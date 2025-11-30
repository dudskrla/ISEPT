# MODEL/model_utils/MLP.py

import os
import torch
import joblib
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from MODEL.model_utils.CAE import CAE
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.decomposition import IncrementalPCA

# ──────────────────────────────────────────────────────────────────────────────
# 상수 및 설정 (Constants & Configuration)
# ──────────────────────────────────────────────────────────────────────────────
IMAGE_CAE_DIR = "./DATA/IMAGE_CAE"
PCA_DIM       = 512
BATCH_SIZE    = 512
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
EPOCHS        = 50
PATIENCE      = 5

# 전역 변수
_ENC = None

# 디바이스 설정 (GPU 우선)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 전처리 (CAE 입력용)
_transform_cae = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# ──────────────────────────────────────────────────────────────────────────────
# 모델 정의 (Model Definitions)
# ──────────────────────────────────────────────────────────────────────────────
class SharpeRegressor(nn.Module):
    """
    임베딩 벡터를 입력받아 Sharpe Ratio를 예측하는 MLP 회귀 모델입니다.
    
    Args:
        input_dim (int): 입력 벡터의 차원 수 (PCA 축소 후 차원)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512),       nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128),        nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ──────────────────────────────────────────────────────────────────────────────
# 임베딩 헬퍼 함수 (Embedding Helpers)
# ──────────────────────────────────────────────────────────────────────────────
def get_ticker_embedding(ticker, encoder, image_month):
    """
    특정 종목(Ticker)의 이미지들에 대해 인코더를 통과시킨 후, 평균 임베딩 벡터를 반환합니다.

    Args:
        ticker (str): 종목 코드
        encoder (nn.Module): 학습된 CAE의 인코더 모듈
        image_month (str): 이미지 데이터가 위치한 월(Month) 디렉토리명

    Returns:
        np.ndarray or None: 평균 임베딩 벡터 (1D array). 데이터가 없으면 None.
    """
    folder = os.path.join(IMAGE_CAE_DIR, image_month, ticker)
    if not os.path.isdir(folder):
        return None
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".png")]
    if not paths:
        return None

    class _DS(Dataset):
        def __init__(self, paths): self.paths = paths
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")
            return _transform_cae(img), 0

    loader = DataLoader(_DS(paths), batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    vecs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            z = encoder(imgs)
            vecs.append(z.view(z.size(0), -1).cpu().numpy())
    return np.vstack(vecs).mean(axis=0)

# ──────────────────────────────────────────────────────────────────────────────
# 1) 임베딩 캐시 생성 (Embedding Caching)
# ──────────────────────────────────────────────────────────────────────────────
class _AllImgsDS(Dataset):
    """
    한 달 치 모든 종목의 이미지를 (종목 인덱스, 이미지) 형태로 제공하는 데이터셋입니다.
    
    Args:
        image_month (str): 대상 월 디렉토리명
        tickers (list): 처리할 종목 리스트
    """
    def __init__(self, image_month, tickers):
        self.paths = []      # List[(ticker_idx, image_path)]
        for idx, t in enumerate(tickers):
            folder = os.path.join(IMAGE_CAE_DIR, image_month, t)
            if not os.path.isdir(folder): continue
            for f in os.listdir(folder):
                if f.lower().endswith(".png"):
                    self.paths.append((idx, os.path.join(folder, f)))
                    
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, i):
        idx, p = self.paths[i]
        img = Image.open(p).convert("RGB")
        return idx, _transform_cae(img)
    
def compute_and_cache_embeddings(
    image_month: str,
    save_emb_path: str,
    load_cae_path: str
):
    """
    모든 종목의 이미지를 한 번의 DataLoader 흐름으로 처리하여 임베딩을 계산하고 저장합니다.
    (CAE 체크포인트 로드 시 '_orig_mod.' 접두사를 제거합니다.)

    Args:
        image_month (str): 이미지 데이터 월 디렉토리
        save_emb_path (str): 계산된 임베딩(.npz)을 저장할 경로
        load_cae_path (str): 학습된 CAE 모델 가중치 경로
    """
    # 1. CAE 인코더 준비
    cae = CAE().to(DEVICE)

    # 체크포인트 키 수정 ("_orig_mod." 제거)
    raw_ckpt = torch.load(load_cae_path, map_location=DEVICE)
    fixed_ckpt = {}
    for key, val in raw_ckpt.items():
        if key.startswith("_orig_mod."):
            fixed_ckpt[key[len("_orig_mod."):]] = val
        else:
            fixed_ckpt[key] = val
    cae.load_state_dict(fixed_ckpt, strict=False)

    cae.eval()
    encoder = cae.encoder

    # 2. 티커 리스트 및 데이터셋 구성
    tickers = [
        d for d in os.listdir(os.path.join(IMAGE_CAE_DIR, image_month))
        if os.path.isdir(os.path.join(IMAGE_CAE_DIR, image_month, d))
    ]
    ds = _AllImgsDS(image_month, tickers)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"[INFO] 총 이미지 개수: {len(ds)}")

    # 3. 이미지별 임베딩 계산 및 티커별 합계/카운트 집계
    sums   = [None] * len(tickers)
    counts = [0] * len(tickers)

    with torch.no_grad():
        for idxs, imgs in tqdm(loader, desc="Embedding all images"):
            imgs = imgs.to(DEVICE, non_blocking=True)
            z = encoder(imgs)
            z = z.view(z.size(0), -1).cpu().numpy()
            for i, emb in zip(idxs.numpy(), z):
                if sums[i] is None:
                    sums[i] = emb.copy()
                else:
                    sums[i] += emb
                counts[i] += 1

    # 4. 티커별 평균 계산 및 저장
    emb_dict = {
        t: sums[i] / counts[i]
        for i, t in enumerate(tickers)
        if counts[i] > 0
    }
    os.makedirs(os.path.dirname(save_emb_path), exist_ok=True)
    np.savez_compressed(
        save_emb_path,
        tickers=list(emb_dict.keys()),
        embeddings=np.vstack(list(emb_dict.values()))
    )
    print(f"[SAVE] {len(emb_dict)} embeddings → {save_emb_path}")

# ──────────────────────────────────────────────────────────────────────────────
# 2) PCA 차원 축소 및 Regressor 학습 (Training)
# ──────────────────────────────────────────────────────────────────────────────
def train_regressor(
    pairs_csv: str,
    image_month: str,
    load_cae_path: str,
    save_emb_path: str,
    save_pca_path: str,
    save_mlp_checkpoint: str
):
    """
    임베딩 데이터를 로드하여 PCA를 수행하고, SharpeRegressor를 학습시킵니다.
    
    Args:
        pairs_csv (str): 학습용 Pair 및 Label 정보가 담긴 CSV 경로
        image_month (str): 이미지 데이터 월
        load_cae_path (str): CAE 모델 경로
        save_emb_path (str): 임베딩 저장/로드 경로
        save_pca_path (str): PCA 모델 저장 경로
        save_mlp_checkpoint (str): 학습된 MLP 모델 저장 경로
    """
    # 1. 임베딩 데이터 준비
    if not os.path.isfile(save_emb_path):
        compute_and_cache_embeddings(image_month, save_emb_path, load_cae_path)

    npz = np.load(save_emb_path, allow_pickle=True)
    tickers_list = npz['tickers']               # ndarray of shape (N,), dtype=object
    emb_array     = npz['embeddings']           # shape (N, emb_dim), dtype=float32

    # 문자열 티커 -> 인덱스 매핑 생성
    ticker2idx = {t: i for i, t in enumerate(tickers_list.tolist())}

    # 2. 레이블 데이터 로드
    df = pd.read_csv(pairs_csv)
    print(f"[DBG] columns in {pairs_csv}:", df.columns.tolist())

    if 'pred_sharpe' in df.columns:
        score_col = 'pred_sharpe'
    elif 'sharpe' in df.columns:
        score_col = 'sharpe'
    else:
        raise KeyError(f"Neither 'pred_sharpe' nor 'sharpe' in {pairs_csv}")

    df['pair'] = df['pair'].str.strip()
    spl = df['pair'].str.split('_', n=1, expand=True)
    df['A'] = spl[0].str.strip()
    df['B'] = spl[1].str.strip()

    # 3. 학습용 X, y 생성 (Vectorized Operation)
    A_vals = df['A'].values
    B_vals = df['B'].values
    scores = df[score_col].values.astype(np.float32)

    # A, B 모두 매핑 테이블에 존재하는 행만 필터링
    maskA = np.fromiter((t in ticker2idx for t in A_vals), dtype=bool)
    maskB = np.fromiter((t in ticker2idx for t in B_vals), dtype=bool)
    valid_mask = maskA & maskB

    # 유효 행의 인덱스를 emb_array 인덱스로 변환
    A_idxs = np.fromiter((ticker2idx[t] for t in A_vals[valid_mask]), dtype=np.int64)
    B_idxs = np.fromiter((ticker2idx[t] for t in B_vals[valid_mask]), dtype=np.int64)
    y      = scores[valid_mask]  # shape (M,)

    # emb_array를 인덱싱하여 임베딩 추출
    embA = emb_array[A_idxs]   # (M, emb_dim)
    embB = emb_array[B_idxs]   # (M, emb_dim)

    # X 데이터 구성: (M, 2*emb_dim)
    X = np.hstack([embA, embB]).astype(np.float32)
    y = y.astype(np.float32)

    # 4. PCA 수행 (GPU Accelerated SVD)
    # 4-1. 데이터를 GPU로 이동
    X_tensor = torch.from_numpy(X).to(DEVICE)      # (M, 2*emb_dim), float32

    # 4-2. 평균 제거 (Centering)
    mean = X_tensor.mean(dim=0, keepdim=True)      # (1, 2*emb_dim)
    X_center = X_tensor - mean                     # (M, 2*emb_dim)

    # 4-3. torch.pca_lowrank를 사용한 주성분 계산
    n = min(PCA_DIM, X_center.shape[0], X_center.shape[1])
    U, S, V = torch.pca_lowrank(X_center, q=n)     # V: (2*emb_dim, n), on GPU

    # 4-4. 데이터 투영 (Projection)
    Xp_tensor = X_center @ V                      # (M, n)

    # 4-5. CPU로 이동 및 joblib 저장
    Xp = Xp_tensor.cpu().numpy().astype(np.float32)   # (M, n)
    pca_data = {
        'components_': V.cpu().numpy(),                # (2*emb_dim, n)
        'mean_':       mean.cpu().numpy().reshape(-1), # (2*emb_dim,)
        'n_components': n
    }
    os.makedirs(os.path.dirname(save_pca_path), exist_ok=True)
    joblib.dump(pca_data, save_pca_path)

    # 5. MLP 모델 학습
    model = SharpeRegressor(n).to(DEVICE)
    crit  = nn.MSELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    loader = DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xp), torch.from_numpy(y)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        pin_memory=True
    )

    best, wait = float('inf'), 0
    for ep in range(1, EPOCHS+1):
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = crit(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        avg = tot / len(loader.dataset)
        print(f"[{ep:02d}] loss={avg:.4f}")
        
        # Early Stopping 및 체크포인트 저장
        if avg < best:
            best, wait = avg, 0
            os.makedirs(os.path.dirname(save_mlp_checkpoint), exist_ok=True)
            torch.save(model.state_dict(), save_mlp_checkpoint)
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    print(f"[DONE] best={best:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 3) Sharpe Ratio 예측 (Inference)
# ──────────────────────────────────────────────────────────────────────────────
from itertools import combinations

def predict_sharpe(
    tickers_list: list[str],
    embeddings_array: np.ndarray,   # (num_tickers, emb_dim)
    load_pca_path: str,
    load_mlp_checkpoint: str,
    output_csv: str,
    batch_size: int = 50_000,
    device: str = "cuda"
) -> pd.DataFrame:
    """
    학습된 PCA 및 MLP 모델을 사용하여 주어진 종목들의 모든 조합(Pair)에 대해 Sharpe Ratio를 예측합니다.
    메모리 효율을 위해 GPU에서 배치 단위로 처리를 수행합니다.

    Args:
        tickers_list (list[str]): 종목 코드 리스트
        embeddings_array (np.ndarray): 종목별 임베딩 배열 (shape: N x emb_dim)
        load_pca_path (str): PCA 모델(또는 딕셔너리) 경로
        load_mlp_checkpoint (str): 학습된 MLP 체크포인트 경로
        output_csv (str): 예측 결과를 저장할 CSV 경로
        batch_size (int): 인퍼런스 배치 크기 (기본값: 50,000)
        device (str): 사용할 디바이스 ('cuda' or 'cpu')

    Returns:
        pd.DataFrame: 예측 결과 데이터프레임 (columns: pair, pred_sharpe)
    """
    # 1. 디바이스 설정
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 2. PCA 정보 로드
    pca_obj = joblib.load(load_pca_path)

    if isinstance(pca_obj, dict):
        # train_regressor에서 저장한 dict 형태 (torch.pca_lowrank 결과)
        components_np = pca_obj['components_'].astype(np.float32)  # (2*emb_dim, n)
        mean_np       = pca_obj['mean_'].astype(np.float32)        # (2*emb_dim,)
        n_comp        = int(pca_obj['n_components'])
    else:
        # sklearn IncrementalPCA/PCA 객체인 경우
        components_np = pca_obj.components_.astype(np.float32)  # 보통 (n, 2*emb_dim)
        mean_np       = pca_obj.mean_.astype(np.float32)        # (2*emb_dim,)
        n_comp        = int(pca_obj.n_components_)

        # (n_comp, 2*emb_dim) -> (2*emb_dim, n_comp) 로 전치 필요
        components_np = components_np.T 

    # 3. PCA 파라미터를 Tensor로 변환 및 GPU 이동
    pca_components = torch.from_numpy(components_np).to(device)  # (2*emb_dim, n_comp)
    pca_mean       = torch.from_numpy(mean_np).to(device)        # (2*emb_dim,)
    pca_comp_T     = pca_components  # 연산 시 (Batch, Dim) @ (Dim, Comp) 형태가 되므로 그대로 사용

    # 4. 임베딩 배열(Tensor) 준비
    embeddings_torch = torch.from_numpy(embeddings_array.astype(np.float32)).to(device)
    # (num_tickers, emb_dim)

    # 5. MLP 모델 로드 및 설정
    model = SharpeRegressor(input_dim=n_comp).to(device)
    state = torch.load(load_mlp_checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # 6. 모든 종목 쌍(Pair) 인덱스 생성
    # 메모리 문제 방지를 위해 인덱스는 CPU에서 생성 후 배치 단위로 GPU 이동
    idx_pairs = np.array(list(combinations(range(len(tickers_list)), 2)), dtype=np.int64)

    all_preds = []
    pair_names = []

    # 7. 배치별 예측 루프
    for start in tqdm(range(0, len(idx_pairs), batch_size), desc="Predict GPU"):
        end = min(start + batch_size, len(idx_pairs))
        slice_idx = idx_pairs[start:end]                                # NumPy (batch_size, 2)
        slice_idx_t = torch.from_numpy(slice_idx).long().to(device)     # GPU LongTensor (batch_size, 2)

        # 7-1. 임베딩 슬라이싱 (GPU)
        emb_a = embeddings_torch[slice_idx_t[:, 0]]   # (batch_size, emb_dim)
        emb_b = embeddings_torch[slice_idx_t[:, 1]]   # (batch_size, emb_dim)

        # 7-2. 두 임베딩 결합
        raw_pairs = torch.cat([emb_a, emb_b], dim=1)   # (batch_size, 2*emb_dim)

        # 7-3. PCA 변환 (Mean Centering -> Projection)
        Xp_batch = (raw_pairs - pca_mean) @ pca_comp_T  # (batch_size, n_comp)

        # 7-4. MLP 예측 수행 (GPU -> CPU)
        with torch.no_grad():
            preds_t = model(Xp_batch)                # (batch_size,) on GPU
            preds_np = preds_t.cpu().numpy()         # (batch_size,) NumPy

        all_preds.append(preds_np)

        # 7-5. Pair 이름 저장
        for i, j in slice_idx:
            pair_names.append(f"{tickers_list[i]}_{tickers_list[j]}")

        # GPU 캐시 정리
        del emb_a, emb_b, raw_pairs, Xp_batch, preds_t
        torch.cuda.empty_cache()

    # 8. 결과 병합 및 CSV 저장
    preds = np.concatenate(all_preds)   # (total_num_pairs,)
    df = pd.DataFrame({"pair": pair_names, "pred_sharpe": preds})
    df.sort_values("pred_sharpe", ascending=False, inplace=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[SAVE] {len(df)} rows → {output_csv!r}")
    return df