# MODEL/model_utils/CAE.py

import os
import time
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from PIL import Image
from tqdm import tqdm
from datetime import datetime
from torchvision import transforms
from pandas.tseries.offsets import MonthEnd
from torch.cuda.amp import autocast, GradScaler
from dateutil.relativedelta import relativedelta
from torch.utils.data import DataLoader, Dataset, random_split

# 최적화를 위한 벤치마크 모드 활성화
cudnn.benchmark = True

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# 하이퍼파라미터 및 설정 (Hyperparameters & Configuration)
# ──────────────────────────────────────────────────────────────────────────────
CAE_BATCH_SIZE      = 2048  # 배치 크기 (필요시 조정)
CAE_NUM_EPOCHS      = 20    # 총 학습 에포크 수
CAE_PATIENCE        = 3     # Early Stopping 대기 횟수
CAE_LEARNING_RATE   = 1e-4  # 초기 학습률
CAE_TRAIN_VAL_RATIO = 0.7   # 학습/검증 데이터 분할 비율

# 학습률 스케줄러 설정
LR_STEP_SIZE = 5
LR_GAMMA     = 0.5

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# 모델 및 데이터셋 정의 (Model & Dataset Definitions)
# ──────────────────────────────────────────────────────────────────────────────
class CAE(nn.Module):
    """
    이미지 재구성을 위한 합성곱 오토인코더(Convolutional Autoencoder) 모델입니다.
    
    구조:
        - Encoder: 3단계의 Conv2d -> BatchNorm -> PReLU -> MaxPool 블록
        - Decoder: 3단계의 Conv2d -> BatchNorm -> PReLU -> Upsample 블록
    """
    def __init__(self):
        super(CAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z   = self.encoder(x)
        out = self.decoder(z)
        return out


class CAE_Dataset(Dataset):
    """
    CAE 학습을 위한 이미지 데이터셋 클래스입니다.
    Autoencoder 학습이므로 __getitem__에서 (input, target)으로 (img, img)를 반환합니다.

    Args:
        image_paths (list): 이미지 파일 경로 리스트
        transform (callable, optional): 이미지 전처리 변환 함수
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img


# ──────────────────────────────────────────────────────────────────────────────
# 학습 함수 (Training Function)
# ──────────────────────────────────────────────────────────────────────────────
def train_cae_model(
    target_month,
    model_class,
    device,
    learning_rate=CAE_LEARNING_RATE,
    num_epoch=CAE_NUM_EPOCHS,
    patience=CAE_PATIENCE,
    cae_dir="./MODEL_CAE",
    image_root="./DATA/IMAGE_CAE",
):
    """
    지정된 월(target_month)을 기준으로 과거 1년치 데이터를 사용하여 CAE 모델을 학습합니다.
    학습 과정에서의 디버깅 정보 및 타이밍 로그를 출력합니다.

    Args:
        target_month (str): 학습 기준 월 (예: '2023-10')
        model_class (class): 인스턴스화할 모델 클래스 (예: CAE)
        device (torch.device): 학습을 수행할 디바이스
        learning_rate (float): 학습률
        num_epoch (int): 최대 에포크 수
        patience (int): Early Stopping 허용 횟수
        cae_dir (str): 모델 체크포인트 저장 경로
        image_root (str): 이미지 데이터 루트 경로
    """

    # 1. 모델 초기화
    model = model_class().to(device)
    # model = torch.compile(model)  # 디버깅 시 주석 처리
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[DEBUG] 모델 파라미터 수 = {total_params}", flush=True)

    # 2. 학습 기간 설정 (target_month 기준 과거 12개월)
    try:
        end_dt   = datetime.strptime(target_month + "-01", "%Y-%m-%d")
    except Exception as e:
        print(f"[ERROR] target_month 형식 오류: {target_month}", flush=True)
        return
    
    start_dt = end_dt - relativedelta(months=12)
    months   = []
    cur      = start_dt
    while cur <= end_dt:
        months.append(cur.strftime("%Y-%m"))
        cur += relativedelta(months=1)
    print(f"[DEBUG] {target_month}: 학습용 기간(months) = {months}", flush=True)

    # 3. 이미지 경로 수집
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    all_paths = []
    for m in months:
        folder = os.path.join(image_root, m)
        if not os.path.isdir(folder):
            # 폴더가 없는 경우 경고 (주석 처리됨)
            # print(f"[WARN] {target_month}: 폴더 없음 {folder}", flush=True)
            continue
        for root, _, files in os.walk(folder):
            for fn in files:
                if fn.lower().endswith(".png"):
                    all_paths.append(os.path.join(root, fn))
    
    print(f"[DEBUG] {target_month}: 수집된 이미지 파일 경로 개수 = {len(all_paths)}", flush=True)
    if not all_paths:
        print(f"[WARN] {target_month}: 학습할 이미지 없음. 함수 종료.", flush=True)
        return

    # 4. DataLoader 성능 테스트 및 Worker 설정
    full_ds = CAE_Dataset(all_paths, transform=transform)
    print(f"[DEBUG] {target_month}: Dataset 크기 = {len(full_ds)}", flush=True)
    
    # 다양한 num_workers로 첫 배치 로딩 속도 테스트
    for nw in [0, 4, 8, 16]:
        try:
            loader_test = DataLoader(
                full_ds, batch_size=CAE_BATCH_SIZE,
                shuffle=True, num_workers=nw,
                pin_memory=True, persistent_workers=False
            )
            it = iter(loader_test)
            t0 = time.time()
            _ = next(it)
            t1 = time.time()
            print(f"[DEBUG] num_workers={nw}, 첫 배치 로딩: {t1-t0:.2f}s", flush=True)
        except Exception as e:
            print(f"[DEBUG] num_workers={nw}, 첫 배치 로딩 중 예외: {e}", flush=True)

    # 테스트 결과에 따라 적절한 worker 수 설정 (예: 8)
    chosen_workers = 8
    print(f"[DEBUG] DataLoader에서 사용할 num_workers={chosen_workers}", flush=True)

    # 5. 데이터셋 분할 (Train/Validation)
    n_train = int(CAE_TRAIN_VAL_RATIO * len(full_ds))
    n_val   = len(full_ds) - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    print(f"[DEBUG] train dataset size={len(train_ds)}, val dataset size={len(val_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=CAE_BATCH_SIZE, shuffle=True,
        num_workers=chosen_workers, pin_memory=True, persistent_workers=False, prefetch_factor=2
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=CAE_BATCH_SIZE, shuffle=False,
        num_workers=chosen_workers, pin_memory=True, persistent_workers=False, prefetch_factor=2
    )

    # 6. Optimizer, Scheduler, Scaler 설정
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    scaler    = GradScaler()  # AMP(Automatic Mixed Precision) 사용

    # 체크포인트 저장 경로 생성
    save_dir = os.path.join(cae_dir, target_month)
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')
    wait      = 0

    # 7. 학습 루프 (Training Loop)
    for epoch in range(1, num_epoch+1):
        print(f"[DEBUG] Epoch {epoch} 시작", flush=True)
        
        # ── Train ──
        model.train()
        running = 0.0
        for batch_idx, (x, _) in enumerate(train_loader):
            # 타이밍 측정: 데이터 로드 및 GPU 이동
            t0 = time.time()
            x = x.to(device, non_blocking=True)
            t1 = time.time()

            # Forward
            with autocast():
                pred = model(x)
                loss = criterion(pred, x)
            t2 = time.time()

            # Backward + Optimizer
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            t3 = time.time()

            running += loss.item() * x.size(0)

            # 주기적 타이밍 로그 출력
            if batch_idx % 10 == 0:
                print(f"[TIMING][Epoch {epoch}] batch {batch_idx}/{len(train_loader)}: "
                      f"to(device)={(t1-t0):.2f}s, forward={(t2-t1):.2f}s, backward+step={(t3-t2):.2f}s",
                      flush=True)

        train_loss = running / len(train_loader.dataset)
        print(f"[DEBUG] Epoch {epoch} 학습 종료: train_loss={train_loss:.4f}", flush=True)

        # ── Validation ──
        model.eval()
        running_val = 0.0
        print(f"[DEBUG] Epoch {epoch} 검증 시작", flush=True)
        
        for val_idx, (x_val, _) in enumerate(val_loader):
            t0 = time.time()
            x_val = x_val.to(device, non_blocking=True)
            
            with torch.no_grad():
                with autocast():
                    pred_val = model(x_val)
                    loss_val = criterion(pred_val, x_val)
            
            running_val += loss_val.item() * x_val.size(0)
            t1 = time.time()
            
            if val_idx % 10 == 0:
                print(f"[TIMING][Epoch {epoch} 검증] batch {val_idx}/{len(val_loader)}: {(t1-t0):.2f}s", flush=True)
        
        val_loss = running_val / len(val_loader.dataset)
        print(f"[DEBUG] Epoch {epoch} 검증 종료: val_loss={val_loss:.4f}", flush=True)

        # 스케줄러 업데이트
        scheduler.step()

        # 체크포인트 저장 및 Early Stopping 체크
        if val_loss < best_loss:
            best_loss = val_loss
            wait      = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
            print(f"[DEBUG] Epoch {epoch}: best 모델 저장, val_loss={val_loss:.4f}", flush=True)
        else:
            wait += 1
            print(f"[DEBUG] Epoch {epoch}: 개선 없음 (val_loss={val_loss:.4f}), wait={wait}/{patience}", flush=True)
            if wait >= patience:
                print(f"[DEBUG] Epoch {epoch}: Early stopping 발동", flush=True)
                break

    print(f"[DONE] {target_month} CAE 학습 종료, best_val_loss={best_loss:.4f}", flush=True)