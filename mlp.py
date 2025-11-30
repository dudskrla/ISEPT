# mlp.py

import os
import csv
import torch
import shutil
import numpy as np
import pandas as pd

from datetime import datetime
from typing import List, Tuple
from MODEL.model_utils.CAE import CAE
from MODEL.model_utils.MLP import (
    compute_and_cache_embeddings,
    train_regressor,
    predict_sharpe,
)

# ──────────────────────────────────────────────────────────────────────────────
# 데이터 로드 및 전처리 함수 (Data Loading & Preprocessing)
# ──────────────────────────────────────────────────────────────────────────────

def build_price_data(
    df_universe: pd.DataFrame,
    months: List[str],
    base_dir: str = "./OHLC_DATA",
    strict: bool = True,
) -> Tuple[dict, pd.DataFrame]:
    """
    지정된 월(months)에 해당하는 종목들의 종가(CLOSE) 데이터를 로드하고 정리합니다.

    Args:
        df_universe (pd.DataFrame): 유니버스 정보가 담긴 데이터프레임 (필수 컬럼: 'Month', 'Code')
        months (List[str]): 로드할 월 리스트 (예: ["1991-01"])
        base_dir (str): OHLC 데이터가 저장된 기본 디렉토리
        strict (bool): 파일이나 티커가 없을 때 에러 발생 여부 (True면 에러, False면 건너뜀)

    Returns:
        Tuple[dict, pd.DataFrame]:
            - price_data (dict): {ticker: [price_list]} 형태의 딕셔너리 (NaN 제거됨)
            - price_df (pd.DataFrame): 정제된 가격 데이터프레임 (Index=Date, Columns=Tickers)
                                       (모든 값이 NaN인 컬럼은 제거됨)
    
    Raises:
        TypeError: df_universe가 DataFrame이 아닌 경우
        ValueError: 필수 컬럼이 누락되었거나 데이터가 없는 경우
        FileNotFoundError: strict=True이고 파일이 없는 경우
    """
    if not isinstance(df_universe, pd.DataFrame):
        raise TypeError("build_price_data expects df_universe as a pandas DataFrame.")

    if not months:
        return {}, pd.DataFrame()

    # 1. 유니버스에서 대상 티커 추출
    if ("Month" not in df_universe.columns) or ("Code" not in df_universe.columns):
        raise ValueError("df_universe must have columns: ['Month', 'Code'].")

    desired = sorted(
        set(df_universe[df_universe["Month"].isin(months)]["Code"].astype(str).tolist())
    )
    if not desired:
        return {}, pd.DataFrame()

    # 2. 월별 파일 로드
    close_dfs = []
    for m in months:
        close_path = os.path.join(base_dir, "CLOSE", f"{m}-CLOSE.csv")
        if not os.path.exists(close_path):
            if strict:
                raise FileNotFoundError(f"Missing CLOSE file: {close_path}")
            else:
                continue

        # 헤더를 먼저 읽어 존재하는 티커만 필터링 (메모리 효율화)
        header_cols = pd.read_csv(close_path, nrows=0).columns.tolist()
        available = [t for t in desired if t in header_cols]

        if not available:
            if strict:
                raise ValueError(f"No desired tickers found in {close_path}")
            else:
                continue

        usecols = ["Date"] + available
        df = pd.read_csv(close_path, usecols=usecols, parse_dates=["Date"])
        df.set_index("Date", inplace=True)

        # 데이터 타입 변환 (숫자형 보장)
        for c in available:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        close_dfs.append(df)

    if not close_dfs:
        return {}, pd.DataFrame()

    # 3. 데이터 병합 및 정제
    price_df = pd.concat(close_dfs, axis=0)
    price_df.sort_index(inplace=True)

    # 중복 날짜 처리 (마지막 값 유지)
    price_df = price_df[~price_df.index.duplicated(keep="last")]

    # 모든 값이 NaN인 컬럼 제거
    price_df.dropna(axis=1, how="all", inplace=True)

    # 딕셔너리 형태로 변환 (NaN 제거된 리스트)
    price_data = {
        t: price_df[t].dropna().tolist()
        for t in price_df.columns
    }

    return price_data, price_df

def get_subfolders(path: str) -> List[str]:
    """지정된 경로의 하위 폴더 이름 리스트를 반환합니다."""
    return [
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ]

def get_month_strings(year: int, month: int):
    """
    기준 연월을 받아 다음 달, 현재 달, 이전 달의 문자열 정보를 반환합니다.
    
    Returns:
        tuple: ([next_m], [curr_m], [prev_m], curr_m_str, prev_m_str)
    """
    curr = pd.Timestamp(year=year, month=month, day=1)
    prev = curr - pd.DateOffset(months=1)
    nxt  = curr + pd.DateOffset(months=1)
    return [nxt.strftime("%Y-%m")], \
           [curr.strftime("%Y-%m")], \
           [prev.strftime("%Y-%m")], \
           curr.strftime("%Y-%m"), \
           prev.strftime("%Y-%m")

def next_n_months(month_str: str, n: int) -> List[str]:
    """
    기준 월(month_str)부터 n개월 간의 월 문자열 리스트를 반환합니다.
    
    Args:
        month_str (str): "YYYY-MM" (예: "2025-06")
        n (int): 개월 수
        
    Returns:
        List[str]: ["YYYY-MM", "YYYY-MM+1", ...]
    """
    base = pd.Timestamp(month_str + "-01")
    return [(base + pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(n)]
    
def has_complete_close_data(ticker: str, months: List[str], base_dir="./OHLC_DATA") -> bool:
    """
    특정 티커의 데이터가 지정된 기간(months) 동안의 모든 파일에 존재하는지 확인합니다.
    파일 내에서 해당 티커 컬럼이 존재하고, 유효한 데이터가 하나라도 있어야 합니다.
    """
    for m in months:
        fpath = os.path.join(base_dir, "CLOSE", f"{m}-CLOSE.csv")
        if (not os.path.exists(fpath)):
            return False
        df = pd.read_csv(fpath, nrows=1)
        if ticker not in df.columns:
            return False
        full_df = pd.read_csv(fpath, usecols=[ticker])
        if full_df[ticker].dropna().empty:
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 학습 데이터 준비 함수 (Training Data Preparation)
# ──────────────────────────────────────────────────────────────────────────────

def _load_top_bottom_from_all_results(path: str, top_n=20, bottom_n=20) -> pd.DataFrame:
    """
    이전 결과 파일에서 상위 N개와 하위 N개의 페어를 추출하여 학습용 라벨 데이터를 생성합니다.
    """
    df = pd.read_csv(path)

    # 컬럼명 호환성 처리 (Sharpe Ratio vs sharpe)
    if "Sharpe Ratio" in df.columns:
        sharpe_col = "Sharpe Ratio"
    elif "sharpe" in df.columns:
        sharpe_col = "sharpe"
    else:
        raise ValueError(f"[MLP] Sharpe column not found in {path}. cols={df.columns.tolist()}")

    df = df.dropna(subset=[sharpe_col]).copy()
    df = df.sort_values(sharpe_col, ascending=False)

    top = df.head(top_n)
    bot = df.tail(bottom_n)
    out = pd.concat([top, bot], ignore_index=True)

    # 티커 컬럼 확인 및 분리
    if "ticker1" not in out.columns or "ticker2" not in out.columns:
        if "pair" in out.columns:
            tmp = out["pair"].astype(str).str.split("_", n=1, expand=True)
            out["ticker1"], out["ticker2"] = tmp[0], tmp[1]
        else:
            raise ValueError(f"[MLP] ticker columns not found in {path}. cols={out.columns.tolist()}")

    # train_regressor 호환을 위해 컬럼명을 'sharpe'로 통일
    return out[["ticker1", "ticker2", sharpe_col]].rename(columns={sharpe_col: "sharpe"})

def ensure_train_csv(
    train_csv: str,
    prev_results_path: str = None,
    warmstart_pairs_path: str = None,
    top_n: int = 20,
    bottom_n: int = 20
):
    """
    학습에 사용할 CSV 파일이 존재하는지 확인하고, 없으면 생성합니다.
    
    생성 우선순위:
    1. 이전 트레이딩 결과(prev_results_path)에서 Top/Bottom 추출
    2. Warm-start 파일(warmstart_pairs_path) 복사
    """
    if os.path.isfile(train_csv):
        return

    # 1. 이전 결과에서 라벨 추출
    if prev_results_path and os.path.isfile(prev_results_path):
        df = _load_top_bottom_from_all_results(prev_results_path, top_n=top_n, bottom_n=bottom_n)
        os.makedirs(os.path.dirname(train_csv), exist_ok=True)
        df.to_csv(train_csv, index=False)
        print(f"[MLP] Built train csv from all_results → {train_csv} (rows={len(df)})")
        return

    # 2. Warm-start 파일 복사 (초기 학습용)
    if warmstart_pairs_path and os.path.isfile(warmstart_pairs_path):
        os.makedirs(os.path.dirname(train_csv), exist_ok=True)
        shutil.copy(warmstart_pairs_path, train_csv)
        print(f"[MLP] Warm-start train csv copied → {train_csv}")
        return

    raise FileNotFoundError(
        f"Missing train csv: {train_csv}\n"
        f"and no prev_results_path/warmstart_pairs_path available."
    )


# ──────────────────────────────────────────────────────────────────────────────
# MLP 파이프라인 실행 함수 (Main Pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def run_mlp_pipeline(
    year: int,
    month: int,
    df_universe: pd.DataFrame,
    cae_dir: str = "./MODEL_CAE",
    mlp_dir: str = "./MODEL_MLP",
    pair_select_dir: str = "./PAIR_SELECT_100",
    pair_dir: str = "./BEST_PAIR_100",
    base_ohlc_dir: str = "./OHLC_DATA",
    TOP_K_PAIRS: int = 100,
    test_month: str = "",
    batch_size: int = 1024,
    device_infer: str = "cpu",
    prev_results_path=None, 
    warmstart_pairs_path=None
) -> List[Tuple[str, str]]:
    """
    MLP 모델을 사용하여 페어를 선정하는 전체 파이프라인을 실행합니다.

    절차:
    1. 임베딩 계산 및 캐싱 (CAE Encoder 사용)
    2. Regressor(MLP) 학습 (Training Mode인 경우)
    3. Universe 필터링 (데이터 완전성 및 이미지 존재 여부 확인)
    4. Sharpe Ratio 예측 (MLP Inference)
    5. 상위 K개 페어 선정 및 저장

    Args:
        year, month: 기준 연월
        df_universe: 유니버스 데이터프레임
        cae_dir, mlp_dir: 모델 저장 경로
        pair_select_dir, pair_dir: 결과 저장 경로
        base_ohlc_dir: OHLC 데이터 경로
        TOP_K_PAIRS: 선정할 페어 수
        test_month: 테스트 모드일 경우 해당 월 문자열 (비어있으면 학습 모드)
        prev_results_path: 이전 결과 파일 경로 (라벨링용)
        warmstart_pairs_path: 웜스타트 파일 경로 (초기 라벨링용)

    Returns:
        List[Tuple[str, str]]: 선정된 상위 페어 리스트 [(ticker1, ticker2), ...]
    """
    # 1. 월 문자열 설정
    NEXT_M, CURR_M, PRIOR_M, m_str, prev_m_str = get_month_strings(year, month)

    # 2. 경로 설정
    SAVE_EMB_PATH   = os.path.join(mlp_dir, m_str, "embeddings.npz")   
    if test_month:
        LOAD_CAE_PATH   = os.path.join(cae_dir,  test_month, "best.pth")
        SAVE_MLP_PATH   = os.path.join(mlp_dir, test_month, "sharpe_regressor.pth")
        SAVE_PCA_PATH   = os.path.join(mlp_dir, test_month, "pca_model.pkl") 
    else:
        LOAD_CAE_PATH   = os.path.join(cae_dir,  prev_m_str, "best.pth")
        SAVE_MLP_PATH   = os.path.join(mlp_dir, m_str, "sharpe_regressor.pth")
        SAVE_PCA_PATH   = os.path.join(mlp_dir, m_str, "pca_model.pkl") 
    
    for p in (SAVE_EMB_PATH, SAVE_PCA_PATH, SAVE_MLP_PATH):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        
    # 학습 데이터(CSV) 준비
    train_csv = os.path.join(pair_select_dir, f"{m_str}_pairs.csv")
    ensure_train_csv(
        train_csv=train_csv,
        prev_results_path=prev_results_path,
        warmstart_pairs_path=warmstart_pairs_path,
        top_n=20,
        bottom_n=20
    )

    # 3. 임베딩 생성 (Computing Embeddings)
    if not os.path.exists(SAVE_EMB_PATH):
        print("▶ Computing embeddings …")
        compute_and_cache_embeddings(
            image_month   = m_str,
            save_emb_path = SAVE_EMB_PATH,
            load_cae_path = LOAD_CAE_PATH
        )
        
    # 4. 모델 학습 (Training Stage) - 테스트 모드가 아닐 때만 수행
    if not test_month:
        if not os.path.exists(SAVE_PCA_PATH) or not os.path.exists(SAVE_MLP_PATH):
            print("▶ Training Sharpe regressor …")
            train_regressor(
                pairs_csv           = train_csv,
                image_month         = m_str,
                load_cae_path       = LOAD_CAE_PATH,
                save_emb_path       = SAVE_EMB_PATH,
                save_pca_path       = SAVE_PCA_PATH,
                save_mlp_checkpoint = SAVE_MLP_PATH
            )
        
    # 5. Universe 필터링 (Universe Filtering)
    # 현재 달과 다음 달의 데이터, 유니버스, 가격 정보가 모두 존재하는 종목만 선별
    all_close, all_price, all_codes = [], [], []
    for m in (CURR_M[0], NEXT_M[0]):
        # Close Data
        close_df = pd.read_csv(
            os.path.join(base_ohlc_dir, "CLOSE", f"{m}-CLOSE.csv"),
            parse_dates=['Date'], index_col='Date'
        )
        all_close.append(close_df)
        # Price Data
        _, price_df = build_price_data(df_universe, [m], base_dir=base_ohlc_dir)
        all_price.append(price_df)
        # Code List
        codes = df_universe[df_universe["Month"] == m]["Code"].tolist()
        all_codes.append(codes)

    # 교집합(Intersection) 구하기
    common = (
        set(all_close[0].columns) &
        set(all_close[1].columns) &
        set(all_price[0].columns) &
        set(all_price[1].columns) &
        set(all_codes[0]) &
        set(all_codes[1])
    )    
    months_to_check = next_n_months(NEXT_M[0], 6)

    # 직후 6개월치 데이터가 완전한지 확인
    universe = [
        t for t in common
        if has_complete_close_data(t, months_to_check, base_ohlc_dir)
    ]
    
    # 이미지 파일 존재 여부 필터링
    IMAGE_CAE_DIR = "./DATA/IMAGE_CAE"
    exist_imgs = set(os.listdir(os.path.join(IMAGE_CAE_DIR, m_str)))
    universe = [t for t in universe if t in exist_imgs]

    print(f"Universe size: {len(universe)}")
    if len(universe) < 2:
        raise ValueError("유효 universe 종목이 2개 미만입니다")

    # 6. Sharpe 예측 (Predicting Sharpe)
    # 6-1. 저장된 임베딩 로드
    npz = np.load(SAVE_EMB_PATH, allow_pickle=True)
    all_tickers = npz["tickers"].tolist()
    all_emb     = npz["embeddings"]
    
    # 6-2. Universe에 포함된 티커만 필터링
    uni_idx     = [i for i,t in enumerate(all_tickers) if t in universe]
    tickers     = [all_tickers[i] for i in uni_idx]
    emb         = all_emb[uni_idx]
    
    # 6-3. 예측 수행
    print("▶ Predicting with cached embeddings …")
    output_csv = os.path.join(pair_dir, f"{NEXT_M[0]}", "best_pair_list.csv")
    os.makedirs(f"{pair_dir}/{NEXT_M[0]}", exist_ok=True)
    
    df_pred = predict_sharpe(
        tickers_list        = tickers,
        embeddings_array    = emb,
        load_pca_path       = SAVE_PCA_PATH,
        load_mlp_checkpoint = SAVE_MLP_PATH,
        output_csv          = output_csv
    )

    # 7. 결과 저장 (Top-K Selection)
    topk = df_pred.head(TOP_K_PAIRS)[['pair','pred_sharpe']]
    topk.to_csv(output_csv, index=False)
    print(f"[INFO] TOP-{TOP_K_PAIRS} saved → {output_csv}")

    return [(p.split('_')[0], p.split('_')[1]) for p in topk['pair']]