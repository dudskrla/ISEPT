# main_ours_vidyamurthy.py

# ──────────────────────────────────────────────────────────────────────────────
# 실행 가이드 (Execution Guide)
# ──────────────────────────────────────────────────────────────────────────────
# 필수 패키지 설치:
# pip install torch torchvision scikit-learn pandas numpy tqdm openpyxl statsmodels

# 주의사항:
# 1. 1991년은 데이터 특성상 2월부터 실행해야 합니다.
#    python3 main_ours_vidyamurthy.py --start_year 1991 --start_month 2
# 2. 실행 전에 반드시 CAE 이미지 생성 스크립트가 선행되어야 합니다.
#    ./MODEL/model_utils/Image_CAE.py 실행 필요

import os
import csv
import time
import shutil
import argparse
import numpy as np
import pandas as pd
import multiprocessing

from tqdm import tqdm
from datetime import datetime
from pandas.tseries.offsets import MonthEnd

# 커스텀 모듈 임포트
from mlp import run_mlp_pipeline
from MODEL.model_utils.Pair_Trading import Pair_Trading
from MODEL.model_utils.CAE import train_cae_model, CAE, device as CAE_DEVICE


# ──────────────────────────────────────────────────────────────────────────────
# argparse 설정 및 상수 정의
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="CAE+MLP 기반 페어 선정 및 Vidyamurthy 트레이딩 시뮬레이션")
parser.add_argument('--start_year',  type=int, default=1991, help="시뮬레이션 시작 연도")
parser.add_argument('--start_month', type=int, choices=range(1,13), default=1, help="시뮬레이션 시작 월")
parser.add_argument('--train_months', type=int, default=156, help="Training 기간 (기본 20년)")
parser.add_argument('--test_months',  type=int, default=252, help="Testing 기간 (기본 10년)")
# Vidyamurthy 임계값 설정 (Z-score 방식; dev = spread - mu)
parser.add_argument('--k_entry', type=float, default=2.0, help="진입 임계값 (Sigma 단위)")
parser.add_argument('--k_exit',  type=float, default=0.0, help="청산 임계값 (Sigma 단위, 0.0=평균)")
args = parser.parse_args()

INIT_YEAR    = args.start_year
INIT_MONTH   = args.start_month
TRAIN_MONTHS = args.train_months
TEST_MONTHS  = args.test_months

K_ENTRY      = args.k_entry
K_EXIT       = args.k_exit

TOP_K_TRAIN  = 100
TOP_K_TEST   = 30
OHLC_DIR     = "./DATA/OHLC_DATA"

# 경로 설정
IMAGE_CAE_DIR = "./DATA/IMAGE_CAE"
CAE_MODEL_DIR = f"./MODEL/MODEL_CAE/OURS/{INIT_YEAR}"
MLP_MODEL_DIR = f"./MODEL/MODEL_MLP/TRAIN_ONE/OURS/{INIT_YEAR}"

BEST_PAIR_DIR = f"./BEST_PAIR/TRAIN_ONE/2_VIDYAMURTHY/{INIT_YEAR}"
OUTPUT_DIR    = f"./PAIR_RESULT/TRAIN_ONE/2_VIDYAMURTHY/{INIT_YEAR}"
OUTPUT_DAILY_DIR = OUTPUT_DIR + "_DAILY"


# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티 함수 (Utility Functions)
# ──────────────────────────────────────────────────────────────────────────────
def load_snp500_universe(file_path: str = './RAW_DATA/SNP500_UNIVERSE.xlsx'):
    """
    S&P500 유니버스 데이터를 로드하여 DataFrame과 월별 딕셔너리로 반환합니다.

    Args:
        file_path (str): 엑셀 파일 경로

    Returns:
        tuple: (df_snp500_universe, monthly_snp500_universe)
    """
    sheets = pd.ExcelFile(file_path).sheet_names
    df_snp500_universe = pd.DataFrame()

    for sheet in sheets:
        sheet_data = pd.read_excel(file_path, sheet_name=sheet, nrows=510, header=None)
        month_end_dates = sheet_data.iloc[0, :].dropna()
        codes = sheet_data.iloc[2:, 0].reset_index(drop=True)
        names = sheet_data.iloc[2:, 1].reset_index(drop=True)

        for date in month_end_dates:
            temp_df = pd.DataFrame({
                'Code': codes,
                'Name': names,
                'Month_End': pd.to_datetime(date) + MonthEnd(0)
            })
            df_snp500_universe = pd.concat([df_snp500_universe, temp_df], ignore_index=True)

    df_snp500_universe['Month'] = df_snp500_universe['Month_End'].dt.strftime('%Y-%m')
    monthly_snp500_universe = df_snp500_universe.groupby('Month')['Code'].apply(list).to_dict()
    return df_snp500_universe, monthly_snp500_universe


def last_n_months(year: int, month: int, n: int) -> list:
    """(year, month) 기준 과거 n개월(해당 월 미포함) 리스트를 반환합니다."""
    end = pd.Period(f"{year}-{month:02d}", freq="M") - 1
    return [(end - i).strftime("%Y-%m") for i in reversed(range(n))]

def next_n_months(year: int, month: int, n: int) -> list:
    """(year, month) 기준 미래 n개월(해당 월 포함) 리스트를 반환합니다."""
    start = pd.Period(f"{year}-{month:02d}", freq="M")
    return [(start + i).strftime("%Y-%m") for i in range(1, n+1)]

def prev_month_str(m: str) -> str:
    """주어진 월('YYYY-MM')의 이전 달 문자열을 반환합니다."""
    return (pd.Period(m, freq="M") - 1).strftime("%Y-%m")


# ──────────────────────────────────────────────────────────────────────────────
# 가격 데이터 로딩 및 처리 (Price Data Loading & Processing)
# ──────────────────────────────────────────────────────────────────────────────
def load_price_dict(tickers: list, months: list, base_dir: str, field='CLOSE') -> dict:
    """
    지정된 Ticker와 기간에 대한 가격 데이터를 로드하여 Dictionary로 반환합니다.
    """
    temp = {t: [] for t in tickers}
    for m in months:
        path = os.path.join(base_dir, field, f"{m}-{field}.csv")
        if not os.path.exists(path):
            for t in tickers:
                temp[t].append(pd.Series(dtype=float))
            continue

        try:
            df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
        except Exception:
            for t in tickers:
                temp[t].append(pd.Series(dtype=float))
            continue

        for t in tickers:
            if t in df.columns:
                temp[t].append(df[t])
            else:
                temp[t].append(pd.Series([np.nan]*len(df), index=df.index))

    out = {}
    for t, lst in temp.items():
        valid = [s for s in lst if not s.empty]
        out[t] = pd.concat(valid).sort_index() if valid else pd.Series(dtype=float)
    return out

def _normalize(s: pd.Series):
    """시리즈의 첫 유효값으로 나누어 정규화합니다."""
    if s is None or s.empty:
        return None
    idx = s.first_valid_index()
    if idx is None or s.loc[idx] == 0 or np.isnan(s.loc[idx]):
        return None
    return s / s.loc[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Vidyamurthy 트레이딩 시뮬레이터 (Trading Simulator)
# ──────────────────────────────────────────────────────────────────────────────
def simulate_pair_vidya(pa: pd.Series, pb: pd.Series, mu: float, sigma: float,
                        k_entry=0.75, k_exit=0.0):
    """
    Vidyamurthy 스타일의 페어 트레이딩을 시뮬레이션합니다.
    
    전략:
        - Spread Deviation: dev = (pa - pb) - mu
        - Entry: |dev| >= k_entry * sigma
        - Exit: |dev| <= k_exit * sigma (Mean Reversion)

    Args:
        pa, pb (pd.Series): 종목 A, B의 가격 시계열
        mu (float): Spread의 평균
        sigma (float): Spread의 표준편차
        k_entry (float): 진입 임계값 계수
        k_exit (float): 청산 임계값 계수

    Returns:
        tuple: (perf, daily, events)
    """
    pm = Pair_Trading()
    events = []
    pos_side = None
    pending = None  # ('open', params) or ('close', None)

    entry = k_entry * sigma
    exit_thr = k_exit * sigma

    dates = pa.index.intersection(pb.index).sort_values()

    for dt in dates:
        a = pa.loc[dt]
        b = pb.loc[dt]
        cur = {'A': a, 'B': b}

        # 1. 상장 폐지(Delisting) 또는 결측 처리
        if np.isnan(a) or np.isnan(b):
            if pm.positions:
                pm.close_position(cur)
                events.append({'Date': dt, 'Action': 'DELIST_CLOSE', 'PortVal': pm.portfolio_values[-1]})
            break

        # 2. 예약된 주문 실행 (전일 시그널 -> 당일 체결)
        if pending:
            side, prm = pending
            if side == 'open':
                L, S, pl, ps = prm
                pm.open_position(L, S, pl, ps)
                pos_side = +1 if L == 'A' else -1
                events.append({'Date': dt, 'Action': 'OPEN', 'Long': L, 'Short': S,
                               'PriceLong': pl, 'PriceShort': ps, 'PortVal': pm.portfolio_values[-1]})
            else:  # close
                pm.close_position(cur)
                pos_side = None
                events.append({'Date': dt, 'Action': 'CLOSE', 'PortVal': pm.portfolio_values[-1]})
            pending = None

        # 3. Spread Deviation 계산
        dev = (a - b) - mu
        abs_dev = abs(dev)

        # 4. 다음날 주문 생성
        if pos_side is None and abs_dev >= entry:
            if dev > 0:
                # A 고평가 -> Long B, Short A
                pending = ('open', ('B', 'A', b, a))
            else:
                # A 저평가 -> Long A, Short B
                pending = ('open', ('A', 'B', a, b))
        elif pos_side is not None:
            if abs_dev <= exit_thr:
                pending = ('close', None)

        # 5. MTM 업데이트
        pm.hold_position(cur)
        events.append({'Date': dt, 'Action': 'HOLD', 'PortVal': pm.portfolio_values[-1]})

    # 만기 강제 청산
    if pm.positions and len(dates) > 0:
        last_dt = dates[-1]
        pm.close_position({'A': pa.loc[last_dt], 'B': pb.loc[last_dt]})
        events.append({'Date': last_dt, 'Action': 'FORCE_CLOSE', 'PortVal': pm.portfolio_values[-1]})

    perf = pm.finalize_performance()

    pv = pm.portfolio_values
    if len(pv) == 0 or len(dates) == 0:
        daily = pd.Series(dtype=float)
    else:
        n = min(len(pv), len(dates))
        daily = pd.Series(pv[:n], index=dates[:n], name='PortVal')

    return perf, daily, pd.DataFrame(events)


# ──────────────────────────────────────────────────────────────────────────────
# 월별 트레이딩 실행 (Execution Wrapper)
# ──────────────────────────────────────────────────────────────────────────────
def run_pair_trading_vidya_like(pair_list, prior_y, prior_m, ym_tag, base_ohlc_dir, out_dir, out_daily_dir):
    """
    선정된 페어 리스트에 대해 월별 Vidyamurthy 트레이딩을 수행합니다.

    Args:
        pair_list (list): 트레이딩할 페어 리스트
        prior_y, prior_m (int): 기준 연도 및 월 (전월)
        ym_tag (str): 결과 파일명 태그 ('YYYY-MM')
        base_ohlc_dir (str): OHLC 데이터 경로
        out_dir (str): 결과 저장 경로
        out_daily_dir (str): 일별 수익률 저장 경로
    """
    if not pair_list:
        print(f"[WARN] Empty pair_list for {ym_tag}")
        return

    months_form  = last_n_months(prior_y, prior_m, 12)
    months_trade = next_n_months(prior_y, prior_m, 6)

    tickers = sorted({t for a, b in pair_list for t in (a, b)})

    pf = load_price_dict(tickers, months_form,  base_ohlc_dir, 'CLOSE')
    pt = load_price_dict(tickers, months_trade, base_ohlc_dir, 'CLOSE')

    metrics_cols = ["Count", "ROI", "Sharpe Ratio", "Cumulative Return",
                    "Sortino Ratio", "Maximum Drawdown", "Calmar Ratio",
                    "Volatility", "Hit Ratio"]

    all_res = {}
    all_rets = {}

    for (p1, p2) in tqdm(pair_list, desc=f"Trade {ym_tag} (Vidya)"):
        # Formation 기간 통계 산출
        s1_form = _normalize(pf.get(p1, pd.Series(dtype=float)))
        s2_form = _normalize(pf.get(p2, pd.Series(dtype=float)))
        if s1_form is None or s2_form is None:
            continue

        idx_common = s1_form.index.intersection(s2_form.index).sort_values()
        if idx_common.empty:
            continue

        spread = s1_form.loc[idx_common] - s2_form.loc[idx_common]
        mu = float(spread.mean())
        sigma = float(spread.std(ddof=0))
        if not np.isfinite(sigma) or sigma == 0:
            continue

        # Trading 기간 데이터 로드
        s1_trade = _normalize(pt.get(p1, pd.Series(dtype=float)))
        s2_trade = _normalize(pt.get(p2, pd.Series(dtype=float)))
        if s1_trade is None or s2_trade is None:
            continue

        idx_t = s1_trade.index.intersection(s2_trade.index).sort_values()
        if idx_t.empty:
            continue

        pa = s1_trade.loc[idx_t]
        pb = s2_trade.loc[idx_t]

        # 시뮬레이션
        perf, daily_nav, _ = simulate_pair_vidya(pa, pb, mu, sigma)

        key = f"{p1}_{p2}"
        all_res[key] = perf
        if daily_nav is not None and not daily_nav.empty:
            # Pandas 기본 fill_method 전파 방지
            all_rets[key] = daily_nav.pct_change(fill_method=None).fillna(0.0)

    # 결과 저장
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_daily_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{ym_tag}_all_results.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker1", "ticker2"] + metrics_cols)
        for pair, stats in all_res.items():
            t1, t2 = pair.split("_", 1)
            row = [stats.get(m, "") for m in metrics_cols]
            w.writerow([t1, t2] + row)
    print(f"[INFO] Saved: {out_path} (pairs={len(all_res)})")

    if all_rets:
        df_rets = pd.DataFrame(all_rets).rename_axis("Date")
        df_rets.to_csv(os.path.join(out_daily_dir, f"{ym_tag}_Return.csv"))
        print(f"[INFO] Saved daily returns: {os.path.join(out_daily_dir, f'{ym_tag}_Return.csv')}")


# ──────────────────────────────────────────────────────────────────────────────
# 부트스트래핑: SSD 기반 초기 페어 선정 (Bootstrap / Warm-start)
# ──────────────────────────────────────────────────────────────────────────────
def make_pairs_csv_for_month_vidya(
    month_str: str,
    df_univ: pd.DataFrame,
    ohlc_dir: str,
    out_dir: str,
    top_candidates: int = 5000,
    min_days: int = 120,
    k_entry: float = 0.75,
    k_exit: float = 0.0,
):
    """
    초기 학습 라벨이 없을 때, SSD 방식으로 후보 페어를 선정하고 
    Vidya 전략으로 Sharpe Ratio(라벨)를 계산하여 라벨 파일(pairs.csv)을 생성합니다.

    Args:
        month_str (str): 기준 월 ('YYYY-MM')
        df_univ (pd.DataFrame): 유니버스 데이터
        ohlc_dir (str): OHLC 데이터 경로
        out_dir (str): 결과 저장 경로
        top_candidates (int): SSD로 선발할 후보 수
        min_days (int): 최소 데이터 길이
        k_entry, k_exit: Vidya 전략 파라미터

    Returns:
        str: 생성된 CSV 파일 경로
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{month_str}_pairs.csv")
    if os.path.exists(out_path):
        return out_path

    per = pd.Period(month_str, freq="M")
    prior_y, prior_m = per.year, per.month

    months_form  = last_n_months(prior_y, prior_m, 12)  # month_str-1 포함 과거 12개월
    months_trade = next_n_months(prior_y, prior_m, 6)   # month_str+1부터 6개월

    # Universe Tickers
    tickers = sorted(set(df_univ[df_univ["Month"] == month_str]["Code"].astype(str).tolist()))
    if len(tickers) < 2:
        raise ValueError(f"[{month_str}] universe tickers < 2")

    # Formation Data 로딩
    pf = load_price_dict(tickers, months_form, ohlc_dir, field="CLOSE")

    # Normalize 및 정제
    series = {}
    for t in tickers:
        s = _normalize(pf.get(t, pd.Series(dtype=float)))
        if s is not None and (s.dropna().shape[0] >= min_days):
            series[t] = s

    if len(series) < 2:
        raise ValueError(f"[{month_str}] not enough tickers with valid formation data")

    # 공통 날짜 Inner Join
    X = pd.concat(series, axis=1, join="inner").dropna(how="any")
    if X.shape[0] < min_days or X.shape[1] < 2:
        raise ValueError(f"[{month_str}] aligned formation matrix too small: {X.shape}")

    arr  = X.to_numpy(dtype=np.float64)   # T x N
    cols = X.columns.tolist()             # tickers aligned

    # SSD Calculation: sum_t (x_i - x_j)^2
    G = arr.T @ arr
    diag = np.diag(G)
    ssd = diag[:, None] + diag[None, :] - 2.0 * G

    iu = np.triu_indices(ssd.shape[0], k=1)
    ssd_vals = ssd[iu]

    K = min(top_candidates, ssd_vals.shape[0])
    top_idx = np.argpartition(ssd_vals, K - 1)[:K]

    cand_pairs = [(cols[iu[0][k]], cols[iu[1][k]], float(ssd_vals[k])) for k in top_idx]
    cand_pairs.sort(key=lambda x: x[2])  # SSD Ascending

    # Trade Period 시뮬레이션 (라벨 생성)
    pt = load_price_dict(tickers, months_trade, ohlc_dir, field="CLOSE")

    rows = []
    for a, b, ssd_score in tqdm(cand_pairs, desc=f"Bootstrap pairs (VIDYA) {month_str}"):
        sa_f = X[a]
        sb_f = X[b]
        spread_f = sa_f - sb_f
        mu = float(spread_f.mean())
        sigma = float(spread_f.std(ddof=0))
        if (not np.isfinite(sigma)) or sigma == 0:
            continue

        sa_t = _normalize(pt.get(a, pd.Series(dtype=float)))
        sb_t = _normalize(pt.get(b, pd.Series(dtype=float)))
        if sa_t is None or sb_t is None:
            continue

        idx_t = sa_t.index.intersection(sb_t.index).sort_values()
        if idx_t.empty:
            continue

        pa = sa_t.loc[idx_t]
        pb = sb_t.loc[idx_t]

        perf, _, _ = simulate_pair_vidya(
            pa, pb, mu, sigma,
            k_entry=k_entry, k_exit=k_exit
        )
        sharpe = perf.get("Sharpe Ratio", np.nan)

        rows.append({
            "ticker1": a,
            "ticker2": b,
            "pair": f"{a}_{b}",
            "ssd": ssd_score,
            "sharpe": sharpe,
            "roi": perf.get("ROI", np.nan),
            "count": perf.get("Count", np.nan),
        })

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        raise ValueError(f"[{month_str}] failed to generate any labeled pairs (VIDYA)")

    df_out.sort_values("sharpe", ascending=False, inplace=True)
    df_out.to_csv(out_path, index=False)
    print(f"[BOOTSTRAP-VIDYA] Saved pairs csv -> {out_path} (rows={len(df_out)})")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 메인 실행 함수 (Main Execution Loop)
# ──────────────────────────────────────────────────────────────────────────────
def main():
    df_univ, _ = load_snp500_universe("./DATA/SNP500_UNIVERSE.xlsx")

    # 1. TRAINING PERIOD 실행
    for offset in range(1, TRAIN_MONTHS):
        dt = datetime(INIT_YEAR, INIT_MONTH, 1) + pd.DateOffset(months=offset)
        dt_prior = datetime(INIT_YEAR, INIT_MONTH, 1) + pd.DateOffset(months=offset - 1)

        y, m = dt.year, dt.month
        prior_y, prior_m = dt_prior.year, dt_prior.month

        period = dt.strftime("%Y-%m")        # 이번달 trade 결과 저장 태그
        prior_period = dt_prior.strftime("%Y-%m")  # MLP 입력용 m_str

        # 이미 결과가 있으면 스킵
        if os.path.exists(os.path.join(OUTPUT_DIR, f"{period}_all_results.csv")):
            continue

        total_start_time = time.time()

        # ─── [2nd] 페어 선정 (CAE + MLP Pipeline) ───
        print(f"[2nd] 페어 선정 (이번 달): {period}")

        best_csv = os.path.join(BEST_PAIR_DIR, period, "best_pair_list.csv")
        if os.path.exists(best_csv):
            print(f"Already exists, Loading... {best_csv}")
            df_pairs = pd.read_csv(best_csv)

            # 컬럼명 호환성 체크
            if "pair" in df_pairs.columns:
                pair_list = [p.split("_", 1) for p in df_pairs["pair"].astype(str).tolist()]
                pair_list = [(a, b) for a, b in pair_list]
            elif ("ticker1" in df_pairs.columns) and ("ticker2" in df_pairs.columns):
                pair_list = list(zip(df_pairs["ticker1"].astype(str), df_pairs["ticker2"].astype(str)))
            else:
                raise ValueError(f"[PAIR LOAD] unexpected columns: {df_pairs.columns.tolist()}")

        else:
            # MLP 파이프라인으로 페어 선정
            prev_res_month = prev_month_str(prior_period)  # prior_period의 전월
            prev_results_path = os.path.join(OUTPUT_DIR, f"{prev_res_month}_all_results.csv")

            warmstart_pairs_path = None
            if not os.path.isfile(prev_results_path):
                # ✅ 첫 실행 시 Warm-start(Bootstrap)
                warmstart_pairs_path = make_pairs_csv_for_month_vidya(
                    month_str=prior_period,
                    df_univ=df_univ,
                    ohlc_dir=OHLC_DIR,
                    out_dir=OUTPUT_DIR,
                    top_candidates=5000,
                    min_days=120,
                    k_entry=K_ENTRY, k_exit=K_EXIT
                )
                prev_results_path = None

            pair_list = run_mlp_pipeline(
                year=prior_y, month=prior_m,
                df_universe=df_univ,
                cae_dir=CAE_MODEL_DIR, mlp_dir=MLP_MODEL_DIR,
                base_ohlc_dir=OHLC_DIR,
                pair_select_dir=OUTPUT_DIR, pair_dir=BEST_PAIR_DIR,
                TOP_K_PAIRS=TOP_K_TRAIN,
                prev_results_path=prev_results_path,
                warmstart_pairs_path=warmstart_pairs_path,
            )

        # ─── [1st] 트레이딩 시뮬레이션 (Vidyamurthy) ───
        print(f"[1st] 페어 트레이딩 (Vidya-sim): {period}")
        run_pair_trading_vidya_like(
            pair_list=pair_list,
            prior_y=prior_y, prior_m=prior_m,
            ym_tag=period,
            base_ohlc_dir=OHLC_DIR,
            out_dir=OUTPUT_DIR,
            out_daily_dir=OUTPUT_DAILY_DIR
        )

        # ─── [4th] CAE 학습 (Feature Extraction) ───
        if not os.path.exists(os.path.join(CAE_MODEL_DIR, period, "best.pth")):
            print(f"[4th] CAE 학습: {period}")
            train_cae_model(
                target_month=period,
                model_class=CAE,
                device=CAE_DEVICE,
                cae_dir=CAE_MODEL_DIR,
                image_root=IMAGE_CAE_DIR
            )

        # 이미지 폴더 정리 (Embeddings 존재 시)
        emb_path = os.path.join(MLP_MODEL_DIR, prior_period, "embeddings.npz")
        folder_path = os.path.join(IMAGE_CAE_DIR, prior_period)

        if os.path.isfile(emb_path) and os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"폴더 '{folder_path}' 삭제 완료 (embeddings 존재)")
            except Exception as e:
                print(f"삭제 중 오류 발생: {e}")
        else:
            print(f"[SKIP DELETE] embeddings 없거나 폴더 없음: {folder_path}")

        end_time = time.time()
        print(f"[{period}] 전체 완료 → 소요 {(end_time - total_start_time)/60:.2f}분\n")

    # 2. TESTING PERIOD 실행
    for offset in range(TRAIN_MONTHS, TRAIN_MONTHS + TEST_MONTHS):
        dt = datetime(INIT_YEAR, INIT_MONTH, 1) + pd.DateOffset(months=offset)
        dt_prior = datetime(INIT_YEAR, INIT_MONTH, 1) + pd.DateOffset(months=offset - 1)

        y, m = dt.year, dt.month
        prior_y, prior_m = dt_prior.year, dt_prior.month
        period = dt.strftime("%Y-%m")

        if os.path.exists(os.path.join(OUTPUT_DIR, f"{period}_all_results.csv")):
            print(f"[SKIP TEST] 이미 결과 존재: {period}")
            continue

        # 테스트 기간 페어 선정 (MLP 추론)
        print(f"[2nd-TEST] 페어 선정(테스트): {period}")
        pair_list = run_mlp_pipeline(
            year=prior_y, month=prior_m,
            df_universe=df_univ,
            cae_dir=CAE_MODEL_DIR, mlp_dir=MLP_MODEL_DIR,
            base_ohlc_dir=OHLC_DIR,
            pair_select_dir=OUTPUT_DIR, pair_dir=BEST_PAIR_DIR,
            TOP_K_PAIRS=TOP_K_TEST,
            test_month=None
        )

        # 테스트 기간 트레이딩
        print(f"[1st-TEST] 페어 트레이딩 (Vidya-sim): {period}")
        run_pair_trading_vidya_like(
            pair_list=pair_list,
            prior_y=prior_y, prior_m=prior_m,
            ym_tag=period,
            base_ohlc_dir=OHLC_DIR,
            out_dir=OUTPUT_DIR,
            out_daily_dir=OUTPUT_DAILY_DIR
        )


if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)
    main()