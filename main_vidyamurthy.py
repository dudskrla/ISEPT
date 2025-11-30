# main_vidyamurthy.py

"""
Vidyamurthy 방식의 페어 선정(Selection) 및 페어 트레이딩(Trading)

실행 예시:
    python3 main_vidyamurthy.py --start_year 2004 --start_month 1 \
        --train_months 156 --test_months 252 \
        --top_k_train 100 --top_k_test 20 \
        --ohlc_dir ./DATA/OHLC_DATA \
        --universe_path ./DATA/SNP500_UNIVERSE.xlsx \
        --best_pair_root ./BEST_PAIR/TRAIN_ONE/VIDYAMURTHY \
        --output_root ./PAIR_RESULT/TRAIN_ONE/VIDYAMURTHY
"""

import os
import csv
import argparse
import itertools
import numpy as np
import pandas as pd
import multiprocessing

from tqdm import tqdm
from datetime import datetime
from typing import List, Tuple
from pandas.tseries.offsets import MonthEnd
from statsmodels.tsa.stattools import coint

# 커스텀 모듈 임포트
from MODEL.model_utils.Pair_Trading import Pair_Trading    # simulate용

# ──────────────────────────────────────────────────────────────────────────────
# argparse 설정 및 상수 정의
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Vidyamurthy 방식 페어 선정 및 페어 트레이딩")
parser.add_argument('--start_year',  type=int, default=1991, help="초기 연도")
parser.add_argument('--start_month', type=int, choices=range(1,13), default=1, help="초기 월")
parser.add_argument('--train_months', type=int, default=156, help="트레이닝 기간 개월수")
parser.add_argument('--test_months',  type=int, default=252, help="테스트 기간 개월수")
parser.add_argument('--top_k_train', type=int, default=100, help="트레이닝 시 Vidyamurthy 상위 N 페어 개수")
parser.add_argument('--top_k_test',  type=int, default=20, help="테스트 시 Vidyamurthy 상위 N 페어 개수")
parser.add_argument('--alpha', type=float, default=0.05, help="공적분 검정 p-value 임계치")
parser.add_argument('--ohlc_dir', type=str, default='./DATA/OHLC_DATA', help="OHLC 데이터 루트 디렉토리")
parser.add_argument('--universe_path', type=str, default='./DATA/SNP500_UNIVERSE.xlsx', help="S&P500 유니버스 엑셀 파일 경로")
parser.add_argument('--best_pair_root', type=str, default='./BEST_PAIR/TRAIN_ONE/VIDYAMURTHY', help="Vidyamurthy best_pair 저장 루트 디렉토리")
parser.add_argument('--output_root', type=str, default='./PAIR_RESULT/TRAIN_ONE/VIDYAMURTHY', help="페어 트레이딩 결과 저장 루트 디렉토리")
args = parser.parse_args()

INIT_YEAR       = args.start_year
INIT_MONTH      = args.start_month
TRAIN_MONTHS    = args.train_months
TEST_MONTHS     = args.test_months
TOP_K_TRAIN     = args.top_k_train
TOP_K_TEST      = args.top_k_test
ALPHA           = args.alpha
OHLC_DIR        = args.ohlc_dir.rstrip("/")
UNIV_PATH       = args.universe_path
BEST_PAIR_ROOT  = args.best_pair_root.rstrip("/")
OUTPUT_ROOT     = args.output_root.rstrip("/")

# ──────────────────────────────────────────────────────────────────────────────
# 데이터 로드 및 유틸리티 함수
# ──────────────────────────────────────────────────────────────────────────────
def load_snp500_universe(file_path: str):
    """
    Excel 파일에서 시트별로 S&P500 종목 데이터를 읽어와 DataFrame 및 Dictionary로 반환합니다.

    Args:
        file_path (str): S&P500 유니버스 엑셀 파일 경로

    Returns:
        tuple: (df_univ, monthly_univ)
            - df_univ (pd.DataFrame): ['Code', 'Month_End', 'Month'] 컬럼 포함
            - monthly_univ (dict): {'YYYY-MM': [Code 목록], ...}
    """
    sheets = pd.ExcelFile(file_path).sheet_names
    df_list = []
    for sheet in sheets:
        data = pd.read_excel(file_path, sheet_name=sheet, nrows=510, header=None)
        month_end_dates = data.iloc[0, :].dropna()
        codes = data.iloc[2:, 0].reset_index(drop=True)
        for date in month_end_dates:
            tmp = pd.DataFrame({
                'Code': codes,
                'Month_End': pd.to_datetime(date) + MonthEnd(0)
            })
            df_list.append(tmp)
    if not df_list:
        raise RuntimeError(f"유니버스 파일에서 데이터를 읽지 못했습니다: {file_path}")
    df_univ = pd.concat(df_list, ignore_index=True)
    df_univ['Month'] = df_univ['Month_End'].dt.strftime('%Y-%m')
    monthly_univ = df_univ.groupby('Month')['Code'].apply(
        lambda s: [c for c in s.dropna() if isinstance(c, str) and c.strip()!=""]
    ).to_dict()
    return df_univ, monthly_univ

def last_n_months(year: int, month: int, n: int) -> list:
    """
    지정된 (year, month) 이전 n개월(해당 월 미포함)의 리스트를 반환합니다.
    """
    end = pd.Period(f"{year}-{month:02d}", freq="M") - 1
    return [(end - i).strftime("%Y-%m") for i in reversed(range(n))]

def next_n_months(year: int, month: int, n: int) -> list:
    """
    지정된 (year, month) 이후 n개월(해당 월 포함)의 리스트를 반환합니다.
    """
    start = pd.Period(f"{year}-{month:02d}", freq="M")
    result = []
    for i in range(1, n+1):
        cand = start + i
        result.append(cand.strftime("%Y-%m"))
    return result

def has_complete_close_data(ticker: str, months: list, base_dir: str) -> bool:
    """
    해당 Ticker가 지정된 기간(months) 동안의 CLOSE 데이터 파일에 모두 존재하고, 
    데이터가 결측 없이 유효한지 확인합니다.
    """
    for m in months:
        fpath = os.path.join(base_dir, "CLOSE", f"{m}-CLOSE.csv")
        if not os.path.exists(fpath):
            return False
        try:
            df = pd.read_csv(fpath, nrows=1)
        except Exception:
            return False
        if ticker not in df.columns:
            return False
        try:
            full_df = pd.read_csv(fpath, usecols=[ticker])
        except Exception:
            return False
        if full_df[ticker].dropna().empty:
            return False
    return True

def build_price_data(
    df_universe: pd.DataFrame,
    months: List[str],
    base_dir: str = "./OHLC_DATA",
    strict: bool = True,
) -> Tuple[dict, pd.DataFrame]:
    """
    df_universe (DataFrame)와 months(예: ["1991-01"])를 받아서
    해당 월들의 CLOSE 데이터를 읽고,

    반환:
      - price_data: dict[ticker] -> List[float] (NaN 제거된 시계열)
      - price_df  : DataFrame (index=Date, columns=tickers, values=close)
                   * 전부 NaN인 컬럼은 제거
    """
    if not isinstance(df_universe, pd.DataFrame):
        raise TypeError("build_price_data expects df_universe as a pandas DataFrame.")

    if not months:
        return {}, pd.DataFrame()

    # df_universe에서 대상 티커 뽑기 (Month/Code 컬럼 기준)
    if ("Month" not in df_universe.columns) or ("Code" not in df_universe.columns):
        raise ValueError("df_universe must have columns: ['Month', 'Code'].")

    desired = sorted(
        set(df_universe[df_universe["Month"].isin(months)]["Code"].astype(str).tolist())
    )
    if not desired:
        return {}, pd.DataFrame()

    close_dfs = []
    for m in months:
        close_path = os.path.join(base_dir, "CLOSE", f"{m}-CLOSE.csv")
        if not os.path.exists(close_path):
            if strict:
                raise FileNotFoundError(f"Missing CLOSE file: {close_path}")
            else:
                continue

        # 헤더로 실제 존재하는 티커만 골라서 usecols 구성 (메모리/속도 유리)
        header_cols = pd.read_csv(close_path, nrows=0).columns.tolist()
        available = [t for t in desired if t in header_cols]

        if not available:
            # 이 달 파일에 원하는 티커가 하나도 없으면 스킵(또는 strict면 에러)
            if strict:
                raise ValueError(f"No desired tickers found in {close_path}")
            else:
                continue

        usecols = ["Date"] + available
        df = pd.read_csv(close_path, usecols=usecols, parse_dates=["Date"])
        df.set_index("Date", inplace=True)

        # 혹시 문자열/혼합형이면 숫자로 변환
        for c in available:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        close_dfs.append(df)

    if not close_dfs:
        return {}, pd.DataFrame()

    # 월별 데이터 이어붙이기
    price_df = pd.concat(close_dfs, axis=0)
    price_df.sort_index(inplace=True)

    # 중복 날짜가 있으면 마지막 값 기준 유지(필요시 정책 변경 가능)
    price_df = price_df[~price_df.index.duplicated(keep="last")]

    # 전부 NaN인 종목 제거
    price_df.dropna(axis=1, how="all", inplace=True)

    # dict 형태도 같이 만들어 반환 (NaN 제거된 list)
    price_data = {
        t: price_df[t].dropna().tolist()
        for t in price_df.columns
    }

    return price_data, price_df


# ──────────────────────────────────────────────────────────────────────────────
# Vidyamurthy 페어 선정 (Selection)
# ──────────────────────────────────────────────────────────────────────────────
def _log_returns(price_series):
    """
    가격 시리즈를 입력받아 표준화된(Standardized) 로그 수익률을 계산합니다.
    변동성이 0이거나 데이터가 부족한 경우 None을 반환합니다.
    """
    px = np.asarray(price_series, dtype=float)
    mask = (px > 0) & ~np.isnan(px)
    px = px[mask]
    if len(px) < 3:
        return None
    ret = np.diff(np.log(px))
    if ret.std() == 0:
        return None
    return (ret - ret.mean()) / ret.std()

def _coint_test_rets(args):
    """
    병렬 처리를 위한 공적분(Cointegration) 검정 래퍼 함수입니다.
    """
    t1, t2, rets_dict, alpha = args
    r1, r2 = rets_dict[t1], rets_dict[t2]
    n = min(len(r1), len(r2))
    if n < 30:
        return None
    score, pval, _ = coint(r1[-n:], r2[-n:])
    if pval < alpha:
        return (t1, t2, pval)
    return None

def vidyamurthy_select(
    year: int,
    month: int,
    df_univ: pd.DataFrame,
    base_ohlc_dir: str,
    top_n: int,
    alpha: float,
    output_dir: str
) -> list:
    """
    Vidyamurthy 방식으로 페어를 선정합니다.
    
    절차:
    1. Formation 기간(12개월)의 데이터를 로드하여 로그 수익률을 계산합니다.
    2. 모든 페어 조합에 대해 공적분 검정(Cointegration Test)을 수행합니다 (P-value < alpha).
    3. 공적분을 만족하는 페어들을 상관계수(Correlation) 순으로 정렬하여 상위 top_n개를 선정합니다.

    Args:
        year (int): 기준 연도
        month (int): 기준 월
        df_univ (pd.DataFrame): 유니버스 데이터
        base_ohlc_dir (str): OHLC 데이터 경로
        top_n (int): 선정할 페어 수
        alpha (float): 공적분 유의수준
        output_dir (str): 결과 저장 경로

    Returns:
        list: 선정된 페어 리스트 [(t1, t2), ...]
    """
    # 1) 기간 설정 (Formation 12개월, Trading 이후 6개월)
    months_form = last_n_months(year, month, 12)
    months_next = next_n_months(year, month, 6)
    if not months_form or not months_next:
        return []

    # 유니버스 필터: Formation 첫 달과 Trading 첫 달의 교집합
    f0 = months_form[0]
    c0 = months_next[0]
    A_raw = df_univ.loc[df_univ["Month"] == f0, "Code"].dropna()
    B_raw = df_univ.loc[df_univ["Month"] == c0, "Code"].dropna()
    A = {t for t in set(A_raw) if isinstance(t, str) and t.strip()!=""}
    B = {t for t in set(B_raw) if isinstance(t, str) and t.strip()!=""}
    try:
        common = A & B
    except TypeError as e:
        print(f"[ERROR] Vidya select 정렬 중 TypeError: {e}")
        return []
    tickers = list(common)
    
    # 데이터 완전성 검사
    tickers = [t for t in tickers if has_complete_close_data(t, months_next, base_ohlc_dir)]
    if len(tickers) < 2:
        print(f"[WARN] Vidya select: 유니버스 ticker 개수 2개 미만 (year={year},month={month})")
        return []

    # 2) 가격 로드 및 로그수익률 생성
    prices, _ = build_price_data(tickers, months_form, base_dir=base_ohlc_dir)
    rets = {}
    for tk, px in prices.items():
        r = _log_returns(px)
        if r is not None:
            rets[tk] = r
    tickers = list(rets.keys())
    if len(tickers) < 2:
        print(f"[ERROR] Vidya select: 로그수익률 생성 실패 or 종목 부족 (year={year},month={month})")
        return []

    # 3) 공적분 검정 (Multiprocessing)
    pair_iter = itertools.combinations(tickers, 2)
    args_iter = ((a, b, rets, alpha) for a, b in pair_iter)
    
    passed = []
    with multiprocessing.Pool() as pool:
        for res in pool.imap_unordered(_coint_test_rets, args_iter, chunksize=512):
            if res:
                passed.append(res)
    if not passed:
        print(f"[WARN] Vidya select: 공적분 만족 페어 없음 (year={year},month={month})")
        return []

    # 4) 상관계수 계산 및 정렬
    corr_list = []
    for t1, t2, pval in passed:
        r1, r2 = rets[t1], rets[t2]
        n = min(len(r1), len(r2))
        if n <= 0:
            continue
        rho = np.corrcoef(r1[-n:], r2[-n:])[0,1]
        if np.isfinite(rho):
            corr_list.append((f"{t1}_{t2}", rho, pval))
            
    if not corr_list:
        print(f"[WARN] Vidya select: 상관계수 계산 결과 없음 (year={year},month={month})")
        return []
        
    # 높은 상관계수 우선 정렬 (내림차순)
    corr_list.sort(key=lambda x: -x[1])
    selected = corr_list[:top_n]

    # 5) CSV 저장
    formation_next = months_next[0]
    save_dir = os.path.join(output_dir, formation_next)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_pair_list.csv")
    df_sel = pd.DataFrame(selected, columns=["pair", "corr", "p_value"])
    df_sel.to_csv(save_path, index=False)
    print(f"[INFO] Vidya best pair 저장 → {save_path} (pairs={len(selected)})")
    
    return [pair.split("_") for pair, _, _ in selected]


# ──────────────────────────────────────────────────────────────────────────────
# 페어 트레이딩 시뮬레이션 (Simulation)
# ──────────────────────────────────────────────────────────────────────────────
def load_price_dict(tickers: list, months: list, base_dir: str, field='CLOSE') -> dict:
    """
    Ticker별로 지정된 기간(months)의 특정 필드(field, 예: CLOSE) 데이터를 로드합니다.
    데이터가 없는 경우 해당 기간을 NaN으로 채웁니다.
    """
    temp = {t: [] for t in tickers}
    for m in months:
        path = os.path.join(base_dir, field, f"{m}-{field}.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
            except Exception:
                # 실패 시 빈 시리즈로 대체
                for t in tickers:
                    temp[t].append(pd.Series(dtype=float))
                continue
            for t in tickers:
                if t in df.columns:
                    temp[t].append(df[t])
                else:
                    temp[t].append(pd.Series([np.nan]*len(df), index=df.index))
        else:
            for t in tickers:
                temp[t].append(pd.Series(dtype=float))
    out = {}
    for t, lst in temp.items():
        valid = [s for s in lst if not s.empty]
        if valid:
            out[t] = pd.concat(valid).sort_index()
        else:
            out[t] = pd.Series(dtype=float)
    return out

def _normalize(s: pd.Series):
    """
    시리즈의 첫 유효값으로 나누어 정규화합니다.
    """
    if s is None or s.empty:
        return None
    idx = s.first_valid_index()
    if idx is None or s.loc[idx] == 0 or np.isnan(s.loc[idx]):
        return None
    return s / s.loc[idx]

def simulate_pair_vidya(pa: pd.Series, pb: pd.Series, mu: float, sigma: float,
                        k_entry=0.75, k_exit=0.0, k_stop=3.0):
    """
    Vidyamurthy 전략을 시뮬레이션합니다.
    
    전략 규칙:
      - 진입 (Entry): Spread 편차가 ±k_entry * sigma (기본 0.75) 이상일 때
      - 청산 (Exit): Spread 편차가 0으로 회귀하거나 (기본 0.0),
                     Spread 편차가 ±k_stop * sigma (기본 3.0) 이상일 때 (Stop Loss)

    Args:
        pa, pb (pd.Series): 정규화된 가격 시계열
        mu, sigma (float): Spread의 평균 및 표준편차
        k_entry, k_exit, k_stop (float): 진입/청산/손절 임계값 계수

    Returns:
        tuple: (perf, daily, event_df)
    """
    pm = Pair_Trading()
    events = []
    pos_side = None
    pending = None  # ('open', params) or ('close', None)
    
    # 임계값 계산
    entry = k_entry * sigma
    exit_thr = k_exit * sigma
    stop_thr = k_stop * sigma if k_stop else None

    # 공통 인덱스 정렬
    dates = pa.index.intersection(pb.index).sort_values()
    last_prices = {}
    
    for dt in dates:
        a = pa.loc[dt] if dt in pa.index else np.nan
        b = pb.loc[dt] if dt in pb.index else np.nan
        cur = {'A': a, 'B': b}

        # 1. DELISTING: NaN 발생 시 즉시 청산
        if np.isnan(a) or np.isnan(b):
            if pm.positions:
                pm.close_position(cur)
                events.append({'Date': dt, 'Action': 'DELIST_CLOSE', 'PortVal': pm.portfolio_values[-1]})
            break

        # 2. 예약된 주문 실행 (Open/Close)
        if pending:
            side, prm = pending
            if side == 'open':
                L, S, pl, ps = prm
                pm.open_position(L, S, pl, ps)
                pos_side = +1 if L == 'A' else -1  # 기록용
                events.append({'Date': dt, 'Action': 'OPEN', 'Long': L, 'Short': S,
                               'PriceLong': pl, 'PriceShort': ps, 'PortVal': pm.portfolio_values[-1]})
            else:  # 'close'
                pm.close_position(cur)
                pos_side = None
                events.append({'Date': dt, 'Action': 'CLOSE', 'PortVal': pm.portfolio_values[-1]})
            pending = None

        # 3. Spread 편차 계산
        dev = (a - b) - mu
        abs_dev = abs(dev)

        # 4. 다음날 주문 결정 (Signal Generation)
        if pos_side is None and abs_dev >= entry:
            # 진입 신호: dev > 0 이면 (A 고평가/B 저평가) -> Short A, Long B
            if dev > 0:
                pending = ('open', ('B', 'A', b, a))
            else:
                pending = ('open', ('A', 'B', a, b))
        elif pos_side is not None:
            # 청산 신호: Stop Loss 또는 이익 실현(Mean Reversion)
            if stop_thr is not None and abs_dev >= stop_thr:
                pending = ('close', None)
            elif abs_dev <= exit_thr:
                pending = ('close', None)

        # 5. 매일 포트폴리오 가치 갱신 (MTM)
        pm.hold_position(cur)
        events.append({'Date': dt, 'Action': 'HOLD', 'PortVal': pm.portfolio_values[-1]})

        last_prices = {'A': a, 'B': b}

    # 6. 만기 강제 청산
    if pm.positions:
        last = {}
        if len(dates) > 0:
            last_dt = dates[-1]
            a = pa.loc[last_dt] if last_dt in pa.index else np.nan
            b = pb.loc[last_dt] if last_dt in pb.index else np.nan
            last = {'A': a, 'B': b}
        pm.close_position(last)
        events.append({'Date': dates[-1] if len(dates)>0 else None, 'Action': 'FORCE_CLOSE',
                       'PortVal': pm.portfolio_values[-1]})

    perf = pm.finalize_performance()
    
    # 일별 NAV 시리즈 생성
    pv = pm.portfolio_values
    n_pv = len(pv)
    n_dates = len(dates)
    if n_pv == 0 or n_dates == 0:
        daily = pd.Series(dtype=float)
    else:
        n = min(n_pv, n_dates)
        idx = dates[:n]
        daily = pd.Series(pv[:n], index=idx, name='PortVal')
    return perf, daily, pd.DataFrame(events)


def run_pair_trading_vidya(pair_list, months_form, months_trade, ym_tag, output_root):
    """
    선정된 페어 리스트에 대해 Vidyamurthy 전략 트레이딩을 수행하고 결과를 저장합니다.

    Args:
        pair_list (list): 트레이딩할 페어 리스트 [(t1, t2), ...]
        months_form (list): Formation 기간 리스트
        months_trade (list): Trading 기간 리스트
        ym_tag (str): 결과 파일명 태그 (예: 'YYYY-MM')
        output_root (str): 결과 저장 경로
    """
    if not pair_list:
        print(f"[WARN] run_pair_trading_vidya: pair_list 비어있음 for {ym_tag}")
        return

    tickers = sorted({t for pr in pair_list for t in pr})
    pf = load_price_dict(tickers, months_form, OHLC_DIR, 'CLOSE')
    pt = load_price_dict(tickers, months_trade, OHLC_DIR, 'CLOSE')

    all_res = {}
    all_rets = {}

    for (p1, p2) in tqdm(pair_list, desc=f"Trade {ym_tag}"):
        s1_form = _normalize(pf.get(p1, pd.Series(dtype=float)))
        s2_form = _normalize(pf.get(p2, pd.Series(dtype=float)))
        if s1_form is None or s2_form is None:
            continue
        idx_common = s1_form.index.intersection(s2_form.index).sort_values()
        if idx_common.empty:
            continue
        
        # Formation 기간 통계량
        spread = s1_form.loc[idx_common] - s2_form.loc[idx_common]
        mu = spread.mean()
        sigma = spread.std(ddof=0)
        
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
        
        # 시뮬레이션 수행
        perf, daily, _ = simulate_pair_vidya(pa, pb, mu, sigma)
        key = f"{p1}_{p2}"
        all_res[key] = perf
        if not daily.empty:
            all_rets[key] = daily.pct_change().fillna(0)

    # 결과 저장
    perf_dir = output_root
    ret_dir = output_root + "_DAILY"
    os.makedirs(perf_dir, exist_ok=True)
    os.makedirs(ret_dir, exist_ok=True)

    # 일별 수익률 저장
    if all_rets:
        df_rets = pd.DataFrame(all_rets).rename_axis("Date")
        df_rets.to_csv(os.path.join(ret_dir, f"{ym_tag}_Return.csv"))
    
    # 성과 요약 저장
    metrics = ["Count", "ROI", "Sharpe Ratio", "Cumulative Return",
               "Sortino Ratio", "Maximum Drawdown", "Calmar Ratio",
               "Volatility", "Hit Ratio"]
    with open(os.path.join(perf_dir, f"{ym_tag}_all_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker1", "ticker2"] + metrics)
        for pair, stats in all_res.items():
            t1, t2 = pair.split("_")
            row = [stats.get(m, "") for m in metrics]
            writer.writerow([t1, t2] + row)
    print(f"[INFO] run_pair_trading_vidya 완료: {ym_tag}, pairs={len(all_res)}")


# ──────────────────────────────────────────────────────────────────────────────
# 메인 실행 함수 (Main Execution)
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # S&P500 유니버스 로드
    print("[INFO] S&P500 유니버스 로드:", UNIV_PATH)
    df_univ, monthly_univ = load_snp500_universe(UNIV_PATH)

    # 1. TRAINING PERIOD (현재 주석 처리된 상태)
    print("[INFO] TRAINING 시작: start=", INIT_YEAR, INIT_MONTH, "months=", TRAIN_MONTHS)
    for offset in range(1, TRAIN_MONTHS):
        dt = datetime(INIT_YEAR, INIT_MONTH, 1) + pd.DateOffset(months=offset)
        y, m = dt.year, dt.month
        period = dt.strftime("%Y-%m")
        formation_next = (pd.Period(period, freq="M") + 1).strftime("%Y-%m")

        """
        # 이미 결과가 있으면 건너뜀
        if os.path.exists(os.path.join(OUTPUT_ROOT, f"{period}_all_results.csv")):
            print(f"[SKIP] 이미 결과 존재: {period}")
            continue

        print(f"[TRAIN] Vidyamurthy 페어 선정 및 트레이딩: {period}")
        pair_list = vidyamurthy_select(
            year=y, month=m, df_univ=df_univ,
            base_ohlc_dir=OHLC_DIR, top_n=TOP_K_TRAIN,
            alpha=ALPHA, output_dir=BEST_PAIR_ROOT
        )
        if pair_list:
            months_form = last_n_months(y, m, 12)
            months_trade = next_n_months(y, m, 6)
            run_pair_trading_vidya(pair_list, months_form, months_trade, period, OUTPUT_ROOT)
        """
        
    # 2. TESTING PERIOD
    print("[INFO] TESTING 시작: start offset=", TRAIN_MONTHS, "months=", TEST_MONTHS)
    for offset in range(TRAIN_MONTHS, TRAIN_MONTHS + TEST_MONTHS):
        dt = datetime(INIT_YEAR, INIT_MONTH, 1) + pd.DateOffset(months=offset)
        y, m = dt.year, dt.month
        period = dt.strftime("%Y-%m")

        # 이미 결과가 있으면 건너뜀
        if os.path.exists(os.path.join(OUTPUT_ROOT, f"{period}_all_results.csv")):
            print(f"[SKIP TEST] 이미 결과 존재: {period}")
            continue

        print(f"[TEST] Vidyamurthy 페어 선정 및 트레이딩: {period}")
        # trading 직전 월 기준으로 선정
        dt_prior = datetime(INIT_YEAR, INIT_MONTH, 1) + pd.DateOffset(months=offset-1)
        pair_list = vidyamurthy_select(
            year=dt_prior.year, month=dt_prior.month, df_univ=df_univ,
            base_ohlc_dir=OHLC_DIR, top_n=TOP_K_TEST,
            alpha=ALPHA, output_dir=BEST_PAIR_ROOT
        )
        if pair_list:
            months_form = last_n_months(dt_prior.year, dt_prior.month, 12)
            months_trade = next_n_months(dt_prior.year, dt_prior.month, 6)
            run_pair_trading_vidya(pair_list, months_form, months_trade, period, OUTPUT_ROOT)

if __name__ == "__main__":
    # multiprocessing 필요 시 설정
    multiprocessing.set_start_method('fork', force=True)
    main()