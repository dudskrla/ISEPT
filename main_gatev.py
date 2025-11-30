# main_gatev.py

"""
GATEV 방식의 페어 선정(Pair Selection)과 페어 트레이딩(Pair Trading)

실행 예시:
    python3 main_gatev.py --start_year 2004 --start_month 1 \
        --train_months 156 --test_months 252 \
        --top_k_train 100 --top_k_test 20 \
        --ohlc_dir ./DATA/OHLC_DATA \
        --universe_path ./DATA/SNP500_UNIVERSE.xlsx \
        --best_pair_root ./BEST_PAIR/TRAIN_ONE/GATEV \
        --output_root ./PAIR_RESULT/TRAIN_ONE/GATEV
"""

import os
import csv
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Optional
from datetime import datetime
from pandas.tseries.offsets import MonthEnd

# ──────────────────────────────────────────────────────────────────────────────
# argparse 설정 및 상수 정의
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GATEV 방식 페어 선정 및 페어 트레이딩 수행")
parser.add_argument('--start_year',  type=int, default=1991, help="초기 연도")
parser.add_argument('--start_month', type=int, choices=range(1,13), default=1, help="초기 월")
parser.add_argument('--train_months', type=int, default=156, help="트레이닝 기간 개월수")
parser.add_argument('--test_months',  type=int, default=252, help="테스트 기간 개월수")
parser.add_argument('--top_k_train', type=int, default=100, help="트레이닝 시 GATEV 상위 N 페어 개수")
parser.add_argument('--top_k_test',  type=int, default=20, help="테스트 시 GATEV 상위 N 페어 개수")
parser.add_argument('--ohlc_dir', type=str, default='./DATA/OHLC_DATA', help="OHLC 데이터 루트 디렉토리")
parser.add_argument('--universe_path', type=str, default='./DATA/SNP500_UNIVERSE.xlsx', help="S&P500 유니버스 엑셀 파일 경로")
parser.add_argument('--best_pair_root', type=str, default='./BEST_PAIR/TRAIN_ONE/GATEV', help="GATEV best_pair 저장 루트 디렉토리")
parser.add_argument('--output_root', type=str, default='./PAIR_RESULT/TRAIN_ONE/GATEV', help="페어 트레이딩 결과 저장 루트 디렉토리")
args = parser.parse_args()

INIT_YEAR      = args.start_year
INIT_MONTH     = args.start_month
TRAIN_MONTHS   = args.train_months
TEST_MONTHS    = args.test_months
TOP_K_TRAIN    = args.top_k_train
TOP_K_TEST     = args.top_k_test
OHLC_DIR       = args.ohlc_dir.rstrip("/")
UNIV_PATH      = args.universe_path
BEST_PAIR_ROOT = args.best_pair_root.rstrip("/")
OUTPUT_ROOT    = args.output_root.rstrip("/")

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
            - df_univ (pd.DataFrame): ['Code', 'Month_End', 'Month'] 컬럼을 가진 데이터프레임
            - monthly_univ (dict): {'YYYY-MM': [Code 목록], ...} 형태의 딕셔너리

    Raises:
        RuntimeError: 파일에서 데이터를 읽지 못했을 경우 발생
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
    예: (1991, 1), n=12 -> ['1989-01', ..., '1990-12']
    """
    end = pd.Period(f"{year}-{month:02d}", freq="M") - 1
    return [(end - i).strftime("%Y-%m") for i in reversed(range(n))]

def next_n_months(year: int, month: int, n: int) -> list:
    """
    지정된 (year, month) 이후 n개월(해당 월 포함)의 리스트를 반환합니다.
    예: (1991, 1), n=6 -> ['1991-02', ..., '1991-07']
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
    
    경로 포맷: base_dir/CLOSE/{YYYY-MM}-CLOSE.csv
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

def load_close_series_dict(tickers: list, months: list, base_dir: str) -> dict:
    """
    Ticker별로 지정된 기간(months)의 CLOSE 데이터를 연결(Concat)하여 반환합니다.

    Returns:
        dict: {ticker: pd.Series, ...} 형태 (데이터가 없으면 빈 Series)
    """
    monthly_dfs = {}
    for m in months:
        fn = os.path.join(base_dir, "CLOSE", f"{m}-CLOSE.csv")
        if os.path.exists(fn):
            try:
                df = pd.read_csv(fn, index_col="Date", parse_dates=["Date"])
                monthly_dfs[m] = df
            except Exception:
                pass
    out = {}
    for t in tickers:
        pieces = []
        for m in months:
            if m in monthly_dfs and t in monthly_dfs[m].columns:
                pieces.append(monthly_dfs[m][t])
        if pieces:
            s = pd.concat(pieces).sort_index()
            out[t] = s
        else:
            out[t] = pd.Series(dtype=float)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# GATEV 페어 선정 (Selection)
# ──────────────────────────────────────────────────────────────────────────────
def gatev_select(year: int,
                 month: int,
                 df_univ: pd.DataFrame,
                 monthly_univ: dict,
                 ohlc_dir: str,
                 top_n: int,
                 output_dir: str) -> list:
    """
    GATEV 방식으로 페어를 선정합니다.
    Formation 기간(12개월)의 정규화된 가격 거리(SSD)를 기준으로 상위 top_n 페어를 선택합니다.

    Args:
        year (int): 기준 연도
        month (int): 기준 월
        df_univ (pd.DataFrame): 유니버스 전체 데이터프레임
        monthly_univ (dict): 월별 유니버스 딕셔너리
        ohlc_dir (str): OHLC 데이터 경로
        top_n (int): 선정할 페어의 수
        output_dir (str): 선정 결과를 저장할 경로

    Returns:
        list: 선정된 페어 튜플 리스트 [(t1, t2), ...]
    """
    months_form = last_n_months(year, month, 12)
    months_check6 = next_n_months(year, month, 6)
    if not months_form or not months_check6:
        return []
    f0 = months_form[0]
    c0 = months_check6[0]

    # 유니버스 교집합: Formation 시작월(f0)과 Trading 시작월(c0) 모두 존재하는 종목
    A_raw = df_univ.loc[df_univ["Month"] == f0, "Code"].dropna()
    B_raw = df_univ.loc[df_univ["Month"] == c0, "Code"].dropna()
    A = {t for t in set(A_raw) if isinstance(t, str) and t.strip() != ""}
    B = {t for t in set(B_raw) if isinstance(t, str) and t.strip() != ""}
    try:
        common = A & B
        tickers = sorted(common)
    except TypeError as e:
        print(f"[ERROR] GATEV select 정렬 중 TypeError: {e}")
        return []
        
    # 데이터 완전성 검사
    tickers = [t for t in tickers if has_complete_close_data(t, months_check6, ohlc_dir)]
    if len(tickers) < 2:
        print(f"[WARN] GATEV select: 유니버스 ticker 개수 2개 미만 (year={year},month={month})")
        return []

    # 가격 데이터 로드 및 정규화
    raw = load_close_series_dict(tickers, months_form, ohlc_dir)
    norm = {}
    for t, series in raw.items():
        if not series.empty:
            first_idx = series.first_valid_index()
            if first_idx is not None and series.loc[first_idx] != 0:
                norm[t] = series / series.loc[first_idx]
    keys = list(norm.keys())
    n = len(keys)
    if n < 2:
        print(f"[WARN] GATEV select: 정규화 이후 ticker 개수 2개 미만 (year={year},month={month})")
        return []

    # 거리 행렬 계산 (Brute-force SSD)
    D = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i+1, n):
            s1 = norm[keys[i]]
            s2 = norm[keys[j]]
            valid = ~s1.isna() & ~s2.isna()
            s1v = s1[valid]; s2v = s2[valid]
            if len(s1v) > 0:
                dist = np.sum((s1v.values - s2v.values)**2)
                D[i,j] = dist; D[j,i] = dist
            else:
                D[i,j] = np.inf; D[j,i] = np.inf

    # 거리 오름차순 정렬하여 상위 top_n 페어 선택
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((keys[i], keys[j], D[i,j]))
    pairs = sorted(pairs, key=lambda x: x[2])[:top_n]

    # CSV 저장: output_dir/{formation_next}/best_pair_list.csv
    formation_next = months_check6[0]
    save_dir = os.path.join(output_dir, formation_next)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_pair_list.csv")
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair", "distance"])
        for t1, t2, dist in pairs:
            writer.writerow([f"{t1}_{t2}", f"{dist:.6f}"])
    print(f"[INFO] GATEV best pair 저장 → {save_path} (pairs={len(pairs)})")
    return [(t1, t2) for t1, t2, _ in pairs]

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
            s_all = pd.concat(valid).sort_index()
            out[t] = s_all
        else:
            out[t] = pd.Series(dtype=float)
    return out

def _normalize(s: pd.Series) -> Optional[pd.Series]:
    """
    시리즈의 첫 유효값으로 나누어 정규화합니다.
    유효값이 없거나 0인 경우 None을 반환합니다.
    """
    if s is None or s.empty:
        return None
    idx = s.first_valid_index()
    if idx is None or s.loc[idx] == 0:
        return None
    return s / s.loc[idx]

def simulate_pair_gatev(pa: pd.Series, pb: pd.Series, mu: float, sigma: float):
    """
    단일 페어에 대해 GATEV 전략(Standard Distance Method)을 시뮬레이션합니다.
    
    전략:
        - Spread가 Mean ± 2*Sigma를 벗어나면 진입 (Open)
        - Spread가 Mean으로 회귀하면 청산 (Close)
        - 데이터가 끊기면(Delisting) 즉시 강제 청산

    Args:
        pa (pd.Series): 종목 A의 정규화된 가격 시계열
        pb (pd.Series): 종목 B의 정규화된 가격 시계열
        mu (float): Formation 기간 Spread의 평균
        sigma (float): Formation 기간 Spread의 표준편차

    Returns:
        tuple: (perf, daily, event_df)
            - perf (dict): 최종 성과 지표 (ROI, MDD, Sharpe 등)
            - daily (pd.Series): 일별 포트폴리오 가치
            - event_df (pd.DataFrame): 트레이딩 이벤트 로그
    """
    from MODEL.model_utils.Pair_Trading import Pair_Trading  # 실제 경로에 맞게 조정
    pm = Pair_Trading()
    events = []
    last_open = None
    prev_dev = None
    pending_open = False
    open_params = None
    pending_close = False

    # 공통 인덱스 정렬
    dates = pa.index.intersection(pb.index).sort_values()
    last_prices = {}

    for dt in dates:
        a = pa.loc[dt] if dt in pa.index else np.nan
        b = pb.loc[dt] if dt in pb.index else np.nan

        # 1. DELISTING: 가격 결측 시 즉시 청산
        if np.isnan(a) or np.isnan(b):
            if pm.positions:
                prices = {'A': last_prices.get('A'), 'B': last_prices.get('B')}
                pm.close_position(prices)
                L, S = last_open
                price_long = last_prices.get(L)
                price_short = last_prices.get(S)
                events.append({
                    'Date': dt, 'Action': 'DELIST_CLOSE',
                    'Long': L, 'Short': S,
                    'PriceLong': price_long, 'PriceShort': price_short,
                    'PortVal': pm.portfolio_values[-1]
                })
            break

        # 2. 예약된 OPEN 실행
        if pending_open:
            L, S, pl, ps = open_params
            pm.open_position(L, S, pl, ps)
            events.append({
                'Date': dt, 'Action': 'OPEN',
                'Long': L, 'Short': S,
                'PriceLong': pl, 'PriceShort': ps,
                'PortVal': pm.portfolio_values[-1]
            })
            last_open = (L, S)
            pending_open = False

        # 3. 스프레드 편차 계산
        dev = (a - b) - mu
        cur_prices = {'A': a, 'B': b}

        # 4. 포지션 청산 조건 (평균 회귀: 편차의 부호가 바뀔 때)
        if pm.positions and prev_dev is not None and dev * prev_dev < 0:
            pending_close = True

        # 5. 포지션 진입 조건 (±2 Sigma 돌파 시 다음날 진입)
        if not pm.positions and abs(dev) > 2 * sigma:
            if dev > 0:
                open_params = ('B', 'A', b, a)
            else:
                open_params = ('A', 'B', a, b)
            pending_open = True

        # 6. 예약된 CLOSE 실행
        if pending_close:
            pm.close_position(cur_prices)
            L, S = last_open
            events.append({
                'Date': dt, 'Action': 'CLOSE',
                'Long': L, 'Short': S,
                'PriceLong': cur_prices[L], 'PriceShort': cur_prices[S],
                'PortVal': pm.portfolio_values[-1]
            })
            pending_close = False

        # 7. HOLD 업데이트
        if pm.positions:
            pm.hold_position(cur_prices)
            events.append({'Date': dt, 'Action': 'HOLD', 'PortVal': pm.portfolio_values[-1]})
        else:
            events.append({'Date': dt, 'Action': 'HOLD', 'PortVal': pm.portfolio_values[-1]})

        prev_dev = dev
        last_prices = {'A': a, 'B': b}

    # 8. 만기 강제 청산 (Trading Period 종료)
    if pm.positions:
        pm.close_position(last_prices)
        L, S = last_open
        price_long = last_prices.get(L)
        price_short = last_prices.get(S)
        events.append({
            'Date': dates[-1], 'Action': 'FORCE_CLOSE',
            'Long': L, 'Short': S,
            'PriceLong': price_long, 'PriceShort': price_short,
            'PortVal': pm.portfolio_values[-1]
        })

    # 9. 성과 및 일별 시리즈 반환
    pv = pm.portfolio_values
    n_pv = len(pv)
    n_dates = len(dates)
    if n_pv == 0:
        daily = pd.Series(dtype=float)
    else:
        # 날짜와 포트폴리오 값 길이 동기화
        n = min(n_pv, n_dates)
        idx = dates[:n]
        daily = pd.Series(pv[:n], index=idx, name='PortVal')
    perf = pm.finalize_performance()
    return perf, daily, pd.DataFrame(events)

def run_pair_trading_gatev(pair_list: list, months_form: list, months_trade: list, ym_tag: str, output_root: str):
    """
    선정된 페어 리스트에 대해 GATEV 트레이딩을 수행하고 결과를 저장합니다.

    Args:
        pair_list (list): 트레이딩할 페어 리스트 [(t1, t2), ...]
        months_form (list): Formation 기간 리스트
        months_trade (list): Trading 기간 리스트
        ym_tag (str): 결과 파일명에 사용할 태그 (예: 'YYYY-MM')
        output_root (str): 결과 저장 경로
    """
    if not pair_list:
        print(f"[WARN] run_pair_trading_gatev: pair_list가 비어있음 for {ym_tag}")
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
        
        # Formation 기간의 통계량 계산
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
        perf, daily, ev = simulate_pair_gatev(pa, pb, mu, sigma)
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
    print(f"[INFO] run_pair_trading_gatev 완료: {ym_tag}, pairs={len(all_res)}")

# ──────────────────────────────────────────────────────────────────────────────
# 메인 실행 함수 (Main Execution)
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # S&P500 유니버스 로드
    print("[INFO] S&P500 유니버스 로드:", UNIV_PATH)
    df_univ, monthly_univ = load_snp500_universe(UNIV_PATH)

    # 1. TRAINING PERIOD (현재 주석 처리된 상태 유지)
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

        print(f"[TRAIN] GATEV 페어 선정 및 트레이딩: {period}")
        pair_list = gatev_select(year=y, month=m, df_univ=df_univ, monthly_univ=monthly_univ,
                                 ohlc_dir=OHLC_DIR, top_n=TOP_K_TRAIN, output_dir=BEST_PAIR_ROOT)
        if pair_list:
            months_form = last_n_months(y, m, 12)
            months_trade = next_n_months(y, m, 6)
            run_pair_trading_gatev(pair_list, months_form, months_trade, period, OUTPUT_ROOT)
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

        print(f"[TEST] GATEV 페어 선정 및 트레이딩: {period}")
        dt_prior = datetime(INIT_YEAR, INIT_MONTH, 1) + pd.DateOffset(months=offset-1)
        pair_list = gatev_select(year=dt_prior.year, month=dt_prior.month,
                                 df_univ=df_univ, monthly_univ=monthly_univ,
                                 ohlc_dir=OHLC_DIR, top_n=TOP_K_TEST, output_dir=BEST_PAIR_ROOT)
        if pair_list:
            months_form = last_n_months(dt_prior.year, dt_prior.month, 12)
            months_trade = next_n_months(dt_prior.year, dt_prior.month, 6)
            run_pair_trading_gatev(pair_list, months_form, months_trade, period, OUTPUT_ROOT)

if __name__ == "__main__":
    main()