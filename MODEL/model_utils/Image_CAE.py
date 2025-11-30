# MODEL/model_utils/Image_CAE.py
# python3 ./MODEL/model_utils/Image_CAE.py --target_month 1991-02

import os
import time
import logging
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

# ──────────────────────────────────────────────────────────────────────────────
# 로깅 및 전역 설정 (Logging & Configuration)
# ──────────────────────────────────────────────────────────────────────────────
LOG_FILENAME = "process.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILENAME, mode='a', encoding='utf-8')
    ]
)

# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티 함수 (Utility Functions)
# ──────────────────────────────────────────────────────────────────────────────
def sanitize_filename(name: str) -> str:
    """
    파일 이름으로 사용할 수 없는 특수 문자를 언더바(_)로 치환합니다.

    Args:
        name (str): 원본 파일 이름

    Returns:
        str: 정제된 파일 이름
    """
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(ch, '_')
    return name

def process_ticker_task(args):
    """
    단일 종목(Ticker)에 대해 캔들스틱 차트 이미지를 생성하고 저장하는 작업 함수입니다.
    ProcessPoolExecutor에서 호출됩니다.

    Args:
        args (tuple): (ticker, full_open, full_high, full_low, full_close, out_dir, window_size)
            - ticker (str): 종목 코드
            - full_open (pd.DataFrame): 전체 시가 데이터
            - full_high (pd.DataFrame): 전체 고가 데이터
            - full_low (pd.DataFrame): 전체 저가 데이터
            - full_close (pd.DataFrame): 전체 종가 데이터
            - out_dir (str): 이미지가 저장될 기본 경로
            - window_size (int): 롤링 윈도우 크기 (이미지 한 장당 포함될 일수)
    """
    ticker, full_open, full_high, full_low, full_close, out_dir, window_size = args
    ticker_s = sanitize_filename(ticker)
    
    # 해당 종목의 데이터프레임 구성
    df = pd.DataFrame({
        'Open':  full_open.get(ticker, pd.Series(index=full_open.index)),
        'High':  full_high.get(ticker, pd.Series(index=full_high.index)),
        'Low':   full_low.get(ticker, pd.Series(index=full_low.index)),
        'Close': full_close.get(ticker, pd.Series(index=full_close.index)),
    }).sort_index()

    tgt = os.path.join(out_dir, ticker_s)
    os.makedirs(tgt, exist_ok=True)

    # 롤링 윈도우 방식으로 이미지 생성
    for i in range(window_size, len(df) + 1):
        win = df.iloc[i-window_size:i]
        
        # 결측치가 하나라도 있으면 건너뜀
        if win.isna().any().any():
            continue
            
        try:
            # mpf 차트 생성 (축 제거, 캔들 차트)
            fig, _ = mpf.plot(win, type='candle', style='charles',
                              volume=False, returnfig=True, axisoff=True)
            fig.set_size_inches(2, 2)
            fig.set_dpi(32)
            
            # 이미지 저장
            save_path = os.path.join(tgt, f"{ticker_s}_{i-window_size}.png")
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
        except Exception as e:
            logging.error(f"{ticker_s} 에러: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 메인 로직 함수 (Core Logic Functions)
# ──────────────────────────────────────────────────────────────────────────────
def process_month(
    target_month,
    base_dir='./DATA/OHLC_DATA',
    output_dir='./DATA/IMAGE_CAE',
    window_size=21
):
    """
    특정 월(target_month)을 기준으로 과거 12개월 데이터를 로드하여 
    티커별 차트 이미지를 생성합니다.

    Args:
        target_month (str): 처리할 대상 월 ('YYYY-MM')
        base_dir (str): OHLC 데이터가 있는 CSV 폴더 경로
        output_dir (str): 생성된 이미지를 저장할 경로
        window_size (int): 이미지 생성 시 사용할 윈도우 크기
    """
    end_dt   = datetime.strptime(target_month + '-01', '%Y-%m-%d')
    start_dt = end_dt - relativedelta(months=12)

    # YYYY-MM 리스트 생성 (12개월 전 ~ 타겟 월)
    months = []
    cur = start_dt
    while cur <= end_dt:
        months.append(cur.strftime('%Y-%m'))
        cur += relativedelta(months=1)

    logging.info(f"[{target_month}] {months[0]} ~ {months[-1]} 처리 시작")
    start = time.time()

    # 1. 월별 CSV 데이터 로드 및 병합 준비
    data = {'open':[], 'high':[], 'low':[], 'close':[]}
    for m in months:
        for k in ['open','high','low','close']:
            path = os.path.join(base_dir, k.upper(), f"{m}-{k.upper()}.csv")
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
                except Exception as e:
                    logging.error(f"[{m}] {k.upper()} 읽기 실패: {e}")
                    df = pd.DataFrame()
            else:
                logging.warning(f"[{m}] {k.upper()} 파일 누락")
                df = pd.DataFrame()
            data[k].append(df)

    # 2. 전체 데이터 병합 (Concat)
    full_open  = pd.concat(data['open']).sort_index()
    full_high  = pd.concat(data['high']).sort_index()
    full_low   = pd.concat(data['low']).sort_index()
    full_close = pd.concat(data['close']).sort_index()

    # 출력 폴더 생성
    out_dir = os.path.join(output_dir, target_month)
    os.makedirs(out_dir, exist_ok=True)

    # 3. 처리할 전체 티커 집합 추출
    all_tickers = set(full_open.columns) | set(full_high.columns) | set(full_low.columns) | set(full_close.columns)
    if not all_tickers:
        logging.error(f"[{target_month}] 처리할 티커 없음")
        return

    # 4. 병렬 처리 작업 준비 및 실행
    tasks = [
        (t, full_open, full_high, full_low, full_close, out_dir, window_size)
        for t in sorted(all_tickers)
    ]
    
    # max_workers=128은 시스템 리소스에 따라 조정 가능
    with ProcessPoolExecutor(max_workers=128) as exe:
        futures = {exe.submit(process_ticker_task, a): a[0] for a in tasks}
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"{target_month} 티커 처리"):
            pass

    logging.info(f"[{target_month}] 완료 ({time.time()-start:.2f}s)")

def process_all_months(month_list, **kwargs):
    """
    주어진 월 리스트에 대해 순차적으로 process_month를 실행합니다.

    Args:
        month_list (list): ['YYYY-MM', ...] 형태의 리스트
        **kwargs: process_month 함수에 전달할 키워드 인자들 (base_dir, output_dir 등)
    """
    for m in month_list:
        process_month(m, **kwargs)

# ──────────────────────────────────────────────────────────────────────────────
# 실행 진입점 (Main Execution)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_month', type=str, default="", help="e.g., 1991-02")
    parser.add_argument('--start_year', type=int, default=2004)
    parser.add_argument('--end_year', type=int, default=2024)
    parser.add_argument('--base_dir', type=str, default='./DATA/OHLC_DATA')
    parser.add_argument('--output_dir', type=str, default='./DATA/IMAGE_CAE')
    parser.add_argument('--window_size', type=int, default=21)
    args = parser.parse_args()

    if args.target_month:
        # 특정 1개월만 실행
        process_month(
            args.target_month,
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            window_size=args.window_size
        )
    else:
        # start_year ~ end_year 전체 기간 실행
        months_to_run = []
        for year in range(args.start_year, args.end_year + 1):
            for month in range(1, 13):
                months_to_run.append(f"{year}-{month:02d}")
        
        process_all_months(
            months_to_run,
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            window_size=args.window_size
        )