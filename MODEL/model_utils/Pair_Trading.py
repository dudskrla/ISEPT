# MODEL/model_utils/Pair_Trading.py

import numpy as np

class Pair_Trading:
    """
    Gatev 스타일의 단일 페어 트레이딩(Single-pair Tracker) 시뮬레이션 클래스입니다.

    설정값:
        GROSS_EXPOSURE (float): 2.0 (Long $1 + Short $1)
        COMM_RATE (float): 0.0001 (거래대금의 0.01%, 편도 1bp)
                           참고: A Cooperative Dynamic Approach to Pairs Trading
    """
    GROSS_EXPOSURE = 2.0
    COMM_RATE      = 0.0001

    def __init__(self):
        self.realized_pnl  = 0.0
        self.positions     = []          # 현재 보유 중인 포지션 리스트 (한 번에 하나의 페어만 가정)
        self.portfolio_pnl = [0.0]       # 달러 기준 누적 P/L (초기값 0$)
        self.history       = []          # 거래 기록 로그
        self.count         = 0           # 총 트레이드 횟수

        # 하위 호환성을 위한 별칭(Alias)
        self.unrealized_pnl   = self.portfolio_pnl
        self.portfolio_values = self.portfolio_pnl

    # ────────────── 1) 진입 (OPEN) ──────────────
    def open_position(self, long_tkr, short_tkr, pl, ps):
        """
        롱/숏 포지션을 각각 $1씩 진입합니다. (총 노출 $2)
        수수료를 계산하여 실현 손익에서 즉시 차감하고, 히스토리를 기록합니다.

        Args:
            long_tkr (str): 매수(Long)할 종목 코드
            short_tkr (str): 매도(Short)할 종목 코드
            pl (float): 매수 종목의 진입 가격
            ps (float): 매도 종목의 진입 가격
        """
        notional = pl + ps                     # 진입 시 명목 거래 금액 (각 1주 가격 기준이 아니라 로직상 비율)
                                               # *주의: 실제 코드는 아래 amt 계산에서 $1에 맞춰 수량을 조절함
        
        notional = pl + ps 
        commission = self.COMM_RATE * notional 

        self.positions = [dict(
            long=long_tkr, short=short_tkr,
            price_long=pl, price_short=ps,
            amt_long=1.0/pl, amt_short=1.0/ps     # 각 $1 어치 매수/매도
        )]
        self.count += 1

        # 진입 시 수수료 즉시 차감
        self.realized_pnl -= commission

        self.history.append(
            f"[OPEN] long {long_tkr} @ {pl:.4f} / "
            f"short {short_tkr} @ {ps:.4f}  (commission {commission:.5f}$)"
        )
        
        # 진입 직후 시가평가(Mark-to-Market) 수행
        self.mark_to_market({long_tkr: pl, short_tkr: ps})

    # ────────────── 2) 청산 (CLOSE) ──────────────
    def close_position(self, cur):
        """
        현재가(cur)를 기준으로 보유 포지션을 전량 청산합니다.

        Args:
            cur (dict): {ticker: price, ...} 형태의 현재가 딕셔너리

        Returns:
            float: 수수료 차감 후 순 실현 손익(Net PnL)
        """
        if not self.positions:
            return 0.0

        p = self.positions[0]
        
        # 손익 계산: (매도 - 매수) * 수량
        # Long: (현재가 - 진입가)
        # Short: (진입가 - 현재가)
        pl_long  = (cur[p["long"]]  - p["price_long"])  * p["amt_long"]
        pl_short = (p["price_short"] - cur[p["short"]]) * p["amt_short"]
        gross_pnl = pl_long + pl_short

        # 거래 대금 및 수수료 계산
        notional   = cur[p["long"]] + cur[p["short"]]  # 원본 로직 유지 (단순 가격 합 사용)
        commission = self.COMM_RATE * notional
        net_pnl    = gross_pnl - commission

        self.realized_pnl += net_pnl
        self.history.append(
            f"[CLOSE] gross {gross_pnl:.5f}$  − commission {commission:.5f}$"
            f"  = net {net_pnl:.5f}$"
        )

        self.positions.clear()
        
        # 청산 직후 시가평가 수행
        self.mark_to_market(cur)
        return net_pnl

    # ────────────── 3) 평가 (MTM) ──────────────
    def mark_to_market(self, cur):
        """
        현재가를 반영하여 미실현 손익을 포함한 포트폴리오 가치를 갱신합니다.

        Args:
            cur (dict): {ticker: price, ...} 형태의 현재가 딕셔너리
        """
        unreal = 0.0
        for p in self.positions:
            unreal += (cur[p["long"]]  - p["price_long"])  * p["amt_long"]
            unreal += (p["price_short"] - cur[p["short"]]) * p["amt_short"]
        
        # 누적 P/L 기록 (실현 손익 + 미실현 손익)
        self.portfolio_pnl.append(self.realized_pnl + unreal)

    hold_position = mark_to_market   # Alias for backward compatibility

    # ────────────── 4) 성과 지표 (Performance) ──────────────
    def finalize_performance(self, rf: float = 0.0):
        """
        트레이딩 종료 후 최종 성과 지표를 계산합니다.

        Args:
            rf (float): 무위험 이자율 (Risk-free rate), 기본값 0.0

        Returns:
            dict: ROI, Sharpe Ratio, MDD 등 주요 성과 지표가 담긴 딕셔너리
        """
        TRADING_DAYS = 252
        pnl = np.asarray(self.portfolio_pnl, dtype=float)

        # 데이터 부족 시 NaN 반환
        if len(pnl) < 2:
            n, z = float('nan'), 0.0
            return {"Count": self.count, "ROI": z, "Sharpe Ratio": n,
                    "Cumulative Return": z, "Sortino Ratio": n,
                    "Maximum Drawdown": z, "Calmar Ratio": z,
                    "Volatility": n, "Hit Ratio": z}

        # NAV(순자산가치) 계산 (Gross Exposure 대비 수익률)
        nav = 1.0 + pnl / self.GROSS_EXPOSURE

        # ── [핵심 패치] NAV 유효성 검사 ──
        # NAV가 0, 음수, 또는 NaN이 되는 구간은 수익률 계산을 왜곡하므로 필터링
        nav = np.asarray(nav, dtype=float)
        valid_nav = np.isfinite(nav) & (nav > 0)

        # 연율화(Annualization)를 위한 전체 기간
        n_total = len(nav) - 1

        # 수익률(Return) 계산: 연속된 두 시점 모두 NAV가 유효한 경우만 사용
        ret_raw = np.diff(nav) / nav[:-1]
        valid_ret = valid_nav[:-1] & valid_nav[1:] & np.isfinite(ret_raw)
        ret = ret_raw[valid_ret]
        n = len(ret)

        # 누적 수익률(Cumulative Return): 마지막 유효 NAV 기준
        if valid_nav.any():
            nav_last = nav[valid_nav][-1]
            cum_ret = nav_last - 1.0
        else:
            # 전 구간이 유효하지 않음 (파산 등)
            cum_ret = -1.0

        # ROI (연환산 수익률)
        annual_roi = (1 + cum_ret)**(TRADING_DAYS / max(n_total, 1)) - 1 if cum_ret > -1 else -1.0

        # 변동성 및 샤프 지수 계산
        if n < 2:
            sharpe = np.nan
            vol = np.nan
            sortino = np.nan
            hit = float((ret > 0).sum()) / n if n > 0 else 0.0
        else:
            mu = np.mean(ret)
            sig = np.std(ret, ddof=1)
            vol = sig * np.sqrt(TRADING_DAYS) if np.isfinite(sig) else np.nan
            sharpe = np.nan if (not np.isfinite(sig)) or sig == 0 else (mu - rf/252) / sig * np.sqrt(TRADING_DAYS)

            # 소르티노 비율 (Downside Deviation 사용)
            downside = ret[ret < rf/252]
            d_sig = np.std(downside, ddof=1) if len(downside) >= 2 else 0.0
            sortino = np.nan if d_sig == 0 else (mu - rf/252) / d_sig * np.sqrt(TRADING_DAYS)

            hit = float((ret > 0).sum()) / n

        # MDD 및 칼마 비율
        mdd = self._max_drawdown(nav)
        calmar = 0.0 if mdd == 0 else annual_roi / mdd

        return {"Count": self.count, "ROI": annual_roi, "Sharpe Ratio": sharpe,
                "Cumulative Return": cum_ret, "Sortino Ratio": sortino,
                "Maximum Drawdown": mdd, "Calmar Ratio": calmar,
                "Volatility": vol, "Hit Ratio": hit}

    @staticmethod
    def _max_drawdown(curve):
        """
        자산 곡선(Curve)을 입력받아 최대 낙폭(MDD)을 계산합니다.
        """
        peak = curve[0]
        mdd = 0.0
        for v in curve:
            if v <= 0: return 1.0  # 파산 시 MDD 100%
            peak = max(peak, v)
            mdd  = max(mdd, (peak - v)/peak)
        return min(mdd, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# 데이터셋 및 테스트 유틸리티 (Dataset & Test Utils)
# ──────────────────────────────────────────────────────────────────────────────
from PIL import Image
from torch.utils.data import Dataset

class PairTradingTestDataset(Dataset):
    """
    페어 트레이딩 모델 테스트를 위한 이미지 데이터셋 클래스입니다.
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # 레이블은 더미 값(0) 반환
        return img, 0

def test_model(model, loader, device):
    """
    모델 추론을 수행하여 예측값과 실제 레이블을 반환합니다.

    Args:
        model: 학습된 PyTorch 모델
        loader: 테스트 데이터 로더
        device: 추론 디바이스 (CPU/GPU)

    Returns:
        tuple: (predictions list, true labels list)
    """
    import torch
    model.eval()
    preds, true_lab = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
            true_lab.extend(y.numpy())
    return preds, true_lab