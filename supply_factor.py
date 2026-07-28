"""
수급(외국인/기관 순매수) 기반 팩터 백테스트
============================================
cross_sectional_momentum.py의 유니버스/검증 로직을 재사용하고,
팩터 계산 부분만 가격모멘텀 -> 수급강도로 교체한 버전.

필요 라이브러리:
    pip install pykrx pandas numpy scipy pyarrow tqdm python-dotenv --break-system-packages

KRX 로그인 필요 (2025.12.27부로 KRX Data Marketplace 회원제 전환):
    data.krx.co.kr 에서 무료 회원가입 후, 아래 환경변수 설정 필요
    KRX_ID, KRX_PW (.env 파일 + python-dotenv 사용 권장, .env는 .gitignore에 반드시 추가)

주의: KRX 무료 로그인 경로는 2015년 이전 투자자별 매매동향 데이터를 제공하지 않는 것으로
     확인됨 (2014년까지는 빈 응답, 2015년부터 정상 조회). 따라서 START는 2015-01-01 이후로
     설정해야 함 (cross_sectional_momentum.py의 START/END를 그대로 가져다 씀).
"""

import os
import time
import pandas as pd
import numpy as np
from scipy import stats
from pykrx import stock as pykrx_stock
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# 기존 파일에서 그대로 재사용
from cross_sectional_momentum import (
    build_price_pool,
    build_listing_master,
    build_full_universe_history,
    build_monthly_returns,
    test_significance,
    apply_transaction_costs,
    START, END, HOLDING_MONTHS, TOP_N, BOTTOM_N, TXN_COST
)

SUPPLY_WINDOW = 20       # 순매수 롤링 합산 기간 (거래일 기준)
NETBUY_CACHE_DIR = "cache/investor"
VOLUME_CACHE_DIR = "cache/daily_volume"


# -----------------------------
# 1. 종목 하나 순매수 조회 (병렬 처리용 헬퍼)
# -----------------------------
def _fetch_one_netbuy(ticker, start_str, end_str, cache_dir, max_retries, base_sleep):
    cache_path = f"{cache_dir}/{ticker}.parquet"
    if os.path.exists(cache_path):
        return ticker, pd.read_parquet(cache_path), None

    df = None
    for attempt in range(max_retries):
        try:
            df = pykrx_stock.get_market_trading_value_by_date(
                start_str, end_str, ticker, on="순매수",
                etf=False, etn=False, elw=False
            )
            break
        except Exception:
            time.sleep(base_sleep * (2 ** attempt))

    if df is None or df.empty:
        return ticker, None, "fail"

    cols = [c for c in ["외국인합계", "기관합계"] if c in df.columns]
    if not cols:
        return ticker, None, "fail"

    df = df[cols]
    df.to_parquet(cache_path)
    return ticker, df, None


# -----------------------------
# 2. 종목별 투자자 순매수 히스토리 수집 (병렬 + 캐싱 + 재시도)
# -----------------------------
def build_investor_netbuy_history(tickers, start, end, cache_dir=NETBUY_CACHE_DIR,
                                    max_retries=1, base_sleep=0.5, max_workers=4):
    """
    max_workers=4로 보수적으로 시작 - KRX가 로그인 세션 기반이라
    너무 공격적으로 병렬화하면 세션 충돌 가능성 있음. 실패율 안 오르면 6~8로 올려도 됨.
    """
    os.makedirs(cache_dir, exist_ok=True)
    start_str = pd.Timestamp(start).strftime("%Y%m%d")
    end_str = pd.Timestamp(end).strftime("%Y%m%d")

    result = {}
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_one_netbuy, t, start_str, end_str,
                             cache_dir, max_retries, base_sleep): t
            for t in tickers
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc="수급 데이터 수집", unit="종목")
        for future in pbar:
            t, df, err = future.result()
            if err:
                failed.append(t)
            else:
                result[t] = df
            pbar.set_postfix(성공=len(result), 실패=len(failed))

    print(f"\n[수급 데이터 수집 결과] 전체 {len(tickers)}개 중 성공 {len(result)}개 / 실패 {len(failed)}개")
    if len(tickers) > 0 and len(failed) / len(tickers) > 0.1:
        print("  ⚠️ 실패율 10% 초과 - failed 리스트를 상장일 기준으로 정렬해서 특정 구간에 쏠려있는지 확인 권장")

    return result, failed


# -----------------------------
# 3. 실패 종목만 별도로 재수집
# -----------------------------
def retry_failed_tickers(failed_tickers, start, end, cache_dir=NETBUY_CACHE_DIR,
                          max_retries=1, base_sleep=1.0, max_workers=2):
    """
    한 번 전체를 다 돌리고 나서 failed 리스트만 따로 다시 시도할 때 사용.
    재시도인 만큼 max_workers를 더 낮춰서(기본 2) 서버 부하를 줄임.
    """
    print(f"\n[재수집 시도] {len(failed_tickers)}개 종목")
    result, still_failed = build_investor_netbuy_history(
        failed_tickers, start, end, cache_dir=cache_dir,
        max_retries=max_retries, base_sleep=base_sleep, max_workers=max_workers
    )
    return result, still_failed


# -----------------------------
# 4. 거래대금 정규화 기준 만들기 (일별 캐시에서)
# -----------------------------
def load_daily_volume_value(ticker, cache_dir=VOLUME_CACHE_DIR):
    """cross_sectional_momentum.py 실행 시 캐싱해둔 일별 Close/Volume에서 거래대금 계산"""
    path = f"{cache_dir}/{ticker}.parquet"
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if "Close" not in df.columns or "Volume" not in df.columns:
        return None
    return df["Close"] * df["Volume"]


# -----------------------------
# 5. 팩터 계산: 20일 롤링 (외국인+기관 순매수) / 20일 누적 거래대금
# -----------------------------
def compute_supply_factor(netbuy_df, trading_value, window=SUPPLY_WINDOW):
    """절대금액이 아니라 거래대금 대비 비율로 정규화 -> 대형주 쏠림 방지"""
    if netbuy_df is None or trading_value is None:
        return None
    combined_netbuy = netbuy_df.reindex(trading_value.index).fillna(0).sum(axis=1)
    rolling_netbuy = combined_netbuy.rolling(window).sum()
    rolling_value = trading_value.rolling(window).sum()
    factor = rolling_netbuy / rolling_value.replace(0, np.nan)
    return factor


def build_supply_factor_panel(tickers, netbuy_history, window=SUPPLY_WINDOW):
    """전 종목 팩터를 하나의 DataFrame(index=날짜, columns=종목)으로 병합"""
    factor_dict = {}
    for t in tickers:
        netbuy_df = netbuy_history.get(t)
        trading_value = load_daily_volume_value(t)
        factor = compute_supply_factor(netbuy_df, trading_value, window)
        if factor is not None:
            factor_dict[t] = factor
    panel = pd.DataFrame(factor_dict)
    # 월말 리밸런싱 날짜에 맞춰 리샘플 (그 시점까지의 마지막 값 사용 -> 룩어헤드 없음)
    monthly_panel = panel.resample("ME").last()
    return monthly_panel


# -----------------------------
# 6. 수급 팩터 단독 백테스트
# -----------------------------
def run_supply_factor_backtest(monthly_supply_factor, universe_by_date, monthly_ret,
                                holding=HOLDING_MONTHS, top_n=TOP_N, bottom_n=BOTTOM_N):
    """
    핵심 수정: rebalance_date(T월 말) 시점 신호로, T+1월부터 holding개월 보유.
    이전 버전은 idx(T월)부터 보유기간을 시작해서 신호 계산 구간과 보유기간이
    같은 달에 겹치는 룩어헤드 버그가 있었음 (t-stat 12, p=0.0000으로 비정상적으로
    유의미하게 나온 원인). T월 순매수 데이터로 T월 수익률을 맞히는 건 예측이 아니라
    동시성(contemporaneous correlation)일 뿐이라 반드시 다음 달부터 보유해야 함.
    """
    results = []
    dates = monthly_supply_factor.index

    for i in range(len(dates) - holding - 1):  # -1: T+1월부터 시작할 여유분 확보
        rebalance_date = dates[i]
        valid_tickers = universe_by_date.get(rebalance_date, [])
        if not valid_tickers:
            continue

        scores = monthly_supply_factor.loc[rebalance_date].dropna()
        scores = scores[scores.index.isin(valid_tickers)]
        if len(scores) < (top_n + bottom_n):
            continue

        ranked = scores.sort_values(ascending=False)
        top = ranked.index[:top_n]
        bottom = ranked.index[-bottom_n:]

        if rebalance_date not in monthly_ret.index:
            continue
        idx = monthly_ret.index.get_loc(rebalance_date)

        # T월 신호 -> T+1월부터 holding개월 보유 (신호/보유기간 겹침 제거)
        start_idx = idx + 1
        if start_idx + holding > len(monthly_ret):
            continue
        hold_ret = (1 + monthly_ret.iloc[start_idx:start_idx+holding]).prod() - 1

        top_ret = hold_ret.reindex(top).mean()
        bottom_ret = hold_ret.reindex(bottom).mean()

        results.append({
            "date": rebalance_date,
            "top_ret": top_ret,
            "bottom_ret": bottom_ret,
            "spread": top_ret - bottom_ret
        })

    return pd.DataFrame(results)


# -----------------------------
# 7. 모멘텀 팩터와의 상관관계 확인 (결합 전 필수 체크)
# -----------------------------
def check_factor_correlation(monthly_supply_factor, monthly_momentum_score):
    """두 팩터가 사실상 같은 정보(지연된 가격모멘텀)인지 확인."""
    common_dates = monthly_supply_factor.index.intersection(monthly_momentum_score.index)
    corrs = []
    for d in common_dates:
        s = monthly_supply_factor.loc[d].dropna()
        m = monthly_momentum_score.loc[d].dropna()
        common_tickers = s.index.intersection(m.index)
        if len(common_tickers) > 10:
            corrs.append(s[common_tickers].corr(m[common_tickers]))
    corr_series = pd.Series(corrs, index=common_dates[:len(corrs)])
    print(f"\n[팩터 간 횡단면 상관관계] 평균: {corr_series.mean():.3f}, "
          f"중앙값: {corr_series.median():.3f}")
    return corr_series


# -----------------------------
# 7-1. 수급 팩터 전용 거래비용 반영 (컬럼명이 winner/loser가 아니라 top/bottom이라 별도 정의)
# -----------------------------
def apply_transaction_costs_supply(result_df, cost=TXN_COST):
    result_df["top_ret_net"] = result_df["top_ret"] - cost
    result_df["spread_net"] = result_df["spread"] - cost
    print(f"\n[거래비용 {cost*100:.2f}% 반영 후]")
    print(f"평균 순 상위그룹 수익률: {result_df['top_ret_net'].mean()*100:.2f}%")
    print(f"연환산 순 상위그룹 수익률: {result_df['top_ret_net'].mean() * (12/HOLDING_MONTHS) * 100:.2f}%")
    return result_df

# supply_factor.py에 추가
def run_supply_grid_search(tickers, netbuy_history, universe_by_date, monthly_ret,
                            windows, holdings, top_n=TOP_N, bottom_n=BOTTOM_N):
    rows = []
    for window in windows:
        panel = build_supply_factor_panel(tickers, netbuy_history, window=window)
        for holding in holdings:
            result_df = run_supply_factor_backtest(panel, universe_by_date, monthly_ret, holding=holding)
            if result_df.empty or "spread" not in result_df.columns:
                rows.append({"window": window, "holding": holding, "n_obs": 0,
                            "t_stat": None, "p_value": None, "significant": None})
                continue
            spread = result_df["spread"].dropna()
            t_stat, p_value = stats.ttest_1samp(spread, 0)
            rows.append({
                "window": window, "holding": holding, "n_obs": len(spread),
                "mean_spread": spread.mean(), "t_stat": t_stat, "p_value": p_value,
                "significant": p_value < 0.05
            })
    summary = pd.DataFrame(rows)
    print("\n[수급 팩터 윈도우/보유기간 그리드서치]")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    price_pool, delisted_set = build_price_pool(START, END)

    print("\n=== 1. 투자자별 순매수 데이터 수집 ===")
    netbuy_history, failed_netbuy = build_investor_netbuy_history(price_pool, START, END)

    if failed_netbuy:
        print(f"\n=== 1-1. 실패 {len(failed_netbuy)}건 재수집 시도 ===")
        retried, still_failed = retry_failed_tickers(failed_netbuy, START, END)
        netbuy_history.update(retried)
        if still_failed:
            print(f"최종 실패 {len(still_failed)}건 - 이 종목들은 수급 팩터 계산에서 제외됨")

    print("\n=== 2. 수급 팩터 패널 계산 ===")
    supply_panel = build_supply_factor_panel(price_pool, netbuy_history)

    print("\n=== 3. 유니버스 히스토리 (기존 로직 재사용) ===")
    listing_master = build_listing_master()
    universe_by_date = build_full_universe_history(listing_master, START, END, freq="ME")

    print("\n=== 4. 가격 수익률 로드 ===")
    monthly_ret, _ = build_monthly_returns(price_pool, START, END, delisted_set=delisted_set)

    print("\n=== 5. 수급 팩터 단독 백테스트 ===")
    result_df = run_supply_factor_backtest(supply_panel, universe_by_date, monthly_ret)
    test_significance(result_df)
    apply_transaction_costs_supply(result_df)
    
    grid = run_supply_grid_search(price_pool, netbuy_history, universe_by_date, monthly_ret,
                                windows=[10, 20, 40], holdings=[1, 3, 6])