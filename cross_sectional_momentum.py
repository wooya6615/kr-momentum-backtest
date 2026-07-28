"""
한국 주식 크로스섹션 모멘텀 전략 (Jegadeesh & Titman 스타일)
=================================================
2000~2015년 코스피 종목 대상 상위20 vs 하위20 바스켓 수익률 비교

필요 라이브러리:
    pip install FinanceDataReader pandas numpy scipy pyarrow tqdm --break-system-packages
"""

import pandas as pd
import numpy as np
from scipy import stats
import FinanceDataReader as fdr
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------
# 0. 설정
# -----------------------------
START = "2015-01-01"
END = "2025-12-31"
FORMATION_MONTHS = 12
SKIP_MONTHS = 1
HOLDING_MONTHS = 3
TOP_N = 20
BOTTOM_N = 20
TXN_COST = 0.003
PRICE_CACHE_DIR = "cache/monthly_price"


# -----------------------------
# 1. 유니버스 구성
# -----------------------------
def _get_code_col(df):
    return 'Code' if 'Code' in df.columns else 'Symbol'


def build_listing_master():
    alive = fdr.StockListing('KRX-DESC')
    alive_code_col = _get_code_col(alive)
    alive_m = alive[[alive_code_col, 'Market', 'ListingDate']].rename(columns={alive_code_col: 'Code'})
    alive_m['DelistingDate'] = pd.NaT

    delisted = fdr.StockListing('KRX-DELISTING')
    delisted_code_col = _get_code_col(delisted)
    delisted_m = delisted[[delisted_code_col, 'Market', 'ListingDate', 'DelistingDate']].rename(
        columns={delisted_code_col: 'Code'})

    master = pd.concat([alive_m, delisted_m], ignore_index=True)
    master = master.dropna(subset=['ListingDate'])
    master['Code'] = master['Code'].astype(str).str.zfill(6)
    master = master[master['Code'].str.match(r'^\d{6}$')]
    master = master.drop_duplicates(subset=['Code'], keep='first')
    return master


def get_universe_at_date(master, date, market="KOSPI"):
    date = pd.Timestamp(date)
    cond = (
        (master['Market'] == market) &
        (master['ListingDate'] <= date) &
        (master['DelistingDate'].isna() | (master['DelistingDate'] >= date))
    )
    return master.loc[cond, 'Code'].tolist()


def build_full_universe_history(master, start, end, freq="ME"):
    dates = pd.date_range(start, end, freq=freq)
    universe_by_date = {d: get_universe_at_date(master, d) for d in dates}
    return universe_by_date


# -----------------------------
# 1-1. 가격 데이터 수집용 종목 풀
# -----------------------------
def build_price_pool(start, end):
    alive = fdr.StockListing('KRX')
    alive_tickers = alive.loc[alive['Market'] == 'KOSPI', 'Code'].tolist()

    try:
        delisted = fdr.StockListing('KRX-DELISTING')
        mask = (delisted['Market'] == 'KOSPI') & (delisted['DelistingDate'] >= pd.Timestamp(start))
        delisted_tickers = delisted.loc[mask, 'Symbol'].tolist()
    except Exception as e:
        print(f"[경고] KRX-DELISTING 조회 실패, 상장폐지 종목 없이 진행: {e}")
        delisted_tickers = []

    alive_tickers = [t for t in alive_tickers if isinstance(t, str) and len(t) == 6 and t.isdigit()]
    delisted_tickers = [t for t in delisted_tickers if isinstance(t, str) and len(t) == 6 and t.isdigit()]

    delisted_set = set(delisted_tickers) - set(alive_tickers)
    full_pool = sorted(set(alive_tickers) | delisted_set)

    print(f"가격 데이터 수집 대상 종목 수: {len(full_pool)}개 "
          f"(생존 {len(alive_tickers)}개 + 상장폐지 {len(delisted_set)}개)")
    return full_pool, delisted_set


# -----------------------------
# 1-2. 감자/액면분할 미조정 보정
# -----------------------------
def adjust_for_corporate_actions(close, threshold=2.5):
    close = close.copy().sort_index().astype(float)
    ratio = close / close.shift(1)

    for i in range(1, len(close)):
        r = ratio.iloc[i]
        if pd.notna(r) and (r > threshold or r < 1 / threshold):
            close.iloc[:i] = close.iloc[:i] * r

    return close


# -----------------------------
# 1-3. 종목 하나 가격 조회 (병렬 처리용 헬퍼)
# -----------------------------
def _fetch_one_ticker(t, start, end, delisted_set, start_str, end_str, pykrx_available):
    cache_path = f"{PRICE_CACHE_DIR}/{t}.parquet"
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        return t, cached["monthly_close"], None

    df = None
    if t in delisted_set:
        try:
            df = fdr.DataReader(f"KRX-DELISTING:{t}", start, end)
        except Exception:
            df = None
        if (df is None or df.empty) and pykrx_available:
            try:
                from pykrx import stock as pykrx_stock
                pdf = pykrx_stock.get_market_ohlcv_by_date(start_str, end_str, t)
                if pdf is not None and not pdf.empty:
                    pdf = pdf.rename(columns={"종가": "Close"})
                    df = pdf
            except Exception:
                df = None
    else:
        try:
            df = fdr.DataReader(t, start, end)
        except Exception:
            df = None

    if df is None or df.empty:
        return t, None, "fail"

    adjusted_close = adjust_for_corporate_actions(df["Close"])

    os.makedirs("cache/daily_volume", exist_ok=True)
    if "Volume" in df.columns:
        df[["Close", "Volume"]].to_parquet(f"cache/daily_volume/{t}.parquet")

    monthly_close = adjusted_close.resample("ME").last()

    os.makedirs(PRICE_CACHE_DIR, exist_ok=True)
    monthly_close.rename("monthly_close").to_frame().to_parquet(cache_path)

    return t, monthly_close, None


# -----------------------------
# 2. 종목별 월간 수익률 (병렬 + 캐싱)
# -----------------------------
def build_monthly_returns(tickers, start, end, delisted_set=None, max_workers=8):
    """
    종목별 월말 종가 기준 월간 수익률 DataFrame.
    - 종목별 결과를 캐싱해서 재실행 시 스킵
    - ThreadPoolExecutor로 병렬 요청
    """
    delisted_set = delisted_set or set()

    try:
        from pykrx import stock as pykrx_stock
        pykrx_available = True
    except ImportError:
        pykrx_available = False

    start_str = pd.Timestamp(start).strftime("%Y%m%d")
    end_str = pd.Timestamp(end).strftime("%Y%m%d")

    price_data = {}
    failed_tickers = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_one_ticker, t, start, end, delisted_set,
                             start_str, end_str, pykrx_available): t
            for t in tickers
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc="가격 데이터 수집", unit="종목")
        for future in pbar:
            t, monthly_close, err = future.result()
            if err:
                failed_tickers.append(t)
            else:
                price_data[t] = monthly_close
            pbar.set_postfix(성공=len(price_data), 실패=len(failed_tickers))

    price_df = pd.DataFrame(price_data)
    monthly_ret = price_df.pct_change(fill_method=None)

    total = len(tickers)
    failed = len(failed_tickers)
    failed_delisted = sum(1 for t in failed_tickers if t in delisted_set)
    print(f"\n[가격 데이터 수집 결과] 전체 {total}개 중 성공 {total - failed}개 / 실패 {failed}개 "
          f"({failed/total*100:.1f}%)")
    print(f"  - 실패 중 상장폐지 종목: {failed_delisted}개 / 실패 중 생존 종목: {failed - failed_delisted}개")
    if failed_delisted == len(delisted_set) and len(delisted_set) > 0:
        print("  🚨 상장폐지 종목이 100% 실패했습니다 — fdr.__version__ 확인 후 "
              "pip install -U finance-datareader 로 업그레이드해보세요.")
    elif failed / total > 0.05:
        print("  ⚠️ 실패율 5% 초과 — 실패 종목 리스트를 확인해서 특정 유형(부실주 등)에 쏠려있는지 점검 권장")

    return monthly_ret, failed_tickers


# -----------------------------
# 2-1. 이상치(비정상 수익률) 진단
# -----------------------------
def diagnose_extreme_returns(monthly_ret, threshold=3.0):
    extreme = monthly_ret[(monthly_ret.abs() > threshold)]
    extreme_stacked = extreme.stack().sort_values(key=abs, ascending=False)

    if extreme_stacked.empty:
        print(f"수익률 {threshold*100:.0f}% 초과 이상치 없음")
        return extreme_stacked

    print(f"\n[이상치 진단] |월간수익률| > {threshold*100:.0f}% 인 케이스 {len(extreme_stacked)}건 (상위 20개):")
    for (date, ticker), ret in extreme_stacked.head(20).items():
        print(f"  {date.strftime('%Y-%m')}  {ticker}  {ret*100:>10.1f}%")

    affected_tickers = extreme_stacked.index.get_level_values(1).unique().tolist()
    print(f"\n영향받은 종목 수: {len(affected_tickers)}개 -> 이 종목들 개별로 "
          f"fdr vs pykrx 가격을 직접 비교해서 어느 시점에 어긋나는지 확인 권장")
    return extreme_stacked


# -----------------------------
# 3. 매월 랭킹 -> 상위/하위 바스켓 -> 향후 K개월 수익률
# -----------------------------
def run_cross_sectional_momentum(monthly_ret, universe_by_date, formation=12, skip=1, holding=3, top_n=20, bottom_n=20):
    results = []
    dates = monthly_ret.index

    for i in range(formation + skip, len(dates) - holding):
        formation_start = i - formation - skip
        formation_end = i - skip
        rebalance_date = dates[i]

        valid_tickers = universe_by_date.get(rebalance_date, [])
        if not valid_tickers:
            continue

        cum_ret = (1 + monthly_ret.iloc[formation_start:formation_end]).prod() - 1
        cum_ret = cum_ret.dropna()
        cum_ret = cum_ret[cum_ret.index.isin(valid_tickers)]

        if len(cum_ret) < (top_n + bottom_n):
            continue

        ranked = cum_ret.sort_values(ascending=False)
        winners = ranked.index[:top_n]
        losers = ranked.index[-bottom_n:]

        hold_ret = (1 + monthly_ret.iloc[i:i+holding]).prod() - 1

        winner_ret = hold_ret[winners].mean()
        loser_ret = hold_ret[losers].mean()

        results.append({
            "date": dates[i],
            "winner_ret": winner_ret,
            "loser_ret": loser_ret,
            "spread": winner_ret - loser_ret
        })

    return pd.DataFrame(results)


# -----------------------------
# 4. 통계적 유의성 검정
# -----------------------------
def test_significance(result_df):
    if result_df.empty or "spread" not in result_df.columns:
        raise ValueError(
            "result_df가 비어있습니다. universe_by_date의 유니버스가 매달 비어있지 않은지, "
            "monthly_ret 컬럼(종목코드)과 universe_by_date의 코드 형식이 일치하는지(둘 다 6자리 문자열인지) 확인하세요."
        )
    spread = result_df["spread"].dropna()
    t_stat, p_value = stats.ttest_1samp(spread, 0)
    print(f"평균 스프레드(승자-패자): {spread.mean()*100:.2f}%")
    print(f"연환산 스프레드: {spread.mean() * (12/HOLDING_MONTHS) * 100:.2f}%")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"통계적으로 유의미? {'예 (p<0.05)' if p_value < 0.05 else '아니오'}")
    return t_stat, p_value


# -----------------------------
# 5. 거래비용/세금 반영 순수익 계산
# -----------------------------
def apply_transaction_costs(result_df, cost=TXN_COST):
    result_df["winner_ret_net"] = result_df["winner_ret"] - cost
    result_df["spread_net"] = result_df["spread"] - cost
    print(f"\n[거래비용 {cost*100:.2f}% 반영 후]")
    print(f"평균 순 승자수익률: {result_df['winner_ret_net'].mean()*100:.2f}%")
    print(f"연환산 순 승자수익률: {result_df['winner_ret_net'].mean() * (12/HOLDING_MONTHS) * 100:.2f}%")
    return result_df


# -----------------------------
# 6. J/K 조합 그리드서치
# -----------------------------
def run_grid_search(monthly_ret, universe_by_date, combos, skip=1, top_n=20, bottom_n=20):
    rows = []
    for formation, holding in combos:
        result_df = run_cross_sectional_momentum(
            monthly_ret, universe_by_date,
            formation=formation, skip=skip, holding=holding,
            top_n=top_n, bottom_n=bottom_n
        )
        if result_df.empty or "spread" not in result_df.columns:
            rows.append({
                "formation": formation, "holding": holding, "n_obs": 0,
                "mean_spread_monthly": None, "t_stat": None, "p_value": None,
                "significant": None
            })
            continue

        spread = result_df["spread"].dropna()
        monthly_equiv_spread = spread / holding
        if len(monthly_equiv_spread) < 2:
            t_stat, p_value = None, None
        else:
            t_stat, p_value = stats.ttest_1samp(monthly_equiv_spread, 0)

        rows.append({
            "formation": formation,
            "holding": holding,
            "n_obs": len(spread),
            "mean_spread_monthly": monthly_equiv_spread.mean(),
            "t_stat": t_stat,
            "p_value": p_value,
            "significant": (p_value is not None and p_value < 0.05)
        })

    summary = pd.DataFrame(rows)
    print("\n[J/K 그리드서치 결과]")
    print(summary.to_string(index=False))

    n_significant = summary["significant"].sum() if "significant" in summary else 0
    print(f"\n총 {len(combos)}개 조합 중 유의미(p<0.05): {n_significant}개")
    if 0 < n_significant < len(combos):
        print("⚠️ 일부 조합에서만 유의미함 - 여러 조합을 테스트하면 우연히 하나쯤 유의미하게 나올 "
              "확률이 있으므로, 이 결과 하나만 보고 '엣지 있다'고 단정하지 말 것 (다중검정 문제)")

    return summary


if __name__ == "__main__":
    price_pool, delisted_set = build_price_pool(START, END)
    monthly_ret, failed_tickers = build_monthly_returns(price_pool, START, END, delisted_set=delisted_set)

    listing_master = build_listing_master()
    universe_by_date = build_full_universe_history(listing_master, START, END, freq="ME")

    diagnose_extreme_returns(monthly_ret, threshold=3.0)

    result_df = run_cross_sectional_momentum(
        monthly_ret,
        universe_by_date,
        formation=FORMATION_MONTHS,
        skip=SKIP_MONTHS,
        holding=HOLDING_MONTHS,
        top_n=TOP_N,
        bottom_n=BOTTOM_N
    )
    test_significance(result_df)
    apply_transaction_costs(result_df)

    grid_combos = [(3, 1), (6, 3), (6, 6), (9, 3), (12, 3), (12, 6)]
    grid_summary = run_grid_search(monthly_ret, universe_by_date, grid_combos, skip=SKIP_MONTHS,
                                    top_n=TOP_N, bottom_n=BOTTOM_N)