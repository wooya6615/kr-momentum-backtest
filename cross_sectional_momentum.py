"""
한국 주식 크로스섹션 모멘텀 전략 (Jegadeesh & Titman 스타일)
=================================================
2000~2015년 코스피 종목 대상 상위20 vs 하위20 바스켓 수익률 비교

필요 라이브러리:
    pip install FinanceDataReader pandas numpy scipy --break-system-packages
    (pykrx는 더 이상 사용하지 않음 - KRX 서버 응답 불안정 이슈 때문에 FDR의
     상장일/폐지일 데이터만으로 유니버스를 로컬 계산하도록 변경함)

주의: 이 스크립트는 KRX 데이터 서버에 접속해야 하므로 로컬/Colab 환경에서 실행할 것.
     (Claude 컨테이너는 외부 시세 API 접근이 막혀있어 여기서는 실행 불가)
"""

import pandas as pd
import numpy as np
from scipy import stats
import FinanceDataReader as fdr

# -----------------------------
# 0. 설정
# -----------------------------
START = "2000-01-01"
END = "2015-12-31"
FORMATION_MONTHS = 12   # J: 형성기간 (과거 몇 개월 수익률로 랭킹할지)
SKIP_MONTHS = 1         # 단기 반전효과 제거용 skip
HOLDING_MONTHS = 3      # K: 보유기간
TOP_N = 20
BOTTOM_N = 20
TXN_COST = 0.003        # 왕복 거래비용+세금 대략치 (매도세 0.18~0.2% + 슬리피지 등, 실제 수치로 조정할 것)

# -----------------------------
# 1. 유니버스 구성 (pykrx 없이, FDR 상장일/폐지일 기반 - 생존편향 방지)
# -----------------------------
def _get_code_col(df):
    return 'Code' if 'Code' in df.columns else 'Symbol'


def build_listing_master():
    """
    '언제부터 언제까지 코스피에 상장되어 있었는지' 전체 종목 마스터 테이블을 한 번만 만듦.
    - fdr.StockListing('KRX-DESC'): 현재 살아있는 종목 + 상장일(ListingDate)
    - fdr.StockListing('KRX-DELISTING'): 과거 상장폐지된 종목 + 상장일/폐지일
    이렇게 만들어두면, 이후 특정 과거 날짜의 유니버스를 조회할 때 pykrx처럼 매번
    KRX 서버를 다시 호출할 필요 없이 이 테이블만 필터링하면 됨 (서버 불안정 이슈 회피).
    """
    alive = fdr.StockListing('KRX-DESC')
    alive_code_col = _get_code_col(alive)
    alive_m = alive[[alive_code_col, 'Market', 'ListingDate']].rename(columns={alive_code_col: 'Code'})
    alive_m['DelistingDate'] = pd.NaT  # 아직 살아있음

    delisted = fdr.StockListing('KRX-DELISTING')
    delisted_code_col = _get_code_col(delisted)
    delisted_m = delisted[[delisted_code_col, 'Market', 'ListingDate', 'DelistingDate']].rename(
        columns={delisted_code_col: 'Code'})

    master = pd.concat([alive_m, delisted_m], ignore_index=True)
    master = master.dropna(subset=['ListingDate'])  # 상장일 없는 우선주 등은 제외
    master['Code'] = master['Code'].astype(str).str.zfill(6)
    master = master[master['Code'].str.match(r'^\d{6}$')]  # 6자리 숫자 코드만
    master = master.drop_duplicates(subset=['Code'], keep='first')
    return master


def get_universe_at_date(master, date, market="KOSPI"):
    """
    master(build_listing_master 결과)에서 특정 날짜에 실제 상장되어 있던 종목코드 리스트 반환.
    - ListingDate <= date
    - DelistingDate가 없거나(NaT, 아직 생존) DelistingDate >= date
    """
    date = pd.Timestamp(date)
    cond = (
        (master['Market'] == market) &
        (master['ListingDate'] <= date) &
        (master['DelistingDate'].isna() | (master['DelistingDate'] >= date))
    )
    return master.loc[cond, 'Code'].tolist()


def build_full_universe_history(master, start, end, freq="ME"):
    """
    형성기간마다(매달) 그 시점의 유니버스를 로컬 계산으로 미리 다 구성.
    반환값: {날짜: [그 시점 실제 상장종목 리스트]}
    """
    dates = pd.date_range(start, end, freq=freq)
    universe_by_date = {d: get_universe_at_date(master, d) for d in dates}
    return universe_by_date

# -----------------------------
# 1-1. 가격 데이터 수집용 종목 풀 (상장폐지 종목 포함, 넓게)
# -----------------------------
def build_price_pool(start, end):
    """
    가격 시계열을 미리 받아둘 종목 후보 풀을 최대한 넓게 구성.
    - 현재 살아있는 코스피 전체 종목 (Code 컬럼 사용)
    - + 분석 기간(start~end) 중 코스피에서 상장폐지된 종목 (Symbol 컬럼 사용)
    반환값: (전체 티커 리스트, 상장폐지 종목 set)
    상장폐지 종목은 fdr.DataReader() 호출 시 'KRX-DELISTING:코드' 접두어를 붙여야
    상장일~상장폐지일 전체 가격을 받아올 수 있음 (일반 코드로 조회하면 데이터 안 나옴).
    """
    alive = fdr.StockListing('KRX')
    alive_tickers = alive.loc[alive['Market'] == 'KOSPI', 'Code'].tolist()

    try:
        delisted = fdr.StockListing('KRX-DELISTING')
        mask = (delisted['Market'] == 'KOSPI') & (delisted['DelistingDate'] >= pd.Timestamp(start))
        delisted_tickers = delisted.loc[mask, 'Symbol'].tolist()
    except Exception as e:
        print(f"[경고] KRX-DELISTING 조회 실패, 상장폐지 종목 없이 진행: {e}")
        delisted_tickers = []

    # 유효성 검증: 6자리 숫자 코드만 인정
    alive_tickers = [t for t in alive_tickers if isinstance(t, str) and len(t) == 6 and t.isdigit()]
    delisted_tickers = [t for t in delisted_tickers if isinstance(t, str) and len(t) == 6 and t.isdigit()]

    delisted_set = set(delisted_tickers) - set(alive_tickers)  # 혹시 겹치면 alive 쪽 우선
    full_pool = sorted(set(alive_tickers) | delisted_set)

    print(f"가격 데이터 수집 대상 종목 수: {len(full_pool)}개 "
          f"(생존 {len(alive_tickers)}개 + 상장폐지 {len(delisted_set)}개)")
    return full_pool, delisted_set



# -----------------------------
# 1-2. 감자/액면분할 미조정 보정
# -----------------------------
def adjust_for_corporate_actions(close, threshold=2.5):
    """
    일별 종가 시계열에서 전일 대비 threshold배 이상 뛰거나 1/threshold 이하로 급락하는 지점을
    '감자(주식병합)/액면분할이 반영 안 된 원본가격'으로 간주하고, 그 지점 이전 가격 전체를
    비율만큼 곱해서 연속적인(조정된) 가격 시계열로 되돌림.
    (예: 하루 만에 5배 뛰면 1:5 무상감자로 보고, 그 이전 모든 가격을 5배 조정)
    threshold=2.5로 잡은 이유: 일반적인 시장 변동으로는 하루 150% 이상 뛰는 경우가 거의 없어서,
    이 이상이면 실제 급등이라기보단 데이터/기업행위 이슈일 가능성이 훨씬 높음.
    """
    close = close.copy().sort_index().astype(float)
    ratio = close / close.shift(1)

    for i in range(1, len(close)):
        r = ratio.iloc[i]
        if pd.notna(r) and (r > threshold or r < 1 / threshold):
            close.iloc[:i] = close.iloc[:i] * r

    return close


def build_monthly_returns(tickers, start, end, delisted_set=None):
    """
    종목별 월말 종가 기준 월간 수익률 DataFrame (index=월말일, columns=종목코드)
    상장폐지 종목 가격 조회 순서:
      1) fdr.DataReader('KRX-DELISTING:코드', start, end)  (현재 FDR 권장 문법)
      2) pykrx stock.get_market_ohlcv_by_date(start, end, 코드)  (1번 실패 시 대안,
         pykrx의 티커리스트 엔드포인트는 불안정하지만 개별종목 OHLCV 조회는 상대적으로 안정적)
    (구버전 exchange= 파라미터는 최신 FDR에서 deprecated되어 제거함)
    """
    delisted_set = delisted_set or set()
    price_data = {}
    failed_tickers = []

    # pykrx는 필요할 때만 지연 import (없어도 1번 방법으로 대부분 커버 가능)
    try:
        from pykrx import stock as pykrx_stock
        pykrx_available = True
    except ImportError:
        pykrx_available = False

    start_str = pd.Timestamp(start).strftime("%Y%m%d")
    end_str = pd.Timestamp(end).strftime("%Y%m%d")

    for t in tickers:
        df = None
        if t in delisted_set:
            # 방법 1: FDR 콜론 문법
            try:
                df = fdr.DataReader(f"KRX-DELISTING:{t}", start, end)
            except Exception:
                df = None

            # 방법 2: pykrx OHLCV (방법 1 실패 시)
            if (df is None or df.empty) and pykrx_available:
                try:
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
            failed_tickers.append(t)
            continue

        adjusted_close = adjust_for_corporate_actions(df["Close"])
        monthly_close = adjusted_close.resample("ME").last()
        price_data[t] = monthly_close

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
    """
    월간 수익률이 threshold(기본 300%)를 넘는 값들을 찾아냄.
    이런 값은 대부분 실제 시장 움직임이 아니라 데이터 소스 불일치(액면분할/병합 미조정,
    두 소스 간 가격 스케일 차이 등)로 인한 오류일 가능성이 높음.
    스프레드가 비정상적으로 크게(예: -100% 미만) 나올 때 원인 파악용으로 먼저 돌려볼 것.
    """
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
    """
    universe_by_date: build_full_universe_history()로 만든 {날짜: [그 시점 실제 상장종목]} 딕셔너리.
    매달 랭킹할 때, monthly_ret 전체 종목 중 "그 시점에 실제 상장되어 있던 종목"만 후보로 필터링해서
    생존편향(미래에만 존재하는 종목이 과거 시점에 잘못 섞이는 것)을 방지함.
    """
    results = []
    dates = monthly_ret.index

    for i in range(formation + skip, len(dates) - holding):
        formation_start = i - formation - skip
        formation_end = i - skip
        rebalance_date = dates[i]

        # 그 시점 실제 상장종목 리스트 (생존편향 방지 핵심)
        valid_tickers = universe_by_date.get(rebalance_date, [])
        if not valid_tickers:
            continue

        # 형성기간 누적수익률 계산 (skip 반영), 그 시점 유니버스에 있는 종목만 사용
        cum_ret = (1 + monthly_ret.iloc[formation_start:formation_end]).prod() - 1
        cum_ret = cum_ret.dropna()
        cum_ret = cum_ret[cum_ret.index.isin(valid_tickers)]

        if len(cum_ret) < (top_n + bottom_n):
            continue

        ranked = cum_ret.sort_values(ascending=False)
        winners = ranked.index[:top_n]
        losers = ranked.index[-bottom_n:]

        # 보유기간 실현수익률 (동일가중)
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
    """
    매 리밸런싱(보유기간마다)마다 왕복 거래비용을 스프레드에서 차감.
    롱온리(승자군만 매수)로 가정할 경우 winner_ret에서만 차감.
    """
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
    """
    여러 (형성기간, 보유기간) 조합을 한번에 돌려서 비교.
    combos: [(formation, holding), ...] 형태의 리스트. 예: [(12,3), (6,6), (3,1), (9,3)]

    지금 12/3 조합에서 '엣지 없음'이 나온 게 그 조합에서만 우연히 그런 건지,
    아니면 여러 조합에서 일관되게 재현되는 결론인지 확인하는 용도.
    한두 조합만 유의미하고 나머지는 다 아니라면 오히려 데이터 스누핑을 의심해야 함
    (여러 조합을 실험하면 우연히 하나쯤은 p<0.05가 나올 확률이 올라가기 때문).
    """
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
        # 관측치가 holding개월 합산 수익률이므로 월평균으로 환산해서 조합 간 비교 가능하게 함
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
    # 1. 가격 데이터를 받아올 종목 전체 풀(pool): 상장폐지 종목까지 포함해서 최대한 넓게 구성
    price_pool, delisted_set = build_price_pool(START, END)
    monthly_ret, failed_tickers = build_monthly_returns(price_pool, START, END, delisted_set=delisted_set)

    # 2. 상장일/폐지일 마스터 테이블 (pykrx 없이 로컬 계산으로 매달 유니버스 판정)
    listing_master = build_listing_master()
    universe_by_date = build_full_universe_history(listing_master, START, END, freq="ME")

    # 2-1. 이상치 진단 (스프레드가 비정상적으로 크게 나올 때 원인 파악용, 결과 이상하면 여기부터 확인)
    diagnose_extreme_returns(monthly_ret, threshold=3.0)

    # 3. 매달 그 시점 유니버스만 갖고 랭킹 -> 상위20/하위20 -> 보유기간 수익률
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

    # 4. J/K 조합 그리드서치 - 12/3 조합의 결론이 다른 조합에서도 재현되는지 확인
    grid_combos = [(3, 1), (6, 3), (6, 6), (9, 3), (12, 3), (12, 6)]
    grid_summary = run_grid_search(monthly_ret, universe_by_date, grid_combos, skip=SKIP_MONTHS,
                                    top_n=TOP_N, bottom_n=BOTTOM_N)