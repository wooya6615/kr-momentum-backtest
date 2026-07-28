import os
from dotenv import load_dotenv
load_dotenv()
from pykrx import stock

# 상장폐지 종목 중 실패한 게 정확히 언제 상장폐지됐는지 확인
failed_delisted_dates = delisted[delisted['Symbol'].isin(failed_tickers_that_are_delisted)]
print(failed_delisted_dates['DelistingDate'].describe())