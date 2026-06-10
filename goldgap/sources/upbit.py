"""업비트 공개 REST API 소스: 일봉 및 현재가."""

import logging

import requests

from goldgap.sources.common import PUBLIC_API_HEADERS
from goldgap.sources.exchange_candles import fetch_exchange_day_candles

logger = logging.getLogger(__name__)

UPBIT_DAY_CANDLES_URL = "https://api.upbit.com/v1/candles/days"
UPBIT_TICKER_URL = "https://api.upbit.com/v1/ticker"


def fetch_upbit_current_price(ticker):
    """업비트 현재가 REST 조회."""
    try:
        resp = requests.get(
            UPBIT_TICKER_URL,
            params={"markets": ticker},
            headers=PUBLIC_API_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        rows = resp.json()
        if rows:
            return float(rows[0]["trade_price"])
    except Exception as e:
        logger.warning(f"Latest Upbit ticker failed for {ticker}: {e}")
    return None


def fetch_upbit_ohlcv(ticker, start_date=None, include_latest=False):
    """업비트 REST API로 일봉 조회

    업비트 일봉은 00:00 UTC(=09:00 KST)에 개시되어 candle_date_time_kst와
    candle_date_time_utc의 날짜 라벨이 항상 동일하다 — 인덱스를 KST 필드로
    바꿔도 결과는 불변이며, 빗썸(KST 기준)과 의미 일관성을 위해 KST를 쓴다.
    페이지네이션 커서(to 파라미터)는 API 관례대로 UTC 필드를 유지한다. (BUG-04)
    """
    return fetch_exchange_day_candles(
        UPBIT_DAY_CANDLES_URL,
        ticker,
        "Upbit",
        start_date,
        pagination_date_field="candle_date_time_utc",
        index_date_field="candle_date_time_kst",
        latest_price=fetch_upbit_current_price(ticker) if include_latest else None,
    )
