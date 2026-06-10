"""빗썸 공개 REST API 소스: 일봉 및 현재가."""

import logging

import requests

from goldgap.sources.exchange_candles import fetch_exchange_day_candles

logger = logging.getLogger(__name__)

BITHUMB_TICKER_URL = "https://api.bithumb.com/public/ticker/{symbol}"
BITHUMB_DAY_CANDLES_URL = "https://api.bithumb.com/v1/candles/days"


def fetch_bithumb_current_price(symbol):
    """빗썸 현재가 REST 조회."""
    try:
        resp = requests.get(
            BITHUMB_TICKER_URL.format(symbol=symbol),
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") == "0000":
            return float(payload["data"]["closing_price"])
    except Exception as e:
        logger.warning(f"Latest Bithumb ticker failed for {symbol}: {e}")
    return None


def fetch_bithumb_ohlcv(ticker, start_date=None, include_latest=False, public_symbol=None):
    """빗썸 REST API로 일봉 조회"""
    return fetch_exchange_day_candles(
        BITHUMB_DAY_CANDLES_URL,
        ticker,
        "Bithumb",
        start_date,
        pagination_date_field="candle_date_time_kst",
        index_date_field="candle_date_time_kst",
        latest_price=(
            fetch_bithumb_current_price(public_symbol)
            if include_latest and public_symbol
            else None
        ),
    )
