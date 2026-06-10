"""거래소(업비트/빗썸) 공개 REST 일봉 조회 공용 루틴."""

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from goldgap.sources.common import PUBLIC_API_HEADERS, upsert_latest_row

logger = logging.getLogger(__name__)


def fetch_exchange_day_candles(
    url,
    ticker,
    exchange_name,
    start_date=None,
    pagination_date_field="candle_date_time_utc",
    index_date_field="candle_date_time_utc",
    latest_price=None,
):
    """거래소 공개 REST API로 일봉 조회 (페이지네이션으로 전체 기간)"""
    logger.info(f"Fetching {exchange_name} day candles for {ticker}...")

    all_dfs = []
    to = datetime.now()
    target_start = pd.Timestamp(
        start_date or (datetime.now() - timedelta(days=5 * 365))
    ).tz_localize(None).normalize()

    while pd.Timestamp(to).tz_localize(None) > target_start:
        params = {"market": ticker, "count": 200}
        if all_dfs:
            params["to"] = to.strftime("%Y-%m-%dT%H:%M:%S")

        resp = requests.get(
            url,
            params=params,
            headers=PUBLIC_API_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        rows = resp.json()

        if isinstance(rows, dict):
            raise ValueError(f"{exchange_name} returned error for {ticker}: {rows}")
        if not rows:
            break

        df = pd.DataFrame(rows)
        if "trade_price" not in df or index_date_field not in df:
            raise ValueError(f"{exchange_name} response missing candle fields for {ticker}")

        all_dfs.append(df)
        page_dates = pd.to_datetime(df[pagination_date_field], errors="coerce")
        oldest = page_dates.min()
        if pd.isna(oldest):
            raise ValueError(f"{exchange_name} response has invalid candle dates for {ticker}")

        to = oldest.to_pydatetime().replace(tzinfo=None)
        time.sleep(0.2)

    if not all_dfs:
        raise ValueError(f"Failed to fetch {exchange_name} data for {ticker}")

    result = pd.concat(all_dfs).sort_index()
    dates = pd.to_datetime(result[index_date_field], errors="coerce")
    result = result.assign(date=dates.dt.normalize())
    result = result.dropna(subset=["date", "trade_price"])
    result = result.sort_values("date")

    domestic = pd.DataFrame(
        {"domestic_price": result["trade_price"].astype(float).values},
        index=pd.DatetimeIndex(result["date"]),
    )
    domestic = domestic[~domestic.index.duplicated(keep="last")]
    domestic = domestic[domestic.index >= target_start].dropna()

    if domestic.empty:
        raise ValueError(f"No {exchange_name} data for {ticker} after filtering")

    if latest_price:
        domestic = upsert_latest_row(domestic, {"domestic_price": float(latest_price)})

    logger.info(
        f"{exchange_name} {ticker}: {len(domestic)} rows "
        f"({domestic.index.min().strftime('%Y-%m-%d')} ~ {domestic.index.max().strftime('%Y-%m-%d')})"
    )
    return domestic
