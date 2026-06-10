"""World Gold Council/ICE 금 현물 시계열 + Gold API 최신 XAU quote."""

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from goldgap.constants import KST
from goldgap.domain.gap import attach_fx_and_convert
from goldgap.sources.common import upsert_latest_row

logger = logging.getLogger(__name__)

WGC_GOLD_SPOT_URL = "https://fsapi.gold.org/api/goldprice/v13/chart/price/usd/oz/{start},{end}?cache09092024"
GOLD_API_SPOT_URL = "https://api.gold-api.com/price/XAU/USD"


def fetch_london_spot_gold(fx_df, start_date=None):
    """World Gold Council/ICE spot gold 시계열 + Gold API 최신 XAU spot quote."""
    end = datetime.now(KST)
    start = start_date or (end - timedelta(days=5 * 365))
    if getattr(start, "tzinfo", None) is None:
        start = start.replace(tzinfo=KST)
    start_ms = int(start.astimezone(ZoneInfo("UTC")).timestamp() * 1000)
    end_ms = int(end.astimezone(ZoneInfo("UTC")).timestamp() * 1000)

    logger.info("Fetching London spot gold (WGC/ICE spot)...")
    resp = requests.get(
        WGC_GOLD_SPOT_URL.format(start=start_ms, end=end_ms),
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.gold.org/goldhub/data/gold-prices"},
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json()
    points = payload.get("chartData", {}).get("USD", [])
    if not points:
        raise ValueError("Failed to fetch WGC spot gold data")

    rows = []
    for ts_ms, price in points:
        date = pd.to_datetime(ts_ms, unit="ms", utc=True).tz_convert(KST).tz_localize(None).normalize()
        rows.append((date, float(price)))

    gold_df = pd.DataFrame(rows, columns=["date", "gold_usd_oz"]).set_index("date")
    gold_df = gold_df[~gold_df.index.duplicated(keep="last")].sort_index()

    try:
        latest_resp = requests.get(
            GOLD_API_SPOT_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        latest_resp.raise_for_status()
        latest = float(latest_resp.json()["price"])
        gold_df = upsert_latest_row(gold_df, {"gold_usd_oz": latest})
    except Exception as e:
        logger.warning(f"Latest XAU spot quote failed: {e}")

    return attach_fx_and_convert(gold_df, fx_df, "gold_usd_oz", "intl_price")
