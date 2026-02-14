"""
데이터 수집 모듈: 국내/국제 금 가격 및 괴리율 계산

국내 금 가격: pykrx를 통한 KRX 금현물 ETF(411060) 기초지수 + PDF 보정
국제 금 가격: yfinance COMEX 금선물(GC=F) + USD/KRW 환율(KRW=X)
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pykrx import stock as pykrx_stock

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "gold_data.json")
TROY_OZ_TO_GRAM = 31.1035

# KRX Gold Spot Price Index base price (2014-03-24 = 1000)
# Calibrated from 2022-2024 stable period: median factor ≈ 42.3
DEFAULT_BASE_PRICE = 42300  # KRW/g


def fetch_international_gold(years=5):
    """yfinance로 국제 금 선물(GC=F) 및 USD/KRW 환율(KRW=X) 조회"""
    end = datetime.now()
    start = end - timedelta(days=years * 365)

    logger.info("Fetching international gold (GC=F) and USD/KRW (KRW=X)...")
    gold = yf.download("GC=F", start=start, end=end, progress=False)
    fx = yf.download("KRW=X", start=start, end=end, progress=False)

    if gold.empty or fx.empty:
        raise ValueError("Failed to fetch international gold or exchange rate data")

    if isinstance(gold.columns, pd.MultiIndex):
        gold.columns = gold.columns.get_level_values(0)
    if isinstance(fx.columns, pd.MultiIndex):
        fx.columns = fx.columns.get_level_values(0)

    gold_df = gold[["Close"]].rename(columns={"Close": "gold_usd_oz"})
    fx_df = fx[["Close"]].rename(columns={"Close": "usd_krw"})

    gold_df.index = pd.to_datetime(gold_df.index).tz_localize(None)
    fx_df.index = pd.to_datetime(fx_df.index).tz_localize(None)

    intl = gold_df.join(fx_df, how="inner")
    intl["intl_krw_per_gram"] = (intl["gold_usd_oz"] * intl["usd_krw"]) / TROY_OZ_TO_GRAM

    return intl


def _get_pdf_gold_price(date_str):
    """pykrx PDF에서 KRX 금현물 가격(KRW/g) 추출"""
    try:
        pdf = pykrx_stock.get_etf_portfolio_deposit_file("411060", date_str)
        if pdf.empty:
            return None
        gold_row = pdf[pdf["구성종목명"].str.contains("금", na=False)]
        if gold_row.empty:
            return None
        amount = gold_row["금액"].iloc[0]
        contracts = gold_row["계약수"].iloc[0]
        if contracts <= 0:
            return None
        return amount / contracts
    except Exception as e:
        logger.debug(f"PDF fetch failed for {date_str}: {e}")
        return None


def fetch_krx_gold():
    """KRX 금현물 가격 조회 (ETF 기초지수 + PDF 보정)"""
    end_str = datetime.now().strftime("%Y%m%d")
    start_str = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y%m%d")

    # Step 1: Get ETF OHLCV with 기초지수 for full range
    logger.info("Fetching KRX gold ETF base index data...")
    etf_df = pykrx_stock.get_etf_ohlcv_by_date(start_str, end_str, "411060")

    if etf_df.empty:
        raise ValueError("Failed to fetch KRX gold ETF data")

    etf_df.index = pd.to_datetime(etf_df.index)
    base_index = etf_df["기초지수"].copy()

    # Filter out invalid index values
    base_index = base_index[base_index > 100]

    logger.info(f"ETF data: {len(base_index)} trading days")

    # Step 2: Sample PDF data for calibration (~every 20 trading days)
    sample_interval = 20
    sample_dates = base_index.index[::sample_interval].tolist()
    # Always include the latest date
    if base_index.index[-1] not in sample_dates:
        sample_dates.append(base_index.index[-1])

    logger.info(f"Fetching {len(sample_dates)} PDF samples for calibration...")
    calibration_points = {}
    for dt in sample_dates:
        date_str = dt.strftime("%Y%m%d")
        price = _get_pdf_gold_price(date_str)
        if price is not None and price > 0:
            idx_val = base_index.loc[dt]
            if idx_val > 100:
                factor = price / idx_val
                calibration_points[dt] = factor
                logger.debug(f"  {date_str}: gold={price:.0f}, idx={idx_val:.2f}, factor={factor:.2f}")
        time.sleep(0.3)

    if not calibration_points:
        logger.warning("No calibration points, using default factor")
        factor_series = pd.Series(DEFAULT_BASE_PRICE / 1000, index=base_index.index)
    else:
        # Interpolate factor for all dates
        factor_df = pd.Series(calibration_points).sort_index()
        factor_series = factor_df.reindex(base_index.index).interpolate(method="time")
        # Fill edges with nearest value
        factor_series = factor_series.ffill().bfill()

    # Step 3: Compute KRX gold price per gram
    krx_gold = base_index * factor_series
    result = pd.DataFrame({"domestic_krw_per_gram": krx_gold})
    result = result.dropna()

    logger.info(
        f"KRX gold data: {len(result)} rows "
        f"({result.index.min().strftime('%Y-%m-%d')} ~ {result.index.max().strftime('%Y-%m-%d')})"
    )
    return result


def calculate_gap(intl_df, domestic_df):
    """국내/국제 금 가격 병합 및 괴리율 계산"""
    merged = intl_df.join(domestic_df, how="inner")
    merged = merged.dropna()

    if merged.empty:
        raise ValueError("No overlapping dates between international and domestic data")

    merged["gap_pct"] = (
        (merged["domestic_krw_per_gram"] - merged["intl_krw_per_gram"])
        / merged["intl_krw_per_gram"]
        * 100
    )

    return merged


def find_high_gap_periods(merged_df, threshold=5.0):
    """threshold% 이상 괴리율(절대값) 발생 구간을 찾아 요약"""
    periods = []
    in_period = False
    start = None
    max_gap = 0

    for date, row in merged_df.iterrows():
        gap = row["gap_pct"]
        if abs(gap) >= threshold:
            if not in_period:
                in_period = True
                start = date
                max_gap = gap
            else:
                if abs(gap) > abs(max_gap):
                    max_gap = gap
        else:
            if in_period:
                duration = (date - start).days
                periods.append(
                    {
                        "start": start.strftime("%Y-%m-%d"),
                        "end": (date - timedelta(days=1)).strftime("%Y-%m-%d"),
                        "max_gap": round(max_gap, 2),
                        "duration_days": max(duration, 1),
                    }
                )
                in_period = False

    if in_period:
        last_date = merged_df.index[-1]
        duration = (last_date - start).days + 1
        periods.append(
            {
                "start": start.strftime("%Y-%m-%d"),
                "end": last_date.strftime("%Y-%m-%d"),
                "max_gap": round(max_gap, 2),
                "duration_days": max(duration, 1),
            }
        )

    return periods


def get_data(force_refresh=False):
    """캐시된 데이터 반환, 필요시 갱신"""
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not force_refresh and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_time = datetime.fromisoformat(cached["timestamp"])
            if datetime.now() - cached_time < timedelta(hours=24):
                logger.info("Using cached data")
                return cached["data"]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Cache read failed: {e}")

    logger.info("Fetching fresh data...")
    intl_df = fetch_international_gold()
    domestic_df = fetch_krx_gold()
    merged = calculate_gap(intl_df, domestic_df)
    high_gap_periods = find_high_gap_periods(merged)

    data = {
        "dates": [d.strftime("%Y-%m-%d") for d in merged.index],
        "domestic_price": [round(float(v), 0) for v in merged["domestic_krw_per_gram"].tolist()],
        "intl_price": [round(float(v), 0) for v in merged["intl_krw_per_gram"].tolist()],
        "gap_pct": [round(float(v), 2) for v in merged["gap_pct"].tolist()],
        "gold_usd_oz": [round(float(v), 2) for v in merged["gold_usd_oz"].tolist()],
        "usd_krw": [round(float(v), 2) for v in merged["usd_krw"].tolist()],
        "high_gap_periods": high_gap_periods,
    }

    cache_data = {"timestamp": datetime.now().isoformat(), "data": data}
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False)
    logger.info("Data cached successfully")

    return data
