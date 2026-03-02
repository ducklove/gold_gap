"""
데이터 수집 모듈: 김치프리미엄 멀티 자산 대시보드

지원 자산:
- GOLD: KRX 금현물 vs COMEX 금선물
- BITCOIN: 업비트 BTC vs Binance BTC (yfinance BTC-USD)
- USDT: 업비트 USDT vs 1 USD (yfinance USDT-USD ≈ 1)
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyupbit
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "all_data.json")
TROY_OZ_TO_GRAM = 31.1035

NAVER_GOLD_PRICES_URL = "https://api.stock.naver.com/marketindex/metals/CMDT_GD/prices"

# Asset-specific gap thresholds
THRESHOLDS = {
    "gold": 5.0,
    "bitcoin": 5.0,
    "usdt": 3.0,
}


def fetch_exchange_rate(years=5):
    """KRW=X 환율을 공통으로 추출"""
    end = datetime.now()
    start = end - timedelta(days=years * 365)

    logger.info("Fetching USD/KRW exchange rate (KRW=X)...")
    fx = yf.download("KRW=X", start=start, end=end, progress=False)

    if fx.empty:
        raise ValueError("Failed to fetch exchange rate data")

    if isinstance(fx.columns, pd.MultiIndex):
        fx.columns = fx.columns.get_level_values(0)

    fx_df = fx[["Close"]].rename(columns={"Close": "usd_krw"})
    fx_df.index = pd.to_datetime(fx_df.index).tz_localize(None)

    return fx_df


def fetch_international_gold(fx_df):
    """yfinance로 국제 금 선물(GC=F) 조회, 공통 환율 재사용"""
    end = datetime.now()
    start = end - timedelta(days=5 * 365)

    logger.info("Fetching international gold (GC=F)...")
    gold = yf.download("GC=F", start=start, end=end, progress=False)

    if gold.empty:
        raise ValueError("Failed to fetch international gold data")

    if isinstance(gold.columns, pd.MultiIndex):
        gold.columns = gold.columns.get_level_values(0)

    gold_df = gold[["Close"]].rename(columns={"Close": "gold_usd_oz"})
    gold_df.index = pd.to_datetime(gold_df.index).tz_localize(None)

    intl = gold_df.join(fx_df, how="inner")
    intl["intl_price"] = (intl["gold_usd_oz"] * intl["usd_krw"]) / TROY_OZ_TO_GRAM

    return intl


def fetch_naver_gold(years=5):
    """네이버 금융 API로 국내 금 가격(원/g) 조회"""
    logger.info("Fetching domestic gold prices from Naver Finance...")

    page_size = 60
    target_start = datetime.now() - timedelta(days=years * 365)
    rows = []

    for page in range(1, 200):
        resp = requests.get(
            NAVER_GOLD_PRICES_URL,
            params={"page": page, "pageSize": page_size},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break

        for item in data:
            dt = datetime.fromisoformat(item["localTradedAt"]).replace(tzinfo=None)
            price = float(item["closePrice"].replace(",", ""))
            rows.append({"date": dt.replace(hour=0, minute=0, second=0, microsecond=0), "domestic_price": price})

        oldest = datetime.fromisoformat(data[-1]["localTradedAt"])
        if oldest.replace(tzinfo=None) < target_start:
            break
        time.sleep(0.1)

    if not rows:
        raise ValueError("Failed to fetch Naver gold price data")

    result = pd.DataFrame(rows).set_index("date").sort_index()
    result = result[~result.index.duplicated(keep="first")]
    result = result[result.index >= target_start]

    logger.info(
        f"Naver gold data: {len(result)} rows "
        f"({result.index.min().strftime('%Y-%m-%d')} ~ {result.index.max().strftime('%Y-%m-%d')})"
    )
    return result


def fetch_international_crypto(ticker, fx_df):
    """yfinance로 크립토 국제 가격 조회 (BTC-USD, USDT-USD 등)"""
    end = datetime.now()
    start = end - timedelta(days=5 * 365)

    logger.info(f"Fetching international crypto ({ticker})...")
    crypto = yf.download(ticker, start=start, end=end, progress=False)

    if crypto.empty:
        raise ValueError(f"Failed to fetch {ticker} data")

    if isinstance(crypto.columns, pd.MultiIndex):
        crypto.columns = crypto.columns.get_level_values(0)

    crypto_df = crypto[["Close"]].rename(columns={"Close": "crypto_usd"})
    crypto_df.index = pd.to_datetime(crypto_df.index).tz_localize(None).normalize()

    # 크립토는 주말 거래, forex는 평일만 → left join + ffill
    fx_filled = fx_df.reindex(crypto_df.index).ffill().bfill()

    intl = crypto_df.join(fx_filled, how="left")
    intl = intl.dropna()
    intl["intl_price"] = intl["crypto_usd"] * intl["usd_krw"]

    return intl


def fetch_upbit_ohlcv(ticker, years=5):
    """pyupbit로 업비트 일봉 조회 (페이지네이션으로 전체 기간)"""
    logger.info(f"Fetching Upbit OHLCV for {ticker}...")

    all_dfs = []
    to = datetime.now()
    target_start = datetime.now() - timedelta(days=years * 365)

    while to > target_start:
        df = pyupbit.get_ohlcv(ticker, interval="day", count=200, to=to)
        if df is None or df.empty:
            break

        all_dfs.append(df)
        to = df.index[0] - timedelta(days=1)
        time.sleep(0.2)

    if not all_dfs:
        raise ValueError(f"Failed to fetch Upbit data for {ticker}")

    result = pd.concat(all_dfs).sort_index()
    result = result[~result.index.duplicated(keep="first")]

    # Filter to requested period
    result = result[result.index >= target_start]
    result.index = pd.to_datetime(result.index).tz_localize(None)

    # 업비트 인덱스는 09:00:00 KST가 포함됨 → 날짜만 남겨서 yfinance와 조인 가능하게
    result.index = result.index.normalize()

    domestic = pd.DataFrame({"domestic_price": result["close"]})
    domestic = domestic.dropna()

    logger.info(
        f"Upbit {ticker}: {len(domestic)} rows "
        f"({domestic.index.min().strftime('%Y-%m-%d')} ~ {domestic.index.max().strftime('%Y-%m-%d')})"
    )
    return domestic


def calculate_gap(intl_df, domestic_df):
    """국내/국제 가격 병합 및 괴리율 계산 (컬럼명: intl_price, domestic_price)"""
    merged = intl_df.join(domestic_df, how="inner")
    merged = merged.dropna(subset=["intl_price", "domestic_price"])

    if merged.empty:
        raise ValueError("No overlapping dates between international and domestic data")

    merged["gap_pct"] = (
        (merged["domestic_price"] - merged["intl_price"])
        / merged["intl_price"]
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


def serialize_asset_data(merged, periods):
    """DataFrame → dict 변환 헬퍼"""
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in merged.index],
        "domestic_price": [round(float(v), 2) for v in merged["domestic_price"]],
        "intl_price": [round(float(v), 2) for v in merged["intl_price"]],
        "gap_pct": [round(float(v), 2) for v in merged["gap_pct"]],
        "usd_krw": [round(float(v), 2) for v in merged["usd_krw"]],
        "high_gap_periods": periods,
    }


def get_gold_data(fx_df):
    """금 자산 오케스트레이터"""
    logger.info("=== Fetching GOLD data ===")
    intl_df = fetch_international_gold(fx_df)
    domestic_df = fetch_naver_gold()
    merged = calculate_gap(intl_df, domestic_df)
    periods = find_high_gap_periods(merged, threshold=THRESHOLDS["gold"])
    return serialize_asset_data(merged, periods)


def get_bitcoin_data(fx_df):
    """비트코인 자산 오케스트레이터"""
    logger.info("=== Fetching BITCOIN data ===")
    intl_df = fetch_international_crypto("BTC-USD", fx_df)
    domestic_df = fetch_upbit_ohlcv("KRW-BTC")
    merged = calculate_gap(intl_df, domestic_df)
    periods = find_high_gap_periods(merged, threshold=THRESHOLDS["bitcoin"])
    return serialize_asset_data(merged, periods)


def get_usdt_data(fx_df):
    """USDT 자산 오케스트레이터"""
    logger.info("=== Fetching USDT data ===")
    intl_df = fetch_international_crypto("USDT-USD", fx_df)
    domestic_df = fetch_upbit_ohlcv("KRW-USDT")
    merged = calculate_gap(intl_df, domestic_df)
    periods = find_high_gap_periods(merged, threshold=THRESHOLDS["usdt"])
    return serialize_asset_data(merged, periods)


def get_all_data(force_refresh=False):
    """전체 데이터 수집 + 캐시"""
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

    logger.info("Fetching fresh data for all assets...")
    fx_df = fetch_exchange_rate()

    data = {}
    data["gold"] = get_gold_data(fx_df)
    data["bitcoin"] = get_bitcoin_data(fx_df)
    data["usdt"] = get_usdt_data(fx_df)

    cache_data = {"timestamp": datetime.now().isoformat(), "data": data}
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False)
    logger.info("Data cached successfully")

    return data
