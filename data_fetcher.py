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
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyupbit
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "all_data.json")
TROY_OZ_TO_GRAM = 31.1035
KST = ZoneInfo("Asia/Seoul")

NAVER_ETF_BASIC_URL = "https://m.stock.naver.com/api/etf/{code}/basic"
KRX_GOLD_ETF = "411060"
KRX_GOLD_ETF_CU_SIZE = 100_000  # 1 CU = 100,000 좌
WGC_GOLD_SPOT_URL = "https://fsapi.gold.org/api/goldprice/v13/chart/price/usd/oz/{start},{end}?cache09092024"
GOLD_API_SPOT_URL = "https://api.gold-api.com/price/XAU/USD"
BITHUMB_CANDLE_URL = "https://api.bithumb.com/public/candlestick/{symbol}/24h"
BITHUMB_TICKER_URL = "https://api.bithumb.com/public/ticker/{symbol}"

# Asset-specific gap thresholds
THRESHOLDS = {
    "gold": 5.0,
    "bitcoin": 5.0,
    "usdt": 3.0,
}


def today_kst():
    return pd.Timestamp(datetime.now(KST).date())


def normalize_yfinance_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def latest_yfinance_price(ticker):
    """yfinance fast_info에서 최신 quote를 조회한다."""
    try:
        value = yf.Ticker(ticker).fast_info.get("lastPrice")
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception as e:
        logger.warning(f"Latest yfinance quote failed for {ticker}: {e}")
        return None


def upsert_latest_row(df, row):
    """오늘(KST) 행을 추가하거나 최신 quote로 덮어쓴다."""
    if not row:
        return df

    result = df.copy()
    idx = today_kst()
    for key, value in row.items():
        result.loc[idx, key] = value
    return result.sort_index()


def fetch_exchange_rate(years=5):
    """KRW=X 환율을 공통으로 추출"""
    end = datetime.now()
    start = end - timedelta(days=years * 365)

    logger.info("Fetching USD/KRW exchange rate (KRW=X)...")
    fx = yf.download("KRW=X", start=start, end=end, progress=False)

    if fx.empty:
        raise ValueError("Failed to fetch exchange rate data")

    fx = normalize_yfinance_columns(fx)

    fx_df = fx[["Close"]].rename(columns={"Close": "usd_krw"})
    fx_df.index = pd.to_datetime(fx_df.index).tz_localize(None)
    latest = latest_yfinance_price("KRW=X")
    if latest:
        fx_df = upsert_latest_row(fx_df, {"usd_krw": latest})

    return fx_df


def attach_fx_and_convert(gold_df, fx_df, usd_col, price_col):
    fx_filled = fx_df.reindex(gold_df.index).ffill().bfill()
    intl = gold_df.join(fx_filled, how="left").dropna(subset=[usd_col, "usd_krw"])
    intl[price_col] = (intl[usd_col] * intl["usd_krw"]) / TROY_OZ_TO_GRAM
    return intl


def fetch_new_york_gold_futures(fx_df):
    """yfinance로 국제 금 선물(GC=F) 조회, 공통 환율 재사용"""
    end = datetime.now()
    start = end - timedelta(days=5 * 365)

    logger.info("Fetching New York gold futures (GC=F)...")
    gold = yf.download("GC=F", start=start, end=end, progress=False)

    if gold.empty:
        raise ValueError("Failed to fetch international gold data")

    gold = normalize_yfinance_columns(gold)

    gold_df = gold[["Close"]].rename(columns={"Close": "gold_usd_oz"})
    gold_df.index = pd.to_datetime(gold_df.index).tz_localize(None)

    latest = latest_yfinance_price("GC=F")
    if latest:
        gold_df = upsert_latest_row(gold_df, {"gold_usd_oz": latest})

    return attach_fx_and_convert(gold_df, fx_df, "gold_usd_oz", "intl_price")


def fetch_london_spot_gold(fx_df):
    """World Gold Council/ICE spot gold 시계열 + Gold API 최신 XAU spot quote."""
    end = datetime.now(KST)
    start = end - timedelta(days=5 * 365)
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


def _get_gold_grams_per_unit():
    """네이버 ETF API에서 411060 CU 구성을 조회하여 1좌당 금 그램수 계산"""
    resp = requests.get(
        NAVER_ETF_BASIC_URL.format(code=KRX_GOLD_ETF),
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    for item in data.get("constituentList", []):
        if "금" in item.get("itemName", ""):
            cu_grams = item["cuUnitQuantity"]
            grams_per_unit = cu_grams / KRX_GOLD_ETF_CU_SIZE
            logger.info(f"ETF CU gold: {cu_grams}g / {KRX_GOLD_ETF_CU_SIZE} units = {grams_per_unit:.4f} g/unit")
            return grams_per_unit

    raise ValueError("Gold constituent not found in ETF CU data")


def fetch_krx_gold():
    """411060 ETF(ACE KRX금현물) 시세로 국내 금 가격(원/g) 산출"""
    grams_per_unit = _get_gold_grams_per_unit()

    end = datetime.now()
    start = end - timedelta(days=5 * 365)

    logger.info("Fetching KRX gold ETF (411060.KS) via yfinance...")
    etf = yf.download(KRX_GOLD_ETF + ".KS", start=start, end=end, progress=False)

    if etf.empty:
        raise ValueError("Failed to fetch 411060.KS data")

    etf = normalize_yfinance_columns(etf)

    etf.index = pd.to_datetime(etf.index).tz_localize(None)
    domestic_price = etf["Close"] / grams_per_unit

    result = pd.DataFrame({"domestic_price": domestic_price}).dropna()
    latest = latest_yfinance_price(KRX_GOLD_ETF + ".KS")
    if latest:
        result = upsert_latest_row(result, {"domestic_price": latest / grams_per_unit})

    logger.info(
        f"KRX gold data: {len(result)} rows "
        f"({result.index.min().strftime('%Y-%m-%d')} ~ {result.index.max().strftime('%Y-%m-%d')}), "
        f"factor={1/grams_per_unit:.2f} KRW·unit/g"
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

    crypto = normalize_yfinance_columns(crypto)

    crypto_df = crypto[["Close"]].rename(columns={"Close": "crypto_usd"})
    crypto_df.index = pd.to_datetime(crypto_df.index).tz_localize(None).normalize()
    latest = latest_yfinance_price(ticker)
    if latest:
        crypto_df = upsert_latest_row(crypto_df, {"crypto_usd": latest})

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


def fetch_bithumb_ohlcv(symbol, years=5):
    """빗썸 일봉 조회. USDT/KRW처럼 업비트보다 긴 원화 히스토리가 필요한 경우 사용."""
    logger.info(f"Fetching Bithumb OHLCV for {symbol}...")
    resp = requests.get(
        BITHUMB_CANDLE_URL.format(symbol=symbol),
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "0000" or not payload.get("data"):
        raise ValueError(f"Failed to fetch Bithumb data for {symbol}")

    rows = []
    target_start = today_kst() - pd.Timedelta(days=years * 365)
    for ts_ms, _open, close, _high, _low, _volume in payload["data"]:
        date = pd.to_datetime(int(ts_ms), unit="ms", utc=True).tz_convert(KST).tz_localize(None).normalize()
        if date >= target_start:
            rows.append((date, float(close)))

    if not rows:
        raise ValueError(f"No Bithumb rows for {symbol}")

    domestic = pd.DataFrame(rows, columns=["date", "domestic_price"]).set_index("date")
    domestic = domestic[~domestic.index.duplicated(keep="last")].sort_index()

    try:
        ticker_resp = requests.get(
            BITHUMB_TICKER_URL.format(symbol=symbol),
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        ticker_resp.raise_for_status()
        ticker_payload = ticker_resp.json()
        if ticker_payload.get("status") == "0000":
            latest = float(ticker_payload["data"]["closing_price"])
            domestic = upsert_latest_row(domestic, {"domestic_price": latest})
    except Exception as e:
        logger.warning(f"Latest Bithumb ticker failed for {symbol}: {e}")

    logger.info(
        f"Bithumb {symbol}: {len(domestic)} rows "
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


def serialize_asset_data(merged, periods, extra_columns=None):
    """DataFrame → dict 변환 헬퍼"""
    payload = {
        "dates": [d.strftime("%Y-%m-%d") for d in merged.index],
        "domestic_price": [round(float(v), 2) for v in merged["domestic_price"]],
        "intl_price": [round(float(v), 2) for v in merged["intl_price"]],
        "gap_pct": [round(float(v), 2) for v in merged["gap_pct"]],
        "usd_krw": [round(float(v), 2) for v in merged["usd_krw"]],
        "high_gap_periods": periods,
    }
    for col in extra_columns or []:
        if col in merged:
            payload[col] = [round(float(v), 6) for v in merged[col]]
    return payload


def get_gold_data(fx_df):
    """금 자산 오케스트레이터"""
    logger.info("=== Fetching GOLD data ===")
    domestic_df = fetch_krx_gold()
    ny_df = fetch_new_york_gold_futures(fx_df)
    spot_df = fetch_london_spot_gold(fx_df)

    modes = {}
    for mode, intl_df in [("ny_futures", ny_df), ("london_spot", spot_df)]:
        merged = calculate_gap(intl_df, domestic_df)
        periods = find_high_gap_periods(merged, threshold=THRESHOLDS["gold"])
        modes[mode] = serialize_asset_data(merged, periods, extra_columns=["gold_usd_oz"])

    data = dict(modes["ny_futures"])
    data["intl_modes"] = modes
    data["default_intl_mode"] = "ny_futures"
    data["sources"] = {
        "domestic": "ACE KRX금현물(411060.KS) latest/daily close from Yahoo Finance, converted by ETF gold grams per unit",
        "fx": "USD/KRW KRW=X latest/daily close from Yahoo Finance",
        "international": {
            "ny_futures": "COMEX Gold Futures GC=F latest/daily close from Yahoo Finance",
            "london_spot": "World Gold Council/ICE gold spot chart data plus latest XAU spot quote from Gold API",
        },
    }
    return data


def get_bitcoin_data(fx_df):
    """비트코인 자산 오케스트레이터"""
    logger.info("=== Fetching BITCOIN data ===")
    intl_df = fetch_international_crypto("BTC-USD", fx_df)
    domestic_df = fetch_upbit_ohlcv("KRW-BTC")
    latest_domestic = pyupbit.get_current_price("KRW-BTC")
    if latest_domestic:
        domestic_df = upsert_latest_row(domestic_df, {"domestic_price": float(latest_domestic)})
    merged = calculate_gap(intl_df, domestic_df)
    periods = find_high_gap_periods(merged, threshold=THRESHOLDS["bitcoin"])
    data = serialize_asset_data(merged, periods, extra_columns=["crypto_usd"])
    data["sources"] = {
        "domestic": "Upbit KRW-BTC daily candles plus pyupbit current price",
        "international": "BTC-USD latest/daily close from Yahoo Finance, converted to KRW",
        "fx": "USD/KRW KRW=X latest/daily close from Yahoo Finance",
    }
    return data


def get_usdt_data(fx_df):
    """USDT 자산 오케스트레이터"""
    logger.info("=== Fetching USDT data ===")
    intl_df = fetch_international_crypto("USDT-USD", fx_df)
    domestic_df = fetch_bithumb_ohlcv("USDT_KRW")
    merged = calculate_gap(intl_df, domestic_df)
    periods = find_high_gap_periods(merged, threshold=THRESHOLDS["usdt"])
    data = serialize_asset_data(merged, periods, extra_columns=["crypto_usd"])
    data["sources"] = {
        "domestic": "Bithumb USDT_KRW daily candles plus current ticker",
        "international": "USDT-USD latest/daily close from Yahoo Finance, converted to KRW",
        "fx": "USD/KRW KRW=X latest/daily close from Yahoo Finance",
    }
    return data


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
