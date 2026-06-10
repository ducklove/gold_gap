"""Yahoo Finance(yfinance) 소스: 환율, 금 선물, 크립토 국제가, KRX 금 ETF."""

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from goldgap.domain.gap import attach_fx_and_convert
from goldgap.sources.common import upsert_latest_row
from goldgap.sources.naver_etf import KRX_GOLD_ETF, get_gold_grams_per_unit

logger = logging.getLogger(__name__)


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


def fetch_exchange_rate(start_date=None):
    """KRW=X 환율을 공통으로 추출"""
    end = datetime.now()
    start = start_date or (end - timedelta(days=5 * 365))

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


def fetch_new_york_gold_futures(fx_df, start_date=None):
    """yfinance로 국제 금 선물(GC=F) 조회, 공통 환율 재사용"""
    end = datetime.now()
    start = start_date or (end - timedelta(days=5 * 365))

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


def fetch_krx_gold(start_date=None):
    """411060 ETF(ACE KRX금현물) 시세로 국내 금 가격(원/g) 산출"""
    grams_per_unit = get_gold_grams_per_unit()

    end = datetime.now()
    start = start_date or (end - timedelta(days=5 * 365))

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


def fetch_international_crypto(ticker, fx_df, start_date=None):
    """yfinance로 크립토 국제 가격 조회 (BTC-USD, USDT-USD 등)"""
    end = datetime.now()
    start = start_date or (end - timedelta(days=5 * 365))

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
