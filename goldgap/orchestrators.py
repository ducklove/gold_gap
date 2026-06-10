"""자산별 데이터 조립 오케스트레이터: 소스 fetch → 괴리율 계산 → 직렬화."""

import logging

from goldgap import cache
from goldgap.assets import get_threshold
from goldgap.domain.gap import calculate_gap
from goldgap.domain.merge import compute_incremental_start_dates, merge_asset_data
from goldgap.domain.periods import find_high_gap_periods
from goldgap.serialize import build_meta, format_updated_at, serialize_asset_data
from goldgap.sources.bithumb import fetch_bithumb_ohlcv
from goldgap.sources.upbit import fetch_upbit_ohlcv
from goldgap.sources.wgc import fetch_london_spot_gold
from goldgap.sources.yahoo import (
    fetch_exchange_rate,
    fetch_international_crypto,
    fetch_krx_gold,
    fetch_new_york_gold_futures,
)

logger = logging.getLogger(__name__)


def _periods_from_merged(merged, threshold):
    """병합 DataFrame에서 고괴리 구간 계산.

    직렬화 출력(gap_pct 소수 2자리 반올림)과 동일한 값으로 단일 규칙
    find_high_gap_periods를 호출한다 — 증분 병합 재계산/프론트엔드와
    같은 입력 정밀도를 보장하기 위함.
    """
    dates = [d.strftime("%Y-%m-%d") for d in merged.index]
    gap_pcts = [round(float(v), 2) for v in merged["gap_pct"]]
    return find_high_gap_periods(dates, gap_pcts, threshold)


def get_gold_data(fx_df, start_date=None):
    """금 자산 오케스트레이터"""
    logger.info("=== Fetching GOLD data ===")
    domestic_df = fetch_krx_gold(start_date)
    ny_df = fetch_new_york_gold_futures(fx_df, start_date)
    spot_df = fetch_london_spot_gold(fx_df, start_date)

    modes = {}
    for mode, intl_df in [("ny_futures", ny_df), ("london_spot", spot_df)]:
        merged = calculate_gap(intl_df, domestic_df)
        periods = _periods_from_merged(merged, get_threshold("gold"))
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


def get_bitcoin_data(fx_df, start_date=None):
    """비트코인 자산 오케스트레이터"""
    logger.info("=== Fetching BITCOIN data ===")
    intl_df = fetch_international_crypto("BTC-USD", fx_df, start_date)
    domestic_df = fetch_upbit_ohlcv("KRW-BTC", start_date, include_latest=True)
    merged = calculate_gap(intl_df, domestic_df)
    periods = _periods_from_merged(merged, get_threshold("bitcoin"))
    data = serialize_asset_data(merged, periods, extra_columns=["crypto_usd"])
    data["sources"] = {
        "domestic": "Upbit KRW-BTC daily candles plus current ticker",
        "international": "BTC-USD latest/daily close from Yahoo Finance, converted to KRW",
        "fx": "USD/KRW KRW=X latest/daily close from Yahoo Finance",
    }
    return data


def fetch_usdt_domestic(start_date=None):
    """USDT 국내 가격 조회: 빗썸 우선, 실패 시 업비트 백업"""
    sources = [
        (
            "bithumb",
            "빗썸 USDT",
            lambda: fetch_bithumb_ohlcv(
                "KRW-USDT",
                start_date,
                include_latest=True,
                public_symbol="USDT_KRW",
            ),
        ),
        (
            "upbit",
            "업비트 USDT (백업)",
            lambda: fetch_upbit_ohlcv("KRW-USDT", start_date, include_latest=True),
        ),
    ]
    errors = []

    for source_key, source_label, fetcher in sources:
        try:
            domestic_df = fetcher()
            return domestic_df, {
                "domestic_source": source_key,
                "domestic_label": source_label,
            }
        except Exception as e:
            logger.warning(f"USDT domestic fetch failed from {source_label}: {e}")
            errors.append(f"{source_label}: {e}")

    raise ValueError("Failed to fetch USDT domestic data: " + "; ".join(errors))


def get_usdt_data(fx_df, start_date=None):
    """USDT 자산 오케스트레이터"""
    logger.info("=== Fetching USDT data ===")
    intl_df = fetch_international_crypto("USDT-USD", fx_df, start_date)
    domestic_df, metadata = fetch_usdt_domestic(start_date)
    merged = calculate_gap(intl_df, domestic_df)
    periods = _periods_from_merged(merged, get_threshold("usdt"))
    data = serialize_asset_data(
        merged,
        periods,
        extra_columns=["crypto_usd"],
        metadata=metadata,
    )
    data["sources"] = {
        "domestic": f"{metadata['domestic_label']} daily candles; falls back to Upbit when Bithumb fails",
        "international": "USDT-USD latest/daily close from Yahoo Finance, converted to KRW",
        "fx": "USD/KRW KRW=X latest/daily close from Yahoo Finance",
    }
    return data


def fetch_fresh(existing_data=None):
    """세 자산을 순차 fetch (기존 데이터가 있으면 증분 업데이트 후 병합).

    개별 자산 실패 시 기존 데이터를 유지하고, 결과가 하나도 없으면
    RuntimeError를 던진다 (호출 측에서 전체 폴백 처리).
    반환값은 자산 키만 담는다 — updated_at/meta는 호출 측에서 부착.
    """
    start_dates = compute_incremental_start_dates(existing_data)

    if start_dates:
        fx_start = min(start_dates.values())
        logger.info(f"Incremental update from {fx_start.strftime('%Y-%m-%d')}")
    else:
        fx_start = None
        logger.info("Full fetch (no existing data)")

    fx_df = fetch_exchange_rate(fx_start)

    data = {}
    errors = []

    fetchers = [("gold", get_gold_data), ("bitcoin", get_bitcoin_data), ("usdt", get_usdt_data)]
    for name, fetcher in fetchers:
        try:
            start = start_dates.get(name)
            new_data = fetcher(fx_df, start)

            if start and existing_data and existing_data.get(name):
                data[name] = merge_asset_data(existing_data[name], new_data, get_threshold(name))
                logger.info(
                    f"{name}: {len(data[name]['dates'])} points "
                    f"(incremental, +{len(new_data['dates'])} fetched)"
                )
            else:
                data[name] = new_data
                logger.info(f"{name}: {len(data[name]['dates'])} points (full)")
        except Exception as e:
            logger.warning(f"{name}: FAILED - {e}")
            errors.append(name)
            if existing_data and existing_data.get(name):
                data[name] = existing_data[name]
                logger.info(f"{name}: kept existing data ({len(existing_data[name]['dates'])} points)")

    if not data:
        raise RuntimeError(f"All assets failed: {errors}")

    return data


def get_all_data(force_refresh=False):
    """전체 데이터 수집 + 24시간 파일 캐시 (Flask 로컬 미리보기 경로)."""
    if not force_refresh:
        cached = cache.load_cached_data()
        if cached is not None:
            return cached

    logger.info("Fetching fresh data for all assets...")
    fx_df = fetch_exchange_rate()

    data = {}
    data["gold"] = get_gold_data(fx_df)
    data["bitcoin"] = get_bitcoin_data(fx_df)
    data["usdt"] = get_usdt_data(fx_df)
    # 정적 data.json과 같은 계약: KST 갱신 시각(BUG-01) + meta 블록
    data["updated_at"] = format_updated_at()
    data["meta"] = build_meta()

    cache.store_cached_data(data)
    return data
