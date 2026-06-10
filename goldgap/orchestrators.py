"""자산별 데이터 조립 오케스트레이터: 소스 fetch → 괴리율 계산 → 직렬화."""

import logging

from goldgap import cache
from goldgap.assets import ASSETS, get_threshold
from goldgap.domain.gap import calculate_gap
from goldgap.domain.merge import (
    compute_incremental_start_dates,
    merge_asset_data,
    merge_market_data,
)
from goldgap.domain.periods import find_high_gap_periods
from goldgap.serialize import (
    build_meta,
    format_updated_at,
    serialize_asset_data,
    serialize_market_data,
)
from goldgap.sources.bithumb import fetch_bithumb_ohlcv
from goldgap.sources.upbit import fetch_upbit_ohlcv
from goldgap.sources.wgc import fetch_london_spot_gold
from goldgap.sources.yahoo import (
    fetch_exchange_rate,
    fetch_index_series,
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


def get_eth_data(fx_df, start_date=None):
    """이더리움 자산 오케스트레이터"""
    logger.info("=== Fetching ETH data ===")
    intl_df = fetch_international_crypto("ETH-USD", fx_df, start_date)
    domestic_df = fetch_upbit_ohlcv("KRW-ETH", start_date, include_latest=True)
    merged = calculate_gap(intl_df, domestic_df)
    periods = _periods_from_merged(merged, get_threshold("eth"))
    data = serialize_asset_data(merged, periods, extra_columns=["crypto_usd"])
    data["sources"] = {
        "domestic": "Upbit KRW-ETH daily candles plus current ticker",
        "international": "ETH-USD latest/daily close from Yahoo Finance, converted to KRW",
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


def get_market_data(fx_df, start_date=None):
    """시장 지표(KOSPI·S&P500·환율) 오케스트레이터 — data.json 'market' 블록.

    자산과 달리 한국/미국 휴장일이 서로 달라 outer join 합집합 날짜에
    결측은 null로 직렬화한다 (serialize_market_data). 두 지수 중 하나라도
    실패하면 market 전체 실패로 간주한다 — 단순성 우선, fetch_fresh의
    격리가 기존 market 데이터를 유지해준다.
    """
    logger.info("=== Fetching MARKET data ===")
    kospi_df = fetch_index_series("^KS11", "kospi", start_date)
    sp500_df = fetch_index_series("^GSPC", "sp500", start_date)

    fx_part = fx_df[["usd_krw"]]
    if start_date is not None:
        # fx_df는 전체 자산의 min 시작일 기준이라 더 길 수 있다 — 증분 범위로 절단
        fx_part = fx_part[fx_part.index >= start_date]

    market_df = kospi_df.join(sp500_df, how="outer").join(fx_part, how="outer")
    data = serialize_market_data(market_df)
    data["sources"] = {
        "kospi": "KOSPI ^KS11 latest/daily close from Yahoo Finance",
        "sp500": "S&P 500 ^GSPC latest/daily close from Yahoo Finance",
        "fx": "USD/KRW KRW=X latest/daily close from Yahoo Finance",
    }
    return data


def fetch_fresh(existing_data=None):
    """레지스트리 자산과 market 블록을 순차 fetch (기존 데이터가 있으면 증분 병합).

    개별 자산/market 실패 시 기존 데이터를 유지하고, 자산 결과가 하나도
    없으면 RuntimeError를 던진다 (호출 측에서 전체 폴백 처리 — market만
    성공해도 자산 없는 data.json은 만들지 않는다).
    반환값은 자산·market 키만 담는다 — updated_at/meta는 호출 측에서 부착.
    """
    start_dates = compute_incremental_start_dates(existing_data)

    if start_dates:
        fx_start = min(start_dates.values())
        logger.info(f"Incremental update from {fx_start.strftime('%Y-%m-%d')}")
    else:
        fx_start = None
        logger.info("Full fetch (no existing data)")

    fx_df = fetch_exchange_rate(fx_start)

    # 증분 런에서 전체 fetch가 필요한 블록(신규 자산·market 부재)용 전체 환율 캐시.
    # 증분 fx_df로 5년 시계열을 만들면 ffill/bfill이 과거 환율을 최근 값 상수로
    # 백필한다 — ETH 최초 수집에서 실제로 발생한 데이터 오염 사고의 재발 방지.
    full_fx_cache = {"df": fx_df if fx_start is None else None}

    def fx_covering(start):
        """start(None=전체 fetch)를 커버하는 환율 DataFrame을 반환한다."""
        if start is not None or fx_start is None:
            return fx_df
        if full_fx_cache["df"] is None:
            logger.info("Full FX history fetch for full-range block...")
            full_fx_cache["df"] = fetch_exchange_rate(None)
        return full_fx_cache["df"]

    data = {}
    errors = []

    fetchers = [
        ("gold", get_gold_data),
        ("bitcoin", get_bitcoin_data),
        ("eth", get_eth_data),
        ("usdt", get_usdt_data),
    ]
    for name, fetcher in fetchers:
        try:
            start = start_dates.get(name)
            new_data = fetcher(fx_covering(start), start)

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

    # market 블록 — 자산 루프와 동일한 격리 패턴 (실패 시 기존 유지)
    market_start = start_dates.get("market")
    try:
        # market 부재 시 5년 전체 fetch — fx_covering이 전체 환율을 보장(자산과 캐시 공유)
        new_market = get_market_data(fx_covering(market_start), market_start)

        if market_start and existing_data and existing_data.get("market"):
            data["market"] = merge_market_data(existing_data["market"], new_market)
            logger.info(
                f"market: {len(data['market']['dates'])} points "
                f"(incremental, +{len(new_market['dates'])} fetched)"
            )
        else:
            data["market"] = new_market
            logger.info(f"market: {len(data['market']['dates'])} points (full)")
    except Exception as e:
        logger.warning(f"market: FAILED - {e}")
        errors.append("market")
        if existing_data and existing_data.get("market"):
            data["market"] = existing_data["market"]
            logger.info(
                f"market: kept existing data ({len(existing_data['market']['dates'])} points)"
            )

    if not any(key in ASSETS for key in data):
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
    data["eth"] = get_eth_data(fx_df)
    data["usdt"] = get_usdt_data(fx_df)
    data["market"] = get_market_data(fx_df)
    # 정적 data.json과 같은 계약: KST 갱신 시각(BUG-01) + meta 블록
    data["updated_at"] = format_updated_at()
    data["meta"] = build_meta()

    cache.store_cached_data(data)
    return data
