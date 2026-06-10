"""직렬화된 자산 데이터의 증분 병합 (generate_data.py에서 이동).

data.json에 저장되는 리스트 기반 페이로드(dates + 시리즈들)를 다루며,
겹치는 날짜는 새 데이터가 우선한다. 병합 후 고괴리 구간은
domain.periods.find_high_gap_periods 단일 규칙으로 전체 재계산한다.
"""

from datetime import datetime, timedelta

from goldgap.assets import ASSETS
from goldgap.domain.periods import find_high_gap_periods

# 증분 fetch 시 마지막 날짜에서 거슬러 올라가 다시 받는 겹침 일수
OVERLAP_DAYS = 7

SERIES_KEYS = [
    "domestic_price",
    "intl_price",
    "gap_pct",
    "usd_krw",
    "crypto_usd",
    "gold_usd_oz",
]
METADATA_KEYS = [
    "domestic_source",
    "domestic_label",
    "default_intl_mode",
    "sources",
]

# market 블록 시리즈 키 — 자산 페이로드와 달리 결측(null)이 정상 상태(휴장일)
MARKET_SERIES_KEYS = ["kospi", "sp500", "usd_krw"]


def get_last_date(asset_data):
    """자산 데이터에서 마지막 날짜 추출"""
    if asset_data and asset_data.get("dates"):
        return datetime.strptime(asset_data["dates"][-1], "%Y-%m-%d")
    return None


def compute_incremental_start_dates(existing_data, overlap_days=OVERLAP_DAYS):
    """자산·market별 증분 fetch 시작일 계산 (마지막 날짜 - overlap_days).

    기존 데이터가 없거나 해당 블록이 비어 있으면 키를 생략한다 → 전체 fetch.
    market을 포함해야 공통 환율(fx) fetch 시작일(min)이 market 범위도 덮는다.
    """
    start_dates = {}
    if not existing_data:
        return start_dates
    for name in [*ASSETS, "market"]:
        last = get_last_date(existing_data.get(name))
        if last:
            start_dates[name] = last - timedelta(days=overlap_days)
    return start_dates


def merge_asset_data(old_data, new_data, threshold):
    """기존 데이터와 새 데이터를 병합 (겹치는 날짜는 새 데이터 우선)"""
    if not old_data:
        return new_data
    if not new_data:
        return old_data

    combined = {}
    fields = [key for key in SERIES_KEYS if key in old_data or key in new_data]

    for source in [old_data, new_data]:
        for i, d in enumerate(source["dates"]):
            row = combined.setdefault(d, {})
            for key in fields:
                values = source.get(key)
                if values and i < len(values):
                    row[key] = values[i]

    required = ["domestic_price", "intl_price", "gap_pct", "usd_krw"]
    sorted_dates = sorted(
        d for d, row in combined.items()
        if all(key in row for key in required)
    )
    gap_pcts = [combined[d]["gap_pct"] for d in sorted_dates]

    merged = {
        "dates": sorted_dates,
        "domestic_price": [combined[d]["domestic_price"] for d in sorted_dates],
        "intl_price": [combined[d]["intl_price"] for d in sorted_dates],
        "gap_pct": gap_pcts,
        "usd_krw": [combined[d]["usd_krw"] for d in sorted_dates],
        "high_gap_periods": find_high_gap_periods(sorted_dates, gap_pcts, threshold),
    }

    for key in fields:
        if key not in merged and all(key in combined[d] for d in sorted_dates):
            merged[key] = [combined[d][key] for d in sorted_dates]

    for key in METADATA_KEYS:
        value = new_data.get(key) or old_data.get(key)
        if value:
            merged[key] = value

    mode_names = set(old_data.get("intl_modes", {})) | set(new_data.get("intl_modes", {}))
    if mode_names:
        merged["intl_modes"] = {}
        for mode in sorted(mode_names):
            old_mode = old_data.get("intl_modes", {}).get(mode)
            new_mode = new_data.get("intl_modes", {}).get(mode)
            if old_mode and new_mode:
                merged["intl_modes"][mode] = merge_asset_data(old_mode, new_mode, threshold)
            else:
                merged["intl_modes"][mode] = new_mode or old_mode

    return merged


def merge_market_data(old_data, new_data):
    """market 블록 병합: 날짜 합집합, 겹치는 날짜는 새 값 우선.

    자산 병합과 달리 결측(null)이 정상 상태(휴장일)라 필수 필드 기준으로
    행을 제외하지 않으며, periods 재계산도 없다. 새 값이 null이면 기존
    값을 덮어쓰지 않는다 — 한쪽 지수만 갱신돼도 기존 관측값을 보존한다.
    """
    if not old_data:
        return new_data
    if not new_data:
        return old_data

    combined = {}
    for source in [old_data, new_data]:
        for i, d in enumerate(source["dates"]):
            row = combined.setdefault(d, {})
            for key in MARKET_SERIES_KEYS:
                values = source.get(key)
                if values and i < len(values) and values[i] is not None:
                    row[key] = values[i]

    sorted_dates = sorted(combined)
    merged = {"dates": sorted_dates}
    for key in MARKET_SERIES_KEYS:
        merged[key] = [combined[d].get(key) for d in sorted_dates]

    sources = new_data.get("sources") or old_data.get("sources")
    if sources:
        merged["sources"] = sources

    return merged
