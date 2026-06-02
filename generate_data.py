"""정적 배포용 data.json 생성 스크립트 (증분 업데이트 지원)

세 자산(gold, bitcoin, usdt)을 순차 fetch.
기존 data.json이 있으면 마지막 날짜 이후만 가져와서 병합.
개별 자산 fetch 실패 시 기존 데이터 유지, 전체 실패 시 기존 data.json 폴백.
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta

from data_fetcher import (
    THRESHOLDS,
    fetch_exchange_rate,
    get_bitcoin_data,
    get_gold_data,
    get_usdt_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.json")
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


def load_existing_data():
    """기존 data.json 로드"""
    if not os.path.exists(DATA_FILE):
        return None
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if any(data.get(a) for a in ["gold", "bitcoin", "usdt"]):
            return data
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to load existing data.json: {e}")
    return None


def get_last_date(asset_data):
    """자산 데이터에서 마지막 날짜 추출"""
    if asset_data and asset_data.get("dates"):
        return datetime.strptime(asset_data["dates"][-1], "%Y-%m-%d")
    return None


def calc_high_gap_periods(dates, gap_pcts, threshold):
    """직렬화된 리스트에서 고괴리율 구간 재계산"""
    periods = []
    in_period = False
    start = max_gap = None

    for i, gap in enumerate(gap_pcts):
        if abs(gap) >= threshold:
            if not in_period:
                in_period = True
                start = dates[i]
                max_gap = gap
            elif abs(gap) > abs(max_gap):
                max_gap = gap
        elif in_period:
            end = dates[i - 1]
            d = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days + 1
            periods.append({
                "start": start, "end": end,
                "max_gap": round(max_gap, 2),
                "duration_days": max(d, 1),
            })
            in_period = False

    if in_period:
        end = dates[-1]
        d = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days + 1
        periods.append({
            "start": start, "end": end,
            "max_gap": round(max_gap, 2),
            "duration_days": max(d, 1),
        })

    return periods


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
        "high_gap_periods": calc_high_gap_periods(sorted_dates, gap_pcts, threshold),
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


def fetch_fresh(existing_data=None):
    """데이터 수집 (기존 데이터가 있으면 증분 업데이트)"""
    start_dates = {}
    if existing_data:
        for name in ["gold", "bitcoin", "usdt"]:
            last = get_last_date(existing_data.get(name))
            if last:
                start_dates[name] = last - timedelta(days=OVERLAP_DAYS)

    incremental = bool(start_dates)
    if incremental:
        fx_start = min(start_dates.values())
        print(f"Incremental update from {fx_start.strftime('%Y-%m-%d')}")
    else:
        fx_start = None
        print("Full fetch (no existing data)")

    fx_df = fetch_exchange_rate(fx_start)

    data = {}
    errors = []

    fetchers = [("gold", get_gold_data), ("bitcoin", get_bitcoin_data), ("usdt", get_usdt_data)]
    for name, fetcher in fetchers:
        try:
            start = start_dates.get(name)
            new_data = fetcher(fx_df, start)

            if start and existing_data and existing_data.get(name):
                data[name] = merge_asset_data(
                    existing_data[name], new_data, THRESHOLDS[name]
                )
                print(f"  {name}: {len(data[name]['dates'])} points (incremental, +{len(new_data['dates'])} fetched)")
            else:
                data[name] = new_data
                print(f"  {name}: {len(data[name]['dates'])} points (full)")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            errors.append(name)
            if existing_data and existing_data.get(name):
                data[name] = existing_data[name]
                print(f"  {name}: kept existing data ({len(existing_data[name]['dates'])} points)")

    if not data:
        raise RuntimeError(f"All assets failed: {errors}")

    return data


def main():
    existing_data = load_existing_data()

    data = None
    print("Fetching data...")
    try:
        data = fetch_fresh(existing_data)
    except Exception as e:
        print(f"Fetch failed: {e}")

    if data is None:
        if existing_data:
            print("Using existing data.json as fallback")
            data = {k: v for k, v in existing_data.items() if k in ["gold", "bitcoin", "usdt"]}
        else:
            print("ERROR: No data available", file=sys.stderr)
            sys.exit(1)

    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M KST")

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print("data.json written successfully")


if __name__ == "__main__":
    main()
