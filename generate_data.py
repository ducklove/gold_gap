"""정적 배포용 data.json 생성 스크립트

신규 데이터 fetch 실패 시 기존 캐시(cache/gold_data.json)를 폴백으로 사용.
"""

import json
import os
import sys
from datetime import datetime

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "gold_data.json")


def fetch_fresh():
    from data_fetcher import fetch_international_gold, fetch_krx_gold, calculate_gap, find_high_gap_periods

    intl_df = fetch_international_gold()
    domestic_df = fetch_krx_gold()
    merged = calculate_gap(intl_df, domestic_df)
    high_gap_periods = find_high_gap_periods(merged)

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in merged.index],
        "domestic_price": [round(float(v), 0) for v in merged["domestic_krw_per_gram"]],
        "intl_price": [round(float(v), 0) for v in merged["intl_krw_per_gram"]],
        "gap_pct": [round(float(v), 2) for v in merged["gap_pct"]],
        "gold_usd_oz": [round(float(v), 2) for v in merged["gold_usd_oz"]],
        "usd_krw": [round(float(v), 2) for v in merged["usd_krw"]],
        "high_gap_periods": high_gap_periods,
    }


def load_from_cache():
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cached = json.load(f)
    return cached["data"]


def main():
    data = None

    print("Fetching fresh data...")
    try:
        data = fetch_fresh()
        print(f"Fresh data: {len(data['dates'])} data points")
    except Exception as e:
        print(f"Fresh fetch failed: {e}")

    if data is None:
        print("Falling back to cached data...")
        if os.path.exists(CACHE_FILE):
            data = load_from_cache()
            print(f"Cached data: {len(data['dates'])} data points")
        else:
            print("ERROR: No cached data available", file=sys.stderr)
            sys.exit(1)

    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M KST")

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"data.json written successfully")


if __name__ == "__main__":
    main()
