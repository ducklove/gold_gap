"""정적 배포용 data.json 생성 스크립트

세 자산(gold, bitcoin, usdt)을 순차 fetch.
개별 자산 fetch 실패 시 해당 자산만 건너뛰고, 전체 실패 시 캐시 폴백.
"""

import json
import logging
import os
import sys
from datetime import datetime

from data_fetcher import (
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
CACHE_FILE = os.path.join(BASE_DIR, "cache", "all_data.json")
OLD_GOLD_CACHE = os.path.join(BASE_DIR, "cache", "gold_data.json")


def fetch_fresh():
    fx_df = fetch_exchange_rate()

    data = {}
    errors = []

    for name, fetcher in [("gold", get_gold_data), ("bitcoin", get_bitcoin_data), ("usdt", get_usdt_data)]:
        try:
            data[name] = fetcher(fx_df)
            print(f"  {name}: {len(data[name]['dates'])} data points")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            errors.append(name)

    # Gold 실패 시 기존 캐시에서 폴백
    if "gold" not in data and os.path.exists(OLD_GOLD_CACHE):
        try:
            with open(OLD_GOLD_CACHE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            old_gold = cached["data"]
            # 기존 컬럼명 호환: domestic_krw_per_gram → domestic_price, intl_krw_per_gram → intl_price
            if "domestic_krw_per_gram" in str(old_gold.keys()) or "domestic_price" in old_gold:
                data["gold"] = old_gold
                print(f"  gold: {len(old_gold['dates'])} data points (from old cache)")
        except Exception as e:
            print(f"  gold old cache fallback failed: {e}")

    if not data:
        raise RuntimeError(f"All assets failed: {errors}")

    return data


def load_from_cache():
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cached = json.load(f)
    return cached["data"]


def main():
    data = None

    print("Fetching fresh data...")
    try:
        data = fetch_fresh()
    except Exception as e:
        print(f"Fresh fetch failed: {e}")

    if data is None:
        print("Falling back to cached data...")
        if os.path.exists(CACHE_FILE):
            data = load_from_cache()
            print("Cached data loaded")
        else:
            print("ERROR: No cached data available", file=sys.stderr)
            sys.exit(1)

    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M KST")

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print("data.json written successfully")


if __name__ == "__main__":
    main()
