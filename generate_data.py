"""정적 배포용 data.json 생성 스크립트"""

import json
import sys
from datetime import datetime

from data_fetcher import fetch_international_gold, fetch_krx_gold, calculate_gap, find_high_gap_periods


def main():
    print("Fetching data...")
    intl_df = fetch_international_gold()
    domestic_df = fetch_krx_gold()
    merged = calculate_gap(intl_df, domestic_df)
    high_gap_periods = find_high_gap_periods(merged)

    data = {
        "dates": [d.strftime("%Y-%m-%d") for d in merged.index],
        "domestic_price": [round(float(v), 0) for v in merged["domestic_krw_per_gram"]],
        "intl_price": [round(float(v), 0) for v in merged["intl_krw_per_gram"]],
        "gap_pct": [round(float(v), 2) for v in merged["gap_pct"]],
        "gold_usd_oz": [round(float(v), 2) for v in merged["gold_usd_oz"]],
        "usd_krw": [round(float(v), 2) for v in merged["usd_krw"]],
        "high_gap_periods": high_gap_periods,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M KST"),
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"data.json generated: {len(data['dates'])} data points")


if __name__ == "__main__":
    main()
