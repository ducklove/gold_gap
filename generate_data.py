"""정적 배포용 data.json 생성 스크립트 (증분 업데이트 지원) — 얇은 CLI.

세 자산(gold, bitcoin, usdt)을 순차 fetch.
기존 data.json이 있으면 마지막 날짜 이후만 가져와서 병합.
개별 자산 fetch 실패 시 기존 데이터 유지, 전체 실패 시 기존 data.json 폴백.

증분 시작일 계산/병합/자산별 폴백 핵심 로직은 goldgap 패키지에 있고,
이 스크립트는 기존 data.json 로딩 → 수집 호출 → meta/updated_at 부착 →
파일 출력만 담당한다. (운영 cron 인터페이스: `python generate_data.py`)
"""

import json
import logging
import os
import sys

from goldgap.assets import ASSETS
from goldgap.orchestrators import fetch_fresh
from goldgap.serialize import build_meta, format_updated_at

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.json")


def load_existing_data():
    """기존 data.json 로드"""
    if not os.path.exists(DATA_FILE):
        return None
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if any(data.get(a) for a in ASSETS):
            return data
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to load existing data.json: {e}")
    return None


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
            data = {k: v for k, v in existing_data.items() if k in ASSETS}
        else:
            print("ERROR: No data available", file=sys.stderr)
            sys.exit(1)

    # 모든 출력 경로(전체/증분/전체 실패 폴백)에 KST 갱신 시각(BUG-01)과 meta 블록 부착
    data["updated_at"] = format_updated_at()
    data["meta"] = build_meta()

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print("data.json written successfully")


if __name__ == "__main__":
    main()
