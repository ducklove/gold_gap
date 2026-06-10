"""Flask 로컬 미리보기 경로용 24시간 파일 캐시 (기존 data_fetcher 동작 유지).

캐시 위치는 리포 루트의 cache/ 디렉터리 (.gitignore 대상).
"""

import json
import logging
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_REPO_ROOT, "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "all_data.json")
CACHE_TTL = timedelta(hours=24)


def load_cached_data():
    """24시간 이내의 캐시가 있으면 반환, 없거나 손상이면 None."""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_time = datetime.fromisoformat(cached["timestamp"])
        if datetime.now() - cached_time < CACHE_TTL:
            logger.info("Using cached data")
            return cached["data"]
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Cache read failed: {e}")
    return None


def store_cached_data(data):
    """수집 결과를 타임스탬프와 함께 캐시 파일에 저장."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_data = {"timestamp": datetime.now().isoformat(), "data": data}
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False)
    logger.info("Data cached successfully")
