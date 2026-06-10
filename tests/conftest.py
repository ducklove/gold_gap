"""공용 픽스처: 리포 루트 경로와 골든 data.json 로더 (네트워크 불필요)."""

import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_JSON_PATH = os.path.join(REPO_ROOT, "data.json")
CONFIG_JSON_PATH = os.path.join(REPO_ROOT, "config.json")


@pytest.fixture(scope="session")
def golden_data():
    """실제 data.json (읽기 전용 골든 파일).

    data.json은 data 전용 브랜치에 산다 — CI는 자동으로 받아오고,
    로컬에는 없을 수 있으므로 안내와 함께 skip한다.
    """
    if not os.path.exists(DATA_JSON_PATH):
        pytest.skip(
            "data.json 없음 — `git fetch --depth=1 origin data && "
            "git show FETCH_HEAD:data.json > data.json`으로 받아오세요 (CI는 자동)"
        )
    with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def config_json():
    """리포 루트 config.json (프론트/외부 계약)."""
    with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
