"""공용 픽스처: 리포 루트 경로와 골든 data.json 로더 (네트워크 불필요)."""

import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_JSON_PATH = os.path.join(REPO_ROOT, "data.json")
CONFIG_JSON_PATH = os.path.join(REPO_ROOT, "config.json")


@pytest.fixture(scope="session")
def golden_data():
    """리포에 커밋된 실제 data.json (읽기 전용 골든 파일)."""
    with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def config_json():
    """리포 루트 config.json (프론트/외부 계약)."""
    with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
