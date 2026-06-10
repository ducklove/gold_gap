"""generate_data.py CLI 통합 테스트 (네트워크 차단 환경의 전체 실패 폴백 경로).

tmp 디렉터리에 data.json과 스크립트를 복사해 실행한다 — 스크립트는
자신의 디렉터리 기준으로 data.json을 읽고 쓰므로 리포의 골든 data.json은
건드리지 않는다. 외부 API가 차단된 환경에서는 fetch가 전부 실패하고
기존 data.json 폴백 + meta/updated_at 부착 후 exit 0이어야 한다.
"""

import json
import os
import re
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_JSON_PATH = os.path.join(REPO_ROOT, "data.json")


def test_cli_fallback_keeps_existing_data_and_attaches_meta(tmp_path, golden_data):
    shutil.copy(DATA_JSON_PATH, tmp_path / "data.json")
    shutil.copy(os.path.join(REPO_ROOT, "generate_data.py"), tmp_path / "generate_data.py")

    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, "generate_data.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "data.json written successfully" in result.stdout

    with open(tmp_path / "data.json", "r", encoding="utf-8") as f:
        written = json.load(f)

    # 기존 자산 데이터 보존 (전체 실패 폴백 경로)
    for asset_key in ["gold", "bitcoin", "usdt"]:
        assert written[asset_key] == golden_data[asset_key], f"{asset_key} payload changed"

    # updated_at: KST 형식으로 새로 기록
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} KST", written["updated_at"])

    # meta 블록: 모든 출력 경로에 부착되는 프론트 계약
    meta = written["meta"]
    assert meta["schema_version"] == 2
    assert set(meta["assets"]) == {"gold", "bitcoin", "usdt"}
    assert "+09:00" in meta["generated_at"]


def test_cli_exits_nonzero_without_any_data(tmp_path):
    """기존 data.json도 없고 fetch도 전부 실패하면 exit 1 (cron 실패 신호)."""
    shutil.copy(os.path.join(REPO_ROOT, "generate_data.py"), tmp_path / "generate_data.py")

    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, "generate_data.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 1
    assert "ERROR: No data available" in result.stderr
    assert not (tmp_path / "data.json").exists()
