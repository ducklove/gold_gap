"""generate_data.py CLI 통합 테스트 (전체 실패 폴백 경로).

tmp 디렉터리에 data.json과 스크립트를 복사해 실행한다 — 스크립트는
자신의 디렉터리 기준으로 data.json을 읽고 쓰므로 리포의 골든 data.json은
건드리지 않는다. fetch가 전부 실패하면 기존 data.json 폴백 +
meta/updated_at 부착 후 exit 0이어야 한다.

fetch 실패는 도달 불가 프록시 주입으로 강제한다 — CI(GitHub Actions)처럼
실제 네트워크가 있는 환경에서는 수집이 진짜로 성공해버려 폴백 경로가
실행되지 않으므로, 환경과 무관하게 결정적으로 만들기 위함이다.
"""

import json
import os
import re
import shutil
import subprocess
import sys

from goldgap.assets import ASSETS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_JSON_PATH = os.path.join(REPO_ROOT, "data.json")


def _offline_env():
    """모든 외부 HTTP 호출을 즉시 실패시키는 서브프로세스 env.

    127.0.0.1:9(discard 포트)를 프록시로 강제 — requests는 대소문자 env를,
    libcurl 계열은 소문자 env를 읽으므로 둘 다 설정한다.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env[key] = "http://127.0.0.1:9"
    for key in ("NO_PROXY", "no_proxy"):
        env[key] = ""
    return env


def test_cli_fallback_keeps_existing_data_and_attaches_meta(tmp_path, golden_data):
    shutil.copy(DATA_JSON_PATH, tmp_path / "data.json")
    shutil.copy(os.path.join(REPO_ROOT, "generate_data.py"), tmp_path / "generate_data.py")

    env = _offline_env()

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
    # 골든에 실재하는 자산만 검사 — data 브랜치 재수집 윈도우(일시 제거)에 강건
    for asset_key in [k for k in ASSETS if k in golden_data]:
        assert written[asset_key] == golden_data[asset_key], f"{asset_key} payload changed"

    # updated_at: KST 형식으로 새로 기록
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} KST", written["updated_at"])

    # meta 블록: 모든 출력 경로에 부착되는 프론트 계약
    # (meta는 레지스트리 전체를 담는다 — 아직 데이터가 없는 신규 자산 포함)
    meta = written["meta"]
    assert meta["schema_version"] == 2
    assert set(meta["assets"]) == {"gold", "bitcoin", "eth", "usdt"}
    assert "+09:00" in meta["generated_at"]


def test_cli_fallback_preserves_market_block(tmp_path, golden_data):
    """기존 data.json에 market 블록이 있으면 전체 실패 폴백에서도 보존된다."""
    fake_market = {
        "dates": ["2026-06-01", "2026-06-02"],
        "kospi": [2712.14, None],
        "sp500": [5352.96, 5354.03],
        "usd_krw": [1378.5, None],
        "sources": {"kospi": "test-kospi", "sp500": "test-sp500", "fx": "test-fx"},
    }
    seeded = dict(golden_data)  # 골든 픽스처는 불변 — 최상위 키만 추가한 사본
    seeded["market"] = fake_market
    with open(tmp_path / "data.json", "w", encoding="utf-8") as f:
        json.dump(seeded, f, ensure_ascii=False)
    shutil.copy(os.path.join(REPO_ROOT, "generate_data.py"), tmp_path / "generate_data.py")

    result = subprocess.run(
        [sys.executable, "generate_data.py"],
        cwd=tmp_path,
        env=_offline_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    with open(tmp_path / "data.json", "r", encoding="utf-8") as f:
        written = json.load(f)

    assert written["market"] == fake_market, "market block lost in fallback"
    for asset_key in [k for k in ASSETS if k in golden_data]:
        assert written[asset_key] == golden_data[asset_key], f"{asset_key} payload changed"


def test_cli_exits_nonzero_without_any_data(tmp_path):
    """기존 data.json도 없고 fetch도 전부 실패하면 exit 1 (cron 실패 신호)."""
    shutil.copy(os.path.join(REPO_ROOT, "generate_data.py"), tmp_path / "generate_data.py")

    env = _offline_env()

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
