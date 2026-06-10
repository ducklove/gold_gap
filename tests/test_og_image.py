"""OG 이미지 렌더링 테스트 (네트워크 불필요)."""

import os

from PIL import Image

from goldgap.og_image import HEIGHT, WIDTH, render_og_image


def _fake_data():
    return {
        "updated_at": "2026-06-11 01:32 KST",
        "gold": {"dates": ["2026-06-11"], "gap_pct": [1.23]},
        "bitcoin": {"dates": ["2026-06-11"], "gap_pct": [-0.81]},
        "eth": {"dates": ["2026-06-11"], "gap_pct": [0.0]},
        "usdt": {"dates": ["2026-06-11"], "gap_pct": [-3.5]},
    }


def test_render_creates_og_sized_png(tmp_path):
    out = tmp_path / "og.png"
    render_og_image(_fake_data(), str(out))
    assert os.path.getsize(out) > 0
    with Image.open(out) as img:
        assert img.size == (WIDTH, HEIGHT)
        assert img.format == "PNG"


def test_render_tolerates_missing_assets(tmp_path):
    """자산 일부가 없거나 비어 있어도 렌더링이 죽지 않는다."""
    out = tmp_path / "og.png"
    render_og_image({"gold": {"dates": [], "gap_pct": []}, "bitcoin": None}, str(out))
    assert os.path.getsize(out) > 0


def test_render_with_real_golden_data(tmp_path, golden_data):
    out = tmp_path / "og.png"
    render_og_image(golden_data, str(out))
    with Image.open(out) as img:
        assert img.size == (WIDTH, HEIGHT)
