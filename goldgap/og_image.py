"""OG 공유 이미지 생성: data.json의 최신 괴리율로 1200x630 카드 PNG 렌더링.

운영 워크플로우(update-data.yml)가 데이터 갱신 직후 호출해 data 브랜치에
data.json과 함께 커밋하고, 배포가 사이트 루트(/og.png)로 복사한다:

    python -m goldgap.og_image <data.json> <out.png>

GitHub Actions 러너에 한글 폰트가 없으므로 텍스트는 영문/숫자만 사용한다.
이미지 생성 실패가 데이터 갱신을 막으면 안 되므로 호출 측에서 비치명 처리한다.
"""

import sys

from PIL import Image, ImageDraw, ImageFont

from goldgap.assets import ASSETS

WIDTH, HEIGHT = 1200, 630

# 대시보드 다크 테마 팔레트 (static/style.css :root와 동일 계열)
BG = "#0f1117"
SURFACE = "#1a1d27"
BORDER = "#2e3345"
TEXT = "#e4e7f1"
TEXT_DIM = "#8b90a5"
PREMIUM = "#f0615e"   # 국내가 비쌈 (+)
DISCOUNT = "#4a90d9"  # 국내가 쌈 (-)

_DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_DEJAVU_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size, bold=False):
    """DejaVu가 있으면 사용, 없으면 Pillow 내장 폰트로 폴백."""
    try:
        return ImageFont.truetype(_DEJAVU_BOLD if bold else _DEJAVU, size)
    except OSError:
        return ImageFont.load_default(size=size)


def _latest_gap(asset_data):
    if not isinstance(asset_data, dict):
        return None
    gaps = asset_data.get("gap_pct") or []
    if not gaps or not isinstance(gaps[-1], (int, float)):
        return None
    return float(gaps[-1])


def render_og_image(data, out_path):
    """data.json dict에서 자산별 최신 괴리율 카드 PNG를 만든다."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    # 상단 골드 액센트 바
    draw.rectangle([0, 0, WIDTH, 8], fill=ASSETS["gold"].color)

    draw.text((64, 52), "Kimchi Premium", font=_font(62, bold=True), fill=TEXT)
    draw.text(
        (64, 132),
        "Korea vs global price gap - gold & crypto, updated every 30 min",
        font=_font(25),
        fill=TEXT_DIM,
    )

    # 자산 행: 레지스트리 순서, 데이터에 있는 자산만
    rows = [(asset, _latest_gap((data or {}).get(key))) for key, asset in ASSETS.items()]
    rows = [(asset, gap) for asset, gap in rows if gap is not None]

    top, row_h = 208, 92
    label_font, gap_font, sub_font = _font(36, bold=True), _font(40, bold=True), _font(20)
    for i, (asset, gap) in enumerate(rows):
        y = top + i * row_h
        draw.rounded_rectangle([56, y, WIDTH - 56, y + row_h - 14], radius=12, fill=SURFACE, outline=BORDER)
        draw.text((88, y + 18), asset.label, font=label_font, fill=TEXT)

        gap_text = f"{gap:+.2f}%"
        color = PREMIUM if gap > 0 else DISCOUNT if gap < 0 else TEXT_DIM
        gap_w = draw.textlength(gap_text, font=gap_font)
        draw.text((WIDTH - 88 - gap_w, y + 16), gap_text, font=gap_font, fill=color)

        sub = f"alert at +/-{asset.threshold_pct:g}%"
        sub_w = draw.textlength(sub, font=sub_font)
        draw.text((WIDTH - 88 - gap_w - 28 - sub_w, y + 30), sub, font=sub_font, fill=TEXT_DIM)

    footer = "ducklove.github.io/gold_gap"
    updated = str((data or {}).get("updated_at") or "")
    if updated:
        footer = f"Updated {updated}  ·  {footer}"
    draw.text((64, HEIGHT - 48), footer, font=_font(22), fill=TEXT_DIM)

    img.save(out_path, "PNG")


def main(argv):
    if len(argv) != 3:
        print("usage: python -m goldgap.og_image <data.json> <out.png>", file=sys.stderr)
        return 2
    import json

    with open(argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
    render_og_image(data, argv[2])
    print(f"og image written: {argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
