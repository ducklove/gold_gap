"""임계 돌파 알림: 직전 data.json 대비 최신 괴리율의 임계치 신규 진입 탐지.

운영 워크플로우(update-data.yml)가 데이터 갱신 직후 호출한다:

    python -m goldgap.alerts <이전 data.json> <새 data.json>

신규 진입(직전엔 임계 미만 → 지금은 임계 이상)인 자산만 사람용 한 줄씩
stdout으로 출력한다. 출력이 비어 있으면 알림 없음. 이미 임계 이상이던
자산은 다시 알리지 않는다 (30분 주기 실행의 알림 노이즈 방지).

알림은 파이프라인을 실패시키면 안 되므로 어떤 입력 오류에도 exit 0.
"""

import json
import sys

from goldgap.assets import ASSETS


def _latest_gap(asset_data):
    """자산 페이로드에서 (최신 괴리율, 날짜) 추출. 없으면 (None, None)."""
    if not isinstance(asset_data, dict):
        return None, None
    gaps = asset_data.get("gap_pct") or []
    dates = asset_data.get("dates") or []
    if not gaps or not dates:
        return None, None
    gap = gaps[-1]
    if gap is None or not isinstance(gap, (int, float)) or gap != gap:
        return None, None
    return float(gap), str(dates[-1])


def detect_threshold_crossings(old_data, new_data):
    """레지스트리 자산별로 임계치 신규 진입을 찾는다.

    진입 조건: 새 데이터의 최신 |gap| >= threshold 이면서,
    직전 데이터의 최신 |gap| < threshold (또는 직전 데이터 부재).
    """
    crossings = []
    for key, asset in ASSETS.items():
        new_gap, new_date = _latest_gap((new_data or {}).get(key))
        if new_gap is None or abs(new_gap) < asset.threshold_pct:
            continue
        old_gap, _ = _latest_gap((old_data or {}).get(key))
        if old_gap is not None and abs(old_gap) >= asset.threshold_pct:
            continue  # 이미 임계 이상이던 자산 — 신규 진입만 알림
        crossings.append(
            {
                "asset": key,
                "label": asset.label,
                "date": new_date,
                "gap_pct": round(new_gap, 2),
                "threshold_pct": asset.threshold_pct,
            }
        )
    return crossings


def format_crossings(crossings):
    """이슈 본문용 한국어 한 줄 포맷."""
    lines = []
    for c in crossings:
        direction = "프리미엄" if c["gap_pct"] > 0 else "디스카운트"
        lines.append(
            f"- {c['label']} ({c['asset']}): {c['date']} 괴리율 {c['gap_pct']:+.2f}% — "
            f"임계 ±{c['threshold_pct']:g}% {direction} 진입"
        )
    return "\n".join(lines)


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def main(argv):
    if len(argv) != 3:
        print("usage: python -m goldgap.alerts <old_data.json> <new_data.json>", file=sys.stderr)
        return 0
    old_data = _load_json(argv[1])
    new_data = _load_json(argv[2])
    crossings = detect_threshold_crossings(old_data, new_data)
    if crossings:
        print(format_crossings(crossings))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
