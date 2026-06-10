"""임계 돌파 알림 탐지 테스트 (네트워크 불필요)."""

from goldgap.alerts import detect_threshold_crossings, format_crossings


def _asset(gap, date="2026-06-10"):
    return {"dates": [date], "gap_pct": [gap]}


def test_new_crossing_detected():
    """직전 임계 미만 → 새 데이터 임계 이상이면 알림."""
    old = {"bitcoin": _asset(3.2, "2026-06-09")}
    new = {"bitcoin": _asset(5.4)}
    crossings = detect_threshold_crossings(old, new)
    assert len(crossings) == 1
    c = crossings[0]
    assert c["asset"] == "bitcoin"
    assert c["gap_pct"] == 5.4
    assert c["threshold_pct"] == 5.0
    assert c["date"] == "2026-06-10"


def test_negative_gap_crossing_detected():
    """디스카운트(음수) 방향도 절대값 기준으로 진입 탐지."""
    old = {"usdt": _asset(-1.0)}
    new = {"usdt": _asset(-3.5)}
    crossings = detect_threshold_crossings(old, new)
    assert [c["asset"] for c in crossings] == ["usdt"]
    assert crossings[0]["gap_pct"] == -3.5


def test_already_above_threshold_not_realerted():
    """직전에도 임계 이상이던 자산은 다시 알리지 않는다 (노이즈 방지)."""
    old = {"gold": _asset(6.1, "2026-06-09")}
    new = {"gold": _asset(7.2)}
    assert detect_threshold_crossings(old, new) == []


def test_below_threshold_no_alert():
    old = {"bitcoin": _asset(1.0)}
    new = {"bitcoin": _asset(4.99)}
    assert detect_threshold_crossings(old, new) == []


def test_missing_old_data_alerts_if_above():
    """직전 데이터가 없으면(첫 실행 등) 임계 이상일 때 알림."""
    new = {"eth": _asset(5.0)}  # 경계 동치 포함
    crossings = detect_threshold_crossings(None, new)
    assert [c["asset"] for c in crossings] == ["eth"]


def test_multiple_assets_and_threshold_per_asset():
    """자산별 임계(usdt 3%, 나머지 5%)가 각각 적용된다."""
    old = {"gold": _asset(1.0), "bitcoin": _asset(1.0), "eth": _asset(1.0), "usdt": _asset(1.0)}
    new = {
        "gold": _asset(4.0),    # 5% 미만 — 알림 없음
        "bitcoin": _asset(5.1),  # 진입
        "eth": _asset(-5.2),     # 진입(음수)
        "usdt": _asset(3.1),     # 진입(3% 임계)
    }
    keys = [c["asset"] for c in detect_threshold_crossings(old, new)]
    assert keys == ["bitcoin", "eth", "usdt"]  # 레지스트리 order 순


def test_malformed_payloads_are_ignored():
    """결측/비정상 페이로드는 조용히 건너뛴다 (알림이 파이프라인을 깨면 안 됨)."""
    new = {
        "gold": {"dates": [], "gap_pct": []},
        "bitcoin": {"dates": ["2026-06-10"]},  # gap_pct 없음
        "eth": None,
        "usdt": {"dates": ["2026-06-10"], "gap_pct": [None]},
    }
    assert detect_threshold_crossings({}, new) == []


def test_format_crossings_korean_lines():
    crossings = detect_threshold_crossings({}, {"bitcoin": _asset(5.4), "usdt": _asset(-3.5)})
    text = format_crossings(crossings)
    lines = text.splitlines()
    assert len(lines) == 2
    assert "Bitcoin (bitcoin): 2026-06-10 괴리율 +5.40%" in lines[0]
    assert "프리미엄 진입" in lines[0]
    assert "디스카운트 진입" in lines[1]
