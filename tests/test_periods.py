"""domain.periods.find_high_gap_periods 단위 테스트 (통일 규칙 검증)."""

from goldgap.domain.periods import find_high_gap_periods


def test_single_day_period():
    """단일일 구간: start == end, duration_days == 1."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    gaps = [1.0, 6.2, 0.5]
    assert find_high_gap_periods(dates, gaps, 5.0) == [
        {"start": "2024-01-02", "end": "2024-01-02", "max_gap": 6.2, "duration_days": 1},
    ]


def test_multi_day_period_ends_on_last_exceeding_day():
    """다중일 구간: end는 마지막 '초과 데이터일' (임계 미달일 -1 달력일 아님)."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    gaps = [5.5, 6.0, 7.1, 2.0, 1.0]
    assert find_high_gap_periods(dates, gaps, 5.0) == [
        {"start": "2024-01-01", "end": "2024-01-03", "max_gap": 7.1, "duration_days": 3},
    ]


def test_weekend_gap_inside_period():
    """주말이 끼어 데이터일이 연속되지 않아도 하나의 구간으로 보고,
    duration_days는 달력일 기준 (금~월 = 4일)."""
    dates = ["2024-01-05", "2024-01-08", "2024-01-09"]  # 금, 월, 화
    gaps = [5.5, 6.5, 1.0]
    assert find_high_gap_periods(dates, gaps, 5.0) == [
        {"start": "2024-01-05", "end": "2024-01-08", "max_gap": 6.5, "duration_days": 4},
    ]


def test_ongoing_period_closed_at_last_data_day():
    """진행 중 구간은 마지막 데이터일로 닫는다."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    gaps = [1.0, 5.1, 8.0]
    assert find_high_gap_periods(dates, gaps, 5.0) == [
        {"start": "2024-01-02", "end": "2024-01-03", "max_gap": 8.0, "duration_days": 2},
    ]


def test_negative_gap_keeps_sign():
    """음수 괴리: 절대값으로 판정하되 max_gap은 서명값을 유지한다."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    gaps = [-5.5, -7.2, -0.4]
    assert find_high_gap_periods(dates, gaps, 5.0) == [
        {"start": "2024-01-01", "end": "2024-01-02", "max_gap": -7.2, "duration_days": 2},
    ]


def test_mixed_sign_max_gap_by_absolute_value():
    """양/음 혼재 구간에서 max_gap은 절대값 최대의 서명값."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    gaps = [5.5, -8.0, 6.0]
    assert find_high_gap_periods(dates, gaps, 5.0) == [
        {"start": "2024-01-01", "end": "2024-01-03", "max_gap": -8.0, "duration_days": 3},
    ]


def test_threshold_boundary_inclusive():
    """경계 동치(|gap| == threshold)도 구간에 포함된다."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    gaps = [3.0, -3.0, 2.99]
    assert find_high_gap_periods(dates, gaps, 3.0) == [
        {"start": "2024-01-01", "end": "2024-01-02", "max_gap": 3.0, "duration_days": 2},
    ]


def test_empty_input():
    """빈 입력은 빈 결과."""
    assert find_high_gap_periods([], [], 5.0) == []


def test_all_below_threshold():
    """전체 임계 미달이면 구간 없음."""
    dates = ["2024-01-01", "2024-01-02"]
    gaps = [4.99, -4.99]
    assert find_high_gap_periods(dates, gaps, 5.0) == []


def test_multiple_periods():
    """임계 미달일을 사이에 둔 구간은 분리된다."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    gaps = [6.0, 1.0, -5.5, -6.5]
    assert find_high_gap_periods(dates, gaps, 5.0) == [
        {"start": "2024-01-01", "end": "2024-01-01", "max_gap": 6.0, "duration_days": 1},
        {"start": "2024-01-03", "end": "2024-01-04", "max_gap": -6.5, "duration_days": 2},
    ]


def test_max_gap_rounding():
    """max_gap은 소수 2자리 반올림."""
    dates = ["2024-01-01"]
    gaps = [5.6789]
    assert find_high_gap_periods(dates, gaps, 5.0) == [
        {"start": "2024-01-01", "end": "2024-01-01", "max_gap": 5.68, "duration_days": 1},
    ]


def test_none_and_nan_gaps_are_skipped():
    """결측값(None/NaN)은 구간을 닫지도 늘리지도 않는다."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    gaps = [6.0, None, float("nan"), 7.0]
    assert find_high_gap_periods(dates, gaps, 5.0) == [
        {"start": "2024-01-01", "end": "2024-01-04", "max_gap": 7.0, "duration_days": 4},
    ]
