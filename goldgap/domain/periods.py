"""고괴리율 구간 탐지 — 단일 구현 (BUG-02+05).

기존 3중 구현(data_fetcher.find_high_gap_periods / generate_data.calc_high_gap_periods /
프론트엔드 buildHighGapPeriods)을 이 리스트 기반 함수 하나로 통일한다.

확정 규칙:
- 구간 = |gap| >= threshold (경계 동치 포함)인 연속 데이터일
- start = 첫 초과 데이터일, end = 마지막 초과 데이터일 (실존 데이터일)
- max_gap = 절대값이 최대인 gap의 서명값, 소수 2자리 반올림
- duration_days = (end - start).days + 1, 최소 1
- 진행 중 구간은 마지막 데이터일로 닫는다

구 data_fetcher.find_high_gap_periods의 "임계 미달 첫날 - 1달력일"을 end로 쓰던
방식은 폐기한다 (주말/휴장일이 끼면 실존하지 않는 날짜가 end가 되는 문제).
"""

from datetime import datetime


def _close_period(start, end, max_gap):
    duration = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days + 1
    return {
        "start": start,
        "end": end,
        "max_gap": round(max_gap, 2),
        "duration_days": max(duration, 1),
    }


def find_high_gap_periods(dates: list[str], gap_pcts: list[float], threshold: float) -> list[dict]:
    """|gap| >= threshold 연속 구간을 찾아 요약한다.

    Args:
        dates: "%Y-%m-%d" 형식 날짜 문자열 리스트 (오름차순, gap_pcts와 같은 길이)
        gap_pcts: 날짜별 괴리율(%) 리스트 (결측값 None/NaN은 판정에서 건너뜀)
        threshold: 고괴리 임계값(%) — 절대값 기준, 경계 포함

    Returns:
        [{"start", "end", "max_gap", "duration_days"}, ...]
    """
    periods = []
    in_period = False
    start = None
    last_exceed = None
    max_gap = 0.0

    for date, gap in zip(dates, gap_pcts, strict=True):
        if gap is None or gap != gap:  # 결측값(None/NaN)은 구간을 닫지도 늘리지도 않음
            continue
        if abs(gap) >= threshold:
            if not in_period:
                in_period = True
                start = date
                max_gap = gap
            elif abs(gap) > abs(max_gap):
                max_gap = gap
            last_exceed = date
        elif in_period:
            periods.append(_close_period(start, last_exceed, max_gap))
            in_period = False

    if in_period:
        periods.append(_close_period(start, last_exceed, max_gap))

    return periods
