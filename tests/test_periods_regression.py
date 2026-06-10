"""골든 회귀: data.json에 저장된 high_gap_periods가 통일 규칙 재계산값과 정확히 일치.

운영 cron(generate_data.py 증분 병합 경로)이 마지막으로 기록한 값과
domain.periods.find_high_gap_periods 단일 구현의 출력이 같음을 보증한다 —
규칙 통일(BUG-02+05)이 기존 저장 데이터와 호환됨을 확인하는 외부 계약 가드.
"""

import pytest

from goldgap.assets import get_threshold
from goldgap.domain.periods import find_high_gap_periods


@pytest.mark.parametrize("asset_key", ["gold", "bitcoin", "usdt"])
def test_stored_periods_match_unified_rule(golden_data, asset_key):
    asset = golden_data[asset_key]
    recomputed = find_high_gap_periods(asset["dates"], asset["gap_pct"], get_threshold(asset_key))
    assert recomputed == asset["high_gap_periods"]


@pytest.mark.parametrize("mode", ["ny_futures", "london_spot"])
def test_stored_gold_mode_periods_match_unified_rule(golden_data, mode):
    payload = golden_data["gold"]["intl_modes"][mode]
    recomputed = find_high_gap_periods(payload["dates"], payload["gap_pct"], get_threshold("gold"))
    assert recomputed == payload["high_gap_periods"]
