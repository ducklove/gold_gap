"""domain.merge.merge_asset_data 단위 테스트 + data.json 골든 회귀."""

import copy

import pytest

from goldgap.assets import get_threshold
from goldgap.domain.merge import SERIES_KEYS, compute_incremental_start_dates, merge_asset_data


def make_payload(dates, base=100.0, **overrides):
    """테스트용 직렬화 페이로드 생성 헬퍼."""
    n = len(dates)
    payload = {
        "dates": list(dates),
        "domestic_price": [base + i for i in range(n)],
        "intl_price": [base for _ in range(n)],
        "gap_pct": [round((base + i - base) / base * 100, 2) for i in range(n)],
        "usd_krw": [1300.0 + i for i in range(n)],
    }
    payload.update(overrides)
    return payload


def test_empty_old_returns_new():
    new = make_payload(["2024-01-01"])
    assert merge_asset_data(None, new, 5.0) is new


def test_empty_new_returns_old():
    old = make_payload(["2024-01-01"])
    assert merge_asset_data(old, None, 5.0) is old


def test_overlap_new_data_wins():
    """겹치는 날짜는 새 데이터가 우선한다."""
    old = make_payload(["2024-01-01", "2024-01-02"])
    new = {
        "dates": ["2024-01-02", "2024-01-03"],
        "domestic_price": [999.0, 998.0],
        "intl_price": [900.0, 900.0],
        "gap_pct": [11.0, 10.89],
        "usd_krw": [1400.0, 1401.0],
    }
    merged = merge_asset_data(old, new, 5.0)
    assert merged["dates"] == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert merged["domestic_price"] == [100.0, 999.0, 998.0]
    assert merged["usd_krw"] == [1300.0, 1400.0, 1401.0]
    assert merged["gap_pct"] == [0.0, 11.0, 10.89]


def test_output_sorted_by_date():
    """입력 순서와 무관하게 날짜 오름차순으로 정렬된다."""
    old = make_payload(["2024-01-05", "2024-01-06"])
    new = make_payload(["2024-01-02", "2024-01-01"], base=200.0)
    merged = merge_asset_data(old, new, 5.0)
    assert merged["dates"] == ["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-06"]
    # 정렬 후에도 날짜-값 짝이 유지된다 (2024-01-01은 new의 두 번째 항목)
    assert merged["domestic_price"][0] == 201.0


def test_rows_missing_required_fields_are_dropped():
    """필수 필드(usd_krw 등)가 없는 날짜 행은 제외된다."""
    old = make_payload(["2024-01-01"])
    new = {
        "dates": ["2024-01-02", "2024-01-03"],
        "domestic_price": [110.0, 111.0],
        "intl_price": [100.0, 100.0],
        "gap_pct": [10.0, 11.0],
        "usd_krw": [1310.0],  # 2024-01-03 값 누락
    }
    merged = merge_asset_data(old, new, 5.0)
    assert merged["dates"] == ["2024-01-01", "2024-01-02"]
    assert merged["usd_krw"] == [1300.0, 1310.0]


def test_periods_recomputed_with_unified_rule():
    """병합 후 high_gap_periods는 통일 규칙으로 전체 재계산된다."""
    old = make_payload(["2024-01-01", "2024-01-02"], gap_pct=[6.0, 6.5])
    old["high_gap_periods"] = [{"start": "wrong", "end": "wrong", "max_gap": 0, "duration_days": 0}]
    new = make_payload(["2024-01-03"], gap_pct=[1.0])
    merged = merge_asset_data(old, new, 5.0)
    assert merged["high_gap_periods"] == [
        {"start": "2024-01-01", "end": "2024-01-02", "max_gap": 6.5, "duration_days": 2},
    ]


def test_intl_modes_merged_recursively():
    """intl_modes는 모드별로 재귀 병합되고, 한쪽에만 있는 모드는 그대로 채택된다."""
    old = make_payload(["2024-01-01"])
    old["intl_modes"] = {"ny_futures": make_payload(["2024-01-01"])}
    new = make_payload(["2024-01-02"])
    new["intl_modes"] = {
        "ny_futures": make_payload(["2024-01-02"], base=300.0),
        "london_spot": make_payload(["2024-01-02"], base=400.0),
    }
    merged = merge_asset_data(old, new, 5.0)
    assert set(merged["intl_modes"]) == {"ny_futures", "london_spot"}
    assert merged["intl_modes"]["ny_futures"]["dates"] == ["2024-01-01", "2024-01-02"]
    assert merged["intl_modes"]["ny_futures"]["domestic_price"] == [100.0, 300.0]
    assert merged["intl_modes"]["london_spot"]["dates"] == ["2024-01-02"]


def test_metadata_keys_preserved_new_first():
    """메타데이터 키는 새 데이터 우선으로 보존된다."""
    old = make_payload(["2024-01-01"])
    old.update({"domestic_source": "upbit", "domestic_label": "업비트 USDT (백업)", "sources": {"a": 1}})
    new = make_payload(["2024-01-02"])
    new.update({"domestic_source": "bithumb", "domestic_label": "빗썸 USDT"})
    merged = merge_asset_data(old, new, 3.0)
    assert merged["domestic_source"] == "bithumb"
    assert merged["domestic_label"] == "빗썸 USDT"
    assert merged["sources"] == {"a": 1}  # 새 데이터에 없으면 기존 값 유지


def test_extra_series_kept_only_when_complete():
    """모든 날짜에 값이 있는 추가 시리즈(crypto_usd 등)만 포함된다."""
    old = make_payload(["2024-01-01"], crypto_usd=[50000.0])
    new = make_payload(["2024-01-02"])  # crypto_usd 없음
    merged = merge_asset_data(old, new, 5.0)
    assert "crypto_usd" not in merged


def test_compute_incremental_start_dates(golden_data):
    """증분 시작일 = 자산별 마지막 날짜 - OVERLAP_DAYS(7일)."""
    from datetime import datetime, timedelta

    starts = compute_incremental_start_dates(golden_data)
    # 골든 데이터에 존재하는 레지스트리 자산 전체 — 자산 추가에 따라 자동 확장
    from goldgap.assets import ASSETS
    expected = {key for key in ASSETS if (golden_data.get(key) or {}).get("dates")}
    assert starts and set(starts) == expected
    for key in starts:
        last = datetime.strptime(golden_data[key]["dates"][-1], "%Y-%m-%d")
        assert starts[key] == last - timedelta(days=7)
    assert compute_incremental_start_dates(None) == {}
    assert compute_incremental_start_dates({"gold": {"dates": []}}) == {}


def _tail_slice(payload, n=7):
    """페이로드의 마지막 n개 날짜 슬라이스 (실제 증분 fetch 결과 모사)."""
    sliced = {"dates": payload["dates"][-n:]}
    for key in SERIES_KEYS:
        if key in payload:
            sliced[key] = payload[key][-n:]
    for key in ["domestic_source", "domestic_label", "default_intl_mode", "sources"]:
        if key in payload:
            sliced[key] = copy.deepcopy(payload[key])
    if "intl_modes" in payload:
        sliced["intl_modes"] = {
            mode: _tail_slice(mode_payload, n) for mode, mode_payload in payload["intl_modes"].items()
        }
    return sliced


@pytest.mark.parametrize("asset_key", ["gold", "bitcoin", "usdt"])
def test_golden_merge_idempotent(golden_data, asset_key):
    """핵심 회귀: 실제 data.json 골든 기준 merge(old, old 마지막 7일) == old.

    dates/모든 시리즈가 무손상이어야 하고, high_gap_periods는 통일 규칙
    재계산값(저장값과 동일함이 test_periods_regression에서 보증)이어야 한다.
    """
    old = golden_data[asset_key]
    new = _tail_slice(old, 7)
    merged = merge_asset_data(copy.deepcopy(old), new, get_threshold(asset_key))
    assert merged == old
