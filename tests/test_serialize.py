"""serialize 계층 테스트: 페이로드 키/타입, meta 블록 스키마, config.json 계약 가드."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from goldgap.assets import ASSETS
from goldgap.serialize import SCHEMA_VERSION, build_meta, format_updated_at, serialize_asset_data


def _sample_merged():
    index = pd.DatetimeIndex([datetime(2024, 1, 1), datetime(2024, 1, 2)])
    return pd.DataFrame(
        {
            "domestic_price": [1234.5678, 1300.004],
            "intl_price": [1200.0, 1250.0],
            "gap_pct": [2.8806, 4.0003],
            "usd_krw": [1305.999, 1310.0],
            "crypto_usd": [0.12345678, 0.9999999],
        },
        index=index,
    )


def test_serialize_asset_data_keys_and_types():
    periods = [{"start": "2024-01-01", "end": "2024-01-01", "max_gap": 2.88, "duration_days": 1}]
    payload = serialize_asset_data(
        _sample_merged(),
        periods,
        extra_columns=["crypto_usd", "gold_usd_oz"],  # 없는 컬럼은 생략돼야 함
        metadata={"domestic_source": "bithumb", "domestic_label": "빗썸 USDT"},
    )

    assert set(payload) == {
        "dates", "domestic_price", "intl_price", "gap_pct", "usd_krw",
        "high_gap_periods", "crypto_usd", "domestic_source", "domestic_label",
    }
    assert payload["dates"] == ["2024-01-01", "2024-01-02"]
    # 가격/괴리율/환율은 소수 2자리, 추가 시리즈는 소수 6자리 반올림
    assert payload["domestic_price"] == [1234.57, 1300.0]
    assert payload["intl_price"] == [1200.0, 1250.0]
    assert payload["gap_pct"] == [2.88, 4.0]
    assert payload["usd_krw"] == [1306.0, 1310.0]
    assert payload["crypto_usd"] == [0.123457, 1.0]
    assert payload["high_gap_periods"] is periods
    assert all(isinstance(v, float) for v in payload["domestic_price"])
    assert all(isinstance(d, str) for d in payload["dates"])


def test_format_updated_at_kst():
    """updated_at은 'YYYY-MM-DD HH:MM KST' (BUG-01: UTC 러너에서도 KST)."""
    fixed = datetime(2026, 6, 10, 12, 34, tzinfo=timezone(timedelta(hours=9)))
    assert format_updated_at(fixed) == "2026-06-10 12:34 KST"


def test_build_meta_schema_exact():
    """meta 블록 스키마(schema_version=2) — 프론트엔드 계약과 정확히 일치해야 한다."""
    meta = build_meta()

    assert set(meta) == {"schema_version", "generated_at", "assets"}
    assert meta["schema_version"] == SCHEMA_VERSION == 2
    assert isinstance(meta["schema_version"], int)

    # generated_at: ISO8601 + KST 오프셋
    generated = datetime.fromisoformat(meta["generated_at"])
    assert generated.utcoffset() == timedelta(hours=9)
    assert "+09:00" in meta["generated_at"]

    assert set(meta["assets"]) == set(ASSETS) == {"gold", "bitcoin", "eth", "usdt"}

    base_keys = {
        "label", "order", "threshold_pct", "unit", "color",
        "domestic_label", "intl_label", "summary", "source_summary",
    }
    str_keys = base_keys - {"order", "threshold_pct"}

    for key, entry in meta["assets"].items():
        expected_keys = base_keys | ({"intl_modes", "default_intl_mode"} if key == "gold" else set())
        assert set(entry) == expected_keys, f"{key} meta keys mismatch"
        assert isinstance(entry["order"], int)
        assert isinstance(entry["threshold_pct"], float)
        assert all(isinstance(entry[k], str) and entry[k] for k in str_keys)

    gold = meta["assets"]["gold"]
    assert set(gold["intl_modes"]) == {"ny_futures", "london_spot"}
    for mode in gold["intl_modes"].values():
        assert set(mode) == {"label", "intl_label", "source_summary"}
        assert all(isinstance(v, str) and v for v in mode.values())
    assert gold["default_intl_mode"] == "ny_futures"
    assert gold["default_intl_mode"] in gold["intl_modes"]


def test_build_meta_values_match_registry():
    """레지스트리 핵심 값 스팟 체크 (프론트 폴백 ASSETS와 일치하는 값들)."""
    assets = build_meta()["assets"]
    assert assets["gold"]["order"] == 1
    assert assets["bitcoin"]["order"] == 2
    assert assets["eth"]["order"] == 3
    assert assets["usdt"]["order"] == 4
    assert assets["gold"]["threshold_pct"] == 5.0
    assert assets["bitcoin"]["threshold_pct"] == 5.0
    assert assets["eth"]["threshold_pct"] == 5.0
    assert assets["usdt"]["threshold_pct"] == 3.0
    assert assets["gold"]["unit"] == "KRW/g"
    assert assets["bitcoin"]["unit"] == assets["eth"]["unit"] == assets["usdt"]["unit"] == "KRW"
    assert assets["gold"]["color"] == "#d9a441"
    assert assets["bitcoin"]["color"] == "#d9791f"
    assert assets["eth"]["color"] == "#627eea"
    assert assets["usdt"]["color"] == "#1f9b57"


def test_config_json_thresholds_match_registry(config_json):
    """외부 계약 가드: config.json thresholdPct == 자산 레지스트리 threshold_pct."""
    config_assets = config_json["assets"]
    assert set(config_assets) == set(ASSETS)
    for key, asset in ASSETS.items():
        assert config_assets[key]["thresholdPct"] == asset.threshold_pct, key
