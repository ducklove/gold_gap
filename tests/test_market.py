"""market 블록 테스트: 직렬화(null 보존)·병합(null 비파괴)·fetch_fresh 격리.

market은 자산 페이로드와 계약이 다르다 — KOSPI(한국)·S&P500(미국)은 휴장일이
서로 달라 '필수 필드 없는 행 제외' 대신 합집합 날짜 + null 결측이어야
데이터 손실이 없다. 네트워크 불필요: fetch는 monkeypatch로 대체한다.
"""

import json
from datetime import datetime, timedelta

import pandas as pd
import pytest

from goldgap.domain.merge import (
    MARKET_SERIES_KEYS,
    compute_incremental_start_dates,
    merge_market_data,
)
from goldgap.orchestrators import fetch_fresh, get_market_data
from goldgap.serialize import serialize_market_data


def _df(dates, **cols):
    return pd.DataFrame(cols, index=pd.DatetimeIndex(pd.to_datetime(dates)))


def make_market(dates, **overrides):
    """테스트용 market 페이로드 생성 헬퍼."""
    n = len(dates)
    payload = {
        "dates": list(dates),
        "kospi": [2600.0 + i for i in range(n)],
        "sp500": [5000.0 + i for i in range(n)],
        "usd_krw": [1300.0 + i for i in range(n)],
    }
    payload.update(overrides)
    return payload


def make_asset(dates, base=100.0):
    """테스트용 자산 페이로드 생성 헬퍼 (merge_asset_data 필수 키 포함)."""
    n = len(dates)
    return {
        "dates": list(dates),
        "domestic_price": [base + i for i in range(n)],
        "intl_price": [base for _ in range(n)],
        "gap_pct": [round(i / base * 100, 2) for i in range(n)],
        "usd_krw": [1300.0 + i for i in range(n)],
    }


# ---------------------------------------------------------------------------
# serialize_market_data
# ---------------------------------------------------------------------------


def test_serialize_market_rounds_and_preserves_null():
    df = _df(
        ["2024-01-02", "2024-01-03"],
        kospi=[2655.275, float("nan")],
        sp500=[float("nan"), 4763.456],
        usd_krw=[1300.004, 1301.5],
    )
    payload = serialize_market_data(df)
    assert set(payload) == {"dates", "kospi", "sp500", "usd_krw"}
    assert payload["dates"] == ["2024-01-02", "2024-01-03"]
    assert payload["kospi"] == [2655.28, None]  # NaN → null, 소수 2자리
    assert payload["sp500"] == [None, 4763.46]
    assert payload["usd_krw"] == [1300.0, 1301.5]
    # JSON 직렬화 가능 (numpy 타입 누출 없음)
    assert json.loads(json.dumps(payload)) == payload


def test_serialize_market_sorts_union_dates():
    df = _df(
        ["2024-01-05", "2024-01-02"],
        kospi=[2.0, 1.0],
        sp500=[20.0, 10.0],
        usd_krw=[1301.0, 1300.0],
    )
    payload = serialize_market_data(df)
    assert payload["dates"] == ["2024-01-02", "2024-01-05"]
    assert payload["kospi"] == [1.0, 2.0]  # 정렬 후에도 날짜-값 짝 유지
    assert payload["usd_krw"] == [1300.0, 1301.0]


# ---------------------------------------------------------------------------
# merge_market_data
# ---------------------------------------------------------------------------


def test_merge_market_empty_old_returns_new():
    new = make_market(["2024-01-01"])
    assert merge_market_data(None, new) is new
    assert merge_market_data({}, new) is new


def test_merge_market_empty_new_returns_old():
    old = make_market(["2024-01-01"])
    assert merge_market_data(old, None) is old
    assert merge_market_data(old, {}) is old


def test_merge_market_union_sorted():
    """비겹침 입력은 순서와 무관하게 날짜 오름차순 합집합이 된다."""
    old = make_market(["2024-01-05", "2024-01-06"])
    new = make_market(["2024-01-02", "2024-01-01"])
    merged = merge_market_data(old, new)
    assert merged["dates"] == ["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-06"]
    # 2024-01-01은 new의 두 번째 항목 (kospi 2601.0)
    assert merged["kospi"] == [2601.0, 2600.0, 2600.0, 2601.0]


def test_merge_market_overlap_new_wins():
    old = make_market(["2024-01-01", "2024-01-02"])
    new = make_market(["2024-01-02", "2024-01-03"], kospi=[2700.0, 2710.0])
    merged = merge_market_data(old, new)
    assert merged["dates"] == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert merged["kospi"] == [2600.0, 2700.0, 2710.0]
    assert merged["sp500"] == [5000.0, 5000.0, 5001.0]


def test_merge_market_null_does_not_overwrite_existing_value():
    """새 값이 null이면 기존 관측값을 유지한다 (휴장일/부분 실패 보호)."""
    old = make_market(["2024-01-02"], kospi=[2650.0], sp500=[5100.0], usd_krw=[1310.0])
    new = make_market(["2024-01-02"], kospi=[None], sp500=[5200.0], usd_krw=[None])
    merged = merge_market_data(old, new)
    assert merged["kospi"] == [2650.0]  # null이 기존 값을 덮지 않음
    assert merged["sp500"] == [5200.0]  # 새 값 우선
    assert merged["usd_krw"] == [1310.0]


def test_merge_market_new_value_fills_old_null():
    old = make_market(["2024-01-02"], kospi=[None])
    new = make_market(["2024-01-02"], kospi=[2660.0])
    merged = merge_market_data(old, new)
    assert merged["kospi"] == [2660.0]


def test_merge_market_both_null_stays_null():
    old = make_market(["2024-01-02"], sp500=[None])
    new = make_market(["2024-01-02"], sp500=[None])
    merged = merge_market_data(old, new)
    assert merged["sp500"] == [None]


def test_merge_market_missing_dates_become_null():
    """한쪽에만 있는 날짜의 나머지 시리즈는 null로 채워진다 (행 제외 없음)."""
    old = make_market(["2024-01-01"])
    new = {
        "dates": ["2024-01-02"],
        "kospi": [2700.0],
        "sp500": [None],  # 미국 휴장
        "usd_krw": [1305.0],
    }
    merged = merge_market_data(old, new)
    assert merged["dates"] == ["2024-01-01", "2024-01-02"]
    assert merged["sp500"] == [5000.0, None]
    assert all(len(merged[key]) == 2 for key in MARKET_SERIES_KEYS)


def test_merge_market_sources_new_first():
    old = make_market(["2024-01-01"], sources={"kospi": "old", "sp500": "old", "fx": "old"})
    new = make_market(["2024-01-02"])
    assert merge_market_data(old, new)["sources"]["kospi"] == "old"  # 새 데이터에 없으면 유지
    new["sources"] = {"kospi": "new", "sp500": "new", "fx": "new"}
    assert merge_market_data(old, new)["sources"]["kospi"] == "new"


def test_merge_market_no_periods_key():
    merged = merge_market_data(make_market(["2024-01-01"]), make_market(["2024-01-02"]))
    assert "high_gap_periods" not in merged
    assert set(merged) == {"dates", *MARKET_SERIES_KEYS}


# ---------------------------------------------------------------------------
# compute_incremental_start_dates — market 확장
# ---------------------------------------------------------------------------


def test_incremental_start_dates_include_market():
    existing = {"market": make_market(["2026-06-01", "2026-06-08"])}
    starts = compute_incremental_start_dates(existing)
    assert starts == {"market": datetime(2026, 6, 8) - timedelta(days=7)}
    assert compute_incremental_start_dates({"market": {"dates": []}}) == {}


# ---------------------------------------------------------------------------
# get_market_data — outer join 합집합 (fetch는 monkeypatch)
# ---------------------------------------------------------------------------


def _patch_index_fetch(monkeypatch, kospi_df, sp500_df):
    def fake_fetch(ticker, column, start_date=None):
        return {"^KS11": kospi_df, "^GSPC": sp500_df}[ticker]

    monkeypatch.setattr("goldgap.orchestrators.fetch_index_series", fake_fetch)


def test_get_market_data_union_with_nulls(monkeypatch):
    """한국/미국 휴장일이 달라도 합집합 날짜 + null 결측으로 손실이 없다."""
    kospi = _df(["2024-01-02", "2024-01-03"], kospi=[2655.0, 2660.123])  # 1/4 한국 휴장 가정
    sp500 = _df(["2024-01-02", "2024-01-04"], sp500=[4760.0, 4770.456])  # 1/3 미국 휴장 가정
    fx = _df(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        usd_krw=[1299.0, 1300.0, 1301.0, 1302.0],
    )
    _patch_index_fetch(monkeypatch, kospi, sp500)

    data = get_market_data(fx)
    assert data["dates"] == ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    assert data["kospi"] == [None, 2655.0, 2660.12, None]
    assert data["sp500"] == [None, 4760.0, None, 4770.46]
    assert data["usd_krw"] == [1299.0, 1300.0, 1301.0, 1302.0]
    assert set(data["sources"]) == {"kospi", "sp500", "fx"}


def test_get_market_data_slices_fx_to_start_date(monkeypatch):
    """fx_df가 더 긴 범위여도 증분 시작일 이전 fx-only 날짜는 제외된다."""
    kospi = _df(["2024-01-03"], kospi=[2660.0])
    sp500 = _df(["2024-01-03"], sp500=[4765.0])
    fx = _df(["2024-01-01", "2024-01-03"], usd_krw=[1299.0, 1301.0])
    _patch_index_fetch(monkeypatch, kospi, sp500)

    data = get_market_data(fx, datetime(2024, 1, 2))
    assert data["dates"] == ["2024-01-03"]
    assert data["usd_krw"] == [1301.0]


# ---------------------------------------------------------------------------
# fetch_fresh — market 격리/증분/전체 fetch 경로
# ---------------------------------------------------------------------------

ASSET_FETCHERS = ["get_gold_data", "get_bitcoin_data", "get_eth_data", "get_usdt_data"]


def _existing_with_market(dates=("2026-06-01",)):
    existing = {key: make_asset(dates) for key in ["gold", "bitcoin", "eth", "usdt"]}
    existing["market"] = make_market(dates)
    return existing


def _patch_fetch_fresh_deps(monkeypatch, asset_fetcher, market_fetcher, fx_calls=None):
    def fake_fx(start_date=None):
        if fx_calls is not None:
            fx_calls.append(start_date)
        return f"FX:{start_date}"

    monkeypatch.setattr("goldgap.orchestrators.fetch_exchange_rate", fake_fx)
    for name in ASSET_FETCHERS:
        monkeypatch.setattr(f"goldgap.orchestrators.{name}", asset_fetcher)
    monkeypatch.setattr("goldgap.orchestrators.get_market_data", market_fetcher)


def test_fetch_fresh_market_failure_keeps_existing(monkeypatch):
    """market fetch 실패는 격리된다 — 자산은 갱신되고 기존 market은 유지."""
    existing = _existing_with_market()

    def boom(fx_df, start_date=None):
        raise ValueError("index source down")

    _patch_fetch_fresh_deps(monkeypatch, lambda fx, start=None: make_asset(["2026-06-02"]), boom)

    data = fetch_fresh(existing)
    assert data["market"] == existing["market"]
    assert data["gold"]["dates"] == ["2026-06-01", "2026-06-02"]


def test_fetch_fresh_market_incremental_merge(monkeypatch):
    """기존 market이 있으면 마지막 날짜 - 7일부터 fetch하고 병합한다."""
    existing = _existing_with_market(["2026-06-01"])
    captured = {}

    def fake_market(fx_df, start_date=None):
        captured["start"] = start_date
        return make_market(["2026-06-02"], kospi=[2777.0])

    _patch_fetch_fresh_deps(monkeypatch, lambda fx, start=None: make_asset(["2026-06-02"]), fake_market)

    data = fetch_fresh(existing)
    assert captured["start"] == datetime(2026, 6, 1) - timedelta(days=7)
    assert data["market"]["dates"] == ["2026-06-01", "2026-06-02"]
    assert data["market"]["kospi"] == [2600.0, 2777.0]


def test_fetch_fresh_market_missing_triggers_full_fetch_with_full_fx(monkeypatch):
    """기존에 market이 없으면 전체 fetch + market용 전체 환율을 따로 받는다."""
    existing = {key: make_asset(["2026-06-01"]) for key in ["gold", "bitcoin", "eth", "usdt"]}
    fx_calls = []
    captured = {}

    def fake_market(fx_df, start_date=None):
        captured["fx"] = fx_df
        captured["start"] = start_date
        return make_market(["2026-06-02"])

    _patch_fetch_fresh_deps(
        monkeypatch, lambda fx, start=None: make_asset(["2026-06-02"]), fake_market, fx_calls
    )

    data = fetch_fresh(existing)
    assert captured["start"] is None  # 전체 fetch
    assert fx_calls[-1] is None and len(fx_calls) == 2  # 두 번째 호출이 market용 전체 환율
    assert captured["fx"] == "FX:None"
    assert data["market"] == make_market(["2026-06-02"])


def test_fetch_fresh_market_alone_does_not_satisfy_assets(monkeypatch):
    """자산이 전부 실패하면 market만 성공해도 RuntimeError (자산 없는 출력 방지)."""

    def asset_boom(fx_df, start_date=None):
        raise ValueError("asset source down")

    _patch_fetch_fresh_deps(monkeypatch, asset_boom, lambda fx, start=None: make_market(["2026-06-02"]))

    with pytest.raises(RuntimeError, match="All assets failed"):
        fetch_fresh(None)


def test_fetch_fresh_new_asset_gets_full_fx_history(monkeypatch):
    """증분 런에서 신규 자산(기존 데이터 없음)은 전체 환율로 fetch한다.

    ETH 최초 수집 때 증분 fx_df가 ffill/bfill로 과거 5년 환율을 최근 값
    상수로 백필해 괴리율 역사가 오염된 사고의 회귀 방지. 전체 환율은
    전체 fetch가 필요한 블록들(신규 자산·market)이 캐시로 공유해 1회만 받는다.
    """
    # eth와 market이 없는 기존 데이터 → 둘 다 전체 fetch 대상
    existing = {key: make_asset(["2026-06-01"]) for key in ["gold", "bitcoin", "usdt"]}
    fx_calls = []
    received = {"assets": []}

    def recording_fetcher(fx_df, start_date=None):
        received["assets"].append((fx_df, start_date))
        return make_asset(["2026-06-02"])

    def fake_market(fx_df, start_date=None):
        received["market"] = (fx_df, start_date)
        return make_market(["2026-06-02"])

    _patch_fetch_fresh_deps(monkeypatch, recording_fetcher, fake_market, fx_calls)
    data = fetch_fresh(existing)

    calls = received["assets"]  # gold, bitcoin, eth, usdt 순
    incremental_fx = calls[0][0]
    assert incremental_fx.startswith("FX:") and incremental_fx != "FX:None"
    assert calls[0][1] is not None  # gold 증분
    assert calls[2] == ("FX:None", None)  # eth: 전체 환율 + 전체 fetch
    assert calls[3][0] == incremental_fx  # usdt: 증분 환율 그대로
    assert received["market"] == ("FX:None", None)  # market도 전체 환율 공유
    # 환율 fetch는 [증분 1회, 전체 1회] — 전체는 eth·market이 캐시 공유
    assert len(fx_calls) == 2 and fx_calls[0] is not None and fx_calls[1] is None
    assert "eth" in data and "market" in data
