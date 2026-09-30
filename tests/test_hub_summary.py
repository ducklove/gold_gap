"""허브용 summary.json 발행(goldgap.hub_summary) 테스트 — 빌더·envelope 검증·no-op 경로.

네트워크 불필요. 계약: value-invest docs/ecosystem/data-contract.md §6.6
(config/schemas/summary/gold_gap.schema.json의 필수 키를 여기서 직접 확인한다 — CI엔 허브 체크아웃이 없다).
"""

import json
import os

import pytest

from goldgap import hub_summary as hs
from goldgap import vc_publish as vp

GEN = "2026-09-30T09:00:00+09:00"


def _asset(dates, gaps, dom, intl, fx, **extra):
    payload = {
        "dates": dates, "gap_pct": gaps, "domestic_price": dom, "intl_price": intl, "usd_krw": fx,
        "high_gap_periods": [],
    }
    payload.update(extra)
    return payload


def _sample_data(updated_at="2026-09-30 08:55 KST", btc_gap=0.51):
    return {
        "gold": _asset(["2026-09-29", "2026-09-30"], [-1.43, -0.85], [181000.0, 181518.62],
                       [183000.0, 183068.22], [1350.0, 1351.61], default_intl_mode="ny_futures"),
        "bitcoin": _asset(["2026-09-29", "2026-09-30"], [-0.08, btc_gap], [113000000, 113590000],
                          [113100000.0, 113013369.48], [1350.0, 1351.61]),
        "eth": _asset(["2026-09-30"], [0.56], [3638000], [3617854.38], [1351.61]),
        "usdt": _asset([], [], [], [], []),  # 시계열 없음 → 요약에서 제외
        "market": {"dates": ["2026-09-30"], "kospi": [3100.0]},
        "updated_at": updated_at,
        "meta": {"schema_version": 2, "generated_at": "2026-09-30T08:55:12+09:00", "assets": {}},
    }


# ---- 빌더 -----------------------------------------------------------------

def test_build_summary_data_shape_and_order():
    summary = hs.build_summary_data(_sample_data())
    assert summary["updatedAt"] == "2026-09-30 08:55 KST"
    assert summary["goldIntlMode"] == "ny_futures"
    assert [a["key"] for a in summary["assets"]] == ["gold", "bitcoin", "eth"]
    gold = summary["assets"][0]
    assert gold == {
        "key": "gold", "label": "금", "date": "2026-09-30", "gap": -0.85, "prevGap": -1.43,
        "domesticPrice": 181518.62, "intlPrice": 183068.22, "usdKrw": 1351.61,
    }
    eth = summary["assets"][2]
    assert eth["prevGap"] is None  # 한 점뿐 — 모르는 값은 null(0 아님)


def test_build_summary_data_unknown_numbers_are_null():
    data = _sample_data()
    data["bitcoin"]["gap_pct"][-1] = float("nan")
    data["bitcoin"]["usd_krw"][-1] = None
    data["bitcoin"]["intl_price"] = []
    btc = hs.build_summary_data(data)["assets"][1]
    assert btc["gap"] is None and btc["usdKrw"] is None and btc["intlPrice"] is None


def test_summary_satisfies_hub_schema_required_keys():
    """gold_gap.schema.json: required [updatedAt, assets]; assets[] required [key,label,date,gap], ≤4."""
    summary = hs.build_summary_data(_sample_data())
    assert {"updatedAt", "assets"} <= set(summary)
    assert len(summary["assets"]) <= 4
    for asset in summary["assets"]:
        assert {"key", "label", "date", "gap"} <= set(asset)
        assert asset["key"] in {"gold", "bitcoin", "eth", "usdt"}
        assert asset["label"]
        for field in ("gap", "prevGap", "domesticPrice", "intlPrice", "usdKrw"):
            assert asset[field] is None or isinstance(asset[field], (int, float))


def test_envelope_validates_and_as_of_is_data_date():
    env = hs.build_envelope(_sample_data(), generated_at=GEN)
    assert vp.validate_envelope(env) is env
    assert env["tool"] == "gold_gap" and env["kind"] == "summary"
    assert env["asOf"] == "2026-09-30"  # 데이터 날짜 — 실행 시각(updated_at)이 아니다
    assert env["generatedAt"] == GEN
    assert env["contentHash"] == vp.content_hash(env["data"])
    assert [s["id"] for s in env["sources"]] == ["naver", "yahoo", "wgc", "gold-api", "upbit", "bithumb"]
    assert len(vp.dumps_compact(env).encode("utf-8")) < 16 * 1024  # 권장 크기 예산


def test_envelope_refuses_empty_summary():
    with pytest.raises(vp.EnvelopeError):
        hs.build_envelope({"updated_at": "x"}, generated_at=GEN)


def test_updated_at_reused_when_only_timestamp_changes():
    prev = hs.build_envelope(_sample_data("2026-09-30 08:55 KST"), generated_at=GEN)
    same = hs.build_envelope(_sample_data("2026-09-30 09:25 KST"), prev_envelope=prev, generated_at=GEN)
    assert same["data"]["updatedAt"] == "2026-09-30 08:55 KST"
    assert same["contentHash"] == prev["contentHash"]
    moved = hs.build_envelope(_sample_data("2026-09-30 09:25 KST", btc_gap=0.9), prev_envelope=prev,
                              generated_at=GEN)
    assert moved["data"]["updatedAt"] == "2026-09-30 09:25 KST"
    assert moved["contentHash"] != prev["contentHash"]


def test_data_content_hash_ignores_timestamps_only():
    base = hs.data_content_hash(_sample_data("2026-09-30 08:55 KST"))
    later = _sample_data("2026-09-30 09:25 KST")
    later["meta"]["generated_at"] = "2026-09-30T09:25:40+09:00"
    assert hs.data_content_hash(later) == base
    assert hs.data_content_hash(_sample_data(btc_gap=0.9)) != base
    assert hs.data_content_hash(None) is None


# ---- publish (파일 쓰기 · no-op) ----------------------------------------------

def _write(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)


def test_publish_first_run_writes_valid_files(tmp_path):
    data = tmp_path / "data.json"
    _write(data, _sample_data())
    summary, version = tmp_path / "summary.json", tmp_path / "version.json"
    result = hs.publish(str(data), prev_path=str(tmp_path / "missing.json"), summary_path=str(summary),
                        version_path=str(version), generated_at=GEN)
    assert result["data_changed"] is True and result["summary_changed"] is True and result["changed"] is True
    text = summary.read_text(encoding="utf-8")
    assert text.endswith("\n") and text.count("\n") == 1  # canonical 한 줄 + 개행
    env = vp.validate_envelope(json.loads(text))
    assert result["content_hash"] == env["contentHash"]
    ver = json.loads(version.read_text(encoding="utf-8"))
    assert ver["tool"] == "gold_gap" and ver["files"] == {"summary.json": env["contentHash"]}


def test_publish_unchanged_content_does_not_rewrite(tmp_path):
    prev = tmp_path / "prev.json"
    data = tmp_path / "data.json"
    _write(prev, _sample_data("2026-09-30 08:55 KST"))
    summary, version = tmp_path / "summary.json", tmp_path / "version.json"
    _write(data, _sample_data("2026-09-30 08:55 KST"))
    hs.publish(str(data), prev_path=str(prev), summary_path=str(summary), version_path=str(version),
               generated_at=GEN)
    before = (summary.read_bytes(), version.read_bytes())
    mtimes = (os.stat(summary).st_mtime_ns, os.stat(version).st_mtime_ns)

    # 다음 실행: 타임스탬프만 바뀜 → 어떤 파일도 다시 쓰지 않고 커밋/배포 생략 신호
    rerun = _sample_data("2026-09-30 09:25 KST")
    rerun["meta"]["generated_at"] = "2026-09-30T09:25:40+09:00"
    _write(data, rerun)
    result = hs.publish(str(data), prev_path=str(prev), summary_path=str(summary),
                        version_path=str(version), generated_at="2026-09-30T09:25:41+09:00")
    assert result == {**result, "data_changed": False, "summary_changed": False, "changed": False}
    assert (summary.read_bytes(), version.read_bytes()) == before
    assert (os.stat(summary).st_mtime_ns, os.stat(version).st_mtime_ns) == mtimes


def test_publish_changed_content_rewrites(tmp_path):
    prev = tmp_path / "prev.json"
    data = tmp_path / "data.json"
    summary, version = tmp_path / "summary.json", tmp_path / "version.json"
    _write(prev, _sample_data())
    _write(data, _sample_data())
    first = hs.publish(str(data), prev_path=str(prev), summary_path=str(summary), version_path=str(version),
                       generated_at=GEN)
    _write(data, _sample_data("2026-09-30 09:25 KST", btc_gap=0.9))
    second = hs.publish(str(data), prev_path=str(prev), summary_path=str(summary),
                        version_path=str(version), generated_at=GEN)
    assert second["data_changed"] and second["summary_changed"] and second["changed"]
    assert second["content_hash"] != first["content_hash"]
    assert json.loads(version.read_text(encoding="utf-8"))["files"]["summary.json"] == second["content_hash"]


def test_publish_summary_failure_is_non_fatal(tmp_path):
    """자산 괴리율이 하나도 없으면 summary는 건너뛰되(기존 파일 유지) data 변경 판정은 그대로."""
    data = tmp_path / "data.json"
    _write(data, {"updated_at": "2026-09-30 09:25 KST", "gold": {"dates": [], "gap_pct": []}})
    summary = tmp_path / "summary.json"
    summary.write_text("keep\n", encoding="utf-8")
    result = hs.publish(str(data), summary_path=str(summary), version_path=str(tmp_path / "version.json"))
    assert result["data_changed"] is True and result["summary_changed"] is False
    assert result["changed"] is True and result["summary_error"]
    assert summary.read_text(encoding="utf-8") == "keep\n"


def test_cli_writes_github_output(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data.json"
    _write(data, _sample_data())
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    code = hs.main(["--data", str(data), "--prev", str(data), "--summary", str(tmp_path / "summary.json"),
                    "--version", str(tmp_path / "version.json"), "--generated-at", GEN])
    assert code == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert "data_changed=false" in lines and "summary_changed=true" in lines and "changed=true" in lines
    assert "data_changed=false" in capsys.readouterr().out


def test_cli_unreadable_data_fails(tmp_path):
    assert hs.main(["--data", str(tmp_path / "nope.json"), "--summary", str(tmp_path / "s.json"),
                    "--version", str(tmp_path / "v.json")]) == 1


# ---- 골든 data.json (data 브랜치 — 없으면 skip) ----------------------------------

def test_golden_data_summary(golden_data):
    env = hs.build_envelope(golden_data, generated_at=GEN)
    vp.validate_envelope(env)
    keys = [a["key"] for a in env["data"]["assets"]]
    assert keys == [k for k in hs.ASSET_KEYS if golden_data.get(k, {}).get("gap_pct")]
    for asset in env["data"]["assets"]:
        src = golden_data[asset["key"]]
        assert asset["gap"] == src["gap_pct"][-1]
        assert asset["date"] == src["dates"][-1]
    assert env["asOf"] == max(a["date"] for a in env["data"]["assets"])
    assert env["data"]["updatedAt"] == golden_data["updated_at"]
