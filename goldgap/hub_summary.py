"""허브(Value Compass)용 summary.json / version.json 발행 — envelope v1.

허브는 김치프리미엄 카드와 KRX_GOLD / CRYPTO_* 포트폴리오 신호를 만들려고 563 KB짜리
data.json 전체를 받아 왔다. 이 모듈은 자산별 최신 괴리율만 담은 ~1 KB 요약을
value-invest 데이터 계약(docs/ecosystem/data-contract.md §6.6)대로 발행한다.
envelope·canonical JSON·no-op 쓰기는 벤더링된 goldgap/vc_publish.py(직접 수정 금지)가 맡는다.

운영 워크플로우(update-data.yml)가 generate_data.py 직후 호출한다:

    python -m goldgap.hub_summary --data data.json --prev /tmp/data_prev.json \\
        --summary summary.json --version version.json

- data_changed: 직전 data.json 대비 내용이 바뀌었는지(updated_at·meta.generated_at 제외 비교).
  같으면 og 재생성·data.json 커밋·Pages 배포를 건너뛴다(30분 cron의 무의미한 커밋 방지).
- summary_changed: summary.json/version.json을 새로 썼는지(write_if_changed — 같은 내용이면 안 씀).
- changed: 둘 중 하나라도 참 → data 브랜치 커밋 대상.
결과는 stdout과 $GITHUB_OUTPUT(있을 때)에 key=value로 남긴다. summary 생성 실패는 비치명
(경고만, summary_changed=false) — data.json 갱신은 계속된다.
네트워크 접근 없음 — 입력 파일만 읽는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

from goldgap import vc_publish as vp

TOOL_ID = "gold_gap"
# 계약 순서 고정(gold, bitcoin, eth, usdt) — 허브 카드 표시 순서와 같다.
ASSET_KEYS = ("gold", "bitcoin", "eth", "usdt")
# 허브 카드 라벨(value-invest external_tools._GOLD_LABELS와 동일) — 한국어 고정.
HUB_LABELS = {"gold": "금", "bitcoin": "비트코인", "eth": "이더리움", "usdt": "USDT"}
SOURCES = (
    {"id": "naver", "name": "네이버 증권 (ACE KRX금현물 ETF)", "url": "https://m.stock.naver.com/"},
    {"id": "yahoo", "name": "Yahoo Finance", "url": "https://finance.yahoo.com/"},
    {"id": "wgc", "name": "World Gold Council / ICE", "url": "https://www.gold.org/goldhub/data/gold-prices"},
    {"id": "gold-api", "name": "Gold API", "url": "https://gold-api.com/"},
    {"id": "upbit", "name": "업비트", "url": "https://upbit.com/"},
    {"id": "bithumb", "name": "빗썸", "url": "https://www.bithumb.com/"},
)


def _num(value):
    """유한한 숫자만 통과(모르는 값은 None — 0으로 채우지 않는다)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _last(values, offset=1):
    if not isinstance(values, list) or len(values) < offset:
        return None
    return values[-offset]


def data_content_hash(data):
    """data.json 내용 해시 — 실행 시각 필드(updated_at, meta.generated_at)를 뺀다.

    NaN이 섞인 구버전 파일도 비교할 수 있도록 표준 json(allow_nan)으로 직렬화한다.
    """
    if not isinstance(data, dict):
        return None
    stripped = {k: v for k, v in data.items() if k != "updated_at"}
    meta = stripped.get("meta")
    if isinstance(meta, dict):
        stripped["meta"] = {k: v for k, v in meta.items() if k != "generated_at"}
    text = json.dumps(stripped, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_summary_data(data, *, updated_at=None):
    """data.json → summary data(§6.6): {updatedAt, goldIntlMode, assets[]}.

    괴리율 시계열이 없는 자산은 건너뛴다(허브 _summarize_gold와 같은 의미).
    """
    assets = []
    for key in ASSET_KEYS:
        payload = data.get(key) if isinstance(data, dict) else None
        if not isinstance(payload, dict):
            continue
        gaps = payload.get("gap_pct") or []
        if not gaps:
            continue
        date = _last(payload.get("dates"))
        assets.append({
            "key": key,
            "label": HUB_LABELS[key],
            "date": date if isinstance(date, str) else None,
            "gap": _num(_last(gaps)),
            "prevGap": _num(_last(gaps, 2)),
            "domesticPrice": _num(_last(payload.get("domestic_price"))),
            "intlPrice": _num(_last(payload.get("intl_price"))),
            "usdKrw": _num(_last(payload.get("usd_krw"))),
        })
    gold = data.get("gold") if isinstance(data, dict) else None
    mode = gold.get("default_intl_mode") if isinstance(gold, dict) else None
    upd = updated_at if updated_at is not None else (data.get("updated_at") if isinstance(data, dict) else None)
    return {
        "updatedAt": upd if isinstance(upd, str) else None,
        "goldIntlMode": mode if isinstance(mode, str) else None,
        "assets": assets,
    }


def summary_as_of(summary_data):
    """envelope asOf = 자산 최신 데이터 날짜의 최댓값(KST 날짜) — 실행 시각이 아니다."""
    dates = [a["date"] for a in summary_data.get("assets", []) if a.get("date")]
    return max(dates) if dates else None


def _without_updated_at(summary_data):
    return {k: v for k, v in summary_data.items() if k != "updatedAt"}


def build_envelope(data, *, prev_envelope=None, generated_at=None):
    """summary envelope 생성.

    updatedAt은 data.json의 실행 시각이라 매 실행 바뀐다. 나머지 내용이 직전 발행본과 같으면
    직전 updatedAt을 그대로 써서 contentHash를 고정한다(= write_if_changed가 쓰지 않음).
    """
    summary = build_summary_data(data)
    if not summary["assets"]:
        raise vp.EnvelopeError("no asset gaps in data.json — refusing to publish an empty summary")
    prev_data = prev_envelope.get("data") if isinstance(prev_envelope, dict) else None
    if isinstance(prev_data, dict) and _without_updated_at(prev_data) == _without_updated_at(summary):
        summary["updatedAt"] = prev_data.get("updatedAt")
    return vp.build_envelope(
        TOOL_ID,
        summary,
        as_of=summary_as_of(summary),
        sources=[dict(s) for s in SOURCES],
        generated_at=generated_at,
    )


def _read_json(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def publish(data_path, *, prev_path=None, summary_path="summary.json", version_path="version.json",
            generated_at=None):
    """summary.json/version.json을 (내용이 바뀐 경우에만) 쓰고 변경 플래그를 돌려준다."""
    data = _read_json(data_path)
    if not isinstance(data, dict):
        raise vp.EnvelopeError(f"cannot read {data_path}")
    prev = _read_json(prev_path)
    data_changed = prev is None or data_content_hash(prev) != data_content_hash(data)
    result = {"data_changed": data_changed, "summary_changed": False, "changed": data_changed,
              "content_hash": "", "as_of": "", "summary_error": ""}
    # summary 실패는 비치명 — data.json 갱신·배포는 그대로 진행하고(허브는 레거시 data.json으로
    # 폴백), 기존 summary.json은 건드리지 않는다.
    try:
        envelope = build_envelope(data, prev_envelope=_read_json(summary_path), generated_at=generated_at)
        summary_written = vp.write_if_changed(summary_path, envelope)
        version_written = vp.write_version(version_path, {"summary.json": envelope}, generated_at=generated_at)
    except vp.EnvelopeError as exc:
        result["summary_error"] = str(exc).replace("\n", " ")
        return result
    result["summary_changed"] = summary_written or version_written
    result["changed"] = data_changed or result["summary_changed"]
    result["content_hash"] = envelope["contentHash"]
    result["as_of"] = envelope["asOf"]
    return result


def _emit(result):
    lines = [f"{k}={'true' if v is True else 'false' if v is False else v}" for k, v in result.items()]
    print("\n".join(lines))
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="허브용 summary.json/version.json 발행 (envelope v1)")
    parser.add_argument("--data", default="data.json", help="새 data.json")
    parser.add_argument("--prev", default=None, help="직전 data.json (내용 비교용, 없으면 변경으로 간주)")
    parser.add_argument("--summary", default="summary.json")
    parser.add_argument("--version", default="version.json")
    parser.add_argument("--generated-at", default=None, help="테스트·재현용 고정 generatedAt (+09:00)")
    args = parser.parse_args(argv)
    try:
        result = publish(args.data, prev_path=args.prev, summary_path=args.summary,
                         version_path=args.version, generated_at=args.generated_at)
    except vp.EnvelopeError as exc:  # 새 data.json 자체를 못 읽음 — 커밋할 것이 없다
        print(f"::error::data.json 읽기 실패: {exc}", file=sys.stderr)
        return 1
    if result["summary_error"]:
        print(f"::warning::summary.json 발행 건너뜀: {result['summary_error']}", file=sys.stderr)
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
