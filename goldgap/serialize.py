"""직렬화 계층: DataFrame → JSON 페이로드, meta 블록, updated_at 포맷.

meta 블록 스키마(schema_version=2)는 프론트엔드 팀과의 계약이다 — 키 이름/타입을
임의로 바꾸지 말 것. 기존 최상위 키(gold/bitcoin/usdt/updated_at)와 자산 내부
키들의 이름·타입·형식도 불변 계약이다 (배포된 구버전 프론트가 그대로 읽는다).
"""

from datetime import datetime

import pandas as pd

from goldgap.assets import ASSETS
from goldgap.constants import KST
from goldgap.domain.merge import MARKET_SERIES_KEYS

SCHEMA_VERSION = 2

UPDATED_AT_FORMAT = "%Y-%m-%d %H:%M KST"


def serialize_asset_data(merged, periods, extra_columns=None, metadata=None):
    """DataFrame → dict 변환 헬퍼"""
    payload = {
        "dates": [d.strftime("%Y-%m-%d") for d in merged.index],
        "domestic_price": [round(float(v), 2) for v in merged["domestic_price"]],
        "intl_price": [round(float(v), 2) for v in merged["intl_price"]],
        "gap_pct": [round(float(v), 2) for v in merged["gap_pct"]],
        "usd_krw": [round(float(v), 2) for v in merged["usd_krw"]],
        "high_gap_periods": periods,
    }
    for col in extra_columns or []:
        if col in merged:
            payload[col] = [round(float(v), 6) for v in merged[col]]
    if metadata:
        payload.update(metadata)
    return payload


def serialize_market_data(market_df):
    """market 블록 직렬화: 합집합 날짜 + 결측 null (자산 페이로드와 다른 규칙).

    KOSPI(한국)와 S&P500(미국)은 휴장일이 서로 달라, 자산식 '필수 필드
    없는 행 제외'를 쓰면 데이터가 손실된다 — 날짜는 모든 시리즈의 합집합을
    오름차순으로 유지하고 결측은 null로 둔다. 값은 소수 2자리 반올림.
    """
    df = market_df.sort_index()
    payload = {"dates": [d.strftime("%Y-%m-%d") for d in df.index]}
    for col in MARKET_SERIES_KEYS:
        payload[col] = [None if pd.isna(v) else round(float(v), 2) for v in df[col]]
    return payload


def format_updated_at(now=None):
    """KST 기준 'YYYY-MM-DD HH:MM KST' 형식 갱신 시각 (BUG-01)."""
    now = now or datetime.now(KST)
    return now.strftime(UPDATED_AT_FORMAT)


def build_meta(now=None):
    """data.json의 meta 블록 생성 — 자산 레지스트리를 그대로 직렬화한다."""
    now = now or datetime.now(KST)
    assets_meta = {}
    for asset in ASSETS.values():
        entry = {
            "label": asset.label,
            "order": asset.order,
            "threshold_pct": asset.threshold_pct,
            "unit": asset.unit,
            "color": asset.color,
            "domestic_label": asset.domestic_label,
            "intl_label": asset.intl_label,
            "summary": asset.summary,
            "source_summary": asset.source_summary,
        }
        if asset.intl_modes:
            entry["intl_modes"] = {
                mode_key: {
                    "label": mode.label,
                    "intl_label": mode.intl_label,
                    "source_summary": mode.source_summary,
                }
                for mode_key, mode in asset.intl_modes.items()
            }
            entry["default_intl_mode"] = asset.default_intl_mode
        assets_meta[asset.key] = entry

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "assets": assets_meta,
    }
