"""전송 계층 공용 헬퍼: KST 오늘 날짜, 최신 quote upsert, 공용 헤더."""

from datetime import datetime

import pandas as pd

from goldgap.constants import KST

PUBLIC_API_HEADERS = {"Accept": "application/json"}


def today_kst():
    return pd.Timestamp(datetime.now(KST).date())


def upsert_latest_row(df, row):
    """오늘(KST) 행을 추가하거나 최신 quote로 덮어쓴다."""
    if not row:
        return df

    result = df.copy()
    idx = today_kst()
    for key, value in row.items():
        result.loc[idx, key] = value
    return result.sort_index()
