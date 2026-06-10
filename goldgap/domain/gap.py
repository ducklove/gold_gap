"""괴리율 계산: 환율 환산 및 국내/국제 가격 병합 (순수 DataFrame 연산)."""

from goldgap.constants import TROY_OZ_TO_GRAM


def attach_fx_and_convert(gold_df, fx_df, usd_col, price_col):
    """USD/oz 시계열에 환율을 붙여 KRW/g 가격 컬럼을 만든다."""
    fx_filled = fx_df.reindex(gold_df.index).ffill().bfill()
    intl = gold_df.join(fx_filled, how="left").dropna(subset=[usd_col, "usd_krw"])
    intl[price_col] = (intl[usd_col] * intl["usd_krw"]) / TROY_OZ_TO_GRAM
    return intl


def calculate_gap(intl_df, domestic_df):
    """국내/국제 가격 병합 및 괴리율 계산 (컬럼명: intl_price, domestic_price)"""
    merged = intl_df.join(domestic_df, how="inner")
    merged = merged.dropna(subset=["intl_price", "domestic_price"])

    if merged.empty:
        raise ValueError("No overlapping dates between international and domestic data")

    merged["gap_pct"] = (
        (merged["domestic_price"] - merged["intl_price"])
        / merged["intl_price"]
        * 100
    )

    return merged
