"""네이버 증권 ETF API: KRX 금현물 ETF(411060) CU 구성 조회."""

import logging

import requests

logger = logging.getLogger(__name__)

NAVER_ETF_BASIC_URL = "https://m.stock.naver.com/api/etf/{code}/basic"
KRX_GOLD_ETF = "411060"
KRX_GOLD_ETF_CU_SIZE = 100_000  # 1 CU = 100,000 좌


def get_gold_grams_per_unit():
    """네이버 ETF API에서 411060 CU 구성을 조회하여 1좌당 금 그램수 계산"""
    resp = requests.get(
        NAVER_ETF_BASIC_URL.format(code=KRX_GOLD_ETF),
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    for item in data.get("constituentList", []):
        if "금" in item.get("itemName", ""):
            cu_grams = item["cuUnitQuantity"]
            grams_per_unit = cu_grams / KRX_GOLD_ETF_CU_SIZE
            logger.info(f"ETF CU gold: {cu_grams}g / {KRX_GOLD_ETF_CU_SIZE} units = {grams_per_unit:.4f} g/unit")
            return grams_per_unit

    raise ValueError("Gold constituent not found in ETF CU data")
