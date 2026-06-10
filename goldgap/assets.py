"""자산 레지스트리: 자산별 표시/임계값 메타데이터의 단일 진실 원천.

라벨·색상·요약 텍스트는 templates/index.html의 프론트엔드 ASSETS 폴백 객체와
일치해야 한다 (meta 블록을 읽지 못하는 구버전 프론트와의 호환 계약).
임계값(threshold_pct)은 config.json의 thresholdPct와도 일치해야 한다.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IntlMode:
    """국제 가격 기준 모드 (예: 금의 뉴욕선물/런던현물 토글)."""

    label: str
    intl_label: str
    source_summary: str


@dataclass(frozen=True)
class Asset:
    """자산 메타데이터."""

    key: str
    label: str
    order: int
    threshold_pct: float
    unit: str
    color: str
    domestic_label: str
    intl_label: str
    summary: str
    source_summary: str
    intl_modes: dict[str, IntlMode] = field(default_factory=dict)
    default_intl_mode: str | None = None


_ASSET_LIST = (
    Asset(
        key="gold",
        label="Gold",
        order=1,
        threshold_pct=5.0,
        unit="KRW/g",
        color="#d9a441",
        domestic_label="국내 (KRX 금현물)",
        intl_label="뉴욕선물 GC=F (KRW 환산)",
        summary="KRX 금현물과 COMEX 금 가격을 원화 기준으로 비교합니다.",
        source_summary="국내: ACE KRX금현물 411060.KS · 환율: USD/KRW KRW=X · 국제 기준은 토글로 선택",
        intl_modes={
            "london_spot": IntlMode(
                label="런던 현물",
                intl_label="런던 현물 XAU/USD (KRW 환산)",
                source_summary="국제: World Gold Council/ICE spot + Gold API latest XAU",
            ),
            "ny_futures": IntlMode(
                label="뉴욕선물",
                intl_label="뉴욕선물 GC=F (KRW 환산)",
                source_summary="국제: COMEX Gold Futures GC=F (Yahoo Finance)",
            ),
        },
        default_intl_mode="ny_futures",
    ),
    Asset(
        key="bitcoin",
        label="Bitcoin",
        order=2,
        threshold_pct=5.0,
        unit="KRW",
        color="#d9791f",
        domestic_label="업비트 BTC",
        intl_label="BTC-USD (KRW 환산)",
        summary="업비트 BTC와 BTC-USD 국제가의 프리미엄을 추적합니다.",
        source_summary="국내: Upbit KRW-BTC · 국제: BTC-USD (Yahoo Finance) · 환율: USD/KRW KRW=X",
    ),
    Asset(
        key="eth",
        label="Ethereum",
        order=3,
        threshold_pct=5.0,
        unit="KRW",
        color="#627eea",
        domestic_label="업비트 ETH",
        intl_label="ETH-USD (KRW 환산)",
        summary="업비트 ETH와 ETH-USD 국제가의 프리미엄을 추적합니다.",
        source_summary="국내: Upbit KRW-ETH · 국제: ETH-USD (Yahoo Finance) · 환율: USD/KRW KRW=X",
    ),
    Asset(
        key="usdt",
        label="USDT",
        order=4,
        threshold_pct=3.0,
        unit="KRW",
        color="#1f9b57",
        domestic_label="빗썸 USDT",
        intl_label="USDT-USD (KRW 환산)",
        summary="빗썸 USDT/KRW와 USDT-USD 환산 기준의 괴리율을 확인합니다.",
        source_summary="국내: Bithumb USDT_KRW · 국제: USDT-USD (Yahoo Finance) · 환율: USD/KRW KRW=X",
    ),
)

# order 순으로 정렬된 레지스트리 (key → Asset)
ASSETS: dict[str, Asset] = {asset.key: asset for asset in sorted(_ASSET_LIST, key=lambda a: a.order)}


def get_asset(key: str) -> Asset:
    """자산 키로 레지스트리 항목 조회."""
    return ASSETS[key]


def get_threshold(key: str) -> float:
    """자산별 고괴리 임계값(%) 조회 (구 THRESHOLDS dict 대체)."""
    return ASSETS[key].threshold_pct
