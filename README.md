# gold_gap — 김치프리미엄 대시보드

국내 거래가와 환산 국제가의 괴리율(김치프리미엄)을 추적하는 정적 대시보드입니다.

**라이브**: https://ducklove.github.io/gold_gap

| 자산 | 국내 기준 | 국제 기준 | 괴리율 임계치 |
|---|---|---|---|
| Gold | KRX 금현물 (ACE KRX금현물 ETF 411060 → 원/g 환산) | COMEX 선물 GC=F 또는 WGC/ICE 런던 현물 (토글) | ±5% |
| Bitcoin | 업비트 KRW-BTC | BTC-USD × USD/KRW | ±5% |
| Ethereum | 업비트 KRW-ETH | ETH-USD × USD/KRW | ±5% |
| USDT | 빗썸 KRW-USDT (실패 시 업비트 백업) | USDT-USD × USD/KRW | ±3% |

자산 추가는 `goldgap/assets.py` 레지스트리 등록 + 오케스트레이터 함수 하나로 끝난다 —
프론트엔드 탭·라벨·임계치는 data.json의 meta 블록에서 동적으로 구성된다.

## 아키텍처

서버 없는 **정적 우선(static-first)** 구조입니다.

```
외부 API (yfinance · WGC/ICE · gold-api · 네이버 ETF · Upbit · Bithumb)
        │
        ▼
goldgap/ 패키지 ──▶ generate_data.py ──▶ data.json
  (수집·계산·병합)      (30분 cron, 증분 갱신)      │
                                                  ▼
                                    GitHub Pages (templates/ + static/ + data.json)
```

- **데이터 갱신**: `update-data.yml`이 30분마다 `generate_data.py`를 실행해 기존 data.json의 마지막 날짜 −7일부터 증분 수집·병합 후 **`data` 전용 브랜치**에 커밋하고, 변경이 있으면 Pages 배포를 트리거합니다. master 히스토리는 사람의 코드 변경만 담습니다.
- **장애 격리**: 자산 단위 수집 실패 → 해당 자산만 기존 데이터 유지. 전체 실패 → 기존 data.json 폴백(사이트는 계속 동작). 워크플로우 실패 시 `data-update-failure` 라벨로 이슈가 자동 생성됩니다.
- **임계 돌파 알림**: 갱신 시 직전 데이터 대비 괴리율이 임계치에 **신규 진입**한 자산이 있으면 `kimchi-alert` 라벨 이슈로 알립니다 (이미 임계 이상이던 자산은 재알림하지 않음). 웹훅 연동은 아래 [알림 웹훅](#알림-웹훅-선택) 참고.
- **OG 공유 이미지**: 갱신 시 최신 괴리율 카드(`/og.png`)를 자동 생성해 SNS 공유 미리보기로 사용합니다 (`goldgap/og_image.py`).
- **브라우저 새로고침**: 정적 호스팅에서는 백엔드가 없으므로, 새로고침 버튼은 거래소·환율 API를 브라우저에서 직접 호출해 오늘 포인트를 합성합니다(아래 [데이터 주의사항](#데이터-주의사항) 참고).

## 디렉터리 구조

```
goldgap/            # Python 패키지
├── assets.py       #   자산 레지스트리 — 라벨·임계치·색상의 단일 진실 원천
├── sources/        #   전송 계층 (yfinance·Upbit·Bithumb·WGC·네이버 ETF)
├── domain/         #   순수 계산 (괴리율, 고괴리 구간, 증분 병합) — 네트워크 의존 없음
├── serialize.py    #   직렬화 + data.json meta 블록 생성
└── ...
generate_data.py    # 데이터 생성 CLI (cron 진입점)
app.py              # 로컬 미리보기 전용 Flask (배포는 정적)
templates/index.html  # 대시보드 마크업
static/js/          # 프론트엔드 ES 모듈
static/style.css    # 테마(라이트/다크) 스타일
tests/              # pytest (구간 탐지·병합·직렬화 회귀 테스트)
docs/project-review.html  # 구조·품질 리뷰 및 로드맵 문서
```

## 로컬 개발

```bash
pip install -r requirements-dev.txt

# 0) data.json 받아오기 — data 전용 브랜치에 산다 (미리보기·골든 테스트에 필요)
git fetch --depth=1 origin data && git show FETCH_HEAD:data.json > data.json

# 1) 대시보드 미리보기
python app.py            # http://localhost:5000

# 2) 데이터 직접 생성/갱신 (외부 API 접근 필요)
python generate_data.py  # data.json 증분 갱신

# 3) 테스트·린트 (data.json 없으면 골든 회귀 테스트는 안내와 함께 skip)
pytest -q
ruff check .
```

`app.py`는 개발 편의용입니다. `/data.json`·`/config.json`·`/sw.js`·`/manifest.webmanifest`는 리포 루트 파일을 그대로 서빙하고, `/api/data`는 24시간 파일 캐시를 사용한 실시간 수집 경로입니다.

### PWA

배포 사이트는 홈 화면 설치형 PWA입니다 — `manifest.webmanifest` + `sw.js`(서비스 워커).
캐시 전략: `data.json`·`og.png`는 network-first(오프라인 시 마지막 캐시 폴백),
정적 자산·CDN은 stale-while-revalidate. 서비스 워커를 수정하면 `sw.js`의
`CACHE` 버전 문자열을 올려 구캐시를 무효화하세요.

## data.json 스키마

최상위: `meta`, `updated_at`(`"YYYY-MM-DD HH:MM KST"`), 자산 키(`gold`·`bitcoin`·`eth`·`usdt`), `market`(시장 지표).

각 자산 객체(레거시 호환 — 구버전 프론트가 그대로 읽을 수 있도록 유지):

```jsonc
{
  "dates": ["YYYY-MM-DD", ...],
  "domestic_price": [...], "intl_price": [...],   // KRW (금은 KRW/g)
  "gap_pct": [...], "usd_krw": [...],
  "crypto_usd": [...] | "gold_usd_oz": [...],      // 자산별 보조 시리즈
  "high_gap_periods": [{"start", "end", "max_gap", "duration_days"}],
  "intl_modes": { "ny_futures": {...}, "london_spot": {...} },  // 금만
  "default_intl_mode": "ny_futures",                              // 금만
  "sources": { ... }
}
```

`meta` 블록(schema_version 2)은 자산 레지스트리에서 생성되며, 프론트엔드가 탭·라벨·임계치·색상을 동적으로 구성하는 데 사용합니다. **meta가 없는 구버전 data.json도 프론트엔드 내장 폴백으로 동일하게 렌더링됩니다.**

```jsonc
"meta": {
  "schema_version": 2,
  "generated_at": "<ISO8601 KST>",
  "assets": {
    "<key>": {
      "label", "order", "threshold_pct", "unit", "color",
      "domestic_label", "intl_label", "summary", "source_summary",
      "intl_modes": { "<mode>": {"label", "intl_label", "source_summary"} },  // 선택
      "default_intl_mode"                                                       // 선택
    }
  }
}
```

### market 블록 (시장 지표)

KOSPI(^KS11)·S&P500(^GSPC)·USD/KRW의 일별 종가 시계열입니다. 두 시장의 휴장일이
서로 달라 **날짜 합집합 + 결측 `null`** 방식을 씁니다 (자산 페이로드의 "필수 필드
없는 행 제외" 방식과 다름). 프론트엔드는 이 블록으로 시장 지표 카드·정규화 비교
차트·상관계수 매트릭스를 그리며, **블록이 없으면 해당 섹션만 숨깁니다**(하위호환).

```jsonc
"market": {
  "dates": ["YYYY-MM-DD", ...],   // 합집합, 오름차순
  "kospi":   [3120.5, null, ...], // 휴장일은 null
  "sp500":   [...], "usd_krw": [...],
  "sources": { ... }
}
```

상관계수는 프론트엔드에서 일별 로그수익률의 피어슨 상관으로 계산합니다
(쌍별 완전 관측, 표본 20 미만은 표시 안 함). 대상: 금(XAU)·BTC·ETH·달러(USD/KRW)·KOSPI·S&P500.

### 고괴리 구간 규칙 (백엔드·프론트 공통)

구간 = |괴리율| ≥ 임계치인 **연속 데이터일**. `start` = 첫 초과일, `end` = 마지막 초과일(실존 데이터일), `duration_days` = (end − start) + 1일 (최소 1). 진행 중 구간은 마지막 데이터일로 닫습니다.

## URL 파라미터

| 파라미터 | 값 | 설명 |
|---|---|---|
| `asset` | `gold` \| `bitcoin` \| `usdt` | 초기 선택 자산 |
| `gold_source` | `ny_futures` \| `london_spot` | 금 국제가 기준 |
| `range` | `1m` \| `3m` \| `6m` \| `1y` \| `all` | 표시 기간 (기본 1y) |
| `theme` | `light` \| `dark` | 테마 강제 (미지정 시 저장값 → 시스템 설정) |
| `embed` | 존재 시 | 헤더·푸터 숨김 (iframe 임베드용) |

임베드 예시:

```html
<iframe src="https://ducklove.github.io/gold_gap/?embed&theme=dark&asset=gold"
        width="100%" height="900" frameborder="0"></iframe>
```

## 데이터 주의사항

- **일봉 경계**: 업비트·빗썸 일봉은 00:00 UTC(= 09:00 KST) 개시 캔들을 해당 KST 날짜로 라벨링합니다. yfinance 일봉은 각 거래소 현지 날짜 기준입니다. 즉 같은 날짜의 국내·국제 종가는 마감 시각이 서로 다르며, 일봉 데이터의 본질적 한계로 괴리율에 약간의 노이즈가 있습니다.
- **새로고침의 합성 포인트**: 브라우저 새로고침은 "마지막 종가(국내) + 실시간 시세(국제·환율)"를 섞어 오늘 포인트를 근사합니다. 휴장 시간대에는 실제보다 괴리가 크거나 작아 보일 수 있으며, 화면에 "실시간 근사" 배지로 표시됩니다.
- **비공식 API 의존**: ETF 좌당 금 그램수는 네이버 증권 모바일 API, 런던 현물 시계열은 WGC 차트 엔드포인트를 사용합니다. 예고 없이 변경될 수 있는 엔드포인트로, 실패 시 위 장애 격리 규칙으로 흡수됩니다.

## config.json (외부 연동 계약)

`config.json`은 이 리포 내부 코드가 소비하지 않으며, **외부 프로젝트가 읽는 연동 계약**으로서 배포 산출물에 포함됩니다(`baseUrl`·`dataUrl`·자산 라벨·`portfolioCodes`·`thresholdPct`). 스키마를 바꿀 때는 소비처 호환성을 먼저 확인하세요. `thresholdPct`는 자산 레지스트리와의 일치가 테스트로 강제됩니다.

## 워크플로우

| 워크플로우 | 트리거 | 역할 |
|---|---|---|
| `update-data.yml` | 30분 cron / 수동 | data.json 증분 갱신 → `data` 브랜치 커밋, 변경 시 배포 트리거, 임계 돌파·실패 이슈 알림 |
| `deploy.yml` | master push / 수동 | `data` 브랜치의 data.json + master의 정적 산출물 조립 → GitHub Pages 배포 |
| `ci.yml` | push·PR (`data` 브랜치 제외) | `data` 브랜치에서 골든 data.json fetch 후 ruff + pytest |

데이터는 `data` orphan 브랜치에만 커밋된다 — master를 클론해도 data.json이 없으며,
위 로컬 개발 0번 명령으로 받아온다.

## 알림 웹훅 (선택)

임계 돌파 시 이슈 알림에 더해, 리포 시크릿을 설정하면 웹훅으로도 발송됩니다
(시크릿이 없으면 해당 단계는 조용히 건너뜁니다):

| 시크릿 | 용도 |
|---|---|
| `DISCORD_WEBHOOK_URL` | 디스코드 채널 웹훅 URL |
| `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` | 텔레그램 봇 토큰과 대상 채팅 ID (둘 다 필요) |

설정 위치: 리포 Settings → Secrets and variables → Actions.

## 로드맵

구조·품질 평가와 단계별 개선 계획은 [docs/project-review.html](docs/project-review.html) 참고. 남은 주요 항목: 환율 기여도 분해 카드(분해 산식 정의 필요 — 환율 고정 가정 등 정의에 따라 결과가 크게 달라져 보류), i18n(영어).

## 라이선스

미정 (TBD). 공개 배포 전 라이선스 파일 추가 필요.
