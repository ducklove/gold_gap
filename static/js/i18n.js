// i18n.js — 한/영 프론트 전용 번역 레이어.
//
// 언어 결정: ?lang=en|ko > localStorage('lang') > 'ko'.
// 페이지 <title>과 og/meta 설명은 공유 미리보기 일관성을 위해 한국어로 고정하고,
// 화면 텍스트(헤더·카드·차트 라벨·테이블·캡션 등)만 번역한다.
//
// 사용 규약:
//  - t(key, params): STRINGS[현재 언어][key]에 {param} 치환. 키가 없으면 ko → 키 그대로 폴백.
//  - 정적 마크업은 data-i18n / data-i18n-aria / data-i18n-title 속성 + applyStaticStrings().
//  - data.json meta가 내려주는 자산별 한국어 문구(summary, domestic_label, intl_label,
//    source_summary, 모드 라벨)는 자산 키 기반 EN 사전(ASSET_EN)으로 번역하고,
//    사전에 없는 신규 자산은 meta 원문(한국어)으로 폴백한다 — localizeAssetConfig().
//
// ko/en 키 집합은 항상 동일해야 한다(검증 코드가 보장).

const LANG_STORAGE_KEY = 'lang';

export const STRINGS = {
    ko: {
        'app.title': '김치프리미엄 대시보드',
        'updated.prefix': '최종 업데이트: ',
        'live.badge': '실시간 근사',
        'live.updatedPrefix': '브라우저 현재가 갱신: ',
        'live.noBaseData': '기준 데이터가 아직 로드되지 않았습니다.',
        'live.noQuotes': '브라우저에서 접근 가능한 현재가 API 응답이 없습니다.',
        'refresh': '새로고침',
        'refresh.busy': '새로고침 중...',
        'theme.title': '테마 전환',
        'theme.toDark': '다크 모드로 전환',
        'theme.toLight': '일반 모드로 전환',
        'lang.label': 'EN',
        'lang.aria': '영어로 전환',
        'tabs.aria': '자산 선택',
        'asset.kicker': 'Spread Monitor',
        'asset.summaryDefault': '국내 거래가와 환산 국제가의 차이를 추적합니다.',
        'loading': '데이터를 불러오는 중...',
        'error.load': '데이터 로딩 실패: {message}',
        'error.refresh': '현재가 갱신 실패: {message}',
        'range.aria': '조회 기간 선택',
        'range.all': '전체',
        'card.currentGap': '현재 괴리율',
        'card.maxGap': '최대 괴리율',
        'card.avgGap': '평균 괴리율',
        'card.highGap': '{threshold}%+ 발생',
        'card.highGapCount': '{count}회',
        'card.histPosition': '역사적 위치',
        'stats.percentile': '백분위 {value}%',
        'stats.basis': '전체 {count}일 기준',
        'detail.fx': '기준환율',
        'detail.domestic': '국내가격',
        'detail.domesticPerGram': '국내가격 (원/g)',
        'detail.intl': '국제가격',
        'detail.intlKrw': '국제가격 (KRW 환산)',
        'intlMode.aria': '국제가격 기준 선택',
        'intlMode.ariaLabeled': '{label} 국제가격 기준 선택',
        'section.priceCompare': '가격 비교',
        'section.gapTrend': '괴리율 추이 (%)',
        'section.gapHist': '괴리율 분포',
        'section.gapTable': '괴리율 발생 구간',
        'chart.priceTitle': '{label} 가격 비교 ({unit})',
        'chart.gapAxis': '괴리율 (%)',
        'chart.gapTooltip': '괴리율: {value}%',
        'chart.days': '일수',
        'chart.histFreq': '빈도: {count}일',
        'chart.histCurrent': '현재 {value}%',
        'chart.normalizedBase': '시작점 = 100',
        'table.title': '{threshold}% 이상 괴리율 발생 구간',
        'table.start': '시작일',
        'table.end': '종료일',
        'table.maxGap': '최대 괴리율',
        'table.duration': '지속일수',
        'table.days': '{days}일',
        'table.noPeriods': '{threshold}% 이상 괴리율 발생 구간이 없습니다.',
        'market.title': '시장 지표 · 상관관계',
        'market.note': 'KOSPI · S&P 500 · 환율과 자산 USD 시세 — 자산 탭과 무관하게 선택한 조회 기간을 따릅니다.',
        'market.delta': '{value}% 전일 대비',
        'market.normalized': '정규화 비교 (시작점 = 100)',
        'market.corrTitle': '상관계수 매트릭스',
        'market.corrCaption': "일별 로그수익률 · 쌍별 완전 관측 · 표본 20 미만 '-'",
        'market.seriesGold': '금 (XAU)',
        'decomp.title': '가격 변동 분해',
        'decomp.headline': '선택 기간 국내 가격 {value}',
        'decomp.intl': '국제 가격',
        'decomp.fx': '환율 (USD/KRW)',
        'decomp.gap': '김치프리미엄',
        'decomp.caption': '로그수익률 분해 — 세 요인의 합 = 국내 가격 변화 (김프 요인은 잔차)',
        'footer': '데이터 출처: KRX, COMEX, Bithumb, Upbit, yfinance',
    },
    en: {
        'app.title': 'Kimchi Premium Dashboard',
        'updated.prefix': 'Last updated: ',
        'live.badge': 'Live approx.',
        'live.updatedPrefix': 'Browser live update: ',
        'live.noBaseData': 'Base data has not been loaded yet.',
        'live.noQuotes': 'No live quote APIs reachable from the browser.',
        'refresh': 'Refresh',
        'refresh.busy': 'Refreshing...',
        'theme.title': 'Toggle theme',
        'theme.toDark': 'Switch to dark mode',
        'theme.toLight': 'Switch to light mode',
        'lang.label': '한국어',
        'lang.aria': 'Switch to Korean',
        'tabs.aria': 'Asset selection',
        'asset.kicker': 'Spread Monitor',
        'asset.summaryDefault': 'Tracks the spread between domestic and converted international prices.',
        'loading': 'Loading data...',
        'error.load': 'Failed to load data: {message}',
        'error.refresh': 'Failed to refresh live quotes: {message}',
        'range.aria': 'Time range selection',
        'range.all': 'All',
        'card.currentGap': 'Current gap',
        'card.maxGap': 'Max gap',
        'card.avgGap': 'Average gap',
        'card.highGap': '{threshold}%+ periods',
        'card.highGapCount': '{count}',
        'card.histPosition': 'Historical position',
        'stats.percentile': 'Percentile {value}%',
        'stats.basis': 'based on all {count} days',
        'detail.fx': 'Reference FX rate',
        'detail.domestic': 'Domestic price',
        'detail.domesticPerGram': 'Domestic price (KRW/g)',
        'detail.intl': 'International price',
        'detail.intlKrw': 'International price (KRW converted)',
        'intlMode.aria': 'International price basis',
        'intlMode.ariaLabeled': '{label} international price basis',
        'section.priceCompare': 'Price Comparison',
        'section.gapTrend': 'Gap Trend (%)',
        'section.gapHist': 'Gap Distribution',
        'section.gapTable': 'Gap Periods',
        'chart.priceTitle': '{label} Price Comparison ({unit})',
        'chart.gapAxis': 'Gap (%)',
        'chart.gapTooltip': 'Gap: {value}%',
        'chart.days': 'Days',
        'chart.histFreq': 'Frequency: {count}',
        'chart.histCurrent': 'Now {value}%',
        'chart.normalizedBase': 'Start = 100',
        'table.title': 'Gap Periods Above {threshold}%',
        'table.start': 'Start',
        'table.end': 'End',
        'table.maxGap': 'Max Gap',
        'table.duration': 'Duration',
        'table.days': '{days}d',
        'table.noPeriods': 'No gap periods above {threshold}%.',
        'market.title': 'Market Indicators · Correlations',
        'market.note': 'KOSPI · S&P 500 · FX and asset USD prices — follows the selected range regardless of the asset tab.',
        'market.delta': '{value}% vs prev. day',
        'market.normalized': 'Normalized Comparison (start = 100)',
        'market.corrTitle': 'Correlation Matrix',
        'market.corrCaption': "Daily log returns · pairwise complete observations · '-' when n < 20",
        'market.seriesGold': 'Gold (XAU)',
        'decomp.title': 'Price Change Decomposition',
        'decomp.headline': 'Domestic price {value} over the selected period',
        'decomp.intl': 'International price',
        'decomp.fx': 'FX (USD/KRW)',
        'decomp.gap': 'Kimchi premium',
        'decomp.caption': 'Log-return decomposition — the three factors sum to the domestic price change (the premium factor is the residual)',
        'footer': 'Data sources: KRX, COMEX, Bithumb, Upbit, yfinance',
    },
};

// data.json meta(또는 폴백)가 내려주는 자산별 한국어 문구의 EN 사전 — 자산 키 기반.
// 여기 없는 자산(신규 추가 등)은 meta 원문(한국어)을 그대로 노출한다.
const ASSET_EN = {
    gold: {
        summary: 'Compares KRX spot gold and COMEX gold prices in KRW terms.',
        domesticLabel: 'Domestic (KRX spot gold)',
        intlLabel: 'NY futures GC=F (KRW converted)',
        sourceSummary: 'Domestic: ACE KRX Gold Spot 411060.KS · FX: USD/KRW KRW=X · choose the international basis with the toggle',
        intlModes: {
            london_spot: {
                label: 'London spot',
                intlLabel: 'London spot XAU/USD (KRW converted)',
                sourceSummary: 'International: World Gold Council/ICE spot + Gold API latest XAU',
                cardLabel: 'London spot ($/oz)',
            },
            ny_futures: {
                label: 'NY futures',
                intlLabel: 'NY futures GC=F (KRW converted)',
                sourceSummary: 'International: COMEX Gold Futures GC=F (Yahoo Finance)',
                cardLabel: 'NY futures ($/oz)',
            },
        },
    },
    bitcoin: {
        summary: 'Tracks the premium of Upbit BTC over the BTC-USD international price.',
        domesticLabel: 'Upbit BTC',
        intlLabel: 'BTC-USD (KRW converted)',
        sourceSummary: 'Domestic: Upbit KRW-BTC · International: BTC-USD (Yahoo Finance) · FX: USD/KRW KRW=X',
    },
    eth: {
        summary: 'Tracks the premium of Upbit ETH over the ETH-USD international price.',
        domesticLabel: 'Upbit ETH',
        intlLabel: 'ETH-USD (KRW converted)',
        sourceSummary: 'Domestic: Upbit KRW-ETH · International: ETH-USD (Yahoo Finance) · FX: USD/KRW KRW=X',
    },
    usdt: {
        summary: 'Tracks the gap between Bithumb USDT/KRW and the USDT-USD converted basis.',
        domesticLabel: 'Bithumb USDT',
        intlLabel: 'USDT-USD (KRW converted)',
        sourceSummary: 'Domestic: Bithumb USDT_KRW · International: USDT-USD (Yahoo Finance) · FX: USD/KRW KRW=X',
    },
};

// ?lang > localStorage > 'ko'. 노드(테스트)나 저장소 차단 환경에서도 안전하게 폴백.
function detectLang() {
    try {
        const param = new URLSearchParams(window.location.search).get('lang');
        if (param === 'en' || param === 'ko') return param;
    } catch (e) { /* window 없음(노드 등) — 다음 후보로 */ }
    try {
        const stored = localStorage.getItem(LANG_STORAGE_KEY);
        if (stored === 'en' || stored === 'ko') return stored;
    } catch (e) { /* 저장소 접근 불가 — 기본값 */ }
    return 'ko';
}

let currentLang = detectLang();

export function getLang() {
    return currentLang;
}

// 언어 변경: 저장(가능하면) + <html lang> 갱신. 화면 재렌더는 호출 측(main.js) 책임.
export function setLang(lang) {
    currentLang = lang === 'en' ? 'en' : 'ko';
    try {
        localStorage.setItem(LANG_STORAGE_KEY, currentLang);
    } catch (e) { /* 저장 실패는 무시 — 세션 언어만 유지 */ }
    try {
        document.documentElement.lang = currentLang;
    } catch (e) { /* DOM 없음(노드) — 무시 */ }
    return currentLang;
}

// 키 조회 + {param} 치환. 현재 언어에 없으면 ko, 그래도 없으면 키 자체를 반환.
export function t(key, params) {
    const dict = STRINGS[currentLang] || STRINGS.ko;
    let text = Object.prototype.hasOwnProperty.call(dict, key) ? dict[key] : STRINGS.ko[key];
    if (text == null) return key;
    if (params) {
        text = text.replace(/\{(\w+)\}/g, (match, name) =>
            (params[name] != null ? String(params[name]) : match));
    }
    return text;
}

// config.js 병합 결과(정규화된 자산 설정)에 현재 언어를 적용한 사본을 반환.
// EN 모드에서 ASSET_EN[key]가 있으면 표시 문구를 덮어쓰고, 없으면 원본 그대로(한국어 폴백).
export function localizeAssetConfig(config, lang = currentLang) {
    if (!config || lang !== 'en') return config;
    const dict = ASSET_EN[config.key];
    if (!dict) return config; // 사전에 없는 신규 자산 — meta 원문 폴백
    const out = { ...config };
    if (dict.summary) out.summary = dict.summary;
    if (dict.domesticLabel) out.domesticLabel = dict.domesticLabel;
    if (dict.intlLabel) out.intlLabel = dict.intlLabel;
    if (dict.sourceSummary) out.sourceSummary = dict.sourceSummary;
    if (config.intlModes && dict.intlModes) {
        out.intlModes = {};
        Object.entries(config.intlModes).forEach(([modeKey, mode]) => {
            const m = dict.intlModes[modeKey];
            out.intlModes[modeKey] = m ? {
                ...mode,
                label: m.label || mode.label,
                intlLabel: m.intlLabel || mode.intlLabel,
                sourceSummary: m.sourceSummary || mode.sourceSummary,
                cardLabel: m.cardLabel || mode.cardLabel,
            } : mode;
        });
    }
    return out;
}

// 정적 마크업의 data-i18n(텍스트) / data-i18n-aria(aria-label) / data-i18n-title(title)
// 속성을 현재 언어로 채운다. 부트 시 1회 + 언어 전환 시마다 호출.
export function applyStaticStrings(root = typeof document !== 'undefined' ? document : null) {
    if (!root || typeof root.querySelectorAll !== 'function') return;
    root.querySelectorAll('[data-i18n]').forEach(el => {
        el.textContent = t(el.dataset.i18n);
    });
    root.querySelectorAll('[data-i18n-aria]').forEach(el => {
        el.setAttribute('aria-label', t(el.dataset.i18nAria));
    });
    root.querySelectorAll('[data-i18n-title]').forEach(el => {
        el.title = t(el.dataset.i18nTitle);
    });
}
