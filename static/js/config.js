// config.js — 자산 설정. DOM에 의존하지 않는다.
//
// 두 가지 데이터 소스를 하나의 "정규화된 자산 설정" 형태로 합친다.
//  1) FALLBACK_ASSETS: 기존 인라인 스크립트의 하드코딩 ASSETS 객체(구버전 data.json용 폴백).
//  2) data.json 최상위 meta 블록(schema_version 2, 백엔드 계약):
//     meta.assets[key] = { label, order, threshold_pct, unit, color, domestic_label,
//                          intl_label, summary, source_summary,
//                          intl_modes: { mode: { label, intl_label, source_summary } },
//                          default_intl_mode }
//     (intl_modes/default_intl_mode는 모드 없는 자산에서 키 생략)
//
// meta가 없으면(현재 리포의 data.json) 폴백만으로 기존과 완전히 동일하게 렌더링되고,
// meta가 있으면 자산 목록·순서·라벨·임계치·색상을 meta 기준으로 구성하되
// meta 계약에 없는 표시용 세부값(intlColor, 상세카드 라벨 등)은 폴백에서 보충한다.

// 국제가(파란 선) 공통 색 — 폴백 3종 모두 동일 값.
const DEFAULT_INTL_COLOR = '#2a78c9';
// meta에만 존재하는 신규 자산의 기본 강조색(--accent와 동일 계열).
const DEFAULT_ASSET_COLOR = '#6c8cff';

// 기존 인라인 ASSETS 그대로 + 상세카드/포맷용 추가 필드(cardLabel, cardSub,
// krwFractionDigits, usdSubLabel, usdSubFormat, order, defaultIntlMode).
export const FALLBACK_ASSETS = {
    gold: {
        label: 'Gold',
        order: 1,
        summary: 'KRX 금현물과 COMEX 금 가격을 원화 기준으로 비교합니다.',
        domesticLabel: '국내 (KRX 금현물)',
        intlLabel: '뉴욕선물 GC=F (KRW 환산)',
        unit: 'KRW/g',
        threshold: 5,
        domesticColor: '#d9a441',
        intlColor: '#2a78c9',
        gapColor: '#d9a441',
        highlightColor: 'rgba(217, 164, 65, 0.12)',
        sourceSummary: '국내: ACE KRX금현물 411060.KS · 환율: USD/KRW KRW=X · 국제 기준은 토글로 선택',
        intlModes: {
            london_spot: {
                label: '런던 현물',
                intlLabel: '런던 현물 XAU/USD (KRW 환산)',
                sourceSummary: '국제: World Gold Council/ICE spot + Gold API latest XAU',
                cardLabel: '런던 현물 ($/oz)',
                cardSub: 'WGC/ICE spot',
            },
            ny_futures: {
                label: '뉴욕선물',
                intlLabel: '뉴욕선물 GC=F (KRW 환산)',
                sourceSummary: '국제: COMEX Gold Futures GC=F (Yahoo Finance)',
                cardLabel: '뉴욕선물 ($/oz)',
                cardSub: 'COMEX GC=F',
            },
        },
        defaultIntlMode: 'ny_futures',
    },
    bitcoin: {
        label: 'Bitcoin',
        order: 2,
        summary: '업비트 BTC와 BTC-USD 국제가의 프리미엄을 추적합니다.',
        domesticLabel: '업비트 BTC',
        intlLabel: 'BTC-USD (KRW 환산)',
        unit: 'KRW',
        threshold: 5,
        domesticColor: '#d9791f',
        intlColor: '#2a78c9',
        gapColor: '#d9791f',
        highlightColor: 'rgba(217, 121, 31, 0.12)',
        sourceSummary: '국내: Upbit KRW-BTC · 국제: BTC-USD (Yahoo Finance) · 환율: USD/KRW KRW=X',
        usdSubLabel: 'BTC-USD',
        usdSubFormat: { maximumFractionDigits: 2 },
    },
    usdt: {
        label: 'USDT',
        order: 3,
        summary: '빗썸 USDT/KRW와 USDT-USD 환산 기준의 괴리율을 확인합니다.',
        domesticLabel: '빗썸 USDT',
        intlLabel: 'USDT-USD (KRW 환산)',
        unit: 'KRW',
        threshold: 3,
        domesticColor: '#1f9b57',
        intlColor: '#2a78c9',
        gapColor: '#1f9b57',
        highlightColor: 'rgba(31, 155, 87, 0.12)',
        sourceSummary: '국내: Bithumb USDT_KRW · 국제: USDT-USD (Yahoo Finance) · 환율: USD/KRW KRW=X',
        krwFractionDigits: 2,
        usdSubLabel: 'USDT-USD',
        usdSubFormat: { minimumFractionDigits: 4, maximumFractionDigits: 4 },
    },
};

// '#rrggbb'/'#rgb' → 'rgba(r, g, b, a)'. 폴백 highlightColor와 동일한 표기 형식.
export function hexToRgba(hex, alpha) {
    const raw = String(hex || '').replace('#', '').trim();
    const full = raw.length === 3
        ? raw.split('').map(ch => ch + ch).join('')
        : raw;
    if (!/^[0-9a-fA-F]{6}$/.test(full)) return `rgba(108, 140, 255, ${alpha})`;
    const r = parseInt(full.slice(0, 2), 16);
    const g = parseInt(full.slice(2, 4), 16);
    const b = parseInt(full.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// intlLabel('BTC-USD (KRW 환산)' 등)에서 USD 시세 부제 접두('BTC-USD')를 유추.
function deriveUsdSubLabel(intlLabel) {
    const text = String(intlLabel || '').trim();
    if (!text) return 'USD';
    return text.split(' (')[0].trim() || 'USD';
}

// 폴백 항목(없으면 {})을 메인/차트 코드가 쓰는 정규화 형태로 변환.
function normalizeFromFallback(key, fb) {
    const gapColor = fb.gapColor || DEFAULT_ASSET_COLOR;
    const config = {
        key,
        label: fb.label || key,
        order: typeof fb.order === 'number' ? fb.order : null,
        summary: fb.summary || '',
        domesticLabel: fb.domesticLabel || '국내가격',
        intlLabel: fb.intlLabel || '국제가격 (KRW 환산)',
        unit: fb.unit || 'KRW',
        threshold: typeof fb.threshold === 'number' ? fb.threshold : 5,
        domesticColor: fb.domesticColor || gapColor,
        intlColor: fb.intlColor || DEFAULT_INTL_COLOR,
        gapColor,
        highlightColor: fb.highlightColor || hexToRgba(gapColor, 0.12),
        sourceSummary: fb.sourceSummary || '',
        krwFractionDigits: typeof fb.krwFractionDigits === 'number' ? fb.krwFractionDigits : null,
        usdSubLabel: fb.usdSubLabel || deriveUsdSubLabel(fb.intlLabel),
        usdSubFormat: fb.usdSubFormat || { maximumFractionDigits: 2 },
        intlModes: null,
        defaultIntlMode: null,
    };
    if (fb.intlModes && Object.keys(fb.intlModes).length) {
        config.intlModes = {};
        Object.entries(fb.intlModes).forEach(([modeKey, mode]) => {
            config.intlModes[modeKey] = {
                key: modeKey,
                label: mode.label || modeKey,
                intlLabel: mode.intlLabel || config.intlLabel,
                sourceSummary: mode.sourceSummary || '',
                cardLabel: mode.cardLabel || null,
                cardSub: mode.cardSub || mode.intlLabel || '',
            };
        });
        config.defaultIntlMode = fb.defaultIntlMode || Object.keys(config.intlModes)[0];
    }
    return config;
}

// meta.assets[key] 한 항목을 폴백 기반 base 위에 덮어쓴다(meta 우선).
function mergeMetaAsset(base, metaAsset) {
    const m = metaAsset || {};
    const config = { ...base };
    if (m.label) config.label = m.label;
    if (typeof m.order === 'number') config.order = m.order;
    if (m.summary) config.summary = m.summary;
    if (m.domestic_label) config.domesticLabel = m.domestic_label;
    if (m.intl_label) {
        config.intlLabel = m.intl_label;
        if (!base.usdSubLabel || base.usdSubLabel === 'USD') {
            config.usdSubLabel = deriveUsdSubLabel(m.intl_label);
        }
    }
    if (m.unit) config.unit = m.unit;
    if (typeof m.threshold_pct === 'number') config.threshold = m.threshold_pct;
    if (m.source_summary) config.sourceSummary = m.source_summary;
    if (m.color) {
        // 계약상 색은 단일 color 하나 — 국내선/괴리율선/하이라이트가 이 색을 따른다
        // (폴백에서도 domesticColor === gapColor, highlight = gapColor 12%).
        config.domesticColor = m.color;
        config.gapColor = m.color;
        config.highlightColor = hexToRgba(m.color, 0.12);
    }

    // intl_modes는 meta가 기준: meta에 키가 없으면 모드 없는 자산으로 취급.
    if (m.intl_modes && Object.keys(m.intl_modes).length) {
        const baseModes = base.intlModes || {};
        config.intlModes = {};
        Object.entries(m.intl_modes).forEach(([modeKey, mode]) => {
            const fbMode = baseModes[modeKey] || {};
            config.intlModes[modeKey] = {
                key: modeKey,
                label: (mode && mode.label) || fbMode.label || modeKey,
                intlLabel: (mode && mode.intl_label) || fbMode.intlLabel || config.intlLabel,
                sourceSummary: (mode && mode.source_summary) || fbMode.sourceSummary || '',
                cardLabel: fbMode.cardLabel || null, // meta 계약에 없음 — 폴백 또는 라벨에서 유도
                cardSub: fbMode.cardSub || ((mode && mode.intl_label) || fbMode.intlLabel || ''),
            };
        });
        const candidates = [m.default_intl_mode, base.defaultIntlMode, Object.keys(config.intlModes)[0]];
        config.defaultIntlMode = candidates.find(mode => mode && config.intlModes[mode]) || null;
    } else {
        config.intlModes = null;
        config.defaultIntlMode = null;
    }
    return config;
}

// meta(없으면 null)를 받아 { configs: {key: config}, order: [key...] }를 반환.
// - meta 없음/비어 있음 → 폴백 3종(gold, bitcoin, usdt)을 기존 순서대로
// - meta 있음 → meta.assets의 자산 목록을 order 오름차순으로
export function buildAssetConfigs(meta) {
    const hasMeta = !!(meta && meta.assets && typeof meta.assets === 'object'
        && Object.keys(meta.assets).length);
    const keys = hasMeta ? Object.keys(meta.assets) : Object.keys(FALLBACK_ASSETS);

    const configs = {};
    keys.forEach((key, idx) => {
        const base = normalizeFromFallback(key, FALLBACK_ASSETS[key] || {});
        const config = hasMeta ? mergeMetaAsset(base, meta.assets[key]) : base;
        if (typeof config.order !== 'number') config.order = 1000 + idx; // order 미지정은 뒤로
        configs[key] = config;
    });

    const order = keys.slice().sort((a, b) => configs[a].order - configs[b].order);
    return { configs, order };
}
