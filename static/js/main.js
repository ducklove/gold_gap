// main.js — 부트스트랩 · 탭/국제기준 모드/조회 기간 상태 · URL 동기화 · 데이터 로드.
//
// Chart 전역은 index.html <head>의 클래식 CDN <script>(chart.js, annotation 플러그인)가
// 이 모듈보다 먼저 실행되므로(모듈은 항상 defer) charts.js에서 바로 사용할 수 있다.
//
// URL 파라미터: ?asset=gold|bitcoin|...  ?range=1m|3m|6m|1y|all
//              ?<asset>_source=<mode>   (intl_modes 보유 자산 — gold는 기존 ?gold_source 그대로)
//              ?lang=en (영어 모드일 때만 기록 — ko면 파라미터 제거)
//              ?theme=dark|light ?embed (head 부트 스크립트에서 처리)

import { buildAssetConfigs } from './config.js';
import { RANGE_OPTIONS, DEFAULT_RANGE, isValidRange, sliceDataByRange, rangeStartIndex } from './periods.js';
import { latestValue, formatKrw, formatUsd, formatAssetKrw } from './format.js';
import {
    applyChartDefaults, destroyCharts, renderPriceChart, renderGapChart, renderGapHistogram,
    renderTable, renderMarketChart, renderCorrelationTable,
    initPeriodTableSort, MOBILE_CHART_MEDIA_QUERY,
} from './charts.js';
import { fetchJson, applyClientLiveQuotes } from './live-quotes.js';
import { gapHistoricalStats, formatHistoricalStats } from './stats.js';
import { alignSeries, buildCorrelationMatrix, collectMarketSeries, latestWithChange } from './market.js';
import { decomposePriceChange } from './decompose.js';
import { t, getLang, setLang, applyStaticStrings, localizeAssetConfig } from './i18n.js';

const THEME_STORAGE_KEY = 'theme';

let currentTheme = document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
let allData = null;
let assetConfigs = {};   // key -> 정규화된 자산 설정
let assetOrder = [];     // 탭 표시 순서
let currentAsset = 'gold';
let currentRange = DEFAULT_RANGE;
const currentIntlModes = {}; // 자산별 선택된 국제가격 기준 모드
let refreshInFlight = false;
let lastUpdatedRaw = '';     // updated_at 원문 — 언어 전환 시 접두만 다시 붙여 재표시

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function setUpdatedAt(raw) {
    lastUpdatedRaw = raw || '';
    setText('updated-at', lastUpdatedRaw ? t('updated.prefix') + lastUpdatedRaw : '');
}

// 브라우저 합성 시세 사용 중일 때만 '실시간 근사' 배지 노출.
function setLiveBadge(visible) {
    const badge = document.getElementById('live-badge');
    if (badge) badge.hidden = !visible;
}

// ----- 설정/상태 -----

function rebuildConfigs() {
    const built = buildAssetConfigs(allData && allData.meta);
    assetConfigs = built.configs;
    assetOrder = built.order;
    if (!assetConfigs[currentAsset]) currentAsset = assetOrder[0];
}

// 자산 설정 + 데이터가 내려준 국내 라벨 반영(기존 USDT domestic_label 특례의 일반화):
// data.domestic_label이 있으면 국내 라벨과 출처 요약의 '국내:' 첫 구절을 교체한다.
// 언어 적용(localizeAssetConfig)도 여기서 — 모든 표시 경로가 이 함수를 거친다.
function resolveAssetConfig(asset, data) {
    const base = assetConfigs[asset];
    const config = { ...localizeAssetConfig(base) };
    if (data.domestic_label) {
        // EN 모드에서는 키 기반 사전 번역을 우선하되, 데이터가 meta와 다른 라벨을
        // 내려보내면(거래소 변경 등) 원문을 그대로 노출한다(한국어 폴백 규칙).
        if (getLang() !== 'en' || data.domestic_label !== base.domesticLabel) {
            config.domesticLabel = data.domestic_label;
        }
        const prefix = ['국내: ', 'Domestic: ']
            .find(p => config.sourceSummary && config.sourceSummary.startsWith(p));
        if (prefix) {
            const parts = config.sourceSummary.split(' · ');
            parts[0] = prefix + config.domesticLabel;
            config.sourceSummary = parts.join(' · ');
        }
    }
    return config;
}

function getCurrentIntlMode(asset, config, data) {
    if (!config.intlModes) return null;
    const candidates = [currentIntlModes[asset], data && data.default_intl_mode, config.defaultIntlMode];
    return candidates.find(mode => mode && config.intlModes[mode]) || Object.keys(config.intlModes)[0];
}

function getActiveAssetData(data, config, mode) {
    if (!config.intlModes || !data.intl_modes) return data;
    return data.intl_modes[mode] || data.intl_modes[data.default_intl_mode] || data;
}

function getActiveChartConfig(config, mode) {
    if (!config.intlModes) return config;
    const modeConfig = config.intlModes[mode];
    return {
        ...config,
        intlLabel: modeConfig ? modeConfig.intlLabel : config.intlLabel,
    };
}

// ----- URL 동기화 -----

function getRequestedAsset() {
    const asset = new URLSearchParams(window.location.search).get('asset');
    return assetConfigs[asset] ? asset : '';
}

// intl_modes 보유 자산의 ?<asset>_source 파라미터 반영(gold → gold_source 하위호환).
function applyRequestedIntlModes() {
    const params = new URLSearchParams(window.location.search);
    assetOrder.forEach(key => {
        const config = assetConfigs[key];
        if (!config || !config.intlModes) return;
        const mode = params.get(key + '_source');
        if (mode && config.intlModes[mode]) currentIntlModes[key] = mode;
    });
}

function readRequestedRange() {
    const range = new URLSearchParams(window.location.search).get('range');
    if (range && isValidRange(range.toLowerCase())) currentRange = range.toLowerCase();
}

function syncUrl(asset, config, mode) {
    const url = new URL(window.location.href);
    url.searchParams.set('asset', asset);
    if (config.intlModes && mode) {
        url.searchParams.set(asset + '_source', mode);
    }
    url.searchParams.set('range', currentRange);
    applyLangParam(url);
    window.history.replaceState(null, '', url);
}

// lang은 'en'일 때만 기록, 'ko'(기본)면 파라미터 제거.
function applyLangParam(url) {
    if (getLang() === 'en') url.searchParams.set('lang', 'en');
    else url.searchParams.delete('lang');
}

// 데이터 미로드 상태(로딩 실패 등)에서도 언어 전환이 URL에 반영되도록 하는 단독 경로.
function syncLangToUrl() {
    const url = new URL(window.location.href);
    applyLangParam(url);
    window.history.replaceState(null, '', url);
}

// ----- UI 조립(탭/모드 토글/기간 토글) -----

function renderTabs() {
    const nav = document.querySelector('.tab-nav');
    if (!nav) return;
    nav.replaceChildren();
    assetOrder.forEach(key => {
        const config = assetConfigs[key];
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'tab-btn' + (key === currentAsset ? ' active' : '');
        btn.dataset.asset = key;
        btn.setAttribute('role', 'tab');
        btn.setAttribute('aria-selected', key === currentAsset ? 'true' : 'false');
        btn.textContent = String(config.label || key).toUpperCase();
        btn.disabled = !!allData && !allData[key]; // 데이터 없는 자산은 비활성
        btn.addEventListener('click', () => switchTab(key));
        nav.appendChild(btn);
    });
}

function updateModeToggle(config, mode) {
    const toggle = document.getElementById('intl-mode-toggle');
    if (!toggle) return;
    toggle.replaceChildren();
    if (!config.intlModes) {
        toggle.hidden = true;
        return;
    }
    toggle.hidden = false;
    toggle.setAttribute('aria-label', t('intlMode.ariaLabeled', { label: config.label }));
    Object.values(config.intlModes).forEach(modeConfig => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'mode-btn' + (modeConfig.key === mode ? ' active' : '');
        btn.dataset.mode = modeConfig.key;
        btn.setAttribute('aria-pressed', modeConfig.key === mode ? 'true' : 'false');
        btn.textContent = modeConfig.label;
        btn.addEventListener('click', () => {
            currentIntlModes[currentAsset] = modeConfig.key;
            if (allData && allData[currentAsset]) switchTab(currentAsset);
        });
        toggle.appendChild(btn);
    });
}

function renderRangeToggle() {
    const wrap = document.getElementById('range-toggle');
    if (!wrap) return;
    wrap.replaceChildren();
    RANGE_OPTIONS.forEach(option => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'mode-btn';
        btn.dataset.range = option.key;
        // '전체'만 한국어 라벨 — 언어에 따라 치환(나머지 1M/3M/...은 중립).
        btn.textContent = option.key === 'all' ? t('range.all') : option.label;
        btn.addEventListener('click', () => {
            if (currentRange === option.key) return;
            currentRange = option.key;
            updateRangeToggle();
            if (allData && allData[currentAsset]) switchTab(currentAsset);
        });
        wrap.appendChild(btn);
    });
    updateRangeToggle();
}

function updateRangeToggle() {
    document.querySelectorAll('#range-toggle .mode-btn').forEach(btn => {
        const active = btn.dataset.range === currentRange;
        btn.classList.toggle('active', active);
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
}

// ----- 데이터 로드/새로고침 -----

function setRefreshButtonState(isBusy) {
    refreshInFlight = isBusy;
    const refreshBtn = document.getElementById('refreshBtn');
    if (!refreshBtn) return;
    refreshBtn.disabled = isBusy;
    refreshBtn.classList.toggle('is-refreshing', isBusy);
    refreshBtn.textContent = isBusy ? t('refresh.busy') : t('refresh');
}

async function loadData({ showLoading = true, bustCache = false } = {}) {
    if (refreshInFlight) return;
    setRefreshButtonState(true);
    document.getElementById('error').style.display = 'none';

    if (showLoading && !allData) {
        document.getElementById('loading').style.display = 'block';
        document.getElementById('content').style.display = 'none';
    }

    try {
        // 상대경로 fetch — GitHub Pages 서브패스와 Flask 로컬 모두에서 동작.
        const url = bustCache ? 'data.json?t=' + Date.now() : 'data.json';
        const resp = await fetch(url);
        if (!resp.ok) {
            let err = {};
            try {
                err = await resp.json();
            } catch (e) {
                err = {};
            }
            throw new Error(err.error || `HTTP ${resp.status}`);
        }

        allData = await resp.json();
        rebuildConfigs();   // data.json meta(schema v2) 있으면 자산 목록/라벨/임계치 갱신
        renderTabs();
        setLiveBadge(false); // 정식 data.json 로드 — 근사 배지 제거

        document.getElementById('loading').style.display = 'none';
        document.getElementById('content').style.display = 'block';

        if (allData.updated_at) {
            setUpdatedAt(allData.updated_at);
        }

        const requestedAsset = getRequestedAsset();
        applyRequestedIntlModes();
        const firstAvailable = [requestedAsset, currentAsset, ...assetOrder]
            .find(asset => asset && allData[asset]);
        if (firstAvailable) switchTab(firstAvailable);
    } catch (e) {
        if (!allData) {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('content').style.display = 'none';
        }
        document.getElementById('error').style.display = 'block';
        document.querySelector('.error-msg').textContent =
            t('error.load', { message: e.message });
    } finally {
        setRefreshButtonState(false);
    }
}

async function fetchBackendFreshData() {
    try {
        return await fetchJson('api/data?force=1&t=' + Date.now());
    } catch (e) {
        return null;
    }
}

async function refreshCurrentData() {
    if (refreshInFlight) return;
    setRefreshButtonState(true);
    document.getElementById('error').style.display = 'none';
    try {
        const fresh = await fetchBackendFreshData();
        if (fresh) {
            allData = fresh;
            rebuildConfigs();
            renderTabs();
            setLiveBadge(false); // 백엔드 정식 데이터 — 근사 배지 제거
        } else {
            await applyClientLiveQuotes(allData, assetConfigs);
            setLiveBadge(true); // 브라우저 합성 시세 — 근사 배지 표시
        }
        if (allData.updated_at) {
            setUpdatedAt(allData.updated_at);
        }
        switchTab(currentAsset);
    } catch (e) {
        document.getElementById('error').style.display = 'block';
        document.querySelector('.error-msg').textContent = t('error.refresh', { message: e.message });
    } finally {
        setRefreshButtonState(false);
    }
}

// ----- 화면 갱신 -----

function updateSourceNote(config, mode) {
    const note = document.getElementById('source-note');
    if (!note) return;
    if (config.intlModes) {
        const modeConfig = config.intlModes[mode];
        note.textContent = [config.sourceSummary, modeConfig && modeConfig.sourceSummary]
            .filter(Boolean).join(' · ');
        return;
    }
    note.textContent = config.sourceSummary || '';
}

function switchTab(asset) {
    currentAsset = asset;
    const data = allData && allData[asset];
    if (!data) return;
    const config = resolveAssetConfig(asset, data);
    const mode = getCurrentIntlMode(asset, config, data);
    const activeData = getActiveAssetData(data, config, mode);
    const chartConfig = getActiveChartConfig(config, mode);
    // 조회 기간 적용: dates·모든 시리즈를 슬라이스하고 구간은 buildHighGapPeriods로 재계산.
    const rangedData = sliceDataByRange(activeData, currentRange, config.threshold);

    syncUrl(asset, config, mode);

    document.documentElement.style.setProperty('--asset-color', config.gapColor);
    setText('asset-title', config.label);
    setText('asset-summary', config.summary);
    updateSourceNote(config, mode);
    updateModeToggle(config, mode);
    updateRangeToggle();

    document.querySelectorAll('.tab-btn').forEach(btn => {
        const isActive = btn.dataset.asset === asset;
        btn.classList.toggle('active', isActive);
        btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });

    updateCards(rangedData, config, activeData, data);
    updateDecomposition(activeData);

    setText('price-chart-title', t('chart.priceTitle', { label: config.label, unit: chartConfig.unit }));
    setText('table-title', t('table.title', { threshold: chartConfig.threshold }));
    setText('high-gap-label', t('card.highGap', { threshold: chartConfig.threshold }));

    destroyCharts();
    applyChartDefaults();
    renderPriceChart(rangedData, chartConfig);
    renderGapChart(rangedData, chartConfig);
    renderGapHistogram(rangedData, chartConfig);
    renderTable(rangedData.high_gap_periods, chartConfig);

    // 시장 섹션은 자산 탭과 무관하게 currentRange만 따르지만, 구현 단순화를 위해
    // 모든 재렌더 경로(탭/기간/테마/데이터 갱신)가 모이는 이곳에서 함께 갱신한다.
    updateMarketSection();
}

// ----- 가격 변동 분해 패널 -----

// 부호 있는 % 문자열(+1.23 / -1.23). suffix는 '%' 또는 '%p'.
function formatSignedPct(value, suffix) {
    return (value >= 0 ? '+' : '') + value.toFixed(2) + suffix;
}

// 요인 1행: 라벨 · 부호 있는 값(%p) · 중앙 0축 기준 좌우로 뻗는 순수 CSS 가로 막대.
// widthPct는 컨테이너 전체 대비 %(중앙에서 한쪽 최대 50%).
function buildDecompRow(label, value, maxAbs) {
    const positive = value >= 0;
    const row = document.createElement('div');
    row.className = 'decomp-row';

    const labelEl = document.createElement('div');
    labelEl.className = 'decomp-row-label';
    labelEl.textContent = label;
    row.appendChild(labelEl);

    const valueEl = document.createElement('div');
    valueEl.className = 'decomp-row-value ' + (value > 0 ? 'up' : value < 0 ? 'down' : '');
    valueEl.textContent = formatSignedPct(value, '%p');
    row.appendChild(valueEl);

    const bar = document.createElement('div');
    bar.className = 'decomp-bar';
    const fill = document.createElement('span');
    fill.className = 'decomp-bar-fill ' + (positive ? 'up' : 'down');
    const widthPct = maxAbs > 0 ? (Math.abs(value) / maxAbs) * 50 : 0;
    fill.style.width = widthPct.toFixed(2) + '%';
    bar.appendChild(fill);
    row.appendChild(bar);

    return row;
}

// 선택 기간의 국내 가격 변화(로그수익률)를 국제 가격/환율/김프 세 요인으로 분해해
// 표시한다. activeData = 현재 모드 데이터(금은 선택된 국제 기준), 기간은 currentRange.
// 유효 관측 쌍이 2개 미만이면 패널을 숨긴다.
function updateDecomposition(activeData) {
    const section = document.getElementById('decomp-section');
    if (!section) return;
    const startIdx = rangeStartIndex(activeData.dates, currentRange);
    const result = decomposePriceChange(activeData, startIdx);
    section.hidden = !result;
    if (!result) return;

    setText('decomp-headline', t('decomp.headline', { value: formatSignedPct(result.dom, '%') }));
    setText('decomp-dates', result.startDate + ' → ' + result.endDate);

    const rows = [
        { label: t('decomp.intl'), value: result.usd },
        { label: t('decomp.fx'), value: result.fx },
        { label: t('decomp.gap'), value: result.gap },
    ];
    const maxAbs = Math.max(...rows.map(row => Math.abs(row.value)));
    const wrap = document.getElementById('decomp-rows');
    if (!wrap) return;
    wrap.replaceChildren();
    rows.forEach(row => wrap.appendChild(buildDecompRow(row.label, row.value, maxAbs)));
}

// ----- 시장 지표 · 상관관계 섹션 -----

// 지수(KOSPI/S&P500) 카드 값 포맷 — 소수 2자리 고정.
function formatIndexValue(value) {
    return Number(value).toLocaleString('ko-KR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// 시장 카드 1장 채우기. stats는 latestWithChange 결과(없으면 null → '-').
// 전일 대비: 상승 빨강(--up)·하락 파랑(--down) — 국내 관례.
function fillMarketCard(idPrefix, stats, format) {
    setText(idPrefix + '-value', stats ? format(stats.value) : '-');
    const deltaEl = document.getElementById(idPrefix + '-delta');
    if (!deltaEl) return;
    deltaEl.classList.remove('up', 'down');
    if (!stats || stats.changePct == null) {
        deltaEl.textContent = '';
        return;
    }
    const sign = stats.changePct > 0 ? '+' : '';
    deltaEl.textContent = t('market.delta', { value: sign + stats.changePct.toFixed(2) });
    if (stats.changePct > 0) deltaEl.classList.add('up');
    else if (stats.changePct < 0) deltaEl.classList.add('down');
}

// market 블록이 있으면 카드/비교 차트/상관 매트릭스를 currentRange 기준으로 갱신하고,
// 없으면(현행 data.json 포함) 섹션 전체를 숨긴 채 둔다 — 에러 없이 동작해야 하는 절대 요건.
function updateMarketSection() {
    const section = document.getElementById('market-section');
    if (!section) return;
    const market = allData && allData.market;
    const hasMarket = !!(market && Array.isArray(market.dates) && market.dates.length > 0);
    section.hidden = !hasMarket;
    if (!hasMarket) return; // 시장 차트 인스턴스는 switchTab의 destroyCharts가 이미 정리

    // [{key, label, color, dates, values}] — 한국어 라벨('금 (XAU)')만 언어에 따라 치환.
    const collected = collectMarketSeries(allData).map(def =>
        def.key === 'gold' ? { ...def, label: t('market.seriesGold') } : def);
    const byKey = {};
    collected.forEach(def => { byKey[def.key] = def; });

    // 카드 3장: 최신 유효 관측 + 직전 유효 관측 대비(휴장 null은 건너뜀).
    fillMarketCard('market-kospi', byKey.kospi && latestWithChange(byKey.kospi.values), formatIndexValue);
    fillMarketCard('market-sp500', byKey.sp500 && latestWithChange(byKey.sp500.values), formatIndexValue);
    fillMarketCard('market-usdkrw', byKey.usd_krw && latestWithChange(byKey.usd_krw.values),
        v => formatKrw(v, { maximumFractionDigits: 2 }));

    // 합집합 날짜 축으로 정렬 후 조회 기간으로 슬라이스 — 차트/상관이 같은 행을 공유.
    const seriesMap = {};
    collected.forEach(def => { seriesMap[def.key] = { dates: def.dates, values: def.values }; });
    const aligned = alignSeries(seriesMap);
    const startIdx = rangeStartIndex(aligned.dates, currentRange);
    const ranged = { dates: startIdx > 0 ? aligned.dates.slice(startIdx) : aligned.dates };
    collected.forEach(def => {
        const values = aligned.series[def.key];
        ranged[def.key] = startIdx > 0 ? values.slice(startIdx) : values;
    });

    renderMarketChart(ranged, collected);

    const corrMap = {};
    collected.forEach(def => { corrMap[def.label] = { dates: ranged.dates, values: ranged[def.key] }; });
    renderCorrelationTable(buildCorrelationMatrix(corrMap));
}

// 통계 카드는 조회 기간(rangedData) 기준, 상세 카드의 '현재' 값은 최신 지점 기준.
function updateCards(rangedData, config, activeData, assetData) {
    const gaps = rangedData.gap_pct;
    const lastGap = gaps[gaps.length - 1];
    const absGaps = gaps.map(Math.abs);
    const maxGapIdx = absGaps.indexOf(Math.max(...absGaps));
    const maxGap = gaps[maxGapIdx];
    const avgGap = gaps.reduce((a, b) => a + b, 0) / gaps.length;

    setText('current-gap', lastGap.toFixed(2) + '%');
    setText('max-gap', maxGap.toFixed(2) + '%');
    setText('avg-gap', avgGap.toFixed(2) + '%');
    setText('high-gap-count', t('card.highGapCount', { count: rangedData.high_gap_periods.length }));

    // 역사적 위치는 조회 기간과 무관하게 전체 기간(activeData)을 기준으로 한다.
    const histText = formatHistoricalStats(gapHistoricalStats(activeData.gap_pct));
    setText('hist-percentile', histText ? histText.value : '-');
    setText('hist-percentile-sub', histText ? histText.sub : '');

    const el = document.getElementById('current-gap');
    const card = el.closest('.card');
    el.classList.remove('high', 'low');
    card.classList.remove('alert');
    if (Math.abs(lastGap) >= config.threshold) {
        el.classList.add('high');
        card.classList.add('alert');
    } else if (lastGap < 0) {
        el.classList.add('low');
    }

    document.getElementById('asset-detail-cards').style.display = '';
    updateDetailCards(assetData, config, activeData);
}

// 자산 데이터에서 USD 표시용 시리즈 키를 찾는다(gold_usd_oz, crypto_usd 등).
function findUsdSeriesKey(data) {
    if (!data) return null;
    return Object.keys(data).find(key =>
        key !== 'usd_krw' && Array.isArray(data[key]) && /_usd(_oz)?$/.test(key)
    ) || null;
}

// 모드 카드(label/value/sub) 채우기 — slot은 'primary' | 'secondary'.
function fillModeCard(slot, modeConfig, modeData) {
    const usdKey = findUsdSeriesKey(modeData);
    const value = usdKey ? latestValue(modeData, usdKey) : null;
    const unitLabel = usdKey && usdKey.endsWith('_oz') ? '$/oz' : 'USD';
    setText(`detail-intl-${slot}-label`, modeConfig.cardLabel || `${modeConfig.label} (${unitLabel})`);
    setText(`detail-intl-${slot}-value`, formatUsd(value, { maximumFractionDigits: 2 }));
    setText(`detail-intl-${slot}-sub`, modeConfig.cardSub || modeConfig.intlLabel || '');
}

function updateDetailCards(assetData, config, activeData) {
    const lastUsdKrw = latestValue(activeData, 'usd_krw');
    const lastDomestic = latestValue(activeData, 'domestic_price');
    const lastIntl = latestValue(activeData, 'intl_price');
    const secondaryCard = document.getElementById('detail-intl-secondary-card');

    setText('detail-usd-krw', formatKrw(lastUsdKrw, { maximumFractionDigits: 2 }));

    setText('detail-domestic-label', config.unit === 'KRW/g' ? t('detail.domesticPerGram') : t('detail.domestic'));
    setText('detail-domestic-value', formatAssetKrw(lastDomestic,
        typeof config.krwFractionDigits === 'number' ? config.krwFractionDigits : undefined));
    setText('detail-domestic-sub', config.domesticLabel);

    // intl_modes 보유 자산(기존 금 특례의 일반화): 모드별 USD 원자재가 카드를
    // 정의 순서대로 최대 2장 노출 — 금에서는 기존 런던/뉴욕 $/oz 화면과 동일.
    const modeKeys = config.intlModes ? Object.keys(config.intlModes) : [];
    if (modeKeys.length && assetData.intl_modes) {
        const [primaryKey, secondaryKey] = modeKeys;
        fillModeCard('primary', config.intlModes[primaryKey], assetData.intl_modes[primaryKey]);
        if (secondaryKey) {
            secondaryCard.hidden = false;
            fillModeCard('secondary', config.intlModes[secondaryKey], assetData.intl_modes[secondaryKey]);
        } else {
            secondaryCard.hidden = true;
        }
        return;
    }

    setText('detail-intl-primary-label', t('detail.intlKrw'));
    setText('detail-intl-primary-value', formatAssetKrw(lastIntl,
        typeof config.krwFractionDigits === 'number' ? config.krwFractionDigits : undefined));
    const usdKey = findUsdSeriesKey(activeData);
    const lastUsd = usdKey ? latestValue(activeData, usdKey) : null;
    setText('detail-intl-primary-sub',
        (config.usdSubLabel || 'USD') + ' ' + formatUsd(lastUsd, config.usdSubFormat || { maximumFractionDigits: 2 }));

    secondaryCard.hidden = true;
}

// ----- 테마 -----

function updateThemeButtonLabel() {
    const themeBtn = document.getElementById('themeToggle');
    if (!themeBtn) return;
    themeBtn.setAttribute('aria-label', currentTheme === 'dark' ? t('theme.toLight') : t('theme.toDark'));
    themeBtn.title = t('theme.title');
}

function applyTheme(theme, { persist = true, rerender = true } = {}) {
    currentTheme = theme === 'dark' ? 'dark' : 'light';
    document.documentElement.dataset.theme = currentTheme;
    updateThemeButtonLabel();

    if (persist) {
        try {
            localStorage.setItem(THEME_STORAGE_KEY, currentTheme);
        } catch (e) {
            // 저장 실패는 무시하고 세션 테마만 유지.
        }
    }

    if (rerender && allData && allData[currentAsset]) {
        switchTab(currentAsset);
    }
}

function bindThemeButton() {
    const themeBtn = document.getElementById('themeToggle');
    if (!themeBtn) return;
    themeBtn.addEventListener('click', () => {
        applyTheme(currentTheme === 'dark' ? 'light' : 'dark');
    });
    updateThemeButtonLabel();
}

function bindRefreshButton() {
    const refreshBtn = document.getElementById('refreshBtn');
    if (!refreshBtn) return;
    refreshBtn.addEventListener('click', refreshCurrentData);
}

// 회전·리사이즈로 폰↔데스크톱 폭 경계(MOBILE_CHART_MEDIA_QUERY)를 넘으면
// chartAspectRatio가 달라지므로 기존 재렌더 경로(switchTab)를 호출해 비율을
// 재적용한다. 1차 트리거는 matchMedia change(경계 통과 시에만 발화), 2차는
// change가 오지 않는 환경(일부 브라우저 회전·에뮬레이션)을 위한
// resize/orientationchange 디바운스 폴백 — 실제 경계 통과 여부를 검사하므로
// 어느 경로로 와도 재렌더는 경계 통과당 1회다. 부트에서 1회만 등록(중복 방지).
function bindChartRatioWatcher() {
    let mql = null;
    try {
        mql = window.matchMedia(MOBILE_CHART_MEDIA_QUERY);
    } catch (e) {
        return; // matchMedia 미지원 — 렌더 시점 비율 고정(기존 동작)
    }
    let wasMobile = mql.matches;
    const applyIfCrossed = () => {
        const nowMobile = window.matchMedia(MOBILE_CHART_MEDIA_QUERY).matches;
        if (nowMobile === wasMobile) return; // 경계 통과 없음 — 재렌더 불필요
        wasMobile = nowMobile;
        if (allData && allData[currentAsset]) switchTab(currentAsset);
    };
    if (typeof mql.addEventListener === 'function') mql.addEventListener('change', applyIfCrossed);
    else if (typeof mql.addListener === 'function') mql.addListener(applyIfCrossed); // 구형 Safari 폴백

    let resizeTimer = null;
    const debounced = () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(applyIfCrossed, 200);
    };
    window.addEventListener('resize', debounced);
    window.addEventListener('orientationchange', debounced);
}

// ----- 언어 토글 -----

// 언어 전환 후 전체 재렌더: 정적 라벨 → 기간 토글('전체'/'All') → URL →
// 데이터가 있으면 switchTab(카드/차트/테이블/분해 패널/시장 섹션까지 일괄 갱신).
function onLanguageChanged() {
    applyStaticStrings();
    updateThemeButtonLabel();
    renderRangeToggle();
    setUpdatedAt(lastUpdatedRaw);
    syncLangToUrl();
    if (allData && allData[currentAsset]) switchTab(currentAsset);
}

function bindLangButton() {
    const langBtn = document.getElementById('langToggle');
    if (!langBtn) return;
    langBtn.addEventListener('click', () => {
        setLang(getLang() === 'en' ? 'ko' : 'en');
        onLanguageChanged();
    });
}

// ----- 부트스트랩 -----

// 서비스워커 등록(PWA 오프라인 캐시) — 상대 경로라 GitHub Pages 서브패스(/gold_gap/)와
// Flask 로컬 모두에서 동작. 등록 실패는 치명적이지 않으므로 콘솔 경고만 남긴다.
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js')
        .catch(err => console.warn('서비스워커 등록 실패:', err));
}

applyStaticStrings();    // data-i18n 정적 라벨을 현재 언어로 — 모듈은 defer라 FOUC 최소
readRequestedRange();
rebuildConfigs();        // 데이터 로드 전에는 폴백 설정으로 탭/기본 상태 구성
renderTabs();
renderRangeToggle();
applyChartDefaults();
bindThemeButton();
bindRefreshButton();
bindLangButton();
bindChartRatioWatcher();
initPeriodTableSort();  // 구간 테이블 헤더 정렬 — 정적 thead에 1회 바인딩
loadData();
