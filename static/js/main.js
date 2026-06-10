// main.js — 부트스트랩 · 탭/국제기준 모드/조회 기간 상태 · URL 동기화 · 데이터 로드.
//
// Chart 전역은 index.html <head>의 클래식 CDN <script>(chart.js, annotation 플러그인)가
// 이 모듈보다 먼저 실행되므로(모듈은 항상 defer) charts.js에서 바로 사용할 수 있다.
//
// URL 파라미터: ?asset=gold|bitcoin|...  ?range=1m|3m|6m|1y|all
//              ?<asset>_source=<mode>   (intl_modes 보유 자산 — gold는 기존 ?gold_source 그대로)
//              ?theme=dark|light ?embed (head 부트 스크립트에서 처리)

import { buildAssetConfigs } from './config.js';
import { RANGE_OPTIONS, DEFAULT_RANGE, isValidRange, sliceDataByRange } from './periods.js';
import { latestValue, formatKrw, formatUsd, formatAssetKrw } from './format.js';
import { applyChartDefaults, destroyCharts, renderPriceChart, renderGapChart, renderTable } from './charts.js';
import { fetchJson, applyClientLiveQuotes } from './live-quotes.js';
import { gapHistoricalStats, formatHistoricalStats } from './stats.js';

const THEME_STORAGE_KEY = 'theme';

let currentTheme = document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
let allData = null;
let assetConfigs = {};   // key -> 정규화된 자산 설정
let assetOrder = [];     // 탭 표시 순서
let currentAsset = 'gold';
let currentRange = DEFAULT_RANGE;
const currentIntlModes = {}; // 자산별 선택된 국제가격 기준 모드
let refreshInFlight = false;

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function setUpdatedAt(text) {
    setText('updated-at', text);
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
function resolveAssetConfig(asset, data) {
    const config = { ...assetConfigs[asset] };
    if (data.domestic_label) {
        config.domesticLabel = data.domestic_label;
        if (config.sourceSummary && config.sourceSummary.startsWith('국내: ')) {
            const parts = config.sourceSummary.split(' · ');
            parts[0] = '국내: ' + data.domestic_label;
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
    toggle.setAttribute('aria-label', config.label + ' 국제가격 기준 선택');
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
        btn.textContent = option.label;
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
    refreshBtn.textContent = isBusy ? '새로고침 중...' : '새로고침';
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
            setUpdatedAt('최종 업데이트: ' + allData.updated_at);
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
            '데이터 로딩 실패: ' + e.message;
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
            setUpdatedAt('최종 업데이트: ' + allData.updated_at);
        }
        switchTab(currentAsset);
    } catch (e) {
        document.getElementById('error').style.display = 'block';
        document.querySelector('.error-msg').textContent = '현재가 갱신 실패: ' + e.message;
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

    setText('price-chart-title', config.label + ' 가격 비교 (' + chartConfig.unit + ')');
    setText('table-title', chartConfig.threshold + '% 이상 괴리율 발생 구간');
    setText('high-gap-label', chartConfig.threshold + '%+ 발생');

    destroyCharts();
    applyChartDefaults();
    renderPriceChart(rangedData, chartConfig);
    renderGapChart(rangedData, chartConfig);
    renderTable(rangedData.high_gap_periods, chartConfig);
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
    setText('high-gap-count', rangedData.high_gap_periods.length + '회');

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

    setText('detail-domestic-label', config.unit === 'KRW/g' ? '국내가격 (원/g)' : '국내가격');
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

    setText('detail-intl-primary-label', '국제가격 (KRW 환산)');
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
    themeBtn.setAttribute('aria-label', currentTheme === 'dark' ? '일반 모드로 전환' : '다크 모드로 전환');
    themeBtn.title = '테마 전환';
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

// ----- 부트스트랩 -----

readRequestedRange();
rebuildConfigs();        // 데이터 로드 전에는 폴백 설정으로 탭/기본 상태 구성
renderTabs();
renderRangeToggle();
applyChartDefaults();
bindThemeButton();
bindRefreshButton();
loadData();
