// charts.js — Chart.js 렌더링(가격/괴리율/분포/시장 비교 차트, 하이라이트 박스, 테이블).
//
// 전역 Chart와 annotation 플러그인은 index.html <head>의 클래식 CDN <script>가
// 로드한다. 클래식 스크립트는 문서 파싱을 막고 즉시 실행되는 반면 ES 모듈은 항상
// defer로 실행되므로, 이 모듈이 평가되는 시점에는 Chart 전역이 보장된다.

import { formatPrice } from './format.js';
import { buildGapHistogram } from './stats.js';
import { normalizeTo100 } from './market.js';
import { t } from './i18n.js';

let priceChartInstance = null;
let gapChartInstance = null;
let gapHistChartInstance = null;
let marketChartInstance = null;

function getThemeColor(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function getChartTheme() {
    return {
        text: getThemeColor('--text'),
        textDim: getThemeColor('--text-dim'),
        grid: getThemeColor('--grid'),
        surface: getThemeColor('--surface'),
        surface2: getThemeColor('--surface2'),
        border: getThemeColor('--border'),
        up: getThemeColor('--up'),
        down: getThemeColor('--down'),
    };
}

export function applyChartDefaults() {
    // CDN 로드 실패 시 부트 전체가 죽지 않도록 방어 — 이후 렌더 단계 오류는
    // loadData의 try/catch가 에러 박스로 표시한다.
    if (typeof Chart === 'undefined') return;
    const theme = getChartTheme();
    Chart.defaults.color = theme.textDim;
    Chart.defaults.borderColor = theme.grid;
    Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
}

export function destroyCharts() {
    if (priceChartInstance) { priceChartInstance.destroy(); priceChartInstance = null; }
    if (gapChartInstance) { gapChartInstance.destroy(); gapChartInstance = null; }
    if (gapHistChartInstance) { gapHistChartInstance.destroy(); gapHistChartInstance = null; }
    if (marketChartInstance) { marketChartInstance.destroy(); marketChartInstance = null; }
}

// 차트 줌: 드래그 박스 줌(x축) + Ctrl+휠 줌 + Shift+드래그 팬.
// zoom 플러그인 CDN 로드 실패 시 조용히 비활성(차트 자체는 정상 동작).
function zoomOptions() {
    const available = typeof Chart !== 'undefined' && !!Chart.registry.plugins.get('zoom');
    if (!available) return {};
    return {
        zoom: {
            zoom: {
                drag: { enabled: true, backgroundColor: 'rgba(108, 140, 255, 0.18)' },
                wheel: { enabled: true, modifierKey: 'ctrl' },
                mode: 'x',
            },
            pan: { enabled: true, mode: 'x', modifierKey: 'shift' },
        },
    };
}

// 더블클릭 = 줌 리셋. addEventListener는 재렌더마다 누적되므로 프로퍼티 할당으로 교체.
function bindZoomReset(canvas, getInstance) {
    canvas.ondblclick = () => {
        const chart = getInstance();
        if (chart && typeof chart.resetZoom === 'function') chart.resetZoom();
    };
}

export function renderPriceChart(data, config) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    const theme = getChartTheme();
    priceChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: config.domesticLabel,
                    data: data.domestic_price,
                    borderColor: config.domesticColor,
                    backgroundColor: config.domesticColor + '18',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: false,
                    tension: 0.1,
                },
                {
                    label: config.intlLabel,
                    data: data.intl_price,
                    borderColor: config.intlColor,
                    backgroundColor: config.intlColor + '18',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: false,
                    tension: 0.1,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2.35,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    type: 'category',
                    grid: { color: theme.grid },
                    ticks: { maxTicksLimit: 10, maxRotation: 0, font: { size: 11 }, color: theme.textDim },
                },
                y: {
                    grid: { color: theme.grid },
                    ticks: {
                        font: { size: 11 },
                        color: theme.textDim,
                        callback: v => v >= 1e6 ? (v / 1e6).toFixed(0) + 'M' : v.toLocaleString(),
                    },
                    title: { display: true, text: config.unit, font: { size: 11 }, color: theme.textDim },
                },
            },
            plugins: {
                ...zoomOptions(),
                legend: {
                    labels: { usePointStyle: true, pointStyle: 'circle', padding: 20, font: { size: 12 }, color: theme.textDim },
                },
                tooltip: {
                    backgroundColor: theme.surface2,
                    borderColor: theme.border,
                    borderWidth: 1,
                    titleColor: theme.text,
                    bodyColor: theme.text,
                    titleFont: { size: 12 },
                    bodyFont: { family: "'SFMono-Regular', Consolas, monospace", size: 12 },
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: ctx => ctx.dataset.label + ': ' + formatPrice(ctx.parsed.y, config.unit),
                    },
                },
            },
        },
    });
    bindZoomReset(ctx.canvas, () => priceChartInstance);
}

// |gap| >= threshold 구간을 annotation box로 표시(괴리율 차트 배경 하이라이트).
function buildHighlightBoxes(data, config) {
    const boxes = {};
    let inRegion = false, regionStart = null, regionIdx = 0;
    const threshold = config.threshold;

    for (let i = 0; i < data.gap_pct.length; i++) {
        if (Math.abs(data.gap_pct[i]) >= threshold) {
            if (!inRegion) { inRegion = true; regionStart = data.dates[i]; }
        } else if (inRegion) {
            boxes['box' + regionIdx] = {
                type: 'box', xMin: regionStart, xMax: data.dates[i - 1],
                backgroundColor: config.highlightColor, borderWidth: 0,
            };
            regionIdx++;
            inRegion = false;
        }
    }
    if (inRegion) {
        boxes['box' + regionIdx] = {
            type: 'box', xMin: regionStart, xMax: data.dates[data.dates.length - 1],
            backgroundColor: config.highlightColor, borderWidth: 0,
        };
    }
    return boxes;
}

export function renderGapChart(data, config) {
    const ctx = document.getElementById('gapChart').getContext('2d');
    const theme = getChartTheme();
    const boxes = buildHighlightBoxes(data, config);
    const threshold = config.threshold;

    boxes['thresholdLine'] = {
        type: 'line', yMin: threshold, yMax: threshold,
        borderColor: theme.up, borderWidth: 1.5, borderDash: [6, 4],
        label: {
            display: true, content: '+' + threshold + '%', position: 'end',
            backgroundColor: theme.up, color: '#fff',
            font: { size: 10, family: "'SFMono-Regular', Consolas, monospace" }, padding: { x: 6, y: 3 },
        },
    };
    boxes['negThresholdLine'] = {
        type: 'line', yMin: -threshold, yMax: -threshold,
        borderColor: theme.down, borderWidth: 1.5, borderDash: [6, 4],
        label: {
            display: true, content: '-' + threshold + '%', position: 'end',
            backgroundColor: theme.down, color: '#fff',
            font: { size: 10, family: "'SFMono-Regular', Consolas, monospace" }, padding: { x: 6, y: 3 },
        },
    };
    boxes['zeroLine'] = {
        type: 'line', yMin: 0, yMax: 0,
        borderColor: theme.grid, borderWidth: 1,
    };

    const gradient = ctx.createLinearGradient(0, 0, 0, ctx.canvas.clientHeight || 300);
    gradient.addColorStop(0, config.gapColor + '30');
    gradient.addColorStop(1, config.gapColor + '02');

    gapChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [{
                label: t('chart.gapAxis'),
                data: data.gap_pct,
                borderColor: config.gapColor,
                backgroundColor: gradient,
                borderWidth: 1.5,
                pointRadius: 0,
                pointHoverRadius: 4,
                fill: true,
                tension: 0.1,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2.35,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    type: 'category',
                    grid: { color: theme.grid },
                    ticks: { maxTicksLimit: 10, maxRotation: 0, font: { size: 11 }, color: theme.textDim },
                },
                y: {
                    grid: { color: theme.grid },
                    ticks: { font: { size: 11 }, color: theme.textDim, callback: v => v.toFixed(0) + '%' },
                    title: { display: true, text: t('chart.gapAxis'), font: { size: 11 }, color: theme.textDim },
                },
            },
            plugins: {
                ...zoomOptions(),
                legend: { display: false },
                annotation: { annotations: boxes },
                tooltip: {
                    backgroundColor: theme.surface2,
                    borderColor: theme.border,
                    borderWidth: 1,
                    titleColor: theme.text,
                    bodyColor: theme.text,
                    titleFont: { size: 12 },
                    bodyFont: { family: "'SFMono-Regular', Consolas, monospace", size: 12 },
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: ctx => t('chart.gapTooltip', { value: ctx.parsed.y.toFixed(2) }),
                    },
                },
            },
        },
    });
    bindZoomReset(ctx.canvas, () => gapChartInstance);
}

// 신규 차트 공용 툴팁 스타일 — 기존 가격/괴리율 차트와 동일한 톤.
function tooltipStyle(theme) {
    return {
        backgroundColor: theme.surface2,
        borderColor: theme.border,
        borderWidth: 1,
        titleColor: theme.text,
        bodyColor: theme.text,
        titleFont: { size: 12 },
        bodyFont: { family: "'SFMono-Regular', Consolas, monospace", size: 12 },
        padding: 12,
        cornerRadius: 8,
    };
}

// 숫자의 소수 자릿수(최대 3) — 히스토그램 라벨 포맷용. bin 폭은 nice 사다리 값이라 짧다.
function decimalPlaces(value) {
    const text = String(value);
    const idx = text.indexOf('.');
    return idx === -1 ? 0 : Math.min(text.length - idx - 1, 3);
}

// 괴리율 분포 히스토그램(막대). |bin 중심| >= threshold면 자산색 진하게, 아니면 60% 알파.
// annotation 수직선으로 현재(마지막) 괴리율 위치를 표시한다.
export function renderGapHistogram(data, config) {
    const section = document.getElementById('gap-histogram-section');
    const canvas = document.getElementById('gapHistChart');
    if (!canvas) return;
    if (gapHistChartInstance) { gapHistChartInstance.destroy(); gapHistChartInstance = null; }

    const histogram = buildGapHistogram(data && data.gap_pct);
    if (!histogram) {
        if (section) section.hidden = true;
        return;
    }
    if (section) section.hidden = false;

    const theme = getChartTheme();
    const { bins, binWidth } = histogram;
    const edgeDigits = decimalPlaces(binWidth);
    const centerDigits = decimalPlaces(binWidth / 2);
    const labels = bins.map(b => ((b.x0 + b.x1) / 2).toFixed(centerDigits));
    const colors = bins.map(b => {
        const center = (b.x0 + b.x1) / 2;
        // 임계 영역(|gap| >= threshold)은 진한 자산색, 평상 영역은 60% 알파로 연하게.
        return Math.abs(center) >= config.threshold ? config.gapColor : config.gapColor + '99';
    });

    const annotations = {};
    const gaps = (data.gap_pct || []).filter(v => typeof v === 'number' && Number.isFinite(v));
    if (gaps.length) {
        const current = gaps[gaps.length - 1];
        // category 축에서 막대 i의 중심은 좌표 i — 값 current를 bin 좌표로 사상한다.
        const xPos = (current - bins[0].x0) / binWidth - 0.5;
        annotations.currentLine = {
            type: 'line',
            xMin: xPos,
            xMax: xPos,
            borderColor: theme.text,
            borderWidth: 1.5,
            borderDash: [4, 4],
            label: {
                display: true,
                content: t('chart.histCurrent', { value: (current >= 0 ? '+' : '') + current.toFixed(2) }),
                position: 'start',
                backgroundColor: theme.text,
                color: theme.surface,
                font: { size: 10, family: "'SFMono-Regular', Consolas, monospace" },
                padding: { x: 6, y: 3 },
            },
        };
    }

    const ctx = canvas.getContext('2d');
    gapHistChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: t('chart.days'),
                data: bins.map(b => b.count),
                backgroundColor: colors,
                borderColor: config.gapColor,
                borderWidth: 0,
                categoryPercentage: 1,
                barPercentage: 0.94,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2.35,
            scales: {
                x: {
                    type: 'category',
                    grid: { display: false },
                    ticks: { maxTicksLimit: 13, maxRotation: 0, font: { size: 11 }, color: theme.textDim },
                    title: { display: true, text: t('chart.gapAxis'), font: { size: 11 }, color: theme.textDim },
                },
                y: {
                    beginAtZero: true,
                    grid: { color: theme.grid },
                    ticks: { font: { size: 11 }, color: theme.textDim, precision: 0 },
                    title: { display: true, text: t('chart.days'), font: { size: 11 }, color: theme.textDim },
                },
            },
            plugins: {
                legend: { display: false },
                annotation: { annotations },
                tooltip: {
                    ...tooltipStyle(theme),
                    callbacks: {
                        title: items => {
                            const bin = bins[items[0].dataIndex];
                            return bin.x0.toFixed(edgeDigits) + '% ~ ' + bin.x1.toFixed(edgeDigits) + '%';
                        },
                        label: ctx2 => t('chart.histFreq', { count: ctx2.parsed.y }),
                    },
                },
            },
        },
    });
}

// 시장 지표 정규화 비교(선택 기간 시작점=100). ranged: {dates, [key]: values},
// seriesDefs: [{key, label, color}] — 유효 시작점이 없는 시리즈는 제외.
// 범례 클릭 토글은 Chart.js 기본 동작을 그대로 사용한다.
export function renderMarketChart(ranged, seriesDefs) {
    const canvas = document.getElementById('marketChart');
    if (!canvas) return;
    if (marketChartInstance) { marketChartInstance.destroy(); marketChartInstance = null; }
    if (!ranged || !Array.isArray(ranged.dates)) return;

    const theme = getChartTheme();
    const datasets = [];
    (seriesDefs || []).forEach(def => {
        const normalized = normalizeTo100(ranged[def.key]);
        if (!normalized) return;
        datasets.push({
            label: def.label,
            data: normalized,
            borderColor: def.color,
            backgroundColor: def.color + '18',
            borderWidth: 1.6,
            pointRadius: 0,
            pointHoverRadius: 4,
            fill: false,
            tension: 0.1,
            spanGaps: true, // 휴장일 null은 선으로 연결
        });
    });
    if (!datasets.length) return;

    const ctx = canvas.getContext('2d');
    marketChartInstance = new Chart(ctx, {
        type: 'line',
        data: { labels: ranged.dates, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2.35,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    type: 'category',
                    grid: { color: theme.grid },
                    ticks: { maxTicksLimit: 10, maxRotation: 0, font: { size: 11 }, color: theme.textDim },
                },
                y: {
                    grid: { color: theme.grid },
                    ticks: { font: { size: 11 }, color: theme.textDim },
                    title: { display: true, text: t('chart.normalizedBase'), font: { size: 11 }, color: theme.textDim },
                },
            },
            plugins: {
                ...zoomOptions(),
                legend: {
                    labels: { usePointStyle: true, pointStyle: 'circle', padding: 16, font: { size: 12 }, color: theme.textDim },
                },
                tooltip: {
                    ...tooltipStyle(theme),
                    callbacks: {
                        label: ctx2 => ctx2.dataset.label + ': ' + ctx2.parsed.y.toFixed(1),
                    },
                },
            },
        },
    });
    bindZoomReset(ctx.canvas, () => marketChartInstance);
}

// 상관계수 매트릭스 테이블 — textContent로만 조립(데이터가 마크업으로 해석되지 않게).
// 셀 배경: +1 → --up 계열 진하게, -1 → --down 계열, 0 근처 중립(혼합 비율 = |r|).
export function renderCorrelationTable(corr) {
    const table = document.getElementById('corrTable');
    if (!table) return;
    table.replaceChildren();
    if (!corr || !Array.isArray(corr.labels) || corr.labels.length === 0) {
        table.style.display = 'none';
        return;
    }
    table.style.display = '';

    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    headRow.appendChild(document.createElement('th')); // 좌상단 코너
    corr.labels.forEach(label => {
        const th = document.createElement('th');
        th.setAttribute('scope', 'col');
        th.textContent = label;
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    corr.labels.forEach((rowLabel, i) => {
        const tr = document.createElement('tr');
        const th = document.createElement('th');
        th.setAttribute('scope', 'row');
        th.textContent = rowLabel;
        tr.appendChild(th);
        corr.labels.forEach((colLabel, j) => {
            const td = document.createElement('td');
            td.className = 'corr-cell';
            const r = corr.matrix[i][j];
            const count = corr.n[i][j];
            if (r == null) {
                td.textContent = '-'; // 표본 부족(n < 20) 또는 분산 0
            } else {
                td.textContent = r.toFixed(2);
                // 혼합 최대 60%라 양 테마 모두에서 --text 글자 대비가 유지된다.
                const ratio = Math.round(Math.abs(r) * 60); // |r|=1 → 60% 혼합
                const base = r >= 0 ? 'var(--up)' : 'var(--down)';
                td.style.backgroundColor = `color-mix(in srgb, ${base} ${ratio}%, var(--surface))`;
            }
            td.title = rowLabel + ' × ' + colLabel + ' · n=' + count;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
}

export function renderTable(periods, config) {
    const tbody = document.querySelector('#gapTable tbody');
    tbody.replaceChildren();

    const noPeriods = document.getElementById('no-periods');
    const table = document.getElementById('gapTable');

    if (!periods || periods.length === 0) {
        table.style.display = 'none';
        noPeriods.textContent = t('table.noPeriods', { threshold: config.threshold });
        noPeriods.style.display = 'block';
        return;
    }

    table.style.display = '';
    noPeriods.style.display = 'none';

    [...periods].sort((a, b) => b.max_gap - a.max_gap).forEach(p => {
        const tr = document.createElement('tr');
        // innerHTML 대신 textContent로 셀 조립 — 데이터 값이 마크업으로 해석되지 않게.
        const cells = [
            { text: p.start },
            { text: p.end },
            { text: Number(p.max_gap).toFixed(2) + '%', className: 'highlight' },
            { text: t('table.days', { days: p.duration_days }) },
        ];
        cells.forEach(cell => {
            const td = document.createElement('td');
            if (cell.className) td.className = cell.className;
            td.textContent = cell.text;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
}
