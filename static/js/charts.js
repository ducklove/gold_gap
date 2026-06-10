// charts.js — Chart.js 렌더링(가격/괴리율 차트, 하이라이트 박스, 구간 테이블).
//
// 전역 Chart와 annotation 플러그인은 index.html <head>의 클래식 CDN <script>가
// 로드한다. 클래식 스크립트는 문서 파싱을 막고 즉시 실행되는 반면 ES 모듈은 항상
// defer로 실행되므로, 이 모듈이 평가되는 시점에는 Chart 전역이 보장된다.

import { formatPrice } from './format.js';

let priceChartInstance = null;
let gapChartInstance = null;

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
                label: '괴리율 (%)',
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
                    title: { display: true, text: '괴리율 (%)', font: { size: 11 }, color: theme.textDim },
                },
            },
            plugins: {
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
                        label: ctx => '괴리율: ' + ctx.parsed.y.toFixed(2) + '%',
                    },
                },
            },
        },
    });
}

export function renderTable(periods, config) {
    const tbody = document.querySelector('#gapTable tbody');
    tbody.replaceChildren();

    const noPeriods = document.getElementById('no-periods');
    const table = document.getElementById('gapTable');

    if (!periods || periods.length === 0) {
        table.style.display = 'none';
        noPeriods.textContent = config.threshold + '% 이상 괴리율 발생 구간이 없습니다.';
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
            { text: p.duration_days + '일' },
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
