// stats.js — 괴리율 통계 (순수 함수, DOM 의존 없음).
//
// "지금 괴리율이 역사적으로 어느 위치인가"에 답한다:
// - percentile: 전체 기간에서 현재값 이하인 날의 비율 (서명값 기준, 0~100 정수)
// - z: (현재값 − 평균) / 표준편차 (모표준편차). 분산이 0이면 null.

export function gapHistoricalStats(gaps) {
    const values = (gaps || []).filter(v => typeof v === 'number' && !Number.isNaN(v));
    if (values.length < 2) return null;

    const last = values[values.length - 1];
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
    const std = Math.sqrt(variance);
    const below = values.filter(v => v <= last).length;

    return {
        last,
        mean,
        std,
        z: std > 0 ? (last - mean) / std : null,
        percentile: Math.round((below / values.length) * 100),
        count: values.length,
    };
}

// '보기 좋은' bin 폭 사다리: 0.1/0.2/0.25/0.5 × 10^k (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, …).
// rawWidth 이상인 가장 작은 값을 고른다. 최소 폭은 0.1(괴리율 % 표시에 충분한 해상도).
const NICE_STEPS = [0.1, 0.2, 0.25, 0.5];

function niceBinWidth(rawWidth) {
    let scale = 1;
    for (let k = 0; k < 12; k++) {
        for (const step of NICE_STEPS) {
            const width = step * scale;
            if (width >= rawWidth - 1e-12) return width;
        }
        scale *= 10;
    }
    return NICE_STEPS[0] * scale;
}

// 부동소수 노이즈(0.30000000000000004 등) 제거용 — bin 폭이 0.1 이상이므로 6자리면 충분.
function cleanEdge(value) {
    return Math.round(value * 1e6) / 1e6;
}

// 괴리율 분포 히스토그램. null/NaN을 제거한 뒤 min~max를 '보기 좋은' 폭의
// bin으로 나눈다(경계는 폭의 배수에 정렬 — 0이 항상 bin 경계에 온다).
// 반환: { bins: [{x0, x1, count}], binWidth } — 마지막 bin은 x1을 포함. 빈 입력은 null.
export function buildGapHistogram(gaps, targetBins = 30) {
    const values = (gaps || []).filter(v => typeof v === 'number' && Number.isFinite(v));
    if (!values.length) return null;

    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min;
    const binWidth = niceBinWidth(span > 0 ? span / targetBins : 0);

    // 시작 경계를 binWidth 배수로 내림 정렬(+엡실론: min이 정확히 경계일 때 흔들림 방지).
    const start = Math.floor(min / binWidth + 1e-9) * binWidth;
    const binCount = Math.max(1, Math.ceil((max - start) / binWidth - 1e-9));

    const counts = new Array(binCount).fill(0);
    values.forEach(v => {
        const idx = Math.floor((v - start) / binWidth + 1e-9);
        counts[Math.min(binCount - 1, Math.max(0, idx))] += 1; // max는 마지막 bin에 포함
    });

    const bins = counts.map((count, i) => ({
        x0: cleanEdge(start + i * binWidth),
        x1: cleanEdge(start + (i + 1) * binWidth),
        count,
    }));
    return { bins, binWidth };
}

// 통계 카드 표시용 문자열 쌍 {value, sub}. 표본 부족 시 null.
export function formatHistoricalStats(stats) {
    if (!stats) return null;
    const zText = stats.z != null
        ? 'z ' + (stats.z >= 0 ? '+' : '') + stats.z.toFixed(2) + ' · '
        : '';
    return {
        value: '백분위 ' + stats.percentile + '%',
        sub: zText + '전체 ' + stats.count.toLocaleString('ko-KR') + '일 기준',
    };
}
