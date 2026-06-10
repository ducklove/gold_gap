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
