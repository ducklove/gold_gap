// periods.js — 괴리율 구간/조회 기간 계산. 순수 함수만 두며 DOM에 의존하지 않는다.

import { roundNumber } from './format.js';

// 임계치(|gap| >= threshold) 초과 구간 계산.
// 규칙(백엔드 generate_data와 통일된 계약 — 변경 금지):
//  - start = 임계치를 처음 초과한 데이터 일자
//  - end   = 임계치를 마지막으로 초과한 데이터 일자(초과가 끝난 다음 날이 아님)
//  - max_gap = 구간 내 |gap| 최대값의 부호 포함 원본 값 (소수 2자리 반올림)
//  - duration_days = (end - start).days + 1, 최소 1
export function buildHighGapPeriods(data, threshold) {
    const periods = [];
    let inPeriod = false;
    let start = '';
    let maxGap = 0;

    data.dates.forEach((date, idx) => {
        const gap = Number(data.gap_pct[idx]);
        if (Number.isNaN(gap)) return;
        if (Math.abs(gap) >= threshold) {
            if (!inPeriod) {
                inPeriod = true;
                start = date;
                maxGap = gap;
            } else if (Math.abs(gap) > Math.abs(maxGap)) {
                maxGap = gap;
            }
        } else if (inPeriod) {
            const prevDate = data.dates[Math.max(idx - 1, 0)];
            periods.push({
                start,
                end: prevDate,
                max_gap: roundNumber(maxGap, 2),
                duration_days: Math.max(1, Math.round((new Date(prevDate) - new Date(start)) / 86400000) + 1),
            });
            inPeriod = false;
        }
    });

    if (inPeriod && data.dates.length) {
        const end = data.dates[data.dates.length - 1];
        periods.push({
            start,
            end,
            max_gap: roundNumber(maxGap, 2),
            duration_days: Math.max(1, Math.round((new Date(end) - new Date(start)) / 86400000) + 1),
        });
    }
    return periods;
}

// 조회 기간 토글 옵션. key는 URL 파라미터 ?range= 값으로도 사용된다.
export const RANGE_OPTIONS = [
    { key: '1m', label: '1M', months: 1 },
    { key: '3m', label: '3M', months: 3 },
    { key: '6m', label: '6M', months: 6 },
    { key: '1y', label: '1Y', months: 12 },
    { key: 'all', label: '전체', months: null },
];

export const DEFAULT_RANGE = '1y';

export function isValidRange(key) {
    return RANGE_OPTIONS.some(option => option.key === key);
}

// ISO 날짜 문자열에서 months개월 전 날짜를 반환(달력 기준, 말일은 해당 월 말일로 클램프).
function shiftIsoMonths(iso, months) {
    const [y, m, d] = String(iso).split('-').map(Number);
    if (!y || !m || !d) return iso;
    const shifted = new Date(Date.UTC(y, m - 1 - months, 1));
    const daysInMonth = new Date(Date.UTC(shifted.getUTCFullYear(), shifted.getUTCMonth() + 1, 0)).getUTCDate();
    shifted.setUTCDate(Math.min(d, daysInMonth));
    return shifted.toISOString().slice(0, 10);
}

// 오름차순 ISO 날짜 배열에서 rangeKey 범위의 시작 인덱스를 반환(마지막 일자 기준).
// 전체 범위('all'/빈 배열)는 0, 모든 데이터가 범위 밖이면 최신 1점만 남도록 length-1.
// sliceDataByRange와 시장 섹션(main.js)이 같은 컷오프 규칙을 공유하기 위한 유틸.
export function rangeStartIndex(dates, rangeKey) {
    if (!Array.isArray(dates) || dates.length === 0) return 0;
    const option = RANGE_OPTIONS.find(o => o.key === rangeKey);
    const months = option && option.months;
    if (!months) return 0;
    const cutoff = shiftIsoMonths(dates[dates.length - 1], months);
    const startIdx = dates.findIndex(date => date >= cutoff);
    return startIdx === -1 ? dates.length - 1 : startIdx;
}

// 시리즈를 마지막 데이터 일자 기준 rangeKey 범위로 슬라이스한 새 객체를 반환.
// - dates와 길이가 같은 모든 배열 시리즈를 함께 슬라이스
// - high_gap_periods는 슬라이스된 범위에서 buildHighGapPeriods로 재계산
// - 'all'(또는 전체가 범위 안)이면 원본 객체를 그대로 반환(백엔드 계산 high_gap_periods 유지)
export function sliceDataByRange(data, rangeKey, threshold) {
    if (!data || !Array.isArray(data.dates) || data.dates.length === 0) return data;
    const startIdx = rangeStartIndex(data.dates, rangeKey);
    if (startIdx <= 0) return data;

    const total = data.dates.length;
    const sliced = { ...data };
    Object.keys(data).forEach(key => {
        if (key === 'high_gap_periods') return;
        const value = data[key];
        if (Array.isArray(value) && value.length === total) {
            sliced[key] = value.slice(startIdx);
        }
    });
    sliced.high_gap_periods = buildHighGapPeriods(sliced, threshold);
    return sliced;
}
