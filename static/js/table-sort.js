// table-sort.js — 괴리율 구간 테이블 정렬 순수 로직. DOM에 의존하지 않는다
// (헤더 바인딩·aria-sort 반영은 charts.js가 담당).
//
// 계약:
//  - PERIOD_SORT_COLUMNS: 컬럼 key → { type, defaultDirection }.
//    date 컬럼은 ISO(YYYY-MM-DD) 문자열이라 사전순 비교가 곧 시간순이다.
//  - nextSortState: 같은 컬럼 재선택 = 방향 토글, 다른 컬럼 = 그 컬럼의 기본 방향.
//  - sortPeriods: 원본 비파괴(복사 후 정렬). 기본 상태(max_gap desc)는 기존
//    renderTable의 '최대 괴리율 내림차순' 표시 순서와 동일하다.

export const PERIOD_SORT_COLUMNS = {
    start: { type: 'date', defaultDirection: 'asc' },
    end: { type: 'date', defaultDirection: 'asc' },
    max_gap: { type: 'number', defaultDirection: 'desc' },
    duration_days: { type: 'number', defaultDirection: 'desc' },
};

export const DEFAULT_PERIOD_SORT = { key: 'max_gap', direction: 'desc' };

// 숫자 강제 변환 — 비유한값(NaN/undefined 등)은 -Infinity로 취급해 항상 최솟값.
function toComparableNumber(value) {
    const num = Number(value);
    return Number.isFinite(num) ? num : -Infinity;
}

// 타입별 오름차순 비교자(-1/0/1). date/그 외는 문자열 사전순(ISO 날짜 == 시간순).
export function compareValues(a, b, type) {
    if (type === 'number') {
        const na = toComparableNumber(a);
        const nb = toComparableNumber(b);
        return na === nb ? 0 : (na < nb ? -1 : 1);
    }
    const sa = String(a);
    const sb = String(b);
    return sa === sb ? 0 : (sa < sb ? -1 : 1);
}

// key/direction에 따라 정렬된 새 배열을 반환. 미지의 key는 순서 유지 복사본.
export function sortPeriods(periods, key, direction, columns = PERIOD_SORT_COLUMNS) {
    const list = Array.isArray(periods) ? [...periods] : [];
    const column = columns[key];
    if (!column) return list;
    const sign = direction === 'asc' ? 1 : -1;
    return list.sort((a, b) => sign * compareValues(a && a[key], b && b[key], column.type));
}

// 헤더 클릭 시 다음 정렬 상태. 미지의 key는 현재 상태 유지.
export function nextSortState(state, key, columns = PERIOD_SORT_COLUMNS) {
    const column = columns[key];
    if (!column) return state;
    if (state && state.key === key) {
        return { key, direction: state.direction === 'asc' ? 'desc' : 'asc' };
    }
    return { key, direction: column.defaultDirection };
}
