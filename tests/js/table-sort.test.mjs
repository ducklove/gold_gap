// node --test tests/js — table-sort.js 정렬 순수 로직 단위 테스트.
// 괴리율 구간 테이블(시작일/종료일/최대 괴리율/지속일수) 헤더 정렬의 비교자·상태 전환 계약을 고정한다.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
    PERIOD_SORT_COLUMNS,
    DEFAULT_PERIOD_SORT,
    compareValues,
    sortPeriods,
    nextSortState,
} from '../../static/js/table-sort.js';

const PERIODS = [
    { start: '2026-01-05', end: '2026-01-08', max_gap: -9.13, duration_days: 4 },
    { start: '2025-12-30', end: '2026-01-02', max_gap: 5.5, duration_days: 4 },
    { start: '2026-02-01', end: '2026-02-01', max_gap: 7.01, duration_days: 1 },
];

test('PERIOD_SORT_COLUMNS 계약: 테이블 4개 컬럼 key와 타입', () => {
    assert.deepEqual(Object.keys(PERIOD_SORT_COLUMNS), ['start', 'end', 'max_gap', 'duration_days']);
    assert.equal(PERIOD_SORT_COLUMNS.start.type, 'date');
    assert.equal(PERIOD_SORT_COLUMNS.end.type, 'date');
    assert.equal(PERIOD_SORT_COLUMNS.max_gap.type, 'number');
    assert.equal(PERIOD_SORT_COLUMNS.duration_days.type, 'number');
});

test('DEFAULT_PERIOD_SORT: 기존 표시 순서(max_gap 내림차순)와 동일', () => {
    assert.deepEqual(DEFAULT_PERIOD_SORT, { key: 'max_gap', direction: 'desc' });
});

test('compareValues(number): 대소/동등/부호', () => {
    assert.equal(compareValues(1, 2, 'number'), -1);
    assert.equal(compareValues(2, 1, 'number'), 1);
    assert.equal(compareValues(1.5, 1.5, 'number'), 0);
    assert.equal(compareValues(-9.13, 5.5, 'number'), -1); // 절대값 아닌 부호 포함 값 기준
});

test('compareValues(number): 비유한값(NaN/undefined)은 최솟값 취급', () => {
    assert.equal(compareValues(NaN, -100, 'number'), -1);
    assert.equal(compareValues(undefined, 0, 'number'), -1);
    assert.equal(compareValues(NaN, undefined, 'number'), 0);
});

test('compareValues(date): ISO 문자열 사전순 == 시간순', () => {
    assert.equal(compareValues('2025-12-30', '2026-01-05', 'date'), -1);
    assert.equal(compareValues('2026-02-01', '2026-01-05', 'date'), 1);
    assert.equal(compareValues('2026-01-05', '2026-01-05', 'date'), 0);
});

test('sortPeriods: 숫자 컬럼(max_gap) asc/desc', () => {
    const asc = sortPeriods(PERIODS, 'max_gap', 'asc').map(p => p.max_gap);
    assert.deepEqual(asc, [-9.13, 5.5, 7.01]);
    const desc = sortPeriods(PERIODS, 'max_gap', 'desc').map(p => p.max_gap);
    assert.deepEqual(desc, [7.01, 5.5, -9.13]);
});

test('sortPeriods: 숫자 컬럼(duration_days)은 수치 정렬(문자열 아님)', () => {
    const rows = [{ duration_days: 10 }, { duration_days: 2 }, { duration_days: 33 }];
    assert.deepEqual(
        sortPeriods(rows, 'duration_days', 'asc').map(r => r.duration_days),
        [2, 10, 33]
    );
});

test('sortPeriods: 날짜 컬럼(start) asc/desc', () => {
    const asc = sortPeriods(PERIODS, 'start', 'asc').map(p => p.start);
    assert.deepEqual(asc, ['2025-12-30', '2026-01-05', '2026-02-01']);
    const desc = sortPeriods(PERIODS, 'start', 'desc').map(p => p.start);
    assert.deepEqual(desc, ['2026-02-01', '2026-01-05', '2025-12-30']);
});

test('sortPeriods: 동률은 원래 순서 유지(안정 정렬)', () => {
    const sorted = sortPeriods(PERIODS, 'duration_days', 'desc');
    // duration_days 4가 둘 — 입력 순서(1월 구간 → 12월 구간) 유지.
    assert.deepEqual(sorted.map(p => p.start), ['2026-01-05', '2025-12-30', '2026-02-01']);
});

test('sortPeriods: 원본 비파괴 + 미지의 key는 순서 유지 복사본', () => {
    const before = PERIODS.map(p => ({ ...p }));
    const out = sortPeriods(PERIODS, 'nope', 'asc');
    assert.notEqual(out, PERIODS);
    assert.deepEqual(PERIODS, before);
    assert.deepEqual(out, PERIODS);
});

test('sortPeriods: 배열 아닌 입력은 빈 배열', () => {
    assert.deepEqual(sortPeriods(null, 'start', 'asc'), []);
    assert.deepEqual(sortPeriods(undefined, 'max_gap', 'desc'), []);
});

test('nextSortState: 같은 컬럼 재선택 = 방향 토글(왕복)', () => {
    let state = { key: 'max_gap', direction: 'desc' };
    state = nextSortState(state, 'max_gap');
    assert.deepEqual(state, { key: 'max_gap', direction: 'asc' });
    state = nextSortState(state, 'max_gap');
    assert.deepEqual(state, { key: 'max_gap', direction: 'desc' });
});

test('nextSortState: 다른 컬럼 선택 = 그 컬럼의 기본 방향', () => {
    assert.deepEqual(
        nextSortState({ key: 'max_gap', direction: 'asc' }, 'start'),
        { key: 'start', direction: 'asc' } // 날짜 컬럼 기본 오름차순
    );
    assert.deepEqual(
        nextSortState({ key: 'start', direction: 'desc' }, 'duration_days'),
        { key: 'duration_days', direction: 'desc' } // 숫자 컬럼 기본 내림차순
    );
});

test('nextSortState: 미지의 key는 현재 상태 그대로', () => {
    const state = { key: 'max_gap', direction: 'desc' };
    assert.equal(nextSortState(state, 'nope'), state);
});
