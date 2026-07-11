// node --test tests/js — periods.js 구간 탐지·조회 기간 계산 단위 테스트.
// buildHighGapPeriods 규칙은 백엔드 generate_data와 통일된 계약(tests/test_periods.py와 동일 의미).

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
    buildHighGapPeriods,
    RANGE_OPTIONS,
    DEFAULT_RANGE,
    isValidRange,
    rangeStartIndex,
    sliceDataByRange,
} from '../../static/js/periods.js';

function series(dates, gaps) {
    return { dates, gap_pct: gaps };
}

test('buildHighGapPeriods: 중간에 열리고 닫히는 단일 구간', () => {
    const data = series(
        ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04', '2026-01-05'],
        [1, 5, 7, 4.9, 1]
    );
    assert.deepEqual(buildHighGapPeriods(data, 5), [
        { start: '2026-01-02', end: '2026-01-03', max_gap: 7, duration_days: 2 },
    ]);
});

test('buildHighGapPeriods: 데이터 끝까지 이어지는 구간은 마지막 일자로 닫는다', () => {
    const data = series(['2026-01-01', '2026-01-02', '2026-01-03'], [1, 6, 8]);
    assert.deepEqual(buildHighGapPeriods(data, 5), [
        { start: '2026-01-02', end: '2026-01-03', max_gap: 8, duration_days: 2 },
    ]);
});

test('buildHighGapPeriods: 임계치 == |gap| 은 포함(>= 규칙)', () => {
    const data = series(['2026-01-01', '2026-01-02'], [5, 4.99]);
    assert.deepEqual(buildHighGapPeriods(data, 5), [
        { start: '2026-01-01', end: '2026-01-01', max_gap: 5, duration_days: 1 },
    ]);
});

test('buildHighGapPeriods: 음수 gap도 절대값으로 판정, max_gap은 부호 유지', () => {
    const data = series(
        ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04'],
        [-6, -9.129, 6, 1]
    );
    assert.deepEqual(buildHighGapPeriods(data, 5), [
        { start: '2026-01-01', end: '2026-01-03', max_gap: -9.13, duration_days: 3 },
    ]);
});

test('buildHighGapPeriods: max_gap은 소수 2자리 반올림', () => {
    const data = series(['2026-01-01', '2026-01-02'], [7.005, 1]);
    const periods = buildHighGapPeriods(data, 5);
    assert.equal(periods.length, 1);
    assert.equal(periods[0].max_gap, roundTwo(7.005));
});

function roundTwo(v) {
    return Math.round(v * 100) / 100;
}

test('buildHighGapPeriods: NaN은 건너뛰며 진행 중 구간을 닫지 않는다', () => {
    const data = series(
        ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04'],
        [6, NaN, 7, 1]
    );
    assert.deepEqual(buildHighGapPeriods(data, 5), [
        { start: '2026-01-01', end: '2026-01-03', max_gap: 7, duration_days: 3 },
    ]);
});

test('buildHighGapPeriods: 복수 구간 분리', () => {
    const data = series(
        ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04', '2026-01-05'],
        [6, 1, 1, -8, 1]
    );
    assert.deepEqual(buildHighGapPeriods(data, 5), [
        { start: '2026-01-01', end: '2026-01-01', max_gap: 6, duration_days: 1 },
        { start: '2026-01-04', end: '2026-01-04', max_gap: -8, duration_days: 1 },
    ]);
});

test('buildHighGapPeriods: 임계치 0이면 전 기간이 한 구간', () => {
    const data = series(['2026-01-01', '2026-01-03'], [0, 0.5]);
    assert.deepEqual(buildHighGapPeriods(data, 0), [
        { start: '2026-01-01', end: '2026-01-03', max_gap: 0.5, duration_days: 3 },
    ]);
});

test('buildHighGapPeriods: 초과 없음/빈 입력은 빈 배열', () => {
    assert.deepEqual(buildHighGapPeriods(series(['2026-01-01'], [1]), 5), []);
    assert.deepEqual(buildHighGapPeriods(series([], []), 5), []);
});

test('RANGE_OPTIONS/DEFAULT_RANGE 계약: key 목록과 기본값', () => {
    assert.deepEqual(RANGE_OPTIONS.map(o => o.key), ['1m', '3m', '6m', '1y', 'all']);
    assert.equal(DEFAULT_RANGE, '1y');
    assert.ok(isValidRange(DEFAULT_RANGE));
});

test('isValidRange: 유효/무효 키', () => {
    for (const key of ['1m', '3m', '6m', '1y', 'all']) assert.ok(isValidRange(key));
    assert.equal(isValidRange('2y'), false);
    assert.equal(isValidRange(''), false);
    assert.equal(isValidRange(undefined), false);
    assert.equal(isValidRange('ALL'), false); // 대소문자 구분
});

test('rangeStartIndex: 빈 배열/all/미지의 키는 0', () => {
    assert.equal(rangeStartIndex([], '1m'), 0);
    assert.equal(rangeStartIndex(null, '1m'), 0);
    assert.equal(rangeStartIndex(['2026-01-01', '2026-06-01'], 'all'), 0);
    assert.equal(rangeStartIndex(['2026-01-01', '2026-06-01'], 'nope'), 0);
});

test('rangeStartIndex: 1m 컷오프 — 3/31 기준 한 달 전은 2/28로 클램프', () => {
    const dates = ['2025-01-31', '2026-02-27', '2026-02-28', '2026-03-01', '2026-03-31'];
    assert.equal(rangeStartIndex(dates, '1m'), 2);
});

test('rangeStartIndex: 윤년이면 2/29로 클램프', () => {
    const dates = ['2024-02-28', '2024-02-29', '2024-03-31'];
    assert.equal(rangeStartIndex(dates, '1m'), 1);
});

test('rangeStartIndex: 1y 컷오프는 마지막 일자 기준', () => {
    const dates = ['2025-07-10', '2025-07-11', '2026-07-11'];
    assert.equal(rangeStartIndex(dates, '1y'), 1);
});

test('rangeStartIndex: 전체가 범위 안이면 0', () => {
    const dates = ['2026-07-01', '2026-07-05', '2026-07-11'];
    assert.equal(rangeStartIndex(dates, '1m'), 0);
    assert.equal(rangeStartIndex(dates, '1y'), 0);
});

test('sliceDataByRange: 범위 밖이면 원본 객체 그대로(identity) 반환', () => {
    const data = series(['2026-07-01', '2026-07-05'], [1, 2]);
    assert.equal(sliceDataByRange(data, '1m', 5), data);
    assert.equal(sliceDataByRange(data, 'all', 5), data);
    assert.equal(sliceDataByRange(null, '1m', 5), null);
});

test('sliceDataByRange: dates와 같은 길이 배열만 슬라이스, 나머지 필드는 유지', () => {
    const data = {
        dates: ['2026-01-01', '2026-02-01', '2026-03-01', '2026-04-01', '2026-05-01', '2026-06-01'],
        gap_pct: [10, 0, 0, 0, 5, 6],
        domestic_price: [1, 2, 3, 4, 5, 6],
        meta: 'x',
        short_arr: [1, 2],
        high_gap_periods: [{ start: 'backend', end: 'backend', max_gap: 99, duration_days: 1 }],
    };
    const sliced = sliceDataByRange(data, '1m', 4);
    assert.notEqual(sliced, data);
    assert.deepEqual(sliced.dates, ['2026-05-01', '2026-06-01']);
    assert.deepEqual(sliced.gap_pct, [5, 6]);
    assert.deepEqual(sliced.domestic_price, [5, 6]);
    assert.equal(sliced.meta, 'x');
    assert.deepEqual(sliced.short_arr, [1, 2]); // 길이 다른 배열은 그대로
    // 원본은 비파괴
    assert.equal(data.dates.length, 6);
});

test('sliceDataByRange: high_gap_periods는 슬라이스 범위에서 재계산', () => {
    const data = {
        dates: ['2026-01-01', '2026-02-01', '2026-03-01', '2026-04-01', '2026-05-01', '2026-06-01'],
        gap_pct: [10, 0, 0, 0, 5, 6],
        high_gap_periods: [{ start: 'backend', end: 'backend', max_gap: 99, duration_days: 1 }],
    };
    const sliced = sliceDataByRange(data, '1m', 4);
    assert.deepEqual(sliced.high_gap_periods, [
        { start: '2026-05-01', end: '2026-06-01', max_gap: 6, duration_days: 32 },
    ]);
});
