// node --test tests/js — stats.js 괴리율 통계·히스토그램 단위 테스트.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
    gapHistoricalStats,
    buildGapHistogram,
    formatHistoricalStats,
} from '../../static/js/stats.js';
import { setLang } from '../../static/js/i18n.js';

test('gapHistoricalStats: 표본 2개 미만이면 null', () => {
    assert.equal(gapHistoricalStats([]), null);
    assert.equal(gapHistoricalStats(null), null);
    assert.equal(gapHistoricalStats([1]), null);
    assert.equal(gapHistoricalStats([1, NaN]), null); // NaN 제거 후 1개
    assert.equal(gapHistoricalStats(['1', '2']), null); // 문자열은 표본 아님
});

test('gapHistoricalStats: [1..5] 알려진 값 검증', () => {
    const stats = gapHistoricalStats([1, 2, 3, 4, 5]);
    assert.equal(stats.last, 5);
    assert.equal(stats.mean, 3);
    assert.equal(stats.std, Math.sqrt(2)); // 모표준편차
    assert.ok(Math.abs(stats.z - 2 / Math.sqrt(2)) < 1e-12);
    assert.equal(stats.percentile, 100); // 전부 last 이하
    assert.equal(stats.count, 5);
});

test('gapHistoricalStats: 현재값이 최저면 백분위는 이하 비율 기준', () => {
    const stats = gapHistoricalStats([5, 4, 3, 2, 1]);
    assert.equal(stats.last, 1);
    assert.equal(stats.percentile, 20); // 1 이하는 본인 1개 = 1/5
    assert.ok(stats.z < 0);
});

test('gapHistoricalStats: 분산 0이면 z는 null, 백분위 100', () => {
    const stats = gapHistoricalStats([2, 2, 2]);
    assert.equal(stats.std, 0);
    assert.equal(stats.z, null);
    assert.equal(stats.percentile, 100);
});

test('gapHistoricalStats: null/NaN 혼입은 걸러내고 계산', () => {
    const stats = gapHistoricalStats([null, 1, NaN, 3, undefined, 5]);
    assert.equal(stats.count, 3);
    assert.equal(stats.mean, 3);
    assert.equal(stats.last, 5);
});

test('gapHistoricalStats: 음수 시리즈(역프리미엄)도 서명값 기준', () => {
    const stats = gapHistoricalStats([-1, -2, -3]);
    assert.equal(stats.last, -3);
    assert.equal(stats.mean, -2);
    assert.equal(stats.percentile, Math.round((1 / 3) * 100));
});

test('buildGapHistogram: 빈/전부 무효 입력은 null', () => {
    assert.equal(buildGapHistogram([]), null);
    assert.equal(buildGapHistogram(null), null);
    assert.equal(buildGapHistogram([NaN, null, Infinity]), null); // Infinity도 제외(isFinite)
});

test('buildGapHistogram: 단일 값 → bin 1개, 최소 폭 0.1', () => {
    const hist = buildGapHistogram([2.34]);
    assert.equal(hist.binWidth, 0.1);
    assert.equal(hist.bins.length, 1);
    assert.equal(hist.bins[0].count, 1);
    assert.ok(hist.bins[0].x0 <= 2.34 && 2.34 <= hist.bins[0].x1);
});

test('buildGapHistogram: span 3, targetBins 30 → 폭 0.1', () => {
    const hist = buildGapHistogram([0, 3], 30);
    assert.equal(hist.binWidth, 0.1);
    assert.equal(hist.bins.length, 30);
});

test('buildGapHistogram: 보기 좋은 폭 사다리(0.1/0.2/0.25/0.5×10^k)', () => {
    // span 4.5 / 30 = 0.15 → 0.2
    assert.equal(buildGapHistogram([0, 4.5], 30).binWidth, 0.2);
    // span 100 / 30 = 3.33... → 5
    assert.equal(buildGapHistogram([0, 100], 30).binWidth, 5);
    // span 30 / 30 = 1 → 1
    assert.equal(buildGapHistogram([0, 30], 30).binWidth, 1);
});

test('buildGapHistogram: 경계는 폭의 배수 정렬 — 0이 항상 bin 경계', () => {
    const hist = buildGapHistogram([-0.47, 0.73], 30); // 폭 0.1
    assert.equal(hist.binWidth, 0.1);
    assert.equal(hist.bins[0].x0, -0.5); // -0.47을 폭 배수로 내림
    const edges = hist.bins.map(b => b.x0).concat(hist.bins[hist.bins.length - 1].x1);
    assert.ok(edges.includes(0));
});

test('buildGapHistogram: 부동소수 노이즈 없는 경계(cleanEdge)', () => {
    const hist = buildGapHistogram([0, 0.3], 30);
    assert.equal(hist.bins.length, 3);
    assert.equal(hist.bins[2].x1, 0.3); // 0.30000000000000004가 아니라 0.3
});

test('buildGapHistogram: 최대값은 마지막 bin에 포함, 총 카운트 = 표본 수', () => {
    const values = [0, 0.05, 0.15, 0.95, 1];
    const hist = buildGapHistogram(values, 10);
    const total = hist.bins.reduce((a, b) => a + b.count, 0);
    assert.equal(total, values.length);
    assert.equal(hist.bins[hist.bins.length - 1].count, 2); // 0.95와 1(최대값 클램프)
});

test('formatHistoricalStats: null이면 null', () => {
    assert.equal(formatHistoricalStats(null), null);
});

test('formatHistoricalStats: ko — 백분위/z부호/콤마 일수', () => {
    const out = formatHistoricalStats({ z: 1.234, percentile: 85, count: 1234 });
    assert.equal(out.value, '백분위 85%');
    assert.equal(out.sub, 'z +1.23 · 전체 1,234일 기준');
});

test('formatHistoricalStats: 음수 z는 - 부호 그대로', () => {
    const out = formatHistoricalStats({ z: -0.5, percentile: 10, count: 30 });
    assert.equal(out.sub, 'z -0.50 · 전체 30일 기준');
});

test('formatHistoricalStats: z가 null이면 z 부분 생략', () => {
    const out = formatHistoricalStats({ z: null, percentile: 100, count: 3 });
    assert.equal(out.value, '백분위 100%');
    assert.equal(out.sub, '전체 3일 기준');
});

test('formatHistoricalStats: en — 번역 문자열 + en-US 콤마', () => {
    setLang('en');
    try {
        const out = formatHistoricalStats({ z: 0, percentile: 42, count: 1234 });
        assert.equal(out.value, 'Percentile 42%');
        assert.equal(out.sub, 'z +0.00 · based on all 1,234 days');
    } finally {
        setLang('ko');
    }
});
