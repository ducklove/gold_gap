// node --test tests/js — market.js 날짜 정렬·로그수익률·피어슨 상관 단위 테스트.
// (market.js 머리말이 말하는 'node 단위 테스트'가 바로 이 파일이다.)

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
    MIN_CORR_SAMPLES,
    alignSeries,
    logReturns,
    pearson,
    buildCorrelationMatrix,
    normalizeTo100,
    latestWithChange,
    MARKET_SERIES,
    collectMarketSeries,
} from '../../static/js/market.js';

test('alignSeries: 날짜 합집합 오름차순 + 결측은 null', () => {
    const aligned = alignSeries({
        a: { dates: ['2026-01-02', '2026-01-01'], values: [2, 1] },
        b: { dates: ['2026-01-02', '2026-01-03'], values: [20, 30] },
    });
    assert.deepEqual(aligned.dates, ['2026-01-01', '2026-01-02', '2026-01-03']);
    assert.deepEqual(aligned.series.a, [1, 2, null]);
    assert.deepEqual(aligned.series.b, [null, 20, 30]);
});

test('alignSeries: 숫자가 아닌 관측은 null로 정규화', () => {
    const aligned = alignSeries({
        a: { dates: ['2026-01-01', '2026-01-02', '2026-01-03'], values: [NaN, '5', 3] },
    });
    assert.deepEqual(aligned.series.a, [null, null, 3]);
});

test('alignSeries: dates/values 길이 불일치는 짧은 쪽까지만', () => {
    const aligned = alignSeries({
        a: { dates: ['2026-01-01', '2026-01-02', '2026-01-03'], values: [1, 2] },
    });
    assert.deepEqual(aligned.dates, ['2026-01-01', '2026-01-02']);
    assert.deepEqual(aligned.series.a, [1, 2]);
});

test('alignSeries: 빈 입력/무효 시리즈는 제외', () => {
    assert.deepEqual(alignSeries({}), { dates: [], series: {} });
    assert.deepEqual(alignSeries(null), { dates: [], series: {} });
    const aligned = alignSeries({
        empty: { dates: [], values: [] },
        broken: { dates: null, values: [1] },
        ok: { dates: ['2026-01-01'], values: [1] },
    });
    assert.deepEqual(Object.keys(aligned.series), ['ok']);
});

test('logReturns: ln(v[i]/v[i-1]), 길이 = n-1', () => {
    const out = logReturns([100, 105, 105]);
    assert.equal(out.length, 2);
    assert.ok(Math.abs(out[0] - Math.log(1.05)) < 1e-12);
    assert.equal(out[1], 0);
});

test('logReturns: null/0/음수가 낀 쌍은 null(휴장 결측 안전)', () => {
    assert.deepEqual(logReturns([100, null, 110]), [null, null]);
    assert.deepEqual(logReturns([0, 100]), [null]);
    assert.deepEqual(logReturns([-1, 100]), [null]);
    assert.deepEqual(logReturns([100, 0]), [null]);
});

test('logReturns: 빈/한 점/비배열 입력은 빈 배열', () => {
    assert.deepEqual(logReturns([]), []);
    assert.deepEqual(logReturns([100]), []);
    assert.deepEqual(logReturns(null), []);
});

test('pearson: 완전 양/음의 상관 (minN 하향)', () => {
    assert.equal(pearson([1, 2, 3], [2, 4, 6], 3), 1);
    assert.equal(pearson([1, 2, 3], [3, 2, 1], 3), -1);
});

test('pearson: 알려진 중간값 r=0.6', () => {
    const r = pearson([1, 2, 3, 4], [2, 1, 4, 3], 4);
    assert.ok(Math.abs(r - 0.6) < 1e-12);
});

test('pearson: 한쪽 분산 0이면 null', () => {
    assert.equal(pearson([1, 1, 1], [1, 2, 3], 3), null);
    assert.equal(pearson([1, 2, 3], [5, 5, 5], 3), null);
});

test('pearson: 쌍별 완전 관측만 사용 — null 행은 제거', () => {
    const r = pearson([1, null, 2, 3], [2, 100, 4, 6], 3);
    assert.equal(r, 1);
});

test('pearson: 기본 minN은 MIN_CORR_SAMPLES(20)', () => {
    assert.equal(MIN_CORR_SAMPLES, 20);
    const xs19 = Array.from({ length: 19 }, (_, i) => i + 1);
    const xs20 = Array.from({ length: 20 }, (_, i) => i + 1);
    assert.equal(pearson(xs19, xs19.map(v => v * 2)), null); // 표본 19 < 20
    const r = pearson(xs20, xs20.map(v => v * 2 + 1));
    assert.ok(Math.abs(r - 1) < 1e-12); // 표본 20이면 계산
});

test('pearson: 부동소수 오차가 있어도 [-1, 1]로 클램프', () => {
    const xs = Array.from({ length: 50 }, (_, i) => 1e8 + i * 0.1);
    const r = pearson(xs, xs, 2);
    assert.ok(r <= 1 && r >= -1);
});

test('buildCorrelationMatrix: 대칭 행렬 + 대각 r=1 + 쌍별 표본 수', () => {
    const dates = ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04'];
    const { labels, matrix, n } = buildCorrelationMatrix({
        a: { dates, values: [100, 110, 105, 120] },
        b: { dates, values: [200, 220, 210, 240] }, // a의 2배 — 수익률 동일
    }, 2);
    assert.deepEqual(labels, ['a', 'b']);
    assert.ok(Math.abs(matrix[0][0] - 1) < 1e-12); // 자기 상관
    assert.ok(Math.abs(matrix[0][1] - 1) < 1e-12);
    assert.equal(matrix[0][1], matrix[1][0]); // 대칭
    assert.deepEqual(n, [[3, 3], [3, 3]]); // 수익률 3개 전부 유효
});

test('buildCorrelationMatrix: 결측 날짜로 표본 부족이면 r=null, n은 실제 쌍 수', () => {
    const { matrix, n } = buildCorrelationMatrix({
        a: { dates: ['2026-01-01', '2026-01-02'], values: [100, 110] },
        b: { dates: ['2026-01-03', '2026-01-04'], values: [200, 220] }, // 겹치는 날짜 없음
    }, 2);
    assert.equal(matrix[0][1], null);
    assert.equal(n[0][1], 0);
});

test('normalizeTo100: 첫 유효(>0) 관측 = 100 지수', () => {
    assert.deepEqual(normalizeTo100([50, 100, 75]), [100, 200, 150]);
    assert.deepEqual(normalizeTo100([null, 4, 8]), [null, 100, 200]); // 선행 null 유지
    assert.deepEqual(normalizeTo100([0, 5]), [0, 100]); // 0은 기준점 아님, 값은 환산
});

test('normalizeTo100: 유효 시작점 없으면 null(시리즈 제외 신호)', () => {
    assert.equal(normalizeTo100([null, null]), null);
    assert.equal(normalizeTo100([0, 0]), null);
    assert.equal(normalizeTo100([]), null);
    assert.equal(normalizeTo100(null), null);
});

test('latestWithChange: 마지막 유효값과 직전 유효값(휴장 null 건너뜀)', () => {
    assert.deepEqual(latestWithChange([100, null, 150]), { value: 150, prev: 100, changePct: 50 });
    const up = latestWithChange([100, null, 110]);
    assert.equal(up.value, 110);
    assert.equal(up.prev, 100);
    assert.ok(Math.abs(up.changePct - 10) < 1e-12); // 부동소수 허용
    assert.equal(latestWithChange([200, 100]).changePct, -50);
});

test('latestWithChange: 유효값 1개면 changePct null, 없으면 null', () => {
    assert.deepEqual(latestWithChange([null, 42]), { value: 42, prev: null, changePct: null });
    assert.equal(latestWithChange([null, NaN]), null);
    assert.equal(latestWithChange([]), null);
    assert.equal(latestWithChange(null), null);
});

test('latestWithChange: 직전 값이 0이면 changePct null(0 나눗셈 방지)', () => {
    assert.deepEqual(latestWithChange([0, 5]), { value: 5, prev: 0, changePct: null });
});

test('collectMarketSeries: MARKET_SERIES 순서 유지 + 소스 매핑', () => {
    const dates = ['2026-01-01', '2026-01-02'];
    const allData = {
        market: { dates, kospi: [2500, 2510], sp500: [5000, 5010], usd_krw: [1390, 1391] },
        gold: { intl_modes: { ny_futures: { dates, gold_usd_oz: [2600, 2610], usd_krw: [1389, 1390] } } },
        bitcoin: { dates, crypto_usd: [95000, 96000] },
        eth: { dates, crypto_usd: [3500, 3510] },
    };
    const series = collectMarketSeries(allData);
    assert.deepEqual(series.map(s => s.key), ['gold', 'btc', 'eth', 'usd_krw', 'kospi', 'sp500']);
    assert.deepEqual(series.map(s => s.key), MARKET_SERIES.map(d => d.key));
    const gold = series.find(s => s.key === 'gold');
    assert.deepEqual(gold.values, [2600, 2610]); // gold_usd_oz 소스
    const fx = series.find(s => s.key === 'usd_krw');
    assert.deepEqual(fx.values, [1390, 1391]); // market 블록 우선
});

test('collectMarketSeries: market.usd_krw 없으면 gold ny_futures로 폴백', () => {
    const dates = ['2026-01-01', '2026-01-02'];
    const allData = {
        market: { dates, kospi: [2500, 2510] },
        gold: { intl_modes: { ny_futures: { dates, gold_usd_oz: [2600, 2610], usd_krw: [1389, 1390] } } },
    };
    const fx = collectMarketSeries(allData).find(s => s.key === 'usd_krw');
    assert.deepEqual(fx.values, [1389, 1390]);
});

test('collectMarketSeries: 없는 시리즈·전부 결측 시리즈는 조용히 제외', () => {
    const dates = ['2026-01-01', '2026-01-02'];
    const allData = {
        market: { dates, kospi: [null, null], sp500: [5000, 5010] }, // kospi 전부 결측
        bitcoin: { dates, crypto_usd: [95000, 96000] },
    };
    const keys = collectMarketSeries(allData).map(s => s.key);
    assert.deepEqual(keys, ['btc', 'sp500']); // gold/eth/usd_krw/kospi 제외
});

test('collectMarketSeries: null/빈 입력은 빈 배열', () => {
    assert.deepEqual(collectMarketSeries(null), []);
    assert.deepEqual(collectMarketSeries({}), []);
});
