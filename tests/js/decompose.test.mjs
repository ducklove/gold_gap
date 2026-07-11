// node --test tests/js — decompose.js 가격 변동 분해(로그수익률 항등식) 단위 테스트.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { findUsdSeriesKey, decomposePriceChange } from '../../static/js/decompose.js';

test('findUsdSeriesKey: gold_usd_oz / crypto_usd 인식, usd_krw 제외', () => {
    assert.equal(findUsdSeriesKey({ gold_usd_oz: [], usd_krw: [] }), 'gold_usd_oz');
    assert.equal(findUsdSeriesKey({ crypto_usd: [], usd_krw: [] }), 'crypto_usd');
    assert.equal(findUsdSeriesKey({ usd_krw: [] }), null);
});

test('findUsdSeriesKey: 배열이 아닌 값·무관 키·null 입력은 null', () => {
    assert.equal(findUsdSeriesKey({ gold_usd_oz: 123 }), null);
    assert.equal(findUsdSeriesKey({ domestic_price: [] }), null);
    assert.equal(findUsdSeriesKey(null), null);
    assert.equal(findUsdSeriesKey({}), null);
});

const DATES = ['2026-01-01', '2026-01-02', '2026-01-03'];

test('decomposePriceChange: 환율 불변이면 R_fx=0, gap은 잔차', () => {
    const out = decomposePriceChange({
        dates: DATES,
        domestic_price: [100, 110, 121],
        gold_usd_oz: [50, 55, 60.5],
        usd_krw: [1000, 1000, 1000],
    }, 0);
    assert.equal(out.startDate, '2026-01-01');
    assert.equal(out.endDate, '2026-01-03');
    assert.ok(Math.abs(out.dom - Math.log(1.21) * 100) < 1e-12);
    assert.ok(Math.abs(out.usd - Math.log(1.21) * 100) < 1e-12);
    assert.equal(out.fx, 0);
    // 항등식 R_dom = R_usd + R_fx + R_gap 은 구성상 정확히 성립
    assert.equal(out.dom, out.usd + out.fx + out.gap);
});

test('decomposePriceChange: USDT형 — usd 시세 고정이면 R_usd=0', () => {
    const out = decomposePriceChange({
        dates: DATES,
        domestic_price: [1400, 1410, 1420],
        crypto_usd: [1, 1, 1],
        usd_krw: [1390, 1395, 1400],
    }, 0);
    assert.equal(out.usd, 0);
    assert.ok(Math.abs(out.fx - Math.log(1400 / 1390) * 100) < 1e-12);
    assert.ok(Math.abs(out.dom - (out.usd + out.fx + out.gap)) < 1e-15);
});

test('decomposePriceChange: startIdx부터 첫/마지막 유효 인덱스를 찾는다', () => {
    const out = decomposePriceChange({
        dates: ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04'],
        domestic_price: [100, 200, 220, 240],
        crypto_usd: [1, 2, 2.2, 2.4],
        usd_krw: [1000, 1000, 1000, 1000],
    }, 1);
    assert.equal(out.startDate, '2026-01-02');
    assert.equal(out.endDate, '2026-01-04');
    assert.ok(Math.abs(out.dom - Math.log(240 / 200) * 100) < 1e-12);
});

test('decomposePriceChange: 양끝의 null/0/음수 관측은 건너뛴다', () => {
    const out = decomposePriceChange({
        dates: ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04'],
        domestic_price: [null, 100, 110, null],
        crypto_usd: [1, 1, 1, 1],
        usd_krw: [1000, 1000, 1000, 0], // 마지막 fx 무효
    }, 0);
    assert.equal(out.startDate, '2026-01-02');
    assert.equal(out.endDate, '2026-01-03');
});

test('decomposePriceChange: 유효 관측 쌍 2개 미만이면 null', () => {
    const one = decomposePriceChange({
        dates: DATES,
        domestic_price: [null, 100, null],
        crypto_usd: [1, 1, 1],
        usd_krw: [1000, 1000, 1000],
    }, 0);
    assert.equal(one, null);
    const none = decomposePriceChange({
        dates: DATES,
        domestic_price: [null, null, null],
        crypto_usd: [1, 1, 1],
        usd_krw: [1000, 1000, 1000],
    }, 0);
    assert.equal(none, null);
});

test('decomposePriceChange: 시리즈 누락이면 null', () => {
    assert.equal(decomposePriceChange(null, 0), null);
    assert.equal(decomposePriceChange({ dates: DATES }, 0), null); // usd 키 없음
    assert.equal(decomposePriceChange({
        dates: DATES, crypto_usd: [1, 1, 1], usd_krw: [1, 1, 1],
    }, 0), null); // domestic_price 없음
    assert.equal(decomposePriceChange({
        dates: DATES, crypto_usd: [1, 1, 1], domestic_price: [1, 1, 1],
    }, 0), null); // usd_krw 없음
});

test('decomposePriceChange: startIdx가 음수/비정수면 0으로 취급', () => {
    const base = {
        dates: DATES,
        domestic_price: [100, 110, 121],
        crypto_usd: [1, 1, 1],
        usd_krw: [1000, 1000, 1000],
    };
    assert.equal(decomposePriceChange(base, -5).startDate, '2026-01-01');
    assert.equal(decomposePriceChange(base, undefined).startDate, '2026-01-01');
    assert.equal(decomposePriceChange(base, 1.7).startDate, '2026-01-01');
});

test('decomposePriceChange: 길이가 다른 시리즈는 짧은 쪽까지만', () => {
    const out = decomposePriceChange({
        dates: ['2026-01-01', '2026-01-02', '2026-01-03'],
        domestic_price: [100, 110], // dates보다 짧음
        crypto_usd: [1, 1, 1],
        usd_krw: [1000, 1000, 1000],
    }, 0);
    assert.equal(out.endDate, '2026-01-02');
});
