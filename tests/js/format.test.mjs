// node --test tests/js — format.js 순수 포맷터 단위 테스트.
// static/js는 ES 모듈이라 직접 import 한다(루트 package.json "type":"module").
// 통화 접미사('원'/'KRW')는 i18n 언어를 따르므로 setLang으로 양쪽을 검증한다.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
    roundNumber,
    latestValue,
    getKstDateString,
    getKstTimeString,
    formatPrice,
    formatKrw,
    formatUsd,
    formatAssetKrw,
} from '../../static/js/format.js';
import { setLang } from '../../static/js/i18n.js';

// 언어 상태를 바꾸는 테스트용 헬퍼 — 실패해도 반드시 ko로 복원.
function withLang(lang, fn) {
    setLang(lang);
    try {
        fn();
    } finally {
        setLang('ko');
    }
}

test('roundNumber: 기본 2자리 반올림', () => {
    assert.equal(roundNumber(1.234), 1.23);
    assert.equal(roundNumber(1.236), 1.24);
    assert.equal(roundNumber(-1.234), -1.23);
    assert.equal(roundNumber(5), 5);
    assert.equal(roundNumber(0), 0);
});

test('roundNumber: digits 인자', () => {
    assert.equal(roundNumber(1.2345, 3), 1.235);
    assert.equal(roundNumber(2.5, 0), 3); // Math.round 규칙(양수 .5는 올림)
    assert.equal(roundNumber(1234.5678, 1), 1234.6);
});

test('roundNumber: null/NaN/비수치 입력은 null', () => {
    assert.equal(roundNumber(null), null);
    assert.equal(roundNumber(undefined), null);
    assert.equal(roundNumber(NaN), null);
    assert.equal(roundNumber('abc'), null);
});

test('roundNumber: 숫자 문자열은 숫자로 변환 후 반올림', () => {
    assert.equal(roundNumber('1.234'), 1.23);
    assert.equal(roundNumber('-3.999'), -4);
});

test('latestValue: 마지막 값 / 빈·누락 시리즈는 null', () => {
    assert.equal(latestValue({ a: [1, 2, 3] }, 'a'), 3);
    assert.equal(latestValue({ a: [7] }, 'a'), 7);
    assert.equal(latestValue({ a: [] }, 'a'), null);
    assert.equal(latestValue({ a: [1] }, 'b'), null);
    assert.equal(latestValue(null, 'a'), null);
});

test('getKstDateString: UTC→KST(+9) 날짜 경계', () => {
    // UTC 20:00 = KST 다음날 05:00
    assert.equal(getKstDateString(new Date('2026-01-01T20:00:00Z')), '2026-01-02');
    // UTC 14:59 = KST 같은 날 23:59
    assert.equal(getKstDateString(new Date('2026-07-11T14:59:00Z')), '2026-07-11');
    // UTC 15:00 = KST 다음날 00:00
    assert.equal(getKstDateString(new Date('2026-07-11T15:00:00Z')), '2026-07-12');
});

test('getKstTimeString: ko 모드는 ko-KR 표기(연월일 + 시:분)', () => {
    const text = getKstTimeString(new Date('2026-01-02T05:07:00+09:00'));
    assert.match(text, /2026\.\s*01\.\s*02\./);
    assert.match(text, /05:07/);
    assert.doesNotMatch(text, /\.$/); // 말미 마침표 제거 규약
});

test('getKstTimeString: en 모드는 en-CA 24시간 표기', () => {
    withLang('en', () => {
        const text = getKstTimeString(new Date('2026-01-02T05:07:00+09:00'));
        assert.match(text, /2026-01-02/);
        assert.match(text, /05:07/);
    });
});

test('formatPrice: 천 단위 콤마 + 단위별 접미사(ko)', () => {
    assert.equal(formatPrice(1234567, 'KRW'), '1,234,567 원');
    assert.equal(formatPrice(4321, 'KRW/g'), '4,321 원/g');
    assert.equal(formatPrice(0, 'KRW'), '0 원');
});

test('formatPrice: en 모드는 KRW 접미사', () => {
    withLang('en', () => {
        assert.equal(formatPrice(1234567, 'KRW'), '1,234,567 KRW');
        assert.equal(formatPrice(4321, 'KRW/g'), '4,321 KRW/g');
    });
});

test('formatKrw: 콤마 + " 원", null/NaN은 "-"', () => {
    assert.equal(formatKrw(1234567), '1,234,567 원');
    assert.equal(formatKrw(0), '0 원');
    assert.equal(formatKrw(null), '-');
    assert.equal(formatKrw(undefined), '-');
    assert.equal(formatKrw(NaN), '-');
    assert.equal(formatKrw('abc'), '-');
});

test('formatKrw: 숫자 문자열 허용 + toLocaleString 옵션 전달', () => {
    assert.equal(formatKrw('1234'), '1,234 원');
    assert.equal(
        formatKrw(1234.567, { maximumFractionDigits: 0 }),
        '1,235 원'
    );
    assert.equal(
        formatKrw(1393, { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
        '1,393.00 원'
    );
});

test('formatKrw: en 모드는 " KRW" 접미사', () => {
    withLang('en', () => {
        assert.equal(formatKrw(1234567), '1,234,567 KRW');
        assert.equal(formatKrw(null), '-');
    });
});

test('formatUsd: $ 접두사 + en-US 콤마, null/NaN은 "-"', () => {
    assert.equal(formatUsd(1234.5, { minimumFractionDigits: 2, maximumFractionDigits: 2 }), '$1,234.50');
    assert.equal(formatUsd(0), '$0');
    assert.equal(formatUsd(1000000), '$1,000,000');
    assert.equal(formatUsd(null), '-');
    assert.equal(formatUsd(NaN), '-');
});

test('formatAssetKrw: 소수 자리 지정(USDT형) vs 정수(금/BTC형)', () => {
    assert.equal(formatAssetKrw(1393, 2), '1,393.00 원'); // krwFractionDigits=2 고정
    assert.equal(formatAssetKrw(1393.456, 2), '1,393.46 원');
    assert.equal(formatAssetKrw(1234567.6), '1,234,568 원'); // 미지정 → 정수 반올림
    assert.equal(formatAssetKrw(1234567.4, undefined), '1,234,567 원');
});

test('formatAssetKrw: null/NaN은 "-"', () => {
    assert.equal(formatAssetKrw(null, 2), '-');
    assert.equal(formatAssetKrw(NaN), '-');
    assert.equal(formatAssetKrw(undefined), '-');
});
