// node --test tests/js — i18n.js 번역 레이어 단위 테스트.
// i18n.js 머리말의 "ko/en 키 집합은 항상 동일해야 한다(검증 코드가 보장)"를
// 보장하는 검증 코드가 이 파일이다 — 키 집합·플레이스홀더 패리티를 고정한다.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
    STRINGS,
    getLang,
    setLang,
    t,
    localizeAssetConfig,
    applyStaticStrings,
} from '../../static/js/i18n.js';

function placeholders(text) {
    return [...String(text).matchAll(/\{(\w+)\}/g)].map(m => m[1]).sort();
}

test('STRINGS: ko/en 키 집합이 정확히 동일하다', () => {
    const koKeys = Object.keys(STRINGS.ko).sort();
    const enKeys = Object.keys(STRINGS.en).sort();
    assert.deepEqual(enKeys, koKeys);
});

test('STRINGS: 모든 값은 비어 있지 않은 문자열', () => {
    for (const lang of ['ko', 'en']) {
        for (const [key, value] of Object.entries(STRINGS[lang])) {
            assert.equal(typeof value, 'string', `${lang}.${key}`);
            assert.ok(value.length > 0, `${lang}.${key} 빈 문자열`);
        }
    }
});

test('STRINGS: 키별 {param} 플레이스홀더 집합이 ko/en에서 동일하다', () => {
    for (const key of Object.keys(STRINGS.ko)) {
        assert.deepEqual(
            placeholders(STRINGS.en[key]),
            placeholders(STRINGS.ko[key]),
            `플레이스홀더 불일치: ${key}`
        );
    }
});

test('getLang: 노드(window 없음)에서는 기본 ko', () => {
    assert.equal(getLang(), 'ko');
});

test('setLang: en/ko 왕복 + 무효 언어는 ko로 폴백', () => {
    try {
        assert.equal(setLang('en'), 'en');
        assert.equal(getLang(), 'en');
        assert.equal(setLang('fr'), 'ko'); // 무효 → ko
        assert.equal(getLang(), 'ko');
        assert.equal(setLang('ko'), 'ko');
    } finally {
        setLang('ko');
    }
});

test('t: 단일/복수 파라미터 치환', () => {
    assert.equal(t('card.highGap', { threshold: 3 }), '3%+ 발생');
    assert.equal(t('error.load', { message: 'boom' }), '데이터 로딩 실패: boom');
});

test('t: 파라미터 값 0/빈 문자열도 치환(null/undefined만 원문 유지)', () => {
    assert.equal(t('card.highGap', { threshold: 0 }), '0%+ 발생');
    assert.equal(t('error.load', { message: '' }), '데이터 로딩 실패: ');
    assert.equal(t('card.highGap', { threshold: null }), '{threshold}%+ 발생');
    assert.equal(t('card.highGap', {}), '{threshold}%+ 발생');
});

test('t: 없는 키는 키 자체를 반환', () => {
    assert.equal(t('no.such.key'), 'no.such.key');
    assert.equal(t('no.such.key', { a: 1 }), 'no.such.key');
});

test('t: en 모드 조회 + en에 없는 키는 ko로 폴백', () => {
    STRINGS.ko['__test.koOnly'] = '한국어만';
    try {
        setLang('en');
        assert.equal(t('card.highGap', { threshold: 3 }), '3%+ periods');
        assert.equal(t('__test.koOnly'), '한국어만'); // en 사전에 없음 → ko 폴백
    } finally {
        delete STRINGS.ko['__test.koOnly'];
        setLang('ko');
    }
});

const GOLD_CONFIG = {
    key: 'gold',
    summary: 'KRX 금과 COMEX 금을 비교합니다.',
    domesticLabel: '국내 (KRX 금현물)',
    intlLabel: 'NY 선물 GC=F (KRW 환산)',
    sourceSummary: '국내: ACE KRX 금현물',
    intlModes: {
        ny_futures: { label: 'NY 선물', intlLabel: 'NY 선물 (KRW 환산)', cardLabel: 'NY 선물 ($/oz)' },
        custom_mode: { label: '커스텀', intlLabel: '커스텀 환산' },
    },
};

test('localizeAssetConfig: ko 모드에서는 원본 그대로(identity)', () => {
    assert.equal(localizeAssetConfig(GOLD_CONFIG, 'ko'), GOLD_CONFIG);
    assert.equal(localizeAssetConfig(null, 'en'), null);
});

test('localizeAssetConfig: en 모드 — 사전에 있는 자산은 표시 문구 덮어쓰기', () => {
    const out = localizeAssetConfig(GOLD_CONFIG, 'en');
    assert.notEqual(out, GOLD_CONFIG); // 사본
    assert.match(out.summary, /COMEX/);
    assert.notEqual(out.summary, GOLD_CONFIG.summary);
    assert.match(out.domesticLabel, /KRX/);
    assert.equal(out.key, 'gold'); // 비표시 필드 유지
    // 원본은 비파괴
    assert.equal(GOLD_CONFIG.summary, 'KRX 금과 COMEX 금을 비교합니다.');
});

test('localizeAssetConfig: intlModes — 사전에 있는 모드만 번역, 없는 모드는 원본 유지', () => {
    const out = localizeAssetConfig(GOLD_CONFIG, 'en');
    assert.equal(out.intlModes.ny_futures.label, 'NY futures');
    assert.equal(out.intlModes.ny_futures.cardLabel, 'NY futures ($/oz)');
    assert.equal(out.intlModes.custom_mode, GOLD_CONFIG.intlModes.custom_mode); // 사전 없음 → 그대로
});

test('localizeAssetConfig: 사전에 없는 신규 자산은 meta 원문(한국어) 폴백', () => {
    const config = { key: 'silver', summary: '은 괴리율' };
    assert.equal(localizeAssetConfig(config, 'en'), config);
});

test('applyStaticStrings: DOM 없는 환경(null/무효 root)에서 예외 없이 무시', () => {
    assert.doesNotThrow(() => applyStaticStrings(null));
    assert.doesNotThrow(() => applyStaticStrings({}));
    assert.doesNotThrow(() => applyStaticStrings()); // 노드: document 없음 → null 폴백
});
