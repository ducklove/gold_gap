// node --test tests/js — ecosystem.js(Value Compass 셸 연동) 행위 테스트.
// ① 헬퍼 단위: 셸 유무에 따른 테마 토글 경로, vc:themechange 구독, 형제 링크 URL.
// ② 통합: 벤더링된 static/vc-shell.js를 vm 샌드박스(최소 window/document 스텁)에서 실제로 실행해
//    requestTheme → VCShell.setTheme → 공용 'theme' 키 저장 → 'vc:themechange' → 재렌더 콜백,
//    그리고 VCShell.linkTo('all-about-gold')가 ?theme·from을 붙이는지 확인한다.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

import {
    getShell,
    requestTheme,
    onShellThemeChange,
    siblingUrl,
    bindSiblingLink,
    ALL_ABOUT_GOLD_ID,
    ALL_ABOUT_GOLD_URL,
} from '../../static/js/ecosystem.js';

// ---- ① 단위 ---------------------------------------------------------------

test('getShell: setTheme을 갖춘 VCShell만 인정', () => {
    assert.equal(getShell({}), null);
    assert.equal(getShell({ VCShell: {} }), null);
    const shell = { setTheme() {} };
    assert.equal(getShell({ VCShell: shell }), shell);
    assert.equal(getShell(undefined), null);
});

test('requestTheme: 셸이 있으면 셸로, 없거나 실패하면 기존 로직(fallback)', () => {
    const calls = [];
    const shell = { setTheme: t => calls.push(['shell', t]) };
    const fallback = t => calls.push(['fallback', t]);
    assert.equal(requestTheme('dark', { shell, fallback }), 'shell');
    assert.equal(requestTheme('light', { shell: null, fallback }), 'fallback');
    const broken = { setTheme() { throw new Error('boom'); } };
    assert.equal(requestTheme('dark', { shell: broken, fallback }), 'fallback');
    assert.equal(requestTheme('bogus', { shell: null, fallback }), 'fallback');
    assert.deepEqual(calls, [['shell', 'dark'], ['fallback', 'light'], ['fallback', 'dark'], ['fallback', 'light']]);
});

test('onShellThemeChange: 현재 테마와 다를 때만 콜백, 해제 가능', () => {
    const target = new EventTarget();
    let current = 'light';
    const seen = [];
    const off = onShellThemeChange(target, () => current, t => { seen.push(t); current = t; });
    const fire = theme => target.dispatchEvent(new CustomEvent('vc:themechange', { detail: { theme } }));
    fire('light');        // 같은 테마 — 무시(중복 재렌더 방지)
    fire('dark');
    fire('dark');         // 이미 반영됨 — 무시
    fire('sepia');        // 잘못된 값 — 무시
    off();
    fire('light');        // 해제 후 — 무시
    assert.deepEqual(seen, ['dark']);
    assert.equal(typeof onShellThemeChange(null, () => 'light', () => {}), 'function');
});

test('siblingUrl: 셸 linkTo 우선, 폴백은 ?theme·from 부착', () => {
    const shell = { linkTo: (id, vars) => `https://x.test/${id}?v=${JSON.stringify(vars)}` };
    assert.equal(siblingUrl(ALL_ABOUT_GOLD_ID, ALL_ABOUT_GOLD_URL, { shell }), 'https://x.test/all-about-gold?v={}');
    assert.equal(
        siblingUrl(ALL_ABOUT_GOLD_ID, ALL_ABOUT_GOLD_URL, { shell: null, theme: 'dark' }),
        'https://ducklove.github.io/all-about-gold/?theme=dark&from=gold_gap',
    );
    // 셸이 null을 돌려주면(미등록 id) 폴백
    assert.equal(
        siblingUrl('nope', ALL_ABOUT_GOLD_URL, { shell: { linkTo: () => null }, theme: 'bogus' }),
        'https://ducklove.github.io/all-about-gold/?from=gold_gap',
    );
    assert.equal(siblingUrl('x', 'not a url', {}), 'not a url');
});

function fakeAnchor(href) {
    const target = new EventTarget();
    const attrs = { href };
    target.getAttribute = name => attrs[name] ?? null;
    target.setAttribute = (name, value) => { attrs[name] = String(value); };
    return target;
}

test('bindSiblingLink: 바인딩 즉시·클릭 직전 현재 테마로 href 갱신', () => {
    const a = fakeAnchor(ALL_ABOUT_GOLD_URL);
    let theme = 'light';
    bindSiblingLink(a, ALL_ABOUT_GOLD_ID, { getShell: () => null, getTheme: () => theme });
    assert.equal(a.getAttribute('href'), 'https://ducklove.github.io/all-about-gold/?theme=light&from=gold_gap');
    theme = 'dark';
    a.dispatchEvent(new Event('click'));
    assert.equal(a.getAttribute('href'), 'https://ducklove.github.io/all-about-gold/?theme=dark&from=gold_gap');
    assert.equal(bindSiblingLink(null, ALL_ABOUT_GOLD_ID, { getTheme: () => 'light' }), null);
});

// ---- ② 벤더링된 vc-shell.js 통합 ------------------------------------------------

function loadVendoredShell({ search = '', stored = null } = {}) {
    const store = new Map(stored ? [['theme', stored]] : []);
    const attrs = new Map([['data-theme', 'light']]);
    const document = new EventTarget();
    document.documentElement = {
        getAttribute: n => (attrs.has(n) ? attrs.get(n) : null),
        setAttribute: (n, v) => attrs.set(n, String(v)),
    };
    document.querySelector = sel => (sel === 'vc-shell[tool]' ? { getAttribute: () => 'gold_gap' } : null);
    document.querySelectorAll = () => [];
    const window = new EventTarget();
    Object.assign(window, {
        document,
        location: { search, pathname: '/gold_gap/', hash: '' },
        history: { state: null, replaceState() {} },
        localStorage: {
            getItem: k => (store.has(k) ? store.get(k) : null),
            setItem: (k, v) => store.set(k, String(v)),
            removeItem: k => store.delete(k),
        },
        matchMedia: () => ({ matches: false, addEventListener() {} }),
        innerWidth: 1280,
    });
    window.top = window;
    window.self = window;
    class MutationObserver { observe() {} disconnect() {} }
    const context = vm.createContext({
        window, document, URL, URLSearchParams, CustomEvent, EventTarget, MutationObserver, console,
    });
    vm.runInContext(readFileSync(new URL('../../static/vc-shell.js', import.meta.url), 'utf8'), context);
    return { window, document, store, attrs };
}

test('vendored vc-shell: 토글 → 공용 theme 키 저장 → vc:themechange → 재렌더 콜백', () => {
    const { window, document, store, attrs } = loadVendoredShell();
    const shell = getShell(window);
    assert.ok(shell, 'VCShell 전역이 설치돼야 한다');
    let current = 'light';
    const rerenders = [];
    onShellThemeChange(document, () => current, t => { current = t; rerenders.push(t); });
    assert.equal(requestTheme('dark', { shell, fallback: () => assert.fail('fallback 호출 금지') }), 'shell');
    assert.equal(store.get('theme'), 'dark');
    assert.equal(attrs.get('data-theme'), 'dark');
    assert.deepEqual(rerenders, ['dark']);
});

test('vendored vc-shell: linkTo(all-about-gold)가 현재 테마·출처를 붙인다', () => {
    const { window, attrs } = loadVendoredShell();
    attrs.set('data-theme', 'dark');
    const url = new URL(siblingUrl(ALL_ABOUT_GOLD_ID, ALL_ABOUT_GOLD_URL, { shell: getShell(window) }));
    assert.equal(url.origin + url.pathname, 'https://ducklove.github.io/all-about-gold/');
    assert.equal(url.searchParams.get('theme'), 'dark');
    assert.equal(url.searchParams.get('from'), 'gold_gap');
    // 자기 자신(레지스트리 gold_gap 항목)의 자산 딥링크 템플릿도 동작
    const self = new URL(window.VCShell.linkTo('gold_gap', { asset: 'bitcoin' }));
    assert.equal(self.searchParams.get('asset'), 'bitcoin');
});
