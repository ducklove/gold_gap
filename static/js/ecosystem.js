// ecosystem.js — Value Compass 에코시스템 바(window.VCShell, static/vc-shell.js) 연동 헬퍼.
//
// vc-shell.js는 벤더링된 클래식 defer 스크립트라 모듈(main.js)보다 먼저 실행되지만,
// 로드 실패·차단 시에도 대시보드가 기존과 똑같이 동작해야 하므로 모든 진입점에
// "셸 없으면 기존 로직" 폴백을 둔다. DOM·전역에 직접 의존하지 않고 의존성을 주입받아
// node --test로 검증한다(tests/js/ecosystem.test.mjs).

export const TOOL_ID = 'gold_gap';
export const ALL_ABOUT_GOLD_ID = 'all-about-gold';
export const ALL_ABOUT_GOLD_URL = 'https://ducklove.github.io/all-about-gold/';

function validTheme(theme) {
    return theme === 'dark' || theme === 'light' ? theme : null;
}

// window.VCShell — 공개 API(setTheme/linkTo)를 갖춘 경우에만 반환.
export function getShell(win = typeof window !== 'undefined' ? window : undefined) {
    const shell = win && win.VCShell;
    return shell && typeof shell.setTheme === 'function' ? shell : null;
}

// 테마 토글 요청. 셸이 있으면 셸이 공용 'theme' 키 저장 → data-theme 적용 → 'vc:themechange'
// 발행까지 맡고(재렌더는 이벤트 구독자가 수행), 없거나 실패하면 fallback(기존 로컬 로직).
// 반환값: 'shell' | 'fallback' (테스트용).
export function requestTheme(theme, { shell, fallback }) {
    const next = validTheme(theme) || 'light';
    if (shell) {
        try {
            shell.setTheme(next);
            return 'shell';
        } catch (e) {
            // 셸 오류 — 아래 기존 경로로 계속
        }
    }
    fallback(next);
    return 'fallback';
}

// 'vc:themechange' 구독(셸 토글·다른 탭 storage·OS 다크모드 추종·허브 postMessage 모두 이 이벤트로 온다).
// 현재 테마와 다를 때만 onChange(theme)를 불러 중복 차트 재렌더를 막는다. 해제 함수 반환.
export function onShellThemeChange(target, getCurrentTheme, onChange) {
    if (!target || typeof target.addEventListener !== 'function') return () => {};
    const handler = (event) => {
        const theme = validTheme(event && event.detail && event.detail.theme);
        if (!theme || theme === getCurrentTheme()) return;
        onChange(theme);
    };
    target.addEventListener('vc:themechange', handler);
    return () => target.removeEventListener('vc:themechange', handler);
}

// 형제 도구 URL. 셸이 있으면 레지스트리 템플릿(linkTo: ?theme·from 자동 부착)을 쓰고,
// 없으면 고정 URL에 ?theme=<현재 테마>&from=gold_gap만 붙인다(github.io 형제는 ?theme를 받는다).
export function siblingUrl(toolId, fallbackUrl, { shell = null, theme = null, vars = {} } = {}) {
    if (shell && typeof shell.linkTo === 'function') {
        try {
            const url = shell.linkTo(toolId, vars);
            if (url) return url;
        } catch (e) {
            // 폴백으로 계속
        }
    }
    try {
        const url = new URL(fallbackUrl);
        const t = validTheme(theme);
        if (t) url.searchParams.set('theme', t);
        url.searchParams.set('from', TOOL_ID);
        return url.href;
    } catch (e) {
        return fallbackUrl;
    }
}

// <a>의 href를 이동 직전(포커스·포인터·클릭·새 탭 열기)에 현재 테마로 다시 계산한다.
// 클릭 핸들러에서 바꾼 href는 기본 동작(이동)에 그대로 반영된다.
export function bindSiblingLink(anchor, toolId, { getShell: shellOf = getShell, getTheme }) {
    if (!anchor || typeof anchor.getAttribute !== 'function') return null;
    const base = anchor.getAttribute('href');
    const refresh = () => {
        anchor.setAttribute('href', siblingUrl(toolId, base, { shell: shellOf(), theme: getTheme() }));
    };
    ['focus', 'pointerdown', 'click', 'auxclick', 'contextmenu'].forEach((type) => {
        anchor.addEventListener(type, refresh);
    });
    refresh();
    return refresh;
}
