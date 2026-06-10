// sw.js — 김치프리미엄 대시보드 서비스워커(클래식 스크립트 — 모듈 아님).
//
// 스코프 주의: 이 파일은 사이트 루트(GitHub Pages에서는 /gold_gap/, 로컬 Flask에서는 /)에
// 서빙되며, 아래 모든 상대 경로는 이 스크립트의 위치를 기준으로 풀린다.
// 따라서 Pages 서브패스와 로컬 양쪽에서 같은 코드가 동작한다.
//
// 캐시 전략 근거:
//  ① data.json · og.png · api/ — network-first(네트워크 우선, 실패 시 캐시 폴백).
//     30분마다 갱신되는 시계열이므로 신선도가 우선이고, 오프라인/장애 시에만
//     마지막으로 받아둔 사본을 보여준다(완전 실패보다 낫다).
//  ② 그 외 GET(같은 출처 정적 자원 + cdn.jsdelivr.net의 고정 버전 Chart.js) —
//     stale-while-revalidate(캐시 즉시 응답 + 백그라운드 재검증).
//     앱 셸과 버전 고정 CDN은 내용이 거의 변하지 않아 체감 속도를 최우선으로 두되,
//     배포로 바뀐 파일은 다음 방문에 자연히 반영된다.
//  ③ 비GET 요청과 그 외 출처(실시간 시세 API 등)는 가로채지 않는다 — 시세는
//     캐시되면 안 되고, POST 등은 캐시 의미가 없다.

// v2: 가격 변동 분해 모듈(decompose.js) 추가 및 앱 셸 갱신.
const CACHE = 'goldgap-v2';

// 앱 셸 프리캐시 목록 — './'는 SW 위치 기준 사이트 루트 문서.
const APP_SHELL = [
    './',
    'static/style.css',
    'static/js/main.js',
    'static/js/config.js',
    'static/js/charts.js',
    'static/js/periods.js',
    'static/js/stats.js',
    'static/js/format.js',
    'static/js/live-quotes.js',
    'static/js/market.js',
    'static/js/decompose.js',
    'static/icon.svg',
    'manifest.webmanifest',
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE)
            .then(cache => cache.addAll(APP_SHELL))
            .then(() => self.skipWaiting()) // 새 버전 즉시 활성화
    );
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys()
            .then(keys => Promise.all(
                keys.filter(key => key !== CACHE).map(key => caches.delete(key)) // 구버전 캐시 정리
            ))
            .then(() => self.clients.claim()) // 열려 있는 탭도 즉시 제어
    );
});

// 데이터성 요청(전략 ①) 판별 — 같은 출처의 data.json / og.png / api/ 경로.
function isDataRequest(url) {
    return /\/data\.json$/.test(url.pathname)
        || /\/og\.png$/.test(url.pathname)
        || /\/api\//.test(url.pathname);
}

// ① network-first: 성공 응답을 캐시에 갱신해 두고, 실패하면 캐시 폴백.
// data.json?t=...처럼 캐시버스터 쿼리가 붙어도 같은 항목을 쓰도록
// 쿼리를 뗀 URL을 캐시 키로 쓴다.
function networkFirst(request) {
    const url = new URL(request.url);
    const cacheKey = url.origin + url.pathname;
    return caches.open(CACHE).then(cache =>
        fetch(request)
            .then(response => {
                if (response && response.ok) {
                    cache.put(cacheKey, response.clone());
                }
                return response;
            })
            .catch(() => cache.match(cacheKey).then(cached => {
                if (cached) return cached;
                throw new Error('offline and not cached: ' + request.url);
            }))
    );
}

// ② stale-while-revalidate: 캐시가 있으면 즉시 응답하고 백그라운드로 갱신.
// 탐색 요청(?asset=...&range=... 등 쿼리 변형)은 전부 프리캐시된 './' 한 항목으로
// 정규화해 캐시가 무한히 늘지 않게 한다. CDN 응답은 opaque(no-cors)여도 캐시한다.
function staleWhileRevalidate(request) {
    const cacheKey = request.mode === 'navigate' ? './' : request;
    return caches.open(CACHE).then(cache =>
        cache.match(cacheKey).then(cached => {
            const refresh = fetch(request)
                .then(response => {
                    if (response && (response.ok || response.type === 'opaque')) {
                        cache.put(cacheKey, response.clone());
                    }
                    return response;
                })
                .catch(() => cached); // 오프라인이면 캐시만으로 응답
            return cached || refresh;
        })
    );
}

self.addEventListener('fetch', (event) => {
    const request = event.request;
    if (request.method !== 'GET') return; // ③ 비GET 무시

    const url = new URL(request.url);
    const sameOrigin = url.origin === self.location.origin;

    if (sameOrigin && isDataRequest(url)) {
        event.respondWith(networkFirst(request)); // ① 신선도 우선
        return;
    }

    // ② 같은 출처 정적 자원 + 고정 버전 jsdelivr CDN
    if (sameOrigin || url.hostname === 'cdn.jsdelivr.net') {
        event.respondWith(staleWhileRevalidate(request));
        return;
    }
    // ③ 그 외 출처(실시간 시세 API 등)는 브라우저 기본 네트워크 동작에 맡긴다.
});
