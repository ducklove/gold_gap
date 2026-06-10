// market.js — 시장 지표(KOSPI/S&P500/USD·KRW)와 자산 USD 시세의
// 날짜 정렬·로그수익률·피어슨 상관 계산. 순수 함수만 두며 DOM에 의존하지 않는다
// (node 단위 테스트 대상). 화면 반영은 main.js/charts.js 책임.

// 상관 행렬에서 유효 표본(쌍별 완전 관측) 최소 개수 — 미만이면 null('-' 표시).
export const MIN_CORR_SAMPLES = 20;

function isFiniteNumber(value) {
    return typeof value === 'number' && Number.isFinite(value);
}

// {이름: {dates, values}} 묶음을 공통 날짜 축(합집합, 오름차순)으로 행 단위 정렬한다.
// 반환: { dates: [...], series: {이름: [value|null, ...]} } — 시리즈마다 길이가 dates와
// 같고, 해당 날짜 관측이 없거나 숫자가 아니면 null. 쌍별로 따로 맞추지 않고 한 번에
// 정렬해 두면 비교 차트(spanGaps)와 상관 계산이 같은 행 인덱스를 공유한다.
export function alignSeries(seriesMap) {
    const entries = Object.entries(seriesMap || {}).filter(([, s]) =>
        s && Array.isArray(s.dates) && Array.isArray(s.values) && s.dates.length > 0
    );

    const dateSet = new Set();
    entries.forEach(([, s]) => {
        const n = Math.min(s.dates.length, s.values.length);
        for (let i = 0; i < n; i++) dateSet.add(String(s.dates[i]));
    });
    const dates = [...dateSet].sort(); // ISO(YYYY-MM-DD) 문자열은 사전순 = 시간순

    const series = {};
    entries.forEach(([name, s]) => {
        const byDate = new Map();
        const n = Math.min(s.dates.length, s.values.length);
        for (let i = 0; i < n; i++) {
            const v = s.values[i];
            byDate.set(String(s.dates[i]), isFiniteNumber(v) ? v : null);
        }
        series[name] = dates.map(d => (byDate.has(d) ? byDate.get(d) : null));
    });
    return { dates, series };
}

// 일별 로그수익률 ln(v[i]/v[i-1]). 결과 길이 = values.length - 1.
// 인접 두 관측 중 하나라도 null/0/음수면 해당 수익률은 null(휴장 결측 안전).
export function logReturns(values) {
    const list = Array.isArray(values) ? values : [];
    const out = [];
    for (let i = 1; i < list.length; i++) {
        const prev = list[i - 1];
        const cur = list[i];
        out.push(isFiniteNumber(prev) && prev > 0 && isFiniteNumber(cur) && cur > 0
            ? Math.log(cur / prev)
            : null);
    }
    return out;
}

// 피어슨 상관계수 — 쌍별 완전 관측(둘 다 유한한 행)만 사용.
// 표본 < minN 또는 한쪽 분산이 0이면 r = null. n은 항상 실제 쌍 표본 수.
function pearsonWithN(xs, ys, minN) {
    const len = Math.min(
        Array.isArray(xs) ? xs.length : 0,
        Array.isArray(ys) ? ys.length : 0
    );
    const px = [];
    const py = [];
    for (let i = 0; i < len; i++) {
        if (isFiniteNumber(xs[i]) && isFiniteNumber(ys[i])) {
            px.push(xs[i]);
            py.push(ys[i]);
        }
    }
    const n = px.length;
    if (n < minN) return { r: null, n };

    const meanX = px.reduce((a, b) => a + b, 0) / n;
    const meanY = py.reduce((a, b) => a + b, 0) / n;
    let cov = 0;
    let varX = 0;
    let varY = 0;
    for (let i = 0; i < n; i++) {
        const dx = px[i] - meanX;
        const dy = py[i] - meanY;
        cov += dx * dy;
        varX += dx * dx;
        varY += dy * dy;
    }
    if (varX === 0 || varY === 0) return { r: null, n };
    const r = cov / Math.sqrt(varX * varY);
    return { r: Math.max(-1, Math.min(1, r)), n }; // 부동소수 오차로 ±1을 살짝 넘는 것 방지
}

export function pearson(xs, ys, minN = MIN_CORR_SAMPLES) {
    return pearsonWithN(xs, ys, minN).r;
}

// 일별 로그수익률 기반 상관계수 행렬.
// 입력: {이름: {dates, values}} — 내부에서 alignSeries로 행 정렬 후 수익률 계산.
// 반환: { labels: [이름...], matrix: [[r|null]], n: [[쌍별 표본수]] } (대칭, 대각도 계산값).
export function buildCorrelationMatrix(seriesMap, minN = MIN_CORR_SAMPLES) {
    const aligned = alignSeries(seriesMap);
    const labels = Object.keys(aligned.series);
    const returns = labels.map(name => logReturns(aligned.series[name]));

    const matrix = labels.map(() => labels.map(() => null));
    const n = labels.map(() => labels.map(() => 0));
    for (let i = 0; i < labels.length; i++) {
        for (let j = i; j < labels.length; j++) {
            const { r, n: count } = pearsonWithN(returns[i], returns[j], minN);
            matrix[i][j] = r;
            matrix[j][i] = r;
            n[i][j] = count;
            n[j][i] = count;
        }
    }
    return { labels, matrix, n };
}

// 비교 차트용 정규화: 첫 유효 관측을 100으로 둔 지수. null은 null 유지(spanGaps용).
// 유효 시작점(>0)이 없으면 null 반환(시리즈 제외 신호).
export function normalizeTo100(values) {
    const list = Array.isArray(values) ? values : [];
    const base = list.find(v => isFiniteNumber(v) && v > 0);
    if (base == null) return null;
    return list.map(v => (isFiniteNumber(v) ? (v / base) * 100 : null));
}

// 카드용 최신값/전일 대비: 마지막 유효 관측과 그 직전 유효 관측(휴장 null 건너뜀).
// 반환 {value, prev, changePct|null} — 유효 관측이 없으면 null.
export function latestWithChange(values) {
    const list = Array.isArray(values) ? values : [];
    let value = null;
    let prev = null;
    for (let i = list.length - 1; i >= 0; i--) {
        if (!isFiniteNumber(list[i])) continue;
        if (value == null) {
            value = list[i];
        } else {
            prev = list[i];
            break;
        }
    }
    if (value == null) return null;
    const changePct = prev != null && prev !== 0 ? ((value / prev) - 1) * 100 : null;
    return { value, prev, changePct };
}

// 상관/비교 대상 시리즈 정의(표시 순서·라벨·색 — 테마와 어울리는 고정 팔레트).
export const MARKET_SERIES = [
    { key: 'gold', label: '금 (XAU)', color: '#d9a441' },
    { key: 'btc', label: 'BTC', color: '#d9791f' },
    { key: 'eth', label: 'ETH', color: '#627eea' },
    { key: 'usd_krw', label: 'USD/KRW', color: '#1f9b57' },
    { key: 'kospi', label: 'KOSPI', color: '#c0504d' },
    { key: 'sp500', label: 'S&P 500', color: '#2a78c9' },
];

function pickSeries(source, valueKey) {
    if (!source || !Array.isArray(source.dates) || !Array.isArray(source[valueKey])) return null;
    const values = source[valueKey];
    if (!values.some(isFiniteNumber)) return null; // 전부 결측인 시리즈는 제외
    return { dates: source.dates, values };
}

// data.json 전체(allData)에서 상관 대상 시리즈를 추출한다. 없는 시리즈는 조용히 제외.
// - 금: gold.intl_modes.ny_futures의 gold_usd_oz (XAU USD)
// - BTC/ETH: 각 자산의 crypto_usd
// - USD/KRW: market.usd_krw, 없으면 gold ny_futures의 usd_krw로 폴백
// - KOSPI/S&P500: market 블록(없으면 제외 — market 부재 시 섹션 자체가 숨음)
// 반환: [{key, label, color, dates, values}] (MARKET_SERIES 순서 유지)
export function collectMarketSeries(allData) {
    if (!allData) return [];
    const market = allData.market;
    const goldNy = allData.gold && allData.gold.intl_modes && allData.gold.intl_modes.ny_futures;

    const sources = {
        gold: pickSeries(goldNy, 'gold_usd_oz'),
        btc: pickSeries(allData.bitcoin, 'crypto_usd'),
        eth: pickSeries(allData.eth, 'crypto_usd'),
        usd_krw: pickSeries(market, 'usd_krw') || pickSeries(goldNy, 'usd_krw'),
        kospi: pickSeries(market, 'kospi'),
        sp500: pickSeries(market, 'sp500'),
    };

    return MARKET_SERIES
        .filter(def => sources[def.key])
        .map(def => ({ ...def, ...sources[def.key] }));
}
