// decompose.js — 가격 변동 분해(로그수익률 항등식). 순수 함수만 두며 DOM에 의존하지 않는다.
//
// "국내 가격 변화가 국제 가격 때문인가, 환율 때문인가, 김프 변동 때문인가"에 답한다.
// 선택 기간 [i0, i1]에서 (i0/i1 = domestic·usd·fx가 모두 유효한 첫/마지막 인덱스):
//   R_dom = ln(domestic[i1]/domestic[i0]) × 100
//   R_usd = ln(usd[i1]/usd[i0]) × 100      (usd = gold_usd_oz | crypto_usd)
//   R_fx  = ln(fx[i1]/fx[i0]) × 100        (fx = usd_krw)
//   R_gap = R_dom − R_usd − R_fx
// R_gap은 잔차로 계산한다 — 저장값(gap_pct)이 소수 2자리 반올림이라 직접 계산하면
// 항등식이 어긋나므로, 잔차 구성으로 R_dom = R_usd + R_fx + R_gap을 정확히 보장한다.
// USDT처럼 환율을 추종하는 자산은 R_usd ≈ 0이 정상이다.

function isPositiveNumber(value) {
    return typeof value === 'number' && Number.isFinite(value) && value > 0;
}

// 자산 데이터에서 USD 시세 시리즈 키를 찾는다(gold_usd_oz, crypto_usd 등).
// main.js 상세 카드와 같은 규칙 — usd_krw(환율)는 제외.
export function findUsdSeriesKey(data) {
    if (!data) return null;
    return Object.keys(data).find(key =>
        key !== 'usd_krw' && Array.isArray(data[key]) && /_usd(_oz)?$/.test(key)
    ) || null;
}

// activeData(모드 보유 자산은 선택된 모드의 데이터)와 시작 인덱스(호출 측이
// periods.js의 rangeStartIndex로 산출)를 받아 분해 결과를 반환한다.
// 반환: { dom, usd, fx, gap, startDate, endDate } | null.
// null 조건: 시리즈 누락, 또는 기간 내 유효(null/0/음수 아님) 관측 쌍이 2개 미만.
export function decomposePriceChange(activeData, startIdx) {
    if (!activeData || !Array.isArray(activeData.dates)) return null;
    const usdKey = findUsdSeriesKey(activeData);
    if (!usdKey) return null;

    const dates = activeData.dates;
    const dom = activeData.domestic_price;
    const fx = activeData.usd_krw;
    const usd = activeData[usdKey];
    if (!Array.isArray(dom) || !Array.isArray(fx)) return null;

    const from = Math.max(0, Number.isInteger(startIdx) ? startIdx : 0);
    const last = Math.min(dates.length, dom.length, fx.length, usd.length) - 1;
    const validAt = i => isPositiveNumber(dom[i]) && isPositiveNumber(usd[i]) && isPositiveNumber(fx[i]);

    let i0 = -1;
    for (let i = from; i <= last; i++) {
        if (validAt(i)) { i0 = i; break; }
    }
    if (i0 === -1) return null;

    let i1 = -1;
    for (let i = last; i > i0; i--) {
        if (validAt(i)) { i1 = i; break; }
    }
    if (i1 === -1) return null; // 유효 쌍 2개 미만 — 패널 숨김 신호

    const rDom = Math.log(dom[i1] / dom[i0]) * 100;
    const rUsd = Math.log(usd[i1] / usd[i0]) * 100;
    const rFx = Math.log(fx[i1] / fx[i0]) * 100;
    return {
        dom: rDom,
        usd: rUsd,
        fx: rFx,
        gap: rDom - rUsd - rFx, // 잔차 — 구성상 R_dom = R_usd + R_fx + R_gap
        startDate: dates[i0],
        endDate: dates[i1],
    };
}
