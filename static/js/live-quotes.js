// live-quotes.js — 브라우저에서 직접 공개 API 현재가를 받아 데이터에 합성(폴백 새로고침 경로).
// DOM에 의존하지 않는다. 배지 표시 등 UI 반영은 main.js 책임.

import { roundNumber, latestValue, getKstDateString, getKstTimeString } from './format.js';
import { buildHighGapPeriods } from './periods.js';
import { t } from './i18n.js';

export const TROY_OZ_TO_GRAM = 31.1035;

export async function fetchJson(url) {
    const resp = await fetch(url, { cache: 'no-store' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
}

function ensureSeries(data, key) {
    if (!Array.isArray(data[key])) {
        data[key] = Array(data.dates.length).fill(null);
    }
    while (data[key].length < data.dates.length) {
        data[key].push(data[key].length ? data[key][data[key].length - 1] : null);
    }
}

// date 지점에 values를 삽입/갱신하고 gap_pct와 high_gap_periods를 재계산한다.
export function upsertQuotePoint(data, date, values, threshold) {
    let idx = data.dates.indexOf(date);
    const arrayKeys = Object.keys(data).filter(key => key !== 'dates' && Array.isArray(data[key]));

    if (idx === -1) {
        data.dates.push(date);
        idx = data.dates.length - 1;
        arrayKeys.forEach(key => {
            const last = data[key].length ? data[key][data[key].length - 1] : null;
            data[key].push(last);
        });
    }

    Object.entries(values).forEach(([key, value]) => {
        ensureSeries(data, key);
        data[key][idx] = value;
    });

    ensureSeries(data, 'gap_pct');
    if (data.domestic_price[idx] != null && data.intl_price[idx] != null && data.intl_price[idx] !== 0) {
        data.gap_pct[idx] = roundNumber(((data.domestic_price[idx] - data.intl_price[idx]) / data.intl_price[idx]) * 100, 2);
    }
    data.high_gap_periods = buildHighGapPeriods(data, threshold);
}

// intl_modes 보유 자산: 기본 모드의 시리즈를 자산 최상위로 복사(미러).
// 기존 syncGoldDefaultMode의 일반화 — 고정 키 목록 대신 모드 데이터의 모든
// 배열 시리즈(high_gap_periods 포함)를 복사한다(현행 데이터에서 결과 동일).
function syncDefaultModeMirror(assetData) {
    if (!assetData || !assetData.intl_modes) return;
    const defaultMode = assetData.default_intl_mode || Object.keys(assetData.intl_modes)[0];
    const modeData = assetData.intl_modes[defaultMode];
    if (!modeData) return;
    Object.keys(modeData).forEach(key => {
        if (Array.isArray(modeData[key])) assetData[key] = [...modeData[key]];
    });
}

// 브라우저에서 접근 가능한 공개 시세 API 모음. 합성 가능한 소스가 자산별로
// 고정돼 있어(금 현물/BTC/USDT) 이 함수는 자산 키에 결합된 채 유지한다.
async function fetchClientLiveQuotes() {
    const [fx, goldSpot, upbitBtc, binanceBtc, bithumbUsdt] = await Promise.allSettled([
        fetchJson('https://open.er-api.com/v6/latest/USD'),
        fetchJson('https://api.gold-api.com/price/XAU/USD'),
        fetchJson('https://api.upbit.com/v1/ticker?markets=KRW-BTC'),
        fetchJson('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT'),
        fetchJson('https://api.bithumb.com/public/ticker/USDT_KRW'),
    ]);

    const quote = {};
    if (fx.status === 'fulfilled' && fx.value && fx.value.rates && fx.value.rates.KRW) {
        quote.usdKrw = Number(fx.value.rates.KRW);
    }
    if (goldSpot.status === 'fulfilled' && goldSpot.value && goldSpot.value.price) {
        quote.goldSpotUsdOz = Number(goldSpot.value.price);
    }
    if (upbitBtc.status === 'fulfilled' && Array.isArray(upbitBtc.value) && upbitBtc.value[0]) {
        quote.btcKrw = Number(upbitBtc.value[0].trade_price);
    }
    if (binanceBtc.status === 'fulfilled' && binanceBtc.value && binanceBtc.value.price) {
        quote.btcUsd = Number(binanceBtc.value.price);
    }
    if (bithumbUsdt.status === 'fulfilled' && bithumbUsdt.value && bithumbUsdt.value.status === '0000') {
        quote.usdtKrw = Number(bithumbUsdt.value.data.closing_price);
    }
    return quote;
}

// allData에 현재가를 합성한다. configs는 buildAssetConfigs 결과의 configs 맵 —
// 임계치만 사용하며, 설정이 없으면 5%로 둔다. 반환값은 합성된 시리즈 수.
export async function applyClientLiveQuotes(allData, configs) {
    if (!allData) throw new Error(t('live.noBaseData'));
    const thresholdOf = key => (configs && configs[key] && typeof configs[key].threshold === 'number')
        ? configs[key].threshold
        : 5;

    const quote = await fetchClientLiveQuotes();
    const date = getKstDateString();
    let applied = 0;

    if (allData.gold && allData.gold.intl_modes && quote.usdKrw) {
        const domestic = latestValue(allData.gold.intl_modes.ny_futures, 'domestic_price');
        const nyUsdOz = latestValue(allData.gold.intl_modes.ny_futures, 'gold_usd_oz');
        if (domestic != null && nyUsdOz != null) {
            upsertQuotePoint(allData.gold.intl_modes.ny_futures, date, {
                domestic_price: roundNumber(domestic, 2),
                gold_usd_oz: roundNumber(nyUsdOz, 6),
                usd_krw: roundNumber(quote.usdKrw, 2),
                intl_price: roundNumber((nyUsdOz * quote.usdKrw) / TROY_OZ_TO_GRAM, 2),
            }, thresholdOf('gold'));
            applied++;
        }
        if (domestic != null && quote.goldSpotUsdOz) {
            upsertQuotePoint(allData.gold.intl_modes.london_spot, date, {
                domestic_price: roundNumber(domestic, 2),
                gold_usd_oz: roundNumber(quote.goldSpotUsdOz, 6),
                usd_krw: roundNumber(quote.usdKrw, 2),
                intl_price: roundNumber((quote.goldSpotUsdOz * quote.usdKrw) / TROY_OZ_TO_GRAM, 2),
            }, thresholdOf('gold'));
            applied++;
        }
        syncDefaultModeMirror(allData.gold);
    }

    if (allData.bitcoin && quote.usdKrw && quote.btcKrw && quote.btcUsd) {
        upsertQuotePoint(allData.bitcoin, date, {
            domestic_price: roundNumber(quote.btcKrw, 2),
            crypto_usd: roundNumber(quote.btcUsd, 6),
            usd_krw: roundNumber(quote.usdKrw, 2),
            intl_price: roundNumber(quote.btcUsd * quote.usdKrw, 2),
        }, thresholdOf('bitcoin'));
        applied++;
    }

    if (allData.usdt && quote.usdKrw && quote.usdtKrw) {
        upsertQuotePoint(allData.usdt, date, {
            domestic_price: roundNumber(quote.usdtKrw, 2),
            crypto_usd: 1,
            usd_krw: roundNumber(quote.usdKrw, 2),
            intl_price: roundNumber(quote.usdKrw, 2),
        }, thresholdOf('usdt'));
        applied++;
    }

    if (!applied) throw new Error(t('live.noQuotes'));
    // 갱신 시점 언어로 기록된다 — 이후 언어를 전환해도 다음 갱신 전까지는 이 문구가 남는다.
    allData.updated_at = t('live.updatedPrefix') + getKstTimeString();
    return applied;
}
