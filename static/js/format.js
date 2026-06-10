// format.js — 숫자/통화 포맷터와 KST 날짜 유틸. 순수 함수만 두며 DOM에 의존하지 않는다.

export function roundNumber(value, digits = 2) {
    if (value == null || Number.isNaN(Number(value))) return null;
    const factor = 10 ** digits;
    return Math.round(Number(value) * factor) / factor;
}

export function latestValue(data, key) {
    const values = data && data[key];
    if (!values || values.length === 0) return null;
    return values[values.length - 1];
}

export function getKstDateString(date = new Date()) {
    const parts = new Intl.DateTimeFormat('en-CA', {
        timeZone: 'Asia/Seoul',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
    }).formatToParts(date).reduce((acc, part) => {
        acc[part.type] = part.value;
        return acc;
    }, {});
    return `${parts.year}-${parts.month}-${parts.day}`;
}

export function getKstTimeString(date = new Date()) {
    return new Intl.DateTimeFormat('ko-KR', {
        timeZone: 'Asia/Seoul',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
    }).format(date).replace(/\.$/, '');
}

// 차트 툴팁용 가격 포맷. unit이 'KRW/g'(금)이면 '원/g', 그 외에는 '원'.
export function formatPrice(value, unit) {
    if (unit === 'KRW/g') return Number(value).toLocaleString() + ' 원/g';
    return Number(value).toLocaleString() + ' 원';
}

export function formatKrw(value, options = {}) {
    if (value == null || Number.isNaN(Number(value))) return '-';
    return Number(value).toLocaleString('ko-KR', options) + ' 원';
}

export function formatUsd(value, options = {}) {
    if (value == null || Number.isNaN(Number(value))) return '-';
    return '$' + Number(value).toLocaleString('en-US', options);
}

// 자산 카드용 원화 포맷. krwFractionDigits가 숫자면 해당 소수 자리 고정(USDT=2),
// 아니면 정수 표기(금/BTC) — 기존 formatAssetKrw(value, asset) 동작과 동일.
export function formatAssetKrw(value, krwFractionDigits) {
    if (value == null || Number.isNaN(Number(value))) return '-';
    const digits = typeof krwFractionDigits === 'number'
        ? { minimumFractionDigits: krwFractionDigits, maximumFractionDigits: krwFractionDigits }
        : { maximumFractionDigits: 0 };
    return formatKrw(value, digits);
}
