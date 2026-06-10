"""goldgap 패키지: 김치프리미엄 멀티 자산 대시보드 백엔드.

구성:
- assets: 자산 레지스트리 (라벨/임계값/색상 등 표시 메타데이터의 단일 진실 원천)
- sources: 전송 계층 (yfinance/거래소/네이버/WGC HTTP 호출)
- domain: 순수 계산 (괴리율, 고괴리 구간, 직렬화 데이터 병합) — 네트워크 의존 없음
- serialize: DataFrame → JSON 페이로드 직렬화, meta 블록/updated_at 생성
- orchestrators: 자산별 데이터 조립과 전체 수집
- cache: Flask 로컬 미리보기 경로용 24시간 파일 캐시
"""
