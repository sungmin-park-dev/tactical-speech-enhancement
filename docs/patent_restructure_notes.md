# Patent Restructure Notes

Updated: 2026-04-04

이 문서는 최근 대화에서 합의된 핵심 판단을 잊지 않기 위한 작업 메모다.
명세서 재구성, 청구항 확장, 실험 프레이밍 재정렬 시 기준점으로 사용한다.

## 1. 현재 코드에 대한 확정 판단

- 현재 주력 구현은 [models/dpcrn_dual.py](/Users/david/GitHub/tactical-speech-enhancement/models/dpcrn_dual.py) 기준 BC/AC 듀얼 입력 음성 향상 모델이다.
- 현재 DPCRNDual은 strict causal 실시간 모델이 아니다.
- 비실시간성의 직접 근거:
  - `bidirectional=True` LSTM 사용
  - `torch.stft(..., center=True)` 사용
- 따라서 현재 DPCRNDual은 실시간 메인 경로의 직접 구현으로 보기보다, 고품질 경로 후보 또는 지연 허용 복원 경로 후보로 보는 것이 안전하다.
- 현재 구현에서 가장 확실한 핵심 자산은 [models/fusion.py](/Users/david/GitHub/tactical-speech-enhancement/models/fusion.py)의 포화 마스크 기반 adaptive fusion이다.
- BWE는 [models/bwe.py](/Users/david/GitHub/tactical-speech-enhancement/models/bwe.py)에 모듈은 존재하지만, 현재 [models/dpcrn_dual.py](/Users/david/GitHub/tactical-speech-enhancement/models/dpcrn_dual.py) forward 경로에는 연결되어 있지 않다.

## 2. 특허 서사에 대한 핵심 판단

- 특허의 중심은 "포화/신뢰도 기반 적응적 융합"과 "적응형 연산 제어"로 유지한다.
- 현재 코드와 명세서의 충돌은 발명 아이디어 자체의 붕괴가 아니라, 현재 구현의 역할 배치가 어긋난 문제로 본다.
- 명세서는 "실시간 저지연 온디바이스 통신"을 유지하되, 현재 DPCRN 계열 구현을 실시간 메인 경로에 직접 대응시키지 않는다.
- 명세서 재구성의 큰 방향은:
  - 실시간 메인 경로
  - 고품질 경로
  - 별도 보조 복원 경로
  를 분리해서 서술하는 것이다.

## 3. 모드와 경로에 대한 현재 합의

- 비실시간 복원은 모드 안에 억지로 넣지 않고, "별도의 복원 경로"로 두는 방향을 우선 선호한다.
- 즉, 실시간 통신 경로와 별개로:
  - 심하게 손상된 구간을 버퍼링하고
  - 후속으로 고복잡도 모델을 적용하여
  - 재청취, 재전송, 기록, 사후검토 등에 활용하는 경로를 생각한다.
- 이 비실시간 복원 경로는 현재 코드 자산을 살리는 중요한 축이다.

## 4. 제0모드에 대한 현재 합의

- 제0모드는 단순한 fail-safe 전용 모드로만 두지 않는다.
- 제0모드는 "DSP 전용 저복잡도 실시간 모드"로 넓게 정의하는 것이 기술적으로 가능하고 바람직하다.
- 제0모드의 진입 사유는 최소 두 축으로 생각한다.
  - 자원 기반: 지연 초과, 부하 초과, 전력 절감 필요, 런타임 오류
  - 신호 기반: 소음이 약하거나 단순하여 DSP만으로 충분한 경우
- 즉, 제0모드는 "고장 시 폴백"이자 "입력 조건이 단순할 때의 합리적 기본 경로"가 될 수 있다.
- 단순 소음 판정은 AI 없이도 가능하다고 본다.
- 후보 지표:
  - short-term RMS / Leq
  - 포화율 / clipping ratio
  - spectral flux
  - frame variance
  - impulsiveness
  - peak-to-RMS
  - 간단한 DSP 후 품질 proxy

## 5. 용어 정리

- CRM: Complex Ratio Mask
  - 모델이 복소수 스펙트럼 보정 마스크를 예측하고, 기준 복소 스펙트럼에 적용해 향상 신호를 복원하는 방식
- BWE: Bandwidth Extension
  - 골전도 우세 복원 시 부족한 고주파 대역을 보완하기 위한 대역 확장 모듈
- Adaptive Fusion
  - 포화 마스크와 BC/AC 특징을 이용해 채널 기여도를 동적으로 조절하는 핵심 메커니즘

## 6. 문서 작업 시 지켜야 할 원칙

- "현재 구현 사실"과 "설계 목표"를 같은 톤으로 쓰지 않는다.
- 현재 구현이 없는 것을 이미 입증된 사실처럼 쓰지 않는다.
- 실시간성은 메인 경로의 속성으로 설명하고, 현재 DPCRN 전체에 일반화하지 않는다.
- 현재 코드에 직접 대응되는 부분과, 후속 구현 예정 부분을 문장 레벨에서 분리한다.
- 명세서 재구성 논의에서는 "수정 방향"보다 "재구성안" 또는 "재배치"라는 표현을 선호한다.

## 7. 아직 열어둔 이슈

- 제1모드 실시간 메인 경로의 실제 causal/lightweight 백본을 무엇으로 할지
- "사용자 듣기 어려운 경우" 또는 "severely degraded"의 판정 기준을 어떻게 정의할지
- 비실시간 복원 결과를 재청취, 재전송, 기록 중 어디까지 핵심 실시예로 둘지
- 모드 정의와 경로 정의를 명세서에서 어디까지 분리해서 쓸지

## 8. 다음 작업 우선순위

- 이 메모를 기준으로 [docs/patent.tex](/Users/david/GitHub/tactical-speech-enhancement/docs/patent.tex)의 재구성 아웃라인을 구체화한다.
- 그 다음 청구항과 실험 문서의 프레이밍을 다시 정리한다.
