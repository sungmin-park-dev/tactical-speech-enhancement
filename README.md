# Tactical Speech Enhancement

골전도(BC) + 공기전도(AC) 융합 기반 음성 향상 — **포화 마스크 기반 적응적 융합** 특허의 진보성 입증을 위한 실험 코드베이스

## 프로젝트 구조

```
tactical-speech-enhancement/
├── configs/
│   └── data_config.yaml          # 데이터 파이프라인 설정
├── data/                         # D1: 합성 파이프라인
│   ├── bc_simulator.py           # IIR BC 필터 (피킹 EQ + Butterworth LPF) + 신체 잡음
│   ├── saturation.py             # 포화 시뮬 (hard/soft/poly) + 마스크 3종 생성
│   ├── impulse_generator.py      # 군용 충격음 합성 (총성/폭발)
│   ├── noise_mixer.py            # Military(실제+합성), General(DEMAND/MUSAN/WHAM) 소음 혼합
│   ├── pipeline.py               # SampleGenerator + DatasetBuilder + 화자 기준 분할
│   ├── dataset.py                # PyTorch Dataset/DataLoader
│   ├── download_noises.py        # 원본 DB 다운로드 스크립트
│   └── verify_distribution.py    # 합성 데이터 분포 검증
├── models/                       # D2: 공통 모듈
│   ├── fusion.py                 # Magnitude 기반 Adaptive Fusion (Ablation 5변형)
│   ├── loss.py                   # SI-SNR + MR-STFT(3해상도) 손실 함수
│   └── bwe.py                    # 대역 확장(BWE) 서브모듈 (~300K 파라미터)
├── notebooks/
│   ├── download_to_drive.ipynb   # [Colab] 최초 1회: DB 다운로드 → 합성 → Drive 백업
│   └── train_colab.ipynb         # [Colab] 훈련: Drive 복원 → 모델 학습 → 결과 저장
├── docs/
│   ├── experiment_design_v3.md   # 실험 설계서
│   └── patent_revised_v3_final.tex  # 특허 명세서
├── evaluate.py                   # PESQ / STOI / SI-SNR + paired t-test
├── requirements.txt
└── README.md
```

## 구현 현황

| Phase | 모듈 | 상태 | 설명 |
|-------|------|------|------|
| **D1** | `data/*` | ✅ 완료 | BC 시뮬, 포화 마스크(3유형), 충격음 합성, 소음 혼합, 파이프라인 |
| **D2** | `models/fusion.py` | ✅ 완료 | Magnitude Attention Gate (A-soft/hard/param, B, C) |
| **D2** | `models/loss.py` | ✅ 완료 | SI-SNR + MR-STFT (3해상도) 통합 손실 |
| **D2** | `models/bwe.py` | ✅ 완료 | BWE 서브모듈 (FC→FC→Conv1D, ~300K params) |
| **D2** | `evaluate.py` | ✅ 완료 | PESQ/STOI/SI-SNR 평가 + paired t-test 통계 검정 |
| **D3** | `models/dpcrn_dual.py` | ⬜ 미구현 | DPCRN 듀얼 인코더 모델 |
| **D4** | `train.py` | ⬜ 미구현 | 훈련 루프 |

## 핵심 설계

### 적응적 융합 (Adaptive Fusion)
```
BC_feat ──┐                        ┌── w × BC_feat
           ├── |BC|, |AC|, mask ──→ Conv1×1 → σ → w
AC_feat ──┘                        └── (1-w) × AC_feat
```
- **가중치 w는 실수** (0~1) → 복소수 특징맵에 스칼라 곱 → **위상 보존**
- Ablation 5변형: A-soft, A-hard, A-param (마스크 유형), B (마스크 미사용), C (BC only)

### 데이터 합성 파이프라인
```
LibriSpeech clean → BC 시뮬 → 포화 시뮬 → BC 채널 (+ 마스크 3종)
                  → 소음 혼합            → AC 채널
```
- **Military**: SNR -20~+5dB, 합성 충격음+실제 녹음, 포화율 30~100%
- **General**: SNR -10~+15dB, DEMAND/MUSAN/WHAM, 포화율 0~50%

## Colab 실행 (Google Drive I/O 최적화)

> **핵심 원칙**: Drive ↔ Colab 전송은 **단일 tar.gz 파일**만 사용 (I/O 할당량 절약)

### 사전 조건
- **레포가 Public**이어야 Colab에서 `git clone` 가능
  - 또는 [GitHub PAT](https://github.com/settings/tokens) 발급 후 Colab 시크릿에 `GITHUB_TOKEN` 등록

### 최초 1회
```
notebooks/download_to_drive.ipynb 실행:
  Step 0 → 환경 설정 (git clone + sys.path + pip install)
  Step 1 → 원본 DB 다운로드 (/content/ 로컬, ~15분)
  Step 2 → 합성 데이터셋 생성
  Step 3 → processed/ → tar.gz 압축 → Drive 저장
```

### 이후 매 세션
```
notebooks/train_colab.ipynb 실행:
  Step 0 → 환경 설정
  Step 1 → Drive tar.gz → /content/ 복원
  Step 2 → 모델 훈련 (D3 완료 후)
  Step 3 → 결과 tar.gz → Drive 저장
```

## 로컬 개발

```bash
git clone https://github.com/sungmin-park-dev/tactical-speech-enhancement.git
cd tactical-speech-enhancement
pip install -r requirements.txt

# 단위 테스트
python data/bc_simulator.py
python data/noise_mixer.py
python models/fusion.py
python models/loss.py
python models/bwe.py
python evaluate.py
```

## 다음 단계 (TODO)

1. **D3**: `models/dpcrn_dual.py` — DPCRN 듀얼 인코더 구현
2. **D4**: `train.py` — 훈련 루프 + 5 ablation × 2 backbone × 3 seeds
3. **평가**: 100 조건 매트릭스 (환경 × 포화율 × 백본 × Ablation)
4. **논문**: paired t-test로 A-soft vs B 통계적 유의성 입증

## 참고

- 실험 설계서: `docs/experiment_design_v3.md`
- 특허 명세서: `docs/patent_revised_v3_final.tex`
