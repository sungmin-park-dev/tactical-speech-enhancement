# CLAUDE.md — AI 어시스턴트 컨텍스트

이 파일은 AI 코딩 어시스턴트가 프로젝트를 빠르게 파악하기 위한 참조 문서입니다.

## 프로젝트 목적

골전도(BC) + 공기전도(AC) 융합 기반 음성 향상 특허의 **진보성 입증**을 위한 실험.
핵심 차별점: **포화 마스크 기반 적응적 융합** (Magnitude Attention Gate).

## 코드 구조 & 의존 관계

```
configs/data_config.yaml       # 데이터 파이프라인 설정
configs/train_config.yaml      # 훈련 하이퍼파라미터 설정
    ↓ (load_config)
data/pipeline.py
    ├── data/bc_simulator.py      # BCSimulator (IIR 필터 + 신체잡음)
    ├── data/saturation.py        # SaturationSimulator (3종 마스크)
    ├── data/impulse_generator.py # generate_impulse_noise, ImpulseGenerator
    └── data/noise_mixer.py       # NoiseMixer → MilitaryNoiseMixer / GeneralNoiseMixer

data/dataset.py                   # SpeechEnhancementDataset (Pre-gen NPY)
    └── OnTheFlyDataset           # 실시간 합성 모드 (메모리 절약)

models/dpcrn_dual.py              # DPCRNDual (BC/AC 듀얼 인코더)
    ├── ComplexConv2d / ComplexConvTranspose2d  # 복소수 Conv
    ├── EncoderBlock / DecoderBlock             # 5층 인코더/디코더
    ├── DualPathBlock                           # BiLSTM(시간) + FC(주파수)
    └── CRM (Complex Ratio Mask)               # 복소수 마스크 → iSTFT

models/fusion.py                  # AdaptiveFusion (5 ablation 변형)
    └── MagnitudeAttentionGate, ConcatFusion

models/loss.py                    # TotalLoss = SI-SNR + α × MR-STFT
models/bwe.py                     # BWESubmodule (~300K params)
evaluate.py                       # PESQ/STOI/SI-SNR + paired t-test

train.py                          # 훈련 루프 (CLI + config 지원)
    ├── train_one_epoch()         # 1 epoch 학습
    ├── validate()                # 검증 루프
    └── main()                    # 설정 로딩 + 훈련 + 체크포인트 관리
```

## 구현 상태 (2026-03-31)

- [x] **D1** — 합성 파이프라인 (data/ 전체)
- [x] **D2** — 공통 모듈 (fusion, loss, bwe, evaluate)
- [x] **D3** — DPCRN 듀얼 인코더 (models/dpcrn_dual.py) ← **신규 완료**
- [x] **D4** — 훈련 루프 (train.py + configs/train_config.yaml) ← **신규 완료**
- [ ] **Colab 데이터 합성** — download_to_drive.ipynb 실행
- [ ] **DPCRN 훈련** — 5 ablation × 3 seeds = 15회
- [ ] **U-Net 구현** — models/unet_dual.py (Phase C)
- [ ] **평가** — 100 조건 매트릭스 + 통계 검정
- [ ] **엣지 실측** — ONNX 변환 + RP5/Jetson 지연

### 미커밋 변경사항 (staged)
- `configs/train_config.yaml` (신규)
- `models/__init__.py` (수정: DPCRNDual export 추가)
- `models/dpcrn_dual.py` (신규)
- `notebooks/download_to_drive.ipynb` (수정)
- `train.py` (신규)

## 알려진 이슈

### 1. Colab git clone 실패 (Private 레포)
- **원인**: 레포가 private → Colab에서 인증 없이 clone 불가
- **해결 A**: 레포를 Public으로 전환
- **해결 B**: GitHub PAT 발급 → Colab 시크릿 `GITHUB_TOKEN`에 등록
  ```python
  # Colab에서
  GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
  clone_url = f'https://{GITHUB_TOKEN}@github.com/user/repo.git'
  ```

### 2. Colab Drive I/O 할당량 초과
- **원인**: 다수 파일을 개별 전송하면 Google Drive I/O 할당량 초과
- **해결**: 단일 tar.gz 파일로만 Drive 전송 (Google 공식 권장)
  - 로컬 압축 → Drive 복사 (1회)
  - Drive 복사 → 로컬 해제 (매 세션)

### 3. VS Code .ipynb SyntaxError 오탐
- **원인**: VS Code 린터가 `!shell cmd`, `%%capture` 등 IPython 매직을 Python 문법으로 파싱
- **영향 없음**: Colab/Jupyter에서 정상 작동. VS Code에서만 발생하는 오탐.

### 4. DPCRN 파라미터 수 조정
- 실험 설계서 목표: ~2M params
- 채널 축소 (1→8→16→32→64→128, LSTM hidden=64) 적용
- 실제 파라미터 수는 단위 테스트로 확인 필요 (`python models/dpcrn_dual.py`)

## DPCRN 모델 아키텍처

```
BC ─→ STFT ─→ [ComplexConv2D ×5] ──→ BC_feat (256ch)
                                         ↓
                                    AdaptiveFusion(BC, AC, mask) ─→ fused
                                         ↑
AC ─→ STFT ─→ [ComplexConv2D ×5] ──→ AC_feat (256ch)

fused ─→ [DualPath ×2] ─→ [ComplexConvTranspose2D ×5 + Skip] ─→ CRM ─→ iSTFT ─→ enhanced
```
- Skip connection: 인코더의 **입력** (not 출력)을 디코더에 연결
- BC/AC encoder 파라미터 비공유
- decoder skip = (BC_skip + AC_skip) / 2 (AC 없으면 BC만 사용)

## 훈련 설정 요약 (train_config.yaml)

| 항목 | 값 |
|------|------|
| Optimizer | AdamW (lr=1e-3, wd=1e-2) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=10) |
| Loss | SI-SNR + 0.5 × MR-STFT |
| Epochs | 50 (early stopping patience=10) |
| Batch size | 16 (A100) / 8 (T4) |
| Grad clip | 5.0 |
| STFT | n_fft=512, hop=160, win=400 |

## Colab 노트북 워크플로우

```
[최초 1회] download_to_drive.ipynb
  Step 0: git clone + sys.path + pip install
  Step 1: LibriSpeech/MUSAN/DEMAND → /content/data/raw/
  Step 2: 합성 → /content/data/processed/
  Step 3: tar.gz → Drive 백업

[매 세션] train_colab.ipynb
  Step 0: git clone + sys.path + pip install
  Step 1: Drive tar.gz → /content/data/processed/ 복원
  Step 2: 훈련 (D3 완료 → 실행 가능)
  Step 3: 결과 tar.gz → Drive 저장
```

> **중요**: 커널 재시작 시 반드시 Step 0부터 재실행 (sys.path 초기화됨)

## 다음 실행 순서

1. **D3+D4 커밋 & Push** — staged 변경사항 커밋
2. **Colab 데이터 합성** — download_to_drive.ipynb (최초 1회)
3. **Smoke Test** — 소량 데이터로 train.py 전체 파이프라인 검증
4. **Alpha 탐색** — DPCRN A-soft: α ∈ {0.3, 0.5, 1.0} × seed 0
5. **DPCRN 본 훈련** — 5 ablation × 3 seeds = 15회
6. **DPCRN 평가** — 중간 결과 확인 → U-Net 진행 결정
7. **U-Net 구현/훈련** — models/unet_dual.py (Phase C)
8. **전체 평가 + 통계** — 100 조건 매트릭스, paired t-test

## 핵심 설계 결정

| 항목 | 결정 | 근거 |
|------|------|------|
| 가중치 w의 도메인 | 실수 (Sigmoid) | 복소수 특징에 스칼라 곱 → 위상 보존 |
| 마스크 유형 | soft 기본 | 실험 설계서 v3 §5.1 |
| MR-STFT 해상도 | 3종 (512/1024/2048) | Yamamoto et al. 표준 |
| α (MR-STFT 가중치) | 0.5 기본 (0.3/1.0 탐색) | 실험 설계서 v3 §6.3 |
| 화자 분할 | speaker-independent | LibriSpeech 경로 기반 화자 ID 추출 |
| 데이터 전송 | 단일 tar.gz | Google Drive I/O 할당량 제한 대응 |
| DPCRN 인코더 채널 | 1→8→16→32→64→128 | 파라미터 ~2M 목표 달성 |
| Dual-Path | BiLSTM(64) + FC, ×2 | 시간+주파수 양방향 모델링 |
| CRM 출력 | tanh 활성화 | 복소수 마스크 범위 (-1, +1) |
| Skip connection | 인코더 입력 연결 | 공간 해상도 보존 |

## 참고 문서

- `docs/experiment_design_v3.md` — 실험 설계서 (모든 사양의 원천)
- `docs/patent_revised_v3_final.tex` — 특허 명세서
