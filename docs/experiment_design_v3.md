# 실험 설계서 v3 (최종본)

## 변경 이력
- v1: 단일 모델(Conv1D) 기반 초안
- v2: 듀얼 모델 비교(DPCRN vs U-Net) + 군용/일반 환경 분리
- **v3: 학술적 엄밀성 강화 + 엣지 실측 + 피드백 5건 반영**
  - 손실 함수 SI-SNR + MR-STFT로 변경
  - 복소수 어텐션 게이트를 Magnitude 기반으로 수정
  - 충격음 하이브리드 파이프라인 (실제 녹음 + 수식 합성)
  - ONNX 변환 + 엣지 디바이스 실측 지연 추가
  - 포화 마스크 Ablation 5변형 확장

---

## 1. 실험 목적

1. **진보성 소명**: 포화 마스크 기반 적응적 융합의 정량적 효과 입증 (심사관 지적 #15)
2. **모델 구조 선정**: DPCRN vs 2D Conv U-Net, 군용/일반 환경 데이터 기반 비교
3. **마스크 유형별 효과**: Hard / Soft / Parametric 마스크 간 차이 → 청구범위 방어
4. **실시가능성**: 합성 데이터 파이프라인 유효성, 엣지 실시간 실측 검증
5. **명세서 보강**: 정량 데이터 삽입, 실험 결과 기반 명세서 수정

---

## 2. 제약 조건

| 항목 | 조건 |
|------|------|
| 훈련 | Colab (T4/A100) |
| 추론 실측 | RP5 (ARM CPU), 가능 시 Jetson Orin NX |
| 데이터 | 합성 기반 + 오픈소스 실제 녹음 하이브리드 |
| 시간 | **유동적** (기존 1주일 → 품질 우선) |
| 프레임워크 | PyTorch, ONNX Runtime |

---

## 3. 실험 구조 총괄

```
[데이터] 합성 + 하이브리드 파이프라인 (실험 3)
    ├── Military 데이터셋 (충격음 + 극저 SNR + 고포화율)
    └── General 데이터셋 (정상 소음 + 표준 SNR)
           ↓
[모델 1: DPCRN 듀얼 인코더] (우선)
    ├── A-hard: Hard Mask 융합
    ├── A-soft: Soft Mask 융합 (주 실시예)
    ├── A-param: Parametric Mask 융합
    ├── B: Concatenation (마스크 없음)
    └── C: BC only
           ↓
[모델 2: 2D Conv U-Net 듀얼 인코더] (후순위)
    └── 동일 5변형
           ↓
[평가] 동일 테스트셋, 동일 지표
           ↓
[엣지 실측] ONNX 변환 → RP5/Jetson 실측 (실험 4)
           ↓
[벤치마크] 필요 시 Denoiser, GTCRN 등 추가
```

---

## 4. 데이터셋 설계 (실험 3)

### 4.1 공통 기반

| 항목 | 사양 |
|------|------|
| 음성 소스 | LibriSpeech train-clean-100 (~20h subset) |
| 분할 | 화자 기준(speaker-independent) train 70% / val 15% / test 15% |
| 샘플링 | 16kHz, 세그먼트 4초 |
| BC 시뮬 | IIR 필터 (2차 피킹 EQ 1.5kHz, Q=2 + 4차 Butterworth LPF 4kHz) |
| 신체 잡음 | 주기성 심박 (1Hz sine, -30dB) + 랜덤 임펄스 (보행, 0.5~10Hz) |
| 포화 시뮬 | 하드/소프트 클리핑 + 회복 시간 (지수 감쇠, τ=2~5ms) + 다항식 비선형 |

### 4.2 환경별 소음 설계

#### Military (군용)

| 항목 | 사양 |
|------|------|
| **실제 녹음 (50%)** | Freesound/AudioSet: 총성, 폭발음, 군용 차량, 항공기 |
| **수식 합성 (50%)** | 아래 충격음 합성 모델 |
| **증강** | 무작위 RIR 적용, 클리핑 레벨 변조, 다중 소음 중첩 |
| SNR | -20 ~ +5 dB |
| 포화율 | 30~100% (고포화 비중) |
| 충격음 비율 | 40~60% |

충격음 합성 모델:
```python
# 총성
def gunshot(duration_ms=50, fs=16000):
    t = np.arange(0, duration_ms/1000, 1/fs)
    # 초기 충격파: 감쇠 정현파 (1~4kHz)
    f0 = np.random.uniform(1000, 4000)
    tau_attack = np.random.uniform(0.001, 0.005)
    attack = np.sin(2*np.pi*f0*t) * np.exp(-t/tau_attack)
    # 마즐 블래스트: 광대역 버스트
    blast = np.random.randn(len(t)) * np.exp(-t/0.003)
    # 잔향: 지수 감쇠
    tau_reverb = np.random.uniform(0.05, 0.2)
    reverb = np.random.randn(int(tau_reverb*fs*3)) * np.exp(-np.arange(int(tau_reverb*fs*3))/(tau_reverb*fs))
    return normalize(np.concatenate([attack + blast, reverb]))

# 폭발음
def explosion(duration_ms=200, fs=16000):
    t = np.arange(0, duration_ms/1000, 1/fs)
    # 저주파 충격파 (20~200Hz)
    f0 = np.random.uniform(20, 200)
    shock = np.sin(2*np.pi*f0*t) * np.exp(-t/0.05)
    # 광대역 잔향
    reverb = np.random.randn(int(0.5*fs)) * np.exp(-np.arange(int(0.5*fs))/(0.1*fs))
    return normalize(np.concatenate([shock, reverb]))
```

#### General (일반)

| 항목 | 사양 |
|------|------|
| 소음 DB | DEMAND, MUSAN, WHAM! |
| SNR | -10 ~ +15 dB |
| 포화율 | 0~50% (저포화 비중) |
| 충격음 | 0~10% |

### 4.3 포화 마스크 생성 (3유형)

```python
# 공통: 시간 영역 클리핑 검출
def detect_clipping(signal, threshold_ratio=0.95, consecutive=3):
    fullscale = np.max(np.abs(signal))
    threshold = fullscale * threshold_ratio
    clipped = np.abs(signal) >= threshold
    # 연속 N개 이상 검출 → 포화 판정
    ...
    return clipped_regions  # 시간 영역 boolean

# 1. Hard Mask: 포화 구간의 전체 freq bin을 1, 아닌 곳 0
def hard_mask(clipped_regions, stft_params):
    mask = np.zeros((n_frames, n_freq))
    for frame in frames_containing_clipping:
        mask[frame, :] = 1.0
    return mask

# 2. Soft Mask: 프레임 내 클리핑 비율 → 시그모이드
def soft_mask(clipped_regions, stft_params, temperature=0.1):
    mask = np.zeros((n_frames, n_freq))
    for frame_idx, frame_samples in enumerate(frames):
        clip_ratio = np.mean(clipped_regions[frame_samples])
        mask[frame_idx, :] = sigmoid(clip_ratio / temperature)
    return mask

# 3. Parametric Mask: 포화 후 회복 시간 반영
def parametric_mask(clipped_regions, stft_params, tau_recovery_ms=5):
    soft = soft_mask(clipped_regions, stft_params)
    # 포화 종료 후 지수적 감쇠 tail 추가
    for frame_idx in range(n_frames):
        if soft[frame_idx, 0] > 0 and (frame_idx+1 < n_frames) and soft[frame_idx+1, 0] == 0:
            # 포화 종료 지점 → 감쇠 tail
            for k in range(1, max_tail_frames):
                decay = np.exp(-k * hop_ms / tau_recovery_ms)
                if frame_idx + k < n_frames:
                    soft[frame_idx + k, :] = max(soft[frame_idx + k, :], soft[frame_idx, :] * decay)
    return soft
```

---

## 5. 모델 구조

### 5.1 DPCRN 기반 듀얼 인코더 (우선)

```
[BC Encoder] Complex Conv2D × 5층
  채널: 16→32→64→128→256, 커널: (2,3), BN+PReLU

[AC Encoder] 동일 구조 (파라미터 비공유)

[Adaptive Fusion] ★ Magnitude 기반 어텐션 게이트 (v3 수정)
  ┌─────────────────────────────────────────────────┐
  │ 입력: |BC_feat| (magnitude), |AC_feat|, soft_mask │
  │ concat → Conv2D 1×1 (실수) → Sigmoid → weight w  │
  │ 출력: w × BC_feat + (1-w) × AC_feat (복소수 곱)   │
  │                                                   │
  │ ★ 가중치 w는 실수 (0~1)                            │
  │ ★ 복소수 특징맵에 스칼라 곱 → 위상 보존             │
  └─────────────────────────────────────────────────┘

[Dual-Path Module]
  시간 경로: Bi-LSTM (hidden=128) → FC
  주파수 경로: FC → PReLU → FC
  교대 2회 반복

[Decoder] Complex ConvTranspose2D × 5층 + Skip connections

[Output] CRM → 향상 스펙트로그램 → iSTFT

[BWE] FC(40→256) + FC(256→256) + Conv1D(256→64), ~300K
```

총 파라미터: ~2M

### 5.2 2D Conv U-Net 듀얼 인코더 (후순위)

```
[BC Encoder] Real Conv2D × 5층
  채널: 16→32→64→128→256, 커널: (3,3), BN+LeakyReLU

[AC Encoder] 동일 구조 (파라미터 비공유)

[Adaptive Fusion] Magnitude 기반 어텐션 게이트 (동일 구조, 실수 도메인)

[Bottleneck] 2층 LSTM (hidden=256)

[Decoder] ConvTranspose2D × 5층 + U-Net Skip connections

[Output] IRM (진폭 마스크 + 입력 위상 재사용)
  - 선택적으로 CRM 확장 가능

[BWE] 동일 구조
```

총 파라미터: ~2M

### 5.3 Ablation 5변형 (양 모델 공통)

| ID | 입력 | 융합 | 마스크 유형 | 목적 |
|----|------|------|-----------|------|
| **A-soft** | BC+AC+Mask | 어텐션 게이트 | Soft (시그모이드) | 주 실시예 |
| **A-hard** | BC+AC+Mask | 어텐션 게이트 | Hard (이진) | 이진 마스크 효과 |
| **A-param** | BC+AC+Mask | 어텐션 게이트 | Parametric (회복 시간) | 회복 시간 반영 효과 |
| **B** | BC+AC | Concatenation | 없음 | '152 특허 유사 (기준선) |
| **C** | BC only | — | — | 골전도 단독 |

### 5.4 벤치마크 후보 (시간 여유 시)

| 모델 | 목적 | 우선순위 |
|------|------|---------|
| Meta Denoiser | 시간 도메인 기준선, 기존 경험 활용 | P2 |
| GTCRN (24K) | 제1 모드 초경량 후보 | P2 |
| DCCRN (3.7M) | DPCRN 상위 비교 | P3 |

---

## 6. 훈련 설정

### 6.1 손실 함수 (v3 변경)

```python
# v2: SI-SNR + 0.3 × STOI (미분 문제)
# v3: SI-SNR + α × MR-STFT Loss

def total_loss(enhanced, clean):
    l_sisnr = -si_snr(enhanced, clean)

    # Multi-Resolution STFT Loss
    # 3개 해상도: (n_fft, hop, win)
    resolutions = [(512, 50, 240), (1024, 120, 600), (2048, 240, 1200)]
    l_mrstft = 0
    for n_fft, hop, win in resolutions:
        # Spectral Convergence + Log Magnitude Loss
        S_enh = stft(enhanced, n_fft, hop, win)
        S_clean = stft(clean, n_fft, hop, win)
        l_sc = ||S_clean - S_enh||_F / ||S_clean||_F       # Spectral Convergence
        l_mag = ||log(S_clean) - log(S_enh)||_1              # Log Magnitude
        l_mrstft += (l_sc + l_mag)
    l_mrstft /= len(resolutions)

    return l_sisnr + alpha * l_mrstft  # alpha=0.5 초기값, 탐색 대상
```

- **STOI**: 평가 지표로만 사용 (훈련에 미사용)
- **alpha**: 0.3 / 0.5 / 1.0 탐색 (val loss 기준 선택)

### 6.2 공통 하이퍼파라미터

| 항목 | 값 |
|------|-----|
| Loss | SI-SNR + α × MR-STFT (α=0.5 기본) |
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-2 |
| Scheduler | CosineAnnealingWarmRestarts (T_0=10) |
| Epochs | 50 (early stopping patience=10) |
| Seeds | 3 |
| Batch size | 8 (T4) / 16 (A100) |
| STFT | 윈도우 25ms (400 samples), 홉 10ms (160 samples), NFFT 512, Hann |

### 6.3 훈련 규모

| 항목 | 수량 |
|------|------|
| Ablation 변형 | 5 (A-soft, A-hard, A-param, B, C) |
| 모델 백본 | 2 (DPCRN, U-Net) |
| Seeds | 3 |
| **총 훈련 횟수** | **30회** |
| 1회 추정 시간 | A100: ~3h / T4: ~6h |
| 총 시간 | A100: ~90h / T4: ~180h |

시간 관리:
- DPCRN 5변형 × 3 seeds = 15회 우선 (~45h A100)
- 중간 결과 확인 → U-Net 15회 후순위 (~45h A100)
- alpha 탐색은 DPCRN A-soft에서 3값 × 1 seed로 사전 수행 (~9h)

---

## 7. 평가 설계

### 7.1 평가 매트릭스

```
환경: {Military, General}
포화율: {S0(0%), S1(30%), S2(50%), S3(70%), S4(100%)}
백본: {DPCRN, U-Net}
Ablation: {A-soft, A-hard, A-param, B, C}
```

총 조합: 2 × 5 × 2 × 5 = **100 조건**
각 조건: PESQ / STOI / SI-SNR (평균 ± std, 3 seeds)

### 7.2 핵심 비교축

| 비교 | 입증 목표 | 명세서/특허 대응 |
|------|----------|----------------|
| A-soft vs B | 포화 마스크 효과 (핵심) | 진보성 #15 |
| A-soft vs A-hard vs A-param | 마스크 유형별 효과 | 청구범위 방어 (§5.2 A.2.5) |
| A vs C | AC 채널 기여도 | 멀티모달 융합 가치 |
| DPCRN-A vs UNet-A | 백본 비교 | 최적 구조 선정 |
| Military vs General | 군용 특화 효과 | 전장 적용 가치 |
| S0→S4 추이 | 포화 심화 시 마스크 효과 증가 | 적응적 융합 필연성 |

### 7.3 교차 환경 평가

| 훈련 | 테스트 | 목적 |
|------|--------|------|
| Military | Military | In-domain |
| Military | General | 범용성 |
| General | Military | 전장 특화 필요성 |
| Mixed | Military | 혼합 학습 효과 |
| Mixed | General | 혼합 학습 효과 |

### 7.4 BWE 평가 (실험 2)

Model C + BWE 유/무, 양 백본에서 비교.
지표: PESQ-WB, STOI

### 7.5 통계 처리

- 3 seeds 평균 ± std 보고
- 핵심 비교(A-soft vs B)에 대해 paired t-test (p < 0.05)
- 결과 테이블 + 박스플롯/바 그래프

---

## 8. 엣지 실측 벤치마크 (실험 4, v3 확장)

### 8.1 변환 파이프라인

```
PyTorch 모델 (.pt)
    ↓ torch.onnx.export()
ONNX 모델 (.onnx)
    ↓ onnxruntime / onnx-simplifier
ONNX 최적화 모델
    ↓
┌─────────────────────────────────┐
│ 실측 환경:                       │
│  (1) Colab CPU (참조)            │
│  (2) RP5 ARM CPU (실측)          │
│  (3) Jetson Orin NX GPU (가능 시) │
└─────────────────────────────────┘
```

### 8.2 양자화 실험

| 정밀도 | 변환 방법 | 측정 항목 |
|--------|----------|----------|
| FP32 | ONNX 기본 | 추론 시간, PESQ |
| FP16 | ONNX float16 변환 | 추론 시간, PESQ |
| INT8 | ONNX Runtime 동적 양자화 | 추론 시간, PESQ |

### 8.3 지연 분해 테이블 (목표)

| 단계 | DPCRN (RP5) | U-Net (RP5) | DPCRN (Jetson) | 비고 |
|------|-------------|-------------|----------------|------|
| STFT (25ms) | ~25ms | ~25ms | ~25ms | 알고리즘 지연 |
| 모델 추론 | 실측 | 실측 | 실측 | FP32/FP16/INT8 |
| iSTFT + 후처리 | 실측 | 실측 | 실측 | |
| **합계** | **실측** | **실측** | **실측** | 목표 ≤40ms |
| DSP-only (폴백) | 실측 | — | — | 빔포밍+위너 |

---

## 9. 일정 (유동적)

```
[Phase A] 기반 구축
  D1: 합성 파이프라인 구현
      — IIR BC 필터, 포화 시뮬 (3유형 마스크), 충격음 하이브리드
      — 소음 DB 수집 (DEMAND, MUSAN, Freesound 군용)
      — Military/General 데이터셋 생성 + 분포 검증

  D2: 공통 모듈 구현
      — Adaptive Fusion (Magnitude 기반 어텐션 게이트)
      — MR-STFT Loss
      — 평가 코드 (PESQ/STOI/SI-SNR)
      — BWE 서브모듈

[Phase B] DPCRN 실험 (우선)
  D3: DPCRN 듀얼 인코더 구현
  D4: alpha 탐색 (A-soft, 3값 × 1 seed)
  D5-6: DPCRN 5변형 × 3 seeds 훈련
  D7: DPCRN 중간 평가 + 결과 확인

[Phase C] U-Net 실험
  D8: 2D Conv U-Net 듀얼 인코더 구현 (공통 모듈 재사용)
  D9-10: U-Net 5변형 × 3 seeds 훈련
  D11: U-Net 평가

[Phase D] 교차 비교 + 엣지 실측
  D12: 전체 평가 매트릭스 산출 (100 조건)
  D13: 교차 환경 평가
  D14: ONNX 변환 + RP5 실측 지연
  D15: 양자화 (FP16/INT8) 성능·지연 비교

[Phase E] 정리
  D16: 결과 테이블/그래프 정리
  D17: 모델 선정 + 명세서 반영 방향 확정
  D18: 명세서 수정 (§7 실험 결과 삽입)
```

---

## 10. 결과 → 명세서 반영

### 10.1 손실 함수 수정

§7.3.3 (f) 현재: "SI-SNR + 0.3 × STOI"
→ 수정: "SI-SNR 손실 및 다중 해상도 STFT 손실(Multi-Resolution STFT Loss)의 가중 합으로 구성된다. 일 실시예에서 $\mathcal{L} = \mathcal{L}_{\text{SI-SNR}} + \alpha \cdot \mathcal{L}_{\text{MR-STFT}}$, $\alpha = 0.5$를 사용한다."

### 10.2 어텐션 게이트 수정

§7.3.4 (a) 현재: "1×1 컨볼루션 및 시그모이드 활성화 함수에 의해 산출"
→ 보강: "입력 특징의 크기(magnitude)로부터 실수 가중치를 산출하고, 상기 실수 가중치를 복소수 특징 맵에 스칼라 곱으로 적용하여 위상 정보를 보존한다"

### 10.3 포화 마스크 유형 추가

§5.2 A.2.5에 이미 이진/소프트 기재.
A-param이 유효하면 추가: "포화 구간 종료 후의 센서 회복 시간을 반영하여 시간적 감쇠 특성을 포함하는 파라메트릭 마스크를 더 포함할 수 있다"

### 10.4 모델 선정 결과

| 시나리오 | 조치 |
|---------|------|
| DPCRN 우세 | §7.3.3을 DPCRN 기반으로 수정. "듀얼패스 구조를 포함하는 복소수 영역 인코더-디코더" |
| U-Net 우세 | §7.3.3 현행 유지. DPCRN을 변형예 |
| 동등 | 양 구조 병렬 실시예 |

### 10.5 실험 결과 삽입

- §7 신규 §7.3.7: 포화율별 PESQ/STOI 테이블 + 마스크 유형별 비교
- 도 6 (신규): 포화율 vs PESQ 그래프 (A-soft/A-hard/A-param/B/C)
- 필수 단서: "시뮬레이션 기반 합성 데이터를 이용한 예비 실험 결과이며, 실 운용 환경에서의 성능은 차이가 있을 수 있다"

---

## 11. 새 대화 전달 파일

1. patent_revised_v3_final.tex — 명세서 현행본
2. **이 파일 (experiment_design_v3.md)** — 실험 설계 최종본
3. experiment_roadmap_final.md — Phase 1 검토 결과
4. SOTA 조사 결과 (대화 내 텍스트)

---

## 12. GitHub Repo 구조

```
bone-conduction-enhancement/
├── README.md
├── docs/
│   └── experiment_design_v3.md
├── configs/
│   ├── dpcrn_config.yaml
│   ├── unet_config.yaml
│   └── data_config.yaml
├── data/
│   ├── pipeline.py              # 합성 파이프라인 통합
│   ├── bc_simulator.py          # IIR 골전도 시뮬
│   ├── saturation.py            # 포화 시뮬 + 3유형 마스크
│   ├── impulse_generator.py     # 충격음 합성 (총성/폭발)
│   ├── noise_mixer.py           # 소음 혼합 (하이브리드)
│   └── dataset.py               # PyTorch Dataset/DataLoader
├── models/
│   ├── dpcrn_dual.py            # DPCRN 듀얼 인코더
│   ├── unet_dual.py             # 2D Conv U-Net 듀얼 인코더
│   ├── fusion.py                # Magnitude 기반 어텐션 게이트 (공통)
│   ├── bwe.py                   # 대역 확장 서브모듈
│   └── loss.py                  # SI-SNR + MR-STFT Loss
├── train.py
├── evaluate.py
├── export_onnx.py               # ONNX 변환
├── benchmark_latency.py         # 지연 측정 (PyTorch + ONNX)
├── notebooks/
│   ├── train_colab.ipynb
│   └── analysis.ipynb           # 결과 분석 + 그래프
└── results/
    ├── dpcrn/
    └── unet/
```
