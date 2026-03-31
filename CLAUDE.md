# CLAUDE.md — AI 어시스턴트 컨텍스트

이 파일은 AI 코딩 어시스턴트가 프로젝트를 빠르게 파악하기 위한 참조 문서입니다.

## 프로젝트 목적

골전도(BC) + 공기전도(AC) 융합 기반 음성 향상 특허의 **진보성 입증**을 위한 실험.
핵심 차별점: **포화 마스크 기반 적응적 융합** (Magnitude Attention Gate).

## 코드 구조 & 의존 관계

```
configs/data_config.yaml
    ↓ (load_config)
data/pipeline.py
    ├── data/bc_simulator.py      # BCSimulator (IIR 필터 + 신체잡음)
    ├── data/saturation.py        # SaturationSimulator (3종 마스크)
    ├── data/impulse_generator.py # generate_impulse_noise, ImpulseGenerator
    └── data/noise_mixer.py       # NoiseMixer → MilitaryNoiseMixer / GeneralNoiseMixer

models/fusion.py                  # AdaptiveFusion (5 ablation 변형)
    └── MagnitudeAttentionGate, ConcatFusion

models/loss.py                    # TotalLoss = SI-SNR + α × MR-STFT
models/bwe.py                     # BWESubmodule (~300K params)
evaluate.py                       # PESQ/STOI/SI-SNR + paired t-test
```

## 구현 상태 (2026-03-31)

- [x] **D1** — 합성 파이프라인 (data/ 전체)
- [x] **D2** — 공통 모듈 (fusion, loss, bwe, evaluate)
- [ ] **D3** — DPCRN 듀얼 인코더 (models/dpcrn_dual.py)
- [ ] **D4** — 훈련 루프 (train.py)
- [ ] **평가** — 100 조건 매트릭스 + 통계 검정

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
  Step 2: 훈련 (D3 완료 후)
  Step 3: 결과 tar.gz → Drive 저장
```

> **중요**: 커널 재시작 시 반드시 Step 0부터 재실행 (sys.path 초기화됨)

## 다음 구현 순서

1. **models/dpcrn_dual.py** — DPCRN 듀얼 인코더
   - BC 인코더 + AC 인코더 → AdaptiveFusion → 공유 디코더
   - 복소수 STFT 도메인 처리, BWE 서브모듈 연결
2. **train.py** — 훈련 루프
   - 5 ablation × 2 backbone × 3 seeds = 30 실험
   - TotalLoss (SI-SNR + α × MR-STFT)
   - 체크포인트/로그 저장
3. **평가 매트릭스** — evaluate.py의 `evaluate_full_matrix()`
   - 100 조건: 2 환경 × 5 포화율 × 2 백본 × 5 ablation
   - paired t-test: A-soft vs B 유의성 검정

## 핵심 설계 결정

| 항목 | 결정 | 근거 |
|------|------|------|
| 가중치 w의 도메인 | 실수 (Sigmoid) | 복소수 특징에 스칼라 곱 → 위상 보존 |
| 마스크 유형 | soft 기본 | 실험 설계서 v3 §5.1 |
| MR-STFT 해상도 | 3종 (512/1024/2048) | Yamamoto et al. 표준 |
| α (MR-STFT 가중치) | 0.5 기본 (0.3/1.0 탐색) | 실험 설계서 v3 §6.3 |
| 화자 분할 | speaker-independent | LibriSpeech 경로 기반 화자 ID 추출 |
| 데이터 전송 | 단일 tar.gz | Google Drive I/O 할당량 제한 대응 |

## 참고 문서

- `docs/experiment_design_v3.md` — 실험 설계서 (모든 사양의 원천)
- `docs/patent_revised_v3_final.tex` — 특허 명세서
