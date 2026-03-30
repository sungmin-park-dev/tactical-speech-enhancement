# Tactical Speech Enhancement

골전도 음향 장치 및 딥러닝을 활용한 극단적 소음 제거 — 실험 코드베이스

## 구조

```
├── docs/           # 실험 설계서, 명세서
├── configs/        # 하이퍼파라미터 yaml
├── data/           # 합성 파이프라인, Dataset
├── models/         # DPCRN, U-Net, 융합 블록, BWE
├── notebooks/      # Colab 훈련, 결과 분석
├── results/        # 실험 결과
├── train.py
├── evaluate.py
├── export_onnx.py
└── benchmark_latency.py
```

## 실험 개요

- **모델**: DPCRN 듀얼 인코더 vs 2D Conv U-Net 듀얼 인코더
- **핵심**: 포화 마스크 기반 적응적 융합 (Ablation 5변형)
- **환경**: Military / General
- **상세**: `docs/experiment_design_v3.md` 참조
