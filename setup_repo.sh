#!/bin/bash
# tactical-speech-enhancement 초기 설정 스크립트
# 사용법: chmod +x setup_repo.sh && ./setup_repo.sh

set -e

REPO_NAME="tactical-speech-enhancement"

# 1. GitHub repo 생성 (gh CLI 필요, 없으면 수동 생성)
echo "=== GitHub repo 생성 ==="
if command -v gh &> /dev/null; then
    gh repo create "$REPO_NAME" --private --clone
    cd "$REPO_NAME"
else
    echo "gh CLI 미설치. GitHub에서 수동으로 private repo 생성 후:"
    echo "  git clone https://github.com/<USERNAME>/$REPO_NAME.git"
    echo "  cd $REPO_NAME"
    echo ""
    read -p "수동 생성 후 Enter를 누르세요 (이미 clone한 디렉토리에서 실행)..."
    cd "$REPO_NAME" 2>/dev/null || true
fi

# 2. 디렉토리 구조 생성
echo "=== 디렉토리 구조 생성 ==="
mkdir -p docs configs data models notebooks results

# 3. .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
*.egg

# 가상환경
venv/
.venv/
env/

# 데이터 (대용량, repo에 포함 X)
data/raw/
data/processed/
*.wav
*.flac
*.tar.gz
*.zip

# 모델 체크포인트
*.pt
*.pth
*.onnx
checkpoints/

# 결과 (선택적 추적)
results/**/*.wav

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/
*.ipynb_metadata

# OS
.DS_Store
Thumbs.db

# 환경변수
.env
EOF

# 4. requirements.txt
cat > requirements.txt << 'EOF'
torch>=2.0
torchaudio>=2.0
numpy
scipy
librosa
pesq
pystoi
onnx
onnxruntime
matplotlib
pyyaml
soundfile
tqdm
tensorboard
EOF

# 5. README.md
cat > README.md << 'EOF'
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
EOF

# 6. 빈 디렉토리에 .gitkeep
touch configs/.gitkeep
touch data/.gitkeep
touch models/.gitkeep
touch notebooks/.gitkeep
touch results/.gitkeep

# 7. 초기 커밋
echo "=== 초기 커밋 ==="
git add -A
git commit -m "init: 프로젝트 구조 + 실험 설계서"

echo ""
echo "=== 완료 ==="
echo "다음 단계:"
echo "  1. docs/ 에 experiment_design_v3.md 와 patent_revised_v3_final.tex 복사"
echo "  2. git add docs/ && git commit -m 'docs: 실험 설계서 + 명세서 추가'"
echo "  3. git push origin main"
echo "  4. Claude Code에서 시작"
