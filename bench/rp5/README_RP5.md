# RP5 측정 절차 (소유자 실행용)

이 폴더의 스크립트는 저장소 코드를 수정하지 않고 외부에서 계측합니다. 각 Pi에서 저장소를 설치한 가상환경(`pip install -e '.[audio]'`, `tse model fetch`)을 활성화한 뒤 실행합니다. 모든 결과에 **Pi 모델·OS 이미지·커널·Python·onnxruntime/torch 버전·냉각(팬/방열판/케이스)·전원 어댑터·헤드셋 모델·연결 방식(유선/블루투스)·입력 이득**을 함께 기록하세요.

## 0. 준비 (각 Pi)

```bash
sudo apt install -y libportaudio2 libsndfile1
python -m pip install scipy
```

Demucs까지 측정하려면 `pip install torch`(aarch64 휠)와 공식 denoiser 소스(`denoiser/` 폴더: `demucs.py`, `resample.py`, `utils.py`, `__init__.py`), 가중치(`dns48-11decc9d8e3f0998.th`, `dns64-a7761ff99a7d5bb6.th`)가 필요합니다. denoiser는 CC BY-NC 4.0입니다.

## 1. 모델 처리 시간 (약 15–30분)

```bash
python model_rtf.py --gtcrn ~/.cache/tactical-speech-enhancement/gtcrn_simple.onnx \
  --dtln-dir ./dtln --demucs-dir ./fb_denoiser --demucs-weights ./weights \
  --seconds 60 --threads 1 2 4 --out rtf_pi5.json
```

판정: 각 모델의 `p99_ms`가 홉 길이(GTCRN 16 ms, DTLN 8 ms, Demucs 16 ms)보다 충분히 작아야 합니다(권장 p99 < 홉의 60%). `first_vs_last_minute_mean_ms`가 커지거나 `soc_after.throttled`가 0x0이 아니면 열·전원 문제입니다.

## 2. 30분 통화 계측 (두 Pi 동시에)

```bash
# Pi A
python peer_monitor.py --bind A_IP:5000 --peer B_IP:5000 --input-device N --output-device M \
  --duration 1800 --outdir run_A --dump-audio
# Pi B
python peer_monitor.py --bind B_IP:5000 --peer A_IP:5000 --input-device N --output-device M \
  --duration 1800 --outdir run_B --dump-audio
```

`run_*/per_second.jsonl`에 초 단위로 스레드별 CPU, 추론 지연·마감 초과, 원음 전환·음소거 프레임, 지터 버퍼 수위, 소켓 수신 대기열, 콜백 시간, 입력 overflow/출력 underflow 횟수, SoC 온도·클럭이 남습니다. `--dump-audio`는 마이크 원시 입력(`tap_mic_raw.wav`), 송신 프레임(`tap_tx_sent.wav`), 수신 재생(`tap_rx_play.wav`)을 저장합니다(16 kHz PCM16, 30분당 탭별 약 58 MB).

판정 기준:
- 지터 수위가 4에서 7–8로 올라 머무르면 → 버퍼 래칫(시계 차·Wi-Fi 끊김).
- 추론 지연이 16 ms에 근접·초과하고 `c_dry_fallback_frames`가 증가 → CPU·열 문제(잡음 제거 꺼짐).
- `xrun_in_total`/`xrun_out_total`이 늘 때마다 `c_muted_source_frames`가 같이 늘면 → xrun이 음소거를 유발(D4).
- `sock_rx_queue_bytes`가 계속 0이 아니면 → 수신 스레드가 밀림(GIL·CPU).

## 3. 입-귀 종단 지연 (공통 녹음기 필요)

두 Pi의 시계는 동기화되어 있지 않으므로 타임스탬프 차이로 지연을 계산하면 안 됩니다. 노트북 + 2채널 USB 오디오 인터페이스(또는 2채널 녹음기)를 씁니다.

1. `python e2e_latency.py make --speech-dir LibriSpeech/test-clean --minutes 40 --out ref_signal.wav`
2. 작은 스피커로 `ref_signal.wav`를 Pi A 마이크 옆에서 재생하고, 같은 신호(스피커 입력 전기 신호 또는 기준 마이크)를 녹음기 ch0에 연결합니다.
3. Pi B 헤드셋 출력을 녹음기 ch1에 연결합니다(헤드폰 단자 분기, 또는 골전도 트랜스듀서에 붙인 소형 마이크/접촉 마이크).
4. 40분 녹음 후 `python e2e_latency.py analyse --recording two_channel.wav --out latency.json`.

해석: `trend_ms_per_minute`와 `step_changes_over_10ms`를 봅니다. 선형으로 빠르게(분당 100 ms 이상) 늘면 처리 지연(RTF>1)+무한 대기열, 분당 1–20 ms면 시계 드리프트, 망 끊김 직후 계단식 증가 후 복귀 없음이면 지터 버퍼 래칫입니다. 같은 녹음으로 2.의 로그와 시각을 맞추면 원인을 직접 대조할 수 있습니다.

## 4. "폭음 앞뒤 클리핑" 발생 지점 판별

1. 같은 폭음 파일을 재생 레벨 5단계(−24, −18, −12, −6, 0 dB)로, 마이크 이득 2단계에서 반복합니다.
2. 탭: T1 시험 스피커 기준(전기 신호 또는 고AOP 측정 마이크), T2 `tap_mic_raw.wav`, T3 `tap_tx_sent.wav`, T4 상대 Pi의 `tap_rx_play.wav`, T5 트랜스듀서 출력(외부 녹음).
3. `python clip_source.py --t1 t1.wav --t2 tap_mic_raw.wav --t3 tap_tx_sent.wav --t4 tap_rx_play.wav --t5 t5.wav --out clip_source.json`

판정: 평탄 상단(`flat_top_runs`)·전후 고역 에너지 상승이 처음 나타나는 탭이 발생 지점입니다. T1부터 있으면 시험 장비, 마이크 이득을 따라 변하면 마이크/ADC/AGC, T3부터면 모델·보호 제어·리미터, T5에서만이면 트랜스듀서입니다.

주의: 고음압 재생은 청력에 위험합니다. 시험 중 귀마개를 착용하고 음량을 단계적으로 올리세요.
