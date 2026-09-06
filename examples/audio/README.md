# 음성 처리 전후 예제

고정한 공개 음성으로 실제 GTCRN 모델과 입력 보호·출력 제한을 실행한 파일이다.
모두 11초이며, 앞 1초는 시작 보호를 완료하기 위한 무음이다. 나머지 10초는
LibriSpeech test-clean의 화자 61 시험 음성을 사용한다.

## 듣기와 내려받기

| 파일 | MP3 듣기용 | WAV 검증용 |
|---|---|---|
| 잡음을 더하기 전 기준 음성 | [MP3](clean-reference.mp3) | [WAV](clean-reference.wav) |
| 배경 잡음 입력 | [MP3](noise-input.mp3) | [WAV](noise-input.wav) |
| 배경 잡음 처리 결과 | [MP3](noise-output.mp3) | [WAV](noise-output.wav) |
| 충격·포화 입력 | [MP3](impulse-input.mp3) | [WAV](impulse-input.wav) |
| 충격·포화 처리 결과 | [MP3](impulse-output.mp3) | [WAV](impulse-output.wav) |

GitHub에서 미리보기를 지원하지 않으면 파일을 내려받아 재생한다.
WAV는 모노·16 kHz·PCM16이다. MP3는 같은 WAV를 64 kbps로 변환한 듣기용
사본으로, 압축 과정에서 샘플값이 바뀌므로 정확한 0과 디지털 피크 상한은
WAV에서 검증한다. 처리 결과를 별도로 증폭하거나 음량 정규화하지 않았다.

## 생성 조건

- **기준 음성:** 저장소의 `speaker-0061.wav`를 피크 0.30으로 조정하고 앞에
  1초의 무음을 추가했다. 원본 음성의 상세 구간과 출처는
  [시험 자료 기록](../../tests/fixtures/speech/manifest.json)에 있다.
- **배경 잡음:** 고정 난수 시드 `20260906`으로 만든 백색 잡음과 120 Hz
  사인파를 혼합했다. 무음을 제외한 10초 전체의 RMS를 기준으로 신호 대
  잡음비를 5dB로 맞췄다. 실제 현장의 녹음이 아니다.
- **충격·포화:** 잡음 입력을 바탕으로, 파일 시작 기준 4.000초와 4.192초에
  각각 5ms·최대 진폭 0.95의 합성 잡음 펄스를 삽입했다. 두 번째 충격은 보호
  이득의 복귀 도중 발생한다. 7.000초에는 50ms·최대 진폭 0.999로 클리핑한
  잡음 펄스를 넣어 포화를 만들었다.
  충격 구간은 원래 샘플을 교체하며 음성 보존을 평가하지 않는다.
- **처리:** `tse process`와 동일한 파일 처리 경로에 기본 설정과 무결성을
  확인한 GTCRN ONNX를 적용했다. 파일의 길이를 유지하고 고정 2프레임 정렬
  지연을 제거했다. 실시간 마감이나 통신 지연을 측정한 파일이 아니다.

## 확인한 결과

| 항목 | 배경 잡음 처리 | 충격·포화 처리 |
|---|---|---|
| 출력 길이·샘플레이트 | 176,000샘플·16 kHz | 176,000샘플·16 kHz |
| 출력 피크 | −10.91dBFS | −11.26dBFS |
| −6dBFS 상한 | 통과 | 통과 |
| 입력 이상 프레임 | 0개 | 6개 |
| 보호 이득이 0인 구간의 출력 누출 | 0샘플 | 0샘플 |
| 마지막 보호 이득 | 1.0 | 1.0 |

이번 예제 추가 시 기존 자동 테스트 **187개가 모두 통과**했으며 건너뛴 시험은
없었다. 실행 환경은 macOS arm64·Python 3.13.5다. 정확한 설정, 모델·음성
파일 해시, 패키지 버전과 처리 결과는 [생성·검증 기록](manifest.json)에 있다.
실행 시간은 파일 처리 경로에서 측정한 값이며 Pi 성능을 의미하지 않는다.

이 예제는 잡음 감소와 음소거·복귀 동작을 들어보기 위한 자료다. 특정 녹음
하나로 전반적인 음질 향상이나 실제 기기 성능을 입증하지 않는다. 음질 점수와
실제 Pi의 종단 지연은 측정하지 않았다.

## 다시 만들기

프로젝트를 설치하고 모델을 받은 뒤 저장소 최상위 폴더에서 실행한다.

```sh
tse model fetch
python scripts/make_audio_examples.py --output-dir /tmp/tse-audio-examples
```

MP3도 만들려면 `libmp3lame` 인코더가 포함된 FFmpeg를 설치하고 실행한다.

```sh
python scripts/make_audio_examples.py --output-dir /tmp/tse-audio-examples --mp3
```

동일한 입력 WAV를 개별 처리할 수도 있다.

```sh
tse process examples/audio/noise-input.wav /tmp/noise-output.wav
tse process examples/audio/impulse-input.wav /tmp/impulse-output.wav
```

스크립트는 원본 시험 음성의 해시를 검사하고 생성 파일의 길이·출력 상한·
보호 차단 구간을 확인한다. 입력은 고정된 방법으로 만들지만 모델 실행
환경이나 MP3 인코더 버전에 따라 출력 파일의 해시가 달라질 수 있다.

## 출처와 라이선스

음성 출처는 [LibriSpeech / OpenSLR SLR12](https://www.openslr.org/12/)이며
자료 저작자는 Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur다.
원 오디오북 녹음은 LibriVox에서 왔다. 이 폴더의 WAV·MP3에는
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)을 적용한다.
[라이선스 원문](../../licenses/CC-BY-4.0.txt)과
[외부 구성요소·데이터 라이선스 안내](../../THIRD_PARTY_NOTICES.md)를 함께 제공한다.

원본 발화 선택·연결·10초 자르기 이후, 음량 조정·무음 추가·합성 잡음과 충격
삽입·GTCRN 처리·MP3 변환을 수행했다. 각 파일의 변경 내용은 위 표와 생성
조건에 대응한다. 원저작자가 이 프로젝트나 처리 결과를 보증한다는 뜻은 아니다.
