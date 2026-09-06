# Tactical Speech Enhancement

경량 엣지 장치에서 음성 잡음을 줄이고, Raspberry Pi 5 두 대 사이에
Wi-Fi 또는 핫스팟으로 음성을 주고받는 Python 프로젝트다. 국방 분야의
음성 통신을 배경으로 하며, 공개본은 단일 마이크·헤드셋을 사용하는
양방향 통신과 그 소프트웨어 검증에 집중한다.

GTCRN streaming ONNX로 음성을 처리한다. 큰 피크나 포화가 감지되면
해당 구간을 음소거하고, 정상 입력을 확인한 뒤 음량을 점진적으로
복구한다. 폭음 중 음성 보존은 목표에 포함하지 않는다.

## 구현 구조

```mermaid
flowchart LR
    A[마이크 입력] --> B[독립 입력 검사]
    B --> C[GTCRN 추론]
    B --> D[원음 정렬]
    C --> E[결과 선택과 보호 이득]
    D --> E
    E --> F[최종 출력 제한]
    F --> G[UDP 송신]
    G --> H[상대 수신 지터 버퍼]
    H --> I[독립 출력 제한과 재생]
```

- **16 kHz·256샘플:** 프레임당 16ms, 상대 한 명과 PCM16 음성을 송수신한다.
- **추론과 독립적인 보호:** 입력을 먼저 검사한다. 느리거나 멈춘 모델이
  보호 판단과 송신 선택을 막지 않으며, 늦은 결과는 보호를 해제하지 못한다.
- **조건부 원음 전환:** 정상 입력에서 모델만 실패하면 지연을 맞춘 원음을
  사용한다. 이상 구간과 보호 유지 구간에는 이 우회를 허용하지 않는다.
- **제한된 대기열:** 원음 정렬은 고정 2프레임이다. 목표 4·최대 8프레임은
  수신 지터 버퍼에만 적용하며 재생 대기열을 별도로 추가하지 않는다.
- **출력 제한:** 송신과 상대 재생에서 각각 최종 디지털 피크를 −6dBFS로
  제한한다. 이 값은 실제 음압이나 청력보호 성능을 나타내지 않는다.

모델, 보호·복귀 제어, 통신, 오디오 장치 처리를 나누어 기기 측정과
후속 모델 비교 시 각 원인을 구분할 수 있도록 구성했다.

## 설치와 파일 실행

Python 3.11 이상을 사용한다. 저장소를 내려받은 폴더에서 실행한다.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[audio,dev]'
tse model fetch
tse model verify
tse process tests/fixtures/speech/speaker-0061.wav output.wav
tse benchmark --frames 10000
```

모델 다운로드 후 SHA-256과 입출력 계약을 검사한다. 파일 입력은 모노
16 kHz를 사용한다. 자동 샘플레이트 변환은 제공하지 않는다.
오디오 장치 실행에는 운영체제의 PortAudio 지원이 필요하다.

초기 설정 전체는 [software-defaults.toml](examples/software-defaults.toml)에
있다. 설정 파일을 명시해서 같은 조건으로 실행할 수 있다.

```sh
tse --config examples/software-defaults.toml benchmark --frames 10000
```

## 두 장치 실행

두 장치를 같은 Wi-Fi 또는 핫스팟에 연결한 뒤 `tse devices`로 각 장치의
입력·출력 번호를 확인한다. 아래 주소와 장치 번호는 예제이며 실제 연결에
맞게 바꾼다. 16 kHz 모노 입력·출력을 지원하는 헤드셋을 사용한다.

```sh
# 장치 A
tse peer --bind 192.0.2.10:5000 --peer 192.0.2.20:5000 \
  --input-device 1 --output-device 2 --duration 60 --report peer-a.json

# 장치 B
tse peer --bind 192.0.2.20:5000 --peer 192.0.2.10:5000 \
  --input-device 1 --output-device 2 --duration 60 --report peer-b.json
```

`192.0.2.0/24`는 문서용 주소다. 설정과 진단 절차는
[Raspberry Pi 실행 안내](docs/raspberry-pi.md)에 설명되어 있다.
통신은 동일 로컬 네트워크의 명시한 상대를 전제로 하며, 응용 계층 암호화나
상대 인증을 제공하지 않는다.

## 결과를 해석하는 범위

| 구분 | 내용 |
|---|---|
| 이전 장치 경험 — 소유자 보고 | 이전 구현에서 Raspberry Pi 5 두 대의 모델 실행과 Wi-Fi·핫스팟 음성 통신 목표를 달성했다고 보고했다. |
| 이 공개본 | 처리·통신 구조를 재구성하고, 고정된 시험 입력과 자동 테스트로 검증한다. 실행 환경과 실제 결과는 [검증 문서](docs/validation.md)에 기록한다. |
| 공개본의 실제 Pi 측정 | 아직 실시하지 않았다. 음질, 종단 지연, 장치 설정별 오차단율은 **Unknown**이다. |

이전 장치 경험은 현재 공개 코드의 하드웨어 검증을 대신하지 않는다.
기본 임계값과 복귀 시간은 **소프트웨어 검증용 초기값**이다. 다음 단계는
실제 Pi 측정에 따른 입력 이득·시간 설정 조정과 동일 조건의 모델 비교다.

## 검증과 문서

```sh
python -m pytest
ruff check .
```

정상 음성 시험은 공개 LibriSpeech에서 고정한 10개 음성의 세 음량 조건을
사용한다. 충격·포화·잘못된 입력, 느린 추론, 패킷 오류와 시계 차이를
별도로 시험한다. 모델 통합 시험에는 먼저 `tse model fetch`를 실행한다.

- [처리 순서·버퍼·초기 설정](docs/architecture.md)
- [시험 입력·합격 기준·검증 결과](docs/validation.md)
- [Raspberry Pi 연결·실행·측정](docs/raspberry-pi.md)
- [개발 판단과 에이전트 협업 기록](docs/development.md)

프로젝트 코드는 [MIT](LICENSE), GTCRN은 원 저작자의 MIT 라이선스다.
공개 음성 시험 자료에는 CC BY 4.0을 적용한다. 출처·원문 라이선스·변경
내역은 [제삼자 고지](THIRD_PARTY_NOTICES.md)에 모았다.
