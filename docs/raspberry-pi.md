# Raspberry Pi 연결과 측정

이 안내는 공개본을 Raspberry Pi 5 두 대에서 실행하기 위한 절차다. 이번
공개본의 실제 Pi 실행 결과는 아직 없으며 장치·음질·지연 수치는 **Unknown**이다.
공개본의 로컬 자동 검증은 [검증 문서](validation.md)에 별도로 기록한다.

## 준비와 설치

64비트 Raspberry Pi OS, 두 장치가 접근할 로컬 Wi-Fi 또는 핫스팟,
모노 16 kHz 입력·출력을 지원하는 오디오 장치와 헤드셋을 준비한다.
실제 사용한 OS 버전·오디오 장치·입력 이득·전원·냉각 조건은 측정 시 기록한다.
반향 제거를 구현하지 않았으므로 스피커의 음성이 마이크로 다시 들어가는
구성은 기본 시험 조건에 포함하지 않는다.

각 Pi에서 저장소를 내려받은 폴더를 열고 실행한다.

```sh
sudo apt update
sudo apt install python3-venv libportaudio2 libsndfile1
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[audio]'
tse model fetch
tse model verify
tse devices
```

프로젝트는 모노 16 kHz를 직접 지원하는 장치를 요구한다. 출력된 번호에서
입력·출력 장치를 각각 선택한다. PortAudio가 보고하는 장치 지연은 운영체제
추정값이며 실제 음향 루프백으로 측정한 종단 지연과 구분한다.

## 네트워크 연결

Raspberry Pi OS의 네트워크 설정은
[공식 설정 안내](https://www.raspberrypi.com/documentation/computers/configuration.html#networking)를
참조한다. 두 장치를 같은 공유기나 휴대전화 핫스팟에 연결하는 방법이 가장
짧은 절차다. 핫스팟의 클라이언트 간 통신 허용 여부도 확인한다.

Pi 자체를 핫스팟으로 사용할 때는 NetworkManager가 관리하는 Wi-Fi 장치를
확인하고 아래 명령을 **핫스팟을 만들 Pi**에서 실행한다. `wlan0`는 확인한
인터페이스 이름으로 바꾼다. 이 작업은 기존 Wi-Fi 연결을 바꿀 수 있으므로
로컬 터미널이나 별도의 관리 연결에서 수행한다.

```sh
nmcli device status
sudo nmcli device wifi hotspot ifname wlan0 con-name tse-link ssid tse-demo
nmcli device wifi show-password ifname wlan0
```

NetworkManager가 생성한 암호로 상대 Pi를 연결한다. 연결 암호는 명령 인수에
적는 대신 프롬프트에서 입력한다. 위 `show-password` 출력은 로컬 연결
설정에만 사용하고 공유할 실행 기록에는 포함하지 않는다.

```sh
nmcli --ask device wifi connect tse-demo
ip -4 address show
```

명령 구문과 핫스팟 중지 방법은
[NetworkManager 공식 nmcli 문서](https://networkmanager.dev/docs/api/latest/nmcli.html)에
따른다. 핫스팟을 마치면 생성한 연결을 내릴 수 있다.

```sh
sudo nmcli connection down tse-link
```

## 양방향 실행

각 장치에서 자신의 연결 주소와 상대 주소를 확인한다. 아래는 문서용 주소로
쓴 실행 예이며 주소·장치 번호를 실제 값으로 바꿔야 한다. 양쪽이 지정한
UDP 포트에 서로 접근할 수 있어야 한다.

```sh
# 장치 A
tse peer --bind 192.0.2.10:5000 --peer 192.0.2.20:5000 \
  --input-device 1 --output-device 2 --duration 60 --report peer-a.json

# 장치 B
tse peer --bind 192.0.2.20:5000 --peer 192.0.2.10:5000 \
  --input-device 1 --output-device 2 --duration 60 --report peer-b.json
```

시작 시 정상 확인과 이득 복귀, 최초 수신 지터 버퍼 시간이 필요하다.
먼저 정상 음성으로 양쪽 입력·출력을 확인하고, 장치 입력 이득을 기록한 뒤
시험을 진행한다. 프로그램은 지정한 상대 주소의 패킷을 받지만 이는
암호학적 인증이 아니다. 로컬 네트워크에서 사용한다.

| 증상 | 확인할 항목 |
|---|---|
| 장치를 열 수 없음 | `tse devices`, 선택한 방향·채널·16 kHz 지원, PortAudio 설치 |
| 원음만 들리거나 모델 누락 증가 | 모델 검증, 추론 기한 초과, CPU 부하·온도 |
| 반복적인 음소거 | 입력 최대값·포화·장치 overload, 실제 입력 이득, 보호 상태 |
| 상대 음성 없음 | 상대 주소·포트·방화벽·핫스팟 클라이언트 격리, 수신 패킷 수 |
| 끊김이나 지연 증가 | 지터 손실·폐기·재버퍼링, 장치 underrun·overrun, 드라이버 지연 |

## 후속 측정

먼저 같은 설정으로 `tse benchmark --frames 10000 --report benchmark.json`을
실행한다. 이 결과는 모델과 WOLA 처리 시간만 포함한다. 장치·네트워크·보호
제어를 포함한 종단 지연은 별도로 측정한다.

Pi 검증에서는 다음 자료를 동일 설정과 함께 남긴다.

- 두 장치의 OS·Python·모델 해시·오디오 장치·입력 이득·냉각 조건.
- 두 방향 정상 음성, 충격 후 복귀, 입력 overload, 모델 기한 초과,
  일시적인 네트워크 단절 후 복귀 결과.
- 동일한 측정 기준시계 또는 물리적 루프백을 사용한 종단 지연. 동기화되지
  않은 두 장치의 시각 차이를 지연으로 계산하지 않는다.
- 모델 처리 시간, 콜백 처리 시간, 패킷 손실·순서 변경, 지터 버퍼 최댓값,
  장치가 보고한 지연과 실제 루프백 지연.

임계값 조정과 모델 비교는 이 기초 측정을 확보한 다음 진행한다. 폭음 중
음성 보존과 실제 음압·청력보호 인증은 소프트웨어 합격 기준이 아니다.
