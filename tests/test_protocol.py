import math

import numpy as np
import pytest

from tactical_speech_enhancement.protocol import (
    FRAME_SAMPLES,
    HEADER,
    MAGIC,
    PACKET_BYTES,
    PCM16_LIMIT,
    VERSION,
    PacketError,
    decode_packet,
    encode_packet,
)


def test_round_trip_preserves_source_sequence_and_pcm_quantization():
    stream_id = bytes(range(16))
    source = np.linspace(-0.45, 0.45, FRAME_SAMPLES, dtype=np.float32)
    encoded = encode_packet(stream_id, 987654321, source)
    assert len(encoded) == PACKET_BYTES
    result = decode_packet(encoded)
    assert result.stream_id == stream_id
    assert result.sequence == 987654321
    np.testing.assert_allclose(result.audio, source, atol=0.5 / 32768)


@pytest.mark.parametrize("peak", [1, 1e30, -1, -1e30])
def test_wire_peak_never_exceeds_ceiling(peak):
    raw = encode_packet(b"a" * 16, 0, np.full(FRAME_SAMPLES, peak, dtype=np.float64))
    pcm = np.frombuffer(raw, dtype="<i2", offset=HEADER.size)
    assert np.max(np.abs(pcm)) == PCM16_LIMIT
    assert np.max(np.abs(decode_packet(raw).audio)) <= 10 ** (-6 / 20)


def test_custom_ceiling_can_lower_but_cannot_raise_wire_ceiling():
    frame = np.ones(FRAME_SAMPLES)
    quiet = decode_packet(encode_packet(b"a" * 16, 0, frame, ceiling_dbfs=-12))
    loud = decode_packet(encode_packet(b"a" * 16, 0, frame, ceiling_dbfs=-1))
    assert np.max(quiet.audio) == math.floor(32768 * 10 ** (-12 / 20)) / 32768
    assert np.max(loud.audio) == PCM16_LIMIT / 32768


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_one_nonfinite_value_silences_entire_frame(bad):
    values = np.full(FRAME_SAMPLES, 0.2)
    values[73] = bad
    assert not decode_packet(encode_packet(b"a" * 16, 0, values)).audio.any()


@pytest.mark.parametrize("size", [0, 1, 255, 257])
def test_encoder_rejects_wrong_frame_size(size):
    with pytest.raises(ValueError):
        encode_packet(b"a" * 16, 0, np.zeros(size))


@pytest.mark.parametrize("sequence", [-1, 2**64, 1.5, True])
def test_encoder_rejects_invalid_source_sequence(sequence):
    with pytest.raises(ValueError):
        encode_packet(b"a" * 16, sequence, np.zeros(FRAME_SAMPLES))


@pytest.mark.parametrize("mutation", ["short", "long", "magic", "version", "reserved"])
def test_decoder_rejects_malformed_packet(mutation):
    raw = bytearray(encode_packet(b"a" * 16, 0, np.zeros(FRAME_SAMPLES)))
    if mutation == "short":
        raw.pop()
    elif mutation == "long":
        raw.append(0)
    elif mutation == "magic":
        raw[0] ^= 1
    elif mutation == "version":
        raw[4] += 1
    else:
        raw[5] = 1
    with pytest.raises(PacketError):
        decode_packet(bytes(raw))


def test_external_sender_cannot_bypass_receive_pcm_limit():
    pcm = np.array([32767, -32768] * 128, dtype="<i2")
    raw = HEADER.pack(MAGIC, VERSION, b"b" * 16, 0) + pcm.tobytes()
    assert np.max(np.abs(decode_packet(raw).audio)) == PCM16_LIMIT / 32768
