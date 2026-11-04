import numpy as np
import pytest

from tactical_speech_enhancement.jitter import MAX_RETIRED_STREAMS, JitterBuffer
from tactical_speech_enhancement.protocol import FRAME_SAMPLES, FRAME_SECONDS, Packet


def packet(sequence, stream=1):
    return Packet(
        stream.to_bytes(16, "big"), sequence, np.full(FRAME_SAMPLES, 0.1, dtype=np.float32)
    )


def test_first_playout_is_first_arrival_plus_four_frames():
    jitter = JitterBuffer()
    jitter.push(packet(10), 10.0)
    assert not jitter.playout(10.063999).any()
    assert jitter.playout(10.064).any()
    assert jitter.summary()["next_sequence"] == 11


def test_reordering_before_start_does_not_move_first_deadline():
    jitter = JitterBuffer()
    jitter.push(packet(11), 0)
    jitter.push(packet(10), 0.010)
    assert jitter.summary()["next_deadline"] == 0.064
    assert jitter.playout(0.064).any()
    assert jitter.summary()["next_sequence"] == 11
    assert jitter.playout(0.080).any()


def test_duplicate_loss_late_and_ordering():
    jitter = JitterBuffer()
    jitter.push(packet(0), 0)
    jitter.push(packet(2), 0.02)
    assert not jitter.push(packet(2), 0.021)
    assert jitter.playout(0.064).any()
    assert not jitter.playout(0.080).any()
    assert not jitter.push(packet(1), 0.081)
    assert jitter.playout(0.096).any()
    assert jitter.summary()["duplicate"] == 1
    assert jitter.summary()["late"] == 1
    assert jitter.summary()["lost"] == 1


def test_capacity_includes_prestart_reordering():
    jitter = JitterBuffer()
    for sequence in range(100):
        jitter.push(packet(sequence), 0)
        assert jitter.size <= 8
    for sequence in reversed(range(100)):
        jitter.push(packet(sequence), 0.001)
        assert jitter.size <= 8
    assert jitter.summary()["high_watermark"] <= 8
    assert jitter.summary()["sequence_skipped"] > 0


def test_stream_restart_and_delayed_retired_stream():
    jitter = JitterBuffer()
    jitter.push(packet(20), 0)
    assert jitter.playout(0.064).any()
    jitter.push(packet(0, stream=2), 0.080)
    assert not jitter.push(packet(21, stream=1), 0.081)
    assert not jitter.playout(0.1).any()
    assert jitter.playout(0.144).any()
    assert jitter.summary()["restarts"] == 1
    assert jitter.summary()["retired"] == 1


def test_retirement_memory_is_bounded_and_evicted_sessions_cannot_interrupt_healthy_stream():
    jitter = JitterBuffer()
    for stream in range(1, MAX_RETIRED_STREAMS + 3):
        jitter.push(packet(0, stream), float(stream * 2))
    assert jitter.summary()["retired_streams"] == MAX_RETIRED_STREAMS
    now = float((MAX_RETIRED_STREAMS + 2) * 2)
    assert not jitter.push(packet(1, stream=1), now + 0.1)
    assert jitter.summary()["session_wait"] == 1
    assert jitter.push(packet(99, stream=999), now + 1.1)


def test_callback_pause_drops_stale_audio():
    jitter = JitterBuffer()
    for sequence in range(8):
        jitter.push(packet(sequence), sequence * 0.016)
    assert not jitter.playout(5).any()
    assert jitter.size == 0
    assert jitter.summary()["sequence_skipped"] > 200


def test_continuous_reception_during_callback_pause_does_not_skip_elapsed_time_twice():
    jitter = JitterBuffer()
    for sequence in range(645):
        now = sequence * FRAME_SECONDS
        jitter.push(packet(sequence), now)
        if sequence < 20:
            jitter.playout(now)
        assert jitter.size <= 8
    # Ten seconds without a playback callback must not add another ten seconds
    # of silence. The receive side already dropped old source frames meanwhile.
    resumed = jitter.playout(now)
    assert resumed.any()
    assert jitter.summary()["next_sequence"] <= 645
    assert jitter.summary()["next_deadline"] <= now + FRAME_SECONDS + 1e-9
    for sequence in range(645, 661):
        now = sequence * FRAME_SECONDS
        jitter.push(packet(sequence), now)
        assert jitter.playout(now).any()
        assert jitter.size <= 8


def test_continuous_overflow_does_not_push_playout_deadline_into_future():
    jitter = JitterBuffer()
    played = 0
    for tick in range(200):
        now = tick * FRAME_SECONDS
        jitter.push(packet(tick * 2), now)
        jitter.push(packet(tick * 2 + 1), now)
        played += bool(jitter.playout(now).any())
        assert jitter.size <= 8
    assert played >= 190


@pytest.mark.parametrize("ppm", [-300, 300])
def test_one_hour_virtual_clock_drift_is_bounded_and_recovers(ppm):
    jitter = JitterBuffer()
    sender_step = FRAME_SECONDS / (1 + ppm / 1_000_000)
    next_arrival = 0.0
    next_sequence = 0
    pause_begin, pause_end = 1800.0, 1800.4
    recovered_after_pause = False
    for tick in range(int(3600 / FRAME_SECONDS)):
        now = tick * FRAME_SECONDS
        while next_arrival <= now + 1e-9:
            if not pause_begin <= next_arrival < pause_end:
                jitter.push(packet(next_sequence), next_arrival)
            next_arrival += sender_step
            next_sequence += 1
        output = jitter.playout(now)
        if pause_end + 0.5 < now < pause_end + 1 and output.any():
            recovered_after_pause = True
        assert jitter.size <= 8
    stats = jitter.summary()
    assert recovered_after_pause
    assert stats["played"] > 220_000
    assert stats["high_watermark"] <= 8
    assert stats["rebuffered"] < 100
    assert stats["retired_streams"] == 0


def test_short_random_jitter_and_losses_keep_order_without_replay():
    jitter = JitterBuffer()
    rng = np.random.default_rng(40)
    events = []
    for sequence in range(1000):
        if sequence % 17 == 8:
            continue
        arrival = sequence * FRAME_SECONDS + rng.uniform(0, 0.025)
        events.append((arrival, sequence))
        if sequence % 13 == 0:
            events.append((arrival + 0.002, sequence))
    events.sort()
    cursor = 0
    consumed = []
    for tick in range(1010):
        now = tick * FRAME_SECONDS
        while cursor < len(events) and events[cursor][0] <= now:
            arrival, sequence = events[cursor]
            jitter.push(packet(sequence), arrival)
            cursor += 1
        before = jitter.summary()["next_sequence"]
        if jitter.playout(now).any():
            consumed.append(before)
        assert jitter.size <= 8
    assert len(consumed) > 900
    assert all(a < b for a, b in zip(consumed, consumed[1:]))
    assert jitter.summary()["duplicate"] > 0
    assert jitter.summary()["lost"] > 0


def test_level_above_target_converges_back_after_a_burst():
    jitter = JitterBuffer(4, 8)
    now = 0.0
    # A burst delivers 8 frames at once (e.g. after a Wi-Fi stall); afterwards
    # one frame arrives per playout tick, so without convergence the level stays at 8.
    for s in range(8):
        jitter.push(packet(s), now)
    sequence = 8
    now = jitter.summary()["next_deadline"]
    played = []
    for _ in range(62 * 6):
        played.append(jitter.playout(now).copy())
        jitter.push(packet(sequence), now)
        sequence += 1
        now += FRAME_SECONDS
    stats = jitter.summary()
    assert stats["converge_dropped"] >= 2   # 8 -> target + 1 (hysteresis)
    assert stats["frames"] <= 6
    assert stats["lost"] == 0
    assert all(frame.any() for frame in played)


def test_convergence_can_be_disabled_and_keeps_level_at_burst_size():
    jitter = JitterBuffer(4, 8, converge_ticks=0)
    for s in range(8):
        jitter.push(packet(s), 0.0)
    sequence, now = 8, jitter.summary()["next_deadline"]
    for _ in range(62 * 6):
        jitter.playout(now)
        jitter.push(packet(sequence), now)
        sequence += 1
        now += FRAME_SECONDS
    assert jitter.summary()["converge_dropped"] == 0
    assert jitter.summary()["frames"] >= 7


def test_converge_fade_is_continuous_at_the_splice():
    jitter = JitterBuffer(1, 8, converge_ticks=1)
    ramp = [np.full(FRAME_SAMPLES, 0.1 * (k + 1), dtype=np.float32) for k in range(4)]
    for s, audio in enumerate(ramp):
        jitter.push(Packet((1).to_bytes(16, "big"), s, audio), 0.0)
    out = jitter.playout(jitter.summary()["next_deadline"])
    assert out[0] == pytest.approx(0.1)          # starts as the frame that was due
    assert out[-1] == pytest.approx(0.2)          # ends in the following frame
    assert np.all(np.diff(out) >= -1e-7)          # monotonic blend, no jump


def test_concealment_repeats_last_frame_with_decay_then_silence():
    jitter = JitterBuffer(1, 8, converge_ticks=0, conceal_frames=3)
    jitter.push(packet(0), 0.0)
    now = jitter.summary()["next_deadline"]
    first = jitter.playout(now)
    assert first.any()
    levels = [float(np.abs(jitter.playout(now + k * FRAME_SECONDS)).max()) for k in range(1, 6)]
    assert levels[0] == pytest.approx(0.1, rel=1e-6)          # starts at the last frame's level
    assert levels[0] > levels[1] > levels[2] > 0              # decays over 3 frames
    assert levels[3] == 0 and levels[4] == 0                  # then silence
    assert jitter.summary()["concealed"] == 3


def test_concealment_off_by_default():
    jitter = JitterBuffer()
    assert jitter.conceal_frames == 0
