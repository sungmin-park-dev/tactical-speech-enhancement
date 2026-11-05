"""Build a fixed, synthetic stress test set from LibriSpeech test-clean (CC BY 4.0).

Every noise below is synthetic or derived from LibriSpeech. Nothing here is a
battlefield recording; results on it are not field performance.

Conditions (24 utterances each, 1 s noise-only lead-in):
  c00_clean         speech only, -26 dBFS active level
  c01_pink5         pink noise, 5 dB SNR
  c02_vehicle0      synthetic engine-harmonic + rumble noise, 0 dB SNR
  c03_babble5       6-talker LibriSpeech babble, 5 dB SNR
  c04_gunshot       3 synthetic gunshots (shock + muzzle blast + tail), mic peak 0.95, pink 20 dB
  c05_blast_sat     2 synthetic blasts, acoustic peak 4x full scale, hard clip at ADC (+ pink 20 dB)
  c06_blast_hdr     2 blasts with no clipping (speech -40 dBFS, blast peak 0.95), pink 20 dB
  c07_speech_loud   loud speech, peak 0.95, unclipped, pink 20 dB
  c08_speech_clip   shouted speech, acoustic peak 3x full scale, hard clip at ADC, pink 20 dB
  c09_reverb        speech at 0.5 m, RT60 ~0.6 s room, pink 20 dB; reference = direct path
  c10_echo_only     far-end echo via nonlinear path (SER 0 dB nominal), near-end silent, pink 30 dB
  c11_doubletalk    near-end speech + echo at SER 0 dB, pink 30 dB
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import scipy.signal as ss
import soundfile as sf

SP = Path(os.environ.get("TSE_BENCH_DIR", Path(__file__).resolve().parents[1] / "work"))
LS = SP / "data/ls/LibriSpeech/test-clean"
OUT = SP / "eval/testset"
FS = 16000
LEAD = FS  # 1 s lead-in
N_UTT = 24


def active_rms(x: np.ndarray) -> float:
    frames = x[: len(x) // 256 * 256].reshape(-1, 256)
    e = (frames ** 2).mean(1)
    thr = e.max() * 10 ** (-40 / 10)
    return float(np.sqrt(e[e > thr].mean()))


def set_level(x: np.ndarray, dbfs: float) -> np.ndarray:
    return x * (10 ** (dbfs / 20) / active_rms(x))


def pink(n: int, rng) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    y = np.fft.irfft(spec, n)
    return y / y.std()


def vehicle(n: int, rng) -> np.ndarray:
    t = np.arange(n) / FS
    f0 = 45 + 8 * np.sin(2 * np.pi * 0.07 * t + rng.uniform(0, 6)) + 2 * np.sin(2 * np.pi * 0.31 * t)
    phase = 2 * np.pi * np.cumsum(f0) / FS
    y = sum((1.0 / k ** 0.8) * np.sin(k * phase + rng.uniform(0, 6)) for k in range(1, 25))
    b, a = ss.butter(2, 900 / (FS / 2))
    rumble = ss.lfilter(b, a, rng.standard_normal(n))
    clank = np.zeros(n)
    period = int(FS / 9.0)
    for start in range(rng.integers(0, period), n, period):
        L = min(200, n - start)
        clank[start:start + L] += rng.standard_normal(L) * np.exp(-np.arange(L) / 40) * rng.uniform(0.5, 1.5)
    bh, ah = ss.butter(2, 1500 / (FS / 2), "high")
    clank = ss.lfilter(bh, ah, clank)
    y = y / y.std() + 0.8 * rumble / rumble.std() + 0.3 * clank / (clank.std() + 1e-9)
    return y / y.std()


def friedlander(td: float, b: float, length: float, rng) -> np.ndarray:
    t = np.arange(int(length * FS)) / FS
    p = (1 - t / td) * np.exp(-b * t / td)
    p[:2] = np.linspace(0, 1, 2)  # finite rise
    return p


def blast(rng) -> np.ndarray:
    td = rng.uniform(0.008, 0.03)
    main = friedlander(td, rng.uniform(1.0, 2.0), 1.5, rng)
    n = len(main)
    t = np.arange(n) / FS
    b, a = ss.butter(2, 250 / (FS / 2))
    rumble = ss.lfilter(b, a, rng.standard_normal(n)) * np.exp(-t / rng.uniform(0.3, 0.7))
    rumble *= 0.6 / (np.abs(rumble).max() + 1e-9)
    crackle = (rng.random(n) < 0.004) * rng.standard_normal(n) * np.exp(-t / 0.25)
    bh, ah = ss.butter(2, 2000 / (FS / 2), "high")
    crackle = ss.lfilter(bh, ah, crackle)
    crackle *= 0.25 / (np.abs(crackle).max() + 1e-9)
    y = main + rumble + crackle
    return y / np.abs(y).max()


def gunshot(rng) -> np.ndarray:
    n = int(0.6 * FS)
    t = np.arange(n) / FS
    y = np.zeros(n)
    # supersonic N-wave (~0.4 ms), then muzzle blast ~3 ms later
    nw = int(0.0004 * FS) + 2
    y[:nw] = np.linspace(1, -1, nw) * 0.8
    off = int(rng.uniform(0.002, 0.006) * FS)
    mb = friedlander(rng.uniform(0.0015, 0.004), 1.5, 0.05, rng)
    y[off:off + len(mb)] += mb
    tail = rng.standard_normal(n) * np.exp(-t / rng.uniform(0.08, 0.2)) * 0.15
    b, a = ss.butter(2, [300 / (FS / 2), 4000 / (FS / 2)], "band")
    y += ss.lfilter(b, a, tail)
    return y / np.abs(y).max()


def nonlinear_echo(far: np.ndarray, rng) -> np.ndarray:
    drive = 2.5
    x = np.tanh(drive * far / (np.abs(far).max() + 1e-9)) / np.tanh(drive)  # transducer/amp saturation
    L = int(0.012 * FS)
    ir = rng.standard_normal(L) * np.exp(-np.arange(L) / (0.002 * FS))
    b, a = ss.iirpeak(1800 / (FS / 2), 2.0)
    ir = ss.lfilter(b, a, ir)
    ir /= np.abs(ir).max()
    bulk = int(0.030 * FS)  # device playback->capture delay
    y = np.concatenate([np.zeros(bulk), ss.fftconvolve(x, ir)[: len(x) - bulk]])
    return y


def load_utts():
    speakers = sorted(int(p.name) for p in LS.iterdir())
    chosen = []
    for spk in speakers:
        cands = []
        for chap in sorted(int(p.name) for p in (LS / str(spk)).iterdir()):
            tr = LS / f"{spk}/{chap}/{spk}-{chap}.trans.txt"
            for line in tr.read_text().splitlines():
                uid, text = line.split(" ", 1)
                info = sf.info(LS / f"{spk}/{chap}/{uid}.flac")
                if 5.0 <= info.duration <= 10.0:
                    cands.append((uid, text, LS / f"{spk}/{chap}/{uid}.flac"))
        if cands:
            chosen.append((spk, cands[0]))
    return chosen


def main() -> None:
    rng = np.random.default_rng(20261103)
    utts = load_utts()
    targets = utts[:N_UTT]
    others = utts[N_UTT:]
    babble_src = [sf.read(u[1][2], dtype="float64")[0] for u in others[:6]]
    far_src = [(u[1][0], sf.read(u[1][2], dtype="float64")[0]) for u in others[6:]]
    room_rt60 = 0.6
    room_dim = [5.0, 4.0, 3.0]
    e_abs, max_order = pra.inverse_sabine(room_rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=FS, materials=pra.Material(e_abs), max_order=max_order)
    room.add_source([2.0, 2.0, 1.6])
    room.add_microphone([2.5, 2.0, 1.6])
    room.compute_rir()
    rir = np.asarray(room.rir[0][0], dtype=np.float64)
    direct = int(np.argmax(np.abs(rir)))
    measured_rt60 = float(pra.experimental.measure_rt60(rir, fs=FS))
    meta = {"description": __doc__, "license": "LibriSpeech CC BY 4.0; synthetic noise generated here",
            "seed": 20261103, "fs": FS, "lead_in_s": 1.0, "clips": []}
    OUT.mkdir(parents=True, exist_ok=True)
    for idx, (spk, (uid, text, path)) in enumerate(targets):
        speech, fs = sf.read(path, dtype="float64")
        assert fs == FS
        n = LEAD + len(speech) + FS // 2
        clean = np.zeros(n)
        clean[LEAD:LEAD + len(speech)] = speech
        clean26 = set_level(clean, -26)

        def mix(signal, noise, snr):
            noise = noise[: len(signal)]
            gain = active_rms(signal) / (10 ** (snr / 20) * np.sqrt(np.mean(noise[LEAD:LEAD + len(speech)] ** 2)))
            return noise * gain

        records = {}
        records["c00_clean"] = (clean26, clean26, {})
        records["c01_pink5"] = (clean26 + mix(clean26, pink(n, rng), 5), clean26, {})
        records["c02_vehicle0"] = (clean26 + mix(clean26, vehicle(n, rng), 0), clean26, {})
        bab = np.zeros(n)
        for b in babble_src:
            start = rng.integers(0, max(1, len(b) - 1))
            seg = np.tile(b, 3)[start:start + n]
            bab[: len(seg)] += seg / (seg.std() + 1e-9)
        records["c03_babble5"] = (clean26 + mix(clean26, bab, 5), clean26, {})

        def place_events(kind, count, peak, base, ref, acoustic_gain=1.0, clip=False, min_gap=1.2):
            x = base.copy()
            events = []
            usable = np.arange(LEAD + int(0.3 * FS), LEAD + len(speech) - int(0.4 * FS))
            starts = []
            for _ in range(200):
                if len(starts) == count:
                    break
                s = int(rng.choice(usable))
                if all(abs(s - o) > min_gap * FS for o in starts):
                    starts.append(s)
            for s in sorted(starts):
                ev = blast(rng) if kind == "blast" else gunshot(rng)
                ev = ev[: n - s] * peak
                x[s:s + len(ev)] += ev
                env = np.abs(ev) > 0.01 * peak
                end = s + (np.nonzero(env)[0].max() + 1 if env.any() else len(ev))
                events.append({"start": int(s), "end": int(end)})
            x = x * acoustic_gain
            if clip:
                x = np.clip(x, -1.0, 1.0)
            return x, events

        base = clean26 + mix(clean26, pink(n, rng), 20)
        x, ev = place_events("gunshot", 3, 0.95 - np.abs(base).max() * 0.0, base, clean26)
        records["c04_gunshot"] = (np.clip(x, -1, 1), clean26, {"events": ev})
        x, ev = place_events("blast", 2, 4.0, base, clean26, clip=True, min_gap=2.0)
        records["c05_blast_sat"] = (x, clean26, {"events": ev, "mic_clipped": True})
        clean40 = set_level(clean, -40)
        base40 = clean40 + mix(clean40, pink(n, rng), 20)
        x, ev = place_events("blast", 2, 0.95, base40, clean40, min_gap=2.0)
        x = np.clip(x, -1, 1)
        records["c06_blast_hdr"] = (x, clean40, {"events": ev, "mic_clipped": False})
        loud = clean * (0.95 / np.abs(clean).max())
        records["c07_speech_loud"] = (np.clip(loud + mix(loud, pink(n, rng), 20), -1, 1), loud, {})
        shout = clean * (3.0 / np.abs(clean).max())
        noisy = shout + mix(shout, pink(n, rng), 20)
        clipped = np.clip(noisy, -1, 1)
        records["c08_speech_clip"] = (clipped, shout / 3.0, {"clip_fraction": float(np.mean(np.abs(noisy) > 1))})
        rev = ss.fftconvolve(clean26, rir)[direct: direct + n]
        scale = active_rms(clean26) / active_rms(rev)
        rev = rev * scale
        dp = clean26 * rir[direct] * scale
        records["c09_reverb"] = (rev + mix(rev, pink(n, rng), 20), dp,
                                 {"rt60_target": room_rt60, "rt60_measured": measured_rt60})
        far_uid, far = far_src[idx % len(far_src)]
        farsig = np.zeros(n)
        L = min(len(far), n - LEAD)
        farsig[LEAD:LEAD + L] = far[:L]
        farsig = set_level(farsig, -20)          # what the device plays
        echo = nonlinear_echo(farsig, rng)
        echo = echo * (10 ** (-26 / 20) / active_rms(echo))   # SER 0 dB vs -26 dBFS near-end
        floor = mix(clean26, pink(n, rng), 30)
        records["c10_echo_only"] = (echo + floor, np.zeros(n), {"far_end": far_uid, "echo_rms": active_rms(echo)})
        records["c11_doubletalk"] = (clean26 + echo + floor, clean26, {"far_end": far_uid})
        for cond, (noisy_sig, ref, extra) in records.items():
            d = OUT / cond
            d.mkdir(parents=True, exist_ok=True)
            peak = np.abs(noisy_sig).max()
            assert peak <= 1.0 + 1e-9, (cond, peak)
            sf.write(d / f"{uid}.wav", noisy_sig.astype(np.float32), FS, subtype="FLOAT")
            sf.write(d / f"{uid}.ref.wav", ref.astype(np.float32), FS, subtype="FLOAT")
            if cond in ("c10_echo_only", "c11_doubletalk"):
                sf.write(d / f"{uid}.far.wav", farsig.astype(np.float32), FS, subtype="FLOAT")
                sf.write(d / f"{uid}.echo.wav", echo.astype(np.float32), FS, subtype="FLOAT")
            meta["clips"].append({"cond": cond, "uid": uid, "speaker": spk, "text": text,
                                  "speech_start": LEAD, "speech_end": LEAD + len(speech),
                                  "input_peak": float(peak), **extra})
        print(idx, uid, len(speech) / FS)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
