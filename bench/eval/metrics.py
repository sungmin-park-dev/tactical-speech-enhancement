"""Objective metrics. Intrusive metrics use the clean (or direct-path) reference."""

from __future__ import annotations

import re

import numpy as np

FS = 16000
FR = 256


def si_sdr(est, ref, eps=1e-10):
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    return 10 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps))


def pesq_wb(est, ref):
    from pesq import pesq
    try:
        return float(pesq(FS, ref.astype(np.float64), est.astype(np.float64), "wb"))
    except Exception:
        return float("nan")


def estoi(est, ref):
    from pystoi import stoi
    return float(stoi(ref, est, FS, extended=True))


def speech_frames(ref, floor_db=-40):
    n = len(ref) // FR
    e = (ref[: n * FR].reshape(n, FR) ** 2).mean(1)
    return e > e.max() * 10 ** (floor_db / 10), e


def speech_loss_ratio(est, ref, drop_db=-20):
    """Fraction of speech-active frames whose output energy fell >20 dB below the reference."""
    act, e_ref = speech_frames(ref)
    n = len(e_ref)
    scale = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    scale = scale if scale > 1e-3 else 1.0
    e_out = ((est[: n * FR] / scale).reshape(n, FR) ** 2).mean(1)
    lost = e_out < e_ref * 10 ** (drop_db / 10)
    return float(lost[act].mean()) if act.any() else float("nan")


def window_metrics(est, ref, events, before_ms=100, after_ms=500):
    """Around each impulse: residual distortion before onset and speech energy ratio after it."""
    if not events:
        return {}
    scale = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    scale = scale if abs(scale) > 1e-3 else 1.0
    pre, post_ratio, post_sisdr = [], [], []
    act, e_ref = speech_frames(ref)
    for ev in events:
        s, e = ev["start"], ev["end"]
        a = max(0, s - int(before_ms * FS / 1000))
        if s - a > FR:
            r = ref[a:s]
            o = est[a:s] / scale
            pre.append(10 * np.log10((np.sum((o - r) ** 2) + 1e-12) / (np.sum(r ** 2) + 1e-12)))
        b = min(len(ref), e + int(after_ms * FS / 1000))
        if b - e > 4 * FR:
            r = ref[e:b]
            o = est[e:b] / scale
            fa = np.arange(e // FR, b // FR)
            fa = fa[fa < len(act)]
            sel = act[fa]
            if sel.any():
                rr = np.concatenate([ref[f * FR:(f + 1) * FR] for f in fa[sel]])
                oo = np.concatenate([est[f * FR:(f + 1) * FR] / scale for f in fa[sel]])
                post_ratio.append(10 * np.log10((np.sum(oo ** 2) + 1e-12) / (np.sum(rr ** 2) + 1e-12)))
            if np.sum(r ** 2) > 0:
                post_sisdr.append(si_sdr(o, r))
    return {
        "pre_event_distortion_db": float(np.mean(pre)) if pre else float("nan"),
        "post_event_speech_energy_db": float(np.mean(post_ratio)) if post_ratio else float("nan"),
        "post_event_si_sdr": float(np.mean(post_sisdr)) if post_sisdr else float("nan"),
    }


def erle_db(mic_echo, out, start, end):
    """Echo-only: 10log10(E[mic]/E[out]) over the far-end active region."""
    m = mic_echo[start:end]
    o = out[start:end]
    return float(10 * np.log10((np.sum(m ** 2) + 1e-12) / (np.sum(o ** 2) + 1e-12)))


_norm_re = re.compile(r"[^a-z' ]+")


def normalise_text(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = _norm_re.sub(" ", text)
    return " ".join(text.split())
