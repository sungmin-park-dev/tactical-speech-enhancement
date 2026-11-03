"""Systems under test. Each returns output aligned sample-by-sample with its input.

Repository code is imported read-only from the installed package; nothing in
the repository is modified. Official denoiser code is imported from its own
directory. All are run offline (file-by-file) with streaming semantics where
the deployed system streams.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import scipy.special as sps

SP = Path(os.environ.get("TSE_BENCH_DIR", Path(__file__).resolve().parents[1] / "work"))
HOP = 256
sys.path.insert(0, str(SP / "ext/fb_denoiser"))

# ---------------------------------------------------------------- repository
from tactical_speech_enhancement.artifacts import default_model_path  # noqa: E402
from tactical_speech_enhancement.config import Settings  # noqa: E402
from tactical_speech_enhancement.engine import TxController  # noqa: E402
from tactical_speech_enhancement.gtcrn import (  # noqa: E402
    OFFICIAL_MODEL_SHA256,
    GtcrnStream,
    GtcrnStreamingEnhancer,
    PassthroughSpectralEngine,
)
from tactical_speech_enhancement.guard import PeakLimiter  # noqa: E402
from tactical_speech_enhancement.worker import SynchronousWorker  # noqa: E402

MODEL = default_model_path()  # honours TSE_MODEL_PATH / XDG_CACHE_HOME


def _pad(x):
    n = math.ceil(len(x) / HOP) * HOP
    out = np.zeros(n + 2 * HOP, dtype=np.float32)
    out[: len(x)] = x
    return out


def gtcrn_only(x, **_):
    enh = GtcrnStreamingEnhancer(MODEL, expected_sha256=OFFICIAL_MODEL_SHA256)
    xp = _pad(x)
    out = np.concatenate([enh.process_hop(xp[i:i + HOP]) for i in range(0, len(xp), HOP)])
    return out[HOP:HOP + len(x)], {}


def _controller_run(x, enhancer, settings=None):
    settings = settings or Settings()
    worker = SynchronousWorker(enhancer, settings=settings)
    ctrl = TxController(enhancer, settings=settings, worker=worker)
    count = math.ceil(len(x) / HOP)
    out = np.zeros(count * HOP, dtype=np.float32)
    routes = np.empty(count, dtype=object)
    for seq in range(count + 2):
        block = np.zeros(HOP, dtype=np.float32)
        piece = x[seq * HOP:(seq + 1) * HOP]
        block[: len(piece)] = piece
        r = ctrl.process(block, now=seq * settings.frame_seconds)
        if 0 <= r.sequence < count:
            out[r.sequence * HOP:(r.sequence + 1) * HOP] = r.audio
            routes[r.sequence] = r.diagnostics["route"]
    ctrl.close()
    s = ctrl.summary()
    diag = {"muted_frames": int(sum(1 for v in routes if v in ("muted", "startup"))),
            "dry_frames": int(sum(1 for v in routes if v == "dry")),
            "anomalies": s["guard"]["anomalies"]}
    return out[: len(x)], diag


def current_pipeline(x, **_):
    enh = GtcrnStreamingEnhancer(MODEL, expected_sha256=OFFICIAL_MODEL_SHA256)
    return _controller_run(x, enh)


def guard_only(x, **_):
    return _controller_run(x, GtcrnStream(PassthroughSpectralEngine()))


# ------------------------------------------------------------------ classic
def _sqrt_hann(n=512):
    i = np.arange(n)
    return np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * i / n))


def mmse_lsa(x, floor_db=-15.0, **_):
    """Causal MMSE log-spectral amplitude estimator with minimum-statistics noise PSD.

    Same 512/256 sqrt-Hann WOLA as GTCRN (one-hop delay, compensated).
    """
    win = _sqrt_hann()
    xp = _pad(x)
    n_frames = len(xp) // HOP
    buf = np.zeros(512)
    ola = np.zeros(512)
    out = np.zeros(len(xp))
    S = None
    lam = None
    sub_min = None
    sub_hist = []
    gamma_prev = None
    g_prev = None
    gmin = 10 ** (floor_db / 20)
    U, V = 8, 12  # ~1.5 s minimum window
    count = 0
    for f in range(n_frames):
        buf[:-HOP] = buf[HOP:]
        buf[-HOP:] = xp[f * HOP:(f + 1) * HOP]
        Y = np.fft.rfft(buf * win)
        P = np.abs(Y) ** 2 + 1e-12
        if S is None:
            S = P.copy()
            lam = P.copy()
            sub_min = P.copy()
            gamma_prev = np.ones_like(P)
            g_prev = np.ones_like(P)
        S = 0.85 * S + 0.15 * P
        sub_min = np.minimum(sub_min, S)
        count += 1
        if count == V:
            sub_hist.append(sub_min.copy())
            sub_hist = sub_hist[-U:]
            sub_min = S.copy()
            count = 0
        smin = np.minimum.reduce(sub_hist + [sub_min]) if sub_hist else sub_min
        if f < 30:  # lead-in initialisation (noise only)
            lam = (lam * f + P) / (f + 1)
        else:
            lam = np.maximum(1.5 * smin, 1e-12)
        gamma = np.minimum(P / lam, 1e4)
        xi = 0.98 * (g_prev ** 2) * gamma_prev + 0.02 * np.maximum(gamma - 1, 0)
        xi = np.maximum(xi, 10 ** (-25 / 10))
        v = xi * gamma / (1 + xi)
        G = xi / (1 + xi) * np.exp(0.5 * sps.exp1(np.maximum(v, 1e-8)))
        G = np.clip(G, gmin, 1.0)
        g_prev, gamma_prev = G, gamma
        frame = np.fft.irfft(Y * G, 512) * win
        ola[:-HOP] = ola[HOP:]
        ola[-HOP:] = 0
        ola += frame
        out[f * HOP:(f + 1) * HOP] = ola[:HOP]
    return out[HOP:HOP + len(x)].astype(np.float32), {}


def input_soft_limiter(x, ceiling=0.25, lookahead=32, release_ms=30.0):
    """Feed-forward peak limiter with lookahead (attenuates instead of muting)."""
    rel = math.exp(-1 / (16000 * release_ms / 1000))
    a = np.abs(np.concatenate([x, np.zeros(lookahead)]))
    # max over the lookahead window
    from scipy.ndimage import maximum_filter1d
    env = maximum_filter1d(a, size=lookahead + 1, origin=-(lookahead // 2))[: len(x)]
    target = np.minimum(1.0, ceiling / np.maximum(env, 1e-9))
    g = np.empty(len(x))
    cur = 1.0
    for i, t in enumerate(target):
        cur = t if t < cur else rel * cur + (1 - rel) * t
        g[i] = cur
    return (x * g).astype(np.float32), g


def hybrid_soft(x, **_):
    """Input soft limiter -> GTCRN -> output limiter (no muting)."""
    y, g = input_soft_limiter(x)
    out, _ = gtcrn_only(y)
    lim = PeakLimiter(-6.0, 16000, 100.0)
    res = np.concatenate([lim.process(out[i:i + HOP]) for i in range(0, len(out), HOP)])[: len(x)]
    return res, {"attenuated_samples": int((g < 0.5).sum())}


# -------------------------------------------------------------- AEC hybrid
def pbfdaf_aec(mic, far, partitions=6, mu=0.4, **_):
    """Linear partitioned-block frequency-domain NLMS echo canceller (block 256)."""
    B = HOP
    N = 2 * B
    P = partitions
    n_blocks = len(mic) // B
    W = np.zeros((P, N // 2 + 1), dtype=complex)
    Xh = np.zeros((P, N // 2 + 1), dtype=complex)
    xbuf = np.zeros(N)
    e_out = np.zeros(len(mic))
    pw = np.full(N // 2 + 1, 1e-6)
    for k in range(n_blocks):
        xbuf[:B] = xbuf[B:]
        xbuf[B:] = far[k * B:(k + 1) * B]
        X = np.fft.rfft(xbuf)
        Xh = np.roll(Xh, 1, axis=0)
        Xh[0] = X
        Yhat = (W * Xh).sum(0)
        yhat = np.fft.irfft(Yhat)[B:]
        d = mic[k * B:(k + 1) * B]
        e = d - yhat
        e_out[k * B:(k + 1) * B] = e
        E = np.fft.rfft(np.concatenate([np.zeros(B), e]))
        pw = 0.9 * pw + 0.1 * (np.abs(X) ** 2)
        grad = np.conj(Xh) * E[None, :] / (pw[None, :] * P + 1e-6)
        g = np.fft.irfft(grad, axis=1)
        g[:, B:] = 0
        W += mu * np.fft.rfft(g, axis=1)
    return e_out.astype(np.float32)


def aec_gtcrn(x, far=None, **_):
    if far is None:
        return gtcrn_only(x)
    e = pbfdaf_aec(x, far)
    out, _ = gtcrn_only(e)
    return out, {}


def aec_only(x, far=None, **_):
    if far is None:
        return x.copy(), {}
    return pbfdaf_aec(x, far), {}


# ------------------------------------------------------------------- DTLN
_DTLN = {}


def dtln(x, **_):
    import onnxruntime as ort
    d = SP / "ext/dtln"
    if not _DTLN:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        _DTLN["m1"] = ort.InferenceSession(str(d / "model_1.onnx"), so, providers=["CPUExecutionProvider"])
        _DTLN["m2"] = ort.InferenceSession(str(d / "model_2.onnx"), so, providers=["CPUExecutionProvider"])
    m1, m2 = _DTLN["m1"], _DTLN["m2"]
    n1 = [i.name for i in m1.get_inputs()]
    n2 = [i.name for i in m2.get_inputs()]
    in1 = {i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], np.float32) for i in m1.get_inputs()}
    in2 = {i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], np.float32) for i in m2.get_inputs()}
    bl, bs = 512, 128
    xp = np.concatenate([x, np.zeros(bl)]).astype(np.float32)
    nb = (len(xp) - (bl - bs)) // bs
    inb = np.zeros(bl, np.float32)
    outb = np.zeros(bl, np.float32)
    out = np.zeros(len(xp), np.float32)
    for idx in range(nb):
        inb[:-bs] = inb[bs:]
        inb[-bs:] = xp[idx * bs:(idx + 1) * bs]
        F = np.fft.rfft(inb)
        mag = np.abs(F).reshape(1, 1, -1).astype(np.float32)
        in1[n1[0]] = mag
        o1 = m1.run(None, in1)
        in1[n1[1]] = o1[1]
        est = np.fft.irfft(mag * o1[0] * np.exp(1j * np.angle(F))).reshape(1, 1, -1).astype(np.float32)
        in2[n2[0]] = est
        o2 = m2.run(None, in2)
        in2[n2[1]] = o2[1]
        outb[:-bs] = outb[bs:]
        outb[-bs:] = 0
        outb += np.squeeze(o2[0])
        out[idx * bs:(idx + 1) * bs] = outb[:bs]
    delay = bl - bs
    return out[delay:delay + len(x)], {}


# ------------------------------------------------------------ FB denoiser
_DEMUCS = {}


def _demucs_model(name):
    if name not in _DEMUCS:
        import torch
        sys.path.insert(0, str(SP / "ext/fb_denoiser"))
        from denoiser.demucs import Demucs
        hidden = 48 if name == "dns48" else 64
        files = {"dns48": "dns48-11decc9d8e3f0998.th", "dns64": "dns64-a7761ff99a7d5bb6.th",
                 "master64": "master64-8a5dfb4bb92753dd.th"}
        model = Demucs(hidden=hidden, sample_rate=16_000)
        state = torch.load(SP / "weights" / files[name], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _DEMUCS[name] = model
    return _DEMUCS[name]


def demucs_stream(x, name="dns48", clamp=True, chunk=HOP, **_):
    """Official DemucsStreamer, fed like denoiser/live.py (default: hard clamp to [-1, 1])."""
    import torch
    from denoiser.demucs import DemucsStreamer
    model = _demucs_model(name)
    streamer = DemucsStreamer(model, dry=0, num_frames=1)
    t = torch.from_numpy(x.astype(np.float32))[None]
    outs = []
    over = 0
    with torch.no_grad():
        for i in range(0, t.shape[1], chunk):
            o = streamer.feed(t[:, i:i + chunk])
            if o.numel():
                outs.append(o)
        outs.append(streamer.flush())
    y = torch.cat(outs, 1)[0].numpy()
    over = int((np.abs(y) > 1).sum())
    if clamp:
        y = np.clip(y, -1, 1)
    y = y[: len(x)]
    if len(y) < len(x):
        y = np.concatenate([y, np.zeros(len(x) - len(y), np.float32)])
    return y.astype(np.float32), {"samples_over_full_scale": over, "variance_final": float(streamer.variance)}


def demucs_offline(x, name="dns48", **_):
    import torch
    model = _demucs_model(name)
    with torch.no_grad():
        y = model(torch.from_numpy(x.astype(np.float32))[None, None])[0, 0].numpy()
    return y[: len(x)], {}


SYSTEMS = {
    "S0_noisy": lambda x, **k: (x.copy(), {}),
    "S1_gtcrn_only": gtcrn_only,
    "S2_current_pipeline": current_pipeline,
    "S3_guard_only": guard_only,
    "S4_mmse_lsa": mmse_lsa,
    "S5_softlimit_gtcrn": hybrid_soft,
    "S6_demucs_dns48": lambda x, **k: demucs_stream(x, "dns48"),
    "S6b_demucs_dns48_offline": lambda x, **k: demucs_offline(x, "dns48"),
    "S7_demucs_dns64_offline": lambda x, **k: demucs_offline(x, "dns64"),
    "S8_demucs_master64_offline": lambda x, **k: demucs_offline(x, "master64"),
    "S9_dtln": dtln,
    "S10_aec_gtcrn": aec_gtcrn,
    "S11_aec_only": aec_only,
}
