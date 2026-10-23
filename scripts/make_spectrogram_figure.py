"""Draw the before/after spectrogram used in the README (numpy only).

Usage: python scripts/make_spectrogram_figure.py examples/audio docs/figures/noise-spectrogram.svg
"""

import base64
import struct
import sys
import wave
import zlib

import numpy as np

PANELS = [
    ("clean-reference.wav", "Clean reference"),
    ("noise-input.wav", "Input: speech + noise (5 dB SNR)"),
    ("noise-output.wav", "Output: GTCRN + protection"),
]
# Five viridis anchor colours, interpolated linearly.
ANCHORS = np.array(
    [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float
)


def load(path):
    with wave.open(path) as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(float) / 32768


def spectrogram_db(x, n=512, hop=256):
    win = np.hanning(n)
    frames = np.array([x[i : i + n] * win for i in range(0, len(x) - n, hop)])
    mag = np.abs(np.fft.rfft(frames, axis=1)).T
    mag = mag[:256].reshape(128, 2, -1).mean(1)
    return 20 * np.log10(mag + 1e-6)


def colormap(v):
    v = np.clip(v, 0, 1) * (len(ANCHORS) - 1)
    i = np.minimum(v.astype(int), len(ANCHORS) - 2)
    f = (v - i)[..., None]
    return (ANCHORS[i] * (1 - f) + ANCHORS[i + 1] * f).astype(np.uint8)


def png_bytes(rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))

    def chunk(tag, data):
        body = struct.pack(">I", len(data)) + tag + data
        return body + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def main(audio_dir, out_path):
    specs = [spectrogram_db(load(f"{audio_dir}/{name}")) for name, _ in PANELS]
    vmax = max(s.max() for s in specs)
    vmin = vmax - 80
    width, height, top, gap, left = 760, 150, 34, 34, 56
    parts = []
    for k, (s, (_, title)) in enumerate(zip(specs, PANELS)):
        img = colormap((s[::-1] - vmin) / (vmax - vmin))
        data = base64.b64encode(png_bytes(img)).decode()
        y = top + k * (height + gap)
        parts.append(f'<text x="{left}" y="{y - 8}" class="t">{title}</text>')
        parts.append(
            f'<image x="{left}" y="{y}" width="{width}" height="{height}" '
            f'preserveAspectRatio="none" href="data:image/png;base64,{data}"/>'
        )
        parts.append(
            f'<rect x="{left}" y="{y}" width="{width}" height="{height}" '
            'fill="none" stroke="#888" stroke-width="0.6"/>'
        )
        for khz in (0, 4, 8):
            yy = y + height - height * khz / 8
            parts.append(
                f'<text x="{left - 6}" y="{yy + 4}" class="a" text-anchor="end">{khz}</text>'
            )
    y_axis = top + 3 * (height + gap) - gap + 16
    for t in range(12):
        xx = left + width * t / 11
        parts.append(f'<text x="{xx}" y="{y_axis}" class="a" text-anchor="middle">{t}</text>')
    parts.append(
        f'<text x="{left + width / 2}" y="{y_axis + 18}" class="a" '
        'text-anchor="middle">time (s)</text>'
    )
    yc = top + 1.5 * height + gap
    parts.append(
        f'<text x="16" y="{yc}" class="a" text-anchor="middle" '
        f'transform="rotate(-90 16 {yc})">frequency (kHz)</text>'
    )
    total_h, total_w = y_axis + 30, left + width + 12
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} {total_h}" '
        f'width="{total_w}" height="{total_h}" font-family="Helvetica, Arial, sans-serif">\n'
        "<style>.t{font-size:13px;fill:#444}.a{font-size:11px;fill:#666}</style>\n"
        '<rect width="100%" height="100%" fill="#ffffff"/>\n' + "\n".join(parts) + "\n</svg>"
    )
    with open(out_path, "w") as f:
        f.write(svg)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
