# adk_project/agents/voice_agent/agent.py
# Voice Agent (conservative, quality-gated):
# Flags spans only when BOTH (a) spectral flatness is unusually high AND (b) MFCC variance is unusually low,
# and only if audio is sufficiently long and strong. Skips weak/short audio and filters micro-blips.

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import librosa

from adk_project.tools.media_io import load_audio_mono

# ------------------------ helpers ------------------------

def _rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    if x.size == 0 or w <= 1 or x.size < w:
        return x.copy()
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(x, w)
    med = np.median(sw, axis=1)
    # center back to original length by padding
    pad = w // 2
    return np.pad(med, (pad, x.size - med.size - pad), mode="edge")

def _iqr_bounds(x: np.ndarray, k: float = 1.0) -> Tuple[float, float]:
    """Robust bounds using IQR (conservative)."""
    x = x[np.isfinite(x)]
    if x.size < 16:
        return np.inf, -np.inf  # essentially disables
    q1, q3 = np.percentile(x, [25, 75])
    iqr = max(1e-8, q3 - q1)
    upper = q3 + k * iqr
    lower = q1 - k * iqr
    return float(upper), float(lower)

def _merge_spans(spans: List[Tuple[float, float, str]], join_gap: float = 0.2) -> List[Tuple[float, float, str]]:
    spans = sorted(spans, key=lambda s: s[0])
    out: List[Tuple[float, float, str]] = []
    for s, e, r in spans:
        if not out:
            out.append([s, e, r])  # type: ignore
        else:
            ls, le, lr = out[-1]  # type: ignore
            if s <= le + join_gap:
                out[-1][1] = max(le, e)  # type: ignore
            else:
                out.append([s, e, r])  # type: ignore
    return [(float(s), float(e), r) for s, e, r in out]  # type: ignore

def _span_len(s: Tuple[float, float]) -> float:
    return max(0.0, float(s[1]) - float(s[0]))

# ------------------------ API ------------------------

@dataclass
class VoiceResult:
    suspicious_spans: List[dict]
    metric: str = "voice_anomaly_mfcc_flatness"
    details: Optional[Dict] = None

def analyze_voice(
    path: str,
    *,
    target_sr: int = 16000,
    win_s: float = 0.5,
    hop_s: float = 0.25,
    min_span_s: float = 1.0,
    weak_audio_rms: float = 0.01,     # raised from 0.005 → require stronger audio
    min_duration_s: float = 4.0,      # require ≥4s of audio
    iqr_k: float = 1.0,
    fps: int = 25,   # optional, ignored (compatibility with coordinator)
) -> VoiceResult:
    """
    Conservative voice anomaly detector.
    Flags spans only where BOTH conditions hold (after smoothing):
      - spectral flatness is unusually high (upper IQR bound)
      - MFCC variance is unusually low  (lower IQR bound)
    Gated by simple audio-quality checks (duration & RMS).
    """

    # ---- load audio ----
    y, sr, dur = load_audio_mono(path, target_sr=target_sr)

    # duration gate
    if (dur or 0.0) < min_duration_s:
        return VoiceResult(
            suspicious_spans=[],
            details={
                "note": f"audio too short ({dur:.2f}s < {min_duration_s}s)",
                "sr": int(sr),
                "duration_s": float(dur or 0.0),
            },
        )

    if y is None or len(y) < sr // 2:
        return VoiceResult(
            suspicious_spans=[],
            details={"note": "no/short audio", "sr": int(sr), "duration_s": float(dur or 0.0)}
        )

    # quick RMS gate (ultra-quiet audio can look "flat" but isn't evidence)
    rms_full = float(np.sqrt(np.mean(y**2)))
    if rms_full < weak_audio_rms:
        return VoiceResult(
            suspicious_spans=[],
            details={
                "note": f"very weak audio (rms={rms_full:.4f} < {weak_audio_rms})",
                "sr": int(sr),
                "duration_s": float(dur or 0.0),
            },
        )

    # ---- frame STFT ----
    hop = int(sr * hop_s)
    win = int(sr * win_s)
    hop = max(1, hop)
    win = max(hop * 2, win)

    # n_fft must be >= win_length; choose next power of two for efficiency
    import math
    n_fft = 1 << int(math.ceil(math.log2(max(2048, win))))

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win))  # (freq, frames)

    # spectral flatness in [0..1], higher = flatter
    flat = librosa.feature.spectral_flatness(S=S).flatten()

    # MFCC variance over coefficients per frame (proxy for formant dynamics)
    power = (S ** 2).astype(np.float32)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(power + 1e-12), n_mfcc=13)
    mfcc_var = np.var(mfcc, axis=0)

    # ---- smooth with rolling median (~0.6 s) ----
    smooth_w = max(5, int(0.6 / hop_s))  # e.g., hop=0.25s -> w≈2-3, clamp to >=5
    flat_s = _rolling_median(flat, smooth_w)
    mfcc_s = _rolling_median(mfcc_var, smooth_w)

    n = min(flat_s.size, mfcc_s.size)
    if n == 0:
        return VoiceResult(
            suspicious_spans=[],
            details={"note": "too short after smoothing", "sr": int(sr), "duration_s": float(dur or 0.0)}
        )

    flat_s = flat_s[:n]
    mfcc_s = mfcc_s[:n]

    # ---- robust thresholds (conservative) ----
    up_flat, _ = _iqr_bounds(flat_s, k=iqr_k)  # high flatness
    _, lo_mvar = _iqr_bounds(mfcc_s, k=iqr_k)  # low mfcc variance

    # logical AND mask
    flags = (flat_s >= up_flat) & (mfcc_s <= lo_mvar)

    # ---- build spans (sustain & merge) ----
    spans: List[Tuple[float, float, str]] = []
    if np.any(flags):
        # timestamps per frame center
        t = (np.arange(n) * hop / sr) + (win / (2 * sr))
        # group consecutive flagged frames
        idx = np.where(flags)[0]
        groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for g in groups:
            if g.size == 0:
                continue
            s = float(t[g[0]])
            e = float(t[g[-1]])
            if _span_len((s, e)) >= min_span_s:
                spans.append((s, e, "Audio anomaly (flatness↑ & MFCC var↓)"))
    spans = _merge_spans(spans, join_gap=0.2)

    out_spans = [{"start": float(s), "end": float(e), "reason": r} for s, e, r in spans]
    details = {
        "sr": int(sr),
        "duration_s": float(dur or 0.0),
        "window_s": float(win_s),
        "hop_s": float(hop_s),
        "smooth_w_frames": int(smooth_w),
        "weak_audio_rms": float(weak_audio_rms),
        "min_duration_s": float(min_duration_s),
        "iqr_k": float(iqr_k),
        "thresholds": {"flat_upper": float(up_flat), "mfcc_var_lower": float(lo_mvar)},
        "frames_flagged": int(np.count_nonzero(flags)),
        "frames_total": int(n),
        "rms_full": float(round(rms_full, 5)),
    }
    return VoiceResult(suspicious_spans=out_spans, details=details)

# ------------------------ CLI ------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Voice Agent (conservative)")
    p.add_argument("--in", dest="inp", required=True, help="Path to video/audio (mp4/wav)")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--win", type=float, default=0.5)
    p.add_argument("--hop", type=float, default=0.25)
    p.add_argument("--minspan", type=float, default=1.0)
    p.add_argument("--weak_rms", type=float, default=0.01)
    p.add_argument("--mindur", type=float, default=4.0)
    p.add_argument("--iqrk", type=float, default=1.0)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    res = analyze_voice(
        args.inp,
        target_sr=args.sr,
        win_s=args.win,
        hop_s=args.hop,
        min_span_s=args.minspan,
        weak_audio_rms=args.weak_rms,
        min_duration_s=args.mindur,
        iqr_k=args.iqrk,
    )
    payload = {
        "agent": "voice",
        "metric": res.metric,
        "suspicious_spans": res.suspicious_spans,
        "details": res.details,
    }
    print(json.dumps(payload if args.json else payload, indent=None if args.json else 2))
