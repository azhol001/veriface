# adk_project/agents/lipsync_agent/agent.py
# LipSync Agent (MVP): mouth-open signal from Face Mesh vs audio envelope
# Flags spans where correlation is low or lag exceeds a threshold.

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import mediapipe as mp
import librosa

from adk_project.tools.media_io import load_video_frames, load_audio_mono

# ---- Landmark indices (MediaPipe Face Mesh, 468 pts) ----
# We'll compute a simple "mouth aspect ratio" (MAR):
# vertical = distance between upper-inner (13) and lower-inner (14)
# horizontal = distance between left (61) and right (291) mouth corners
UP_INNER = 13
LOW_INNER = 14
MOUTH_L = 61
MOUTH_R = 291

def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def mouth_open_ratio(pts: np.ndarray) -> float:
    """pts: (468,2) pixel coords -> MAR ~ vertical / horizontal"""
    v = _euclidean(pts[UP_INNER], pts[LOW_INNER])
    h = _euclidean(pts[MOUTH_L], pts[MOUTH_R]) + 1e-6
    return v / h

def extract_mouth_series(frames: List[np.ndarray], fps: int) -> Tuple[np.ndarray, List[float]]:
    mp_face_mesh = mp.solutions.face_mesh
    fm = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    series = []
    for f in frames:
        h, w = f.shape[:2]
        res = fm.process(f)
        if not res.multi_face_landmarks:
            series.append(np.nan)
            continue
        face = res.multi_face_landmarks[0]
        pts = np.array([(lm.x * w, lm.y * h) for lm in face.landmark], dtype=np.float32)
        series.append(mouth_open_ratio(pts))
    fm.close()

    times = [i / float(fps) for i in range(len(series))]
    # fill missing with forward fill -> median
    arr = np.array(series, dtype=np.float32)
    if np.isnan(arr).any():
        # simple fill: replace nans with last valid, then overall median if leading NaNs
        isn = np.isnan(arr)
        if (~isn).any():
            last = np.nanmedian(arr)
            for i in range(len(arr)):
                if isn[i]:
                    arr[i] = last
                else:
                    last = arr[i]
        else:
            arr[:] = 0.0
    return arr, times

def audio_envelope(y: np.ndarray, sr: int, frame_times_s: List[float]) -> np.ndarray:
    """Compute RMS envelope and resample to video frame times"""
    # frame RMS with ~25 fps equivalent hop to get smooth curve
    hop = max(1, int(sr / 50))
    win = max(hop * 2, int(0.04 * sr))
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop)[0]  # (T,)
    t_env = np.arange(len(rms)) * (hop / sr)
    # resample envelope to frame timestamps with linear interp
    env = np.interp(frame_times_s, t_env, rms, left=rms[0], right=rms[-1]).astype(np.float32)
    # z-normalize
    mu, sd = float(env.mean()), float(env.std() + 1e-6)
    return (env - mu) / sd

def zscore(x: np.ndarray) -> np.ndarray:
    mu, sd = float(x.mean()), float(x.std() + 1e-6)
    return (x - mu) / sd

def sliding_corr(a: np.ndarray, b: np.ndarray, win: int) -> np.ndarray:
    """Pearson r over a centered sliding window (odd win)."""
    n = len(a)
    w = max(3, win | 1)  # force odd
    pad = w // 2
    A = np.pad(a, (pad, pad), mode='edge')
    B = np.pad(b, (pad, pad), mode='edge')
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        aa = A[i:i+w]; bb = B[i:i+w]
        ra = aa - aa.mean(); rb = bb - bb.mean()
        denom = (np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum()) + 1e-6)
        out[i] = float((ra*rb).sum() / denom)
    return out

def estimate_lag(a: np.ndarray, b: np.ndarray, fps: int, max_ms: int = 400) -> float:
    """
    Return lag in milliseconds (positive = audio leads) using full cross-correlation.
    Uses np.correlate (length 2N-1) and searches only within ±max_ms.
    """
    n = len(a)
    if n == 0 or len(b) != n:
        return 0.0

    a0 = zscore(a)
    b0 = zscore(b)

    # full cross-correlation (lags from -(n-1) .. +(n-1))
    xcorr = np.correlate(a0, b0, mode="full")  # length 2n-1
    lags = np.arange(-(n - 1), n)

    # restrict to ±max_ms
    max_lag_frames = max(1, int((max_ms / 1000.0) * fps))
    m = (lags >= -max_lag_frames) & (lags <= max_lag_frames)

    if not np.any(m):
        return 0.0

    best_lag = int(lags[m][np.argmax(xcorr[m])])
    lag_ms = best_lag * 1000.0 / fps
    return float(lag_ms)

def spans_from_masks(mask: np.ndarray, times: List[float], min_s: float = 0.6) -> List[Tuple[float, float]]:
    spans = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = times[i]
        if (not flag) and start is not None:
            end = times[i]
            if end - start >= min_s:
                spans.append((start, end))
            start = None
    if start is not None:
        end = times[-1]
        if end - start >= min_s:
            spans.append((start, end))
    return spans

@dataclass
class LipSyncResult:
    suspicious_spans: List[dict]
    metric: str = "lipsync_corr_lag"
    details: dict = None

def analyze_lipsync(video_path: str, fps: int = 25,
                    corr_win_s: float = 1.0,
                    corr_thresh: float = 0.25,
                    lag_bad_ms: int = 220) -> LipSyncResult:
    frames, frame_times, vid_dur = load_video_frames(video_path, fps=fps)
    mouth, _ = extract_mouth_series(frames, fps=fps)
    mouth = zscore(mouth)

    y, sr, _ = load_audio_mono(video_path, target_sr=16000)
    env = audio_envelope(y, sr, frame_times)

    # sliding correlation
    win = max(3, int(corr_win_s * fps))
    r = sliding_corr(mouth, env, win=win)

    # low correlation mask
    low_corr = r < corr_thresh

    # overall lag estimate
    lag_ms = estimate_lag(mouth, env, fps=fps, max_ms=400)
    bad_lag = abs(lag_ms) >= lag_bad_ms

    spans = spans_from_masks(low_corr, frame_times, min_s=max(0.6, corr_win_s * 0.6))
    out_spans = [{"start": float(s), "end": float(e),
                  "reason": f"Low lip↔audio correlation (r<{corr_thresh})"} for (s, e) in spans]

    if bad_lag:
        # add a whole-clip lag note (UX: coordinator will merge)
        out_spans.append({"start": 0.0, "end": float(vid_dur),
                          "reason": f"Lip–audio lag ≈ {lag_ms:.0f} ms (>|={lag_bad_ms} ms)"})

    details = {
        "fps": int(fps),
        "corr_window_s": float(corr_win_s),
        "corr_thresh": float(corr_thresh),
        "lag_ms": float(round(lag_ms, 1)),
        "video_duration_s": float(vid_dur),
        "mean_corr": float(round(np.nanmean(r), 3)),
    }
    return LipSyncResult(suspicious_spans=out_spans, details=details)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="LipSync Agent MVP")
    p.add_argument("--in", dest="inp", required=True, help="Path to video (mp4)")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--win", type=float, default=1.0, help="Correlation window seconds")
    p.add_argument("--rmin", type=float, default=0.25, help="Low-corr threshold")
    p.add_argument("--lagbad", type=int, default=220, help="ms lag threshold for flag")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    res = analyze_lipsync(args.inp, fps=args.fps, corr_win_s=args.win, corr_thresh=args.rmin, lag_bad_ms=args.lagbad)
    payload = {
        "agent": "lipsync",
        "metric": res.metric,
        "suspicious_spans": res.suspicious_spans,
        "details": res.details,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print("✅ LipSync Agent")
        print(json.dumps(payload, indent=2))
