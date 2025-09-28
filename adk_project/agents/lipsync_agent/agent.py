# adk_project/agents/lipsync_agent/agent.py
# LipSync Agent (conservative): mouth-open signal vs audio envelope
# Flags only sustained, confident low-correlation spans.

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import mediapipe as mp
import librosa

from adk_project.tools.media_io import load_video_frames, load_audio_mono

# ---- Mouth landmarks (MediaPipe Face Mesh, 468 pts) ----
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

def extract_mouth_series(frames: List[np.ndarray], fps: int) -> Tuple[np.ndarray, List[float], float]:
    """Returns (series, frame_times, face_coverage)."""
    mp_face_mesh = mp.solutions.face_mesh
    fm = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    raw = []
    have_face = 0
    for f in frames:
        h, w = f.shape[:2]
        res = fm.process(f)
        if not res.multi_face_landmarks:
            raw.append(np.nan)
            continue
        have_face += 1
        face = res.multi_face_landmarks[0]
        pts = np.array([(lm.x * w, lm.y * h) for lm in face.landmark], dtype=np.float32)
        raw.append(mouth_open_ratio(pts))
    fm.close()

    face_coverage = float(have_face) / float(len(frames) or 1)

    # Fill NaNs conservatively: forward-fill with median fallback
    arr = np.array(raw, dtype=np.float32)
    if np.isnan(arr).any():
        isn = np.isnan(arr)
        if (~isn).any():
            last = np.nanmedian(arr[~isn])
            for i in range(len(arr)):
                if isn[i]:
                    arr[i] = last
                else:
                    last = arr[i]
        else:
            arr[:] = 0.0

    times = [i / float(fps) for i in range(len(arr))]
    return arr, times, face_coverage

def audio_envelope(y: np.ndarray, sr: int, frame_times_s: List[float]) -> Tuple[np.ndarray, float]:
    """Compute RMS envelope aligned to video frames; returns (env_z, avg_rms)."""
    hop = max(1, int(sr / 50))
    win = max(hop * 2, int(0.04 * sr))
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop)[0]  # (T,)
    avg_rms = float(np.mean(rms)) if rms.size else 0.0
    t_env = np.arange(len(rms)) * (hop / sr)
    env = np.interp(frame_times_s, t_env, rms, left=rms[0] if rms.size else 0.0, right=rms[-1] if rms.size else 0.0).astype(np.float32)
    mu, sd = float(env.mean()), float(env.std() + 1e-6)
    return (env - mu) / sd, avg_rms

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

def spans_from_mask(mask: np.ndarray, times: List[float], min_s: float) -> List[Tuple[float, float]]:
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
    details: Dict | None = None

def analyze_lipsync(
    video_path: str,
    fps: int = 25,
    corr_win_s: float = 0.8,     # shorter window smooths less (was 1.0)
    corr_thresh: float = 0.20,   # more tolerant (was 0.25)
    min_span_s: float = 1.00,    # require sustained issue (was ~0.6)
    face_cov_req: float = 0.60,  # need ≥60% tracked face frames
    weak_audio_rms: float = 0.005  # skip if audio is ultra-quiet
) -> LipSyncResult:
    # ---- video → mouth series ----
    frames, frame_times, vid_dur = load_video_frames(video_path, fps=fps)
    mouth, _, face_cov = extract_mouth_series(frames, fps=fps)

    # not enough face → don't judge
    if face_cov < face_cov_req:
        return LipSyncResult(
            suspicious_spans=[],
            details={
                "video_duration_s": float(vid_dur),
                "fps": int(fps),
                "face_coverage": round(float(face_cov), 3),
                "note": f"insufficient face coverage (<{face_cov_req:.0%})",
            },
        )

    mouth = zscore(mouth)

    # ---- audio → env ----
    y, sr, _ = load_audio_mono(video_path, target_sr=16000)
    if y is None or len(y) < sr // 2:
        return LipSyncResult(
            suspicious_spans=[],
            details={
                "video_duration_s": float(vid_dur),
                "fps": int(fps),
                "face_coverage": round(float(face_cov), 3),
                "note": "no/short audio",
            },
        )
    env, avg_rms = audio_envelope(y, sr, frame_times)
    if avg_rms < weak_audio_rms:
        return LipSyncResult(
            suspicious_spans=[],
            details={
                "video_duration_s": float(vid_dur),
                "fps": int(fps),
                "face_coverage": round(float(face_cov), 3),
                "note": f"very weak audio (avg_rms={avg_rms:.4f})",
            },
        )

    # ---- sliding correlation ----
    win = max(3, int(corr_win_s * fps))
    r = sliding_corr(mouth, env, win=win)

    # ---- conservative flagging: only sustained low correlation ----
    low_corr = r < corr_thresh
    spans = spans_from_mask(low_corr, frame_times, min_s=min_span_s)

    out_spans = [
        {"start": float(s), "end": float(e), "reason": f"Low lip↔audio correlation (r<{corr_thresh})"}
        for (s, e) in spans
    ]

    # NOTE: We do NOT add a whole-clip lag span; we only report lag in details.
    # Global lag is useful context but easily over-flags; keep it descriptive only.
    # (If you still want it, gate by low-corr coverage first.)
    # Global lag estimate (for details)
    try:
        a0, b0 = mouth, env
        xcorr = np.correlate(a0, b0, mode="full")
        lags = np.arange(-(len(a0) - 1), len(a0))
        max_ms = 400
        max_lag_frames = max(1, int((max_ms / 1000.0) * fps))
        m = (lags >= -max_lag_frames) & (lags <= max_lag_frames)
        best_lag = int(lags[m][np.argmax(xcorr[m])])
        lag_ms = float(best_lag * 1000.0 / fps)
    except Exception:
        lag_ms = 0.0

    details = {
        "fps": int(fps),
        "corr_window_s": float(corr_win_s),
        "corr_thresh": float(corr_thresh),
        "min_span_s": float(min_span_s),
        "face_coverage": round(float(face_cov), 3),
        "video_duration_s": float(vid_dur),
        "avg_audio_rms": float(round(avg_rms, 5)),
        "lag_ms": float(round(lag_ms, 1)),
        "mean_corr": float(round(np.nanmean(r), 3)),
    }
    return LipSyncResult(suspicious_spans=out_spans, details=details)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="LipSync Agent (conservative)")
    p.add_argument("--in", dest="inp", required=True, help="Path to video (mp4)")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--win", type=float, default=0.8, help="Correlation window seconds")
    p.add_argument("--rmin", type=float, default=0.20, help="Low-corr threshold")
    p.add_argument("--minspan", type=float, default=1.0, help="Min suspicious span seconds")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    res = analyze_lipsync(
        args.inp, fps=args.fps, corr_win_s=args.win, corr_thresh=args.rmin, min_span_s=args.minspan
    )
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

