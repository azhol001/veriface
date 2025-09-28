# adk_project/agents/lipsync_agent/agent.py
# LipSync Agent (conservative + talk-gated + lag-comp + dynamic threshold)

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
        res = fm.process(f)  # frames should be RGB from loader
        if not res.multi_face_landmarks:
            raw.append(np.nan)
            continue
        have_face += 1
        face = res.multi_face_landmarks[0]
        pts = np.array([(lm.x * w, lm.y * h) for lm in face.landmark], dtype=np.float32)
        raw.append(mouth_open_ratio(pts))
    fm.close()

    face_coverage = float(have_face) / float(len(frames) or 1)

    # Fill NaNs conservatively: forward fill with median fallback
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
    hop = max(1, int(sr / 50))              # ~20 ms
    win = max(hop * 2, int(0.04 * sr))      # >=40 ms
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop)[0] if len(y) else np.zeros(1)
    avg_rms = float(np.mean(rms)) if rms.size else 0.0
    t_env = np.arange(len(rms)) * (hop / sr)

    env = np.interp(frame_times_s, t_env, rms, left=rms[0], right=rms[-1]).astype(np.float32) if rms.size else \
          np.zeros(len(frame_times_s), dtype=np.float32)

    mu, sd = float(env.mean()), float(env.std() + 1e-6)
    return (env - mu) / sd, avg_rms

def zscore(x: np.ndarray) -> np.ndarray:
    mu, sd = float(x.mean()), float(x.std() + 1e-6)
    return (x - mu) / sd

def _moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    w = max(1, w)
    k = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x.astype(np.float32), k, mode="same")

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

def _boolean_close(mask: np.ndarray, min_len_frames: int) -> np.ndarray:
    """Fill gaps shorter than min_len_frames between True-runs."""
    if min_len_frames <= 1:
        return mask
    idx = np.where(mask)[0]
    if idx.size == 0:
        return mask
    out = mask.copy()
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    for a, b in zip(runs[:-1], runs[1:]):
        gap = b[0] - a[-1] - 1
        if 0 < gap < min_len_frames:
            out[a[-1]+1 : b[0]] = True
    return out

@dataclass
class LipSyncResult:
    suspicious_spans: List[dict]
    metric: str = "lipsync_corr_lag"
    details: Dict | None = None

def analyze_lipsync(
    video_path: str,
    fps: int = 25,
    # safer defaults
    corr_win_s: float = 0.9,        # a bit longer window for stability
    corr_thresh: float = 0.18,      # absolute ceiling; dynamic will lower but not below 0.12
    min_span_s: float = 1.20,       # require longer sustained issue
    face_cov_req: float = 0.60,     # need ≥60% tracked face frames
    weak_audio_rms: float = 0.012,  # skip if audio is very quiet
    talk_z_thresh: float = 0.50,    # speech gate is slightly stricter
    talk_close_s: float = 0.25,
    talk_cov_req: float = 0.70,     # inside a flagged span, ≥70% must be talk
    lag_search_ms: int = 350,
    min_useful_lag_ms: int = 80,
    lag_gain_thresh: float = 0.05   # only accept lag if r improves enough
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

    # normalize mouth series
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
    env_z, avg_rms = audio_envelope(y, sr, frame_times)
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

    # ---- talk mask (gate evaluation to speech segments) ----
    env_s = _moving_avg(env_z, max(3, int(0.10 * fps)))
    talk_mask = env_s > talk_z_thresh
    talk_mask = _boolean_close(talk_mask, min_len_frames=max(2, int(talk_close_s * fps)))
    talk_fraction = float(np.mean(talk_mask)) if talk_mask.size else 0.0

    # If almost no talk, don't judge
    if talk_fraction < 0.20:
        return LipSyncResult(
            suspicious_spans=[],
            details={
                "video_duration_s": float(vid_dur),
                "fps": int(fps),
                "face_coverage": round(float(face_cov), 3),
                "note": f"insufficient speech (talk_fraction={talk_fraction:.2f})",
            },
        )

    # ---- sliding correlation ----
    win = max(5, int(corr_win_s * fps))
    r_raw = sliding_corr(mouth, env_z, win=win)

    # ---- lag search (±lag_search_ms), accept only if it measurably helps ----
    lags = np.arange(-(len(mouth) - 1), len(mouth))
    xcorr = np.correlate(mouth, env_z, mode="full")
    max_lag_frames = max(1, int((lag_search_ms / 1000.0) * fps))
    m = (lags >= -max_lag_frames) & (lags <= max_lag_frames)
    best_lag_frames = int(lags[m][np.argmax(xcorr[m])]) if np.any(m) else 0
    lag_ms = float(best_lag_frames * 1000.0 / fps)

    def _shift_env(env: np.ndarray, lag_frames: int) -> np.ndarray:
        dt = 1.0 / float(fps)
        shifted = np.interp(
            np.arange(len(env)) * dt,
            np.arange(len(env)) * dt + (lag_frames * dt),
            env,
            left=env[0],
            right=env[-1],
        ).astype(np.float32)
        return shifted

    used_lag = False
    env_aligned = env_z.copy()
    mean_corr = float(np.nanmean(r_raw))
    if abs(best_lag_frames) >= int((min_useful_lag_ms / 1000.0) * fps):
        env_try = _shift_env(env_z, best_lag_frames)
        r_try = sliding_corr(mouth, env_try, win=win)
        mean_try = float(np.nanmean(r_try))
        if mean_try - mean_corr >= lag_gain_thresh:
            env_aligned = env_try
            r_raw = r_try
            used_lag = True

    mean_corr_aligned = float(np.nanmean(r_raw))

    # ---- dynamic threshold relative to talk baseline ----
    r_on_talk = r_raw[talk_mask] if r_raw.size and talk_mask.any() else np.array([], dtype=np.float32)
    if r_on_talk.size:
        p30 = float(np.percentile(r_on_talk, 30))
        dyn_thresh = min(corr_thresh, max(0.12, p30 - 0.05))  # never go below 0.12
    else:
        dyn_thresh = corr_thresh

    # If overall aligned correlation is already healthy, don't accuse
    if mean_corr_aligned >= 0.22:
        return LipSyncResult(
            suspicious_spans=[],
            details={
                "fps": int(fps),
                "corr_window_s": float(corr_win_s),
                "corr_thresh": float(corr_thresh),
                "min_span_s": float(min_span_s),
                "face_coverage": round(float(face_cov), 3),
                "video_duration_s": float(vid_dur),
                "avg_audio_rms": float(round(avg_rms, 5)),
                "lag_ms": float(round(lag_ms, 1)),
                "used_lag_comp": bool(used_lag),
                "mean_corr": float(round(mean_corr, 3)),
                "mean_corr_aligned": float(round(mean_corr_aligned, 3)),
                "talk_fraction": float(round(talk_fraction, 3)),
                "note": "good overall correlation; skip flags",
            },
        )

    # ---- conservative flagging: only sustained low correlation during talk ----
    low_corr = (r_raw < dyn_thresh) & talk_mask
    if low_corr.any():
        low_corr = _boolean_close(low_corr, min_len_frames=max(2, int(0.20 * fps)))

    spans = spans_from_mask(low_corr, frame_times, min_s=min_span_s)

    # enforce talk coverage inside each span
    valid_spans: List[Tuple[float, float]] = []
    if spans:
        talk_ts = np.array(frame_times, dtype=np.float32)
        for (s, e) in spans:
            in_span = (talk_ts >= s) & (talk_ts <= e)
            cov = float(np.mean(talk_mask[in_span])) if np.any(in_span) else 0.0
            if cov >= talk_cov_req:
                valid_spans.append((s, e))

    out_spans = [
        {"start": float(s), "end": float(e), "reason": f"Low lip↔audio correlation (r<{dyn_thresh:.2f})"}
        for (s, e) in valid_spans
    ]

    # fraction of talk time that is flagged
    flagged_talk_fraction = 0.0
    total_talk_secs = 0.0
    if talk_mask.any():
        dt = 1.0 / float(fps)
        total_talk_secs = float(np.sum(talk_mask)) * dt
        if valid_spans:
            flagged_mask = np.zeros_like(talk_mask, dtype=bool)
            ft = np.array(frame_times)
            for s, e in valid_spans:
                idx = (ft >= s) & (ft <= e)
                flagged_mask[idx] = True
            flagged_talk_fraction = float(np.sum(flagged_mask & talk_mask) * dt / (total_talk_secs + 1e-6))

    # Final conservative gate: require meaningful flagged talk coverage
    min_flag_secs = max(1.6, 0.10 * total_talk_secs)  # ≥1.6s or 10% of talk, whichever larger
    total_flag_secs = sum(e - s for (s, e) in valid_spans)
    if total_flag_secs < min_flag_secs:
        out_spans = []  # not enough sustained evidence

    details = {
        "fps": int(fps),
        "corr_window_s": float(corr_win_s),
        "corr_thresh": float(corr_thresh),
        "min_span_s": float(min_span_s),
        "face_coverage": round(float(face_cov), 3),
        "video_duration_s": float(vid_dur),
        "avg_audio_rms": float(round(avg_rms, 5)),
        "lag_ms": float(round(lag_ms, 1)),
        "used_lag_comp": bool(used_lag),
        "mean_corr": float(round(mean_corr, 3)),
        "mean_corr_aligned": float(round(mean_corr_aligned, 3)),
        "talk_fraction": float(round(talk_fraction, 3)),
        "flagged_talk_fraction": float(round(flagged_talk_fraction, 3)),
        "min_flag_secs": float(round(min_flag_secs, 2)),
        "total_flag_secs": float(round(total_flag_secs, 2)),
        "dyn_thresh_final": float(round(dyn_thresh, 2)),
    }
    return LipSyncResult(suspicious_spans=out_spans, details=details)

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="LipSync Agent (talk-gated, lag-comp, dynamic, conservative)")
    p.add_argument("--in", dest="inp", required=True, help="Path to video (mp4)")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--win", type=float, default=0.9, help="Correlation window seconds")
    p.add_argument("--rmin", type=float, default=0.18, help="Low-corr ceiling (dynamic may go lower, not below 0.12)")
    p.add_argument("--minspan", type=float, default=1.2, help="Min suspicious span seconds")
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


