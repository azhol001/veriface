# adk_project/agents/blink_agent/agent.py
# Blink Agent (conservative): uses MediaPipe Face Mesh to estimate eye aspect ratio (EAR)
# Finds blinks and flags unusually long no-blink gaps as "suspicious" spans,
# BUT only if the clip is long enough and the face was tracked reliably.

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import mediapipe as mp

from adk_project.tools.media_io import load_video_frames

# -------------- EAR helpers --------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]    # [outer, upper1, upper2, inner, lower1, lower2]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]  # mirrored indices

def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def _ear_from_landmarks(landmarks_2d: np.ndarray, eye_idx: List[int]) -> float:
    p0, p1, p2, p3, p4, p5 = [landmarks_2d[i] for i in eye_idx]
    v1 = _euclidean(p1, p4)
    v2 = _euclidean(p2, p5)
    v = (v1 + v2) / 2.0
    h = _euclidean(p0, p3) + 1e-6
    return v / h

# -------------- Blink detection --------------
def detect_blinks(
    frames: List[np.ndarray],
    fps: int = 25,
    ear_thresh: float = 0.17,        # slightly stricter closing threshold (↓ from 0.19)
    min_closed_frames: int = 3
) -> tuple[list[float], list[float], float]:
    """
    Returns:
      blink_times_s: list of timestamps (seconds) where a blink is detected
      ear_series: list of EAR per frame (np.nan if face not found)
      face_coverage: fraction of frames with a detected face (0..1)
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    blink_times: List[float] = []
    ear_series: List[float] = []
    closed_streak = 0
    have_face = 0

    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        results = face_mesh.process(frame)

        if not results.multi_face_landmarks:
            ear_series.append(np.nan)
            closed_streak = 0
            continue

        have_face += 1
        face = results.multi_face_landmarks[0]
        pts = np.array([(lm.x * w, lm.y * h) for lm in face.landmark], dtype=np.float32)  # (468,2)

        le = _ear_from_landmarks(pts, LEFT_EYE)
        re = _ear_from_landmarks(pts, RIGHT_EYE)
        ear = (le + re) / 2.0
        ear_series.append(ear)

        if ear < ear_thresh:
            closed_streak += 1
        else:
            if closed_streak >= min_closed_frames:
                blink_center_idx = i - closed_streak // 2
                blink_times.append(blink_center_idx / float(fps))
            closed_streak = 0

    if closed_streak >= min_closed_frames:
        blink_center_idx = len(frames) - closed_streak // 2
        blink_times.append(blink_center_idx / float(fps))

    face_mesh.close()
    face_coverage = float(have_face) / float(len(frames) or 1)
    return blink_times, ear_series, face_coverage

def suspicious_no_blink_spans(
    blink_times_s: List[float],
    video_duration_s: float,
    face_coverage: float,
    *,
    max_gap_ok: float = 16.0,     # ↑ more tolerant than 12s
    min_video_len: float = 12.0,  # don’t judge very short clips
    require_coverage: float = 0.60
) -> List[Tuple[float, float, str]]:
    """
    Mark spans only if: clip is long enough AND we had enough tracked-face coverage.
    """
    spans: List[Tuple[float, float, str]] = []

    # Not enough evidence → do not flag
    if video_duration_s < min_video_len or face_coverage < require_coverage:
        return spans

    if not blink_times_s:
        # Only flag a full-clip "no blink" if the clip is clearly long enough
        long_enough = video_duration_s >= max(max_gap_ok + 4.0, 20.0)
        if long_enough:
            spans.append((0.0, video_duration_s,
                          f"No blinks detected for {video_duration_s:.1f}s"))
        return spans

    times = [0.0] + sorted(blink_times_s) + [video_duration_s]
    for a, b in zip(times[:-1], times[1:]):
        gap = b - a
        if gap >= max_gap_ok:
            spans.append((a, b, f"No blink for {gap:.1f}s (>= {max_gap_ok:.0f}s)"))
    return spans

# -------------- Result wrapper --------------
@dataclass
class BlinkResult:
    suspicious_spans: List[dict]
    metric: str = "blink_no_gap"
    details: Dict | None = None

def analyze_blinks(video_path: str, fps: int = 25) -> BlinkResult:
    frames, times_s, vid_dur = load_video_frames(video_path, fps=fps)
    blink_times, ear_series, face_coverage = detect_blinks(frames, fps=fps)

    spans = suspicious_no_blink_spans(
        blink_times_s=blink_times,
        video_duration_s=vid_dur,
        face_coverage=face_coverage,
        max_gap_ok=16.0,
        min_video_len=12.0,
        require_coverage=0.60,
    )

    out_spans = [
        {"start": float(s), "end": float(e), "reason": reason}
        for (s, e, reason) in spans
    ]
    details = {
        "blink_times_s": [float(t) for t in blink_times],
        "video_duration_s": float(vid_dur),
        "fps": int(fps),
        "face_coverage": round(float(face_coverage), 3),
        "params": {
            "ear_thresh": 0.17,
            "min_closed_frames": 3,
            "max_gap_ok": 16.0,
            "min_video_len": 12.0,
            "require_coverage": 0.60,
        }
    }
    return BlinkResult(suspicious_spans=out_spans, details=details)

# -------------- CLI --------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Blink Agent (conservative)")
    p.add_argument("--in", dest="inp", required=True, help="Path to video (mp4)")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--json", action="store_true", help="Print JSON only")
    args = p.parse_args()

    result = analyze_blinks(args.inp, fps=args.fps)
    payload = {
        "agent": "blink",
        "metric": result.metric,
        "suspicious_spans": result.suspicious_spans,
        "details": result.details,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print("✅ Blink Agent")
        print(json.dumps(payload, indent=2))

