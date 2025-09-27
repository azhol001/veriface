# adk_project/agents/voice_agent/agent.py
# Voice Agent (MVP): windowed MFCC features + IsolationForest anomaly score.
# Outputs merged suspicious spans with reasons.

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
import librosa

from adk_project.tools.media_io import load_audio_mono


def frame_windows(y: np.ndarray, sr: int, win_s: float = 0.5, hop_s: float = 0.25) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[float, float]]]:
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    idx = []
    times = []
    feats = []

    for start in range(0, max(1, len(y) - win + 1), hop):
        end = start + win
        seg = y[start:end]
        if len(seg) < win:
            # pad last frame
            seg = np.pad(seg, (0, win - len(seg)))
        # 13 MFCC + deltas (39 dims)
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        vec = np.hstack([mfcc.mean(axis=1), d1.mean(axis=1), d2.mean(axis=1)])  # (39,)
        feats.append(vec)
        idx.append((start, end))
        times.append((start / sr, end / sr))
    X = np.vstack(feats) if feats else np.zeros((0, 39), dtype=np.float32)
    return X.astype(np.float32), idx, times


def to_spans(flags: np.ndarray, times: List[Tuple[float, float]], min_span_s: float = 0.75) -> List[Tuple[float, float]]:
    # merge consecutive flagged windows
    spans = []
    cur = None
    for f, (s, e) in zip(flags, times):
        if f:
            if cur is None:
                cur = [s, e]
            else:
                cur[1] = e
        else:
            if cur is not None:
                spans.append(tuple(cur))
                cur = None
    if cur is not None:
        spans.append(tuple(cur))

    # drop very short spans
    spans = [sp for sp in spans if (sp[1] - sp[0]) >= min_span_s]
    return spans


@dataclass
class VoiceResult:
    suspicious_spans: List[dict]
    metric: str = "voice_anomaly_mfcc"
    details: dict = None


def analyze_voice(path: str, target_sr: int = 16000, win_s: float = 0.5, hop_s: float = 0.25, contamination: float = 0.12) -> VoiceResult:
    y, sr, dur = load_audio_mono(path, target_sr=target_sr)
    # features over windows
    X, _, times = frame_windows(y, sr, win_s=win_s, hop_s=hop_s)
    if len(X) == 0:
        return VoiceResult(suspicious_spans=[], details={"reason": "no audio frames"})

    # robust scaling
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    Xz = (X - mu) / sd

    # IsolationForest to flag anomalies
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iso.fit(Xz)
    pred = iso.predict(Xz)   # -1 = anomaly, 1 = normal
    flags = (pred == -1)

    spans = to_spans(flags, times, min_span_s=max(win_s, 0.75))

    out_spans = [
        {"start": float(s), "end": float(e), "reason": "Audio anomaly (MFCC/IForest)"}
        for (s, e) in spans
    ]
    details = {
        "sr": int(sr),
        "duration_s": float(dur),
        "window_s": float(win_s),
        "hop_s": float(hop_s),
        "anomalous_windows": int(flags.sum()),
        "total_windows": int(len(flags)),
        "contamination": float(contamination),
    }
    return VoiceResult(suspicious_spans=out_spans, details=details)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Voice Agent MVP")
    p.add_argument("--in", dest="inp", required=True, help="Path to video/audio (mp4/wav)")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--win", type=float, default=0.5)
    p.add_argument("--hop", type=float, default=0.25)
    p.add_argument("--contam", type=float, default=0.12)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    res = analyze_voice(args.inp, target_sr=args.sr, win_s=args.win, hop_s=args.hop, contamination=args.contam)
    payload = {
        "agent": "voice",
        "metric": res.metric,
        "suspicious_spans": res.suspicious_spans,
        "details": res.details,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print("✅ Voice Agent")
        print(json.dumps(payload, indent=2))
