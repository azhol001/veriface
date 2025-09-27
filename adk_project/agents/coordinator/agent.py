# adk_project/agents/coordinator/agent.py
# Local coordinator (non-ADK) to run 3 agents and fuse their spans.

from __future__ import annotations
import json
from typing import List, Dict, Tuple

def _merge_spans(spans: List[Dict], join_tol: float = 0.5) -> List[Dict]:
    """
    Input items: {"start": float, "end": float, "reason": str, "source": str}
    Merge overlaps and near-adjacent spans (<= join_tol) and concatenate reasons.
    """
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x["start"], x["end"]))
    out = []
    cur = dict(spans[0])
    cur_reasons = [f'[{cur.get("source","?")}] {cur["reason"]}']

    for s in spans[1:]:
        if s["start"] <= cur["end"] + join_tol:  # overlap or near
            cur["end"] = max(cur["end"], s["end"])
            cur_reasons.append(f'[{s.get("source","?")}] {s["reason"]}')
        else:
            cur["reason"] = " | ".join(cur_reasons)
            out.append(cur)
            cur = dict(s)
            cur_reasons = [f'[{cur.get("source","?")}] {cur["reason"]}']
    cur["reason"] = " | ".join(cur_reasons)
    out.append(cur)
    return out


def run_all(video_path: str, fps: int = 25) -> Dict:
    # Import here to avoid circular deps
    from adk_project.agents.blink_agent.agent import analyze_blinks
    from adk_project.agents.voice_agent.agent import analyze_voice
    from adk_project.agents.lipsync_agent.agent import analyze_lipsync

    # Run sub-analyzers (sequential MVP; we’ll swap to ADK ParallelAgent later)
    blink = analyze_blinks(video_path, fps=fps)
    voice = analyze_voice(video_path)
    lips  = analyze_lipsync(video_path, fps=fps)

    # Tag sources
    spans_all = []
    for sp in blink.suspicious_spans:
        spans_all.append({**sp, "source": "blink"})
    for sp in voice.suspicious_spans:
        spans_all.append({**sp, "source": "voice"})
    for sp in lips.suspicious_spans:
        spans_all.append({**sp, "source": "lipsync"})

    fused = _merge_spans(spans_all, join_tol=0.5)

    # Coordinator output schema
    out = {
        "project": "VeriFace DIA",
        "summary": {
            "clip_seconds": lips.details.get("video_duration_s") if lips.details else None,
            "sources_flagged": sorted(list({s["source"] for s in spans_all})) if spans_all else [],
            "total_spans": len(fused),
        },
        "timeline": fused,  # list of {start, end, reason}
        "per_agent": {
            "blink": {
                "metric": "blink_no_gap",
                "spans": blink.suspicious_spans,
                "details": blink.details,
            },
            "voice": {
                "metric": "voice_anomaly_mfcc",
                "spans": voice.suspicious_spans,
                "details": voice.details,
            },
            "lipsync": {
                "metric": "lipsync_corr_lag",
                "spans": lips.suspicious_spans,
                "details": lips.details,
            },
        },
    }
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="VeriFace Coordinator (local MVP)")
    p.add_argument("--in", dest="inp", required=True, help="Path to video (mp4)")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    result = run_all(args.inp, fps=args.fps)
    if args.json:
        print(json.dumps(result))
    else:
        print("✅ VeriFace Coordinator (local)")
        print(json.dumps(result, indent=2))
