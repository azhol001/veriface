# adk_project/agents/coordinator/agent.py
# Coordinator: run 3 agents, filter tiny blips, conservatively merge, and summarize.

from __future__ import annotations
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Import agents locally to avoid circular deps at module import time
from adk_project.agents.blink_agent.agent import analyze_blinks
from adk_project.agents.voice_agent.agent import analyze_voice
from adk_project.agents.lipsync_agent.agent import analyze_lipsync

# If you have media probing helpers, you can import; otherwise we’ll infer from agent details.
try:
    from adk_project.tools.media_io import probe_media_meta  # optional
except Exception:
    probe_media_meta = None

# --------------------------- helpers ---------------------------

def _span_len(s: Dict) -> float:
    try:
        return max(0.0, float(s.get("end", 0)) - float(s.get("start", 0)))
    except Exception:
        return 0.0

def _filter_short(spans: List[Dict], min_len: float = 0.75) -> List[Dict]:
    """Drop micro-blips that are usually noise."""
    return [s for s in (spans or []) if _span_len(s) >= min_len]

def _merge_spans(spans: List[Dict], join_tol: float = 0.15) -> List[Dict]:
    """
    Input items: {"start": float, "end": float, "reason": str, "source": str}
    Merge overlaps and very-near spans (<= join_tol). Keep reasons short (dedup).
    """
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (float(x["start"]), float(x["end"])))
    merged: List[Dict] = []
    cur = dict(spans[0])
    cur_reasons = {f'[{cur.get("source","?")}] {cur.get("reason","")}'.strip()}

    for s in spans[1:]:
        if float(s["start"]) <= float(cur["end"]) + join_tol:  # overlap or near
            cur["end"] = max(float(cur["end"]), float(s["end"]))
            cur_reasons.add(f'[{s.get("source","?")}] {s.get("reason","")}'.strip())
        else:
            # finalize current
            reasons_list = sorted(r for r in cur_reasons if r)
            # cap to keep human-readable
            cur["reason"] = " | ".join(reasons_list[:3]) + ("" if len(reasons_list) <= 3 else " | …")
            merged.append(cur)
            # reset
            cur = dict(s)
            cur_reasons = {f'[{cur.get("source","?")}] {cur.get("reason","")}'.strip()}

    # finalize tail
    reasons_list = sorted(r for r in cur_reasons if r)
    cur["reason"] = " | ".join(reasons_list[:3]) + ("" if len(reasons_list) <= 3 else " | …")
    merged.append(cur)
    return merged

def _sum_raw(per_agent: Dict[str, Dict]) -> Tuple[int, float]:
    """Sum raw (filtered) per-detector spans and seconds."""
    count = 0
    secs = 0.0
    for _, block in (per_agent or {}).items():
        spans = block.get("spans", []) or []
        count += len(spans)
        for s in spans:
            secs += _span_len(s)
    return count, round(secs, 2)

# --------------------------- main ---------------------------

def run_all(video_path: str, fps: int = 25) -> Dict:
    """
    Runs lipsync, blink, voice agents; filters tiny spans; merges conservatively;
    returns judge-friendly summary + timeline + per-agent details.
    """

    # Run sub-analyzers (sequential MVP)
    bl = analyze_blinks(video_path, fps=fps)
    vo = analyze_voice(video_path, fps=fps)
    ls = analyze_lipsync(video_path, fps=fps)

    # Raw spans from each detector (then filter out micro-blips)
    per_agent = {
        "lipsync": {
            "metric": ls.metric,
            "spans": _filter_short(ls.suspicious_spans, min_len=0.75),
            "details": ls.details,
        },
        "blink": {
            "metric": bl.metric,
            "spans": _filter_short(bl.suspicious_spans, min_len=0.75),
            "details": bl.details,
        },
        "voice": {
            "metric": vo.metric,
            "spans": _filter_short(vo.suspicious_spans, min_len=0.75),
            "details": vo.details,
        },
    }

    # Build flat list with sources for merging
    spans_all: List[Dict] = []
    for src in ("blink", "voice", "lipsync"):
        for sp in (per_agent[src]["spans"] or []):
            spans_all.append({**sp, "source": src})

    # Conservative merge: join only overlaps / very-close spans
    fused = _merge_spans(spans_all, join_tol=0.15)

    # Totals
    raw_count, raw_secs = _sum_raw(per_agent)
    timeline_secs = round(sum(_span_len(s) for s in fused), 2)

    # Media meta / duration
    clip_seconds = None
    if probe_media_meta:
        try:
            meta = probe_media_meta(video_path) or {}
            clip_seconds = float(meta.get("duration") or 0.0)
        except Exception:
            meta = {}
    else:
        meta = {}
    # fallback: try from agent detail (lipsync or blink)
    if not clip_seconds:
        for det in (ls.details, bl.details, vo.details):
            try:
                clip_seconds = float(det.get("video_duration_s"))
                break
            except Exception:
                continue
        clip_seconds = float(clip_seconds or 0.0)

    sources_flagged = [k for k, v in per_agent.items() if len(v.get("spans", []) or []) > 0]

    summary = {
        "clip_seconds": clip_seconds,
        "sources_flagged": sources_flagged,
        "total_spans": len(fused),                 # windows after merge (judge-facing)
        "detector_spans_total": raw_count,         # raw sum across detectors
        "total_suspicious_seconds": raw_secs,      # raw seconds (before merge)
        "params": {"min_span": 0.75, "join_tol": 0.15},
    }

    return {
        "project": "VeriFace DIA",
        "summary": summary,
        "timeline": fused,                         # [{start,end,reason}]
        "per_agent": per_agent,                    # raw filtered spans by detector
        "consistency": {
            "raw_span_count": raw_count,
            "raw_total_seconds": raw_secs,
            "timeline_windows": len(fused),
            "timeline_total_seconds": timeline_secs,
            "note": "Timeline merges overlapping/nearby spans using join_tol=0.15; "
                    "therefore timeline_windows can be < raw_span_count.",
        },
        "meta": meta,
    }

# --------------------------- CLI ---------------------------

if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(description="VeriFace Coordinator (conservative)")
    p.add_argument("--in", dest="inp", required=True, help="Path to video (mp4)")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    result = run_all(args.inp, fps=args.fps)
    if args.json:
        print(json.dumps(result))
    else:
        print("✅ VeriFace Coordinator")
        print(json.dumps(result, indent=2))
