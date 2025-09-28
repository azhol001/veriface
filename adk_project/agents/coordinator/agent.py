# adk_project/agents/coordinator/agent.py
# Coordinator: run 3 agents, filter tiny blips, conservatively merge, and summarize.

from __future__ import annotations
from typing import List, Dict, Tuple
import json
import math

from adk_project.agents.blink_agent.agent import analyze_blinks
from adk_project.agents.voice_agent.agent import analyze_voice
from adk_project.agents.lipsync_agent.agent import analyze_lipsync

try:
    from adk_project.tools.media_io import probe_media_meta  # optional
except Exception:
    probe_media_meta = None

COORD_VERSION = "coordinator/1.2.0"

# --------------------------- helpers ---------------------------

def _f(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)

def _span_len(s: Dict) -> float:
    try:
        return max(0.0, _f(s.get("end")) - _f(s.get("start")))
    except Exception:
        return 0.0

def _filter_short(spans: List[Dict], min_len: float = 0.75) -> List[Dict]:
    """Drop micro-blips that are usually noise."""
    out = []
    for s in (spans or []):
        L = _span_len(s)
        if L >= min_len:
            out.append({"start": _f(s.get("start")), "end": _f(s.get("end")), "reason": str(s.get("reason", ""))})
    return out

def _merge_spans(spans: List[Dict], join_tol: float = 0.15) -> List[Dict]:
    """
    Input items: {"start": float, "end": float, "reason": str, "source": str}
    Merge overlaps and very-near spans (<= join_tol). Keep concise reasons.
    """
    if not spans:
        return []
    spans = sorted(
        [{"start": _f(s["start"]), "end": _f(s["end"]), "reason": str(s.get("reason","")), "source": str(s.get("source","?"))}
         for s in spans],
        key=lambda x: (x["start"], x["end"])
    )
    merged: List[Dict] = []
    cur = dict(spans[0])
    reasons = {f'[{cur["source"]}] {cur["reason"]}'.strip()}

    for s in spans[1:]:
        if s["start"] <= cur["end"] + join_tol:
            cur["end"] = max(cur["end"], s["end"])
            reasons.add(f'[{s["source"]}] {s["reason"]}'.strip())
        else:
            rl = sorted(r for r in reasons if r)
            cur["reason"] = " | ".join(rl[:3]) + ("" if len(rl) <= 3 else " | …")
            merged.append(cur)
            cur = dict(s)
            reasons = {f'[{cur["source"]}] {cur["reason"]}'.strip()}
    rl = sorted(r for r in reasons if r)
    cur["reason"] = " | ".join(rl[:3]) + ("" if len(rl) <= 3 else " | …")
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
    bl = analyze_blinks(video_path, fps=fps, use_stream=True, frame_stride=1)
    vo = analyze_voice(video_path, fps=fps)
    ls = analyze_lipsync(video_path, fps=fps)

    # Raw spans from each detector (then filter out micro-blips)
    per_agent = {
        "lipsync": {
            "metric": ls.metric,
            "spans": _filter_short(ls.suspicious_spans, min_len=0.75),
            "details": ls.details or {},
        },
        "blink": {
            "metric": bl.metric,
            "spans": _filter_short(bl.suspicious_spans, min_len=0.75),
            "details": bl.details or {},
        },
        "voice": {
            "metric": vo.metric,
            "spans": _filter_short(vo.suspicious_spans, min_len=0.75),
            "details": vo.details or {},
        },
    }

    # Insufficient-source flags the UI can show
    insufficient_sources = []
    insufficient_notes = []
    bdet = per_agent["blink"]["details"] or {}
    if isinstance(bdet.get("description", ""), str) and "insufficient" in bdet["description"]:
        insufficient_sources.append("blink")
        insufficient_notes.append(f'blink: {bdet["description"]}')
    ldet = per_agent["lipsync"]["details"] or {}
    if _f(ldet.get("face_coverage"), 1.0) < 0.60:
        insufficient_sources.append("lipsync")
        insufficient_notes.append(f'lipsync: insufficient face coverage ({_f(ldet.get("face_coverage")):.2f})')

    # Build flat list with sources for merging
    spans_all: List[Dict] = []
    for src in ("blink", "voice", "lipsync"):  # blink first to keep large gaps dominant
        for sp in (per_agent[src]["spans"] or []):
            spans_all.append({**sp, "source": src})

    # Conservative merge
    fused = _merge_spans(spans_all, join_tol=0.15)

    # Totals
    raw_count, raw_secs = _sum_raw(per_agent)
    timeline_secs = round(sum(_span_len(s) for s in fused), 2)

    # Media meta / duration
    clip_seconds = None
    meta = {}
    if probe_media_meta:
        try:
            meta = probe_media_meta(video_path) or {}
            clip_seconds = _f(meta.get("duration"))
        except Exception:
            meta = {}
    if not clip_seconds:
        for det in (ls.details, bl.details, vo.details):
            try:
                clip_seconds = _f((det or {}).get("video_duration_s"))
                if clip_seconds:
                    break
            except Exception:
                continue
        clip_seconds = _f(clip_seconds or 0.0)

    sources_flagged = [k for k, v in per_agent.items() if len((v.get("spans") or [])) > 0]
    timeline_ratio = (timeline_secs / clip_seconds) if clip_seconds else 0.0

    def _verdict(sources: List[str], t_secs: float, clip_secs: float, insufficient: List[str]) -> str:
        if insufficient and not sources:
            return "⚠️ Inconclusive — Insufficient evidence (input quality too low)."
        if not sources and t_secs < max(2.0, 0.02 * clip_secs):
            return "✅ Likely Genuine — No sustained anomalies."
        if len(sources) == 1 and t_secs < max(3.0, 0.08 * clip_seconds):
            return "⚠️ Needs Review — Some signals, not decisive."
        if len(sources) >= 2 and t_secs >= max(4.0, 0.15 * clip_secs):
            return "❌ Likely Deepfake — Multiple detectors and sustained suspicious activity."
        return "⚠️ Needs Review — Some signals, not decisive."

    summary = {
        "clip_seconds": clip_seconds,
        "sources_flagged": sources_flagged,
        "insufficient_sources": insufficient_sources,
        "insufficient_notes": insufficient_notes,
        "total_spans": len(fused),
        "detector_spans_total": raw_count,
        "total_suspicious_seconds": timeline_secs,
        "raw_total_seconds": raw_secs,
        "timeline_ratio": round(timeline_ratio, 3),
        "params": {"min_span": 0.75, "join_tol": 0.15},
        "verdict": _verdict(sources_flagged, timeline_secs, clip_seconds, insufficient_sources),
        "version": COORD_VERSION,
    }

    return {
        "project": "VeriFace DIA",
        "summary": summary,
        "timeline": fused,
        "per_agent": per_agent,
        "consistency": {
            "raw_span_count": raw_count,
            "raw_total_seconds": raw_secs,
            "timeline_windows": len(fused),
            "timeline_total_seconds": timeline_secs,
            "note": "Timeline merges overlapping/nearby spans using join_tol=0.15; therefore timeline_windows can be < raw_span_count.",
        },
        "meta": meta,
    }

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="VeriFace Coordinator (conservative)")
    p.add_argument("--in", dest="inp", required=True, help="Path to video (mp4)")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    result = run_all(args.inp, fps=args.fps)
    print(json.dumps(result) if args.json else json.dumps(result, indent=2))
