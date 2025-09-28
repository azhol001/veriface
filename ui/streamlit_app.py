import os, json, time, requests
import streamlit as st

# ============ Config ============
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")   # change if your FastAPI runs elsewhere
API_URL_DEFAULT = f"{API_BASE}/analyze"

st.set_page_config(page_title="VeriFace — Deepfake Intrusion Alarm", layout="centered")

# ============ Header ============
st.title("🔎 VeriFace — Deepfake Intrusion Alarm")
st.caption("Upload a short clip. We flag where it looks fake and explain why.")

with st.sidebar:
    st.subheader("Settings")
    st.text_input("API endpoint", value=API_URL_DEFAULT, key="api_url")
    st.markdown("Detectors used: **LipSync, Blink, Voice**")

# ============ Helpers ============
def fmt_s(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return x

def short_reason(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    if "lip" in t:
        return "👄 Lip-sync issue — mouth doesn’t match audio"
    if "blink" in t:
        return "👁 Blink issue — unnatural or missing blinks"
    if "voice" in t or "audio" in t:
        return "🎤 Voice artifact — robotic/flat audio patterns"
    return text

def verdict_from_counts(flagged_detectors, detector_spans_total, total_sus_seconds):
    """
    Conservative rules (reduce false alarms):
    - Likely Real if no spans OR total suspicious time < 2.0s
    - Likely Deepfake if (>=2 detectors AND total suspicious time >= 4.0s) OR total suspicious time >= 8.0s
    - Otherwise Inconclusive
    """
    if detector_spans_total == 0 or total_sus_seconds < 2.0:
        return ("✅ Likely Real", "No meaningful suspicious activity.", "ok")
    if (len(flagged_detectors) >= 2 and total_sus_seconds >= 4.0) or total_sus_seconds >= 8.0:
        return ("❌ Likely Deepfake", "Multiple detectors and sustained suspicious activity.", "err")
    return ("🤔 Inconclusive", "Some signals look odd, but not enough evidence.", "warn")

# ============ Upload ============
uploaded = st.file_uploader(
    "Upload video/audio (mp4/mov/mkv/webm/wav/mp3)",
    type=["mp4","mov","mkv","webm","wav","mp3"]
)

if uploaded and st.button("Analyze"):
    # ---- Call backend API ----
    with st.spinner("Analyzing…"):
        files = {"file": (uploaded.name, uploaded.getbuffer(), uploaded.type or "application/octet-stream")}
        t0 = time.time()
        try:
            r = requests.post(st.session_state.api_url, files=files, timeout=300)
        except Exception as e:
            st.error(f"Could not reach API: {e}")
            st.stop()
        took = time.time() - t0

    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {getattr(r, 'text', r)}")
        st.stop()

    data = r.json()
    st.success(f"Done in {took:.1f}s")

    # ---- Extract data ----
    s = data.get("summary", {}) or {}
    timeline = data.get("timeline", []) or []
    pa = data.get("per_agent", {}) or {}
    detectors = ["lipsync", "blink", "voice"]

    # ---- Filter tiny blips + compute total suspicious time ----
    MIN_SPAN = 0.75  # ignore very short flickers to reduce false positives
    total_sus_seconds = 0.0
    filtered_per_agent = {}
    for d in detectors:
        raw_spans = (pa.get(d, {}) or {}).get("spans", []) or []
        good = []
        for sp in raw_spans:
            start = float(sp.get("start", 0) or 0)
            end = float(sp.get("end", 0) or 0)
            if end - start >= MIN_SPAN:
                good.append(sp)
                total_sus_seconds += (end - start)
        if good:
            filtered_per_agent[d] = {**(pa.get(d, {}) or {}), "spans": good}

    # ---- Recompute counts using filtered spans ----
    flagged = [d for d in detectors if len((filtered_per_agent.get(d, {}) or {}).get("spans", [])) > 0]
    detector_spans_total = sum(len((filtered_per_agent.get(d, {}) or {}).get("spans", [])) for d in detectors)
    pa = filtered_per_agent  # use filtered data from now on

    # ---- Verdict ----
    ver_title, ver_reason, ver_kind = verdict_from_counts(flagged, detector_spans_total, total_sus_seconds)
    if ver_kind == "ok":
        st.success(f"Verdict: {ver_title} — {ver_reason}")
    elif ver_kind == "warn":
        st.warning(f"Verdict: {ver_title} — {ver_reason}")
    else:
        st.error(f"Verdict: {ver_title} — {ver_reason}")

    # ---- Metrics ----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🎬 Clip length (s)", fmt_s(s.get("clip_seconds", 0), 1))
    with c2:
        st.metric("🧭 Detectors triggered", ", ".join(flagged) or "none")
    with c3:
        st.metric("🧩 Detector spans (total)", detector_spans_total)
    with c4:
        st.metric("⏱ Total suspicious time (s)", f"{total_sus_seconds:.1f}")

    # ---- Why (top 3 reasons from detector spans) ----
    st.subheader("Why we think so")
    if detector_spans_total > 0:
        reasons = []
        for d in detectors:
            for sp in (pa.get(d, {}) or {}).get("spans", []) or []:
                reasons.append((sp.get("start", 0), sp.get("end", 0), short_reason(sp.get("reason", ""))))
        reasons.sort(key=lambda x: x[0])
        for i, (stt, endt, why) in enumerate(reasons[:3], 1):
            st.write(f"**{i}. {fmt_s(stt)}s → {fmt_s(endt)}s** — {why}")
    else:
        st.write("No suspicious segments reported by detectors.")

    # ---- Timeline (from backend merge) ----
    st.subheader("Suspicious Timeline")
    if timeline:
        rows = []
        for i, span in enumerate(timeline, 1):
            rows.append({
                "#": i,
                "Start (s)": fmt_s(span.get("start", 0)),
                "End (s)": fmt_s(span.get("end", 0)),
                "Why": short_reason(span.get("reason", "")),
            })
        st.dataframe(rows, use_container_width=True)
        st.caption("Note: Timeline windows may merge nearby spans for easier review.")
    else:
        st.success("✅ No suspicious spans detected.")

    # ---- Per-detector (unique keys; no nested expanders) ----
    st.subheader("Per-Detector Results")
    for d in detectors:
        block = pa.get(d, {}) or {}
        spans = block.get("spans", []) or []
        with st.expander(f"{d.title()} — {len(spans)} span(s)"):
            if spans:
                nice = []
                for sp in spans:
                    nice.append({
                        "Start (s)": fmt_s(sp.get("start", 0)),
                        "End (s)": fmt_s(sp.get("end", 0)),
                        "Why": short_reason(sp.get("reason", "")),
                    })
                st.dataframe(nice, use_container_width=True)
            else:
                st.write("Nothing suspicious here. ✅")

            det = block.get("details", {})
            if det:
                show_details = st.toggle(f"Show technical details ({d})", value=False, key=f"details_{d}")
                if show_details:
                    st.json(det)

    # ---- Export ----
    st.download_button(
        "⬇️ Download JSON report",
        data=json.dumps(data, indent=2),
        file_name=f"veriface_report_{uploaded.name}.json",
        mime="application/json"
    )

# Legend
st.caption("Legend: 👄 Lip-sync • 👁 Blink • 🎤 Voice")



