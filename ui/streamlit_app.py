import os, json, time, requests
import streamlit as st

# ---------- Config ----------
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")   # change if your FastAPI runs elsewhere
API_URL_DEFAULT = f"{API_BASE}/analyze"

st.set_page_config(page_title="VeriFace — Deepfake Intrusion Alarm", layout="centered")

# ---------- Light design ----------
st.markdown("""
<style>
/* nicer table + paddings */
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
.dataframe td, .dataframe th { font-size: 0.95rem; }
.badge { padding: .35rem .6rem; border-radius: .5rem; font-weight: 600; }
.badge-ok { background: #1b5e20; color: #fff; }
.badge-warn { background: #ff8f00; color: #111; }
.badge-err { background: #b71c1c; color: #fff; }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.title("🔎 VeriFace — Deepfake Intrusion Alarm")
st.caption("Upload a short clip. We flag where it looks fake and explain why.")

with st.sidebar:
    st.subheader("Settings")
    st.text_input("API endpoint", value=API_URL_DEFAULT, key="api_url")
    st.markdown("Detectors used: **LipSync, Blink, Voice**")

# ---------- Helpers ----------
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

def verdict_from_counts(flagged_detectors, detector_spans_total):
    # simple, judge-friendly rule
    if detector_spans_total == 0:
        return ("Likely real-ish", "No suspicious segments found by our lightweight checks.", "ok")
    if len(flagged_detectors) >= 2 or detector_spans_total >= 3:
        return ("⚠️ Likely deepfake", "Multiple detectors fired and several segments look suspicious.", "err")
    return ("🤔 Inconclusive", "Some signals look odd, but evidence is limited.", "warn")

# ---------- Upload ----------
uploaded = st.file_uploader(
    "Upload video/audio (mp4/mov/mkv/webm/wav/mp3)",
    type=["mp4","mov","mkv","webm","wav","mp3"]
)

if uploaded and st.button("Analyze"):
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

    # ---------- Extract ----------
    s = data.get("summary", {}) or {}
    timeline = data.get("timeline", []) or []
    pa = data.get("per_agent", {}) or {}

    # Compute counts directly from per-detector results so numbers match what you see
    detectors = ["lipsync", "blink", "voice"]
    flagged = []
    detector_spans_total = 0
    for d in detectors:
        spans = (pa.get(d, {}) or {}).get("spans", []) or []
        if len(spans) > 0:
            flagged.append(d)
        detector_spans_total += len(spans)

    # verdict
    ver_title, ver_reason, ver_kind = verdict_from_counts(flagged, detector_spans_total)
    if ver_kind == "ok":
        st.markdown(f'<span class="badge badge-ok">Verdict: {ver_title}</span> &nbsp; {ver_reason}', unsafe_allow_html=True)
    elif ver_kind == "warn":
        st.markdown(f'<span class="badge badge-warn">Verdict: {ver_title}</span> &nbsp; {ver_reason}', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge badge-err">Verdict: {ver_title}</span> &nbsp; {ver_reason}', unsafe_allow_html=True)

    # ---------- Metrics (clear & consistent) ----------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🎬 Clip length (s)", fmt_s(s.get("clip_seconds", 0), 1))
    with c2:
        st.metric("🧭 Detectors triggered", ", ".join(flagged) or "none")
    with c3:
        st.metric("🧩 Detector spans (total)", detector_spans_total)  # ← now shows 5 for 4 lipsync + 1 blink
    with c4:
        st.metric("🕒 Timeline windows", len(timeline))  # can differ if the backend merges spans

    # ---------- Why (top 3) ----------
    st.subheader("Why we think so")
    if detector_spans_total > 0:
        # build from detectors (more granular than timeline)
        reasons = []
        for d in detectors:
            for sp in (pa.get(d, {}) or {}).get("spans", []) or []:
                reasons.append((sp.get("start", 0), sp.get("end", 0), short_reason(sp.get("reason", ""))))
        reasons.sort(key=lambda x: x[0])
        for i, (stt, endt, why) in enumerate(reasons[:3], 1):
            st.write(f"**{i}. {fmt_s(stt)}s → {fmt_s(endt)}s** — {why}")
    else:
        st.write("No suspicious segments reported by detectors.")

    # ---------- Timeline (from backend merge) ----------
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
        st.caption("Note: Timeline windows may merge close spans for easier review.")
    else:
        st.success("✅ No suspicious spans detected.")

    # ---------- Per-detector (unique toggle keys; no nesting) ----------
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

            # FIX: unique key to avoid DuplicateWidgetID
            det = block.get("details", {})
            if det:
                show_details = st.toggle(f"Show technical details ({d})", value=False, key=f"details_{d}")
                if show_details:
                    st.json(det)

    # ---------- Export ----------
    st.download_button(
        "⬇️ Download JSON report",
        data=json.dumps(data, indent=2),
        file_name=f"veriface_report_{uploaded.name}.json",
        mime="application/json"
    )

# Legend
st.caption("Legend: 👄 Lip-sync • 👁 Blink • 🎤 Voice")

