import os, json, time, requests
import streamlit as st

# ---------- Config ----------
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")  # change if your FastAPI runs elsewhere
API_URL = f"{API_BASE}/analyze"

st.set_page_config(page_title="VeriFace — Deepfake Intrusion Alarm", layout="centered")

# ---------- Header ----------
st.title("🔍 VeriFace — Deepfake Intrusion Alarm")
st.caption("Upload a short clip. We flag suspicious moments and explain why.")

with st.sidebar:
    st.subheader("Settings")
    st.text_input("API endpoint", API_URL, key="api_url")
    st.markdown("**Detectors shown:** LipSync, Blink, Voice")

# ---------- Helper(s) ----------
def short_reason(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    if "lip" in t:
        return "👄 Lip-sync issue (mouth ↔ audio mismatch)"
    if "blink" in t:
        return "👁 Unnatural/missing blinks"
    if "voice" in t or "audio" in t:
        return "🎤 Voice artifact (synthetic/flat/unstable)"
    return text

def fmt_s(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return x

# ---------- Uploader ----------
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

    # ---------- Summary ----------
    st.subheader("Summary")
    s = data.get("summary", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎬 Clip length (s)", fmt_s(s.get("clip_seconds", 0), 1))
    with col2:
        st.metric("🧭 Detectors triggered", ", ".join(s.get("sources_flagged", [])) or "none")
    with col3:
        st.metric("🚨 Suspicious spans", s.get("total_spans", 0))

    # ---------- Timeline ----------
    st.subheader("🚦 Suspicious Timeline")
    timeline = data.get("timeline", [])
    if timeline:
        # Build a small, readable table
        rows = []
        for i, span in enumerate(timeline, 1):
            rows.append({
                "#": i,
                "Start (s)": fmt_s(span.get("start", 0)),
                "End (s)": fmt_s(span.get("end", 0)),
                "Why": short_reason(span.get("reason", "")),
            })
        st.dataframe(rows, use_container_width=True)
        st.caption("Tip: Judges can skim this table to see exactly where to jump.")
    else:
        st.success("✅ No suspicious spans detected.")

    # ---------- Per-Detector ----------
    st.subheader("Per-Detector Results")
    pa = data.get("per_agent", {})
    for detector in ["lipsync", "blink", "voice"]:
        block = pa.get(detector, {}) or {}
        spans = block.get("spans", []) or []
        with st.expander(f"{detector.title()} — {len(spans)} span(s)"):
            if spans:
                nice = []
                for sspan in spans:
                    nice.append({
                        "Start (s)": fmt_s(sspan.get("start", 0)),
                        "End (s)": fmt_s(sspan.get("end", 0)),
                        "Why": short_reason(sspan.get("reason", "")),
                    })
                st.dataframe(nice, use_container_width=True)
            else:
                st.write("Nothing suspicious here. ✅")
            # Optional raw details for debugging/judging
            det = block.get("details", {})
            if det:
                with st.expander("Show technical details (optional)"):
                    st.json(det)

    # ---------- Export ----------
    st.download_button(
        "⬇️ Download JSON report",
        data=json.dumps(data, indent=2),
        file_name=f"veriface_report_{uploaded.name}.json",
        mime="application/json"
    )

# Always show a small legend for judges
st.caption("Legend: 👄 Lip-sync • 👁 Blink • 🎤 Voice")

