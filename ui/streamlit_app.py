import os, json, time, requests
import streamlit as st

# ------------ Config ------------
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")   # change if your FastAPI runs elsewhere
API_URL_DEFAULT = f"{API_BASE}/analyze"

st.set_page_config(page_title="VeriFace — Deepfake Intrusion Alarm", layout="centered")

# ------------ Header ------------
st.title("🔎 VeriFace — Deepfake Intrusion Alarm")
st.caption("Upload a short clip. We flag where it looks fake and explain why.")

with st.sidebar:
    st.subheader("Settings")
    st.text_input("API endpoint", value=API_URL_DEFAULT, key="api_url")
    st.markdown("Detectors used: **LipSync, Blink, Voice**")

# ------------ Helpers ------------
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

def overall_verdict(summary, timeline):
    spans = int(summary.get("total_spans", 0) or 0)
    flagged = set(summary.get("sources_flagged", []))
    if spans == 0:
        return ("Likely real-ish", "No suspicious segments found by our lightweight checks.", "success")
    if len(flagged) >= 2 or spans >= 3:
        return ("⚠️ Likely deepfake", "Multiple detectors fired and several segments look suspicious.", "error")
    return ("🤔 Inconclusive", "Some signals look odd, but evidence is limited.", "warning")

# ------------ UI: Upload ------------
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

    # ---------- Summary + Verdict ----------
    s = data.get("summary", {})
    timeline = data.get("timeline", [])

    verdict_title, verdict_reason, verdict_kind = overall_verdict(s, timeline)
    if verdict_kind == "success":
        st.success(f"Verdict: {verdict_title} — {verdict_reason}")
    elif verdict_kind == "warning":
        st.warning(f"Verdict: {verdict_title} — {verdict_reason}")
    else:
        st.error(f"Verdict: {verdict_title} — {verdict_reason}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎬 Clip length (s)", fmt_s(s.get("clip_seconds", 0), 1))
    with col2:
        flags = ", ".join(s.get("sources_flagged", [])) or "none"
        st.metric("🧭 Detectors triggered", flags)
    with col3:
        st.metric("🚨 Suspicious spans", s.get("total_spans", 0))

    # ---------- Human reasons (top 3) ----------
    st.subheader("Why we think so")
    if timeline:
        top3 = timeline[:3]
        for i, span in enumerate(top3, 1):
            st.write(f"**{i}. {fmt_s(span.get('start',0))}s → {fmt_s(span.get('end',0))}s** — {short_reason(span.get('reason',''))}")
    else:
        st.write("No suspicious segments reported by detectors.")

    # ---------- Timeline table ----------
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
        st.caption("Tip: Click a row to discuss that spot; judges can skim quickly.")
    else:
        st.success("✅ No suspicious spans detected.")

    # ---------- Per-detector (no nested expanders) ----------
    st.subheader("Per-Detector Results")
    pa = data.get("per_agent", {})
    for detector in ["lipsync", "blink", "voice"]:
        block = pa.get(detector, {}) or {}
        spans = block.get("spans", []) or []
        exp = st.expander(f"{detector.title()} — {len(spans)} span(s)")
        with exp:
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

            # Show raw technical details under a *separate* toggle (not an expander inside an expander)
            det = block.get("details", {})
            if det:
                show_details = st.toggle("Show technical details", value=False)
                if show_details:
                    st.json(det)

    # ---------- Export ----------
    st.download_button(
        "⬇️ Download JSON report",
        data=json.dumps(data, indent=2),
        file_name=f"veriface_report_{uploaded.name}.json",
        mime="application/json"
    )

# Always-on legend
st.caption("Legend: 👄 Lip-sync • 👁 Blink • 🎤 Voice")


