import os, json, requests, time
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8080")

st.set_page_config(page_title="VeriFace — Deepfake Intrusion Alarm", layout="centered")
st.title("VeriFace — Deepfake Intrusion Alarm (DIA)")
st.caption("Upload a short clip. Three detectors run; coordinator fuses results into a timeline.")

api_url = f"{API_BASE}/analyze"

with st.sidebar:
    st.markdown("### Settings")
    st.write("API:", api_url)
    fps = st.slider("Frame rate (for analysis)", 20, 30, 25)

uploaded = st.file_uploader("Upload video/audio (mp4/mov/mkv/webm/wav/mp3)", type=["mp4","mov","mkv","webm","wav","mp3"])

if uploaded and st.button("Analyze"):
    with st.spinner("Analyzing…"):
        files = {"file": (uploaded.name, uploaded.getbuffer(), uploaded.type or "application/octet-stream")}
        t0 = time.time()
        try:
            r = requests.post(api_url, files=files, timeout=300)
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

        dt = time.time() - t0
        if r.status_code != 200:
            st.error(f"API error {r.status_code}: {getattr(r, 'text', r)}")
            st.stop()

        data = r.json()
        st.success(f"Done in {dt:.1f}s")

        # Summary
        st.subheader("Summary")
        s = data.get("summary", {})
        st.write({
            "clip_seconds": s.get("clip_seconds"),
            "sources_flagged": s.get("sources_flagged"),
            "total_spans": s.get("total_spans"),
        })

        # Timeline (simple table)
        st.subheader("Suspicious Timeline")
        timeline = data.get("timeline", [])
        if timeline:
            for i, span in enumerate(timeline, 1):
                st.write(f"**{i}. {span['start']:.2f}s → {span['end']:.2f}s**")
                st.write(span.get("reason", ""))
                st.divider()
        else:
            st.info("No suspicious spans detected.")

        # Per-agent details
        st.subheader("Per-Agent")
        pa = data.get("per_agent", {})
        for name in ["lipsync", "blink", "voice"]:
            block = pa.get(name, {})
            st.markdown(f"**{name}** — _{block.get('metric','')}_")
            spans = block.get("spans", [])
            st.write(f"Spans: {len(spans)}")
            if spans:
                st.json(spans)
            det = block.get("details", {})
            if det:
                with st.expander("Details"):
                    st.json(det)
