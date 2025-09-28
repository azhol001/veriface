import os, json, requests, time
import streamlit as st

# ----------------- App config -----------------
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
API_URL = f"{API_BASE}/analyze"

st.set_page_config(page_title="VeriFace — Deepfake Intrusion Alarm", layout="wide")

# ----------------- CSS theme (dark gradient + cards) -----------------
st.markdown("""
<style>
/* page background gradient */
.stApp {
  background: radial-gradient(1200px 600px at 85% 20%, rgba(92,61,255,0.25), rgba(0,0,0,0)) ,
              radial-gradient(900px 500px at 10% 10%, rgba(0,135,255,0.18), rgba(0,0,0,0)) ,
              #0B1020;
  color: #E6E8F0;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.block-container{ padding-top: 2rem; max-width: 1200px; }

/* Hero */
.hero-title{ font-size: 3rem; font-weight: 800; line-height: 1.1; margin: .25rem 0 0.25rem 0; }
.hero-sub{ color:#B6BCD1; font-size:1.05rem; margin-bottom: 1.25rem; }
.logo{ font-weight:700; font-size: 1.1rem; letter-spacing: .3px; }

/* Buttons */
.btn-primary button{
  background: linear-gradient(90deg,#7C5CFF,#6E40FF);
  border: 0; color: white; font-weight: 600;
}
.btn-primary button:hover{ filter: brightness(1.08); }

/* Cards */
.card{ border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 16px 16px; background: rgba(255,255,255,0.03); }
.card h4{ margin: 0 0 6px 0; }
.card p{ color: #ADB4C8; margin: 0; font-size: 0.95rem; }

/* Table compact */
.small-table table{ font-size: 0.95rem; }

/* badge */
.badge{ display:inline-block; padding:.25rem .5rem; border-radius:999px; background:#1D2338; border:1px solid rgba(255,255,255,.08); color:#BFC6DB; font-size:.85rem; }

/* top nav (right) */
.nav a{ color:#C9D0E3; text-decoration:none; margin-left: 18px; }
.nav a:hover{ color:white; text-decoration:underline; }
</style>
""", unsafe_allow_html=True)

# ----------------- Top bar -----------------
left, right = st.columns([1,1], gap="large")
with left:
    st.markdown("<div class='logo'>🔎 VeriFace</div>", unsafe_allow_html=True)
with right:
    # Use page links if available (Streamlit 1.25+), else simple links
    try:
        col1, col2, col3 = st.columns([1,1,1])
        with col1:  st.page_link("ui/streamlit_app.py", label="Home", icon="🏠")
        with col2:  st.page_link("pages/2_FAQ.py", label="FAQ", icon="❓")
        with col3:  st.page_link("pages/1_About_Us.py", label="About", icon="ℹ️")
        st.write("")  # spacer
    except Exception:
        st.markdown("<div class='nav' style='text-align:right;'>"
                    "<a href='#'>Home</a>"
                    " · <a>FAQ</a>"
                    " · <a>About</a>"
                    "</div>", unsafe_allow_html=True)

st.markdown("---")

# ----------------- Hero -----------------
c1, c2 = st.columns([1.15, 0.85])
with c1:
    st.markdown("<div class='hero-title'>Deepfake Detection.<br/>Transparent. Fast. Trusted.</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Upload a short clip to detect AI-generated or manipulated videos. We show a timeline of suspicious moments and explain why.</div>", unsafe_allow_html=True)
with c2:
    st.markdown("")  # (optional space for future illustration)

# ----------------- Upload & settings -----------------
with st.container():
    st.markdown("##### Upload video/audio *(mp4/mov/mkv/webm/wav/mp3)*")
    uploaded = st.file_uploader("Drag and drop file here", type=["mp4","mov","mkv","webm","wav","mp3"])
    cols = st.columns([1,1,3], gap="small")
    with cols[0]:
        fps = st.slider("Analysis FPS", 20, 30, 25)
    with cols[1]:
        st.markdown(f"<span class='badge'>API: {API_URL}</span>", unsafe_allow_html=True)

    go = st.button("Analyze", type="primary", use_container_width=False)
    if uploaded and go:
        with st.spinner("Analyzing…"):
            files = {"file": (uploaded.name, uploaded.getbuffer(), uploaded.type or "application/octet-stream")}
            t0 = time.time()
            try:
                r = requests.post(API_URL, files=files, timeout=300)
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()
            dt = time.time() - t0
            if r.status_code != 200:
                st.error(f"API error {r.status_code}: {getattr(r,'text',r)}")
                st.stop()
            data = r.json()
            st.success(f"Done in {dt:.1f}s")

            # ---------- Summary strip ----------
            s = data.get("summary", {})
            colA, colB, colC = st.columns(3)
            with colA:
                st.markdown("**Summary**")
                st.metric("Clip length (s)", f"{s.get('clip_seconds',0):.1f}")
            with colB:
                st.metric("Detectors triggered", ", ".join(s.get("sources_flagged", [])) or "—")
            with colC:
                st.metric("Suspicious spans", s.get("total_spans", 0))

            # ---------- Timeline ----------
            st.markdown("### Suspicious Timeline")
            timeline = data.get("timeline", [])
            if timeline:
                import pandas as pd
                df = pd.DataFrame([{
                    "#": i+1,
                    "Start (s)": round(it["start"],2),
                    "End (s)": round(it["end"],2),
                    "Why": it.get("reason","")
                } for i, it in enumerate(timeline)])
                st.dataframe(df, use_container_width=True, hide_index=True, column_config={
                    "Why": st.column_config.TextColumn(width="large")
                })
                st.caption("Tip: Judges can skim this table to see exactly where to jump.")
            else:
                st.info("No suspicious spans detected.")

            # ---------- Per-detector ----------
            st.markdown("### Per-Detector Results")
            per = data.get("per_agent", {})
            for name, block in per.items():
                with st.expander(f"{name.capitalize()} — {len(block.get('spans',[]))} span(s)"):
                    st.json(block.get("spans", []))
                    if block.get("details"):
                        with st.expander("Technical details (optional)"):
                            st.json(block["details"])

# ----------------- Feature cards -----------------
st.markdown("---")
cc1, cc2, cc3 = st.columns(3)
with cc1:
    st.markdown("<div class='card'><h4>✅ Transparent</h4><p>We show exactly where it looks wrong with timestamps and reasons.</p></div>", unsafe_allow_html=True)
with cc2:
    st.markdown("<div class='card'><h4>🔒 Private</h4><p>Local demo: files are analyzed on your machine during the hackathon.</p></div>", unsafe_allow_html=True)
with cc3:
    st.markdown("<div class='card'><h4>🎯 Accurate</h4><p>Conservative rules to avoid false alarms; clear limits for each detector.</p></div>", unsafe_allow_html=True)

# ----------------- Footer -----------------
st.markdown("<br/>", unsafe_allow_html=True)
f1, f2, f3 = st.columns([2,1,1])
with f1:
    st.markdown("© 2025 VeriFace — Deepfake Intrusion Alarm")
with f2:
    try:
        st.page_link("pages/3_Privacy_Policy.py", label="Privacy Policy")
    except Exception:
        st.write("Privacy Policy")
with f3:
    try:
        st.page_link("pages/4_Contact.py", label="Contact")
    except Exception:
        st.write("Contact")


