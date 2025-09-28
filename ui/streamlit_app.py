# ui/streamlit_app.py
import os, json, time, requests
import streamlit as st
import pandas as pd

# ================= Config =================
st.set_page_config(page_title="VeriFace — Deepfake Intrusion Alarm", layout="wide")

API_BASE = os.getenv("API_BASE_URL", "").rstrip("/")
if not API_BASE:
    st.error("API_BASE_URL is not set on the service.")
    st.stop()

DEFAULT_ENDPOINT = f"{API_BASE}/analyze"

st.sidebar.caption(f"API base: {API_BASE}")
ui_build_sha = os.getenv("UI_BUILD_SHA", "unknown")
st.sidebar.caption(f"UI build: {ui_build_sha}")

# ================= Theme / CSS =================
st.markdown("""
<style>
/* page background gradient (hero style) */
.stApp {
  background: radial-gradient(1200px 600px at 85% 20%, rgba(92,61,255,0.25), rgba(0,0,0,0)),
              radial-gradient(900px 500px at 10% 10%, rgba(0,135,255,0.18), rgba(0,0,0,0)),
              #0B1020;
  color: #E6E8F0;
}
[data-testid="stHeader"]{ background: rgba(0,0,0,0); }
.block-container{ max-width: 1180px; padding-top: 1.5rem; }

/* hero */
.hero-title{ font-size: 3rem; font-weight: 800; line-height: 1.1; margin: .25rem 0 .5rem; }
.hero-sub{ color:#B6BCD1; font-size:1.05rem; margin-bottom: 1.2rem; }
.logo{ font-weight:700; font-size: 1.05rem; letter-spacing:.3px; }

/* primary button */
.btn-primary button{
  background: linear-gradient(90deg,#7C5CFF,#6E40FF);
  border: 0; color: #fff; font-weight: 600;
}
.btn-primary button:hover{ filter: brightness(1.08); }

/* metric tweaks */
.css-1xarl3l, .stMetric { background: transparent; }

/* cards */
.card{ border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:16px; background:rgba(255,255,255,.03); }
.card h4{ margin:0 0 6px 0; }
.card p{ color:#ADB4C8; margin:0; font-size:.95rem; }

/* small table look */
.small-table table{ font-size: .96rem; }

/* top links */
.nav a{ color:#C9D0E3; text-decoration:none; margin-left:16px; }
.nav a:hover{ color:#fff; text-decoration:underline; }
</style>
""", unsafe_allow_html=True)

# ================= Header / Nav =================
left, right = st.columns([1,1])
with left:
    st.markdown("<div class='logo'>🔎 VeriFace</div>", unsafe_allow_html=True)
with right:
    try:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.page_link("ui/streamlit_app.py", label="Home", icon="🏠")
        with c2: st.page_link("pages/2_FAQ.py", label="FAQ", icon="❓")
        with c3: st.page_link("pages/1_About_Us.py", label="About", icon="ℹ️")
        with c4: st.page_link("pages/3_Privacy_Policy.py", label="Privacy", icon="🔒")
    except Exception:
        st.markdown("<div class='nav' style='text-align:right;'>Home · FAQ · About · Privacy</div>", unsafe_allow_html=True)

st.markdown("---")

# ================= Hero =================
hero_l, hero_r = st.columns([1.2, .8])
with hero_l:
    st.markdown("<div class='hero-title'>Deepfake Detection.<br/>Transparent. Fast. Trusted.</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Upload a short clip to detect AI-generated or manipulated videos. We show a timeline of suspicious moments and explain why.</div>", unsafe_allow_html=True)
with hero_r:
    pass  # (room for future illustration)

# ================= Sidebar =================
with st.sidebar:
    st.subheader("Settings")
    st.text_input("API endpoint", value=DEFAULT_ENDPOINT, key="api_url")
    st.markdown("Detectors used: **LipSync, Blink, Voice**")

# ================= Helpers =================
def fmt_s(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return x

def short_reason(text: str) -> str:
    if not text: return ""
    t = text.lower()
    if "lip" in t:
        return "👄 Lip-sync issue — mouth doesn’t match audio"
    if "blink" in t:
        return "👁 Blink issue — unnatural or missing blinks"
    if "voice" in t or "audio" in t:
        return "🎤 Voice artifact — robotic/flat audio patterns"
    return text

def verdict_style(verdict: str) -> str:
    v = (verdict or "").lower()
    if "likely deepfake" in v:
        return "err"
    if "likely real" in v or "likely genuine" in v:
        return "ok"
    return "warn"

# ================= Upload & Analyze =================
st.markdown("#### Upload video/audio *(mp4/mov/mkv/webm/wav/mp3)*")
up_col1, up_col2 = st.columns([1.6, .4])
with up_col1:
    uploaded = st.file_uploader("Drag & drop or browse", type=["mp4","mov","mkv","webm","wav","mp3"])
with up_col2:
    st.markdown("<div class='btn-primary'>", unsafe_allow_html=True)
    go = st.button("Analyze", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded and go:
    # ---- Call backend API ----
    with st.spinner("Analyzing…"):
        endpoint = (st.session_state.get("api_url") or DEFAULT_ENDPOINT).rstrip("/")
        files = {"file": (uploaded.name, uploaded.getbuffer(), uploaded.type or "application/octet-stream")}
        t0 = time.time()
        r = None
        try:
            r = requests.post(endpoint, files=files, timeout=300)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach API: {e}")
            if r is not None and getattr(r, "text", ""):
                st.code(r.text)
            st.stop()
        took = time.time() - t0

        data = r.json() if r and r.headers.get("content-type","").startswith("application/json") else {}
        st.success(f"Done in {took:.1f}s")

        # ---- Extract backend fields (robust defaults) ----
        s = data.get("summary", {}) or {}
        timeline = data.get("timeline", []) or []
        pa = data.get("per_agent", {}) or {}

        clip_len = s.get("clip_seconds", 0.0)
        sources_flagged = s.get("sources_flagged", []) or []
        detector_spans_total = s.get("detector_spans_total", 0)
        total_sus_seconds = float(s.get("total_suspicious_seconds", 0.0) or 0.0)
        verdict = s.get("verdict") or "⚠️ Needs Review — Some signals, not decisive."

        # ---- Verdict strip ----
        kind = verdict_style(verdict)
        if kind == "ok":
            st.success(f"Verdict: {verdict}")
        elif kind == "warn":
            st.warning(f"Verdict: {verdict}")
        else:
            st.error(f"Verdict: {verdict}")

        # ---- Metrics ----
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("🎬 Clip length (s)", fmt_s(clip_len, 1))
        with m2: st.metric("🧭 Detectors triggered", ", ".join(sources_flagged) if sources_flagged else "none")
        with m3: st.metric("🧩 Detector spans (total)", detector_spans_total)
        with m4: st.metric("⏱ Suspicious time (s)", f"{total_sus_seconds:.1f}")

        # ---- Why we think so (Top 3) ----
        st.markdown("### Why we think so")
        if timeline:
            reasons = sorted(timeline, key=lambda sp: float(sp.get("start", 0)))[:3]
            for i, sp in enumerate(reasons, 1):
                stt = fmt_s(sp.get("start", 0)); endt = fmt_s(sp.get("end", 0))
                st.write(f"**{i}. {stt}s → {endt}s** — {short_reason(sp.get('reason',''))}")
        else:
            st.write("No suspicious segments reported by detectors.")

        # ---- Suspicious Timeline (merged by backend) ----
        st.markdown("### Suspicious Timeline")
        if timeline:
            df = pd.DataFrame([{
                "#": i+1,
                "Start (s)": float(sp.get("start", 0)),
                "End (s)": float(sp.get("end", 0)),
                "Why": short_reason(sp.get("reason", "")),
            } for i, sp in enumerate(timeline)])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption("Note: windows may merge nearby spans for easier review.")
        else:
            st.success("✅ No suspicious spans detected.")

        # ---- Per-Detector Results ----
        st.markdown("### Per-Detector Results")
        for d in ["lipsync", "blink", "voice"]:
            block = pa.get(d, {}) or {}
            spans = block.get("spans", []) or []
            with st.expander(f"{d.title()} — {len(spans)} span(s)"):
                if spans:
                    df2 = pd.DataFrame([{
                        "Start (s)": float(sp.get("start", 0)),
                        "End (s)": float(sp.get("end", 0)),
                        "Why": short_reason(sp.get("reason", "")),
                    } for sp in spans])
                    st.dataframe(df2, use_container_width=True, hide_index=True)
                else:
                    st.write("Nothing suspicious here. ✅")

                det = block.get("details", {})
                if det:
                    st.toggle(f"Show technical details ({d})", value=False, key=f"details_{d}")
                    if st.session_state.get(f"details_{d}", False):
                        st.json(det)

        # ---- Export ----
        st.download_button(
            "⬇️ Download JSON report",
            data=json.dumps(data, indent=2),
            file_name=f"veriface_report_{uploaded.name}.json",
            mime="application/json"
        )

# Feature cards
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("<div class='card'><h4>✅ Transparent</h4><p>We show where it looks wrong with timestamps and reasons.</p></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='card'><h4>🔒 Private</h4><p>Local demo: files are analyzed during the session, not stored.</p></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='card'><h4>🎯 Accurate</h4><p>Conservative thresholds reduce false alarms. Clear limits shown.</p></div>", unsafe_allow_html=True)

st.caption("Legend: 👄 Lip-sync • 👁 Blink • 🎤 Voice")
