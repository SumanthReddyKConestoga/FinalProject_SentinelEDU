"""Student Profile — deep-dive view."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Profile | SentinelEDU", layout="wide")

css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "theme", "custom.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

import dashboard.api_client as api
from dashboard.components.cards import alert_card, recommendation_card, risk_badge_html, page_header
from dashboard.components.charts import trend_chart, prediction_timeline

page_header("Student Profile", "Full academic history, predictions, alerts and recommendations.", "👤")

# ── Student selector ──────────────────────────────────────────────────────────
default_id = st.session_state.get("selected_student", "")
scol1, scol2 = st.columns([3, 1])
with scol1:
    sid = st.text_input("Student ID", value=default_id, placeholder="e.g. S0042",
                        label_visibility="collapsed")
with scol2:
    st.markdown("<div style='margin-top:0.05rem;'></div>", unsafe_allow_html=True)
    load_btn = st.button("Load Profile", use_container_width=True)

if not sid:
    st.markdown(
        '<div class="info-box">Enter a student ID above to load their profile.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

profile = api.get_profile(sid)
if not profile:
    st.error(f"Student '{sid}' not found or API unavailable.")
    st.stop()

student   = profile.get("student", {})
preds     = profile.get("latest_predictions", [])
weekly_raw = profile.get("weekly_records", [])
alerts_raw = profile.get("alerts", [])
recs_raw   = profile.get("recommendations", [])

risk    = student.get("current_risk_class", "Low")
pred_g3 = student.get("current_predicted_g3")
sf      = student.get("static_features") or {}
weekly_df = pd.DataFrame(weekly_raw)

# ── Hero header card ──────────────────────────────────────────────────────────
_RISK_GRADIENT = {
    "Low":      ("linear-gradient(135deg,#0F172A,#065F46)", "#10B981"),
    "Medium":   ("linear-gradient(135deg,#0F172A,#78350F)", "#F59E0B"),
    "High":     ("linear-gradient(135deg,#0F172A,#7F1D1D)", "#EF4444"),
    "Critical": ("linear-gradient(135deg,#450A0A,#7F1D1D)", "#DC2626"),
}
gradient, accent = _RISK_GRADIENT.get(risk, ("linear-gradient(135deg,#0F172A,#1E3A5F)", "#3B82F6"))

g3_display = f"{pred_g3:.1f} / 20" if pred_g3 is not None else "—"
school  = student.get("school", "—")
sex     = student.get("sex", "—")
age     = student.get("age", "—")
program = student.get("program", "—")
segment = student.get("segment", "—")

_pill = (
    f'<span style="font-size:.7rem;color:rgba(255,255,255,.55);'
    f'background:rgba(255,255,255,.08);border-radius:6px;'
    f'padding:.18em .55em;font-weight:500;">'
)
st.markdown(
    f'<div style="background:{gradient};border-radius:18px;padding:1.6rem 2rem 1.4rem;'
    f'margin-bottom:1.25rem;box-shadow:0 8px 24px rgba(15,23,42,.25);">'
    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;'
    f'flex-wrap:wrap;gap:1rem;">'
    f'<div>'
    f'<div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.6rem;">'
    f'<span style="font-size:1.6rem;font-weight:900;color:#fff;'
    f'letter-spacing:-.03em;">{sid}</span>'
    f'{risk_badge_html(risk)}'
    f'</div>'
    f'<div style="display:flex;gap:.5rem;flex-wrap:wrap;">'
    f'{_pill}🏫 {school}</span>'
    f'{_pill}👤 {sex}, {age} yrs</span>'
    f'{_pill}📚 {program}</span>'
    f'{_pill}🎯 {segment}</span>'
    f'</div>'
    f'</div>'
    f'<div style="text-align:right;">'
    f'<div style="font-size:.6rem;font-weight:700;text-transform:uppercase;'
    f'letter-spacing:.1em;color:rgba(255,255,255,.4);margin-bottom:.3rem;">Predicted G3</div>'
    f'<div style="font-size:2.4rem;font-weight:900;color:{accent};'
    f'letter-spacing:-.04em;line-height:1;">{g3_display}</div>'
    f'</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Metric row ────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

att   = weekly_df["attendance_pct"].mean() if not weekly_df.empty and "attendance_pct" in weekly_df else sf.get("mean_attendance_pct", 0)
quiz  = weekly_df["quiz_score"].mean()     if not weekly_df.empty and "quiz_score"      in weekly_df else sf.get("mean_quiz_score", 0)
late  = weekly_df["late_count"].sum()      if not weekly_df.empty and "late_count"      in weekly_df else 0
logins = weekly_df["lms_logins"].sum()     if not weekly_df.empty and "lms_logins"      in weekly_df else 0

with m1: st.metric("Avg Attendance",    f"{att:.0f}%")
with m2: st.metric("Avg Quiz Score",    f"{quiz:.1f}")
with m3: st.metric("Late Submissions",  int(late))
with m4: st.metric("Total LMS Logins",  int(logins))

st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

# ── Charts side by side ───────────────────────────────────────────────────────
ch1, ch2 = st.columns(2, gap="medium")

with ch1:
    st.markdown('<div class="section-title">Weekly Performance Trends</div>',
                unsafe_allow_html=True)
    if not weekly_df.empty:
        st.plotly_chart(trend_chart(weekly_df), use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.info("No weekly records yet.")

with ch2:
    st.markdown('<div class="section-title">Predicted Grade Over Time</div>',
                unsafe_allow_html=True)
    preds_df = pd.DataFrame(preds)
    if not preds_df.empty:
        st.plotly_chart(prediction_timeline(preds_df), use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.info("No prediction history yet.")

st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)

# ── Alerts + Recommendations side by side ─────────────────────────────────────
al_col, rec_col = st.columns(2, gap="medium")

with al_col:
    n_alerts = len(alerts_raw)
    st.markdown(
        f'<div class="section-title">Alerts '
        f'<span style="background:#FEF2F2;color:#991B1B;border-radius:999px;'
        f'padding:.05em .5em;font-size:.7rem;">{n_alerts}</span></div>',
        unsafe_allow_html=True,
    )
    if alerts_raw:
        for a in alerts_raw[:10]:
            alert_card(a, show_student=False)
            if not a.get("acknowledged"):
                if st.button("Acknowledge", key=f"ack_{a['id']}",
                             use_container_width=False):
                    api.acknowledge_alert(a["id"])
                    st.rerun()
    else:
        st.success("No active alerts for this student. 🎉")

with rec_col:
    n_recs = len(recs_raw)
    st.markdown(
        f'<div class="section-title">Recommendations '
        f'<span style="background:#FFFBEB;color:#92400E;border-radius:999px;'
        f'padding:.05em .5em;font-size:.7rem;">{n_recs}</span></div>',
        unsafe_allow_html=True,
    )

    def _take_action(rec: dict):
        api.log_advisor_action(
            sid,
            action_taken=rec.get("action", ""),
            rec_id=rec.get("id"),
            notes="Taken from dashboard",
        )
        st.success(f"Action logged: {rec.get('action','')}")
        st.rerun()

    if recs_raw:
        for r in recs_raw:
            recommendation_card(r, on_action=_take_action)
    else:
        st.info("No recommendations at this time.")

# ── Static features accordion ─────────────────────────────────────────────────
st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
with st.expander("📋 Static Features & Academic History"):
    if sf:
        cols = st.columns(3)
        for i, (k, v) in enumerate(sf.items()):
            with cols[i % 3]:
                st.metric(k, round(float(v), 2) if isinstance(v, float) else v)
    else:
        st.info("No static features stored.")
