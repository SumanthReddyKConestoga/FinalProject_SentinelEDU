"""Home — Mission Control with hero splash."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=5000, key="home_refresh")
except ImportError:
    pass

st.set_page_config(page_title="Home | SentinelEDU", layout="wide")

css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "theme", "custom.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

import dashboard.api_client as api
from dashboard.components.cards import alert_card
from dashboard.components.charts import risk_donut, segment_bar, alert_severity_bar

# ── Data fetch ────────────────────────────────────────────────────────────────
students      = api.list_students(limit=1000)
alerts_feed   = api.recent_alerts(limit=200)
segments_data = api.segment_summary()

total         = len(students)
active_alerts = sum(1 for a in alerts_feed if not a.get("acknowledged", False))
high_risk     = sum(1 for s in students if s.get("current_risk_class") == "High")
critical_risk = sum(1 for s in students if s.get("current_risk_class") == "Critical")

# ── Hero splash ───────────────────────────────────────────────────────────────
st.markdown(
    '<div style="'
    'background:linear-gradient(135deg,#060B18 0%,#0F172A 28%,#1E1B4B 62%,#1E3A8A 100%);'
    'border-radius:24px;padding:2.75rem 2.75rem 2.25rem;margin-bottom:1.75rem;'
    'position:relative;overflow:hidden;'
    'box-shadow:0 25px 60px rgba(15,23,42,.45),0 0 0 1px rgba(255,255,255,.06);">'

    # decorative radial orbs
    '<div style="position:absolute;top:-90px;right:-50px;width:380px;height:380px;border-radius:50%;'
    'background:radial-gradient(circle,rgba(37,99,235,.22) 0%,transparent 70%);pointer-events:none;"></div>'
    '<div style="position:absolute;bottom:-70px;left:25%;width:280px;height:280px;border-radius:50%;'
    'background:radial-gradient(circle,rgba(99,102,241,.16) 0%,transparent 70%);pointer-events:none;"></div>'
    '<div style="position:absolute;top:35%;left:-40px;width:200px;height:200px;border-radius:50%;'
    'background:radial-gradient(circle,rgba(16,185,129,.09) 0%,transparent 70%);pointer-events:none;"></div>'

    '<div style="position:relative;z-index:1;">'

    # brand row
    '<div style="display:flex;align-items:center;gap:1.1rem;margin-bottom:1.4rem;">'
    '<div style="width:64px;height:64px;flex-shrink:0;'
    'background:linear-gradient(135deg,rgba(37,99,235,.45),rgba(99,102,241,.35));'
    'border-radius:20px;display:flex;align-items:center;justify-content:center;'
    'font-size:2rem;border:1px solid rgba(255,255,255,.18);'
    'box-shadow:0 8px 28px rgba(37,99,235,.35);">&#x1F6E1;</div>'
    '<div>'
    '<div style="font-size:2.15rem;font-weight:900;color:#fff;letter-spacing:-.055em;'
    'line-height:1;text-shadow:0 2px 24px rgba(37,99,235,.55);">SentinelEDU</div>'
    '<div style="font-size:.68rem;color:rgba(255,255,255,.38);font-weight:600;'
    'letter-spacing:.14em;text-transform:uppercase;margin-top:.28rem;">'
    'AI Student Risk Intelligence Platform</div>'
    '</div>'
    '</div>'

    # tagline
    '<div style="font-size:.97rem;color:rgba(255,255,255,.68);font-weight:400;'
    'max-width:560px;line-height:1.7;margin-bottom:1.65rem;">'
    'Detect student dropout risk before it\'s too late — real-time engagement monitoring,'
    ' ML-powered predictions, and actionable recommendations for every educator.'
    '</div>'

    # live + feature badges
    '<div style="display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;margin-bottom:2rem;">'
    '<div style="display:inline-flex;align-items:center;gap:.42rem;'
    'background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.32);'
    'border-radius:999px;padding:.32em .9em;">'
    '<div style="width:7px;height:7px;background:#10B981;border-radius:50%;'
    'box-shadow:0 0 0 0 rgba(16,185,129,.6);animation:pulse-live 2s ease-in-out infinite;"></div>'
    '<span style="font-size:.68rem;color:#6EE7B7;font-weight:700;letter-spacing:.07em;">'
    'LIVE &nbsp;·&nbsp; AUTO-REFRESH 5s</span>'
    '</div>'
    '<div style="display:inline-flex;align-items:center;gap:.35rem;'
    'background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.13);'
    'border-radius:999px;padding:.32em .9em;">'
    '<span style="font-size:.68rem;color:rgba(255,255,255,.62);font-weight:600;">'
    '&#x1F916; ML-Powered Predictions</span>'
    '</div>'
    '<div style="display:inline-flex;align-items:center;gap:.35rem;'
    'background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.13);'
    'border-radius:999px;padding:.32em .9em;">'
    '<span style="font-size:.68rem;color:rgba(255,255,255,.62);font-weight:600;">'
    '&#x26A1; Early Risk Alerts</span>'
    '</div>'
    '</div>'

    # stat strip inside the hero
    f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.85rem;">'

    f'<div style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);'
    f'border-radius:16px;padding:.95rem 1.1rem;text-align:center;">'
    f'<div style="font-size:1.9rem;font-weight:900;color:#fff;letter-spacing:-.04em;line-height:1;">{total}</div>'
    f'<div style="font-size:.65rem;color:rgba(255,255,255,.45);font-weight:600;'
    f'text-transform:uppercase;letter-spacing:.1em;margin-top:.3rem;">Total Students</div>'
    f'</div>'

    f'<div style="background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.25);'
    f'border-radius:16px;padding:.95rem 1.1rem;text-align:center;">'
    f'<div style="font-size:1.9rem;font-weight:900;color:#FCA5A5;letter-spacing:-.04em;line-height:1;">{active_alerts}</div>'
    f'<div style="font-size:.65rem;color:rgba(252,165,165,.55);font-weight:600;'
    f'text-transform:uppercase;letter-spacing:.1em;margin-top:.3rem;">Unacked Alerts</div>'
    f'</div>'

    f'<div style="background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.22);'
    f'border-radius:16px;padding:.95rem 1.1rem;text-align:center;">'
    f'<div style="font-size:1.9rem;font-weight:900;color:#FCD34D;letter-spacing:-.04em;line-height:1;">{high_risk}</div>'
    f'<div style="font-size:.65rem;color:rgba(252,211,77,.5);font-weight:600;'
    f'text-transform:uppercase;letter-spacing:.1em;margin-top:.3rem;">High Risk</div>'
    f'</div>'

    f'<div style="background:rgba(220,38,38,.14);border:1px solid rgba(220,38,38,.28);'
    f'border-radius:16px;padding:.95rem 1.1rem;text-align:center;">'
    f'<div style="font-size:1.9rem;font-weight:900;color:#F87171;letter-spacing:-.04em;line-height:1;">{critical_risk}</div>'
    f'<div style="font-size:.65rem;color:rgba(248,113,113,.5);font-weight:600;'
    f'text-transform:uppercase;letter-spacing:.1em;margin-top:.3rem;">Critical Risk</div>'
    f'</div>'

    f'</div>'   # end grid
    '</div>'    # end z-index wrapper
    '</div>',   # end hero
    unsafe_allow_html=True,
)

# ── Two-column layout ─────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="section-title">Live Alert Feed</div>', unsafe_allow_html=True)
    recent = [a for a in alerts_feed if not a.get("acknowledged")][:20]
    if recent:
        for alert in recent:
            alert_card(alert, show_student=True)
            sid = alert.get("student_id", "")
            if sid and st.button(f"→ {sid}", key=f"view_{alert['id']}",
                                 use_container_width=False):
                st.session_state["selected_student"] = sid
                st.switch_page("pages/3_👤_Student_Profile.py")
    else:
        st.markdown(
            '<div class="info-box">No unacknowledged alerts — '
            'start the stream to see live events.</div>',
            unsafe_allow_html=True,
        )

with right:
    st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
    risk_counts: dict = {}
    for s in students:
        rc = s.get("current_risk_class") or "Low"
        risk_counts[rc] = risk_counts.get(rc, 0) + 1

    if risk_counts:
        st.plotly_chart(risk_donut(risk_counts), use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.info("No student data available.")

    st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)

    if alerts_feed:
        st.markdown('<div class="section-title">Alerts by Severity</div>', unsafe_allow_html=True)
        st.plotly_chart(alert_severity_bar(alerts_feed), use_container_width=True,
                        config={"displayModeBar": False})

    if segments_data:
        st.markdown('<div class="section-title">Student Segments</div>', unsafe_allow_html=True)
        st.plotly_chart(segment_bar(segments_data), use_container_width=True,
                        config={"displayModeBar": False})
