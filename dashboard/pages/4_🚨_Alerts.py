"""Alert Center — global alert timeline with filters."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

st.set_page_config(page_title="Alerts | SentinelEDU", layout="wide")

css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "theme", "custom.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

import dashboard.api_client as api
from dashboard.components.cards import alert_card, page_header
from dashboard.components.charts import alert_severity_bar

page_header("Alert Center", "All system alerts with severity filtering and bulk acknowledgement.", "🚨")

# ── Filter bar ────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="background:#fff;border-radius:14px;padding:1rem 1.25rem;'
    'border:1px solid #E2E8F0;box-shadow:0 1px 3px rgba(15,23,42,.07);'
    'margin-bottom:1rem;">',
    unsafe_allow_html=True,
)
f1, f2, f3, f4 = st.columns([1.5, 1.5, 1.5, 1])
with f1:
    sev_filter = st.selectbox(
        "Severity", ["All", "critical", "high", "medium", "low"],
        label_visibility="collapsed",
    )
    st.caption("Severity")
with f2:
    ack_filter = st.selectbox(
        "Status", ["All", "Unacknowledged", "Acknowledged"],
        label_visibility="collapsed",
    )
    st.caption("Status")
with f3:
    limit = st.select_slider("Show up to", [25, 50, 100, 200], value=50,
                              label_visibility="collapsed")
    st.caption("Max alerts")
with f4:
    st.markdown("<div style='margin-top:0.05rem;'></div>", unsafe_allow_html=True)
    refresh = st.button("Refresh", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

alerts = api.recent_alerts(limit=limit)
if sev_filter != "All":
    alerts = [a for a in alerts if a.get("severity") == sev_filter]
if ack_filter == "Unacknowledged":
    alerts = [a for a in alerts if not a.get("acknowledged")]
elif ack_filter == "Acknowledged":
    alerts = [a for a in alerts if a.get("acknowledged")]

# ── Summary bar + bulk acknowledge ────────────────────────────────────────────
unacked   = [a for a in alerts if not a.get("acknowledged")]
n_total   = len(alerts)
n_unacked = len(unacked)

sum1, sum2 = st.columns([3, 1])
with sum1:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:.6rem;padding:.25rem 0;flex-wrap:wrap;">
          <span style="font-size:0.82rem;font-weight:700;color:#334155;">
            {n_total} alert{"s" if n_total!=1 else ""} shown
          </span>
          {"<span style='background:#FEF2F2;color:#991B1B;border-radius:999px;padding:.12em .6em;font-size:.72rem;font-weight:700;'>" + str(n_unacked) + " unacknowledged</span>" if n_unacked else "<span style='background:#F0FDF4;color:#166534;border-radius:999px;padding:.12em .6em;font-size:.72rem;font-weight:700;'>All acknowledged</span>"}
        </div>
        """,
        unsafe_allow_html=True,
    )
with sum2:
    if unacked:
        if st.button(f"Ack All ({n_unacked})", use_container_width=True):
            for a in unacked:
                api.acknowledge_alert(a["id"])
            st.success(f"Acknowledged {n_unacked} alerts.")
            st.rerun()

# ── Severity chart ────────────────────────────────────────────────────────────
if alerts:
    all_raw = api.recent_alerts(limit=200)
    st.plotly_chart(alert_severity_bar(all_raw), use_container_width=True,
                    config={"displayModeBar": False})

st.markdown("<div style='margin-top:0.25rem;'></div>", unsafe_allow_html=True)

# ── Alert list ────────────────────────────────────────────────────────────────
if not alerts:
    st.info("No alerts match the selected filters.")
else:
    for a in alerts:
        row1, row2 = st.columns([6, 1])
        with row1:
            alert_card(a, show_student=True)
        with row2:
            st.markdown("<div style='margin-top:0.15rem;display:flex;flex-direction:column;gap:.3rem;'>",
                        unsafe_allow_html=True)
            if not a.get("acknowledged"):
                if st.button("Ack", key=f"ack_{a['id']}", use_container_width=True):
                    api.acknowledge_alert(a["id"])
                    st.rerun()
            if st.button("Profile", key=f"prof_{a['id']}", use_container_width=True):
                st.session_state["selected_student"] = a.get("student_id", "")
                st.switch_page("pages/3_👤_Student_Profile.py")
            st.markdown("</div>", unsafe_allow_html=True)
