"""Roster — Student list with filtering."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Roster | SentinelEDU", layout="wide")

css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "theme", "custom.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

import dashboard.api_client as api
from dashboard.components.cards import page_header
from dashboard.components.tables import styled_roster

page_header("Student Roster", "Browse, filter and navigate to any student profile.", "👥")

students = api.list_students(limit=1000)
if not students:
    st.error("No student data — ensure API is running and DB is seeded.")
    st.stop()

df = pd.DataFrame(students)

# ── Filter bar ────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="background:#fff;border-radius:14px;padding:1rem 1.25rem;'
    'border:1px solid #E2E8F0;box-shadow:0 1px 3px rgba(15,23,42,.07);'
    'margin-bottom:1rem;">',
    unsafe_allow_html=True,
)
f1, f2, f3, f4 = st.columns([2, 1.5, 1.5, 1])
with f1:
    search = st.text_input("Search by ID", placeholder="e.g. S0042", label_visibility="collapsed")
    st.caption("Search by ID")
with f2:
    risk_options = ["All Risk Levels"] + sorted(df["current_risk_class"].dropna().unique().tolist())
    risk_filter  = st.selectbox("Risk", risk_options, label_visibility="collapsed")
    st.caption("Risk level")
with f3:
    seg_options = ["All Segments"] + sorted(df["segment"].dropna().unique().tolist())
    seg_filter  = st.selectbox("Segment", seg_options, label_visibility="collapsed")
    st.caption("Segment")
with f4:
    school_opts = ["All Schools"] + sorted(df["school"].dropna().unique().tolist()) if "school" in df else ["All Schools"]
    school_filter = st.selectbox("School", school_opts, label_visibility="collapsed")
    st.caption("School")
st.markdown("</div>", unsafe_allow_html=True)

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = df.copy()
if search:
    filtered = filtered[filtered["id"].str.contains(search, case=False, na=False)]
if risk_filter != "All Risk Levels":
    filtered = filtered[filtered["current_risk_class"] == risk_filter]
if seg_filter != "All Segments":
    filtered = filtered[filtered["segment"] == seg_filter]
if school_filter != "All Schools" and "school" in filtered.columns:
    filtered = filtered[filtered["school"] == school_filter]

# ── Summary pills ─────────────────────────────────────────────────────────────
n = len(filtered)
n_high = len(filtered[filtered["current_risk_class"].isin(["High", "Critical"])]) if "current_risk_class" in filtered else 0

_high_pill = (
    f'<span style="background:#FEF2F2;color:#991B1B;border-radius:999px;'
    f'padding:.15em .65em;font-size:.72rem;font-weight:700">'
    f'{n_high} high/critical</span>'
) if n_high else ''
st.markdown(
    f'<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.75rem;flex-wrap:wrap">'
    f'<span style="font-size:.82rem;font-weight:600;color:#334155">'
    f'{n} student{"s" if n != 1 else ""} shown</span>'
    f'{_high_pill}'
    f'</div>',
    unsafe_allow_html=True,
)

styled_roster(filtered)

# ── Quick-navigate ────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
st.markdown(
    '<div style="background:#fff;border-radius:14px;padding:1rem 1.25rem;'
    'border:1px solid #E2E8F0;box-shadow:0 1px 3px rgba(15,23,42,.07);">',
    unsafe_allow_html=True,
)
st.markdown(
    '<span style="font-size:0.8rem;font-weight:700;color:#334155;">Quick navigate to student</span>',
    unsafe_allow_html=True,
)
nav1, nav2 = st.columns([3, 1])
with nav1:
    sid_input = st.text_input("Student ID", placeholder="e.g. S0042", label_visibility="collapsed")
with nav2:
    st.markdown("<div style='margin-top:0.1rem;'></div>", unsafe_allow_html=True)
    if st.button("View Profile →", use_container_width=True):
        if sid_input:
            st.session_state["selected_student"] = sid_input
            st.switch_page("pages/3_👤_Student_Profile.py")
        else:
            st.warning("Enter a student ID first.")
st.markdown("</div>", unsafe_allow_html=True)
