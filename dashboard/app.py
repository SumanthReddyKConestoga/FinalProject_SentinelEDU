"""SentinelEDU Dashboard — entry point & shared sidebar."""
import sys
import os
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="SentinelEDU",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject global CSS on every page load
css_path = os.path.join(os.path.dirname(__file__), "theme", "custom.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

import dashboard.api_client as api

# ── Shared sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    # Section label
    st.markdown(
        '<div style="margin-bottom:.45rem;">'
        '<span style="font-family:\'Space Grotesk\',sans-serif;font-size:.54rem;'
        'font-weight:700;letter-spacing:.2em;color:#CBD5E1;text-transform:uppercase;">'
        'SYSTEM</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Live status pill
    status  = api.stream_status()
    running = status.get("running", False) if status else False
    label   = "Live" if running else "Stopped"
    pill_bg     = "#F0FDF4" if running else "#F8FAFC"
    pill_border = "#BBF7D0" if running else "#E2E8F0"
    dot_color   = "#10B981" if running else "#94A3B8"
    text_color  = "#065F46" if running else "#64748B"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:.5rem;'
        f'background:{pill_bg};border:1px solid {pill_border};'
        f'border-radius:8px;padding:.4rem .65rem;margin-bottom:.5rem;">'
        f'<span style="width:7px;height:7px;border-radius:50%;'
        f'background:{dot_color};display:inline-block;flex-shrink:0;"></span>'
        f'<span style="font-family:\'Space Grotesk\',sans-serif;'
        f'font-size:.75rem;font-weight:600;color:{text_color};">'
        f'Stream: {label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Controls
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start", use_container_width=True, key="stream_start"):
            api.start_stream()
            st.rerun()
    with c2:
        if st.button("Stop", use_container_width=True, key="stream_stop"):
            api.stop_stream()
            st.rerun()

    if status:
        depth     = status.get("queue_depth", 0)
        processed = status.get("consumer", {}).get("events_processed", 0)
        st.markdown(
            f'<div style="font-family:\'Space Grotesk\',sans-serif;'
            f'font-size:.62rem;color:#94A3B8;letter-spacing:.03em;margin-top:.3rem;">'
            f'Queue: {depth} &nbsp;·&nbsp; Processed: {processed}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="border-top:1px solid #F1F5F9;margin:.65rem 0;"></div>',
        unsafe_allow_html=True,
    )

    # API health badge
    health = api.health()
    if health:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:.45rem;'
            'background:#F0FDF4;border:1px solid #BBF7D0;'
            'border-radius:8px;padding:.38rem .65rem;">'
            '<span style="color:#10B981;font-size:.85rem;font-weight:700;">&#10003;</span>'
            '<span style="font-family:\'Space Grotesk\',sans-serif;font-size:.73rem;'
            'color:#065F46;font-weight:600;">API Connected</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:.45rem;'
            'background:#FEF2F2;border:1px solid #FECACA;'
            'border-radius:8px;padding:.38rem .65rem;">'
            '<span style="color:#EF4444;font-size:.85rem;font-weight:700;">&#10005;</span>'
            '<span style="font-family:\'Space Grotesk\',sans-serif;font-size:.73rem;'
            'color:#991B1B;font-weight:600;">API Offline</span>'
            '</div>',
            unsafe_allow_html=True,
        )

# Auto-redirect root URL → Home page
st.switch_page("pages/1_🏠_Home.py")
