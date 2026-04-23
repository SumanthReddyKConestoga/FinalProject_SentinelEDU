"""AI Advisor — RAG-powered chat for academic advisors."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

st.set_page_config(page_title="AI Advisor | SentinelEDU", layout="wide")

css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "theme", "custom.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

import dashboard.api_client as api
from dashboard.components.cards import page_header

_API_ERROR_MSG = (
    "⚠️ **Could not reach the AI Advisor API.** "
    "Make sure the FastAPI server is running on port 8001 "
    "(`python -m uvicorn src.api.main:app --port 8001 --reload`) "
    "and that it was restarted after the RAG routes were added."
)


def _push_reply(result: dict | None):
    """Append the assistant reply to session messages, always — even on API failure."""
    if result:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": result.get("answer", "No answer returned."),
            "sources": result.get("sources", []),
        })
    else:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": _API_ERROR_MSG,
            "sources": [],
        })

page_header(
    "AI Advisor",
    "RAG-powered advisor assistant — ask anything about a student or intervention strategy.",
    "🤖",
)

# ── Quick-question bank ───────────────────────────────────────────────────────
QUICK_QUESTIONS = [
    "What should I do first for a High Risk student?",
    "How do I conduct an effective advisor meeting?",
    "What are signs of a mental health crisis in academic data?",
    "How do I make a tutoring referral that students actually follow?",
    "What does the CNN risk model detect that others miss?",
    "How do I write an Academic Improvement Plan?",
    "What protective factors predict academic success?",
    "How do I handle an academic probation conversation?",
]

# ── Layout: left panel (controls) + right panel (chat) ───────────────────────
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown(
        '<div class="section-title">Student Context (optional)</div>',
        unsafe_allow_html=True,
    )
    sid = st.text_input(
        "Student ID",
        value=st.session_state.get("selected_student", ""),
        placeholder="e.g. S0042",
        label_visibility="collapsed",
    )

    profile_data = None
    if sid:
        with st.spinner("Loading profile…"):
            profile_data = api.get_profile(sid)
        if profile_data:
            s = profile_data.get("student", {})
            risk = s.get("current_risk_class", "—")
            g3   = s.get("current_predicted_g3")
            seg  = s.get("segment", "—")
            _RISK_COLOR = {
                "Low": "#10B981", "Medium": "#F59E0B",
                "High": "#EF4444", "Critical": "#DC2626",
            }
            rc = _RISK_COLOR.get(risk, "#94A3B8")
            st.markdown(
                f'<div style="background:#F8FAFC;border:1px solid #E2E8F0;'
                f'border-radius:12px;padding:.85rem 1rem;margin-top:.5rem;">'
                f'<div style="font-size:.62rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:.1em;color:#94A3B8;margin-bottom:.5rem;">Profile Summary</div>'
                f'<div style="display:flex;flex-direction:column;gap:.35rem;">'
                f'<div style="display:flex;justify-content:space-between;">'
                f'<span style="font-size:.75rem;color:#64748B;">Risk Class</span>'
                f'<span style="font-size:.75rem;font-weight:700;color:{rc};">{risk}</span>'
                f'</div>'
                f'<div style="display:flex;justify-content:space-between;">'
                f'<span style="font-size:.75rem;color:#64748B;">Predicted G3</span>'
                f'<span style="font-size:.75rem;font-weight:700;color:#0F172A;">'
                f'{f"{g3:.1f}/20" if g3 is not None else "—"}</span>'
                f'</div>'
                f'<div style="display:flex;justify-content:space-between;">'
                f'<span style="font-size:.75rem;color:#64748B;">Segment</span>'
                f'<span style="font-size:.75rem;font-weight:600;color:#334155;">{seg}</span>'
                f'</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # One-click AI recommendation button
            st.markdown("<div style='margin-top:.6rem;'></div>", unsafe_allow_html=True)
            if st.button("Generate AI Recommendations", use_container_width=True, type="primary"):
                with st.spinner("Analysing student profile…"):
                    result = api.rag_recommend(sid)
                st.session_state.setdefault("messages", [])
                st.session_state["messages"].append({
                    "role": "user",
                    "content": f"Generate recommendations for student {sid}",
                })
                _push_reply(result)
                st.rerun()
        else:
            st.warning(f"Student '{sid}' not found.")

    st.markdown(
        '<div class="section-title" style="margin-top:1rem;">Quick Questions</div>',
        unsafe_allow_html=True,
    )
    for q in QUICK_QUESTIONS:
        if st.button(q, use_container_width=True, key=f"qq_{q[:20]}"):
            st.session_state.setdefault("messages", [])
            st.session_state["messages"].append({"role": "user", "content": q})
            with st.spinner("Thinking…"):
                result = api.rag_query(q, student_id=sid if sid else None)
            _push_reply(result)
            st.rerun()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


with right:
    st.markdown(
        '<div class="section-title">Advisor Chat</div>',
        unsafe_allow_html=True,
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Render chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state["messages"]:
            st.markdown(
                '<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:16px;'
                'padding:2rem;text-align:center;color:#94A3B8;">'
                '<div style="font-size:2rem;margin-bottom:.5rem;">🤖</div>'
                '<div style="font-size:.85rem;font-weight:600;color:#64748B;">'
                'Ask anything about intervention strategies,<br>'
                'or load a student profile for personalised advice.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state["messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and msg.get("sources"):
                        with st.expander("Sources used", expanded=False):
                            for src in msg["sources"]:
                                cat = src.get("category", "").replace("_", " ").title()
                                title = src.get("title", "")
                                st.markdown(
                                    f'<div style="font-size:.75rem;color:#64748B;'
                                    f'padding:.2rem 0;">'
                                    f'<span style="background:#EFF6FF;color:#1D4ED8;'
                                    f'border-radius:4px;padding:.1em .4em;font-size:.65rem;'
                                    f'font-weight:600;margin-right:.4rem;">{cat}</span>'
                                    f'{title}</div>',
                                    unsafe_allow_html=True,
                                )

    # Chat input
    if prompt := st.chat_input("Ask about interventions, risk factors, or a specific student…"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.spinner("Thinking…"):
            result = api.rag_query(prompt, student_id=sid if sid else None)
        _push_reply(result)
        st.rerun()
