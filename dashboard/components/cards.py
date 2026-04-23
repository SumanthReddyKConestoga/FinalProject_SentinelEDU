"""Card components — v3.
All HTML is built with implicit Python string concatenation, producing a
single line with no embedded newlines. This avoids the CommonMark rule that
terminates a raw-HTML block on any blank (whitespace-only) line, which caused
HTML tags to leak as visible text whenever an interpolated variable was empty.
"""
import streamlit as st

_SEV_COLOR = {
    "low":      "#10B981",
    "medium":   "#F59E0B",
    "high":     "#EF4444",
    "critical": "#DC2626",
}

_SEV_BG = {
    "low":      ("#ECFDF5", "#065F46"),
    "medium":   ("#FFFBEB", "#92400E"),
    "high":     ("#FEF2F2", "#991B1B"),
    "critical": ("#FEE2E2", "#7F1D1D"),
}

_RISK_BG = {
    "Low":      ("#ECFDF5", "#065F46"),
    "Medium":   ("#FFFBEB", "#92400E"),
    "High":     ("#FEF2F2", "#991B1B"),
    "Critical": ("#FEE2E2", "#7F1D1D"),
}


def kpi_card(label: str, value, delta: str = None, color: str = "#2563EB", icon: str = ""):
    icon_html = f'<span style="font-size:1.45rem;line-height:1">{icon}</span>' if icon else ''
    delta_html = (f'<div style="font-size:.72rem;color:#64748B;margin-top:.15rem;font-weight:500">'
                  f'{delta}</div>') if delta else ''
    st.markdown(
        f'<div style="background:#fff;border-radius:16px;padding:1.3rem 1.5rem;'
        f'box-shadow:0 1px 3px rgba(15,23,42,.06),0 4px 12px rgba(37,99,235,.04);'
        f'border:1px solid #E2E8F2;border-top:4px solid {color}">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;gap:.75rem">'
        f'<div>'
        f'<div style="font-size:.62rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.1em;color:#8094AE;margin-bottom:.35rem">{label}</div>'
        f'<div style="font-size:2.1rem;font-weight:800;color:#0F172A;'
        f'letter-spacing:-.04em;line-height:1">{value}</div>'
        f'{delta_html}'
        f'</div>'
        f'<div style="width:46px;height:46px;border-radius:13px;background:{color}15;'
        f'display:flex;align-items:center;justify-content:center;flex-shrink:0">'
        f'{icon_html}'
        f'</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def alert_card(alert: dict, show_student: bool = True):
    sev    = alert.get("severity", "low").lower()
    color  = _SEV_COLOR.get(sev, "#94A3B8")
    bg, fg = _SEV_BG.get(sev, ("#F1F5F9", "#334155"))
    ack    = alert.get("acknowledged", False)
    ts     = str(alert.get("triggered_at", ""))[:19].replace("T", " ")
    desc   = alert.get("description", "")
    sid    = alert.get("student_id", "")
    rule   = alert.get("rule_id", "").replace("_", " ").title()
    ack_badge = (
        '<span style="font-size:.59rem;background:#F0FDF4;color:#166534;border-radius:999px;'
        'padding:.1em .5em;font-weight:700;border:1px solid #BBF7D0;margin-left:.25rem">✓ Acked</span>'
    ) if ack else ''
    student_span = (
        f'<span style="font-size:.72rem;font-weight:700;color:#1E293B">{sid}</span>'
        f'<span style="margin:0 .35rem;color:#CBD5E1">·</span>'
    ) if (show_student and sid) else ''
    st.markdown(
        f'<div style="background:#fff;border-radius:14px;padding:.9rem 1rem .9rem 1.1rem;'
        f'margin-bottom:.5rem;box-shadow:0 1px 3px rgba(15,23,42,.05);'
        f'border:1px solid #E2E8F2;border-left:4px solid {color};'
        f'opacity:{"0.52" if ack else "1"}">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.28rem">'
        f'<div style="display:flex;align-items:center;gap:.3rem;flex-wrap:wrap">'
        f'{student_span}'
        f'<span style="background:{bg};color:{fg};border-radius:999px;'
        f'padding:.12em .55em;font-size:.62rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:.07em">{sev}</span>'
        f'<span style="font-size:.7rem;color:#94A3B8">{rule}</span>'
        f'{ack_badge}'
        f'</div>'
        f'<span style="font-size:.67rem;color:#94A3B8;white-space:nowrap;margin-left:.5rem">{ts}</span>'
        f'</div>'
        f'<div style="font-size:.855rem;color:#0F172A;font-weight:500;line-height:1.45">{desc}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def recommendation_card(rec: dict, on_action=None):
    priority  = rec.get("priority", 0)
    pct       = int(priority * 100)
    action    = rec.get("action", "").replace("_", " ").title()
    rationale = rec.get("rationale", "")
    status    = rec.get("status", "pending")
    rec_id    = rec.get("id", 0)
    s_bg, s_fg, s_label = (
        ("#F0FDF4", "#166534", "✓ Done") if status == "completed"
        else ("#FFFBEB", "#92400E", "Pending")
    )
    bar_color = "#10B981" if pct >= 70 else "#F59E0B" if pct >= 40 else "#94A3B8"
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(
            f'<div style="background:#fff;border-radius:14px;padding:.9rem 1.1rem;'
            f'margin-bottom:.45rem;box-shadow:0 1px 3px rgba(15,23,42,.05);'
            f'border:1px solid #E2E8F2">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.28rem">'
            f'<span style="font-weight:700;font-size:.87rem;color:#0F172A">{action}</span>'
            f'<span style="font-size:.6rem;background:{s_bg};color:{s_fg};'
            f'border-radius:999px;padding:.1em .5em;font-weight:700">{s_label}</span>'
            f'</div>'
            f'<div style="font-size:.78rem;color:#64748B;margin-bottom:.45rem;line-height:1.4">{rationale}</div>'
            f'<div style="display:flex;align-items:center;gap:.5rem">'
            f'<div style="flex:1;height:4px;background:#EEF2FB;border-radius:999px;overflow:hidden">'
            f'<div style="width:{pct}%;height:100%;background:{bar_color};border-radius:999px"></div>'
            f'</div>'
            f'<span style="font-size:.65rem;color:#94A3B8;font-weight:600;white-space:nowrap">'
            f'Priority {pct}%</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        if status != "completed" and on_action:
            st.markdown("<div style='margin-top:.55rem'></div>", unsafe_allow_html=True)
            if st.button("Act", key=f"rec_{rec_id}", use_container_width=True):
                on_action(rec)


def risk_badge_html(risk: str) -> str:
    bg, fg = _RISK_BG.get(risk, ("#F1F5F9", "#334155"))
    return (
        f'<span style="background:{bg};color:{fg};border-radius:999px;'
        f'padding:.22em .85em;font-size:.7rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:.06em">{risk}</span>'
    )


def page_header(title: str, subtitle: str = "", icon: str = ""):
    icon_part = f"{icon}&nbsp;&nbsp;" if icon else ""
    sub_html = (
        f'<p style="color:rgba(255,255,255,.65);font-size:.83rem;margin:.3rem 0 0;font-weight:400">'
        f'{subtitle}</p>'
    ) if subtitle else ''
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1E1B4B 0%,#1E3A8A 48%,#2563EB 100%);'
        f'border-radius:18px;padding:1.5rem 2rem;margin-bottom:1.4rem;'
        f'box-shadow:0 8px 24px rgba(37,99,235,.22)">'
        f'<div style="color:#fff;font-size:1.5rem;font-weight:800;'
        f'letter-spacing:-.025em;line-height:1.2">{icon_part}{title}</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )
