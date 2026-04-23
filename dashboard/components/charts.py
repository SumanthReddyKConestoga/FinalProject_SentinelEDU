"""Reusable Plotly chart components — premium v2."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_RISK_COLORS = {
    "Low":      "#10B981",
    "Medium":   "#F59E0B",
    "High":     "#EF4444",
    "Critical": "#DC2626",
}

_FONT = dict(family="Inter, system-ui, sans-serif", size=12, color="#334155")

_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=_FONT,
)

_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor="#F1F5F9",
    gridwidth=1,
    zeroline=False,
    linecolor="#E2E8F0",
    tickfont=dict(size=11, color="#64748B"),
    title_font=dict(size=11, color="#64748B"),
)


def risk_donut(counts: dict) -> go.Figure:
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [_RISK_COLORS.get(l, "#94A3B8") for l in labels]
    total  = sum(values)
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, family="Inter, sans-serif"),
        hovertemplate="<b>%{label}</b><br>%{value} students (%{percent})<extra></extra>",
        direction="clockwise",
        sort=False,
    ))
    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:10px'>Students</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color="#0F172A", family="Inter, sans-serif"),
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        showlegend=True,
        legend=dict(
            orientation="h", y=-0.08, x=0.5, xanchor="center",
            font=dict(size=11, color="#64748B"),
        ),
        height=280,
        margin=dict(t=16, b=16, l=16, r=16),
    )
    return fig


def segment_bar(segments: list) -> go.Figure:
    if not segments:
        return go.Figure()
    df = pd.DataFrame(segments).sort_values("count", ascending=True)
    accent_shades = ["#BFDBFE", "#93C5FD", "#60A5FA", "#3B82F6", "#2563EB", "#1D4ED8"]
    colors = accent_shades[:len(df)] if len(df) <= len(accent_shades) else ["#3B82F6"] * len(df)
    fig = go.Figure(go.Bar(
        x=df["count"],
        y=df["segment"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="white", width=0.5)),
        hovertemplate="<b>%{y}</b>: %{x} students<extra></extra>",
        text=df["count"],
        textposition="outside",
        textfont=dict(size=11, color="#334155"),
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=230,
        xaxis={**_AXIS_STYLE, "title": None, "showgrid": False},
        yaxis=dict(showgrid=False, zeroline=False, linecolor="#E2E8F0",
                   tickfont=dict(size=11, color="#334155")),
        bargap=0.35,
        margin=dict(t=16, b=16, l=16, r=16),
    )
    return fig


def trend_chart(weekly_df: pd.DataFrame) -> go.Figure:
    if weekly_df.empty:
        return go.Figure()
    fig = go.Figure()

    if "attendance_pct" in weekly_df.columns:
        fig.add_trace(go.Scatter(
            x=weekly_df["week"], y=weekly_df["attendance_pct"],
            name="Attendance %",
            line=dict(color="#3B82F6", width=2.5, shape="spline"),
            mode="lines+markers",
            marker=dict(size=7, color="#3B82F6", line=dict(color="white", width=1.5)),
            hovertemplate="Week %{x}: <b>%{y:.1f}%</b><extra>Attendance</extra>",
        ))

    if "quiz_score" in weekly_df.columns:
        fig.add_trace(go.Scatter(
            x=weekly_df["week"], y=weekly_df["quiz_score"],
            name="Quiz Score",
            line=dict(color="#10B981", width=2.5, shape="spline"),
            mode="lines+markers",
            marker=dict(size=7, color="#10B981", line=dict(color="white", width=1.5)),
            yaxis="y2",
            hovertemplate="Week %{x}: <b>%{y:.1f}</b><extra>Quiz Score</extra>",
        ))

    if "late_count" in weekly_df.columns:
        fig.add_trace(go.Bar(
            x=weekly_df["week"], y=weekly_df["late_count"],
            name="Late Count",
            marker=dict(color="#F59E0B", opacity=0.55, line=dict(color="white", width=0.5)),
            yaxis="y2",
            hovertemplate="Week %{x}: <b>%{y}</b> late<extra></extra>",
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        height=320,
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                    font=dict(size=11, color="#64748B")),
        xaxis=dict(**_AXIS_STYLE, title="Week"),
        yaxis=dict(**_AXIS_STYLE, title="Attendance %", tickformat=".0f"),
        yaxis2=dict(title="Score / Count", overlaying="y", side="right",
                    showgrid=False, zeroline=False,
                    tickfont=dict(size=11, color="#64748B")),
        margin=dict(t=16, b=55, l=50, r=50),
        bargap=0.5,
    )
    return fig


def prediction_timeline(preds_df: pd.DataFrame) -> go.Figure:
    if preds_df.empty:
        return go.Figure()
    reg = preds_df[preds_df["model_name"].str.contains("regression", case=False, na=False)].copy()
    if reg.empty:
        return go.Figure()
    reg = reg.sort_values("created_at")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=reg["created_at"], y=reg["prediction_value"],
        mode="lines+markers",
        line=dict(color="#3B82F6", width=2.5, shape="spline"),
        marker=dict(size=7, color="#3B82F6", line=dict(color="white", width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.06)",
        name="Predicted G3",
        hovertemplate="<b>G3: %{y:.1f}</b><br>%{x}<extra></extra>",
    ))
    fig.add_hline(
        y=10, line_dash="dot", line_color="#EF4444", line_width=1.5,
        annotation_text="Pass threshold",
        annotation_font=dict(size=10, color="#EF4444"),
        annotation_position="top left",
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        height=260,
        xaxis=dict(**_AXIS_STYLE, title=None),
        yaxis=dict(**_AXIS_STYLE, title="Predicted Grade (G3)", range=[0, 21]),
        margin=dict(t=16, b=30, l=50, r=16),
        showlegend=False,
    )
    return fig


def confusion_matrix_fig(cm: list, labels: list) -> go.Figure:
    z = [[cm[i][j] for j in range(len(cm[i]))] for i in range(len(cm))]
    total = sum(sum(row) for row in z)
    pct   = [[f"{v}<br>({100*v/total:.1f}%)" if total else str(v) for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale=[[0, "#EFF6FF"], [1, "#1D4ED8"]],
        text=pct,
        texttemplate="%{text}",
        textfont=dict(size=11),
        showscale=False,
        hovertemplate="Pred: <b>%{x}</b><br>Actual: <b>%{y}</b><br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=320,
        xaxis=dict(title="Predicted", tickfont=dict(size=11, color="#334155"),
                   side="bottom", showgrid=False),
        yaxis=dict(title="Actual", tickfont=dict(size=11, color="#334155"),
                   autorange="reversed", showgrid=False),
        margin=dict(t=16, b=16, l=16, r=16),
    )
    return fig


def alert_severity_bar(alerts: list) -> go.Figure:
    from collections import Counter
    order = ["critical", "high", "medium", "low"]
    colors_map = {
        "critical": "#DC2626", "high": "#EF4444",
        "medium": "#F59E0B", "low": "#10B981",
    }
    counts = Counter(a.get("severity", "low").lower() for a in alerts)
    labels = [s for s in order if s in counts]
    values = [counts[s] for s in labels]
    colors = [colors_map[s] for s in labels]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=colors, line=dict(color="white", width=0.5)),
        text=values, textposition="outside",
        textfont=dict(size=12, color="#334155"),
        hovertemplate="<b>%{x}</b>: %{y} alerts<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=200,
        xaxis=dict(showgrid=False, zeroline=False, linecolor="#E2E8F0",
                   tickfont=dict(size=12, color="#334155")),
        yaxis=dict(**_AXIS_STYLE, title=None),
        bargap=0.4,
        showlegend=False,
        margin=dict(t=16, b=16, l=16, r=16),
    )
    return fig


def gauge_fig(value: float, title: str = "", as_pct: bool = True) -> go.Figure:
    """Speedometer gauge for a single metric (value 0–1 or 0–100)."""
    display = value * 100 if as_pct and value <= 1.0 else value
    dmax = 100 if as_pct else max(display * 1.25, 1)
    ratio = display / dmax
    bar_color = "#10B981" if ratio >= 0.80 else "#F59E0B" if ratio >= 0.60 else "#EF4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display,
        number=dict(
            suffix="%" if as_pct else "",
            valueformat=".1f",
            font=dict(size=26, color="#0F172A", family="Inter, sans-serif"),
        ),
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=11, color="#64748B", family="Inter, sans-serif")),
        gauge=dict(
            axis=dict(range=[0, dmax], nticks=5,
                      tickfont=dict(size=9, color="#94A3B8"), tickcolor="#E2E8F0"),
            bar=dict(color=bar_color, thickness=0.65),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,          dmax * 0.60], color="#FEF2F2"),
                dict(range=[dmax * 0.60, dmax * 0.80], color="#FFFBEB"),
                dict(range=[dmax * 0.80, dmax],        color="#ECFDF5"),
            ],
        ),
    ))
    fig.update_layout(**_LAYOUT_BASE, height=190,
                      margin=dict(t=40, b=8, l=16, r=16))
    return fig


def model_radar_fig(models_dict: dict, metrics: list) -> go.Figure:
    """Radar/spider chart comparing multiple models across shared metrics."""
    palette = ["#2563EB", "#10B981", "#F59E0B", "#8B5CF6", "#EF4444", "#EC4899"]
    fig = go.Figure()
    for i, (name, vals) in enumerate(models_dict.items()):
        r_vals = [min(float(vals.get(m, 0)), 1.0) for m in metrics]
        hex_c = palette[i % len(palette)]
        r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
        fig.add_trace(go.Scatterpolar(
            r=r_vals + [r_vals[0]],
            theta=metrics + [metrics[0]],
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.13)",
            line=dict(color=hex_c, width=2.5),
            name=name,
            hovertemplate="%{theta}: <b>%{r:.3f}</b><extra>" + name + "</extra>",
        ))
    fig.update_layout(
        **_LAYOUT_BASE,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%",
                            tickfont=dict(size=9, color="#94A3B8"),
                            gridcolor="#E2E8F0", linecolor="#E2E8F0"),
            angularaxis=dict(tickfont=dict(size=12, color="#334155"),
                             gridcolor="#E2E8F0", linecolor="#E2E8F0"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.14, x=0.5, xanchor="center",
                    font=dict(size=11, color="#64748B")),
        height=400,
        margin=dict(t=20, b=60, l=20, r=20),
    )
    return fig
