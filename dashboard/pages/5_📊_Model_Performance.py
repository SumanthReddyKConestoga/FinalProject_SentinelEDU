"""Model Performance — evaluation metrics, plots, and comparison report."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Model Performance | SentinelEDU", layout="wide")

css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "theme", "custom.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

import dashboard.api_client as api
from dashboard.components.cards import page_header
from dashboard.components.charts import (
    confusion_matrix_fig, gauge_fig, model_radar_fig,
)

page_header(
    "Model Performance",
    "Regression · Classification · Deep Learning · CNN · Clustering",
    "📊",
)


@st.cache_data(ttl=120, show_spinner=False)
def _load_metrics():
    return api.model_metrics()


@st.cache_data(ttl=120, show_spinner=False)
def _load_figures():
    return api.list_figures()


@st.cache_data(ttl=120, show_spinner=False)
def _load_comparison():
    return api.model_comparison()


with st.spinner("Loading model metrics…"):
    metrics       = _load_metrics()
    figures       = _load_figures()
    comparison_md = _load_comparison()
REPORTS_FIG   = Path(__file__).resolve().parent.parent.parent / "reports" / "figures"

_CLF_KEY_ORDER = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr"]
_CLF_LABELS    = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]

# ── helpers ───────────────────────────────────────────────────────────────────

def _val_color(v: float, invert: bool = False) -> str:
    good = v >= 0.80 if not invert else v <= 0.20
    ok   = v >= 0.60 if not invert else v <= 0.40
    return "#10B981" if good else "#F59E0B" if ok else "#EF4444"


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _metric_card(label: str, value: str, color: str = "#3B82F6"):
    st.markdown(
        f'<div style="background:#fff;border-radius:14px;padding:.9rem 1.1rem;'
        f'border:1px solid #E2E8F0;border-top:3px solid {color};'
        f'box-shadow:0 1px 3px rgba(15,23,42,.07);text-align:center;">'
        f'<div style="font-size:.6rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.09em;color:#64748B;margin-bottom:.25rem;">{label}</div>'
        f'<div style="font-size:1.3rem;font-weight:800;color:#0F172A;'
        f'letter-spacing:-.02em;">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _leaderboard_row(rank: int, name: str, metrics_row: dict,
                     cols_def: list, best_key: str, lower_is_better: bool = False):
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
    best_val = metrics_row.get(best_key, 0)
    color = _val_color(best_val, invert=lower_is_better)
    cells = "".join(
        f'<td style="padding:.55rem .7rem;font-size:.8rem;color:#334155;'
        f'font-weight:{"700" if k == best_key else "400"};'
        f'color:{color if k == best_key else "#334155"};">'
        f'{round(v, 4) if isinstance(v, float) else v}</td>'
        for k, v in metrics_row.items()
    )
    st.markdown(
        f'<tr>'
        f'<td style="padding:.55rem .7rem;font-size:.82rem;font-weight:700;'
        f'color:#0F172A;">{medal}&nbsp;{name}</td>'
        f'{cells}'
        f'</tr>',
        unsafe_allow_html=True,
    )


def _show_figs(names: list, cols: int = 3):
    if not names:
        return
    c = st.columns(cols)
    for i, fname in enumerate(names):
        fp = REPORTS_FIG / fname
        if fp.exists():
            with c[i % cols]:
                cap = fname.replace(".png", "").replace("_", " ").title()
                st.image(str(fp), caption=cap, use_container_width=True)


def _section(title: str):
    st.markdown(
        f'<div class="section-title" style="margin-top:.6rem;">{title}</div>',
        unsafe_allow_html=True,
    )


# ── Top summary strip ─────────────────────────────────────────────────────────
reg_data  = metrics.get("regression_metrics", {})
clf_data  = metrics.get("classification_metrics", {})
deep_data = metrics.get("deep_metrics", {})
cnn_data  = metrics.get("cnn_metrics", {})
clust_data = metrics.get("clustering_metrics", {})

_best_clf  = max(clf_data.items(),  key=lambda x: x[1].get("f1_macro",  0), default=("—", {}))
_best_deep = max(deep_data.items(), key=lambda x: x[1].get("f1_macro",  0), default=("—", {}))
_best_reg  = min(reg_data.items(),  key=lambda x: x[1].get("rmse",    999), default=("—", {}))

strip_items = [
    ("Best Clf Model",  _best_clf[0],  "#3B82F6"),
    ("Best Clf F1",     _pct(_best_clf[1].get("f1_macro", 0))  if clf_data  else "—", "#10B981"),
    ("Best Reg Model",  _best_reg[0],  "#8B5CF6"),
    ("Best Reg RMSE",   str(round(_best_reg[1].get("rmse", 0), 3)) if reg_data else "—", "#F59E0B"),
    ("CNN Accuracy",    _pct(cnn_data.get("accuracy", 0))      if cnn_data  else "—", "#EC4899"),
    ("Silhouette",      str(round(clust_data.get("silhouette", 0), 4)) if clust_data else "—", "#06B6D4"),
]

strip_cols = st.columns(len(strip_items))
for col, (label, val, color) in zip(strip_cols, strip_items):
    with col:
        _metric_card(label, val, color)

st.markdown("<div style='margin-top:1.1rem;'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_reg, tab_clf, tab_deep, tab_cnn, tab_clust = st.tabs([
    "📈 Regression", "🏷️ Classification", "🧠 Deep Learning", "🔁 CNN", "🔵 Clustering",
])

# ═══════════════════════════════════════════════════════════════════════════════
# REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_reg:
    _section("Regression Model Leaderboard")
    cv_data = metrics.get("regression_cv", {})

    if reg_data:
        rows = []
        for model, m in reg_data.items():
            row = {
                "Model": model,
                "RMSE":  round(m.get("rmse", 0), 4),
                "MAE":   round(m.get("mae",  0), 4),
                "R²":    round(m.get("r2",   0), 4),
            }
            if model in cv_data:
                cv = cv_data[model]
                row["CV RMSE"] = round(cv.get("cv_rmse_mean", 0), 4)
                row["CV ± σ"]  = round(cv.get("cv_rmse_std",  0), 4)
            rows.append(row)

        df_reg = pd.DataFrame(rows).sort_values("RMSE")
        st.dataframe(
            df_reg.style.background_gradient(subset=["R²"], cmap="Greens")
                        .background_gradient(subset=["RMSE"], cmap="Reds_r"),
            use_container_width=True, hide_index=True,
        )

        best = min(reg_data.items(), key=lambda x: x[1].get("rmse", 999))
        _section("Best Regression Model — Key Gauges")
        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(
                gauge_fig(best[1].get("r2", 0), "R² Score", as_pct=True),
                use_container_width=True, config={"displayModeBar": False},
                key="gauge_r2",
            )
        with g2:
            st.plotly_chart(
                gauge_fig(min(1.0 - best[1].get("rmse", 0), 1.0),
                          "1 − RMSE (higher = better)", as_pct=True),
                use_container_width=True, config={"displayModeBar": False},
                key="gauge_rmse",
            )
        with g3:
            st.plotly_chart(
                gauge_fig(min(1.0 - best[1].get("mae", 0), 1.0),
                          "1 − MAE  (higher = better)", as_pct=True),
                use_container_width=True, config={"displayModeBar": False},
                key="gauge_mae",
            )
    else:
        st.info("No regression metrics available.")

    _show_figs([f for f in figures if "residual" in f], cols=3)
    _show_figs([f for f in figures if "regression_rmse" in f], cols=1)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_clf:
    _section("Classification Model Leaderboard")

    if clf_data:
        rows = []
        for model, m in clf_data.items():
            rows.append({
                "Model":      model,
                "Accuracy":   round(m.get("accuracy",         0), 4),
                "Precision":  round(m.get("precision_macro",  0), 4),
                "Recall":     round(m.get("recall_macro",     0), 4),
                "F1 Macro":   round(m.get("f1_macro",         0), 4),
                "ROC AUC":    round(m.get("roc_auc_ovr", 0) or 0, 4),
            })
        df_clf = pd.DataFrame(rows).sort_values("F1 Macro", ascending=False)
        st.dataframe(
            df_clf.style.background_gradient(subset=["F1 Macro", "Accuracy"], cmap="Blues"),
            use_container_width=True, hide_index=True,
        )

        best = max(clf_data.items(), key=lambda x: x[1].get("f1_macro", 0))

        # gauge row for best model
        _section(f"Best Model: {best[0]} — Performance Gauges")
        g1, g2, g3, g4 = st.columns(4)
        pairs = [
            (g1, "Accuracy",  best[1].get("accuracy",        0), "gauge_clf_acc"),
            (g2, "Precision", best[1].get("precision_macro", 0), "gauge_clf_pre"),
            (g3, "Recall",    best[1].get("recall_macro",    0), "gauge_clf_rec"),
            (g4, "F1 Macro",  best[1].get("f1_macro",        0), "gauge_clf_f1"),
        ]
        for col, label, val, key in pairs:
            with col:
                st.plotly_chart(
                    gauge_fig(val, label, as_pct=True),
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key=key,
                )

        # radar chart comparing all models
        if len(clf_data) > 1:
            _section("Multi-Model Radar Comparison")
            radar_metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
            radar_labels  = ["Accuracy", "Precision", "Recall", "F1"]
            radar_models  = {
                name: {lbl: m.get(k, 0) for k, lbl in zip(radar_metrics, radar_labels)}
                for name, m in clf_data.items()
            }
            _, rc, _ = st.columns([1, 3, 1])
            with rc:
                st.plotly_chart(
                    model_radar_fig(radar_models, radar_labels),
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key="clf_radar",
                )

        # confusion matrices
        _section("Confusion Matrices")
        clf_cols = st.columns(min(len(clf_data), 3))
        labels_ref = None
        for idx, (model, m) in enumerate(clf_data.items()):
            cm  = m.get("confusion_matrix")
            lbl = m.get("labels") or labels_ref
            if cm and lbl:
                labels_ref = lbl
                with clf_cols[idx % 3]:
                    st.markdown(
                        f'<div style="font-size:.75rem;font-weight:700;color:#334155;'
                        f'margin-bottom:.25rem;text-align:center;">{model}</div>',
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(
                        confusion_matrix_fig(cm, lbl),
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key=f"cm_{model}",
                    )
    else:
        st.info("No classification metrics available.")

    _show_figs(
        [f for f in figures if "confusion" in f and "cnn" not in f and "deep" not in f],
        cols=3,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DEEP LEARNING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_deep:
    _section("Deep Learning — SLP / MLP / Tuned ANN")

    if deep_data:
        rows = [
            {
                "Model":     model,
                "Accuracy":  round(m.get("accuracy",        0), 4),
                "Precision": round(m.get("precision_macro", 0), 4),
                "Recall":    round(m.get("recall_macro",    0), 4),
                "F1 Macro":  round(m.get("f1_macro",        0), 4),
            }
            for model, m in deep_data.items()
            if "history" not in m
        ]
        if rows:
            df_deep = pd.DataFrame(rows).sort_values("F1 Macro", ascending=False)
            st.dataframe(
                df_deep.style.background_gradient(
                    subset=["F1 Macro", "Accuracy"], cmap="Purples"),
                use_container_width=True, hide_index=True,
            )

            # radar for deep models if more than one
            deep_comparable = {
                name: {
                    "Accuracy":  m.get("accuracy",        0),
                    "Precision": m.get("precision_macro", 0),
                    "Recall":    m.get("recall_macro",    0),
                    "F1":        m.get("f1_macro",        0),
                }
                for name, m in deep_data.items()
                if "history" not in m
            }
            if len(deep_comparable) > 1:
                _section("Deep Model Comparison")
                _, rc, _ = st.columns([1, 3, 1])
                with rc:
                    st.plotly_chart(
                        model_radar_fig(deep_comparable,
                                        ["Accuracy", "Precision", "Recall", "F1"]),
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key="deep_radar",
                    )

    deep_figs = [f for f in figures if "loss" in f or "deep" in f or "lr_" in f]
    if deep_figs:
        _section("Loss & Learning Curves")
        _show_figs(deep_figs[:9], cols=3)
    else:
        st.info("No deep learning figures found.")


# ═══════════════════════════════════════════════════════════════════════════════
# CNN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_cnn:
    _section("1D CNN — Sequential Features")

    if cnn_data:
        m = cnn_data
        _section("Performance Gauges")
        g1, g2, g3, g4 = st.columns(4)
        cnn_pairs = [
            (g1, "Accuracy",  m.get("accuracy",        0), "cnn_acc"),
            (g2, "Precision", m.get("precision_macro", 0), "cnn_pre"),
            (g3, "Recall",    m.get("recall_macro",    0), "cnn_rec"),
            (g4, "F1 Macro",  m.get("f1_macro",        0), "cnn_f1"),
        ]
        for col, label, val, key in cnn_pairs:
            with col:
                st.plotly_chart(
                    gauge_fig(val, label, as_pct=True),
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key=key,
                )

        cm  = m.get("confusion_matrix")
        lbl = m.get("labels")
        if cm and lbl:
            _section("Confusion Matrix")
            _, cc, _ = st.columns([1, 2, 1])
            with cc:
                st.plotly_chart(
                    confusion_matrix_fig(cm, lbl),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
    else:
        st.info("No CNN metrics available.")

    _show_figs([f for f in figures if "cnn" in f], cols=3)


# ═══════════════════════════════════════════════════════════════════════════════
# CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_clust:
    import plotly.express as px
    _section("K-Means Clustering")

    if clust_data:
        b1, b2, b3 = st.columns(3)
        with b1:
            _metric_card("Silhouette",
                         str(round(clust_data.get("silhouette", 0), 4)), "#3B82F6")
        with b2:
            _metric_card("Inertia",
                         f"{clust_data.get('inertia', 0):,.0f}", "#F59E0B")
        with b3:
            _metric_card("Clusters",
                         str(clust_data.get("n_clusters", 4)), "#10B981")

        sil_val = clust_data.get("silhouette", 0)
        _section("Silhouette Score Gauge")
        _, gc, _ = st.columns([2, 2, 2])
        with gc:
            st.plotly_chart(
                gauge_fig(sil_val, "Silhouette Score", as_pct=True),
                use_container_width=True,
                config={"displayModeBar": False},
                key="clust_sil",
            )

        segs = clust_data.get("segment_counts", {})
        if segs:
            _section("Segment Distribution")
            seg_df = pd.DataFrame(
                [{"Segment": k, "Count": v} for k, v in segs.items()]
            )
            colors = ["#BFDBFE", "#93C5FD", "#60A5FA", "#3B82F6"]
            fig = px.bar(
                seg_df, x="Segment", y="Count", text="Count",
                color="Segment", color_discrete_sequence=colors,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False, height=280,
                margin=dict(t=16, b=16, l=16, r=16),
                xaxis=dict(showgrid=False, linecolor="#E2E8F0"),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False),
                bargap=0.4,
            )
            fig.update_traces(textposition="outside",
                              textfont=dict(size=12, color="#334155"))
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
    else:
        st.info("No clustering metrics available.")

    _show_figs([f for f in figures if "cluster" in f], cols=2)


# ── Model Comparison Report ────────────────────────────────────────────────────
st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
_section("Model Comparison Report")
if comparison_md:
    st.markdown(
        f'<div style="background:#fff;border-radius:14px;padding:1.25rem 1.5rem;'
        f'border:1px solid #E2E8F0;box-shadow:0 1px 3px rgba(15,23,42,.07);">'
        f'{comparison_md}</div>',
        unsafe_allow_html=True,
    )
else:
    st.info("No model comparison report found — run `dvc repro` first.")
