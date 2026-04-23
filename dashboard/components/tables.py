"""Styled dataframe helpers — premium v2."""
import pandas as pd
import streamlit as st

_RISK_STYLE = {
    "Low":      "background-color:#ECFDF5;color:#065F46;font-weight:600;",
    "Medium":   "background-color:#FFFBEB;color:#92400E;font-weight:600;",
    "High":     "background-color:#FEF2F2;color:#991B1B;font-weight:600;",
    "Critical": "background-color:#FFF1F1;color:#7F1D1D;font-weight:700;",
}


def _color_risk(val):
    return _RISK_STYLE.get(val, "")


def styled_roster(df: pd.DataFrame):
    if df.empty:
        st.info("No students match the selected filters.")
        return
    display_cols = [c for c in [
        "id", "sex", "school", "current_risk_class", "current_predicted_g3", "segment"
    ] if c in df.columns]
    rename = {
        "id": "ID", "sex": "Sex", "school": "School",
        "current_risk_class": "Risk", "current_predicted_g3": "Pred G3",
        "segment": "Segment",
    }
    disp = df[display_cols].rename(columns=rename)
    if "Pred G3" in disp.columns:
        disp["Pred G3"] = disp["Pred G3"].round(1)
    styled = disp.style
    if "Risk" in disp.columns:
        styled = styled.applymap(_color_risk, subset=["Risk"])
    if "Pred G3" in disp.columns:
        styled = styled.background_gradient(
            subset=["Pred G3"], cmap="RdYlGn", vmin=0, vmax=20
        )
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.TextColumn("ID", width="small"),
            "Pred G3": st.column_config.NumberColumn("Pred G3", format="%.1f"),
        },
    )
