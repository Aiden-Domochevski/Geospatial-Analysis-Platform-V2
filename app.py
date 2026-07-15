import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from scipy.spatial import cKDTree
import xgboost as xgb
import warnings, os, json, time, subprocess, threading, queue, re
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Attijariwafa — Branch Intelligence",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
#  DESIGN SYSTEM — CUSTOM CSS
# ═══════════════════════════════════════════════════════════

st.markdown('<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">', unsafe_allow_html=True)

st.markdown("""
<style>

/* ── Root & Reset ── */
:root {
    --red:      #CC1B1B;
    --red-deep: #8B0000;
    --red-glow: rgba(204,27,27,0.15);
    --gold:     #D4A017;
    --gold-lt:  #F5C842;
    --bg:       #0A0A0A;
    --bg2:      #111111;
    --bg3:      #1A1A1A;
    --bg4:      #222222;
    --border:   rgba(255,255,255,0.07);
    --text:     #F0F0F0;
    --muted:    #888888;
    --green:    #22C55E;
    --orange:   #F97316;
    --yellow:   #EAB308;
    --blue:     #3B82F6;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }
.block-container {
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Syne', sans-serif;
    color: var(--gold) !important;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover {
    border-color: var(--red) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: var(--text) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="stMetricDelta"] svg { display: none; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--bg2) !important;
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
    border: 1px solid var(--border);
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: var(--red) !important;
    color: white !important;
}

/* ── Selectbox / Multiselect ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] .stSlider > div > div > div {
    background: var(--red) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--bg4); border-radius: 3px; }

/* ── Folium map container ── */
iframe { border-radius: 12px !important; }

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════

ACTION_COLORS = {
    "KEEP — Invest":                "#22C55E",
    "KEEP — Flagship (merge nearby)":"#86EFAC",
    "MONITOR — Growth Potential":   "#3B82F6",
    "REVIEW — Merger Candidate":    "#F97316",
    "CLOSE — Redundant":            "#EF4444",
    "CLOSE — Underperformer":       "#991B1B",
    "REPOSITION — High Competition":"#A855F7",
}

ACTION_ICONS = {
    "KEEP — Invest":                "",
    "KEEP — Flagship (merge nearby)":"",
    "MONITOR — Growth Potential":   "",
    "REVIEW — Merger Candidate":    "",
    "CLOSE — Redundant":            "",
    "CLOSE — Underperformer":       "",
    "REPOSITION — High Competition":"",
}

CLUSTER_COLORS = {
    "High Performer":      "#22C55E",
    "Commercial Hub":      "#3B82F6",
    "Residential Anchor":  "#F59E0B",
    "Underperformer":      "#F97316",
    "Redundant / At Risk": "#EF4444",
}

def get_cluster_color(label):
    """Robust color lookup — handles partial matches, integers, and case differences."""
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return "#888888"
    label = str(label).strip()
    if label in ("nan", "", "None"):
        return "#888888"
    # Integer cluster id fallback
    int_map = {"0":"#22C55E","1":"#3B82F6","2":"#F59E0B","3":"#F97316","4":"#EF4444"}
    if label in int_map:
        return int_map[label]
    # Exact match
    if label in CLUSTER_COLORS:
        return CLUSTER_COLORS[label]
    # Partial match
    ll = label.lower()
    if "high" in ll or "performer" in ll:   return "#22C55E"
    if "commercial" in ll or "hub" in ll:   return "#3B82F6"
    if "residential" in ll or "anchor" in ll: return "#F59E0B"
    if "redundant" in ll or "risk" in ll:   return "#EF4444"
    if "under" in ll:                        return "#F97316"
    return "#888888"

MAP_MARKER_COLORS = {
    "KEEP — Invest":                "green",
    "KEEP — Flagship (merge nearby)":"lightgreen",
    "MONITOR — Growth Potential":   "blue",
    "REVIEW — Merger Candidate":    "orange",
    "CLOSE — Redundant":            "red",
    "CLOSE — Underperformer":       "darkred",
    "REPOSITION — High Competition":"purple",
}

# ═══════════════════════════════════════════════════════════
#  DATA LOADER
# ═══════════════════════════════════════════════════════════

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    for c in ["latitude","longitude","composite_score",
              "score_commercial","score_office","score_residential",
              "score_accessibility","score_competition","score_cannibalization",
              "commercial_1km","office_1km","residential_1km",
              "transit_1km","competitor_1km","own_branches_2km","total_poi_1km"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derive city from name heuristic if not present
    if "city" not in df.columns:
        city_keywords = {
            "Casa":"Casablanca","casa":"Casablanca","CASA":"Casablanca",
            "Rabat":"Rabat","Tanger":"Tanger","Fès":"Fès","Fes":"Fès",
            "Marrakech":"Marrakech","Agadir":"Agadir","Oujda":"Oujda",
            "Meknès":"Meknès","Meknes":"Meknès","Nador":"Nador",
            "Kénitra":"Kénitra","Kenitra":"Kénitra","Salé":"Salé",
            "Sale":"Salé","Tétouan":"Tétouan","Safi":"Safi",
            "Mohammedia":"Mohammedia","Settat":"Settat","Beni Mellal":"Béni Mellal",
            "Béni Mellal":"Béni Mellal","Taza":"Taza","Marjane":"Marjane",
        }
        def infer_city(name):
            for k, v in city_keywords.items():
                if k in str(name):
                    return v
            return "Other"
        df["city"] = df["name"].apply(infer_city)

    # Ensure action column exists
    if "action" not in df.columns:
        def classify(row):
            s = row.get("composite_score", 0.2)
            c = row.get("is_overcrowded", False)
            if s >= 0.35 and not c: return "KEEP — Invest"
            elif s >= 0.35 and c:   return "KEEP — Flagship (merge nearby)"
            elif s >= 0.20 and not c: return "MONITOR — Growth Potential"
            elif s >= 0.20 and c:   return "REVIEW — Merger Candidate"
            elif s < 0.20 and c:    return "CLOSE — Redundant"
            else:                   return "CLOSE — Underperformer"
        df["action"] = df.apply(classify, axis=1)

    # ── Cluster labels — always recompute from scores, never trust CSV ──
    SCORE_FEATS = [c for c in ["score_commercial","score_office","score_residential",
                                "score_accessibility","score_competition","score_cannibalization"]
                   if c in df.columns]

    if SCORE_FEATS:
        X = StandardScaler().fit_transform(df[SCORE_FEATS].fillna(0))
        df["_cid"] = KMeans(5, random_state=42, n_init=10).fit_predict(X)

        # Rank each cluster by its mean composite score: rank 0 = best, rank 4 = worst
        cluster_means = df.groupby("_cid")["composite_score"].mean().sort_values(ascending=False)
        rank_map = {cid: rank for rank, cid in enumerate(cluster_means.index)}

        RANK_TO_LABEL = {
            0: "High Performer",
            1: "Commercial Hub",
            2: "Residential Anchor",
            3: "Underperformer",
            4: "Redundant / At Risk",
        }
        df["cluster_label"] = df["_cid"].map(rank_map).map(RANK_TO_LABEL)
        df.drop(columns=["_cid"], inplace=True)


    if "pca_x" not in df.columns:
        feats = ["score_commercial","score_office","score_residential",
                 "score_accessibility","score_competition","score_cannibalization"]
        feats = [f for f in feats if f in df.columns]
        if feats:
            X = StandardScaler().fit_transform(df[feats].fillna(0))
            pc = PCA(2, random_state=42).fit_transform(X)
            df["pca_x"] = pc[:, 0].round(4)
            df["pca_y"] = pc[:, 1].round(4)

    if "rank" not in df.columns:
        df = df.sort_values("composite_score", ascending=False)
        df["rank"] = range(1, len(df)+1)

    return df.dropna(subset=["latitude","longitude"])

# ═══════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════

def render_header():
    import base64
    logo_html = ""
    if os.path.exists("logo.png"):
        with open("logo.png", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/png;base64,{b64}" style="width:52px;height:52px;object-fit:contain;border-radius:12px;">'
    else:
        logo_html = '<div style="width:52px;height:52px;background:#CC1B1B;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.6rem;">🏦</div>'

    st.markdown(f"""
    <div style="
        display:flex; align-items:center; justify-content:space-between;
        padding: 1.2rem 1.5rem;
        background: linear-gradient(135deg, #111111 0%, #1a0505 100%);
        border: 1px solid rgba(204,27,27,0.3);
        border-radius: 16px;
        margin-bottom: 1.5rem;
    ">
        <div style="display:flex; align-items:center; gap:1.2rem;">
            <div style="
                width:52px; height:52px;
                border-radius:12px;
                overflow:hidden;
                box-shadow: 0 0 24px rgba(204,27,27,0.5);
                flex-shrink:0;
            ">{logo_html}</div>
            <div>
                <div style="font-family:'Syne',sans-serif; font-size:1.4rem;
                            font-weight:800; color:#F0F0F0; letter-spacing:-0.02em;">
                    Attijariwafa Bank
                </div>
                <div style="font-size:0.75rem; color:#888; letter-spacing:0.15em;
                            text-transform:uppercase; font-family:'DM Sans',sans-serif;">
                    Branch Intelligence Platform
                </div>
            </div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:0.7rem; color:#555; text-transform:uppercase;
                        letter-spacing:0.1em; font-family:'Syne',sans-serif;">
                Powered by ML
            </div>
            <div style="font-size:0.7rem; color:#D4A017; margin-top:2px;">
                KMeans · DBSCAN · PCA
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Global download button in header
    if os.path.exists("data/inputs/branches_scored.csv"):
        try:
            branches_df = pd.read_csv("data/inputs/branches_scored.csv")
            branches_csv = branches_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Scored Branches (CSV)",
                data=branches_csv,
                file_name="data/inputs/branches_scored.csv",
                mime="text/csv",
                key="download_branches_header",
            )
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════
#  KPI CARDS
# ═══════════════════════════════════════════════════════════

def render_kpis(df):
    total     = len(df)
    keep      = len(df[df["action"].str.startswith("KEEP")])
    close     = len(df[df["action"].str.startswith("CLOSE")])
    review    = len(df[df["action"].str.startswith("REVIEW")])
    monitor   = len(df[df["action"].str.startswith("MONITOR")])
    top_score = df["composite_score"].max()
    avg_score = df["composite_score"].mean()
    cities    = df["city"].nunique() if "city" in df.columns else "—"

    c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
    pct = lambda x: f"{x/total*100:.0f}%" if total > 0 else "—"
    c1.metric("Total Branches", f"{total:,}")
    c2.metric(" Keep", f"{keep:,}", pct(keep))
    c3.metric(" Review", f"{review:,}", pct(review))
    c4.metric(" Monitor", f"{monitor:,}", pct(monitor))
    c5.metric(" Close", f"{close:,}", pct(close))
    c6.metric("Top Score", f"{top_score:.3f}")
    c7.metric("Avg Score", f"{avg_score:.3f}")
    c8.metric("Cities", f"{cities}")

# ═══════════════════════════════════════════════════════════
#  SIDEBAR FILTERS
# ═══════════════════════════════════════════════════════════

def render_sidebar(df):
    with st.sidebar:
        st.markdown("""
        <div style="padding:1rem 0 0.5rem; text-align:center;">
            <div style="font-family:'Syne',sans-serif; font-size:1rem;
                        font-weight:800; color:#CC1B1B; letter-spacing:0.05em;">
                 FILTERS
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### City")
        cities = sorted(df["city"].unique()) if "city" in df.columns else []
        selected_cities = st.multiselect("", ["All"] + cities, default=["All"],
                                          label_visibility="collapsed")

        st.markdown("### Action")
        actions = sorted(df["action"].unique())
        selected_actions = st.multiselect("", actions, default=list(actions),
                                           label_visibility="collapsed")

        st.markdown("### Cluster")
        clusters = sorted(df["cluster_label"].dropna().unique()) if "cluster_label" in df.columns else []
        selected_clusters = st.multiselect("", clusters, default=list(clusters),
                                            label_visibility="collapsed")

        st.markdown("### Composite Score")
        score_min = float(df["composite_score"].min())
        score_max = float(df["composite_score"].max())
        score_range = st.slider("", score_min, score_max,
                                (score_min, score_max), step=0.01,
                                label_visibility="collapsed")

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.65rem; color:#555; text-align:center;
                    text-transform:uppercase; letter-spacing:0.1em;">
            Attijariwafa Bank<br>Branch Intelligence v1.0
        </div>
        """, unsafe_allow_html=True)

    # Apply filters
    filtered = df.copy()
    if "All" not in selected_cities and selected_cities:
        filtered = filtered[filtered["city"].isin(selected_cities)]
    if selected_actions:
        filtered = filtered[filtered["action"].isin(selected_actions)]
    if selected_clusters and "cluster_label" in filtered.columns:
        filtered = filtered[filtered["cluster_label"].isin(selected_clusters)]
    filtered = filtered[
        filtered["composite_score"].between(score_range[0], score_range[1])
    ]
    return filtered

# ═══════════════════════════════════════════════════════════
#  TAB 1 — INTERACTIVE MAP
# ═══════════════════════════════════════════════════════════

def render_map_tab(df):
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:0.7rem; font-weight:700;
                color:#888; letter-spacing:0.2em; text-transform:uppercase;
                margin-bottom:0.8rem;">
         Geographic Branch Intelligence
    </div>
    """, unsafe_allow_html=True)

    col_ctrl, col_map = st.columns([1, 4])

    with col_ctrl:
        st.markdown("**Map Options**")
        map_style = st.selectbox("Basemap", ["Dark","Light","Satellite-ish"], label_visibility="visible")
        color_by   = st.selectbox("Color by", ["Action","Cluster","Score"], label_visibility="visible")
        show_radius= st.checkbox("Show 1km radius", value=False)
        cluster_markers = st.checkbox("Cluster markers", value=True)
        n = len(df)
        if n <= 1:
            max_display = n
        elif n <= 10:
            max_display = n
            st.caption(f"Showing all {n} branches")
        else:
            sl_max     = n
            sl_min     = min(5, n - 1)
            sl_default = min(500, n)
            sl_step    = max(1, sl_min)
            max_display = st.slider("Max markers", sl_min, sl_max, sl_default, sl_step)

        # Legend
        st.markdown("**Legend**")
        for action, color in ACTION_COLORS.items():
            icon = ACTION_ICONS.get(action, "●")
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:6px;
                        margin-bottom:3px; font-size:0.72rem;">
                <div style="width:10px; height:10px; border-radius:50%;
                            background:{color}; flex-shrink:0;"></div>
                <span style="color:#aaa;">{action.split('—')[1].strip() if '—' in action else action}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_map:
        tiles_map = {
            "Dark": "CartoDB dark_matter",
            "Light": "CartoDB positron",
            "Satellite-ish": "OpenStreetMap"
        }

        m = folium.Map(
            location=[31.5, -6.5],
            zoom_start=6,
            tiles=tiles_map[map_style],
            prefer_canvas=True,
        )

        display_df = df.head(max_display)

        if cluster_markers:
            from folium.plugins import MarkerCluster
            mc = MarkerCluster(
                options={"spiderfyOnMaxZoom": True,
                         "disableClusteringAtZoom": 12}
            )
        
        for _, row in display_df.iterrows():
            if color_by == "Action":
                fc = MAP_MARKER_COLORS.get(row.get("action", ""), "gray")
            elif color_by == "Cluster":
                cl = row.get("cluster_label", "")
                fc = {"High Performer":"green","Commercial Hub":"blue",
                      "Residential Anchor":"orange","Underperformer":"lightred",
                      "Redundant / At Risk":"red"}.get(cl, "gray")
            else:  # Score
                s = row.get("composite_score", 0)
                fc = "green" if s > 0.35 else "orange" if s > 0.2 else "red"

            popup_html = f"""
            <div style="font-family:sans-serif; width:220px;">
                <div style="font-weight:700; font-size:0.9rem;
                            border-bottom:2px solid #CC1B1B; padding-bottom:4px;
                            margin-bottom:6px;">{row['name']}</div>
                <div style="font-size:0.78rem; color:#333;">
                    <b>Score:</b> {row.get('composite_score',0):.3f}<br>
                    <b>Action:</b> {row.get('action','—')}<br>
                    <b>Cluster:</b> {row.get('cluster_label','—')}<br>
                    <hr style="margin:4px 0;">
                    <b>Commercial 1km:</b> {int(row.get('commercial_1km',0))}<br>
                    <b>Office 1km:</b> {int(row.get('office_1km',0))}<br>
                    <b>Competitors 1km:</b> {int(row.get('competitor_1km',0))}<br>
                    <b>Own branches 2km:</b> {int(row.get('own_branches_2km',0))}
                </div>
            </div>
            """

            marker = folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(popup_html, max_width=240),
                tooltip=f"{row['name']} | {row.get('composite_score',0):.3f}",
                icon=folium.Icon(color=fc, icon="building", prefix="fa"),
            )

            if show_radius:
                folium.Circle(
                    location=[row["latitude"], row["longitude"]],
                    radius=1000,
                    color=ACTION_COLORS.get(row.get("action",""), "#888"),
                    fill=True, fill_opacity=0.05, weight=1,
                ).add_to(m)

            if cluster_markers:
                marker.add_to(mc)
            else:
                marker.add_to(m)

        if cluster_markers:
            mc.add_to(m)

        st_folium(m, width="100%", height=560, returned_objects=[])

# ═══════════════════════════════════════════════════════════
#  TAB 2 — BRANCH RANKINGS
# ═══════════════════════════════════════════════════════════

def render_rankings_tab(df):
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:0.7rem; font-weight:700;
                color:#888; letter-spacing:0.2em; text-transform:uppercase;
                margin-bottom:0.8rem;">
         Branch Performance Rankings
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        # Horizontal bar — top/bottom
        mode = st.radio("Show", ["Top 20", "Bottom 20", "All"], horizontal=True)
        
        if mode == "Top 20":
            plot_df = df.nlargest(20, "composite_score")
            title = "Top 20 Branches by Composite Score"
        elif mode == "Bottom 20":
            plot_df = df.nsmallest(20, "composite_score")
            title = "Bottom 20 Branches — Candidates for Review"
        else:
            plot_df = df.sort_values("composite_score", ascending=False)
            title = f"All {len(df)} Branches"

        colors = [ACTION_COLORS.get(a, "#888") for a in plot_df["action"]]

        fig = go.Figure(go.Bar(
            x=plot_df["composite_score"],
            y=plot_df["name"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=plot_df["composite_score"].round(3),
            textposition="outside",
            textfont=dict(size=10, color="#888"),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(family="Syne", size=14, color="#F0F0F0")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                       color="#888", tickfont=dict(size=10)),
            yaxis=dict(color="#ccc", tickfont=dict(size=10), autorange="reversed"),
            margin=dict(l=10, r=60, t=40, b=10),
            height=max(400, len(plot_df) * 22),
            bargap=0.3,
        )
        st.plotly_chart(fig, use_container_width=True)



    with col2:
        # Action donut
        action_counts = df["action"].value_counts().reset_index()
        action_counts.columns = ["action", "count"]
        fig2 = go.Figure(go.Pie(
            labels=action_counts["action"],
            values=action_counts["count"],
            hole=0.65,
            marker=dict(colors=[ACTION_COLORS.get(a, "#888") for a in action_counts["action"]]),
            textinfo="percent",
            textfont=dict(size=11, color="white"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        ))
        fig2.add_annotation(
            text=f"<b>{len(df)}</b><br><span style='font-size:10px'>Branches</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color="#F0F0F0", family="Syne"),
        )
        fig2.update_layout(
            title=dict(text="Action Distribution", font=dict(family="Syne", size=14, color="#F0F0F0")),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(font=dict(color="#888", size=10), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=40, b=10),
            height=320,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Score histogram
        fig3 = go.Figure(go.Histogram(
            x=df["composite_score"],
            nbinsx=30,
            marker=dict(
                color=df["composite_score"],
                colorscale=[[0,"#8B0000"],[0.5,"#CC1B1B"],[1,"#22C55E"]],
                line=dict(width=0),
            ),
        ))
        fig3.update_layout(
            title=dict(text="Score Distribution", font=dict(family="Syne", size=14, color="#F0F0F0")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#888", tickfont=dict(size=10), gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(color="#888", tickfont=dict(size=10), gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=10, r=10, t=40, b=10),
            height=240,
            bargap=0.05,
        )
        st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════
#  TAB 3 — ML CLUSTERS
# ═══════════════════════════════════════════════════════════

def render_clusters_tab(df):
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:0.7rem; font-weight:700;
                color:#888; letter-spacing:0.2em; text-transform:uppercase;
                margin-bottom:0.8rem;">
         KMeans Cluster Analysis (ML)
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        if "pca_x" in df.columns:
            fig = go.Figure()
            for cl in df["cluster_label"].dropna().unique():
                sub = df[df["cluster_label"] == cl]
                fig.add_trace(go.Scatter(
                    x=sub["pca_x"], y=sub["pca_y"],
                    mode="markers",
                    name=cl,
                    marker=dict(
                        size=7,
                        color=get_cluster_color(cl),
                        opacity=0.8,
                        line=dict(width=0),
                    ),
                    hovertemplate="<b>%{text}</b><br>Score: %{customdata:.3f}<extra></extra>",
                    text=sub["name"],
                    customdata=sub["composite_score"],
                ))
            fig.update_layout(
                title=dict(text="PCA — Branch Clusters in 2D Space",
                           font=dict(family="Syne", size=14, color="#F0F0F0")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,15,1)",
                xaxis=dict(color="#555", showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                           zeroline=False, title=dict(text="PC1", font=dict(color="#555"))),
                yaxis=dict(color="#555", showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                           zeroline=False, title=dict(text="PC2", font=dict(color="#555"))),
                legend=dict(font=dict(color="#888", size=11), bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10, r=10, t=40, b=10),
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)



    with col2:
        # Radar chart per cluster
        feats = ["score_commercial","score_office","score_residential",
                 "score_accessibility","score_competition","score_cannibalization"]
        feat_labels = ["Commercial","Office","Residential","Accessibility","Competition","Cannibalization"]
        feats = [f for f in feats if f in df.columns]

        cluster_means = df.groupby("cluster_label")[feats].mean()

        def hex_to_rgba(hex_color, alpha=0.15):
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            return f"rgba({r},{g},{b},{alpha})"

        fig2 = go.Figure()
        for cl, row in cluster_means.iterrows():
            vals = list(row.values) + [row.values[0]]
            labs = feat_labels[:len(feats)] + [feat_labels[0]]
            base_color = get_cluster_color(cl)
            fig2.add_trace(go.Scatterpolar(
                r=vals, theta=labs,
                fill="toself",
                name=cl,
                line=dict(color=base_color, width=2),
                fillcolor=hex_to_rgba(base_color, 0.15),
                opacity=0.9,
            ))
        fig2.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,1], color="#555",
                                tickfont=dict(size=9, color="#555"),
                                gridcolor="rgba(255,255,255,0.06)"),
                angularaxis=dict(color="#888", tickfont=dict(size=10, color="#ccc"),
                                 gridcolor="rgba(255,255,255,0.06)"),
            ),
            title=dict(text="Cluster Profiles (Radar)",
                       font=dict(family="Syne", size=14, color="#F0F0F0")),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(color="#888", size=10), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=40, b=10),
            height=360,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Cluster sizes
        cl_sizes = df["cluster_label"].value_counts().reset_index()
        cl_sizes.columns = ["Cluster","Count"]
        fig3 = go.Figure(go.Bar(
            x=cl_sizes["Count"], y=cl_sizes["Cluster"],
            orientation="h",
            marker=dict(color=[get_cluster_color(c) for c in cl_sizes["Cluster"]]),
            text=cl_sizes["Count"],
            textposition="outside",
            textfont=dict(size=11, color="#888"),
        ))
        fig3.update_layout(
            title=dict(text="Branches per Cluster",
                       font=dict(family="Syne", size=13, color="#F0F0F0")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#555", gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(color="#ccc", tickfont=dict(size=10)),
            margin=dict(l=10, r=40, t=40, b=10),
            height=220,
            bargap=0.3,
        )
        st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════
#  TAB 4 — MERGER ANALYSIS
# ═══════════════════════════════════════════════════════════

def render_merger_tab(df):
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:0.7rem; font-weight:700;
                color:#888; letter-spacing:0.2em; text-transform:uppercase;
                margin-bottom:0.8rem;">
         Merger & Closure Recommendations
    </div>
    """, unsafe_allow_html=True)

    from scipy.spatial.distance import cdist

    coords = df[["latitude","longitude"]].values
    names  = df["name"].values
    scores = df["composite_score"].values

    # Find pairs within 2km
    col_thresh, col_info = st.columns([1, 3])
    with col_thresh:
        radius_km = st.slider("Detection radius (km)", 0.5, 5.0, 2.0, 0.5)
        radius_deg = radius_km * 0.009

    pairs = []
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            dist_deg = np.sqrt((coords[i,0]-coords[j,0])**2 + (coords[i,1]-coords[j,1])**2)
            dist_km = dist_deg / 0.009
            if dist_km <= radius_km:
                keep_idx = i if scores[i] >= scores[j] else j
                close_idx = j if keep_idx == i else i
                pairs.append({
                    "keep":        names[keep_idx],
                    "keep_score":  round(scores[keep_idx], 3),
                    "close":       names[close_idx],
                    "close_score": round(scores[close_idx], 3),
                    "distance_km": round(dist_km, 2),
                    "score_diff":  round(abs(scores[i]-scores[j]), 3),
                })

    pairs_df = pd.DataFrame(pairs).sort_values("distance_km") if pairs else pd.DataFrame()

    with col_info:
        c1,c2,c3 = st.columns(3)
        c1.metric(" Merger Pairs Found", len(pairs_df))
        c2.metric(" Branches to Close", len(pairs_df["close"].unique()) if not pairs_df.empty else 0)
        c3.metric(" Branches to Keep", len(pairs_df["keep"].unique()) if not pairs_df.empty else 0)

    if not pairs_df.empty:
        col_l, col_r = st.columns([2, 1])

        with col_l:
            st.markdown("**Merger Pair Details**")
            # Color-code the table
            def style_score(val):
                if isinstance(val, float):
                    if val >= 0.35: return "color: #22C55E"
                    elif val >= 0.2: return "color: #F97316"
                    else: return "color: #EF4444"
                return ""

            display_cols = ["keep","keep_score","close","close_score","distance_km","score_diff"]
            st.dataframe(
                pairs_df[display_cols].head(50),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "keep":        st.column_config.TextColumn("✅ Keep Branch", width="large"),
                    "keep_score":  st.column_config.NumberColumn("Score", format="%.3f"),
                    "close":       st.column_config.TextColumn("🔴 Close Branch", width="large"),
                    "close_score": st.column_config.NumberColumn("Score", format="%.3f"),
                    "distance_km": st.column_config.NumberColumn("Distance (km)", format="%.2f km"),
                    "score_diff":  st.column_config.ProgressColumn("Score Diff", min_value=0, max_value=0.5),
                },
            )

        with col_r:
            # Distance distribution
            fig = go.Figure(go.Histogram(
                x=pairs_df["distance_km"],
                nbinsx=20,
                marker=dict(color="#CC1B1B", line=dict(width=0)),
            ))
            fig.update_layout(
                title=dict(text="Distance Distribution", font=dict(family="Syne",size=13,color="#F0F0F0")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#888", title="km", gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(color="#888", gridcolor="rgba(255,255,255,0.04)"),
                margin=dict(l=10,r=10,t=40,b=10),
                height=220,
            )
            st.plotly_chart(fig, use_container_width=True)



            # Score diff distribution
            fig2 = go.Figure(go.Scatter(
                x=pairs_df["distance_km"],
                y=pairs_df["score_diff"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=pairs_df["score_diff"],
                    colorscale=[[0,"#CC1B1B"],[1,"#22C55E"]],
                    opacity=0.8,
                    line=dict(width=0),
                ),
                hovertemplate="Keep: %{text}<br>Distance: %{x:.2f}km<br>Score diff: %{y:.3f}<extra></extra>",
                text=pairs_df["keep"],
            ))
            fig2.update_layout(
                title=dict(text="Distance vs Score Gap", font=dict(family="Syne",size=13,color="#F0F0F0")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#888", title="Distance (km)", gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(color="#888", title="Score Diff", gridcolor="rgba(255,255,255,0.04)"),
                margin=dict(l=10,r=10,t=40,b=10),
                height=230,
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info(f"No pairs found within {radius_km}km. Try increasing the radius.")

# ═══════════════════════════════════════════════════════════
#  TAB 5 — CITY ANALYSIS
# ═══════════════════════════════════════════════════════════

def render_city_tab(df):
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:0.7rem; font-weight:700;
                color:#888; letter-spacing:0.2em; text-transform:uppercase;
                margin-bottom:0.8rem;">
         City-Level Intelligence
    </div>
    """, unsafe_allow_html=True)

    if "city" not in df.columns:
        st.warning("City column not available.")
        return

    city_stats = df.groupby("city").agg(
        total=("name","count"),
        avg_score=("composite_score","mean"),
        max_score=("composite_score","max"),
        min_score=("composite_score","min"),
        avg_commercial=("score_commercial","mean") if "score_commercial" in df.columns else ("composite_score","count"),
        avg_competition=("score_competition","mean") if "score_competition" in df.columns else ("composite_score","count"),
    ).reset_index().sort_values("total", ascending=False)

    city_stats["avg_score"]  = city_stats["avg_score"].round(3)
    city_stats["health"]     = city_stats["avg_score"].apply(
        lambda x: "🟢 Healthy" if x > 0.3 else "🟡 Moderate" if x > 0.2 else "🔴 At Risk"
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Branches",
            x=city_stats["city"].head(20),
            y=city_stats["total"].head(20),
            marker=dict(color="#CC1B1B", line=dict(width=0)),
            yaxis="y",
        ))
        fig.add_trace(go.Scatter(
            name="Avg Score",
            x=city_stats["city"].head(20),
            y=city_stats["avg_score"].head(20),
            mode="lines+markers",
            marker=dict(color="#D4A017", size=8),
            line=dict(color="#D4A017", width=2),
            yaxis="y2",
        ))
        fig.update_layout(
            title=dict(text="Branches & Avg Score by City",
                       font=dict(family="Syne",size=14,color="#F0F0F0")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#888", tickangle=-35, tickfont=dict(size=10),
                       gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(color="#CC1B1B", title="Branches", gridcolor="rgba(255,255,255,0.04)"),
            yaxis2=dict(color="#D4A017", title="Avg Score", overlaying="y", side="right"),
            legend=dict(font=dict(color="#888"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10,r=50,t=40,b=60),
            height=380,
            barmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)



    with col2:
        # Bubble chart: total branches vs avg score vs avg competition
        fig2 = go.Figure(go.Scatter(
            x=city_stats["avg_score"],
            y=city_stats["total"],
            mode="markers+text",
            text=city_stats["city"],
            textposition="top center",
            textfont=dict(size=9, color="#888"),
            marker=dict(
                size=city_stats["total"].clip(5, 50),
                color=city_stats["avg_score"],
                colorscale=[[0,"#8B0000"],[0.5,"#CC1B1B"],[1,"#22C55E"]],
                showscale=True,
                colorbar=dict(title=dict(text="Score", font=dict(color="#888")), tickfont=dict(color="#888")),
                line=dict(width=0),
            ),
            hovertemplate="<b>%{text}</b><br>Branches: %{y}<br>Score: %{x:.3f}<extra></extra>",
        ))
        fig2.update_layout(
            title=dict(text="City Portfolio Map (size = branch count)",
                       font=dict(family="Syne",size=14,color="#F0F0F0")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#888", title="Avg Composite Score",
                       gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(color="#888", title="Number of Branches",
                       gridcolor="rgba(255,255,255,0.04)"),
            margin=dict(l=10,r=10,t=40,b=10),
            height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # City health table
    st.markdown("**City Health Summary**")
    st.dataframe(
        city_stats[["city","total","avg_score","max_score","min_score","health"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "city":      st.column_config.TextColumn("City"),
            "total":     st.column_config.NumberColumn("Branches"),
            "avg_score": st.column_config.ProgressColumn("Avg Score", min_value=0, max_value=0.6),
            "max_score": st.column_config.NumberColumn("Max", format="%.3f"),
            "min_score": st.column_config.NumberColumn("Min", format="%.3f"),
            "health":    st.column_config.TextColumn("Health"),
        },
    )

# ═══════════════════════════════════════════════════════════
#  TAB 6 — BRANCH DEEP DIVE
# ═══════════════════════════════════════════════════════════

def render_deepdive_tab(df):
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:0.7rem; font-weight:700;
                color:#888; letter-spacing:0.2em; text-transform:uppercase;
                margin-bottom:0.8rem;">
         Branch Deep Dive
    </div>
    """, unsafe_allow_html=True)

    selected = st.selectbox(
        "Select a branch",
        df.sort_values("composite_score", ascending=False)["name"].tolist()
    )

    row = df[df["name"] == selected].iloc[0]
    action = row.get("action", "—")
    ac = ACTION_COLORS.get(action, "#888")

    # Header card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #111 0%, #1a0505 100%);
        border: 1px solid {ac};
        border-radius:14px; padding:1.5rem 2rem;
        display:flex; justify-content:space-between; align-items:center;
        margin-bottom:1rem;
    ">
        <div>
            <div style="font-family:'Syne',sans-serif; font-size:1.3rem;
                        font-weight:800; color:#F0F0F0;">{row['name']}</div>
            <div style="font-size:0.8rem; color:#888; margin-top:4px;">
                📍 {row.get('address','—')} &nbsp;|&nbsp;
                🏙️ {row.get('city','—')} &nbsp;|&nbsp;
                📞 {row.get('phone','—')}
            </div>
        </div>
        <div style="text-align:right;">
            <div style="font-family:'Syne',sans-serif; font-size:2.5rem;
                        font-weight:800; color:{ac};">
                {row.get('composite_score',0):.3f}
            </div>
            <div style="font-size:0.75rem; font-weight:600; color:{ac};
                        background:{ac}22; padding:3px 10px; border-radius:20px;
                        display:inline-block; margin-top:4px;">
                {action}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # Score breakdown radar
    with col1:
        feats = {
            "Commercial":     row.get("score_commercial", 0),
            "Office":         row.get("score_office", 0),
            "Residential":    row.get("score_residential", 0),
            "Accessibility":  row.get("score_accessibility", 0),
            "Competition":    row.get("score_competition", 0),
            "Cannibalization":row.get("score_cannibalization", 0),
        }
        vals = list(feats.values()) + [list(feats.values())[0]]
        labs = list(feats.keys()) + [list(feats.keys())[0]]
        h = ac.lstrip("#")
        ac_rgba = f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},0.15)"
        fig = go.Figure(go.Scatterpolar(
            r=vals, theta=labs, fill="toself",
            line=dict(color=ac, width=2),
            fillcolor=ac_rgba,
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,1], color="#555",
                                tickfont=dict(size=9,color="#555"),
                                gridcolor="rgba(255,255,255,0.06)"),
                angularaxis=dict(color="#ccc", tickfont=dict(size=10,color="#ccc"),
                                 gridcolor="rgba(255,255,255,0.06)"),
            ),
            title=dict(text="Score Profile", font=dict(family="Syne",size=13,color="#F0F0F0")),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20,r=20,t=40,b=20),
            height=280,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)



    with col2:
        # Raw counts bar
        raw = {
            "Commercial 1km":  row.get("commercial_1km", 0),
            "Office 1km":      row.get("office_1km", 0),
            "Residential 1km": row.get("residential_1km", 0),
            "Transit 1km":     row.get("transit_1km", 0),
            "Competitors 1km": row.get("competitor_1km", 0),
            "Own Branches 2km":row.get("own_branches_2km", 0),
        }
        fig2 = go.Figure(go.Bar(
            x=list(raw.values()),
            y=list(raw.keys()),
            orientation="h",
            marker=dict(color=["#3B82F6","#3B82F6","#3B82F6","#3B82F6","#EF4444","#F97316"],
                        line=dict(width=0)),
            text=list(raw.values()),
            textposition="outside",
            textfont=dict(size=10,color="#888"),
        ))
        fig2.update_layout(
            title=dict(text="POI Counts", font=dict(family="Syne",size=13,color="#F0F0F0")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#555", gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(color="#ccc", tickfont=dict(size=10), autorange="reversed"),
            margin=dict(l=10,r=40,t=40,b=10),
            height=280,
            bargap=0.3,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        # Gauge for composite score
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=row.get("composite_score", 0),
            delta={"reference": df["composite_score"].mean(), "valueformat":".3f"},
            number={"valueformat":".3f", "font":{"family":"Syne","size":36,"color":"#F0F0F0"}},
            gauge={
                "axis":{"range":[0, df["composite_score"].max()],
                        "tickcolor":"#555","tickfont":{"color":"#555","size":10}},
                "bar":{"color":ac,"thickness":0.25},
                "bgcolor":"rgba(0,0,0,0)",
                "borderwidth":0,
                "steps":[
                    {"range":[0, 0.20],"color":"#1a0505"},
                    {"range":[0.20, 0.35],"color":"#1a1205"},
                    {"range":[0.35, df["composite_score"].max()],"color":"#051a08"},
                ],
                "threshold":{"line":{"color":"#D4A017","width":3},
                             "thickness":0.75,"value":df["composite_score"].mean()},
            },
            title={"text":"Composite Score<br><span style='font-size:0.7rem;color:#888'>Gold line = avg</span>",
                   "font":{"family":"Syne","size":13,"color":"#F0F0F0"}},
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20,r=20,t=40,b=10),
            height=280,
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown(f"""
        <div style="background:var(--bg3); border:1px solid var(--border);
                    border-radius:10px; padding:1rem; font-size:0.8rem;">
            <div style="color:#888; margin-bottom:4px;">Cluster</div>
            <div style="color:{CLUSTER_COLORS.get(row.get('cluster_label',''),ac)};
                        font-family:'Syne',sans-serif; font-weight:700; font-size:1rem;">
                {row.get('cluster_label','—')}
            </div>
            <div style="color:#888; margin-top:8px; margin-bottom:4px;">DBSCAN Zone</div>
            <div style="color:#ccc;">
                {' Overcrowded Zone #'+str(int(row.get('dbscan_zone',0))) if row.get('is_overcrowded',False) else '✅ Isolated (healthy spacing)'}
            </div>
            <div style="color:#888; margin-top:8px; margin-bottom:4px;">Global Rank</div>
            <div style="color:#D4A017; font-family:'Syne',sans-serif; font-weight:700;">
                #{int(row.get('rank',0))} of {len(df)}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  TAB 7 — DATA TABLE
# ═══════════════════════════════════════════════════════════

def render_data_tab(df):
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:0.7rem; font-weight:700;
                color:#888; letter-spacing:0.2em; text-transform:uppercase;
                margin-bottom:0.8rem;">
         Full Data Table
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    search = col1.text_input(" Search branch name", "")
    sort_by = col2.selectbox("Sort by", ["composite_score","name","city","action"])
    asc = col3.checkbox("Ascending", False)

    display = df.copy()
    if search:
        display = display[display["name"].str.lower().str.contains(search.lower(), na=False)]
    display = display.sort_values(sort_by, ascending=asc)

    show_cols = ["rank","name","city","composite_score","action","cluster_label",
                 "score_commercial","score_office","score_residential",
                 "score_accessibility","score_competition","score_cannibalization",
                 "commercial_1km","competitor_1km","own_branches_2km"]
    show_cols = [c for c in show_cols if c in display.columns]

    st.dataframe(
        display[show_cols],
        use_container_width=True,
        hide_index=True,
        height=520,
        column_config={
            "rank":             st.column_config.NumberColumn("#", width="small"),
            "name":             st.column_config.TextColumn("Branch", width="large"),
            "city":             st.column_config.TextColumn("City", width="medium"),
            "composite_score":  st.column_config.ProgressColumn("Score", min_value=0, max_value=0.6, format="%.3f"),
            "action":           st.column_config.TextColumn("Action", width="large"),
            "cluster_label":    st.column_config.TextColumn("Cluster", width="medium"),
            "score_commercial": st.column_config.ProgressColumn("Commercial", min_value=0, max_value=1, format="%.2f"),
            "score_office":     st.column_config.ProgressColumn("Office", min_value=0, max_value=1, format="%.2f"),
            "score_residential":st.column_config.ProgressColumn("Residential", min_value=0, max_value=1, format="%.2f"),
            "score_accessibility":st.column_config.ProgressColumn("Accessibility",min_value=0,max_value=1,format="%.2f"),
            "score_competition":st.column_config.ProgressColumn("Competition",min_value=0,max_value=1,format="%.2f"),
            "score_cannibalization":st.column_config.ProgressColumn("Cannibalization",min_value=0,max_value=1,format="%.2f"),
            "commercial_1km":   st.column_config.NumberColumn("Comm. POIs"),
            "competitor_1km":   st.column_config.NumberColumn("Competitors"),
            "own_branches_2km": st.column_config.NumberColumn("Own 2km"),
        },
    )

    st.markdown(f"**{len(display):,} branches** | ")
    csv_data = display[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        csv_data,
        file_name="filtered_branches.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════
#  TAB 8 — RUN SITE SELECTION (Interactive Model Runner)
# ═══════════════════════════════════════════════════════════

def render_run_site_selection_tab(df):
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:0.7rem; font-weight:700;
                color:#888; letter-spacing:0.2em; text-transform:uppercase;
                margin-bottom:0.8rem;">
        Run Site Selection Model — Upload Data & Generate Recommendations
    </div>
    """, unsafe_allow_html=True)

    # Session state for results
    if 'site_selection_results' not in st.session_state:
        st.session_state.site_selection_results = None
    if 'site_selection_candidates' not in st.session_state:
        st.session_state.site_selection_candidates = None
    if 'site_selection_metrics' not in st.session_state:
        st.session_state.site_selection_metrics = None

    # ── File Upload Section ─────────────────────────────────────
    st.markdown("#### Step 1: Upload Required Data Files")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Branch Scores**")
        branches_file = st.file_uploader(
            "Upload branches_scored.csv",
            type=["csv"],
            help="Scored branch data with composite_score and location features",
            key="upload_branches"
        )
        
    with col2:
        st.markdown("**Branch Performance**")
        performance_file = st.file_uploader(
            "Upload branch_performance.csv",
            type=["csv"],
            help="Branch performance data with monthly_revenue_mad",
            key="upload_performance"
        )
        
    with col3:
        st.markdown("**POI Data**")
        poi_file = st.file_uploader(
            "Upload POI data (merged_data_clean.csv)",
            type=["csv"],
            help="Merged POI data from all sources",
            key="upload_poi"
        )

    with col4:
        st.markdown("**Population Data**")
        density_file = st.file_uploader(
            "Upload density.csv",
            type=["csv"],
            help="Population density data per city",
            key="upload_density"
        )

    # ── Model Parameters ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Step 2: Configure Model Parameters")
    
    col_params1, col_params2, col_params3, col_params4, col_params5 = st.columns(5)
    
    with col_params1:
        n_clusters = st.slider("Number of Clusters", 5, 12, 8, 1,
                              help="Number of KMeans clusters for branch grouping")
    with col_params2:
        poi_weight = st.slider("POI Weight", 0.0, 1.0, 0.35, 0.05,
                              help="Weight for POI score in combined score")
    with col_params3:
        pop_weight = st.slider("Pop Weight", 0.0, 1.0, 0.30, 0.05,
                              help="Weight for Population Density score")
    with col_params4:
        min_poi = st.slider("Min POI Required", 3, 10, 5, 1,
                           help="Minimum POIs within 2km to consider a location")
    with col_params5:
        grid_spacing = st.selectbox("Grid Spacing (km)", [0.5, 0.7, 1.0], index=1,
                                   help="Grid point spacing")

    similarity_weight = max(0.0, round(1.0 - poi_weight - pop_weight, 2))
    st.caption(f"Similarity Weight: {similarity_weight} (auto-calculated to balance 1.0)")

    # ── Geographic Diversity Parameters ─────────────────────────
    st.markdown("---")
    st.markdown("#### Step 2b: Geographic Diversity Constraints")
    
    col_geo1, col_geo2, col_geo3 = st.columns(3)
    
    with col_geo1:
        enable_city_limit = st.checkbox("Enable Branch Limit Per City", value=True,
                                       help="Limit the number of branches per city for geographic diversity")
    
    with col_geo2:
        if enable_city_limit:
            max_per_city = st.slider("Max Branches Per City", 1, 5, 2, 1,
                                    help="Maximum number of branches allowed per city")
        else:
            max_per_city = 999
            st.info("No city limit — all top sites selected")
    
    with col_geo3:
        min_distance_km = st.slider("Min Distance Between Sites (km)", 0.3, 2.0, 0.7, 0.1,
                                   help="Minimum distance between any two selected sites")

    # ── Run Model Button ─────────────────────────────────────────
    st.markdown("---")
    run_col1, run_col2 = st.columns([3, 1])
    
    with run_col1:
        st.markdown("""
        <div style="background:var(--bg3); border:1px solid var(--border); border-radius:12px;
                    padding:1rem; font-size:0.8rem; color:#aaa; line-height:1.6;">
            <b style="color:#F0F0F0;">What will happen:</b><br>
            1. Merge branch scores with performance data<br>
            2. Cluster branches by location profile<br>
            3. Identify top 10% performers per cluster<br>
            4. Score grid points across Morocco using POI density<br>
            5. Compute cosine similarity against top performers<br>
            6. Rank and select optimal new branch locations<br>
            7. Apply geographic diversity constraints (city limits & minimum distance)
        </div>
        """, unsafe_allow_html=True)
    
    with run_col2:
        run_button = st.button(
            "Run Site Selection Model",
            type="primary",
            use_container_width=True
        )

    # ── Model Execution ──────────────────────────────────────────
    if run_button:
        if not branches_file or not performance_file or not poi_file or not density_file:
            st.error("Please upload all four required files before running the model.")
            return
        
        with st.spinner("Running site selection model... This may take 2-5 minutes."):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Load uploaded data
                status_text.text("Loading uploaded data...")
                progress_bar.progress(10)
                
                scores = pd.read_csv(branches_file)
                perf = pd.read_csv(performance_file)
                poi_df = pd.read_csv(poi_file, low_memory=False)
                density_df = pd.read_csv(density_file)
                
                scores.columns = scores.columns.str.strip().str.lower()
                perf.columns = perf.columns.str.strip().str.lower()
                poi_df.columns = poi_df.columns.str.strip().str.lower()
                
                status_text.text("Processing Population Density data...")
                density_valid = density_df[["lat", "lng", "population"]].dropna()
                pop_coords = density_valid[["lat", "lng"]].values
                pop_tree = cKDTree(pop_coords)
                pop_densities = density_valid["population"].values
                
                progress_bar.progress(15)
                
                # Merge data
                status_text.text("Merging branch data...")
                overlap = [c for c in perf.columns if c in scores.columns and c != "name"]
                if overlap:
                    perf = perf.drop(columns=overlap)
                
                df_model = scores.merge(perf, on="name", how="inner")
                df_model = df_model.drop_duplicates(subset=["name"]).reset_index(drop=True)
                
                progress_bar.progress(20)
                
                # Ensure required columns
                poi_cols = ["commercial_1km","office_1km","residential_1km",
                           "transit_1km","competitor_1km","own_branches_2km"]
                score_cols = ["score_commercial","score_office","score_residential",
                             "score_accessibility","score_competition","score_cannibalization"]
                
                for col in poi_cols + score_cols:
                    if col not in df_model.columns:
                        df_model[col] = 0
                
                numeric_cols = [c for c in df_model.columns if pd.api.types.is_numeric_dtype(df_model[c])]
                df_clean = df_model[numeric_cols + ["name"] if "name" in df_model.columns else numeric_cols].copy()
                df_clean = df_clean.dropna()
                
                if "name" not in df_clean.columns and "name" in df_model.columns:
                    df_clean["name"] = df_model["name"].values[:len(df_clean)]
                    
                # Add population density to existing branches for model training
                status_text.text("Attaching population density to branches...")
                df_clean["population_density"] = [
                    float(pop_densities[pop_tree.query([r["latitude"], r["longitude"]])[1]])
                    for _, r in df_clean.iterrows()
                ]
                
                target_revenue = "monthly_revenue_mad"
                if target_revenue not in df_clean.columns:
                    st.error(f"Column '{target_revenue}' not found in performance data.")
                    return
                
                progress_bar.progress(25)
                
                # Cluster branches
                status_text.text("Clustering branches...")
                cluster_features = [c for c in score_cols if c in df_clean.columns]
                X_cluster = df_clean[cluster_features].values
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df_clean["location_cluster"] = kmeans.fit_predict(X_cluster)
                unique_labels = np.unique(kmeans.labels_)
                
                progress_bar.progress(35)
                
                # Identify top 10% performers
                status_text.text("Identifying top performers...")
                benchmark_rows = []
                for cluster_id in unique_labels:
                    cluster_df = df_clean[df_clean["location_cluster"] == cluster_id].copy()
                    if len(cluster_df) == 0:
                        continue
                    n_top = max(1, int(len(cluster_df) * 0.10))
                    top_branches = cluster_df.nlargest(n_top, target_revenue)
                    for _, row in top_branches.iterrows():
                        entry = {
                            "cluster": cluster_id,
                            "revenue": row[target_revenue],
                            "lat": row["latitude"],
                            "lon": row["longitude"],
                        }
                        for col in cluster_features:
                            entry[col] = row[col]
                        benchmark_rows.append(entry)
                
                benchmark_df = pd.DataFrame(benchmark_rows)
                progress_bar.progress(40)
                
                # Process POI data
                status_text.text("Processing POI data...")
                plat = next((c for c in poi_df.columns if "lat" in c), None)
                plon = next((c for c in poi_df.columns if "lon" in c), None)
                pcat = next((c for c in poi_df.columns if "cat" in c or "category" in c), None)
                
                poi_df = poi_df.rename(columns={plat:"latitude", plon:"longitude", pcat:"category"})
                poi_valid = poi_df.dropna(subset=["latitude","longitude"])
                poi_valid = poi_valid[poi_valid["category"].notna()].reset_index(drop=True)
                
                # Category classification (reuse from site_selection2.py)
                COMMERCIAL_CATS = {
                    "restaurant","cafe","coffee_shop","fast_food_restaurant","bakery","pizzeria",
                    "boulangerie","patisserie","sandwich_shop","casual_eatery","bar","pub",
                    "bar_and_grill","buffet_restaurant","breakfast_restaurant","juice_bar",
                    "food_or_beverage_store","eating_drinking_location","shop","retail_location",
                    "specialty_store","clothing_store","fashion_or_apparel_store","jewelry_store",
                    "shoe_store","gift_shop","electronics_store","boutique","magasin","souk",
                    "marche","mall","shopping_mall","supermarket","grocery_store","convenience_store",
                    "food_bank","wholesale","wholesaler","department_store",
                    "hardware_store","furniture","home_improvement_center",
                }
                OFFICE_CATS = {
                    "office","corporate_or_business_office","bureau","b2b_service",
                    "b2b_supplier_distributor","financial_service","insurance_agency",
                    "real_estate_agency","real_estate_service","attorney_or_law_firm",
                    "legal_service","accountant_or_bookkeeper","engineering_service",
                    "software_development","information_technology_service",
                    "business_management_service","consulting","company","ngo",
                    "advertising_agency","human_resource_service","travel_service",
                    "travel_agency","import_export_company","telecommunications_service",
                    "government_office","diplomatic","association","construction_company",
                }
                RESIDENTIAL_CATS = {
                    "mosque","muslim_place_of_worship","place_of_worship","church",
                    "pharmacy","school","elementary_school","high_school","middle_school",
                    "college_university","preschool","kindergarten","university",
                    "hospital","clinic_or_treatment_center","doctors_office","healthcare_location",
                    "dental_care","eye_care","supermarket","grocery_store","epicerie",
                    "hammam","salon","hair_salon","beauty_salon","barber_shop","coiffeur",
                    "laundromat","dry_cleaning_service","post_office","community_center",
                    "social_or_community_service","neighbourhood","suburb","residential",
                }
                TRANSIT_CATS = {
                    "bus_stop","bus_station","train_station","stop_position","taxi",
                    "taxi_service","taxi_or_ride_share_service","ferry_terminal",
                    "airport","aerodrome","transportation_location","parking","station",
                    "public_transport","tram_stop",
                }
                COMPETITOR_KW = ["bmce","bmci","cih","credit agricole","banque populaire",
                                "societe generale","al barid","barid bank","cdm","credor",
                                "wafacash","cash plus","western union","money transfer",
                                "bureau_de_change","credit_union","loan_provider"]
                OWN_BANK_KW = ["attijariwafa","attijari wafa","wafa","attijariwafa bank"]
                
                def classify_category(cat_str):
                    c = str(cat_str).lower().strip()
                    for kw in COMPETITOR_KW:
                        if kw in c: return "competitor_bank"
                    for kw in OWN_BANK_KW:
                        if kw in c: return "own_branch"
                    for cat in COMMERCIAL_CATS:
                        if cat in c: return "commercial"
                    for cat in OFFICE_CATS:
                        if cat in c: return "office"
                    for cat in RESIDENTIAL_CATS:
                        if cat in c: return "residential"
                    for cat in TRANSIT_CATS:
                        if cat in c: return "transit"
                    return "other"
                
                poi_valid["category_type"] = poi_valid["category"].apply(classify_category)
                cat_types = ["commercial","office","residential","transit","competitor_bank","own_branch"]
                poi_coords = poi_valid[["latitude","longitude"]].values
                poi_tree = cKDTree(poi_coords)
                
                progress_bar.progress(50)
                
                # Coastline validation
                def strict_coastline_validation(lat, lon, buffer_km=5.0):
                    if not (21.0 <= lat <= 35.9): return False
                    if not (-17.0 <= lon <= -0.5): return False
                    if lat >= 35.5:   max_west = -5.75
                    elif lat >= 34.5: max_west = -6.75
                    elif lat >= 33.5: max_west = -7.75
                    elif lat >= 32.5: max_west = -8.75
                    elif lat >= 31.5: max_west = -9.65
                    elif lat >= 30.0: max_west = -10.0
                    elif lat >= 29.0: max_west = -10.5
                    elif lat >= 28.0: max_west = -11.5
                    elif lat >= 27.0: max_west = -13.0
                    else:             max_west = -15.5
                    buffer_deg = buffer_km / 111.0
                    if lon < max_west - buffer_deg: return False
                    if lat > 32.0 and lon > -1.5: return False
                    if lat > 35.75 and lon < -5.70: return False
                    if lat < 27.0 and lon < -14.0: return False
                    return True
                
                # Create grid points
                status_text.text(f"Creating grid points ({grid_spacing}km spacing)...")
                DEG_PER_KM = 0.009
                CELL_DEG = grid_spacing * DEG_PER_KM
                
                grid_points = []
                for lat in np.arange(21.0 + CELL_DEG/2, 35.9 - CELL_DEG/2, CELL_DEG):
                    for lon in np.arange(-17.0 + CELL_DEG/2, -0.5 + CELL_DEG/2, CELL_DEG):
                        if strict_coastline_validation(lat, lon, buffer_km=5.0):
                            grid_points.append([lat, lon])
                
                grid_points = np.array(grid_points)
                
                # Filter near existing branches
                existing_coords = df_clean[["latitude","longitude"]].values
                existing_tree = cKDTree(existing_coords)
                R_0_7KM = 0.0063
                
                valid_mask = np.array([
                    float(existing_tree.query(pt, k=1)[0]) > R_0_7KM
                    for pt in grid_points
                ])
                grid_points = grid_points[valid_mask]
                
                progress_bar.progress(60)
                
                # Score grid points
                status_text.text("Scoring grid points with POI density...")
                SEARCH_RADIUS_DEG = 2.0 * DEG_PER_KM
                
                grid_results = []
                total_pts = len(grid_points)
                t0 = time.time()
                
                for i, pt in enumerate(grid_points):
                    if i % 2000 == 0:
                        elapsed = time.time() - t0
                        pct = i / max(total_pts, 1) * 100
                        progress_bar.progress(60 + int(pct * 0.25))
                        status_text.text(f"Scoring grid points... {i:,}/{total_pts:,} ({pct:.1f}%)")
                    
                    lat, lon = pt[0], pt[1]
                    idxs = poi_tree.query_ball_point([lat, lon], SEARCH_RADIUS_DEG)
                    
                    if len(idxs) < min_poi:
                        continue
                    
                    cat_counts = {ct: 0 for ct in cat_types}
                    for idx in idxs:
                        ct = poi_valid.iloc[idx]["category_type"]
                        if ct in cat_counts:
                            cat_counts[ct] += 1
                    
                    meaningful = (cat_counts["commercial"] + cat_counts["office"] +
                                cat_counts["residential"] + cat_counts["transit"])
                    if meaningful < 2:
                        continue
                    
                    score = 0.0
                    poi_weights = {
                        "commercial": 1.5,
                        "office": 2.0,
                        "residential": 0.8,
                        "transit": 1.2,
                        "competitor_bank": 0.3,
                        "own_branch": 0.2,
                    }
                    for idx in idxs:
                        ct = poi_valid.iloc[idx]["category_type"]
                        d_lat = abs(lat - poi_coords[idx][0])
                        d_lon = abs(lon - poi_coords[idx][1])
                        d = np.sqrt(d_lat**2 + d_lon**2)
                        weight = 1.0 / (d + 0.001)
                        score += weight * poi_weights.get(ct, 0.0)
                    
                    if len(idxs) > 20:
                        score *= 0.7
                    
                    total_active = meaningful + 1
                    
                    res_dict = {
                        "lat": lat,
                        "lon": lon,
                        "poi_score": score,
                        "poi_count": len(idxs),
                        "commercial_1km": cat_counts["commercial"],
                        "office_1km": cat_counts["office"],
                        "residential_1km": cat_counts["residential"],
                        "transit_1km": cat_counts["transit"],
                        "competitor_1km": cat_counts["competitor_bank"],
                        "own_branches_2km": cat_counts["own_branch"],
                        "score_commercial": float(cat_counts["commercial"]) / total_active,
                        "score_office": float(cat_counts["office"]) / total_active,
                        "score_residential": float(cat_counts["residential"]) / total_active,
                        "score_accessibility": float(cat_counts["transit"]) / total_active,
                        "score_competition": float(cat_counts["competitor_bank"]) / total_active,
                        "score_cannibalization": float(cat_counts["own_branch"]) / total_active,
                    }
                    
                    # Population Density
                    _, pop_idx = pop_tree.query([lat, lon], k=1)
                    res_dict["population_density"] = float(pop_densities[pop_idx])
                    
                    grid_results.append(res_dict)
                
                candidate_df = pd.DataFrame(grid_results)
                if len(candidate_df) == 0:
                    st.error("No candidates remain after filtering. Try adjusting parameters.")
                    return
                
                progress_bar.progress(75)
                
                # Cosine similarity
                status_text.text("Computing cosine similarity...")
                site_profiles = np.column_stack([
                    candidate_df["score_commercial"].values.astype(float),
                    candidate_df["score_office"].values.astype(float),
                    candidate_df["score_residential"].values.astype(float),
                    candidate_df["score_accessibility"].values.astype(float),
                    candidate_df["score_competition"].values.astype(float),
                    candidate_df["score_cannibalization"].values.astype(float),
                ])
                
                bench_profiles = np.column_stack([
                    benchmark_df["score_commercial"].values.astype(float),
                    benchmark_df["score_office"].values.astype(float),
                    benchmark_df["score_residential"].values.astype(float),
                    benchmark_df["score_accessibility"].values.astype(float),
                    benchmark_df["score_competition"].values.astype(float),
                    benchmark_df["score_cannibalization"].values.astype(float),
                ])
                bench_revenues = benchmark_df["revenue"].values.astype(float)
                
                site_norms = np.linalg.norm(site_profiles, axis=1, keepdims=True)
                site_norms = np.where(site_norms < 1e-10, 1.0, site_norms)
                site_profiles_norm = site_profiles / site_norms
                
                bench_norms = np.linalg.norm(bench_profiles, axis=1, keepdims=True)
                bench_norms = np.where(bench_norms < 1e-10, 1.0, bench_norms)
                bench_profiles_norm = bench_profiles / bench_norms
                
                status_text.text("Training RandomForestRegressor for revenue prediction...")
                rf_features = ["score_commercial", "score_office", "score_residential", 
                               "score_accessibility", "score_competition", "score_cannibalization", "population_density"]
                
                X_train = df_clean[rf_features].values
                y_train = df_clean[target_revenue].values
                
                q_lo, q_hi = np.percentile(y_train, [1, 99])
                keep = (y_train >= q_lo) & (y_train <= q_hi)
                X_train_clean = X_train[keep]
                y_train_clean = y_train[keep]
                
                rf_scaler = StandardScaler()
                X_train_sc = rf_scaler.fit_transform(X_train_clean)
                
                rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
                rf_model.fit(X_train_sc, y_train_clean)
                
                status_text.text("Predicting candidate revenues...")
                X_cand = candidate_df[rf_features].values
                X_cand_sc = rf_scaler.transform(X_cand)
                candidate_df["predicted_revenue_mad"] = rf_model.predict(X_cand_sc).round().astype(int)
                
                BATCH_SIZE = 50000
                all_best_sims = []
                all_best_revenues = []
                processed = 0
                total_sites = len(candidate_df)
                t0 = time.time()
                
                while processed < total_sites:
                    end_idx = min(processed + BATCH_SIZE, total_sites)
                    batch_sites = site_profiles_norm[processed:end_idx]
                    batch_sims = np.dot(batch_sites, bench_profiles_norm.T)
                    batch_sims = np.nan_to_num(batch_sims, nan=0.0)
                    batch_sims = np.clip(batch_sims, 0.0, 1.0)
                    
                    k_neighbors = min(5, bench_profiles.shape[0])
                    batch_top_k_idx = np.argpartition(batch_sims, -k_neighbors, axis=1)[:, -k_neighbors:]
                    batch_top_k_sims = np.take_along_axis(batch_sims, batch_top_k_idx, axis=1)
                    batch_top_k_rev = bench_revenues[batch_top_k_idx]
                    
                    best_sim_per_site = batch_top_k_sims.max(axis=1)
                    best_idx_per_site = batch_top_k_sims.argmax(axis=1)
                    
                    all_best_sims.extend(best_sim_per_site)
                    all_best_revenues.extend(batch_top_k_rev[np.arange(len(batch_sims)), best_idx_per_site])
                    
                    processed = end_idx
                    pct = processed / total_sites * 100
                    progress_bar.progress(75 + int(pct * 0.15))
                    status_text.text(f"Computing similarity... {processed:,}/{total_sites:,} ({pct:.1f}%)")
                
                candidate_df["similarity_to_best"] = all_best_sims
                candidate_df["best_match_revenue"] = [round(r) for r in all_best_revenues]
                
                progress_bar.progress(90)
                
                # Combined score
                status_text.text("Computing combined scores...")
                poi_min = candidate_df["poi_score"].min()
                poi_max = candidate_df["poi_score"].max()
                if poi_max > poi_min:
                    candidate_df["poi_score_norm"] = (candidate_df["poi_score"] - poi_min) / (poi_max - poi_min)
                else:
                    candidate_df["poi_score_norm"] = 1.0
                    
                pop_min = candidate_df["population_density"].min()
                pop_max = candidate_df["population_density"].max()
                if pop_max > pop_min:
                    candidate_df["pop_score_norm"] = (candidate_df["population_density"] - pop_min) / (pop_max - pop_min)
                else:
                    candidate_df["pop_score_norm"] = 1.0
                
                candidate_df["combined_score"] = (
                    poi_weight * candidate_df["poi_score_norm"] +
                    similarity_weight * candidate_df["similarity_to_best"] +
                    pop_weight * candidate_df["pop_score_norm"]
                )
                
                candidate_df = candidate_df.sort_values("combined_score", ascending=False).reset_index(drop=True)
                candidate_df["rank_overall"] = candidate_df.index + 1
                
                # Assign nearest city
                CITIES = {
                    "Casablanca": (33.5731, -7.5898), "Rabat": (34.0209, -6.8416),
                    "Tanger": (35.7595, -5.8340), "Fes": (34.0533, -5.0033),
                    "Meknes": (33.8935, -5.5473), "Marrakech": (31.6295, -7.9811),
                    "Agadir": (30.4278, -9.5981), "Oujda": (34.6867, -1.9114),
                    "Nador": (35.1688, -3.9285), "Beni Mellal": (32.3373, -6.3498),
                    "Safi": (32.2994, -9.2372), "El Jadida": (33.2561, -8.5059),
                    "Kenitra": (34.2936, -6.5498), "Tetouan": (35.5785, -5.3684),
                    "Settat": (33.0000, -7.6167), "Taza": (34.2247, -4.0083),
                    "Ouarzazate": (31.9310, -5.7246), "Laayoune": (27.1536, -13.2034),
                    "Mohammedia": (33.5503, -7.6667), "Sale": (34.0370, -6.8326),
                    "Taroudant": (30.4746, -8.8658), "Berkane": (34.9274, -2.3309),
                    "Inezgane": (30.3600, -9.5300), "Khemisset": (33.8333, -6.0667),
                }
                city_tree = cKDTree(list(CITIES.values()))
                city_names_list = list(CITIES.keys())
                candidate_df["nearest_city"] = [
                    city_names_list[city_tree.query([r["lat"], r["lon"]])[1]]
                    for _, r in candidate_df.iterrows()
                ]
                
                # Select top 20 with geographic diversity
                status_text.text("Selecting top 20 sites...")
                selected_indices = []
                selected_coords = []
                city_counts = {}
                
                # Use the user-configurable parameters
                R_MIN_DISTANCE = min_distance_km * DEG_PER_KM
                
                for idx, row in candidate_df.iterrows():
                    if len(selected_indices) >= 20:
                        break
                    
                    city = row["nearest_city"]
                    # Use user-configurable city limit
                    if city_counts.get(city, 0) >= max_per_city:
                        continue
                    
                    pt = [row["lat"], row["lon"]]
                    too_close = any(
                        np.sqrt((pt[0] - sc[0])**2 + (pt[1] - sc[1])**2) < R_MIN_DISTANCE
                        for sc in selected_coords
                    )
                    if too_close:
                        continue
                    
                    selected_indices.append(idx)
                    selected_coords.append(pt)
                    city_counts[city] = city_counts.get(city, 0) + 1
                
                top20 = candidate_df.loc[selected_indices].reset_index(drop=True)
                top20["rank"] = range(1, len(top20) + 1)
                top20["recommendation"] = "Open New Branch"
                
                # Save to session state
                st.session_state.site_selection_results = top20
                st.session_state.site_selection_candidates = candidate_df
                st.session_state.site_selection_metrics = {
                    "method": "Cluster-Based Benchmark KNN (Top 10% Only)",
                    "clusters_used": int(n_clusters),
                    "benchmarks_used": int(len(benchmark_df)),
                    "total_candidates": int(len(candidate_df)),
                    "poi_weight": float(poi_weight),
                    "pop_weight": float(pop_weight),
                    "similarity_weight": float(similarity_weight),
                    "min_poi_required": int(min_poi),
                    "grid_spacing_km": float(grid_spacing),
                    "enable_city_limit": enable_city_limit,
                    "max_per_city": int(max_per_city) if enable_city_limit else "Unlimited",
                    "min_distance_km": float(min_distance_km),
                    "selected_sites": len(top20),
                    "top_site_revenue": int(top20.iloc[0]["predicted_revenue_mad"]),
                    "top_site_city": top20.iloc[0]["nearest_city"],
                    "top_site_lat": float(top20.iloc[0]["lat"]),
                    "top_site_lon": float(top20.iloc[0]["lon"]),
                    "top_site_combined": float(top20.iloc[0]["combined_score"]),
                    "top_site_similarity": float(top20.iloc[0]["similarity_to_best"]),
                }
                
                progress_bar.progress(100)
                status_text.text("Model complete!")
                
                st.success(f"Site selection complete! Top 20 sites identified from {len(candidate_df):,} candidates.")
                
            except Exception as e:
                st.error(f"Error running model: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return

    # ── Display Results ──────────────────────────────────────────
    if st.session_state.site_selection_results is not None:
        top20 = st.session_state.site_selection_results
        candidates = st.session_state.site_selection_candidates
        metrics = st.session_state.site_selection_metrics
        
        st.markdown("---")
        st.markdown("#### Step 3: View Results")
        
        # Model KPIs
        st.markdown("#### Model Overview")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Method", metrics.get("method", "Cluster Benchmark"))
        c2.metric("Clusters", str(metrics.get("clusters_used", "—")))
        c3.metric("Candidates", f"{metrics.get('total_candidates', 0):,}")
        c4.metric("Top Revenue", f"{metrics.get('top_site_revenue', 0):,} MAD")
        c5.metric("Selected Sites", str(metrics.get("selected_sites", len(top20))))
        
        # Geographic Diversity Settings
        st.markdown("#### Geographic Diversity Settings")
        geo_c1, geo_c2, geo_c3 = st.columns(3)
        geo_c1.metric("City Limit Enabled", "Yes" if metrics.get("enable_city_limit", False) else "No")
        geo_c2.metric("Max Per City", str(metrics.get("max_per_city", "2")))
        geo_c3.metric("Min Distance", f"{metrics.get('min_distance_km', 0.7):.1f} km")
        
        st.markdown("---")
        
        # Map and Charts
        col_map, col_info = st.columns([3, 1])
        
        with col_map:
            st.markdown("#### Top 20 Recommended Locations")
            m = folium.Map(location=[31.5, -6.5], zoom_start=6, tiles="OpenStreetMap", prefer_canvas=True)
            
            # Existing branches (grey)
            for _, row in df.head(300).iterrows():
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=3, color="#555", fill=True,
                    fill_opacity=0.5, weight=0,
                    tooltip=row["name"],
                ).add_to(m)
            
            # New sites (orange markers)
            for _, row in top20.iterrows():
                city = row.get("nearest_city", "")
                rev = int(row["predicted_revenue_mad"])
                sim = f"{row['similarity_to_best']:.3f}" if "similarity_to_best" in row else "—"
                popup_html = (
                    f"<b>#{int(row['rank'])} — {city}</b><br>"
                    f"Lat: {row['lat']:.4f}<br>"
                    f"Lon: {row['lon']:.4f}<br>"
                    f"Revenue: {rev:,} MAD/mo<br>"
                    f"Combined Score: {row['combined_score']:.4f}<br>"
                    f"Similarity: {sim}"
                )
                folium.Marker(
                    location=[row["lat"], row["lon"]],
                    tooltip=f"#{int(row['rank'])} {city} — {rev:,} MAD/mo",
                    popup=folium.Popup(popup_html, max_width=220),
                    icon=folium.Icon(color="orange", icon="star", prefix="fa"),
                ).add_to(m)
            
            st_folium(m, width="100%", height=480, returned_objects=[])
        
        with col_info:
            st.markdown("**Top 5 Sites**")
            for _, row in top20.head(5).iterrows():
                city = row.get("nearest_city", "—")
                score = f"{row['combined_score']:.4f}" if "combined_score" in row else "—"
                st.markdown(f"""
                <div style="background:var(--bg4); border-radius:8px; padding:0.6rem 0.8rem;
                            margin-bottom:0.4rem; border-left:3px solid #D4A017;">
                    <div style="font-family:'Syne',sans-serif; font-size:0.8rem;
                                color:#D4A017;">#{int(row['rank'])} — {city}</div>
                    <div style="font-size:0.72rem; color:#aaa;">
                        {row['lat']:.4f}, {row['lon']:.4f}
                    </div>
                    <div style="font-size:0.75rem; color:#22C55E;">
                        {int(row['predicted_revenue_mad']):,} MAD/mo
                    </div>
                    <div style="font-size:0.7rem; color:#888;">
                        Score: {score}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Charts
        st.markdown("---")
        st.markdown("#### Analytics")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Revenue bar chart
            fig1 = go.Figure(go.Bar(
                x=top20["predicted_revenue_mad"],
                y=[f"#{int(r)} {c}" for r, c in zip(top20["rank"], top20["nearest_city"])],
                orientation="h",
                marker=dict(
                    color=top20["predicted_revenue_mad"],
                    colorscale=[[0,"#8B0000"],[0.5,"#CC1B1B"],[1,"#22C55E"]],
                    line=dict(width=0),
                ),
                text=top20["predicted_revenue_mad"].apply(lambda x: f"{int(x):,}"),
                textposition="outside",
                textfont=dict(size=10, color="#888"),
            ))
            fig1.update_layout(
                title=dict(text="Top 20 Sites — Predicted Revenue", font=dict(family="Syne", size=14, color="#F0F0F0")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#555", gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(color="#ccc", tickfont=dict(size=10), autorange="reversed"),
                margin=dict(l=10, r=80, t=40, b=10),
                height=520,
                bargap=0.25,
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col_chart2:
            # Similarity scores
            fig2 = go.Figure(go.Bar(
                x=[f"#{int(r)}" for r in top20["rank"]],
                y=top20["similarity_to_best"],
                marker=dict(
                    color=top20["similarity_to_best"],
                    colorscale=[[0,"#8B0000"],[1,"#22C55E"]],
                    line=dict(width=0),
                ),
            ))
            fig2.update_layout(
                title=dict(text="Similarity to Best Benchmark", font=dict(family="Syne", size=14, color="#F0F0F0")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#888"),
                yaxis=dict(color="#888", gridcolor="rgba(255,255,255,0.04)", range=[0, 1]),
                margin=dict(l=10, r=10, t=40, b=10),
                height=320,
                bargap=0.2,
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # POI distribution pie
            poi_count_cols = {
                "Commercial": "commercial_1km",
                "Office": "office_1km",
                "Residential": "residential_1km",
                "Transit": "transit_1km",
                "Competitor": "competitor_1km",
            }
            available = {k: v for k, v in poi_count_cols.items() if v in top20.columns}
            if available:
                totals = {k: top20[v].sum() for k, v in available.items()}
                poi_colors_map = {
                    "Commercial": "#3B82F6",
                    "Office": "#22C55E",
                    "Residential": "#F59E0B",
                    "Transit": "#A855F7",
                    "Competitor": "#EF4444",
                }
                fig3 = go.Figure(go.Pie(
                    labels=list(totals.keys()),
                    values=list(totals.values()),
                    hole=0.6,
                    marker=dict(colors=[poi_colors_map.get(k, "#888") for k in totals.keys()]),
                    textinfo="label+percent",
                    textfont=dict(size=11, color="white"),
                ))
                fig3.add_annotation(
                    text="POI Mix<br>Top 20",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=13, color="#F0F0F0", family="Syne"),
                )
                fig3.update_layout(
                    title=dict(text="POI Type Distribution", font=dict(family="Syne", size=14, color="#F0F0F0")),
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=300,
                )
                st.plotly_chart(fig3, use_container_width=True)
        
        # Candidate revenue distribution
        st.markdown("---")
        st.markdown("#### All Candidates Distribution")
        
        fig4 = go.Figure(go.Histogram(
            x=candidates["predicted_revenue_mad"],
            nbinsx=50,
            marker=dict(
                color=candidates["predicted_revenue_mad"],
                colorscale=[[0,"#8B0000"],[0.5,"#CC1B1B"],[1,"#22C55E"]],
                line=dict(width=0),
            ),
        ))
        fig4.update_layout(
            title=dict(text="Predicted Revenue Distribution — All Candidates", font=dict(family="Syne", size=14, color="#F0F0F0")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#888", title="Predicted Revenue (MAD)", gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(color="#888", title="Count", gridcolor="rgba(255,255,255,0.04)"),
            margin=dict(l=10, r=10, t=40, b=10),
            height=320,
        )
        st.plotly_chart(fig4, use_container_width=True)
        
        # Download buttons
        st.markdown("---")
        st.markdown("#### Download Results")
        
        dl1, dl2, dl3 = st.columns(3)
        
        with dl1:
            top20_csv = top20.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Top 20 Sites (CSV)",
                data=top20_csv,
                file_name="top_20_sites_custom.csv",
                mime="text/csv",
                use_container_width=True,
            )
        
        with dl2:
            candidates_csv = candidates.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download All Candidates (CSV)",
                data=candidates_csv,
                file_name="all_candidates_custom.csv",
                mime="text/csv",
                use_container_width=True,
            )
        
        with dl3:
            metrics_json = json.dumps(metrics, indent=2).encode('utf-8')
            st.download_button(
                label="Download Model Metrics (JSON)",
                data=metrics_json,
                file_name="model_metrics_custom.json",
                mime="application/json",
                use_container_width=True,
            )
        
        # Results table
        st.markdown("---")
        st.markdown("#### Full Results Table")
        
        show_cols = [c for c in [
            "rank", "nearest_city", "lat", "lon",
            "predicted_revenue_mad", "combined_score",
            "similarity_to_best", "poi_score",
            "commercial_1km", "office_1km", "residential_1km",
        ] if c in top20.columns]
        
        st.dataframe(
            top20[show_cols],
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("Upload the required files and click 'Run Site Selection Model' to generate recommendations.")


# ═══════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
#  TAB 9 — NETWORK OPTIMIZATION (ML BCG MATRIX)
# ═══════════════════════════════════════════════════════════

def render_network_optimization_tab(df):
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700;
                color:#F0F0F0; letter-spacing:0.05em; margin-bottom:0.4rem;">
          Branch Network Optimization (Auto-ML)
    </div>
    <div style="color:#888; font-size:0.9rem; margin-bottom:1.5rem;">
         Use Machine Learning to isolate <b>Location Potential</b> from <b>Management Execution</b>.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown("### Controls")
        if st.button("Run Auto-ML Optimization Pipeline", type="primary", use_container_width=True):
            st.session_state["run_optimization"] = True
            
        st.markdown("---")
        st.markdown("#### Gap Analysis Thresholds")
        overperf_thresh = st.slider("Overperforming Ratio (>)", 1.0, 1.5, 1.15, 0.05)
        underperf_thresh = st.slider("Underperforming Ratio (<)", 0.5, 1.0, 0.85, 0.05)
        
    with col2:
        if st.session_state.get("run_optimization", False):
            with st.spinner("Training XGBoost Regressor on the fly..."):
                try:
                    perf_df = pd.read_csv("data/inputs/branch_performance.csv")
                    perf_df.columns = perf_df.columns.str.strip().str.lower()
                    
                    opt_df = df.merge(perf_df, on="name", how="left")
                    
                    feats = ["score_commercial", "score_office", "score_residential", 
                             "score_accessibility", "score_competition", "score_cannibalization"]
                    
                    # Density
                    try:
                        den = pd.read_csv("data/inputs/density.csv")
                        den.columns = [c.strip().lower() for c in den.columns]
                        if "lng" in den.columns and "lon" not in den.columns:
                            den.rename(columns={"lng": "lon"}, inplace=True)
                        tree = cKDTree(den[["lat", "lon"]].values)
                        dist, idx = tree.query(opt_df[["latitude", "longitude"]].values)
                        opt_df["population_density"] = den.iloc[idx]["population"].values
                        feats.append("population_density")
                    except Exception as e:
                        st.warning(f"Could not load density: {e}")
                    
                    valid_feats = [f for f in feats if f in opt_df.columns]
                    train_data = opt_df.dropna(subset=["monthly_revenue_mad"] + valid_feats)
                    
                    if len(train_data) > 0:
                        X = train_data[valid_feats]
                        y = train_data["monthly_revenue_mad"]
                        
                        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                        model.fit(X, y)
                        
                        opt_df["predicted_revenue"] = model.predict(opt_df[valid_feats].fillna(0))
                        # Prevent div by zero
                        opt_df["predicted_revenue"] = opt_df["predicted_revenue"].replace(0, 1)
                        opt_df["performance_ratio"] = opt_df["monthly_revenue_mad"] / opt_df["predicted_revenue"]
                        
                        median_pred = opt_df["predicted_revenue"].median()
                        
                        def categorize(row):
                            pot = row["predicted_revenue"] > median_pred
                            exec_ratio = row["performance_ratio"]
                            
                            if pot and exec_ratio > overperf_thresh: return "FLAGSHIP"
                            if pot and exec_ratio < underperf_thresh: return "REVIEW / INVEST"
                            if pot: return "PERFORMING WELL"
                            if not pot and exec_ratio > overperf_thresh: return "MONITOR"
                            if not pot and exec_ratio < underperf_thresh: return "CLOSE"
                            return "PERFORMING AS EXPECTED"
                            
                        opt_df["strategic_action"] = opt_df.apply(categorize, axis=1)
                        
                        # Geospatial Merger Logic
                        flagships = opt_df[opt_df["strategic_action"] == "FLAGSHIP"]
                        if len(flagships) > 0:
                            f_tree = cKDTree(flagships[["latitude", "longitude"]].values)
                            dist, idx = f_tree.query(opt_df[["latitude", "longitude"]].values)
                            for i, row in opt_df.iterrows():
                                if row["strategic_action"] in ["CLOSE", "REVIEW / INVEST"]:
                                    if dist[i] < 0.012:  # ~1.2km
                                        f_name = flagships.iloc[idx[i]]["name"]
                                        opt_df.at[i, "strategic_action"] = f"MERGE into {f_name}"
                        
                        # Plotly Scatter
                        color_map = {
                            "FLAGSHIP": "#22C55E",
                            "PERFORMING WELL": "#86EFAC",
                            "MONITOR": "#3B82F6",
                            "PERFORMING AS EXPECTED": "#888888",
                            "REVIEW / INVEST": "#F59E0B",
                            "CLOSE": "#EF4444"
                        }
                        
                        # Default color for Merges
                        def get_color(act):
                            if act.startswith("MERGE"): return "#A855F7"
                            return color_map.get(act, "#888888")
                            
                        opt_df["color"] = opt_df["strategic_action"].apply(get_color)
                        
                        fig = px.scatter(
                            opt_df, x="predicted_revenue", y="performance_ratio",
                            color="strategic_action", hover_name="name",
                            hover_data={"monthly_revenue_mad": ":,.0f", "predicted_revenue": ":,.0f", "color": False},
                            color_discrete_map=color_map,
                            title="Strategic BCG Matrix: Location Potential vs Management Execution"
                        )
                        
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="DM Sans", color="#F0F0F0"),
                            xaxis=dict(title="Location Potential (Predicted Revenue)", showgrid=True, gridcolor="#333"),
                            yaxis=dict(title="Management Execution (Actual / Predicted)", showgrid=True, gridcolor="#333"),
                        )
                        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Expected Performance")
                        fig.add_vline(x=median_pred, line_dash="dash", line_color="gray", annotation_text="Median Potential")
                        
                        st.plotly_chart(fig, use_container_width=True)

                        # Map Visualization
                        st.markdown("### Spatial Restructuring Map")
                        
                        m = folium.Map(location=[opt_df["latitude"].mean(), opt_df["longitude"].mean()], zoom_start=6, tiles="CartoDB dark_matter")
                        
                        from folium.plugins import MarkerCluster
                        
                        for i, row in opt_df.iterrows():
                            act = row["strategic_action"]
                            color = row["color"]
                            
                            folium.CircleMarker(
                                location=[row["latitude"], row["longitude"]],
                                radius=5 if act == "FLAGSHIP" else 3,
                                color=color, fill=True, fillOpacity=0.8,
                                popup=f"<b>{row['name']}</b><br>{act}"
                            ).add_to(m)
                            
                            # Draw lines for Mergers
                            if act.startswith("MERGE"):
                                target_name = act.split("MERGE into ")[-1]
                                target_row = opt_df[opt_df["name"] == target_name]
                                if len(target_row) > 0:
                                    target_lat = target_row.iloc[0]["latitude"]
                                    target_lon = target_row.iloc[0]["longitude"]
                                    folium.PolyLine(
                                        locations=[[row["latitude"], row["longitude"]], [target_lat, target_lon]],
                                        color="#A855F7", weight=2, opacity=0.8, dash_array="5"
                                    ).add_to(m)
                                    
                        st_folium(m, width="100%", height=500, returned_objects=[])



                        
                        # Table
                        st.markdown("### Actionable Recommendations")
                        action_df = opt_df[opt_df["strategic_action"].isin(["CLOSE", "REVIEW / INVEST"]) | opt_df["strategic_action"].str.startswith("MERGE")]
                        
                        disp_df = action_df[["name", "city", "monthly_revenue_mad", "predicted_revenue", "performance_ratio", "strategic_action"]].copy()
                        disp_df.columns = ["Branch", "City", "Actual Rev (MAD)", "Expected Rev (MAD)", "Exec Ratio", "Recommendation"]
                        disp_df["Actual Rev (MAD)"] = disp_df["Actual Rev (MAD)"].apply(lambda x: f"{x:,.0f}")
                        disp_df["Expected Rev (MAD)"] = disp_df["Expected Rev (MAD)"].apply(lambda x: f"{x:,.0f}")
                        disp_df["Exec Ratio"] = disp_df["Exec Ratio"].apply(lambda x: f"{x:.2f}x")
                        
                        st.dataframe(disp_df, use_container_width=True, hide_index=True)
                        
                        # Download button
                        csv = opt_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Full Optimization Data", data=csv, file_name="network_optimization.csv", mime="text/csv")
                        
                    else:
                        st.error("No valid training data (monthly_revenue_mad missing).")
                except Exception as e:
                    st.error(f"Error loading performance data: {e}")
        else:
            st.info(" Configure thresholds and click 'Run Auto-ML Optimization Pipeline' to start.")



# ═══════════════════════════════════════════════════════════
#  TAB 10 — DATA ENRICHMENT (Live Scraping Console)
# ═══════════════════════════════════════════════════════════

def _strip_ansi(text):
    """Remove ANSI color escape codes for clean display."""
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)

def _stream_reader(proc, output_queue):
    """Background thread that reads subprocess stdout line by line."""
    try:
        for line in iter(proc.stdout.readline, ''):
            if line:
                output_queue.put(line)
            else:
                break
    except Exception:
        pass
    finally:
        output_queue.put(None)  # sentinel

def render_data_enrichment_tab(df):
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700;
                color:#F0F0F0; letter-spacing:0.05em; margin-bottom:0.4rem;">
         🛰️ Data Enrichment — Google Maps Scraper
    </div>
    <div style="color:#888; font-size:0.9rem; margin-bottom:1.5rem;">
         Enrich the POI dataset by scraping new data from Google Maps.
         The output below mirrors exactly what you would see in the terminal.
    </div>
    """, unsafe_allow_html=True)

    # ── Paths ──────────────────────────────────────────────────
    SCRAPER_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'google-maps-checkpoints'))
    MAPS_SCRIPT   = os.path.join(SCRAPER_DIR, 'maps.py')
    CLEAN_SCRIPT  = os.path.join(SCRAPER_DIR, 'clean.py')
    CHECKPOINT_F  = os.path.join(SCRAPER_DIR, 'scraper_checkpoint.txt')
    RAW_CSV       = os.path.join(SCRAPER_DIR, 'morocco_gmaps_businesses.csv')
    CLEANALL_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts', 'cleanall.py'))

    # ── Prerequisite check ─────────────────────────────────────
    if not os.path.exists(MAPS_SCRIPT):
        st.error(f"Scraper not found at `{MAPS_SCRIPT}`")
        return

    # ── Status info cards ──────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    ckpt_exists = os.path.exists(CHECKPOINT_F)
    csv_exists  = os.path.exists(RAW_CSV)
    with col_a:
        if ckpt_exists:
            with open(CHECKPOINT_F, 'r') as _f:
                ckpt_count = sum(1 for _ in _f)
            st.metric("Checkpoint Entries", f"{ckpt_count:,}")
        else:
            st.metric("Checkpoint Entries", "0")
    with col_b:
        if csv_exists:
            try:
                row_count = sum(1 for _ in open(RAW_CSV, encoding='utf-8')) - 1
                st.metric("Scraped Records", f"{row_count:,}")
            except Exception:
                st.metric("Scraped Records", "—")
        else:
            st.metric("Scraped Records", "0")
    with col_c:
        merged_path = os.path.join(os.path.dirname(__file__), 'data', 'inputs', 'merged_data_clean.csv')
        if os.path.exists(merged_path):
            try:
                mr = sum(1 for _ in open(merged_path, encoding='utf-8')) - 1
                st.metric("Merged POI Total", f"{mr:,}")
            except Exception:
                st.metric("Merged POI Total", "—")
        else:
            st.metric("Merged POI Total", "—")

    st.markdown("---")

    # ── Requirements warning ───────────────────────────────────
    with st.expander("⚠️ Requirements (click to expand)"):
        st.markdown("""
        The scraper requires the following to be installed:
        ```
        pip install selenium webdriver-manager pandas colorama
        ```
        You also need **Google Chrome** installed on this machine.
        The scraper runs in headless mode by default.
        """)

    # ── Scraping Controls ──────────────────────────────────────
    st.markdown("### Scraping Configuration")

    mode = st.radio(
        "Select scraping mode:",
        [" Resume from Checkpoints", " Start Fresh (delete existing data)"],
        index=0,
        key="scrape_mode",
        horizontal=True,
    )

    col1, col2 = st.columns(2)
    start_clicked = col1.button(" Start Scraping", type="primary", use_container_width=True, key="btn_start_scrape")
    stop_clicked  = col2.button(" Stop Scraper",  type="secondary", use_container_width=True, key="btn_stop_scrape")

    # ── State management ───────────────────────────────────────
    if 'scraper_process' not in st.session_state:
        st.session_state.scraper_process = None
    if 'scraper_log' not in st.session_state:
        st.session_state.scraper_log = ""
    if 'scraper_running' not in st.session_state:
        st.session_state.scraper_running = False

    # ── Stop handler ───────────────────────────────────────────
    if stop_clicked and st.session_state.scraper_process is not None:
        try:
            st.session_state.scraper_process.terminate()
            st.session_state.scraper_process.wait(timeout=5)
        except Exception:
            try:
                st.session_state.scraper_process.kill()
            except Exception:
                pass
        st.session_state.scraper_running = False
        st.session_state.scraper_log += "\n\n=== SCRAPER STOPPED BY USER ==="
        st.toast("Scraper stopped. Data saved up to the last checkpoint.", icon="")
        st.rerun()

    # ── Start handler ──────────────────────────────────────────
    if start_clicked:
        # Fresh start: wipe checkpoint and CSV
        if "Fresh" in mode:
            for fpath in [CHECKPOINT_F, RAW_CSV]:
                if os.path.exists(fpath):
                    os.remove(fpath)
            st.session_state.scraper_log = "=== FRESH START — Checkpoints and CSV deleted ===\n\n"
        else:
            st.session_state.scraper_log = "=== RESUMING FROM CHECKPOINTS ===\n\n"

        try:
            scraper_env = os.environ.copy()
            scraper_env['PYTHONIOENCODING'] = 'utf-8'
            proc = subprocess.Popen(
                ['python', '-u', MAPS_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=SCRAPER_DIR,
                bufsize=1,
                env=scraper_env,
            )
            st.session_state.scraper_process = proc
            st.session_state.scraper_running = True

            # Stream output
            output_queue = queue.Queue()
            reader_thread = threading.Thread(target=_stream_reader, args=(proc, output_queue), daemon=True)
            reader_thread.start()

            terminal_placeholder = st.empty()
            status_placeholder   = st.empty()

            while True:
                # Drain the queue
                lines_this_cycle = []
                try:
                    while True:
                        item = output_queue.get_nowait()
                        if item is None:
                            st.session_state.scraper_running = False
                            break
                        lines_this_cycle.append(item)
                except queue.Empty:
                    pass

                if lines_this_cycle:
                    cleaned = _strip_ansi(''.join(lines_this_cycle))
                    st.session_state.scraper_log += cleaned

                # Keep only the last 200 lines for display perf
                display_lines = st.session_state.scraper_log.split('\n')
                if len(display_lines) > 200:
                    display_lines = display_lines[-200:]
                terminal_placeholder.code('\n'.join(display_lines), language='bash')

                # Check if process ended
                if proc.poll() is not None and not st.session_state.scraper_running:
                    break
                if proc.poll() is not None:
                    # flush remaining
                    remaining = proc.stdout.read()
                    if remaining:
                        st.session_state.scraper_log += _strip_ansi(remaining)
                    st.session_state.scraper_running = False
                    break

                time.sleep(0.5)

            # Final update
            display_lines = st.session_state.scraper_log.split('\n')
            if len(display_lines) > 200:
                display_lines = display_lines[-200:]
            terminal_placeholder.code('\n'.join(display_lines), language='bash')
            status_placeholder.success("Scraping session ended. Data has been saved.")
            st.session_state.scraper_process = None

        except FileNotFoundError:
            st.error("Python interpreter not found. Make sure Python is in your PATH.")
        except Exception as e:
            st.error(f"Failed to start the scraper: {e}")

    # ── Display existing log if not actively running ───────────
    if not start_clicked and st.session_state.scraper_log:
        st.markdown("###  Last Scraping Session Output")
        display_lines = st.session_state.scraper_log.split('\n')
        if len(display_lines) > 200:
            display_lines = display_lines[-200:]
        st.code('\n'.join(display_lines), language='bash')

    st.markdown("---")

    # ── Post-Scraping Pipeline ─────────────────────────────────
    st.markdown("### Clean & Merge Pipeline")
    st.markdown("""
    <div style="color:#aaa; font-size:0.85rem; margin-bottom:1rem;">
        After scraping, run the pipeline below to:<br/>
        <b>Step A</b> — Clean the raw Google Maps CSV (fix encoding, select columns).<br/>
        <b>Step B</b> — Merge all 3 data sources (Google Maps + OSM + Geofabrik) and deduplicate.
    </div>
    """, unsafe_allow_html=True)

    if 'pipeline_log' not in st.session_state:
        st.session_state.pipeline_log = ""

    if st.button(" Run Clean & Merge Pipeline", type="primary", use_container_width=True, key="btn_clean_merge"):
        pipeline_output = []

        # Step A: clean.py
        pipeline_output.append("=" * 60)
        pipeline_output.append("STEP A: Cleaning Google Maps data (clean.py)")
        pipeline_output.append("=" * 60)
        try:
            result_a = subprocess.run(
                ['python', CLEAN_SCRIPT],
                capture_output=True, text=True, encoding='utf-8', errors='replace',
                cwd=SCRAPER_DIR, timeout=120,
            )
            pipeline_output.append(result_a.stdout if result_a.stdout else "(no output)")
            if result_a.stderr:
                pipeline_output.append(f"STDERR: {result_a.stderr}")
            if result_a.returncode == 0:
                pipeline_output.append(" clean.py completed successfully.")
            else:
                pipeline_output.append(f" clean.py failed (exit code {result_a.returncode}).")
        except Exception as e:
            pipeline_output.append(f" Error running clean.py: {e}")

        pipeline_output.append("")

        # Copy google_final.csv to allinone
        src_google = os.path.join(SCRAPER_DIR, 'google_final.csv')
        dst_google = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'allinone', 'google_final.csv'))
        if os.path.exists(src_google):
            import shutil
            try:
                shutil.copy2(src_google, dst_google)
                pipeline_output.append(f"📋 Copied google_final.csv to allinone/")
            except Exception as e:
                pipeline_output.append(f" Could not copy to allinone: {e}")

        pipeline_output.append("")

        # Step B: cleanall.py
        pipeline_output.append("=" * 60)
        pipeline_output.append("STEP B: Merging all sources & deduplicating (cleanall.py)")
        pipeline_output.append("=" * 60)
        try:
            result_b = subprocess.run(
                ['python', CLEANALL_SCRIPT],
                capture_output=True, text=True, encoding='utf-8', errors='replace',
                cwd=os.path.dirname(CLEANALL_SCRIPT), timeout=300,
            )
            pipeline_output.append(result_b.stdout if result_b.stdout else "(no output)")
            if result_b.stderr:
                pipeline_output.append(f"STDERR: {result_b.stderr}")
            if result_b.returncode == 0:
                pipeline_output.append(" cleanall.py completed successfully.")
            else:
                pipeline_output.append(f" cleanall.py failed (exit code {result_b.returncode}).")
        except Exception as e:
            pipeline_output.append(f" Error running cleanall.py: {e}")

        pipeline_output.append("")
        pipeline_output.append("=" * 60)
        pipeline_output.append("PIPELINE COMPLETE")
        pipeline_output.append("=" * 60)

        st.session_state.pipeline_log = '\n'.join(pipeline_output)
        st.rerun()

    if st.session_state.pipeline_log:
        st.code(st.session_state.pipeline_log, language='bash')


def main():
    # ── Load data ──
    data_file = "data/inputs/branches_scored.csv"
    if not os.path.exists(data_file):
        st.error(f" File not found: **{data_file}**\n\nRun `scoring_pipeline.py` first to generate it.")
        st.stop()

    df_raw = load_data(data_file)
    render_header()

    # ── Sidebar filters → filtered df ──
    df = render_sidebar(df_raw)

    # ── KPIs ──
    render_kpis(df)
    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

    # ── Tabs ──
    tabs = st.tabs([
        "Map",
        "Rankings",
        "ML Clusters",
        "Mergers",
        "Cities",
        "Deep Dive",
        "Data",
        "Run Site Selection",
        "Network Optimization",
        "Data Enrichment",
    ])

    with tabs[0]: render_map_tab(df)
    with tabs[1]: render_rankings_tab(df)
    with tabs[2]: render_clusters_tab(df)
    with tabs[3]: render_merger_tab(df)
    with tabs[4]: render_city_tab(df)
    with tabs[5]: render_deepdive_tab(df)
    with tabs[6]: render_data_tab(df)
    with tabs[7]: render_run_site_selection_tab(df)
    with tabs[8]: render_network_optimization_tab(df)
    with tabs[9]: render_data_enrichment_tab(df)


if __name__ == "__main__":
    main()