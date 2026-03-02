import streamlit as st
import pandas as pd
import networkx as nx
import folium
import seaborn as sns
from streamlit_folium import st_folium
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="SENTINEL", layout="wide")

# --- Professional CSS ---
st.markdown("""
    <style>
        html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
        h1 { color: #1a1a2e; font-size: 2.2rem; font-weight: 800; }
        h2, h3 { color: #16213e; font-weight: 700; }
        [data-testid="metric-container"] {
            background: #f0f4ff;
            border: 1px solid #d0d9f0;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        }
        hr { border: none; border-top: 2px solid #e0e6f0; margin: 28px 0; }
        section[data-testid="stSidebar"] { background-color: #1a1a2e; }
        section[data-testid="stSidebar"] * { color: #ffffff !important; }
        div.stButton > button {
            background-color: #1a1a2e; color: white;
            border-radius: 8px; padding: 10px 24px;
            font-weight: 600; border: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header Banner ---
st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 28px 36px; border-radius: 14px; margin-bottom: 24px;'>
        <h1 style='color: white; margin: 0; font-size: 2.4rem;'>🛡️ SENTINEL</h1>
        <p style='color: #a0aec0; margin: 6px 0 0 0; font-size: 1.05rem;'>
            Urban Resource Conflict Resolver — Real-time Detection, Prediction & Resolution
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# CONSTANTS
# ---------------------------------------------------
location_coords = {
    "Main_St_Sector_7":  (18.6279, 73.7997),
    "Aundh_Road_Hwy":    (18.5610, 73.8070),
    "Wakad_Chowk":       (18.5995, 73.7620),
    "Punawale_Malwadi":  (18.6200, 73.7400),
    "Hinjewadi_Phase_1": (18.5912, 73.7389)
}

severity_map = {
    "Water": 3, "Electricity": 2, "Transport": 1, "Telecom": 2
}

PLOT_BG    = "#f8faff"
ACCENT1    = "#3b5bdb"
ACCENT2    = "#e63946"
ACCENT3    = "#f4a261"
TEXT_COLOR = "#1a1a2e"
FIG_SIZE   = (5, 3.5)   # ← all charts same size

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PLOT_BG)
    ax.figure.patch.set_facecolor(PLOT_BG)
    if title:  ax.set_title(title, fontsize=11, fontweight='bold', color=TEXT_COLOR, pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10, color=TEXT_COLOR)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#d0d9f0")
    ax.grid(axis='y', color='#e0e6f0', linestyle='--', linewidth=0.7, alpha=0.7)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.markdown("## ⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("📂 Upload Urban Requests CSV", type=["csv"])
min_score     = st.sidebar.slider("Minimum Severity Threshold", 0.0, 40.0, 5.0)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Required CSV Columns")
st.sidebar.code("request_id\ndepartment\nresource_location\nstart_time\nend_time", language="text")

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time']   = pd.to_datetime(df['end_time'])

    G                = nx.Graph()
    conflicts        = []
    hourly_conflicts = defaultdict(int)
    location_count   = defaultdict(int)
    feature_list     = []
    labels           = []
    conflict_edges   = []

    # ---------------------------------------------------
    # DETECTION ENGINE
    # ---------------------------------------------------
    total_pairs   = len(df) * (len(df) - 1) // 2
    progress_bar  = st.progress(0)
    progress_text = st.empty()
    pair_count    = 0
    last_update   = 0

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            pair_count += 1
            progress_percent = pair_count / total_pairs
            current_percent  = int(progress_percent * 100)
            if current_percent > last_update:
                last_update = current_percent
                progress_bar.progress(progress_percent)
                progress_text.markdown(
                    f"🔄 **Processing:** {pair_count}/{total_pairs} — `{current_percent}%`"
                )

            req_a   = df.iloc[i]
            req_b   = df.iloc[j]
            coord_a = location_coords.get(req_a['resource_location'])
            coord_b = location_coords.get(req_b['resource_location'])
            if not coord_a or not coord_b:
                continue

            distance      = geodesic(coord_a, coord_b).km
            overlap_start = max(req_a['start_time'], req_b['start_time'])
            overlap_end   = min(req_a['end_time'],   req_b['end_time'])
            if overlap_start >= overlap_end:
                continue

            overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600
            impact        = (severity_map.get(req_a['department'], 1) +
                             severity_map.get(req_b['department'], 1))
            score         = round(overlap_hours * impact, 2)

            feature_list.append([overlap_hours, impact, distance])
            labels.append(1 if overlap_hours > 0 and distance < 1 else 0)

            if score >= min_score:
                conflicts.append(score)
                G.add_node(req_a['request_id'])
                G.add_node(req_b['request_id'])
                G.add_edge(req_a['request_id'], req_b['request_id'], weight=score)
                hourly_conflicts[overlap_start.hour] += 1
                location_count[req_a['resource_location']] += 1
                location_count[req_b['resource_location']] += 1
                conflict_edges.append({
                    'req_a': req_a['request_id'],  'req_b': req_b['request_id'],
                    'dept_a': req_a['department'],  'dept_b': req_b['department'],
                    'location': req_a['resource_location'], 'score': score,
                    'start_a': req_a['start_time'], 'end_a': req_a['end_time'],
                    'start_b': req_b['start_time'], 'end_b': req_b['end_time'],
                    'overlap_h': round(overlap_hours, 2)
                })

    progress_bar.progress(1.0)
    progress_text.markdown("✅ **Conflict Detection Completed!**")

    # ---------------------------------------------------
    # TRAIN ML MODEL ONCE
    # ---------------------------------------------------
    ml_model    = None
    importances = [0.33, 0.33, 0.34]
    if feature_list:
        X = np.array(feature_list)
        y = np.array(labels)
        ml_model    = RandomForestClassifier(n_estimators=100, random_state=42)
        ml_model.fit(X, y)
        importances = ml_model.feature_importances_

    st.markdown("---")

    # ===================================================
    # SECTION 1 — KPI
    # ===================================================
    st.markdown("### 📌 Dashboard Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Total Requests",      len(df))
    c2.metric("⚠️ Total Conflicts",     len(conflicts))
    c3.metric("🔥 Total Impact Score",  round(sum(conflicts), 2))
    c4.metric("📊 Avg Conflict Severity",
              round(np.mean(conflicts), 2) if conflicts else 0)

    st.markdown("---")

    # ===================================================
    # SECTION 2 — RISK INTELLIGENCE
    # ===================================================
    st.markdown("### 🔥 Risk Intelligence Overview")
    colA, colB = st.columns(2, gap="large")

    with colA:
        st.markdown("#### Top 3 High-Risk Locations")
        top_3     = sorted(location_count.items(), key=lambda x: x[1], reverse=True)[:3]
        max_count = max([c for _, c in top_3], default=1)
        for loc, count in top_3:
            st.markdown(f"**{loc}**")
            st.progress(min(count / max_count, 1.0))
            st.caption(f"{count} conflicts detected")

    with colB:
        st.markdown("#### Conflict Severity Distribution")
        if conflicts:
            fig_hist, ax_hist = plt.subplots(figsize=FIG_SIZE)
            ax_hist.hist(conflicts, bins=10, color=ACCENT1,
                         edgecolor='white', linewidth=0.8)
            style_ax(ax_hist, xlabel="Severity Score", ylabel="Frequency")
            fig_hist.tight_layout()
            st.pyplot(fig_hist)
            plt.close(fig_hist)

    st.markdown("---")

    # ===================================================
    # SECTION 3 — ANALYTICS
    # ===================================================
    st.markdown("### 📊 Conflict Analytics")
    colC, colD = st.columns(2, gap="large")

    with colC:
        st.markdown("#### Hourly Congestion Trend")
        if hourly_conflicts:
            sorted_hours = sorted(hourly_conflicts.items())
            hours  = [h[0] for h in sorted_hours]
            counts = [h[1] for h in sorted_hours]
            fig2, ax2 = plt.subplots(figsize=FIG_SIZE)
            ax2.plot(hours, counts, marker='o', color=ACCENT1,
                     linewidth=2, markersize=6,
                     markerfacecolor='white', markeredgewidth=2)
            ax2.fill_between(hours, counts, alpha=0.12, color=ACCENT1)
            style_ax(ax2, xlabel="Hour of Day", ylabel="Conflict Count")
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    with colD:
        st.markdown("#### ML Feature Importance")
        features = ["Overlap Hours", "Dept Impact", "Distance"]
        fig3, ax3 = plt.subplots(figsize=FIG_SIZE)
        bars = ax3.bar(features, importances,
                       color=[ACCENT1, ACCENT3, ACCENT2],
                       edgecolor='white', linewidth=0.8, width=0.5)
        for bar, val in zip(bars, importances):
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f"{val:.2f}", ha='center', va='bottom',
                     fontsize=9, color=TEXT_COLOR, fontweight='bold')
        style_ax(ax3, ylabel="Importance Score")
        ax3.set_ylim(0, max(importances) * 1.3 + 0.05)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    st.markdown("---")

    # ===================================================
    # SECTION 4 — CONFLICT NETWORK
    # ===================================================
    st.markdown("### 🕸️ Visual Conflict Network")
    if G.number_of_edges() > 0:
        fig_net, ax_net = plt.subplots(figsize=(12, 6))
        fig_net.patch.set_facecolor(PLOT_BG)
        ax_net.set_facecolor(PLOT_BG)
        pos = nx.spring_layout(G, k=0.5, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color=ACCENT3,
                               node_size=600, ax=ax_net, alpha=0.95)
        nx.draw_networkx_labels(G, pos, font_size=7,
                                font_color=TEXT_COLOR, ax=ax_net)
        nx.draw_networkx_edges(G, pos, edge_color=ACCENT2,
                               width=1.5, alpha=0.7, ax=ax_net)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=6, font_color=TEXT_COLOR, ax=ax_net)
        ax_net.axis("off")
        fig_net.tight_layout()
        st.pyplot(fig_net)
        plt.close(fig_net)
    else:
        st.info("No connections found above the threshold.")

    st.markdown("---")

    # ===================================================
    # SECTION 5 — ML PREDICTION (GAP 1)
    # ===================================================
    st.markdown("### 🤖 Predict Conflict for New Request")
    st.caption("Submit a new request below to check for conflicts BEFORE approving it.")

    with st.form("predict_form"):
        col_p1, col_p2 = st.columns(2, gap="large")
        with col_p1:
            new_dept = st.selectbox("Department", list(severity_map.keys()))
            new_loc  = st.selectbox("Location",   list(location_coords.keys()))
        with col_p2:
            new_start_str = st.text_input("Start Time (YYYY-MM-DD HH:MM)", "2024-01-15 08:00")
            new_end_str   = st.text_input("End Time   (YYYY-MM-DD HH:MM)", "2024-01-15 14:00")
        submitted = st.form_submit_button("🔍 Check for Conflicts")

    if submitted:
        if not ml_model:
            st.warning("Not enough data to run ML prediction.")
        else:
            new_start_dt = pd.to_datetime(new_start_str)
            new_end_dt   = pd.to_datetime(new_end_str)
            new_coord    = location_coords.get(new_loc)
            pred_results = []

            for _, row in df.iterrows():
                coord = location_coords.get(row['resource_location'])
                if not coord or not new_coord:
                    continue
                distance = geodesic(new_coord, coord).km
                ov_start = max(new_start_dt, row['start_time'])
                ov_end   = min(new_end_dt,   row['end_time'])
                if ov_start >= ov_end:
                    continue
                ov_hours = (ov_end - ov_start).total_seconds() / 3600
                impact   = (severity_map.get(new_dept, 1) +
                            severity_map.get(row['department'], 1))
                pred = ml_model.predict([[ov_hours, impact, distance]])[0]
                prob = ml_model.predict_proba([[ov_hours, impact, distance]])[0][1]
                if pred == 1:
                    pred_results.append({
                        'Clashes With':           row['request_id'],
                        'Department':             row['department'],
                        'Location':               row['resource_location'],
                        'Overlap Hours':          round(ov_hours, 2),
                        'Conflict Probability %': round(prob * 100, 2)
                    })

            if pred_results:
                st.error(f"⚠️ {len(pred_results)} conflict(s) predicted — REQUEST NEEDS REVIEW")
                st.dataframe(pd.DataFrame(pred_results), use_container_width=True)
            else:
                st.success("✅ No conflicts predicted — Request can be approved safely!")

    st.markdown("---")

    # ===================================================
    # SECTION 6 — RESOLUTION ENGINE (GAP 2)
    # ===================================================
    st.markdown("### 🛠️ Conflict Resolution Suggestions")

    if conflict_edges:
        resolution_rows = []
        available_locs  = list(location_coords.keys())

        for c in conflict_edges:
            pa = severity_map.get(c['dept_a'], 1)
            pb = severity_map.get(c['dept_b'], 1)
            if pa >= pb:
                rid, rdept = c['req_b'], c['dept_b']
                new_s = c['end_a']
                new_e = new_s + (c['end_b'] - c['start_b'])
            else:
                rid, rdept = c['req_a'], c['dept_a']
                new_s = c['end_b']
                new_e = new_s + (c['end_a'] - c['start_a'])

            alt = [l for l in available_locs if l != c['location']]
            resolution_rows.append({
                'Conflict Pair':              f"{c['req_a']} ↔ {c['req_b']}",
                'Location':                   c['location'],
                'Dept A':                     c['dept_a'],
                'Dept B':                     c['dept_b'],
                'Score':                      c['score'],
                'Overlap (hrs)':              c['overlap_h'],
                '✅ Option 1 – Reschedule':  (
                    f"Move '{rid}' ({rdept}) → "
                    f"{new_s.strftime('%Y-%m-%d %H:%M')} to {new_e.strftime('%H:%M')}"
                ),
                '✅ Option 2 – Relocate': (
                    f"Move '{rid}' to '{alt[0]}'" if alt else "No alternative location"
                )
            })

        st.dataframe(pd.DataFrame(resolution_rows), use_container_width=True)
    else:
        st.info("No conflicts found to resolve.")

    st.markdown("---")

    # ===================================================
    # SECTION 7 — CORRELATION ANALYSIS (GAP 3)
    # ===================================================
    st.markdown("### 📈 Correlation Analysis")

    departments  = list(severity_map.keys())
    clash_matrix = pd.DataFrame(0, index=departments, columns=departments)
    loc_count    = {}
    pair_counts  = {}

    for c in conflict_edges:
        da, db = c['dept_a'], c['dept_b']
        if da in departments and db in departments:
            clash_matrix.loc[da, db] += 1
            clash_matrix.loc[db, da] += 1
        loc_count[c['location']] = loc_count.get(c['location'], 0) + 1
        pair = tuple(sorted([da, db]))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    if conflict_edges:
        colE, colF = st.columns(2, gap="large")

        with colE:
            st.markdown("#### Dept vs Dept Clash Heatmap")
            fig_heat, ax_heat = plt.subplots(figsize=FIG_SIZE)
            fig_heat.patch.set_facecolor(PLOT_BG)
            sns.heatmap(clash_matrix, annot=True, fmt='d',
                        cmap='YlOrRd', linewidths=0.6,
                        linecolor='white', ax=ax_heat,
                        annot_kws={"size": 11, "weight": "bold"})
            ax_heat.set_title("Department Clash Matrix",
                              fontsize=11, fontweight='bold', color=TEXT_COLOR)
            ax_heat.tick_params(colors=TEXT_COLOR, labelsize=9)
            fig_heat.tight_layout()
            st.pyplot(fig_heat)
            plt.close(fig_heat)

        with colF:
            st.markdown("#### Conflicts per Location")
            if loc_count:
                locs   = list(loc_count.keys())
                counts = list(loc_count.values())
                fig_loc, ax_loc = plt.subplots(figsize=FIG_SIZE)
                bars = ax_loc.barh(locs, counts, color=ACCENT2,
                                   edgecolor='white', linewidth=0.8, height=0.5)
                for bar, v in zip(bars, counts):
                    ax_loc.text(v + 0.05,
                                bar.get_y() + bar.get_height() / 2,
                                str(v), va='center', fontsize=9,
                                color=TEXT_COLOR, fontweight='bold')
                style_ax(ax_loc, xlabel="Number of Conflicts")
                ax_loc.set_title("Location Conflict Frequency",
                                 fontsize=11, fontweight='bold', color=TEXT_COLOR)
                ax_loc.grid(axis='x', color='#e0e6f0',
                            linestyle='--', linewidth=0.7, alpha=0.7)
                ax_loc.grid(axis='y', visible=False)
                fig_loc.tight_layout()
                st.pyplot(fig_loc)
                plt.close(fig_loc)

        if pair_counts:
            st.markdown("#### Top Clashing Department Pairs")
            pairs_df = pd.DataFrame([
                {"Dept A": p[0], "Dept B": p[1], "Total Clashes": cnt}
                for p, cnt in sorted(pair_counts.items(),
                                     key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(pairs_df, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough conflict data for correlation analysis.")

    st.markdown("---")

    # ---------------------------------------------------
    # GEOSPATIAL MAP
    # ---------------------------------------------------
    st.subheader("🗺 Department-wise Conflict Map")

    m = folium.Map(location=[18.59, 73.76], zoom_start=12)

    for loc, coord in location_coords.items():
        folium.Marker(
            location=coord,
            popup=loc,
            icon=folium.Icon(color="blue")
        ).add_to(m)

    st_folium(m, width=1400, height=620)

    st.markdown("---")

    # ===================================================
    # CONTACT
    # ===================================================
    with st.expander("👨‍💻 Developer & Team Details"):
        st.markdown("""
        **Team KB — SENTINEL Urban Conflict Detector**

        | Member | Contact |
        |---|---|
        | Ayush Bankar | +91 8237184068 |
        | Suraj Madane | +91 9284395390 |
        | Jay Godse | +91 9403316619 |
        | Abhijeet Chavan | +91 8956570991 |
        | Swarangi Kothawade | +91 9604288562 |

        📧 **Email:** teamkb@gmail.com
        🔗 **LinkedIn:** https://www.linkedin.com/in/TEAMKB-a3a934289/
        """)

else:
    st.markdown("""
        <div style='text-align:center; padding:60px 20px; background:#f0f4ff;
                    border-radius:14px; border:2px dashed #c0caf0; margin-top:40px;'>
            <h2 style='color:#1a1a2e;'>📂 No Data Loaded</h2>
            <p style='color:#555; font-size:1.05rem;'>
                Upload your Urban Requests CSV from the sidebar to launch SENTINEL.
            </p>
        </div>
    """, unsafe_allow_html=True)