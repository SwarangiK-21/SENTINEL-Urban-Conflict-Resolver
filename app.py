import streamlit as st
import pandas as pd
import networkx as nx
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="SENTINEL", layout="wide")
st.title("SENTINEL : Urban Conflict Resolver")

# ---------------------------------------------------
# LOCATION DATA
# ---------------------------------------------------
location_coords = {
    "Main_St_Sector_7": (18.6279, 73.7997),
    "Aundh_Road_Hwy": (18.5610, 73.8070),
    "Wakad_Chowk": (18.5995, 73.7620),
    "Punawale_Malwadi": (18.6200, 73.7400),
    "Hinjewadi_Phase_1": (18.5912, 73.7389)
}

severity_map = {
    "Water": 3,
    "Electricity": 2,
    "Transport": 1,
    "Telecom": 2
}

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload Urban Requests CSV", type=["csv"])
min_score = st.sidebar.slider("Minimum Severity Threshold", 0.0, 40.0, 5.0)

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])

    G = nx.Graph()
    conflicts = []
    hourly_conflicts = defaultdict(int)
    location_count = defaultdict(int)
    feature_list = []
    labels = []

    # ---------------------------------------------------
    # DETECTION ENGINE WITH SMOOTH PROGRESS
    # ---------------------------------------------------
    total_pairs = len(df) * (len(df) - 1) // 2
    progress_bar = st.progress(0)
    progress_text = st.empty()

    pair_count = 0
    last_update = 0

    for i in range(len(df)):
        for j in range(i + 1, len(df)):

            pair_count += 1
            progress_percent = pair_count / total_pairs

            # Update only when % changes (reduces blinking)
            current_percent = int(progress_percent * 100)
            if current_percent > last_update:
                last_update = current_percent
                progress_bar.progress(progress_percent)
                progress_text.markdown(
                    f"🔄 Processing Conflicts: {pair_count}/{total_pairs} ({current_percent}%)"
                )

            req_a = df.iloc[i]
            req_b = df.iloc[j]

            coord_a = location_coords.get(req_a['resource_location'])
            coord_b = location_coords.get(req_b['resource_location'])

            if not coord_a or not coord_b:
                continue

            distance = geodesic(coord_a, coord_b).km

            overlap_start = max(req_a['start_time'], req_b['start_time'])
            overlap_end = min(req_a['end_time'], req_b['end_time'])

            if overlap_start >= overlap_end:
                continue

            overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600

            impact = severity_map.get(req_a['department'], 1) + \
                     severity_map.get(req_b['department'], 1)

            score = round(overlap_hours * impact, 2)

            feature_list.append([overlap_hours, impact, distance])
            labels.append(1 if overlap_hours > 0 and distance < 1 else 0)

            if score >= min_score:

                conflicts.append(score)

                G.add_node(req_a['request_id'])
                G.add_node(req_b['request_id'])
                G.add_edge(req_a['request_id'],
                           req_b['request_id'],
                           weight=score)

                hourly_conflicts[overlap_start.hour] += 1
                location_count[req_a['resource_location']] += 1
                location_count[req_b['resource_location']] += 1

    progress_bar.progress(1.0)
    progress_text.markdown("✅ Conflict Detection Completed!")

    # ---------------------------------------------------
    # ML MODEL
    # ---------------------------------------------------
    if feature_list:
        X = np.array(feature_list)
        y = np.array(labels)
        model = RandomForestClassifier(n_estimators=50)
        model.fit(X, y)
        importances = model.feature_importances_
    else:
        importances = [0, 0, 0]

    # ---------------------------------------------------
    # KPI SECTION
    # ---------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Requests", len(df))
    col2.metric("Total Conflicts", len(conflicts))
    col3.metric("Total Impact Score", round(sum(conflicts), 2))
    col4.metric("Avg Conflict Severity",
                round(np.mean(conflicts), 2) if conflicts else 0)

    st.markdown("---")

    # ---------------------------------------------------
    # RISK INTELLIGENCE
    # ---------------------------------------------------
    st.subheader("🔥 Risk Intelligence Overview")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Top 3 High-Risk Locations")
        top_3 = sorted(location_count.items(),
                       key=lambda x: x[1],
                       reverse=True)[:3]

        for loc, count in top_3:
            st.markdown(f"**{loc}**")
            st.progress(min(count / 20, 1.0))
            st.write(f"{count} conflicts detected")

    with colB:
        st.markdown("### Conflict Severity Distribution")
        if conflicts:
            fig_hist, ax_hist = plt.subplots()
            ax_hist.hist(conflicts, bins=10)
            ax_hist.set_xlabel("Severity Score")
            ax_hist.set_ylabel("Frequency")
            st.pyplot(fig_hist)

    st.markdown("---")

    # ---------------------------------------------------
    # ANALYTICS
    # ---------------------------------------------------
    st.subheader("📊 Conflict Analytics")

    colC, colD = st.columns(2)

    with colC:
        st.markdown("### Hourly Congestion Trend")
        if hourly_conflicts:
            sorted_hours = sorted(hourly_conflicts.items())
            hours = [h[0] for h in sorted_hours]
            counts = [h[1] for h in sorted_hours]

            fig2, ax2 = plt.subplots()
            ax2.plot(hours, counts, marker='o')
            ax2.set_xlabel("Hour of Day")
            ax2.set_ylabel("Conflict Count")
            st.pyplot(fig2)

    with colD:
        st.markdown("### ML Feature Importance")
        features = ["Overlap Hours", "Department Impact", "Distance"]
        fig3, ax3 = plt.subplots()
        ax3.bar(features, importances)
        st.pyplot(fig3)

    st.markdown("---")

    # ---------------------------------------------------
    # VISUAL CONFLICT NETWORK
    # ---------------------------------------------------
    st.subheader("🕸️ Visual Conflict Network")

    if G.number_of_edges() > 0:
        fig_net, ax_net = plt.subplots(figsize=(6, 4))
        pos = nx.spring_layout(G, k=0.3, seed=42)

        nx.draw(G, pos,
                with_labels=True,
                node_color='orange',
                edge_color='red',
                node_size=500,
                font_size=8,
                width=1,
                ax=ax_net)

        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos,
                                     edge_labels=edge_labels,
                                     font_size=6,
                                     ax=ax_net)

        ax_net.axis("off")
        st.pyplot(fig_net)
    else:
        st.info("No connections found above the threshold.")

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

    st_folium(m, width=1000, height=500)

    st.markdown("---")

    # ---------------------------------------------------
    # CONTACT SECTION (COLLAPSIBLE)
    # ---------------------------------------------------
    with st.expander("👨‍💻 Developer & Team Details"):
        st.markdown("""
        **Team KB:** SENTINEL Urban Conflict Detector   

        **Team Members:**
        - Ayush Bankar – +91 8237184068  
        - Suraj Madane – +91 9284395390 
        - Jay Godse – +91 9403316619
        - Abhijeet Chavan – +91 8956570991
        - Swarangi Kothawade – +91 9604288562              

        📧 Email: teamkb@gmail.com  
        🔗 LinkedIn: https://www.linkedin.com/in/TEAMKB-a3a934289/  
        """)

else:
    st.info("Upload dataset to launch SENTINEL dashboard.")