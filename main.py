import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

# ---------------------------------------------------
# LOCATION & SEVERITY DATA
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


class UrbanConflictDetector:

    def __init__(self, df):
        self.df = df.copy()
        self.df['start_time'] = pd.to_datetime(self.df['start_time'])
        self.df['end_time'] = pd.to_datetime(self.df['end_time'])
        self.G = nx.Graph()
        self.conflicts_data = []   # stores full conflict details
        self.feature_list = []
        self.labels = []
        self.model = None

    # ---------------------------------------------------
    # DETECT CONFLICTS
    # ---------------------------------------------------
    def detect_conflicts(self):
        for _, row in self.df.iterrows():
            self.G.add_node(row['request_id'],
                            dept=row['department'],
                            loc=row['resource_location'])

        for i in range(len(self.df)):
            for j in range(i + 1, len(self.df)):
                a = self.df.iloc[i]
                b = self.df.iloc[j]

                coord_a = location_coords.get(a['resource_location'])
                coord_b = location_coords.get(b['resource_location'])

                if not coord_a or not coord_b:
                    continue

                distance = geodesic(coord_a, coord_b).km

                overlap_start = max(a['start_time'], b['start_time'])
                overlap_end = min(a['end_time'], b['end_time'])

                if overlap_start >= overlap_end:
                    continue

                overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600
                impact = severity_map.get(a['department'], 1) + severity_map.get(b['department'], 1)
                score = round(overlap_hours * impact, 2)

                self.feature_list.append([overlap_hours, impact, distance])
                self.labels.append(1 if overlap_hours > 0 and distance < 1 else 0)

                if a['resource_location'] == b['resource_location']:
                    self.G.add_edge(a['request_id'], b['request_id'], weight=score)
                    self.conflicts_data.append({
                        'req_a': a['request_id'],
                        'req_b': b['request_id'],
                        'dept_a': a['department'],
                        'dept_b': b['department'],
                        'location': a['resource_location'],
                        'overlap_hours': round(overlap_hours, 2),
                        'score': score,
                        'start_a': a['start_time'],
                        'end_a': a['end_time'],
                        'start_b': b['start_time'],
                        'end_b': b['end_time']
                    })

        return self.G

    # ---------------------------------------------------
    # GAP 1 — ML: TRAIN + ACTUALLY PREDICT NEW REQUESTS
    # ---------------------------------------------------
    def train_ml_model(self):
        if len(self.feature_list) < 5:
            print("Not enough data to train ML model.")
            return [0, 0, 0]

        X = np.array(self.feature_list)
        y = np.array(self.labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        acc = self.model.score(X_test, y_test)
        print(f"✅ ML Model Trained | Test Accuracy: {round(acc * 100, 2)}%")

        return self.model.feature_importances_

    def predict_conflict(self, new_request: dict, existing_df: pd.DataFrame):
        """
        Predict if a new incoming request will conflict BEFORE approving it.

        Usage:
            new_req = {
                'request_id': 'REQ_NEW_001',
                'department': 'Water',
                'resource_location': 'Wakad_Chowk',
                'start_time': '2024-01-15 08:00',
                'end_time': '2024-01-15 14:00'
            }
            detector.predict_conflict(new_req, df)
        """
        if self.model is None:
            print("❌ Train the model first using train_ml_model()")
            return []

        existing_df = existing_df.copy()
        existing_df['start_time'] = pd.to_datetime(existing_df['start_time'])
        existing_df['end_time'] = pd.to_datetime(existing_df['end_time'])

        new_start = pd.to_datetime(new_request['start_time'])
        new_end = pd.to_datetime(new_request['end_time'])
        new_coord = location_coords.get(new_request['resource_location'])

        predicted_conflicts = []

        for _, row in existing_df.iterrows():
            coord = location_coords.get(row['resource_location'])
            if not coord or not new_coord:
                continue

            distance = geodesic(new_coord, coord).km
            overlap_start = max(new_start, row['start_time'])
            overlap_end = min(new_end, row['end_time'])

            if overlap_start >= overlap_end:
                continue

            overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600
            impact = (severity_map.get(new_request['department'], 1) +
                      severity_map.get(row['department'], 1))

            prediction = self.model.predict([[overlap_hours, impact, distance]])[0]
            probability = self.model.predict_proba([[overlap_hours, impact, distance]])[0][1]

            if prediction == 1:
                predicted_conflicts.append({
                    'conflicts_with': row['request_id'],
                    'department': row['department'],
                    'location': row['resource_location'],
                    'overlap_hours': round(overlap_hours, 2),
                    'conflict_probability_%': round(probability * 100, 2)
                })

        print(f"\n{'='*60}")
        print(f"🔍 PREDICTION FOR: {new_request['request_id']}")
        print(f"{'='*60}")
        if predicted_conflicts:
            print(f"⚠️  {len(predicted_conflicts)} CONFLICT(S) PREDICTED — REQUEST SHOULD BE REVIEWED\n")
            for r in predicted_conflicts:
                print(f"  → Clashes with : {r['conflicts_with']} ({r['department']})")
                print(f"     Location     : {r['location']}")
                print(f"     Overlap      : {r['overlap_hours']} hrs")
                print(f"     Probability  : {r['conflict_probability_%']}%\n")
        else:
            print("✅ NO CONFLICTS PREDICTED — REQUEST CAN BE APPROVED SAFELY")
        print(f"{'='*60}\n")

        return predicted_conflicts

    # ---------------------------------------------------
    # GAP 2 — CONFLICT RESOLUTION ENGINE
    # ---------------------------------------------------
    def resolve_conflicts(self):
        """
        For each detected conflict suggests:
        1. Reschedule the lower-priority department
        2. Move to an alternative location
        """
        if not self.conflicts_data:
            print("No conflicts to resolve.")
            return []

        resolutions = []
        available_locations = list(location_coords.keys())

        for conflict in self.conflicts_data:
            priority_a = severity_map.get(conflict['dept_a'], 1)
            priority_b = severity_map.get(conflict['dept_b'], 1)

            # Higher priority keeps slot, lower priority gets rescheduled
            if priority_a >= priority_b:
                reschedule_id = conflict['req_b']
                reschedule_dept = conflict['dept_b']
                new_start = conflict['end_a']
                duration = conflict['end_b'] - conflict['start_b']
            else:
                reschedule_id = conflict['req_a']
                reschedule_dept = conflict['dept_a']
                new_start = conflict['end_b']
                duration = conflict['end_a'] - conflict['start_a']

            new_end = new_start + duration

            alt_locs = [l for l in available_locations if l != conflict['location']]

            resolution = {
                'conflict_pair': f"{conflict['req_a']} ↔ {conflict['req_b']}",
                'location': conflict['location'],
                'dept_a': conflict['dept_a'],
                'dept_b': conflict['dept_b'],
                'overlap_hours': conflict['overlap_hours'],
                'score': conflict['score'],
                'suggestion_reschedule': (
                    f"Move '{reschedule_id}' ({reschedule_dept}) to → "
                    f"{new_start.strftime('%Y-%m-%d %H:%M')} – "
                    f"{new_end.strftime('%H:%M')}"
                ),
                'suggestion_relocate': (
                    f"Move '{reschedule_id}' ({reschedule_dept}) to "
                    f"alternative location → '{alt_locs[0]}'"
                ) if alt_locs else "No alternative location available"
            }

            resolutions.append(resolution)

        print(f"\n{'='*70}")
        print(f"🛠️  CONFLICT RESOLUTION REPORT — {len(resolutions)} conflicts")
        print(f"{'='*70}")
        for r in resolutions:
            print(f"\n🔴 {r['conflict_pair']}")
            print(f"   Location  : {r['location']}")
            print(f"   Depts     : {r['dept_a']} vs {r['dept_b']}")
            print(f"   Overlap   : {r['overlap_hours']} hrs | Score: {r['score']}")
            print(f"   ✅ Option 1: {r['suggestion_reschedule']}")
            print(f"   ✅ Option 2: {r['suggestion_relocate']}")
        print(f"{'='*70}\n")

        return resolutions

    # ---------------------------------------------------
    # GAP 3 — CORRELATION ANALYSIS
    # ---------------------------------------------------
    def correlation_analysis(self):
        """
        Shows:
        1. Department vs Department clash heatmap
        2. Conflicts per location bar chart
        3. Top clashing department pairs in terminal
        """
        if not self.conflicts_data:
            print("No conflict data for correlation analysis.")
            return

        departments = list(severity_map.keys())
        clash_matrix = pd.DataFrame(0, index=departments, columns=departments)
        location_conflict_count = {}
        pair_counts = {}

        for c in self.conflicts_data:
            dept_a = c['dept_a']
            dept_b = c['dept_b']

            if dept_a in departments and dept_b in departments:
                clash_matrix.loc[dept_a, dept_b] += 1
                clash_matrix.loc[dept_b, dept_a] += 1

            loc = c['location']
            location_conflict_count[loc] = location_conflict_count.get(loc, 0) + 1

            pair = tuple(sorted([dept_a, dept_b]))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # --- Plot 1: Dept vs Dept Heatmap ---
        sns.heatmap(clash_matrix, annot=True, fmt='d',
                    cmap='YlOrRd', linewidths=0.5, ax=axes[0])
        axes[0].set_title("Department vs Department Clash Heatmap", fontsize=13)
        axes[0].set_xlabel("Department")
        axes[0].set_ylabel("Department")

        # --- Plot 2: Location conflict bar chart ---
        if location_conflict_count:
            locs = list(location_conflict_count.keys())
            counts = list(location_conflict_count.values())
            bars = axes[1].barh(locs, counts, color='coral')
            axes[1].set_title("Conflicts per Location", fontsize=13)
            axes[1].set_xlabel("Number of Conflicts")
            for bar, v in zip(bars, counts):
                axes[1].text(v + 0.1, bar.get_y() + bar.get_height() / 2,
                             str(v), va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig("correlation_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("✅ Correlation chart saved as 'correlation_analysis.png'")

        # --- Terminal: Top clashing pairs ---
        print("\n📊 TOP DEPARTMENT CLASH PAIRS:")
        print("-" * 35)
        for pair, count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {pair[0]:12} vs {pair[1]:12} → {count} clashes")

    # ---------------------------------------------------
    # VISUALIZE GRAPH
    # ---------------------------------------------------
    def visualize(self):
        if self.G.number_of_edges() == 0:
            print("No conflicts found to visualize.")
            return

        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.G, seed=42)
        nx.draw(self.G, pos, with_labels=True,
                node_color="skyblue", node_size=2000, font_size=8)
        labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels)
        plt.title("Urban Resource Conflict Graph")
        plt.tight_layout()
        plt.savefig("conflict_graph.png", dpi=150)
        plt.show()
        print("✅ Graph saved as 'conflict_graph.png'")


# ---------------------------------------------------
# RUN EXAMPLE
# ---------------------------------------------------
if __name__ == "__main__":

    df = pd.read_csv("urban_requests.csv")

    detector = UrbanConflictDetector(df)

    # Step 1: Detect
    graph = detector.detect_conflicts()
    print(f"✅ {graph.number_of_edges()} conflict edges detected")

    # Step 2: Train ML
    importances = detector.train_ml_model()

    # Step 3: Predict a NEW request BEFORE approving
    new_req = {
        'request_id': 'REQ_NEW_001',
        'department': 'Water',
        'resource_location': 'Wakad_Chowk',
        'start_time': '2024-01-15 08:00',
        'end_time': '2024-01-15 14:00'
    }
    detector.predict_conflict(new_req, df)

    # Step 4: Resolve all detected conflicts
    detector.resolve_conflicts()

    # Step 5: Correlation Analysis
    detector.correlation_analysis()

    # Step 6: Visualize
    detector.visualize()