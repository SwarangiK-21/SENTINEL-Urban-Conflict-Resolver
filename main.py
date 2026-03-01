import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

class UrbanConflictDetector:

    def __init__(self, df):
        self.df = df.copy()
        self.df['start_time'] = pd.to_datetime(self.df['start_time'])
        self.df['end_time'] = pd.to_datetime(self.df['end_time'])
        self.G = nx.Graph()

    def detect_conflicts(self):
        for _, row in self.df.iterrows():
            self.G.add_node(row['request_id'],
                            dept=row['department'],
                            loc=row['resource_location'])

        for i in range(len(self.df)):
            for j in range(i + 1, len(self.df)):
                a = self.df.iloc[i]
                b = self.df.iloc[j]

                if a['resource_location'] == b['resource_location']:

                    overlap_start = max(a['start_time'], b['start_time'])
                    overlap_end = min(a['end_time'], b['end_time'])

                    if overlap_start < overlap_end:
                        hours = (overlap_end - overlap_start).total_seconds() / 3600
                        score = round(hours * 2, 2)

                        self.G.add_edge(a['request_id'],
                                        b['request_id'],
                                        weight=score)

        return self.G

    def visualize(self):
        plt.figure(figsize=(10,6))
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, with_labels=True, node_color="skyblue", node_size=2000)
        labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels)
        plt.title("Urban Resource Conflict Graph")
        plt.show()