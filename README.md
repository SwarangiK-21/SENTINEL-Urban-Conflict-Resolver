# 🛡️ SENTINEL — Urban Resource Conflict Resolver

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-RandomForest-green?style=for-the-badge&logo=scikit-learn)
![NetworkX](https://img.shields.io/badge/Graph-NetworkX-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## 🌐 Live App

### 👉 [Launch SENTINEL](https://sentinel-urban-conflict-resolver-m4uuejrrhb2reahiamyaoa.streamlit.app/)

---

## 📌 Overview

In a rapidly developing city, multiple departments — **Water, Electricity, Transport, and Telecom** — often attempt to utilize the same physical space or resources simultaneously (e.g., digging up a road for cables while it is being paved).

**SENTINEL** is an AI-powered urban conflict resolution dashboard that:
- Detects overlapping resource usage requests in real time
- Scores the severity of each conflict
- Predicts future conflicts using Machine Learning before approving new requests
- Suggests resolution strategies — rescheduling or relocation
- Visualizes conflicts as a graph network and on a geospatial map

---

## 🎯 Problem Statement

> **Hackathon — Code4Society | Problem 04: Intelligent Urban Resource Conflict Detector**

Objective: Prevent operational clashes and resource wastage by predicting conflicting schedules or spatial usage using **graph dependency modeling** and **correlation analysis**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 Conflict Detection Engine | Detects all overlapping department requests using time + geo analysis |
| 📊 Conflict Score Metric | Scores severity using `overlap_hours × department_impact` |
| 🕸️ Visual Conflict Network | Graph with nodes (requests) and edges (conflicts) using NetworkX |
| 🤖 ML Conflict Predictor | Random Forest model predicts if a NEW request will conflict before approval |
| 🛠️ Resolution Engine | Suggests rescheduling or relocation for every detected conflict |
| 📈 Correlation Analysis | Department vs Department clash heatmap + location conflict frequency |
| 🗺️ Geospatial Map | Interactive Folium map showing conflict hotspots across Pune |
| 📋 KPI Dashboard | Total requests, conflicts, impact score, avg severity at a glance |

---


## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/SENTINEL.git
cd SENTINEL
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

### 4. Upload Dataset
- Use the sidebar to upload a CSV file
- Sample datasets are provided in the `/datasets` folder

---

## 📂 Project Structure

```
SENTINEL/
│
├── app.py                  # Main Streamlit dashboard
├── main.py                 # Core UrbanConflictDetector class
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
└── datasets/
    ├── dataset_small.csv       # 20 rows — quick testing
    ├── dataset_medium.csv      # 40 rows — graph & heatmap testing
    ├── dataset_large.csv       # 60 rows — ML & resolution testing
    └── sentinel_dataset.csv    # 120 rows — full combined dataset
```

---

## 📋 Dataset Format

Your CSV file must have these exact column names:

| Column | Description | Example |
|---|---|---|
| `request_id` | Unique request identifier | REQ_001 |
| `department` | Department name | Water / Electricity / Transport / Telecom |
| `resource_location` | Location name | Wakad_Chowk |
| `start_time` | Work start datetime | 2024-01-15 08:00 |
| `end_time` | Work end datetime | 2024-01-15 14:00 |

---

## 🧠 How It Works

```
CSV Upload
    ↓
Conflict Detection Engine
(Time overlap + Geodesic distance check for every pair)
    ↓
Conflict Score = overlap_hours × department_impact
    ↓
Graph Built (NetworkX) → Nodes = Requests, Edges = Conflicts
    ↓
Random Forest ML Model trained on [overlap_hours, impact, distance]
    ↓
┌─────────────────────────────────────┐
│  Dashboard Sections                 │
│  • KPI Metrics                      │
│  • Risk Intelligence                │
│  • Conflict Analytics               │
│  • Visual Network Graph             │
│  • ML Conflict Predictor            │
│  • Resolution Engine                │
│  • Correlation Heatmap              │
│  • Geospatial Map                   │
└─────────────────────────────────────┘
```

---

## 🏗️ Tech Stack

| Technology | Usage |
|---|---|
| Python | Core language |
| Streamlit | Web dashboard framework |
| NetworkX | Graph modeling & visualization |
| Scikit-learn | Random Forest ML model |
| Pandas | Data processing |
| Folium | Interactive geospatial map |
| Matplotlib | Charts and plots |
| Seaborn | Correlation heatmap |
| Geopy | Geodesic distance calculation |
| NumPy | Numerical computation |

---

## 📦 Requirements

```
streamlit
pandas
networkx
folium
streamlit-folium
geopy
scikit-learn
matplotlib
numpy
seaborn
```

---



## 🏆 Built For

> **Hackathon — Code4Society**
> Problem Statement 04 — Intelligent Urban Resource Conflict Detector

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify and distribute.

---

<div align="center">
    <b>Made with ❤️ by Team KB</b><br>
    <a href="https://sentinel-urban-conflict-resolver-m4uuejrrhb2reahiamyaoa.streamlit.app/">🚀 Try SENTINEL Live</a>
</div>
