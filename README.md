# Pothole Detector - Location Clustering

This project identifies potholes from images, extracts their geolocation, and clusters them using unsupervised learning. It then visualizes these clusters on an interactive **Folium map** for easy city-level monitoring and analysis.

---

## 🚀 Project Overview

- Detect potholes using a trained deep learning model.
- Extract geolocations (GPS metadata or manually provided).
- Cluster nearby potholes using algorithms like **KMeans** or **DBSCAN**.
- Visualize clusters on a **Folium map** with interactive marker groups and color-coded clusters.

---

## 📁 Project Structure
Pothole-detector-Location-Clustering/

├── Model/     --Trained model weights and configs <br>
├── Output/    --Clustered results (maps, coordinates, images) <br>
├── Utils/     --Utility functions for model inference and clustering <br>
├── main.py    --Entry point to run full pipeline <br>
└── README.md 


---

## 🧠 Clustering Visualization Example

![Pothole map](https://github.com/your_username/Pothole-detector-Location-Clustering/blob/master/Output/pothole_map.png)

> Each cluster is color-coded and interactive. Clicking on a marker reveals image data and coordinates.

---

## 🛠️ Tech Stack

- **Python**
- **Folium** for interactive map rendering
- **Scikit-learn** for clustering (KMeans/DBSCAN)
- **OpenCV** & **Pillow** for image handling
- **YOLOv8/YOLOv5** (in `Model/`) for pothole detection
- **Jupyter / Google Colab** for experiments

---
## 📦 Installation

```bash
git clone https://github.com/your_username/Pothole-detector-Location-Clustering.git
cd Pothole-detector-Location-Clustering

pip install -r requirements.txt
```
```bash
python main.py
```

---

## 📌 Use Case

This project can be extended to work with **real-time data** for intelligent road monitoring systems. Here's a possible use case:

- **Vehicle-Based Detection**: Mount a pothole detection device (e.g., vibration sensor, camera + ML model) on moving vehicles such as buses, taxis, or municipal trucks.
- **GPS Integration**: Each detected pothole is tagged with real-time GPS coordinates using a GPS module.
- **Data Logging**: The collected location data is either:
  - Stored locally (on edge device), or
  - Streamed to a centralized server via a wireless network.
- **Periodic Clustering**: At regular intervals (e.g., hourly, daily), the system:
  - Clusters the collected pothole coordinates using this project.
  - Identifies **high-density pothole zones** using distance-based clustering.
- **Actionable Insights**: These clusters can be visualized on a map (via Folium or other GIS tools) and shared with municipal road repair teams to prioritize maintenance.

This approach enables **automated, scalable, and real-time road condition monitoring**, significantly reducing manual inspection efforts.
