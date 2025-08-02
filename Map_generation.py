import pandas as pd
import folium
from folium.plugins import MarkerCluster

# --- Load cluster and detection data ---
clusters = pd.read_csv("pothole_clusters.csv")
detections = pd.read_csv("pothole_detections_tagged.csv")  # optional

# --- Create base map centered around mean cluster location ---
mean_lat = clusters['lat'].mean()
mean_lon = clusters['lon'].mean()
m = folium.Map(location=[mean_lat, mean_lon], zoom_start=14, control_scale=True)

# Optional: Show all detections as small red dots
show_detections = True
if show_detections:
    for _, row in detections.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color='red',
            fill=True,
            fill_opacity=0.4,
            popup=f"Confidence: {row['confidence']:.2f}"
        ).add_to(m)

# --- Add clustered pothole markers ---
marker_cluster = MarkerCluster().add_to(m)

for _, row in clusters.iterrows():
    popup_text = (
        f"<b>Cluster ID:</b> {row['cluster_id']}<br>"
        f"<b>Detections:</b> {row['num_detections']}<br>"
        f"<b>Avg Confidence:</b> {row['avg_confidence']:.2f}<br>"
        f"<b>Last Seen:</b> {row['last_seen']}"
    )

    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=popup_text,
        icon=folium.Icon(color='orange', icon='exclamation-sign')
    ).add_to(marker_cluster)

# --- Save map as HTML ---
m.save("pothole_map.html")
print("Map saved as pothole_map.html")
