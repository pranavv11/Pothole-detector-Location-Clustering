import pandas as pd
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import numpy as np
from datetime import datetime

df = pd.read_csv("pothole_detections_with_gps.csv")
df = df[df['confidence'] > 0.5]

#Convert lat/lon to radians for haversine
coords = df[['latitude', 'longitude']].to_numpy()
radians_coords = np.radians(coords)

#DBSCAN clustering
kms_per_radian = 6371.0088
epsilon = 0.4 / kms_per_radian

db = DBSCAN(
    eps=epsilon,
    min_samples=7, 
    algorithm='ball_tree', 
    metric='haversine'
)
labels = db.fit_predict(radians_coords)
df['cluster_id'] = labels

#Extract cluster centroids
clustered = df[df['cluster_id'] != -1]
clusters = clustered.groupby('cluster_id').agg({
    'latitude': 'mean',
    'longitude': 'mean',
    'confidence': ['count', 'mean']
}).reset_index()

clusters.columns = ['cluster_id', 'lat', 'lon', 'num_detections', 'avg_confidence']
clusters['last_seen'] = datetime.utcnow().isoformat()

#Save results
df.to_csv("pothole_detections_tagged.csv", index=False)
clusters.to_csv("pothole_clusters.csv", index=False)

print("Clustering complete. Results saved to pothole_clusters.csv")
