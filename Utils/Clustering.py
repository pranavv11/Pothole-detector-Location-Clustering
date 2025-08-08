import pandas as pd
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import numpy as np
import os
from datetime import datetime

def perform_clustering(data, confidence_threshold=0.5, eps_km=0.4, min_samples=10):
    """
    Perform DBSCAN clustering on pothole detections.

    Args:
        input_csv_path (str): Path to the CSV file with 'latitude', 'longitude', 'confidence'.
        confidence_threshold (float): Minimum confidence for valid detections.
        eps_km (float): Radius in kilometers for DBSCAN.
        min_samples (int): Minimum samples for DBSCAN cluster.

    Returns:
        df_with_labels (pd.DataFrame): DataFrame with cluster_id column.
        clusters_summary (pd.DataFrame): Cluster centroids and metadata.
    """
    df = pd.DataFrame(data)
    df = df[df['confidence'] > confidence_threshold]

    coords = df[['latitude', 'longitude']].to_numpy()
    radians_coords = np.radians(coords)

    if df.empty:
        return df, pd.DataFrame(columns=['cluster_id', 'lat', 'lon', 'num_detections', 'avg_confidence', 'last_seen'])

    #----RADIAN TO KILOMETER CONVERSION PARAMETERS----
    kms_per_radian = 6371.0088
    epsilon = eps_km / kms_per_radian

    #----DBSCAN CLUSTERING----
    db = DBSCAN(
        eps=epsilon,
        min_samples=min_samples, 
        algorithm='ball_tree', 
        metric='haversine'
    )

    labels = db.fit_predict(radians_coords)
    df['cluster_id'] = labels

    clustered = df[df['cluster_id'] != -1]
    clusters = clustered.groupby('cluster_id').agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'confidence': ['count', 'mean']
    }).reset_index()

    clusters.columns = ['cluster_id', 'lat', 'lon', 'num_detections', 'avg_confidence']
    clusters['last_seen'] = datetime.utcnow().isoformat()

    #----APPEND THE CLUSTERS TO THE .csv FILE
    output_csv_path="../Output/Cluster_directory"
    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
        updated_df = pd.concat([existing_df, clusters], ignore_index=True)
    else:
        updated_df = clusters

    updated_df.to_csv(output_csv_path, index=False)

    return df, clusters
