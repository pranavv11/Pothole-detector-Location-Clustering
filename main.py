import os
import pandas as pd
from utils.detection import load_model, parse_gpx, process_video
from utils.clustering import perform_clustering
from utils.map_generation import generate_pothole_map

# ----------- CONFIGURATION -------------
VIDEO_PATH = "Validation/Val_video.mp4"
GPX_PATH = "Validation/gps_data.gpx"
MODEL_PATH = "Model/best.pt"

DETECTIONS_OUTPUT_CSV = "Output/Cluster_directory/detections.csv"
CLUSTERS_OUTPUT_CSV = "Output/Cluster_directory/clusters.csv"
MAP_OUTPUT_HTML = "Output/Cluster_directory/pothole_map.html"
# ---------------------------------------

def main():
    # --- LOAD MODEL AND PARSE .gpx FILE---
    print("[INFO] Loading model...")
    model = load_model(MODEL_PATH)

    print("[INFO] Parsing GPX data...")
    gps_data = parse_gpx(GPX_PATH)

    # --- PROCESS VIDEOS AND DETECT POTHOLES---
    print("[INFO] Running pothole detection on video...")
    detections = process_video(VIDEO_PATH, gps_data, model)

    if not detections:
        print("[INFO] No potholes detected.")
        return

    # --- SAVE DETECTTIONS---
    print(f"[INFO] Saving {len(detections)} detections to CSV...")
    detections_df = pd.DataFrame(detections)
    detections_df.to_csv(DETECTIONS_OUTPUT_CSV, index=False)

    # --- DBSCAN CLUSTERING ---
    print("[INFO] Performing DBSCAN clustering...")
    labeled_df, clusters_df = perform_clustering(detections)
    clusters_df.to_csv(CLUSTERS_OUTPUT_CSV, index=False)

    # --- GENERATE MAP---
    print("[INFO] Generating map with clusters and detections...")
    generate_pothole_map(CLUSTERS_OUTPUT_CSV, DETECTIONS_OUTPUT_CSV, MAP_OUTPUT_HTML)

    print("[DONE] Pipeline completed successfully.")

if __name__ == "__main__":
    main()
