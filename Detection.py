import os
import cv2
import gpxpy
import gpxpy.gpx
from ultralytics import YOLO
from datetime import datetime, timedelta
from bisect import bisect_left

# --- CONFIG ---
video_dir = "Video_validation/Videos" 
gpx_file_path = "Video_validation/gps_data.gpx"
model_path = "Model/best.pt"  
target_class_name = "pothole"  
confidence_threshold = 0.6

# --- LOAD YOLO MODEL ---
model = YOLO(model_path)

# --- PARSE GPX ---
def parse_gpx(gpx_file_path):
    gps_data = []
    with open(gpx_file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    gps_data.append({
                        "time": point.time,
                        "lat": point.latitude,
                        "lon": point.longitude
                    })
    return gps_data

# --- FIND CLOSEST GPS ENTRY ---
def get_closest_gps_data(gps_data, timestamp):
    times = [entry['time'] for entry in gps_data]
    pos = bisect_left(times, timestamp)
    
    if pos == 0:
        return gps_data[0]
    if pos == len(times):
        return gps_data[-1]
    
    before = gps_data[pos - 1]
    after = gps_data[pos]
    
    # Return the closest one
    return before if abs((timestamp - before['time']).total_seconds()) < abs((after['time'] - timestamp).total_seconds()) else after

# --- PROCESS SINGLE VIDEO ---
def process_video(video_path, gps_data):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0

    detections = []

    # Assume video starts at the beginning of GPS track
    video_start_time = gps_data[0]['time']

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Frame timestamp
        frame_time_offset = timedelta(seconds=frame_num / fps)
        frame_timestamp = video_start_time + frame_time_offset

        # Match frame timestamp to GPS
        gps_point = get_closest_gps_data(gps_data, frame_timestamp)

        # Run detection
        results = model.predict(source=frame, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]

                if class_name == target_class_name and conf >= confidence_threshold:
                    detections.append({
                        "video": os.path.basename(video_path),
                        "frame": frame_num,
                        "time": frame_timestamp.isoformat(),
                        "latitude": gps_point["lat"],
                        "longitude": gps_point["lon"],
                        "confidence": conf
                    })
        frame_num += 1

    cap.release()
    return detections

# --- MAIN PIPELINE ---
gps_data = parse_gpx(gpx_file_path)
all_detections = []

for file in sorted(os.listdir(video_dir)):
    if file.endswith(".mp4") or file.endswith(".avi"):
        print(f"Processing: {file}")
        video_path = os.path.join(video_dir, file)
        detections = process_video(video_path, gps_data)
        all_detections.extend(detections)

# --- RESULTS ---
import pandas as pd
df = pd.DataFrame(all_detections)
df.to_csv("pothole_detections_with_gps.csv", index=False)

print("Detection complete. Results saved to pothole_detections_with_gps.csv")
