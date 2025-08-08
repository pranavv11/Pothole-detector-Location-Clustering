import os
import cv2
import gpxpy
import gpxpy.gpx
from ultralytics import YOLO
from datetime import datetime, timedelta
from bisect import bisect_left

#----LOAD MODEL----
def load_model(model_path):
    return YOLO(model_path)

#----PARSE .gpx FILE----
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

#----FIND CLOSEST COORDINATES FOR DETECTION----
def get_closest_gps_data(gps_data, timestamp):
    times = [entry['time'] for entry in gps_data]
    pos = bisect_left(times, timestamp)

    if pos == 0:
        return gps_data[0]
    if pos == len(times):
        return gps_data[-1]

    before = gps_data[pos - 1]
    after = gps_data[pos]
    return before if abs((timestamp - before['time']).total_seconds()) < abs((after['time'] - timestamp).total_seconds()) else after


#----PROCESS VIDEO TO DETECT POTHOLES AND OBTAIN COORDINATES----
def process_video(video_path, gps_data, model, target_class_name="pothole", confidence_threshold=0.6):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0
    detections = []

    video_start_time = gps_data[0]['time']

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_time_offset = timedelta(seconds=frame_num / fps)
        frame_timestamp = video_start_time + frame_time_offset

        gps_point = get_closest_gps_data(gps_data, frame_timestamp)
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
