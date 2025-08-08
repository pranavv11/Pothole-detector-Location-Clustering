import cv2
import os
import gpxpy
from datetime import timedelta
from bisect import bisect_left

# --- CONFIG ---
video_dir = "Video_validation/Videos"  # path to folder containing 8 videos
gpx_file_path = "Video_validation/gap_darta.gpx"

# --- LOAD GPX ---
def load_gpx_data(gpx_path):
    with open(gpx_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    gps_points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                gps_points.append({
                    "time": point.time,
                    "lat": point.latitude,
                    "lon": point.longitude
                })
    return gps_points

# --- MATCH CLOSEST GPS TIMESTAMP ---
def get_closest_gps(gps_points, target_time):
    times = [p['time'] for p in gps_points]
    pos = bisect_left(times, target_time)

    if pos == 0:
        return gps_points[0]
    if pos == len(times):
        return gps_points[-1]

    before = gps_points[pos - 1]
    after = gps_points[pos]

    # Return closest
    return before if abs((target_time - before['time']).total_seconds()) < abs((after['time'] - target_time).total_seconds()) else after

# --- PROCESS VIDEO ---
def verify_video_with_gpx(video_path, gps_data, start_time):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = start_time + timedelta(seconds=frame_num / fps)
        gps = get_closest_gps(gps_data, timestamp)

        # Draw info
        text1 = f"Frame: {frame_num}"
        text2 = f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        text3 = f"GPS: {gps['lat']:.6f}, {gps['lon']:.6f}"

        cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Video-GPS Sync Check", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()

# --- MAIN ---
gps_data = load_gpx_data(gpx_file_path)

# Default start time as first GPS timestamp
video_start_time = gps_data[0]['time']

for video_file in sorted(os.listdir(video_dir)):
    if video_file.endswith(('.mp4', '.avi')):
        print(f"\n▶️ Now playing: {video_file}")
        video_path = os.path.join(video_dir, video_file)
        verify_video_with_gpx(video_path, gps_data, video_start_time)

print("✅ All videos verified.")
