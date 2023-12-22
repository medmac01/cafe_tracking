from flask import Flask, render_template, jsonify
import threading
import datetime
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

app = Flask(__name__)

# Global structure to store hourly counts
hourly_data = {}
current_count = 0


def run_yolo():
    global current_count
    cap = cv2.VideoCapture('video.mp4')
    model = YOLO("yolov8n.pt")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, classes=0)[0]
        current_hour = datetime.datetime.now().hour
        person_count = sum(1 for *_, class_id in result)

        # Update current count
        current_count = person_count

        # Update count for the current hour
        if current_hour not in hourly_data:
            hourly_data[current_hour] = {'total_count': 0, 'detections': 0}

        hourly_data[current_hour]['total_count'] += person_count
        hourly_data[current_hour]['detections'] += 1

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/data')
def data():
    # Calculate average count per hour
    print(hourly_data)
    average_counts = {hour: data['total_count'] / data['detections'] if data['detections'] else 0 
                      for hour, data in hourly_data.items()}
    print("AVG: ",average_counts)
    return jsonify({'average_counts': average_counts, 'current_count': current_count})

if __name__ == '__main__':
    threading.Thread(target=run_yolo, daemon=True).start()
    app.run(debug=True)
