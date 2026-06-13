"""
Catfish Sorting System - Dashboard V1
YOLOv11 NCNN + ByteTrack + Servo Control + FastAPI Dashboard

Fitur:
- MJPEG video stream (/video_feed)
- WebSocket realtime stats (/ws)
- API kontrol: start/stop detection, reset counter, servo test
- Monitoring: FPS, CPU temp, CPU usage, RAM usage, status servo
"""

import cv2
import time
import threading
import asyncio
import numpy as np
import psutil
import os
from ultralytics import YOLO
from adafruit_servokit import ServoKit

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# =============================
# KONFIGURASI
# =============================

MODEL_PATH = "best_ncnn_model"

CONF_THRESHOLD = 0.7
YOLO_INTERVAL = 5
SERVO_COOLDOWN = 2

FRAME_WIDTH = 640
FRAME_HEIGHT = 360

TARGET_FPS = 15
FRAME_TIME = 1.0 / TARGET_FPS

TARGET_GRADE = None
# TARGET_GRADE = ["besar"]

# =============================
# ROI
# =============================

ROI_X1 = 80
ROI_Y1 = 70
ROI_X2 = 560
ROI_Y2 = 285

# =============================
# GARIS TRIGGER
# =============================

TRIGGER_LINE_X = int(FRAME_WIDTH * 0.8)

# =============================
# SERVO PCA9685
# =============================

kit = ServoKit(channels=16)
kit.frequency = 50

SERVO_CHANNEL = 0

kit.servo[SERVO_CHANNEL].set_pulse_width_range(500, 2500)

# =============================
# GLOBAL STATE
# =============================

latest_frame = None
frame_lock = threading.Lock()

detection_enabled = False
servo_status = "READY"
system_status = "IDLE"

total_fish = 0
current_fps = 0.0
last_servo_time = 0

processed_ids = set()
last_positions = {}

# =============================
# FASTAPI APP
# =============================

app = FastAPI(title="Catfish Sorting Dashboard")

# =============================
# SERVO FUNCTIONS
# =============================

def set_servo_nonblocking(angle_start, angle_end):
    global servo_status

    def _run():
        global servo_status
        try:
            servo_status = "MOVING"
            kit.servo[SERVO_CHANNEL].angle = angle_start
            time.sleep(0.5)
            kit.servo[SERVO_CHANNEL].angle = angle_end
            time.sleep(0.3)
            servo_status = "READY"
        except Exception as e:
            print(f"[SERVO ERROR] {e}")
            servo_status = "ERROR"

    threading.Thread(target=_run, daemon=True).start()


def servo_test_sequence():
    """Test servo: 40° → 80° → 40°"""
    global servo_status

    def _run():
        global servo_status
        try:
            servo_status = "TESTING"
            kit.servo[SERVO_CHANNEL].angle = 40
            time.sleep(0.5)
            kit.servo[SERVO_CHANNEL].angle = 80
            time.sleep(0.5)
            kit.servo[SERVO_CHANNEL].angle = 40
            time.sleep(0.3)
            servo_status = "READY"
        except Exception as e:
            print(f"[SERVO TEST ERROR] {e}")
            servo_status = "ERROR"

    threading.Thread(target=_run, daemon=True).start()

# =============================
# SISTEM MONITORING
# =============================

def get_cpu_temperature():
    """Ambil suhu CPU Raspberry Pi"""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read().strip()) / 1000.0
            return round(temp, 1)
    except Exception:
        try:
            result = os.popen("vcgencmd measure_temp").readline()
            temp = float(result.replace("temp=", "").replace("'C\n", ""))
            return round(temp, 1)
        except Exception:
            return 0.0


def get_system_stats():
    """Kumpulkan semua statistik sistem"""
    return {
        "total_fish": total_fish,
        "fps": round(current_fps, 1),
        "cpu_temp": get_cpu_temperature(),
        "cpu_usage": psutil.cpu_percent(interval=None),
        "ram_usage": psutil.virtual_memory().percent,
        "servo_status": servo_status,
        "system_status": system_status,
        "detection_enabled": detection_enabled,
    }

# =============================
# LOAD MODEL
# =============================

print("Loading YOLO model...")
model = YOLO(MODEL_PATH, task='detect')
labels = model.names
print("Model loaded OK")

# =============================
# CAMERA
# =============================

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: Kamera tidak bisa dibuka!")
    exit(1)

# =============================
# MAIN DETECTION LOOP (Thread)
# =============================

def detection_loop():
    global latest_frame, detection_enabled, system_status
    global total_fish, current_fps, last_servo_time
    global processed_ids, last_positions

    frame_count = 0
    cached_results = None
    prev_time = time.time()

    print("Detection loop started.")

    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("WARNING: Gagal baca frame")
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # FPS
        current_time = time.time()
        current_fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        frame_count += 1

        # =============================
        # YOLO + TRACKING
        # =============================

        if detection_enabled:
            roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

            if roi is not None and roi.size > 0:

                if frame_count % YOLO_INTERVAL == 0 or cached_results is None:
                    try:
                        cached_results = model.track(
                            roi,
                            persist=True,
                            tracker="botsort.yaml",
                            conf=CONF_THRESHOLD,
                            imgsz=640,
                            iou=0.5,
                            verbose=False
                        )
                    except Exception as e:
                        print(f"[TRACK ERROR] {e}")
                        cached_results = None

                results = cached_results

                if results is not None and len(results) > 0:
                    boxes = results[0].boxes

                    if boxes is not None and boxes.id is not None:
                        ids = boxes.id.cpu().numpy().astype(int)
                        xyxy = boxes.xyxy.cpu().numpy()
                        classes = boxes.cls.cpu().numpy().astype(int)

                        for box, obj_id, cls in zip(xyxy, ids, classes):

                            if np.any(np.isnan(box)) or np.any(np.isinf(box)):
                                continue

                            xmin, ymin, xmax, ymax = map(int, box)

                            if xmin >= xmax or ymin >= ymax:
                                continue

                            w = xmax - xmin
                            h = ymax - ymin

                            if w < 20 or h < 20:
                                continue

                            label = labels[cls]

                            xmin += ROI_X1
                            xmax += ROI_X1
                            ymin += ROI_Y1
                            ymax += ROI_Y1

                            cx = int((xmin + xmax) / 2)

                            # Visual bounding box
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                            cv2.putText(frame, f"{label} ID:{obj_id}",
                                        (xmin, max(ymin - 10, 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0), 2)

                            # Crossing detection
                            if obj_id in last_positions:
                                prev_x = last_positions[obj_id]

                                if prev_x < TRIGGER_LINE_X and cx >= TRIGGER_LINE_X:

                                    if TARGET_GRADE is not None and label not in TARGET_GRADE:
                                        continue

                                    if obj_id not in processed_ids:
                                        now = time.time()

                                        if now - last_servo_time > SERVO_COOLDOWN:
                                            processed_ids.add(obj_id)
                                            last_servo_time = now
                                            total_fish += 1

                                            print(f"[INFO] {label} ID {obj_id} lewat | Total: {total_fish}")
                                            set_servo_nonblocking(40, 80)

                            last_positions[obj_id] = cx

        # =============================
        # VISUAL OVERLAY
        # =============================

        cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 100, 0), 2)
        cv2.line(frame, (TRIGGER_LINE_X, 0), (TRIGGER_LINE_X, FRAME_HEIGHT), (0, 0, 255), 2)

        status_color = (0, 255, 100) if detection_enabled else (100, 100, 100)
        cv2.putText(frame, f"{'DETECTING' if detection_enabled else 'IDLE'}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        cv2.putText(frame, f"Fish: {total_fish}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.putText(frame, f"FPS: {current_fps:.1f}",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        # Update frame untuk web stream
        with frame_lock:
            latest_frame = frame.copy()

        # FPS Limiter
        elapsed = time.time() - loop_start
        sleep_time = FRAME_TIME - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# Jalankan detection loop di background thread
detection_thread = threading.Thread(target=detection_loop, daemon=True)
detection_thread.start()

# =============================
# FASTAPI ENDPOINTS
# =============================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve dashboard HTML"""
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())


def generate_mjpeg():
    """Generator untuk MJPEG stream"""
    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        ret, buffer = cv2.imencode(
            '.jpg', frame,
            [cv2.IMWRITE_JPEG_QUALITY, 70]
        )
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() +
            b'\r\n'
        )

        time.sleep(1.0 / 20)  # max 20fps untuk stream


@app.get("/video_feed")
def video_feed():
    """MJPEG video stream endpoint"""
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket untuk realtime stats"""
    await websocket.accept()
    try:
        while True:
            stats = get_system_stats()
            await websocket.send_json(stats)
            await asyncio.sleep(0.5)  # update setiap 500ms
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS ERROR] {e}")


@app.post("/start_detection")
async def start_detection():
    global detection_enabled, system_status
    detection_enabled = True
    system_status = "RUNNING"
    print("[API] Detection STARTED")
    return {"status": "ok", "message": "Detection started"}


@app.post("/stop_detection")
async def stop_detection():
    global detection_enabled, system_status
    detection_enabled = False
    system_status = "IDLE"
    print("[API] Detection STOPPED")
    return {"status": "ok", "message": "Detection stopped"}


@app.post("/reset_counter")
async def reset_counter():
    global total_fish, processed_ids, last_positions
    total_fish = 0
    processed_ids.clear()
    last_positions.clear()
    print("[API] Counter RESET")
    return {"status": "ok", "message": "Counter reset"}


@app.post("/servo_test")
async def run_servo_test():
    servo_test_sequence()
    print("[API] Servo TEST started")
    return {"status": "ok", "message": "Servo test running: 40° → 80° → 40°"}


# =============================
# ENTRY POINT
# =============================

if __name__ == "__main__":
    print("=" * 50)
    print("  Catfish Sorting Dashboard V1")
    print("  Buka browser: http://<RaspberryPi-IP>:8000")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
