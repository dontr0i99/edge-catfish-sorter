"""
Catfish Sorting System - Dashboard V2
YOLOv11 NCNN + ByteTrack + GradeStabilizer + Servo Control + FastAPI Dashboard

Fitur baru v2:
- GradeStabilizer: stabilisasi label grade berdasarkan histori deteksi
  * Prioritas: anomali > normal
  * Grade terkunci ke prioritas tertinggi, tidak bisa turun
- Fish counter terpisah per grade (normal & anomali)
"""

import cv2
import time
import threading
import asyncio
import numpy as np
import psutil
import os
from collections import defaultdict
from ultralytics import YOLO
from adafruit_servokit import ServoKit

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
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
# GRADE STABILIZER
# =============================

class GradeStabilizer:
    """
    Menstabilkan label grade objek berdasarkan histori deteksi.
    Prioritas: anomali > normal
    Grade terkunci ke prioritas tertinggi dan tidak bisa turun.
    """

    GRADE_PRIORITY = {
        "anomali": 2,
        "normal":  1,
        "unknown": 0,
    }

    MAX_HISTORY_SIZE      = 9
    MIN_DETECTIONS_TO_LOCK = 3

    def __init__(self):
        # {id: [(label, confidence), ...]}
        self._history: dict[int, list[tuple[str, float]]] = defaultdict(list)
        # {id: str} — grade tertinggi yang sudah terkunci
        self._locked: dict[int, str] = {}

    def add_detection(self, obj_id: int, label: str, confidence: float):
        history = self._history[obj_id]
        history.append((label, confidence))
        if len(history) > self.MAX_HISTORY_SIZE:
            history.pop(0)
        self._update_locked(obj_id)

    def _update_locked(self, obj_id: int):
        history = self._history[obj_id]
        if not history:
            return

        freq: dict[str, int] = defaultdict(int)
        for lbl, _ in history:
            freq[lbl] += 1

        # kandidat yang memenuhi batas minimum deteksi
        candidates = [lbl for lbl, cnt in freq.items() if cnt >= self.MIN_DETECTIONS_TO_LOCK]
        if not candidates:
            return

        best_candidate = max(candidates, key=lambda l: self.GRADE_PRIORITY.get(l, 0))

        current_locked = self._locked.get(obj_id)
        if current_locked is None:
            self._locked[obj_id] = best_candidate
        else:
            current_prio   = self.GRADE_PRIORITY.get(current_locked, 0)
            candidate_prio = self.GRADE_PRIORITY.get(best_candidate, 0)
            if candidate_prio > current_prio:
                self._locked[obj_id] = best_candidate

    def get_stable_label(self, obj_id: int) -> str:
        locked = self._locked.get(obj_id)
        if locked is not None:
            return locked

        history = self._history.get(obj_id, [])
        if not history:
            return "unknown"
        if len(history) == 1:
            return history[0][0]

        # belum terkunci: pakai label dengan prioritas tertinggi dalam histori
        return max(
            (lbl for lbl, _ in history),
            key=lambda l: self.GRADE_PRIORITY.get(l, 0),
            default="unknown",
        )

    def remove_history(self, obj_id: int):
        self._history.pop(obj_id, None)
        self._locked.pop(obj_id, None)

    def reset(self):
        self._history.clear()
        self._locked.clear()


# =============================
# GLOBAL STATE
# =============================

latest_frame = None
frame_lock   = threading.Lock()

detection_enabled = False
servo_status  = "READY"
system_status = "IDLE"

total_fish    = 0
total_normal  = 0
total_anomali = 0
current_fps   = 0.0
last_servo_time = 0

processed_ids  = set()
last_positions = {}

grade_stabilizer = GradeStabilizer()

# =============================
# FASTAPI APP
# =============================

app = FastAPI(title="Catfish Sorting Dashboard V2")

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
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return round(int(f.read().strip()) / 1000.0, 1)
    except Exception:
        try:
            result = os.popen("vcgencmd measure_temp").readline()
            return round(float(result.replace("temp=", "").replace("'C\n", "")), 1)
        except Exception:
            return 0.0


def get_system_stats():
    return {
        "total_fish":    total_fish,
        "total_normal":  total_normal,
        "total_anomali": total_anomali,
        "fps":           round(current_fps, 1),
        "cpu_temp":      get_cpu_temperature(),
        "cpu_usage":     psutil.cpu_percent(interval=None),
        "ram_usage":     psutil.virtual_memory().percent,
        "servo_status":  servo_status,
        "system_status": system_status,
        "detection_enabled": detection_enabled,
    }

# =============================
# LOAD MODEL
# =============================

print("Loading YOLO model...")
model  = YOLO(MODEL_PATH, task='detect')
labels = model.names
print("Model loaded OK")

# =============================
# CAMERA
# =============================

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: Kamera tidak bisa dibuka!")
    exit(1)

# =============================
# BBOX COLOR PER GRADE
# =============================

GRADE_COLOR = {
    "normal":  (0, 255, 80),    # hijau
    "anomali": (0, 80, 255),    # merah-oranye (BGR)
    "unknown": (120, 120, 120), # abu
}

# =============================
# MAIN DETECTION LOOP (Thread)
# =============================

def detection_loop():
    global latest_frame, detection_enabled, system_status
    global total_fish, total_normal, total_anomali
    global current_fps, last_servo_time
    global processed_ids, last_positions

    frame_count    = 0
    cached_results = None
    prev_time      = time.time()

    print("Detection loop started.")

    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("WARNING: Gagal baca frame")
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        current_time = time.time()
        current_fps  = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time    = current_time

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
                            verbose=False,
                        )
                    except Exception as e:
                        print(f"[TRACK ERROR] {e}")
                        cached_results = None

                results = cached_results

                if results is not None and len(results) > 0:
                    boxes = results[0].boxes

                    if boxes is not None and boxes.id is not None:
                        ids       = boxes.id.cpu().numpy().astype(int)
                        xyxy      = boxes.xyxy.cpu().numpy()
                        classes   = boxes.cls.cpu().numpy().astype(int)
                        confs     = boxes.conf.cpu().numpy()

                        active_ids = set(ids.tolist())

                        for box, obj_id, cls, conf in zip(xyxy, ids, classes, confs):

                            if np.any(np.isnan(box)) or np.any(np.isinf(box)):
                                continue

                            xmin, ymin, xmax, ymax = map(int, box)

                            if xmin >= xmax or ymin >= ymax:
                                continue

                            w = xmax - xmin
                            h = ymax - ymin
                            if w < 20 or h < 20:
                                continue

                            raw_label = labels[cls]

                            # tambahkan ke stabilizer setiap frame
                            grade_stabilizer.add_detection(int(obj_id), raw_label, float(conf))
                            stable_label = grade_stabilizer.get_stable_label(int(obj_id))

                            # koordinat absolut
                            xmin += ROI_X1;  xmax += ROI_X1
                            ymin += ROI_Y1;  ymax += ROI_Y1
                            cx = (xmin + xmax) // 2

                            color = GRADE_COLOR.get(stable_label, GRADE_COLOR["unknown"])
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                            cv2.putText(
                                frame,
                                f"{stable_label} ID:{obj_id}",
                                (xmin, max(ymin - 10, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                            )

                            # crossing detection
                            if obj_id in last_positions:
                                prev_x = last_positions[obj_id]
                                if prev_x < TRIGGER_LINE_X and cx >= TRIGGER_LINE_X:
                                    if obj_id not in processed_ids:
                                        now = time.time()
                                        if now - last_servo_time > SERVO_COOLDOWN:
                                            processed_ids.add(obj_id)
                                            last_servo_time = now

                                            total_fish += 1
                                            if stable_label == "normal":
                                                total_normal += 1
                                            elif stable_label == "anomali":
                                                total_anomali += 1

                                            print(
                                                f"[INFO] {stable_label} ID {obj_id} lewat | "
                                                f"Normal: {total_normal}  Anomali: {total_anomali}  "
                                                f"Total: {total_fish}"
                                            )
                                            set_servo_nonblocking(40, 80)

                            last_positions[obj_id] = cx

                        # bersihkan histori ID yang tidak aktif lagi
                        stale_ids = set(grade_stabilizer._history.keys()) - active_ids
                        for sid in stale_ids:
                            if sid not in processed_ids:
                                grade_stabilizer.remove_history(sid)

        # =============================
        # VISUAL OVERLAY
        # =============================

        cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 100, 0), 2)
        cv2.line(frame, (TRIGGER_LINE_X, 0), (TRIGGER_LINE_X, FRAME_HEIGHT), (0, 0, 255), 2)

        status_color = (0, 255, 100) if detection_enabled else (100, 100, 100)
        cv2.putText(frame, "DETECTING" if detection_enabled else "IDLE",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        cv2.putText(frame, f"Normal: {total_normal}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)
        cv2.putText(frame, f"Anomali: {total_anomali}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)
        cv2.putText(frame, f"FPS: {current_fps:.1f}",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        with frame_lock:
            latest_frame = frame.copy()

        elapsed    = time.time() - loop_start
        sleep_time = FRAME_TIME - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


detection_thread = threading.Thread(target=detection_loop, daemon=True)
detection_thread.start()

# =============================
# FASTAPI ENDPOINTS
# =============================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("index-v2.html", "r") as f:
        return HTMLResponse(content=f.read())


def generate_mjpeg():
    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() +
            b'\r\n'
        )
        time.sleep(1.0 / 20)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(get_system_stats())
            await asyncio.sleep(0.5)
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
    global total_fish, total_normal, total_anomali, processed_ids, last_positions
    total_fish    = 0
    total_normal  = 0
    total_anomali = 0
    processed_ids.clear()
    last_positions.clear()
    grade_stabilizer.reset()
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
    print("  Catfish Sorting Dashboard V2")
    print("  Buka browser: http://<RaspberryPi-IP>:8000")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
