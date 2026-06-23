"""
Perekam Data Test - Catfish Sorter
Merekam video dari kamera dengan KONFIGURASI & ROI yang SAMA seperti main-v4.py,
supaya hasil rekaman bisa dipakai untuk test/replay pipeline deteksi.

Cara pakai:
    python record_test_data.py
    python record_test_data.py --name uji_normal_01
    python record_test_data.py --duration 30          # auto stop setelah 30 detik
    python record_test_data.py --no-preview           # mode headless (tanpa window)

Kontrol (saat window preview aktif):
    s        -> START merekam
    e        -> STOP merekam (file disimpan)
    SPACE    -> pause/resume perekaman
    q / ESC  -> keluar (otomatis simpan jika masih merekam)

Catatan:
- Frame yang DISIMPAN adalah frame BERSIH (tanpa overlay ROI/garis),
  agar bisa di-feed ulang ke detektor dengan kondisi sama.
- Overlay ROI + garis trigger HANYA ditampilkan di window preview sebagai panduan.
"""

import cv2
import time
import os
import argparse
from datetime import datetime

# =============================
# KONFIGURASI (samakan dengan main-v4.py)
# =============================

FRAME_WIDTH  = 640
FRAME_HEIGHT = 360

TARGET_FPS = 15
FRAME_TIME = 1.0 / TARGET_FPS

# ROI — HARUS SAMA dengan main-v4.py
ROI_X1 = 80
ROI_Y1 = 70
ROI_X2 = 640
ROI_Y2 = 285

# Garis trigger — sama dengan main-v4.py
TRIGGER_LINE_X = int(FRAME_WIDTH * 0.75)

OUTPUT_DIR = "test_recordings"


# =============================
# ARGUMEN
# =============================

def parse_args():
    parser = argparse.ArgumentParser(description="Perekam data test catfish sorter")
    parser.add_argument("--name", type=str, default=None,
                        help="Nama dasar file output (default: timestamp)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Durasi rekam dalam detik (0 = sampai ditekan q)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Jalankan tanpa window preview (mode headless)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Index kamera (default: 0)")
    return parser.parse_args()


# =============================
# SETUP KAMERA (sama dengan main-v4.py)
# =============================

def open_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def draw_overlay(frame):
    """Gambar panduan ROI + garis trigger (hanya untuk preview)."""
    cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 100, 0), 2)
    cv2.line(frame, (TRIGGER_LINE_X, 0), (TRIGGER_LINE_X, FRAME_HEIGHT), (0, 0, 255), 2)
    cv2.putText(frame, "ROI", (ROI_X1 + 5, ROI_Y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
    return frame


# =============================
# MAIN
# =============================

def main():
    args = parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_name = args.name or datetime.now().strftime("rec_%Y%m%d_%H%M%S")
    video_path = os.path.join(OUTPUT_DIR, f"{base_name}.mp4")
    meta_path  = os.path.join(OUTPUT_DIR, f"{base_name}.txt")

    cap = open_camera(args.camera)
    if not cap.isOpened():
        print("ERROR: Kamera tidak bisa dibuka!")
        return

    # VideoWriter — rekam pada resolusi & FPS yang sama
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, TARGET_FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    if not writer.isOpened():
        print("ERROR: VideoWriter gagal dibuka!")
        cap.release()
        return

    show_preview = not args.no_preview
    # Mode headless tidak punya input tombol -> langsung merekam.
    recording = not show_preview
    paused = False
    frame_count = 0
    start_time = None  # diisi saat perekaman pertama kali dimulai

    print("=" * 55)
    print("  Perekam Data Test - Catfish Sorter")
    print(f"  Output      : {video_path}")
    print(f"  Resolusi    : {FRAME_WIDTH}x{FRAME_HEIGHT} @ {TARGET_FPS} FPS")
    print(f"  ROI         : ({ROI_X1},{ROI_Y1}) - ({ROI_X2},{ROI_Y2})")
    print(f"  Trigger X   : {TRIGGER_LINE_X}")
    if args.duration > 0:
        print(f"  Durasi      : {args.duration} detik")
    if show_preview:
        print("  Kontrol     : s = start, e = stop, SPACE = pause, q/ESC = keluar")
        print("  Status      : MENUNGGU (tekan 's' untuk mulai merekam)")
    print("=" * 55)

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("WARNING: Gagal baca frame")
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Simpan frame BERSIH (tanpa overlay) untuk replay yang akurat
            if recording and not paused:
                if start_time is None:
                    start_time = time.time()
                writer.write(frame)
                frame_count += 1

            elapsed_total = (time.time() - start_time) if start_time else 0.0

            if show_preview:
                preview = draw_overlay(frame.copy())
                if not recording:
                    status, status_color = "STANDBY", (180, 180, 180)
                elif paused:
                    status, status_color = "PAUSED", (0, 200, 255)
                else:
                    status, status_color = "REC", (0, 0, 255)
                cv2.putText(preview, status, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(preview, f"Frames: {frame_count}  t: {elapsed_total:.1f}s",
                            (10, FRAME_HEIGHT - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Recorder - Catfish Sorter", preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q atau ESC = keluar
                    break
                elif key == ord('s'):             # start merekam
                    if not recording:
                        recording = True
                        print("[START] Mulai merekam")
                elif key == ord('e'):             # stop merekam & simpan
                    if recording:
                        print("[STOP] Berhenti merekam")
                        break
                elif key == ord(' '):             # pause/resume
                    if recording:
                        paused = not paused
                        print("[PAUSE]" if paused else "[RESUME]")

            # Auto stop berdasarkan durasi (dihitung sejak rekaman dimulai)
            if args.duration > 0 and recording and elapsed_total >= args.duration:
                print(f"[AUTO STOP] Durasi {args.duration}s tercapai")
                break

            # Jaga FPS konsisten
            elapsed = time.time() - loop_start
            sleep_time = FRAME_TIME - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Dihentikan oleh user")

    finally:
        duration = (time.time() - start_time) if start_time else 0.0
        cap.release()
        writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        # Simpan metadata supaya kondisi rekaman terdokumentasi
        with open(meta_path, "w") as f:
            f.write("Catfish Sorter - Test Recording Metadata\n")
            f.write(f"timestamp     : {datetime.now().isoformat()}\n")
            f.write(f"video         : {os.path.basename(video_path)}\n")
            f.write(f"resolution    : {FRAME_WIDTH}x{FRAME_HEIGHT}\n")
            f.write(f"target_fps    : {TARGET_FPS}\n")
            f.write(f"frames        : {frame_count}\n")
            f.write(f"duration_sec  : {duration:.2f}\n")
            f.write(f"roi           : ({ROI_X1},{ROI_Y1})-({ROI_X2},{ROI_Y2})\n")
            f.write(f"trigger_x     : {TRIGGER_LINE_X}\n")

        print("=" * 55)
        print(f"  Selesai. {frame_count} frame ({duration:.1f}s) tersimpan.")
        print(f"  Video    : {video_path}")
        print(f"  Metadata : {meta_path}")
        print("=" * 55)


if __name__ == "__main__":
    main()
