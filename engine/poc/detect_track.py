import csv
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from video_codec import FOURCC, H264_AVAILABLE, cv2

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# Classi COCO usate dal modello pretrained: 0 = person, 32 = sports ball
TRACKED_CLASSES = [0, 32]

# Limite frame per iterazioni rapide durante lo sviluppo. None = video intero.
MAX_FRAMES = None

# Nome modello YOLO pretrained (yolov8n/s/m/l/x.pt)
# yolov8m scelto dopo confronto n vs m su 3000 frame: quasi 2x detection/frame,
# confidence piu alta ovunque (specie palla, +35%), track piu lunghe e piu stabili.
MODEL_NAME = "yolov8m.pt"


def find_input_video() -> Path:
    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            return path
    print(f"Nessun video trovato in {DATA_DIR}")
    print("Copia un file .mp4/.mov/.avi/.mkv in quella cartella e rilancia lo script.")
    sys.exit(1)


def main():
    video_path = find_input_video()
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Video di input: {video_path}")
    print(f"Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if device == 0 else 'CPU'}")
    print(f"Codec output: {'H.264 (avc1)' if H264_AVAILABLE else 'mp4v (fallback, DLL openh264 non trovata - vedi README)'}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    model_tag = Path(MODEL_NAME).stem
    output_path = OUTPUT_DIR / f"{video_path.stem}_{model_tag}_tracked.mp4"
    tracks_csv_path = OUTPUT_DIR / f"{video_path.stem}_{model_tag}_tracks.csv"

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = cv2.VideoWriter(str(output_path), FOURCC, fps, (width, height))

    model = YOLO(MODEL_NAME)
    if device == 0:
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    frame_count = 0

    results = model.track(
        source=str(video_path),
        classes=TRACKED_CLASSES,
        tracker="bytetrack.yaml",
        device=device,
        stream=True,
        verbose=False,
    )

    with open(tracks_csv_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["frame_idx", "track_id", "cls", "x1", "y1", "x2", "y2", "conf"])

        for result in results:
            annotated = result.plot()
            writer.write(annotated)

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.int().tolist()
                classes = boxes.cls.int().tolist()
                confs = boxes.conf.tolist()
                xyxy = boxes.xyxy.tolist()
                for track_id, cls_id, conf, (x1, y1, x2, y2) in zip(ids, classes, confs, xyxy):
                    csv_writer.writerow([frame_count, track_id, cls_id, f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", f"{conf:.3f}"])

            frame_count += 1
            if frame_count % 200 == 0:
                print(f"  ...{frame_count} frame elaborati")
            if MAX_FRAMES is not None and frame_count >= MAX_FRAMES:
                break

    writer.release()
    elapsed = time.time() - start

    print("\n--- Risultati PoC ---")
    print(f"Frame totali: {frame_count}")
    print(f"Tempo elaborazione: {elapsed:.1f}s")
    print(f"FPS medio elaborazione: {frame_count / elapsed:.1f}")
    if device == 0:
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Picco memoria GPU: {peak_mem_gb:.2f} GB")
    print(f"Video annotato salvato in: {output_path}")
    print(f"Dati di tracking salvati in: {tracks_csv_path}")


if __name__ == "__main__":
    main()
