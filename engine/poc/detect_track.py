import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# Classi COCO usate dal modello pretrained: 0 = person, 32 = sports ball
TRACKED_CLASSES = [0, 32]


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

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"{video_path.stem}_tracked.mp4"

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    model = YOLO("yolov8n.pt")
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

    for result in results:
        annotated = result.plot()
        writer.write(annotated)
        frame_count += 1
        if frame_count % 200 == 0:
            print(f"  ...{frame_count} frame elaborati")

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


if __name__ == "__main__":
    main()
