import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from detect_track import MODEL_NAME, OUTPUT_DIR, find_input_video
from team_classification import MIN_SAMPLES_PER_TRACK, color_signature, torso_crop
from video_codec import cv2

# Firma colore mediana per traccia, salvata su disco separatamente dal resto
# della pipeline: serve a riusare i dati (classificazione cluster, audit,
# soglie) senza dover ridecodificare tutto il video ogni volta.


def main():
    video_path = find_input_video()
    model_tag = Path(MODEL_NAME).stem
    tracks_csv_path = OUTPUT_DIR / f"{video_path.stem}_{model_tag}_tracks.csv"

    frame_tracks = defaultdict(list)
    with open(tracks_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["cls"]) != 0:
                continue
            frame_tracks[int(row["frame_idx"])].append({
                "track_id": int(row["track_id"]),
                "bbox": (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
            })

    signatures = defaultdict(list)
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for det in frame_tracks.get(frame_idx, []):
            crop = torso_crop(frame, det["bbox"])
            sig = color_signature(crop)
            if sig is not None:
                signatures[det["track_id"]].append(sig)
        frame_idx += 1
        if frame_idx % 5000 == 0:
            print(f"  ...{frame_idx} frame")
    cap.release()

    median_sigs = {
        str(tid): np.median(np.stack(sigs), axis=0).tolist()
        for tid, sigs in signatures.items()
        if len(sigs) >= MIN_SAMPLES_PER_TRACK
    }

    out_path = OUTPUT_DIR / f"{video_path.stem}_track_signatures.json"
    with open(out_path, "w") as f:
        json.dump(median_sigs, f)
    print(f"Firme salvate per {len(median_sigs)} tracce in: {out_path}")


if __name__ == "__main__":
    main()
