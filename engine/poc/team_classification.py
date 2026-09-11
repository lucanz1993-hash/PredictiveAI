import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from detect_track import MODEL_NAME, OUTPUT_DIR, find_input_video
from video_codec import FOURCC, cv2

# Numero di cluster attesi: squadra propria, avversari, portiere proprio,
# portiere avversario, arbitro. Da regolare dopo aver visto i risultati.
N_CLUSTERS = 5

# Limite frame per iterazioni rapide durante lo sviluppo. None = tutti i frame nel CSV.
MAX_FRAMES = None

# Regione torso come frazione dell'altezza/larghezza del bounding box persona
TORSO_TOP, TORSO_BOTTOM = 0.15, 0.55
TORSO_MARGIN_X = 0.15

# Maschera "verde campo" in HSV (scala hue OpenCV 0-179)
GREEN_HUE_MIN, GREEN_HUE_MAX = 35, 90
GREEN_SAT_MIN = 40

# Tonalità dominante del crop: istogramma pesato per saturazione (bin da 10° su
# scala OpenCV 0-179) invece di una media circolare grezza. Una media mischia
# colori diversi presenti nello stesso crop (es. maglia rossa + calzettoni gialli)
# in una tonalità intermedia che non corrisponde a nessuno dei due capi.
HUE_BINS = 18
# Sotto questa saturazione media il colore non è affidabile (maglie bianche/nere/
# grigie): si rinuncia alla tonalità e ci si affida a saturazione/luminosità.
MIN_SAT_FOR_HUE = 30

MIN_VALID_PIXEL_FRACTION = 0.2
MIN_SAMPLES_PER_TRACK = 5

THUMB_SIZE = (60, 90)  # (w, h)
SAMPLES_PER_CLUSTER = 6
# Campione riservato di crop per traccia, usato per scegliere la thumbnail più
# vicina alla firma colore aggregata (mediana) invece del primo crop trovato.
RESERVOIR_SIZE = 15

PALETTE_BGR = [
    (0, 0, 255),    # rosso
    (255, 0, 0),    # blu
    (0, 255, 255),  # giallo
    (255, 0, 255),  # magenta
    (0, 165, 255),  # arancione
    (255, 255, 255),  # bianco
]
BALL_COLOR_BGR = (255, 255, 0)  # ciano


def load_tracks(csv_path: Path):
    frame_tracks = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_tracks[int(row["frame_idx"])].append({
                "track_id": int(row["track_id"]),
                "cls": int(row["cls"]),
                "bbox": (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
            })
    return frame_tracks


def torso_crop(frame, bbox):
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = x2 - x1
    top = y1 + TORSO_TOP * h
    bottom = y1 + TORSO_BOTTOM * h
    left = x1 + TORSO_MARGIN_X * w
    right = x2 - TORSO_MARGIN_X * w
    return frame[max(0, int(top)):int(bottom), max(0, int(left)):int(right)]


def color_signature(crop):
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(np.float32), hsv[:, :, 1].astype(np.float32), hsv[:, :, 2].astype(np.float32)

    green_mask = (h >= GREEN_HUE_MIN) & (h <= GREEN_HUE_MAX) & (s >= GREEN_SAT_MIN)
    valid = ~green_mask
    if valid.sum() < MIN_VALID_PIXEL_FRACTION * valid.size:
        return None

    h_valid, s_valid, v_valid = h[valid], s[valid], v[valid]

    if s_valid.mean() >= MIN_SAT_FOR_HUE:
        # tonalità dominante = picco dell'istogramma pesato per saturazione,
        # robusta a colori minoritari nel crop (pelle, ombre, sfondo residuo)
        hist, edges = np.histogram(h_valid, bins=HUE_BINS, range=(0, 180), weights=s_valid)
        peak_deg = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
        hue_rad = peak_deg * (2 * np.pi / 180)
        hue_x, hue_y = float(np.cos(hue_rad)), float(np.sin(hue_rad))
    else:
        # maglia bianca/nera/grigia: tonalità non affidabile, si contano solo
        # saturazione e luminosità medie per distinguerla
        hue_x, hue_y = 0.0, 0.0

    return np.array([hue_x, hue_y, s_valid.mean() / 255, v_valid.mean() / 255])


def make_thumbnail(crop):
    if crop.size == 0:
        return np.zeros((THUMB_SIZE[1], THUMB_SIZE[0], 3), dtype=np.uint8)
    return cv2.resize(crop, THUMB_SIZE)


def extract_signatures(video_path: Path, frame_tracks: dict):
    signatures = defaultdict(list)
    # campione a dimensione limitata di (firma, thumbnail) per traccia, via
    # reservoir sampling: memoria costante anche su video interi con migliaia
    # di frame, invece di tenere ogni crop di ogni traccia
    reservoir = defaultdict(list)
    seen_count = defaultdict(int)

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for det in frame_tracks.get(frame_idx, []):
            if det["cls"] != 0:  # solo persone, non la palla
                continue
            crop = torso_crop(frame, det["bbox"])
            sig = color_signature(crop)
            if sig is not None:
                tid = det["track_id"]
                signatures[tid].append(sig)

                seen_count[tid] += 1
                pool = reservoir[tid]
                if len(pool) < RESERVOIR_SIZE:
                    pool.append((sig, make_thumbnail(crop)))
                else:
                    j = random.randint(0, seen_count[tid] - 1)
                    if j < RESERVOIR_SIZE:
                        pool[j] = (sig, make_thumbnail(crop))

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  ...{frame_idx} frame analizzati per il colore")
        if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES:
            break
    cap.release()
    return signatures, reservoir


def pick_representative_thumbs(track_to_cluster: dict, signatures: dict, reservoir: dict):
    """Per ogni traccia, sceglie tra il campione riservato il crop la cui firma è
    più vicina alla firma aggregata (mediana) usata per il clustering — non il
    primo crop incontrato, che può essere un frame con motion blur o occlusione
    parziale non rappresentativo del colore reale della maglia."""
    representative_thumb = {}
    for tid in track_to_cluster:
        pool = reservoir.get(tid)
        if not pool:
            continue
        median_sig = np.median(np.stack(signatures[tid]), axis=0)
        _, best_thumb = min(pool, key=lambda item: np.linalg.norm(item[0] - median_sig))
        representative_thumb[tid] = best_thumb
    return representative_thumb


def cluster_tracks(signatures: dict):
    track_ids = [tid for tid, sigs in signatures.items() if len(sigs) >= MIN_SAMPLES_PER_TRACK]
    if len(track_ids) < N_CLUSTERS:
        print(f"Solo {len(track_ids)} tracce valide, insufficienti per {N_CLUSTERS} cluster.")
        sys.exit(1)

    features = np.array([np.median(np.stack(signatures[tid]), axis=0) for tid in track_ids])
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=0)
    labels = km.fit_predict(features)

    track_to_cluster = {tid: int(label) for tid, label in zip(track_ids, labels)}
    return track_to_cluster, features, labels


def save_cluster_json(output_path: Path, track_to_cluster: dict, signatures: dict):
    clusters = defaultdict(list)
    for tid, cluster_id in track_to_cluster.items():
        clusters[cluster_id].append(tid)

    data = {
        "track_to_cluster": track_to_cluster,
        "clusters": {
            str(cid): {
                "track_ids": tids,
                "sample_count": sum(len(signatures[tid]) for tid in tids),
            }
            for cid, tids in clusters.items()
        },
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def save_cluster_collage(output_path: Path, track_to_cluster: dict, representative_thumb: dict):
    clusters = defaultdict(list)
    for tid, cluster_id in track_to_cluster.items():
        if tid in representative_thumb:
            clusters[cluster_id].append(tid)

    rows = []
    label_width = 80
    for cluster_id in sorted(clusters):
        sample_ids = clusters[cluster_id][:SAMPLES_PER_CLUSTER]
        thumbs = [representative_thumb[tid] for tid in sample_ids]
        while len(thumbs) < SAMPLES_PER_CLUSTER:
            thumbs.append(np.zeros((THUMB_SIZE[1], THUMB_SIZE[0], 3), dtype=np.uint8))
        row_imgs = np.hstack(thumbs)

        label_panel = np.zeros((THUMB_SIZE[1], label_width, 3), dtype=np.uint8)
        cv2.putText(label_panel, f"C{cluster_id}", (10, THUMB_SIZE[1] // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        rows.append(np.hstack([label_panel, row_imgs]))

    collage = np.vstack(rows)
    cv2.imwrite(str(output_path), collage)


def render_teams_video(video_path: Path, frame_tracks: dict, track_to_cluster: dict, output_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), FOURCC, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for det in frame_tracks.get(frame_idx, []):
            x1, y1, x2, y2 = (int(v) for v in det["bbox"])
            if det["cls"] == 0:
                cluster_id = track_to_cluster.get(det["track_id"])
                color = PALETTE_BGR[cluster_id % len(PALETTE_BGR)] if cluster_id is not None else (128, 128, 128)
                label = f"C{cluster_id}" if cluster_id is not None else "?"
            else:
                color = BALL_COLOR_BGR
                label = "ball"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        writer.write(frame)
        frame_idx += 1
        if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES:
            break
    cap.release()
    writer.release()


def main():
    video_path = find_input_video()
    model_tag = Path(MODEL_NAME).stem
    tracks_csv_path = OUTPUT_DIR / f"{video_path.stem}_{model_tag}_tracks.csv"
    if not tracks_csv_path.exists():
        print(f"Non trovo {tracks_csv_path}")
        print("Lancia prima detect_track.py per generare i dati di tracking.")
        sys.exit(1)

    print(f"Video: {video_path}")
    print(f"Tracking data: {tracks_csv_path}")

    frame_tracks = load_tracks(tracks_csv_path)

    print("Estrazione firme colore per traccia...")
    signatures, reservoir = extract_signatures(video_path, frame_tracks)
    print(f"Tracce con almeno una firma valida: {len(signatures)}")

    print("Clustering...")
    track_to_cluster, _, _ = cluster_tracks(signatures)
    representative_thumb = pick_representative_thumbs(track_to_cluster, signatures, reservoir)

    clusters_json_path = OUTPUT_DIR / f"{video_path.stem}_team_clusters.json"
    save_cluster_json(clusters_json_path, track_to_cluster, signatures)
    print(f"Cluster salvati in: {clusters_json_path}")

    collage_path = OUTPUT_DIR / f"{video_path.stem}_cluster_samples.jpg"
    save_cluster_collage(collage_path, track_to_cluster, representative_thumb)
    print(f"Collage di esempio salvata in: {collage_path}")

    print("Rendering video annotato per squadra...")
    teams_video_path = OUTPUT_DIR / f"{video_path.stem}_teams.mp4"
    render_teams_video(video_path, frame_tracks, track_to_cluster, teams_video_path)
    print(f"Video annotato salvato in: {teams_video_path}")


if __name__ == "__main__":
    main()
