import csv
import json
from collections import defaultdict
from pathlib import Path

from detect_track import MODEL_NAME, OUTPUT_DIR, find_input_video

# Distanza massima (in pixel, video 640px di larghezza) tra palla e giocatore
# piu' vicino perche' il possesso venga assegnato. Oltre questa soglia il
# frame e' "indeterminato" (palla in aria, lancio lungo, nessuno vicino)
# invece di forzare un'assegnazione arbitraria al giocatore piu' vicino
# comunque. ~23% della larghezza frame.
MAX_DIST_PX = 150

# Durata bucket per la timeline (secondi)
BUCKET_SECONDS = 300  # 5 minuti

FPS = 24  # da detect_track.py / video sorgente


def load_detections(csv_path: Path):
    ball_by_frame = {}
    persons_by_frame = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame_idx"])
            bbox = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            if int(row["cls"]) == 32:
                # se piu' palle nello stesso frame, tiene quella con confidenza piu' alta
                conf = float(row["conf"])
                prev = ball_by_frame.get(frame_idx)
                if prev is None or conf > prev[1]:
                    ball_by_frame[frame_idx] = (bbox, conf)
            else:
                persons_by_frame[frame_idx].append((int(row["track_id"]), bbox))
    return ball_by_frame, persons_by_frame


def load_team_map(video_stem: str, model_tag: str):
    clusters_path = OUTPUT_DIR / f"{video_stem}_team_clusters.json"
    labels_path = OUTPUT_DIR / f"{video_stem}_cluster_labels.json"
    clusters_data = json.loads(clusters_path.read_text())
    labels_data = json.loads(labels_path.read_text())

    track_to_cluster = {int(tid): cid for tid, cid in clusters_data["track_to_cluster"].items()}
    cluster_to_label = labels_data["labels"]
    default_label = labels_data.get("default_label", "non_classificato")

    track_to_team = {}
    for tid, cid in track_to_cluster.items():
        track_to_team[tid] = cluster_to_label.get(str(cid), default_label)
    return track_to_team


def foot_point(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)


def ball_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def assign_possession(ball_by_frame, persons_by_frame, track_to_team):
    per_frame = {}  # frame_idx -> "squadra_blu" | "squadra_rossa" | "indeterminato"
    for frame_idx, (bbox, _conf) in ball_by_frame.items():
        bx, by = ball_center(bbox)
        persons = persons_by_frame.get(frame_idx, [])
        best_team = "indeterminato"
        best_d = MAX_DIST_PX
        for tid, pbbox in persons:
            team = track_to_team.get(tid, "non_classificato")
            if team not in ("squadra_blu", "squadra_rossa"):
                continue
            d = dist((bx, by), foot_point(pbbox))
            if d <= best_d:
                best_d = d
                best_team = team
        per_frame[frame_idx] = best_team
    return per_frame


def summarize(per_frame, total_frames):
    counts = defaultdict(int)
    for team in per_frame.values():
        counts[team] += 1
    n_ball_frames = len(per_frame)

    def pct_of_ball_frames(team):
        return round(100 * counts.get(team, 0) / n_ball_frames, 1) if n_ball_frames else 0.0

    return {
        "frame_totali_partita": total_frames,
        "frame_con_palla_rilevata": n_ball_frames,
        "copertura_palla_pct": round(100 * n_ball_frames / total_frames, 1),
        "possesso_su_frame_con_palla": {
            "squadra_blu_pct": pct_of_ball_frames("squadra_blu"),
            "squadra_rossa_pct": pct_of_ball_frames("squadra_rossa"),
            "indeterminato_pct": pct_of_ball_frames("indeterminato"),
        },
        "possesso_su_frame_con_palla_e_assegnato": (
            lambda blu, rossa: {
                "squadra_blu_pct": round(100 * blu / (blu + rossa), 1) if (blu + rossa) else 0.0,
                "squadra_rossa_pct": round(100 * rossa / (blu + rossa), 1) if (blu + rossa) else 0.0,
                "nota": "possesso relativo ai soli frame in cui un giocatore e' stato assegnato (esclude gli indeterminati) - la lettura piu' vicina al 'possesso palla' classico",
            }
        )(counts.get("squadra_blu", 0), counts.get("squadra_rossa", 0)),
    }


def timeline(per_frame, total_frames):
    bucket_frames = BUCKET_SECONDS * FPS
    n_buckets = total_frames // bucket_frames + 1
    buckets = []
    for b in range(n_buckets):
        start = b * bucket_frames
        end = min(start + bucket_frames, total_frames)
        counts = defaultdict(int)
        for frame_idx, team in per_frame.items():
            if start <= frame_idx < end:
                counts[team] += 1
        n_ball = sum(counts.values())
        blu, rossa = counts.get("squadra_blu", 0), counts.get("squadra_rossa", 0)
        assegnati = blu + rossa
        buckets.append({
            "minuto_inizio": round(start / FPS / 60, 1),
            "minuto_fine": round(end / FPS / 60, 1),
            "frame_con_palla": n_ball,
            "squadra_blu_pct": round(100 * blu / assegnati, 1) if assegnati else None,
            "squadra_rossa_pct": round(100 * rossa / assegnati, 1) if assegnati else None,
        })
    return buckets


def main():
    video_path = find_input_video()
    model_tag = Path(MODEL_NAME).stem
    tracks_csv_path = OUTPUT_DIR / f"{video_path.stem}_{model_tag}_tracks.csv"

    print(f"Tracking data: {tracks_csv_path}")
    ball_by_frame, persons_by_frame = load_detections(tracks_csv_path)
    total_frames = max(
        max(ball_by_frame.keys(), default=0),
        max(persons_by_frame.keys(), default=0),
    ) + 1
    print(f"Frame totali: {total_frames}, frame con palla: {len(ball_by_frame)}")

    track_to_team = load_team_map(video_path.stem, model_tag)
    print(f"Tracce con etichetta squadra: {sum(1 for t in track_to_team.values() if t in ('squadra_blu', 'squadra_rossa'))}")

    print("Assegnazione possesso per frame...")
    per_frame = assign_possession(ball_by_frame, persons_by_frame, track_to_team)

    summary = summarize(per_frame, total_frames)
    tl = timeline(per_frame, total_frames)

    out_path = OUTPUT_DIR / f"{video_path.stem}_possession.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "timeline": tl, "max_dist_px": MAX_DIST_PX}, f, indent=2, ensure_ascii=False)

    print("\n--- Possesso palla ---")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSalvato: {out_path}")


if __name__ == "__main__":
    main()
