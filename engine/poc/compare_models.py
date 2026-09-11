import csv
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

MODELS = ["yolov8n", "yolov8m"]
VIDEO_STEM = "videoplayback"

PERSON_CLS = 0
BALL_CLS = 32


def analyze(model_tag):
    csv_path = OUTPUT_DIR / f"{VIDEO_STEM}_{model_tag}_tracks.csv"
    per_frame_person = defaultdict(int)
    per_frame_ball = defaultdict(int)
    person_confs = []
    ball_confs = []
    track_frame_count = defaultdict(int)
    track_frames = defaultdict(list)
    max_frame = -1

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame_idx"])
            cls = int(row["cls"])
            conf = float(row["conf"])
            track_id = int(row["track_id"])
            max_frame = max(max_frame, frame_idx)

            if cls == PERSON_CLS:
                per_frame_person[frame_idx] += 1
                person_confs.append(conf)
                track_frame_count[track_id] += 1
                track_frames[track_id].append(frame_idx)
            elif cls == BALL_CLS:
                per_frame_ball[frame_idx] += 1
                ball_confs.append(conf)

    n_frames = max_frame + 1
    total_person_dets = sum(per_frame_person.values())
    total_ball_dets = sum(per_frame_ball.values())
    frames_with_ball = len(per_frame_ball)

    n_tracks = len(track_frame_count)
    avg_track_len = sum(track_frame_count.values()) / n_tracks if n_tracks else 0
    # tracce "effimere": durano <=3 frame, segno di ID switching / falsi positivi instabili
    short_tracks = sum(1 for v in track_frame_count.values() if v <= 3)
    # tracce "solide": presenti in almeno il 50% dei frame del video
    stable_tracks = sum(1 for v in track_frame_count.values() if v >= 0.5 * n_frames)

    return {
        "model_tag": model_tag,
        "n_frames": n_frames,
        "avg_person_per_frame": total_person_dets / n_frames,
        "avg_person_conf": sum(person_confs) / len(person_confs) if person_confs else 0,
        "min_person_conf": min(person_confs) if person_confs else 0,
        "ball_detection_rate": frames_with_ball / n_frames,
        "avg_ball_conf": sum(ball_confs) / len(ball_confs) if ball_confs else 0,
        "n_person_tracks": n_tracks,
        "avg_track_len_frames": avg_track_len,
        "short_tracks_le3": short_tracks,
        "short_tracks_pct": short_tracks / n_tracks * 100 if n_tracks else 0,
        "stable_tracks_ge50pct": stable_tracks,
    }


def main():
    results = [analyze(m) for m in MODELS]

    print(f"{'Metrica':38s} {'yolov8n':>15s} {'yolov8m':>15s}")
    print("-" * 70)
    rows = [
        ("Frame analizzati", "n_frames", "{:.0f}"),
        ("Persone rilevate / frame (media)", "avg_person_per_frame", "{:.2f}"),
        ("Confidence media (persone)", "avg_person_conf", "{:.3f}"),
        ("Confidence minima (persone)", "min_person_conf", "{:.3f}"),
        ("Frame con palla rilevata", "ball_detection_rate", "{:.1%}"),
        ("Confidence media (palla)", "avg_ball_conf", "{:.3f}"),
        ("Track persona totali (ID unici)", "n_person_tracks", "{:.0f}"),
        ("Lunghezza media track (frame)", "avg_track_len_frames", "{:.1f}"),
        ("Track effimere (<=3 frame)", "short_tracks_le3", "{:.0f}"),
        ("  -> % sul totale track", "short_tracks_pct", "{:.1f}%"),
        ("Track solide (>=50% del video)", "stable_tracks_ge50pct", "{:.0f}"),
    ]
    for label, key, fmt in rows:
        v0 = fmt.format(results[0][key])
        v1 = fmt.format(results[1][key])
        print(f"{label:38s} {v0:>15s} {v1:>15s}")

    n_size = (OUTPUT_DIR / f"{VIDEO_STEM}_yolov8n_tracked.mp4").stat().st_size / (1024**2)
    m_size = (OUTPUT_DIR / f"{VIDEO_STEM}_yolov8m_tracked.mp4").stat().st_size / (1024**2)
    print("-" * 70)
    print(f"{'Video annotato (MB)':38s} {n_size:>14.1f}M {m_size:>14.1f}M")


if __name__ == "__main__":
    main()
