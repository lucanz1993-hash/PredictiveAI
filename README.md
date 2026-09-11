# Match Analysis

Software di match analysis per video di partite di calcio (registrate con VEO cam o altre fonti).

## Stato attuale

PoC tecnica in `engine/poc/detect_track.py`: valida detection e tracking di giocatori e palla con YOLO + ByteTrack su GPU locale.

### Setup

```
cd engine
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Nota: torch va installato per primo e separatamente dall'indice CUDA di PyTorch, altrimenti `pip install -r requirements.txt` installerebbe una build CPU-only come dipendenza di ultralytics/torchvision, disattivando l'accelerazione GPU.

**Codec H.264 (opzionale ma consigliato)**: senza un passo in più, i video annotati vengono salvati in `mp4v` (funziona, ma file 15%+ più pesanti a parità di qualità). Per l'H.264 vero, scarica la DLL ufficiale Cisco (BSD license, non committata per motivi di licenza/dimensione) e mettila nella cartella dell'interprete Python della venv:

```
curl -fsSL -o openh264-2.5.0-win64.dll.bz2 http://ciscobinary.openh264.org/openh264-2.5.0-win64.dll.bz2
bunzip2 openh264-2.5.0-win64.dll.bz2
move openh264-2.5.0-win64.dll engine\.venv\Scripts\
```

Verifica MD5 attesa (dal sito Cisco): `83234500b244daf1e79c8b772c06e66f`. Senza la DLL gli script continuano a funzionare, con fallback silenzioso a `mp4v` (messaggio "Codec output: mp4v (fallback...)" in console).

### Uso

1. Copia un video (`.mp4` o `.mov`) in `engine/data/`
2. Esegui:

```
python poc/detect_track.py
```

3. Il video annotato e i dati di tracking (CSV) vengono salvati in `engine/outputs/`
4. Per la classificazione squadra (colore maglia) sugli stessi dati:

```
python poc/team_classification.py
```

Salva `<video>_team_clusters.json` (mappa track_id -> cluster), `<video>_cluster_samples.jpg` (collage per etichettatura manuale dei cluster) e `<video>_teams.mp4` (video annotato per cluster).
