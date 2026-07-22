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

### Uso

1. Copia un video (`.mp4` o `.mov`) in `engine/data/`
2. Esegui:

```
python poc/detect_track.py
```

3. Il video annotato viene salvato in `engine/outputs/`
