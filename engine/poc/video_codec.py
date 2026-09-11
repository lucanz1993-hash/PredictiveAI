"""Fourcc H.264 per VideoWriter, con fallback automatico a mp4v.

Su Windows, OpenCV(+ffmpeg) sa codificare H.264 solo se trova la DLL
openh264-2.5.0-win64.dll di Cisco: non e' bundlata nella wheel di opencv-python
per motivi di licenza e va scaricata a parte (vedi README, sezione Setup).
Non e' committata nel repo (binario, .venv/ e' in .gitignore).

Con la DLL presente: H.264 vero, file 3-5x piu leggeri a parita' di qualita'
visiva rispetto a mp4v. Senza: fallback silenzioso a mp4v, tutto funziona
comunque, solo file piu pesanti.
"""
import os
import sys
from pathlib import Path

_DLL_NAME = "openh264-2.5.0-win64.dll"
_dll_path = Path(sys.executable).resolve().parent / _DLL_NAME
H264_AVAILABLE = sys.platform == "win32" and _dll_path.exists()

if H264_AVAILABLE:
    os.add_dll_directory(str(_dll_path.parent))

import cv2  # noqa: E402 (import dopo add_dll_directory di proposito)

FOURCC = cv2.VideoWriter_fourcc(*"avc1") if H264_AVAILABLE else cv2.VideoWriter_fourcc(*"mp4v")
