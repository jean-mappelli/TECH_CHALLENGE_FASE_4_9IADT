"""Detecção de objetos/áreas críticas com YOLOv8 + regras de zona.

`YoloDetector` depende do pacote `ultralytics` (import tardio). As funções de
regra (`zone_events`, `absence_events`) são puras e testáveis sem o modelo.

Observação: os pesos pré-treinados do YOLOv8 (COCO) reconhecem "person",
"scissors", "knife" etc., mas NÃO instrumentos cirúrgicos específicos. Para
detectar instrumentais/áreas críticas reais é preciso fine-tuning em dataset
rotulado do domínio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .findings import Finding


@dataclass
class Detection:
    label: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2


class YoloDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics não instalado (pip install ultralytics)") from exc
        self._model = YOLO(weights)
        self._conf = conf

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        res = self._model.predict(frame_bgr, conf=self._conf, verbose=False)[0]
        names = res.names
        return [
            Detection(names[int(c)], float(p), tuple(float(v) for v in b))
            for b, c, p in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy())
        ]


def overlap_ratio(box, zone) -> float:
    """Fração da área da `box` que está dentro de `zone` (ambos x1,y1,x2,y2)."""
    ix1, iy1 = max(box[0], zone[0]), max(box[1], zone[1])
    ix2, iy2 = min(box[2], zone[2]), min(box[3], zone[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))
    return inter / area


def _runs(mask: list[bool]) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def zone_events(
    detections_per_frame: list[list[Detection]],
    fps: float,
    zone: tuple[float, float, float, float],
    labels: set[str],
    min_overlap: float = 0.3,
    min_duration_s: float = 0.5,
    severity: str = "WARNING",
    zone_name: str = "área crítica",
) -> list[Finding]:
    """Objetos de interesse permanecendo dentro da zona por tempo mínimo."""
    findings = []
    for label in sorted(labels):
        mask = [any(d.label == label and overlap_ratio(d.box, zone) >= min_overlap for d in dets)
                for dets in detections_per_frame]
        for a, b in _runs(mask):
            if (b - a + 1) / fps >= min_duration_s:
                findings.append(Finding(
                    "critical_zone_occupied", severity, a / fps, (b + 1) / fps,
                    f"“{label}” permaneceu em {zone_name} por {(b - a + 1) / fps:.1f}s",
                    {"label": label, "zone": list(zone)}))
    return findings


def absence_events(
    detections_per_frame: list[list[Detection]],
    fps: float,
    required_label: str,
    min_duration_s: float = 3.0,
    severity: str = "WARNING",
) -> list[Finding]:
    """Ausência de um objeto/pessoa esperado (ex.: 'person' no leito/sala)."""
    mask = [not any(d.label == required_label for d in dets) for dets in detections_per_frame]
    return [
        Finding("required_object_absent", severity, a / fps, (b + 1) / fps,
                f"“{required_label}” não detectado por {(b - a + 1) / fps:.1f}s", {"label": required_label})
        for a, b in _runs(mask) if (b - a + 1) / fps >= min_duration_s
    ]
