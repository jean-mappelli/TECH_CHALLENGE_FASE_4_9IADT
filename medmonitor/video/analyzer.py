"""Orquestra a análise de um vídeo clínico (fisioterapia/cirurgia gravada).

Em uma única passada pelo vídeo:
- energia de movimento (fluxo óptico Farneback, OpenCV) - sempre disponível;
- pose (OpenPose/MediaPipe) - se um estimador for fornecido;
- objetos (YOLOv8) - se um detector for fornecido.
Gera achados, alertas e um relatório (Markdown/JSON).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import medfilt

from ..alerts import Alert, Severity
from ..anomaly.common import group_events
from .findings import Finding
from .objects import Detection, absence_events, zone_events
from .pose import analyze_pose_sequence


def analyze_motion_series(
    times: np.ndarray, energy: np.ndarray, sample_fps: float,
    min_stop_s: float = 2.0, stop_ratio: float = 0.15, min_activity: float = 0.02,
) -> list[Finding]:
    """Paradas súbitas (congelamento) na energia de fluxo óptico.

    Movimentos bruscos NÃO são detectados aqui: em testes, o fluxo óptico em baixa taxa de
    amostragem não os captura de forma confiável (movimento rápido sofre aliasing). Esse papel
    fica com a análise de pose (aceleração/velocidade angular), em `pose.analyze_pose_sequence`.
    """
    findings: list[Finding] = []
    if len(energy) < 10:
        return findings
    e = medfilt(energy.astype(float), 3)
    ref = float(np.percentile(e, 75))
    if ref < min_activity:  # cena essencialmente estática
        return findings
    dt = 1.0 / sample_fps
    for a, b in group_events(e < stop_ratio * ref, max_gap=1):
        dur = (b - a + 1) * dt
        if dur >= min_stop_s:
            findings.append(Finding(
                "motion_freeze", "WARNING", float(times[a]), float(times[b] + dt),
                f"parada de movimento por {dur:.1f}s durante a atividade (energia {e[a:b+1].mean():.2f} vs ~{ref:.2f})",
                {"mean_energy": float(e[a:b + 1].mean()), "reference_energy": ref}))
    return findings


@dataclass
class VideoReport:
    patient_id: str
    video: str
    session_type: str
    duration_s: float
    sample_fps: float
    modules: dict[str, bool]
    findings: list[Finding]
    alerts: list[Alert] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id, "video": self.video, "session_type": self.session_type,
            "duration_s": self.duration_s, "sample_fps": self.sample_fps, "modules": self.modules,
            "stats": self.stats, "findings": [f.to_dict() for f in self.findings],
            "alerts": [a.to_dict() for a in self.alerts],
        }

    def to_markdown(self) -> str:
        used = ", ".join(k for k, v in self.modules.items() if v) or "nenhum"
        lines = [
            f"# Relatório automático de vídeo - paciente {self.patient_id}",
            "",
            f"- **Arquivo:** `{Path(self.video).name}`",
            f"- **Tipo de sessão:** {self.session_type}",
            f"- **Duração:** {self.duration_s:.1f}s (amostragem {self.sample_fps:g} quadros/s)",
            f"- **Módulos ativos:** {used}",
            f"- **Achados:** {len(self.findings)}",
            "",
        ]
        if self.findings:
            lines += ["| Início | Fim | Tipo | Severidade | Descrição |", "|---|---|---|---|---|"]
            for f in sorted(self.findings, key=lambda f: f.t_start):
                lines.append(f"| {f.t_start:.1f}s | {f.t_end:.1f}s | {f.kind} | {f.severity} | {f.message} |")
        else:
            lines.append("Nenhum desvio detectado nos módulos ativos.")
        lines += ["", "> Triagem automática para apoio à revisão humana; não substitui avaliação clínica."]
        return "\n".join(lines) + "\n"


def analyze_video(
    path: str | Path,
    patient_id: str,
    session_type: str = "fisioterapia",
    sample_fps: float = 10.0,
    pose_estimator=None,
    detector=None,
    critical_zone: tuple[float, float, float, float] | None = None,
    zone_labels: set[str] | None = None,
    required_label: str | None = None,
    flow_size: tuple[int, int] = (160, 120),
    max_seconds: float | None = None,
    event_time: str | None = None,
) -> VideoReport:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"não foi possível abrir o vídeo: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps <= 1e-3 or src_fps > 240:
        src_fps = 25.0
    step = max(1, int(round(src_fps / sample_fps)))
    eff_fps = src_fps / step

    times, energy, pose_frames, det_frames = [], [], [], []
    prev, idx = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            t = idx / src_fps
            if max_seconds and t > max_seconds:
                break
            gray = cv2.cvtColor(cv2.resize(frame, flow_size), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 2, 15, 3, 5, 1.2, 0)
                energy.append(float(np.mean(np.linalg.norm(flow, axis=2))))
                times.append(t)
            prev = gray
            if pose_estimator is not None:
                pose_frames.append(pose_estimator.estimate(frame))
            if detector is not None:
                det_frames.append(detector.detect(frame))
        idx += 1
    cap.release()
    duration = idx / src_fps

    findings = analyze_motion_series(np.array(times), np.array(energy), eff_fps) if energy else []
    stats = {"frames_read": idx, "motion_mean": float(np.mean(energy)) if energy else 0.0}
    if pose_frames:
        pose_findings, angles = analyze_pose_sequence(pose_frames, eff_fps)
        findings += pose_findings
        stats["pose_frames_with_detection"] = int(sum(f is not None for f in pose_frames))
    if det_frames:
        if critical_zone and zone_labels:
            findings += zone_events(det_frames, eff_fps, critical_zone, zone_labels)
        if required_label:
            findings += absence_events(det_frames, eff_fps, required_label)

    now = event_time or datetime.now(timezone.utc).isoformat(timespec="seconds")
    alerts = [
        Alert(patient_id, "video", f.kind, Severity[f.severity],
              f"{f.message} [{f.t_start:.1f}s-{f.t_end:.1f}s]", timestamp=now,
              details={**f.evidence, "t_start": f.t_start, "t_end": f.t_end, "video": Path(path).name})
        for f in findings
    ]
    return VideoReport(
        patient_id, str(path), session_type, duration, eff_fps,
        {"motion_energy": True, "pose": pose_estimator is not None, "yolo": detector is not None},
        findings, alerts, stats,
    )
