"""Camada de serviço: junta detectores, pipelines, fusão e alertas.

É usada pela API (FastAPI), pelo dashboard (Streamlit) e pelo script de demonstração,
para que a lógica exista em um único lugar.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .alerts import Alert, AlertManager, Severity, console_notifier
from .anomaly.movement import detect_movement_anomalies
from .anomaly.prescriptions import detect_prescription_anomalies
from .anomaly.vitals import detect_vital_anomalies
from .audio.azure_clients import SpeechToText, TextAnalytics
from .audio.pipeline import AudioReport, analyze_consultation_audio
from .config import Settings, get_settings
from .fusion.engine import FusionEngine, RiskAssessment
from .video.analyzer import VideoReport, analyze_video
from .video.pose import analyze_pose_sequence
from datetime import datetime, timezone


class MonitoringService:
    def __init__(self, settings: Settings | None = None, alerts: AlertManager | None = None,
                 fusion: FusionEngine | None = None, pose_estimator=None, detector=None, verbose: bool = False):
        self.settings = settings or get_settings()
        self.alerts = alerts or AlertManager(self.settings.alerts_path,
                                             notifiers=[console_notifier] if verbose else [])
        self.fusion = fusion or FusionEngine()
        self.stt = SpeechToText(self.settings)
        self.text = TextAnalytics(self.settings)
        self.pose_estimator = pose_estimator
        self.detector = detector
        self._fusion_keys: dict[str, tuple] = {}

    # ---------------------------------------------------------- status ----
    def status(self) -> dict:
        return {
            "azure_speech": self.settings.azure_speech_enabled,
            "azure_language": self.settings.azure_language_enabled,
            "pose_estimator": type(self.pose_estimator).__name__ if self.pose_estimator else None,
            "yolo": self.detector is not None,
            "alerts_total": len(self.alerts.list()),
        }

    # ------------------------------------------------------ tabulares -----
    def analyze_vitals(self, df: pd.DataFrame, patient_id: str):
        flagged, alerts = detect_vital_anomalies(df, patient_id)
        self.alerts.emit_many(alerts)
        return flagged, alerts

    def analyze_prescriptions(self, df: pd.DataFrame, patient_id: str) -> list[Alert]:
        alerts = detect_prescription_anomalies(df, patient_id)
        self.alerts.emit_many(alerts)
        return alerts

    def analyze_movement(self, df: pd.DataFrame, patient_id: str, **kw) -> list[Alert]:
        alerts = detect_movement_anomalies(df, patient_id, **kw)
        self.alerts.emit_many(alerts)
        return alerts

    # ------------------------------------------------- áudio e vídeo ------
    def analyze_audio(self, wav_path: str | Path, patient_id: str, mock_transcript: str | None = None,
                      event_time: str | None = None) -> AudioReport:
        report = analyze_consultation_audio(wav_path, patient_id, self.stt, self.text,
                                            mock_transcript=mock_transcript, event_time=event_time)
        self.alerts.emit_many(report.alerts)
        return report

    def analyze_video(self, path: str | Path, patient_id: str, event_time: str | None = None, **kw) -> VideoReport:
        report = analyze_video(path, patient_id, pose_estimator=self.pose_estimator, detector=self.detector,
                               event_time=event_time, **kw)
        self.alerts.emit_many(report.alerts)
        return report

    def analyze_pose_keypoints(self, frames: list[dict | None], fps: float, patient_id: str,
                               event_time: str | None = None, source: str = "keypoints") -> list[Alert]:
        """Analisa uma sequência de keypoints já extraída (de qualquer estimador)."""
        findings, _ = analyze_pose_sequence(frames, fps)
        now = event_time or datetime.now(timezone.utc).isoformat(timespec="seconds")
        alerts = [
            Alert(patient_id, "video", f.kind, Severity[f.severity], f"{f.message} [{f.t_start:.1f}s-{f.t_end:.1f}s]",
                  timestamp=now, details={**f.evidence, "source": source})
            for f in findings
        ]
        self.alerts.emit_many(alerts)
        return alerts

    # --------------------------------------------------------- fusão -------
    def assess(self, patient_id: str, emit: bool = True) -> RiskAssessment:
        assessment = self.fusion.assess(patient_id, self.alerts.list(patient_id))
        if emit:
            consolidated = self.fusion.consolidated_alert(assessment)
            key = (assessment.level, assessment.window_start, assessment.window_end)
            if consolidated and self._fusion_keys.get(patient_id) != key:
                self._fusion_keys[patient_id] = key
                self.alerts.emit(consolidated)
        return assessment
