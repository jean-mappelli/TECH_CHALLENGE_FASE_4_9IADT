"""Pipeline de áudio: voz (acústica) + transcrição (Azure Speech) + texto (Azure Language)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..alerts import Alert, Severity
from .azure_clients import SpeechToText, TextAnalysis, TextAnalytics, Transcript
from .voice_features import VoiceFeatures, VoiceFinding, assess_voice, extract_features, load_wav


@dataclass
class AudioReport:
    patient_id: str
    file: str
    transcript: Transcript
    text_analysis: TextAnalysis
    features: VoiceFeatures
    findings: list[VoiceFinding]
    alerts: list[Alert] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "file": self.file,
            "transcript": {"text": self.transcript.text, "confidence": self.transcript.confidence,
                           "source": self.transcript.source},
            "text_analysis": {
                "sentiment": self.text_analysis.sentiment,
                "key_phrases": self.text_analysis.key_phrases,
                "critical_terms": self.text_analysis.critical_terms,
                "source": self.text_analysis.source,
            },
            "voice_features": self.features.to_dict(),
            "findings": [f.__dict__ for f in self.findings],
            "alerts": [a.to_dict() for a in self.alerts],
        }


def analyze_consultation_audio(
    wav_path: str | Path,
    patient_id: str,
    stt: SpeechToText | None = None,
    analytics: TextAnalytics | None = None,
    baseline: VoiceFeatures | None = None,
    mock_transcript: str | None = None,
    event_time: str | None = None,
) -> AudioReport:
    stt = stt or SpeechToText()
    analytics = analytics or TextAnalytics()
    sr, x = load_wav(wav_path)
    features = extract_features(x, sr)
    findings = assess_voice(features, baseline)
    transcript = stt.transcribe(wav_path, mock_text=mock_transcript)
    analysis = analytics.analyze(transcript.text)

    now = event_time or datetime.now(timezone.utc).isoformat(timespec="seconds")
    alerts: list[Alert] = []
    for f in findings:
        alerts.append(Alert(patient_id, "audio", f.kind, Severity[f.severity], f.message,
                            timestamp=now, details=f.evidence))
    for term, sev in analysis.critical_terms:
        alerts.append(Alert(patient_id, "audio", "critical_term", Severity[sev],
                            f"termo crítico na fala do paciente: “{term}”", timestamp=now,
                            details={"term": term, "sentiment": analysis.sentiment}))
    if analysis.sentiment == "negative" and any(a.severity >= Severity.WARNING for a in alerts):
        alerts.append(Alert(patient_id, "audio", "negative_sentiment", Severity.INFO,
                            "sentimento negativo na consulta, associado a outros achados", timestamp=now))
    return AudioReport(patient_id, str(wav_path), transcript, analysis, features, findings, alerts)
