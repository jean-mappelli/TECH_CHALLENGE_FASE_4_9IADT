"""Clientes Azure (REST) para Speech to Text e Language (Text Analytics).

Usa a API REST via `requests` (sem SDK), o que simplifica a instalação.
Sem credenciais, cai automaticamente em um modo mock/local determinístico,
para que o restante do pipeline funcione e seja testável.

Limitação conhecida: o endpoint REST de Speech "short audio" aceita áudios de
até ~60 s. Para consultas longas, use o Speech SDK (reconhecimento contínuo)
ou a Batch Transcription API - ver docs/AZURE_SETUP.md.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import requests

from ..config import Settings, get_settings

# ---------------------------------------------------------------- Speech ----


@dataclass
class Transcript:
    text: str
    confidence: float | None
    source: str  # "azure" | "mock"


class SpeechToText:
    def __init__(self, settings: Settings | None = None, session: requests.Session | None = None):
        self.settings = settings or get_settings()
        self.session = session or requests.Session()

    @property
    def is_azure(self) -> bool:
        return self.settings.azure_speech_enabled

    def transcribe(self, wav_path: str | Path, mock_text: str | None = None) -> Transcript:
        if not self.is_azure:
            return self._mock(wav_path, mock_text)
        s = self.settings
        url = (
            f"https://{s.speech_region}.stt.speech.microsoft.com/speech/recognition/"
            f"conversation/cognitiveservices/v1"
        )
        with open(wav_path, "rb") as fh:
            audio = fh.read()
        resp = self.session.post(
            url,
            params={"language": s.speech_locale, "format": "detailed"},
            headers={
                "Ocp-Apim-Subscription-Key": s.speech_key,
                "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
                "Accept": "application/json",
            },
            data=audio,
            timeout=60,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("RecognitionStatus") != "Success":
            return Transcript("", None, "azure")
        best = (body.get("NBest") or [{}])[0]
        return Transcript(best.get("Display") or body.get("DisplayText", ""), best.get("Confidence"), "azure")

    @staticmethod
    def _mock(wav_path: str | Path, mock_text: str | None) -> Transcript:
        """Mock: usa `mock_text` ou um arquivo irmão `<audio>.txt` com a transcrição."""
        if mock_text is not None:
            return Transcript(mock_text, None, "mock")
        sidecar = Path(wav_path).with_suffix(".txt")
        if sidecar.exists():
            return Transcript(sidecar.read_text(encoding="utf-8").strip(), None, "mock")
        return Transcript("", None, "mock")


# -------------------------------------------------------------- Language ----

# termos críticos (PT-BR) -> severidade. Complementa o Azure; é exibido sempre.
CRITICAL_TERMS: dict[str, str] = {
    "dor no peito": "CRITICAL",
    "nao consigo respirar": "CRITICAL",
    "falta de ar": "CRITICAL",
    "desmaio": "CRITICAL",
    "desmaiei": "CRITICAL",
    "sangramento": "CRITICAL",
    "dificuldade para falar": "CRITICAL",
    "dificuldade para respirar": "CRITICAL",
    "formigamento": "WARNING",
    "dormencia": "WARNING",
    "tontura": "WARNING",
    "palpitacao": "WARNING",
    "visao turva": "WARNING",
    "confusao": "WARNING",
    "cansaco": "WARNING",
    "cansado": "WARNING",
    "cansada": "WARNING",
    "fraqueza": "WARNING",
    "febre": "WARNING",
    "vomito": "WARNING",
    "dor forte": "WARNING",
}

NEGATIVE_WORDS = {"dor", "pior", "mal", "medo", "preocupado", "cansado", "fraco", "ruim", "sufocando", "piorou"}
POSITIVE_WORDS = {"melhor", "bem", "tranquilo", "otimo", "melhorou", "aliviado", "disposto"}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


@dataclass
class TextAnalysis:
    sentiment: str                      # positive | neutral | negative | mixed
    confidence: dict[str, float]
    key_phrases: list[str]
    critical_terms: list[tuple[str, str]] = field(default_factory=list)  # (termo, severidade)
    source: str = "mock"


def find_critical_terms(text: str) -> list[tuple[str, str]]:
    norm = _norm(text)
    return [(term, sev) for term, sev in CRITICAL_TERMS.items() if term in norm]


class TextAnalytics:
    API_VERSION = "2023-04-01"

    def __init__(self, settings: Settings | None = None, session: requests.Session | None = None):
        self.settings = settings or get_settings()
        self.session = session or requests.Session()

    @property
    def is_azure(self) -> bool:
        return self.settings.azure_language_enabled

    def analyze(self, text: str) -> TextAnalysis:
        critical = find_critical_terms(text)
        if not text.strip():
            return TextAnalysis("neutral", {}, [], critical, "mock")
        if not self.is_azure:
            return self._mock(text, critical)
        sentiment = self._call("SentimentAnalysis", text)
        phrases = self._call("KeyPhraseExtraction", text)
        return TextAnalysis(
            sentiment=sentiment["sentiment"],
            confidence=sentiment.get("confidenceScores", {}),
            key_phrases=phrases.get("keyPhrases", []),
            critical_terms=critical,
            source="azure",
        )

    def _call(self, kind: str, text: str) -> dict:
        s = self.settings
        url = f"{s.language_endpoint.rstrip('/')}/language/:analyze-text"
        payload = {
            "kind": kind,
            "parameters": {"modelVersion": "latest"},
            "analysisInput": {"documents": [{"id": "1", "language": s.speech_locale.split("-")[0], "text": text}]},
        }
        resp = self.session.post(
            url,
            params={"api-version": self.API_VERSION},
            headers={"Ocp-Apim-Subscription-Key": s.language_key, "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        docs = resp.json()["results"]["documents"]
        return docs[0] if docs else {}

    @staticmethod
    def _mock(text: str, critical: list[tuple[str, str]]) -> TextAnalysis:
        words = set(_norm(text).replace(",", " ").replace(".", " ").split())
        neg, pos = len(words & NEGATIVE_WORDS), len(words & POSITIVE_WORDS)
        if neg and pos:
            sent = "mixed"
        elif neg > pos:
            sent = "negative"
        elif pos > neg:
            sent = "positive"
        else:
            sent = "neutral"
        phrases = [t for t, _ in critical]
        return TextAnalysis(sent, {}, phrases, critical, "mock")
