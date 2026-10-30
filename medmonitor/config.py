"""Configuração central. Lê chaves do Azure de variáveis de ambiente.

Sem chaves (ou com MEDMON_FORCE_MOCK=1) o sistema roda em modo mock/local,
o que permite desenvolver, testar e demonstrar sem conta Azure.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


@dataclass(frozen=True)
class Settings:
    speech_key: str | None = field(default_factory=lambda: _env("AZURE_SPEECH_KEY"))
    speech_region: str | None = field(default_factory=lambda: _env("AZURE_SPEECH_REGION"))
    language_key: str | None = field(default_factory=lambda: _env("AZURE_LANGUAGE_KEY"))
    language_endpoint: str | None = field(default_factory=lambda: _env("AZURE_LANGUAGE_ENDPOINT"))
    force_mock: bool = field(default_factory=lambda: os.getenv("MEDMON_FORCE_MOCK") == "1")
    speech_locale: str = field(default_factory=lambda: os.getenv("MEDMON_LOCALE", "pt-BR"))
    alerts_path: Path = field(default_factory=lambda: Path(os.getenv("MEDMON_ALERTS_PATH", ROOT / "outputs" / "alerts.jsonl")))

    @property
    def azure_speech_enabled(self) -> bool:
        return bool(self.speech_key and self.speech_region) and not self.force_mock

    @property
    def azure_language_enabled(self) -> bool:
        return bool(self.language_key and self.language_endpoint) and not self.force_mock


def get_settings() -> Settings:
    return Settings()
