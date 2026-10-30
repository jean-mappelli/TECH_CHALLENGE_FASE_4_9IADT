"""Modelo de alertas e gerenciador (console + JSONL)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Callable, Iterable


class Severity(IntEnum):
    INFO = 1
    WARNING = 2
    CRITICAL = 3


@dataclass
class Alert:
    patient_id: str
    modality: str  # vitals | prescription | movement | audio | video | fusion
    kind: str
    severity: Severity
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.name
        return d


Notifier = Callable[[Alert], None]


def console_notifier(alert: Alert) -> None:
    print(f"[{alert.severity.name:8}] {alert.patient_id} | {alert.modality}/{alert.kind}: {alert.message}")


class AlertManager:
    """Guarda alertas em memória e (opcionalmente) em um arquivo JSONL.

    `notifiers` é a extensão para integrar e-mail, Teams, SMS etc.
    """

    def __init__(self, path: Path | str | None = None, notifiers: Iterable[Notifier] | None = None):
        self.path = Path(path) if path else None
        self.notifiers = list(notifiers or [])
        self._alerts: list[Alert] = []

    def emit(self, alert: Alert) -> None:
        self._alerts.append(alert)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(alert.to_dict(), ensure_ascii=False) + "\n")
        for notify in self.notifiers:
            notify(alert)

    def emit_many(self, alerts: Iterable[Alert]) -> None:
        for alert in alerts:
            self.emit(alert)

    def list(self, patient_id: str | None = None, min_severity: Severity = Severity.INFO) -> list[Alert]:
        return [
            a for a in self._alerts
            if (patient_id is None or a.patient_id == patient_id) and a.severity >= min_severity
        ]

    def clear(self) -> None:
        self._alerts.clear()
