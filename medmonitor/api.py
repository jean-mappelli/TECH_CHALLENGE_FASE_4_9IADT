"""API REST (FastAPI). Execute com:  uvicorn medmonitor.api:app --reload

Observação: este módulo não pôde ser executado no ambiente de desenvolvimento original
(FastAPI não estava instalável); a lógica está em `service.py`, que é testada.
Valide com `pip install -r requirements.txt` e a documentação interativa em /docs.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .alerts import Severity
from .service import MonitoringService

app = FastAPI(title="MedMonitor Multimodal", version="1.0.0",
              description="Monitoramento multimodal (vitais, prescrições, movimento, áudio, vídeo) com alertas.")
WEB_ROOT = Path(__file__).resolve().parent.parent / "web"


def _build_service() -> MonitoringService:
    """Cria o serviço COM estimador de pose e YOLO. Sem eles o vídeo só teria energia de movimento
    (e nenhum alerta sobre repetições/equilíbrio). Falhas são avisadas no console, não silenciosas."""
    root = WEB_ROOT.parent
    pose = detector = None
    model = Path(os.getenv("MEDMON_POSE_MODEL", root / "modelos" / "pose_landmarker_lite.task"))
    try:
        from .video.pose import MediaPipePose

        pose = MediaPipePose(model_path=model if model.exists() else None)
    except Exception as exc:
        print(f"[medmonitor] AVISO: pose desativada ({exc})")
    try:
        from .video.objects import YoloDetector

        yolo = root / "yolov8n.pt"
        detector = YoloDetector(str(yolo) if yolo.exists() else "yolov8n.pt")
    except Exception as exc:
        print(f"[medmonitor] AVISO: YOLO desativado ({exc})")
    return MonitoringService(pose_estimator=pose, detector=detector, verbose=True)


service = _build_service()
print(f"[medmonitor] status: {service.status()}")


class Records(BaseModel):
    records: list[dict]


@app.get("/", include_in_schema=False)
def web_app():
    return FileResponse(WEB_ROOT / "index.html")


app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")


def _df(rec: Records) -> pd.DataFrame:
    if not rec.records:
        raise HTTPException(422, "lista de registros vazia")
    df = pd.DataFrame(rec.records)
    if "timestamp" not in df:
        raise HTTPException(422, "campo 'timestamp' obrigatório")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(upload.file, tmp)
    return Path(tmp.name)


@app.get("/health")
def health():
    return {"status": "ok", **service.status()}


@app.post("/patients/{patient_id}/vitals")
def post_vitals(patient_id: str, body: Records):
    try:
        _, alerts = service.analyze_vitals(_df(body), patient_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {"alerts": [a.to_dict() for a in alerts]}


@app.post("/patients/{patient_id}/prescriptions")
def post_prescriptions(patient_id: str, body: Records):
    alerts = service.analyze_prescriptions(_df(body), patient_id)
    return {"alerts": [a.to_dict() for a in alerts]}


@app.post("/patients/{patient_id}/movement")
def post_movement(patient_id: str, body: Records):
    alerts = service.analyze_movement(_df(body), patient_id)
    return {"alerts": [a.to_dict() for a in alerts]}


@app.post("/patients/{patient_id}/audio")
async def post_audio(patient_id: str, file: UploadFile = File(...), transcript: str | None = Form(None)):
    """WAV mono/estéreo. `transcript` só é usado no modo mock (sem Azure Speech)."""
    path = _save_upload(file)
    try:
        report = service.analyze_audio(path, patient_id, mock_transcript=transcript)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc))
    finally:
        path.unlink(missing_ok=True)
    return report.to_dict()


@app.post("/patients/{patient_id}/video")
async def post_video(patient_id: str, file: UploadFile = File(...), session_type: str = Form("fisioterapia")):
    path = _save_upload(file)
    try:
        report = service.analyze_video(path, patient_id, session_type=session_type)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc))
    finally:
        path.unlink(missing_ok=True)
    return {**report.to_dict(), "report_markdown": report.to_markdown()}


@app.get("/patients/{patient_id}/alerts")
def get_alerts(patient_id: str, min_severity: str = "INFO"):
    try:
        sev = Severity[min_severity.upper()]
    except KeyError:
        raise HTTPException(422, "min_severity deve ser INFO, WARNING ou CRITICAL")
    return {"alerts": [a.to_dict() for a in service.alerts.list(patient_id, sev)]}


@app.get("/patients/{patient_id}/risk")
def get_risk(patient_id: str):
    return service.assess(patient_id).to_dict()
