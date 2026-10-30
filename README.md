# MedMonitor — Monitoramento multimodal de pacientes (Tech Challenge · Fase 4)

Sistema que analisa **vídeo, áudio, texto e séries clínicas** de pacientes, detecta anomalias e envia
**alertas** à equipe médica, consolidando tudo em um **escore de risco multimodal**.

| Requisito do desafio | Onde está |
|---|---|
| Análise de vídeo (OpenPose / YOLOv8, relatório automático) | `medmonitor/video/` (`pose.py`, `objects.py`, `analyzer.py`) |
| Análise de áudio (alterações vocais, Azure Speech to Text, Azure Text Analytics) | `medmonitor/audio/` (`voice_features.py`, `azure_clients.py`, `pipeline.py`) |
| Anomalias em sinais vitais, prescrições e movimentação + alertas | `medmonitor/anomaly/`, `medmonitor/alerts.py` |
| Fusão multimodal | `medmonitor/fusion/engine.py` |
| API e dashboard para a demonstração | `medmonitor/api.py` (FastAPI), `dashboard/app.py` (Streamlit) |

## Execução da aplicação

### Pré-requisitos

- Python 3.10 ou superior instalado e disponível no `PATH`.
- Navegador moderno com acesso a microfone e câmera.
- Conexão com a internet na primeira execução, para instalar as dependências.

### Inicialização automática

Os scripts na raiz criam o ambiente virtual, atualizam o `pip`, instalam o `requirements.txt`, criam a pasta `outputs/`
e iniciam o servidor FastAPI:

**Windows (PowerShell ou Prompt de Comando):**

```bat
start.bat
```

**Linux:**

```bash
bash start.sh
```

Ao iniciar, abra [http://127.0.0.1:8000](http://127.0.0.1:8000). O endereço `/docs` disponibiliza a documentação interativa
da API. Para encerrar, pressione `Ctrl+C` no terminal.

Na primeira visita, permita o uso do microfone para gravar o relato em áudio e da câmera para gravar o exercício de
fisioterapia. Depois de parar a gravação, clique em **Enviar para análise**; a página enviará o arquivo ao backend e exibirá
o relatório, os alertas e o escore de risco multimodal.

### Inicialização manual

Use estes comandos quando precisar controlar cada etapa:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn medmonitor.api:app --host 127.0.0.1 --port 8000
```

Em outro terminal, os comandos abaixo executam a demonstração sintética e os testes:

```bash
python scripts/run_demo.py --mock
python -m unittest discover -s tests -t .
```

Sem chaves Azure, o sistema funciona em **modo mock/local**. Para habilitar Azure Speech e Azure Language, configure as
variáveis `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`, `AZURE_LANGUAGE_KEY` e `AZURE_LANGUAGE_ENDPOINT` em um arquivo `.env`
na raiz do projeto. Consulte [`docs/AZURE_SETUP.md`](docs/AZURE_SETUP.md) antes de usar credenciais reais. Para forçar o
modo local mesmo com credenciais configuradas, defina `MEDMON_FORCE_MOCK=1`.

## Dependências opcionais

| Pacote | Habilita | Sem ele |
|---|---|---|
| `ultralytics` (YOLOv8) | detecção de objetos / zonas críticas | análise de vídeo segue por fluxo óptico + pose |
| `mediapipe` ou pesos do OpenPose | keypoints e ângulos articulares | idem (apenas energia de movimento) |
| chaves Azure | Speech to Text e Language reais | modo mock |

## Estrutura

```
medmonitor/
  config.py            # chaves Azure via variáveis de ambiente; MEDMON_FORCE_MOCK
  alerts.py            # Alert, Severity, AlertManager (console + JSONL, ponto de extensão p/ e-mail/Teams)
  service.py           # MonitoringService: orquestra tudo (usado por API, dashboard e demo)
  api.py               # FastAPI
  anomaly/             # vitals.py · prescriptions.py · movement.py · common.py
  audio/               # voice_features.py · azure_clients.py · pipeline.py
  video/               # pose.py · objects.py · analyzer.py · findings.py
  fusion/engine.py     # escore de risco por paciente (late fusion)
  data/                # synthetic.py (dados com anomalias conhecidas) · loaders.py (CSV real, ex.: PhysioNet)
dashboard/app.py       # Streamlit
scripts/               # run_demo.py · evaluate_synthetic.py
tests/                 # unittest (44 testes)
docs/                  # arquitetura, plano, relatório técnico, roteiro do vídeo, setup Azure
outputs/               # artefatos gerados (alertas, relatórios, gráficos)
web/                   # interface web: captura de mídia e relatório do paciente
```

## Documentação

- [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md) — estrutura do projeto e fluxo de dados
- [`docs/PLANO_IMPLEMENTACAO.md`](docs/PLANO_IMPLEMENTACAO.md) — fases, checklist e o que falta fazer
- [`docs/RELATORIO_TECNICO.md`](docs/RELATORIO_TECNICO.md) — **entregável**: fluxo, modelos, resultados, limitações
- [`docs/ROTEIRO_VIDEO.md`](docs/ROTEIRO_VIDEO.md) — roteiro de 15 min para o vídeo de demonstração
- [`docs/AZURE_SETUP.md`](docs/AZURE_SETUP.md) — criar os recursos Azure e testar

## Aviso

Projeto acadêmico. Limiares clínicos e regras de voz são **ilustrativos/heurísticos** e não foram validados
clinicamente; os resultados quantitativos vêm de **dados sintéticos**. Não use para decisões clínicas reais.
