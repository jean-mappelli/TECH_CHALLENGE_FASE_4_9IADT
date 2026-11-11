# Monitoramento multimodal de pacientes

Sistema que analisa **vídeo e audio** de pacientes atravvés de ferramentas de IA, detecta anomalias e sinaliza alertas, retornando um escore de risco multimodal.

## Fluxos de análise

### 1. Análise de áudio

O audio é processado em duas partes: caracteristicas da voz e conteudo da fala. O sistema extrai volume, pausas, ritmo e variação de intensidade para identificar padrões como fadiga vocal, fala fragmentada, pausas
longas e possível alteraçoes. Em seguida, a fala é transcrita pelo Azure Speech e o texto é analisado para obter sentimento, frases-chave e termos clínicos críticos, como falta de ar, dor no peito e desmaio.

```text
Áudio WAV
   │
   ├── Características da voz ─> alterações vocais e pausas
   │
   └── Transcrição ─> análise textual ─> sentimento,
                         frases-chave e termos críticos
                                      │
                                      v
                                  Alertas
```

### 2. Análise de vídeo

No fluxo da aplicação, o vídeo é analisado pelo fluxo óptico e pelo estimador de pose MediaPipe. O fluxo óptico mede a intensidade do movimento entre frames e identifica possíveis paradas prolongadas. O MediaPipe detecta pontos corporais, a partir dos quais são calculados ângulos articulares, amplitude, assimetria, movimentos bruscos,
repetições e equilíbrio.

```text
Vídeo
  │
  ├── Fluxo óptico ──> intensidade do movimento e congelamento
  │
  └── MediaPipe Pose ──> keypoints ──> ângulos, amplitude,
                                      assimetria e movimentos
                                           │
                                           ▼
                                       Alertas
```

## Execução da aplicação

### Pré-requisitos

- Python 3.10 ou superior instalado e disponível no `PATH`.
- Navegador moderno com acesso a microfone e câmera.
- Conexão com a internet na primeira execução, para instalar as dependências.
- Serviços Azure criados e configurados.

## Criação dos serviços Azure

Para habilitar a transcrição e a análise textual dos áudios, crie dois recursos no
[Portal do Azure](https://portal.azure.com):

1. Acesse **Criar um recurso** e pesquise por **Speech**.
2. Crie um recurso **Speech Services**, escolhendo assinatura, grupo de recursos, região e plano de preços.
3. Após a criação, abra o recurso, entre em **Chaves e ponto de extremidade** e copie uma chave e a região.
4. Volte para **Criar um recurso**, pesquise por **Language Service** e crie um recurso de linguagem.
5. No recurso criado, abra **Chaves e ponto de extremidade** e copie uma chave e o endpoint.
6. Na raiz do projeto, crie um arquivo `.env` com os valores obtidos:

```env
AZURE_SPEECH_KEY=sua_chave_do_speech
AZURE_SPEECH_REGION=sua_regiao_do_speech
AZURE_LANGUAGE_KEY=sua_chave_do_language
AZURE_LANGUAGE_ENDPOINT=https://seu-recurso.cognitiveservices.azure.com/
MEDMON_LOCALE=pt-BR
```

O sistema usa o Azure Speech para converter o áudio em texto e o Azure Language para
classificar o sentimento e extrair frases-chave. Os termos clínicos críticos são
identificados pelo próprio projeto a partir da transcrição. Nunca publique o arquivo
`.env` nem compartilhe suas chaves.

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

Ao iniciar, abra [http://127.0.0.1:8000](http://127.0.0.1:8000). 
Para encerrar, pressione `Ctrl+C` no terminal.

## Registro de alertas

O arquivo `outputs/alerts.jsonl` armazena os alertas gerados pelo sistema. O formato
JSONL significa que cada linha do arquivo representa um alerta JSON independente,
permitindo registrar vários eventos sem substituir os registros anteriores.

Os alertas podem ser gerados pelas análises de áudio e vídeo e pela fusão multimodal.
Cada registro contém, entre outros campos:

- `patient_id`: identificador do paciente;
- `modality`: origem do alerta, como `audio`, `video` ou `fusion`;
- `kind`: tipo do evento identificado;
- `severity`: severidade (`INFO`, `WARNING` ou `CRITICAL`);
- `message`: descrição do alerta;
- `timestamp`: data e hora do evento;
- `details`: evidências e informações adicionais da análise.

Por padrão, o caminho é `outputs/alerts.jsonl`. Ele pode ser alterado pela variável
`MEDMON_ALERTS_PATH` no arquivo `.env`.
