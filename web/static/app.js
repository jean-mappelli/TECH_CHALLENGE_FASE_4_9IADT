const $ = (id) => document.getElementById(id);
const state = { audio: null, video: null, transcript: "", audioRecognition: null };

function setStatus(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}
/**
 * Função para obter o ID do paciente.
 * @returns {string} - O ID do paciente.
 */
function patientId() {
  const value = $("patient").value.trim();
  if (!value) throw new Error("Informe um identificador para o paciente.");
  return encodeURIComponent(value);
}

function formatTime(seconds) {
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}
//
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[character]));
}

function startTimer(element, onLimit) {
  const started = Date.now();
  const timer = setInterval(() => {
    const elapsed = Math.floor((Date.now() - started) / 1000);
    element.textContent = formatTime(elapsed);
    if (elapsed >= 60) onLimit();
  }, 250);
  return () => clearInterval(timer);
}

//Exibe uma contagem regressiva antes de iniciar a gravação de vídeo
function videoCountdown(current) {
  return new Promise((resolve) => {
    const overlay = $("video-countdown");
    let remaining = 5;
    overlay.textContent = remaining;
    overlay.hidden = false;
    current.cancelCountdown = () => {
      clearInterval(current.countdownTimer);
      overlay.hidden = true;
      resolve(false);
    };
    current.countdownTimer = setInterval(() => {
      remaining -= 1;
      if (remaining === 0) {
        clearInterval(current.countdownTimer);
        current.countdownTimer = null;
        overlay.hidden = true;
        resolve(true);
      } else {
        overlay.textContent = remaining;
      }
    }, 1000);
  });
}

//usa a API de gravação de áudio para criar um arquivo WAV a partir dos samples capturados
function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset, text) => [...text].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
  write(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); write(8, "WAVE");
  write(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  write(36, "data"); view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, index) => view.setInt16(44 + index * 2, Math.max(-1, Math.min(1, sample)) * 0x7fff, true));
  return new Blob([view], { type: "audio/wav" });
}

/**
 * Função para criar um objeto de reconhecimento de fala para transcrever o audio em tempo real.
 * @returns {SpeechRecognition|null} - Retorna o objeto de reconhecimento de fala ou null se não suportado.
 */
function recognitionForAudio() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) return null;
  const recognition = new Recognition();
  recognition.lang = "pt-BR"; recognition.continuous = true; recognition.interimResults = false;
  recognition.onresult = (event) => {
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      if (event.results[index].isFinal) state.transcript += `${event.results[index][0].transcript} `;
    }
  };
  recognition.onerror = () => {};
  return recognition;
}

async function startAudio() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const context = new AudioContext();
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(4096, 1, 1);
    const samples = [];
    processor.onaudioprocess = (event) => samples.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    source.connect(processor); processor.connect(context.destination);
    state.audio = { stream, context, source, processor, samples, stopTimer: startTimer($("audio-time"), stopAudio) };
    state.transcript = ""; state.audioRecognition = recognitionForAudio();
    if (state.audioRecognition) state.audioRecognition.start();
    $("audio-start").disabled = true; $("audio-stop").disabled = false; setStatus("Gravando seu relato em áudio...");
  } catch (error) { setStatus(`Não foi possível acessar o microfone: ${error.message}`, true); }
}

async function stopAudio() {
  if (!state.audio) return;
  const current = state.audio; current.stopTimer(); current.processor.disconnect(); current.source.disconnect();
  current.stream.getTracks().forEach((track) => track.stop()); await current.context.close();
  if (state.audioRecognition) { try { state.audioRecognition.stop(); } catch (_) {} }
  const samples = new Float32Array(current.samples.reduce((total, chunk) => total + chunk.length, 0));
  let offset = 0; current.samples.forEach((chunk) => { samples.set(chunk, offset); offset += chunk.length; });
  state.audio.blob = encodeWav(samples, current.context.sampleRate);
  $("audio-preview").src = URL.createObjectURL(state.audio.blob); $("audio-preview").hidden = false;
  $("audio-start").disabled = false; $("audio-stop").disabled = true; $("audio-send").disabled = false; setStatus("Áudio pronto para envio.");
}
//Fnção que inicia um timer para iniciar a gravação
async function startVideo() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    const current = { stream, recorder: null, stopTimer: null, countdownTimer: null };
    state.video = current;
    $("video-preview").srcObject = stream; $("video-preview").hidden = false;
    $("video-start").disabled = true; $("video-stop").disabled = false;
    setStatus("Prepare-se: a gravação começa em 5 segundos...");
    if (!await videoCountdown(current) || state.video !== current) return;
    const mimeType = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"].find(MediaRecorder.isTypeSupported);
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    const chunks = []; recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
    recorder.onstop = () => { state.video.blob = new Blob(chunks, { type: recorder.mimeType || "video/webm" }); $("video-send").disabled = false; setStatus("Vídeo pronto para envio."); };
    recorder.start(); current.recorder = recorder; current.stopTimer = startTimer($("video-time"), stopVideo); setStatus("Gravando o exercício...");
  } catch (error) { setStatus(`Não foi possível acessar a câmera: ${error.message}`, true); }
}
//Para de gravar o vídeo
function stopVideo() {
  if (!state.video) return;
  const current = state.video;
  if (!current.recorder) {
    current.cancelCountdown?.(); current.stream.getTracks().forEach((track) => track.stop()); state.video = null;
    $("video-preview").srcObject = null; $("video-preview").hidden = true; $("video-start").disabled = false; $("video-stop").disabled = true; setStatus("Contagem cancelada."); return;
  }
  current.stopTimer(); current.recorder.stop(); current.stream.getTracks().forEach((track) => track.stop());
  $("video-preview").srcObject = null; $("video-start").disabled = false; $("video-stop").disabled = true;
}

//Sintetiza o relatório
function renderReport(kind, report, risk) {
  $("report").hidden = false; $("report-kind").textContent = kind === "audio" ? "ÁUDIO" : "VÍDEO";
  $("risk").innerHTML = `<div><small>Risco multimodal</small><br><strong>${escapeHtml(risk.level || "BAIXO")}</strong><br><small>Modalidades: ${escapeHtml((risk.modalities || []).join(", ") || "nenhuma")}</small></div><div class="score">${escapeHtml(risk.score ?? 0)}<small>/100</small></div>`;
  if (kind === "audio") {
    const terms = (report.text_analysis?.critical_terms || []).map((term) => `${term[0]} (${term[1]})`).join(", ") || "nenhum";
    $("report-content").innerHTML = `<p><strong>Transcrição:</strong> ${escapeHtml(report.transcript?.text || "Não disponível")}</p><p><strong>Sentimento:</strong> ${escapeHtml(report.text_analysis?.sentiment || "não disponível")} · <strong>Termos críticos:</strong> ${escapeHtml(terms)}</p><pre>${escapeHtml(JSON.stringify(report.voice_features || {}, null, 2))}</pre>`;
  } else {
    $("report-content").innerHTML = `<pre>${escapeHtml(report.report_markdown || "Relatório não disponível.")}</pre>`;
  }
  const alerts = report.alerts || [];
  $("report-alerts").innerHTML = alerts.length ? `<h3>Alertas gerados</h3>${alerts.map((alert) => `<div class="finding"><strong>${escapeHtml(alert.severity)}</strong><br>${escapeHtml(alert.message)}</div>`).join("")}` : "<p>Nenhum alerta específico foi gerado.</p>";
  $("report").scrollIntoView({ behavior: "smooth", block: "start" });
}

/**
 * Função para enviar a gravação para o backend e processar a analise.
 */
async function sendRecording(kind) {
  try {
    const current = state[kind]; if (!current?.blob) return;
    setStatus("Enviando arquivo e executando análise...");
    const form = new FormData(); form.append("file", current.blob, kind === "audio" ? "relato.wav" : "fisioterapia.webm");
    if (kind === "audio" && state.transcript.trim()) form.append("transcript", state.transcript.trim());
    if (kind === "video") form.append("session_type", "fisioterapia");
    const response = await fetch(`/patients/${patientId()}/${kind}`, { method: "POST", body: form });
    const body = await response.json(); if (!response.ok) throw new Error(body.detail || "Falha na análise");
    const riskResponse = await fetch(`/patients/${patientId()}/risk`); const risk = await riskResponse.json();
    renderReport(kind, body, risk); setStatus("Análise concluída.");
  } catch (error) { setStatus(error.message, true); }
}

//Event listeners
$("audio-start").addEventListener("click", startAudio); $("audio-stop").addEventListener("click", stopAudio); $("audio-send").addEventListener("click", () => sendRecording("audio"));
$("video-start").addEventListener("click", startVideo); $("video-stop").addEventListener("click", stopVideo); $("video-send").addEventListener("click", () => sendRecording("video"));