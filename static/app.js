const uploadForm = document.getElementById("uploadForm");
const uploadResult = document.getElementById("uploadResult");
const documentsList = document.getElementById("documentsList");
const refreshDocsBtn = document.getElementById("refreshDocsBtn");
const deleteDocsBtn = document.getElementById("deleteDocsBtn");
const chatBtn = document.getElementById("chatBtn");
const chatResult = document.getElementById("chatResult");
const configForm = document.getElementById("configForm");
const configResult = document.getElementById("configResult");
const resetBtn = document.getElementById("resetBtn");
const docsStatus = document.getElementById("docsStatus");
const docsStatusDot = document.getElementById("docsStatusDot");
const chatStatus = document.getElementById("chatStatus");
const chatStatusDot = document.getElementById("chatStatusDot");
const configStatus = document.getElementById("configStatus");
const configStatusDot = document.getElementById("configStatusDot");
const uploadBtn = uploadForm ? uploadForm.querySelector("button[type='submit']") : null;
const configSaveBtn = configForm ? configForm.querySelector("button[type='submit']") : null;
const defaultIndexInput = document.getElementById("defaultIndex");
const retrievalTopKInput = document.getElementById("retrievalTopK");
const defaultSemanticRankerInput = document.getElementById("defaultSemanticRanker");
const chatPromptCard = document.getElementById("chatPromptCard");
const chatPromptText = document.getElementById("chatPromptText");
const chatThinking = document.getElementById("chatThinking");
const chatAnswerCard = document.getElementById("chatAnswerCard");
const chatAnswerText = document.getElementById("chatAnswerText");
const chatAnswerMeta = document.getElementById("chatAnswerMeta");
const chatConfigHint = document.getElementById("chatConfigHint");
const voiceToggleBtn = document.getElementById("voiceToggleBtn");
const recordBtn = document.getElementById("recordBtn");
const speakerToggleBtn = document.getElementById("speakerToggleBtn");
const ttsAudioPlayer = document.getElementById("ttsAudioPlayer");
const chatAnswerAudioContainer = document.getElementById("chatAnswerAudioContainer");
let runtimeConfig = null;
let isSpeakerEnabled = false;
let isVoiceMode = false;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let recordingTimer = null;
const MAX_RECORDING_SECONDS = 60;

function setStatus(dotEl, textEl, state, message) {
  if (!dotEl || !textEl) {
    return;
  }
  dotEl.classList.remove("idle", "working", "success", "error");
  dotEl.classList.add(state);
  textEl.textContent = message;
}

function setBusy(elements, isBusy) {
  elements.filter(Boolean).forEach((el) => {
    el.disabled = isBusy;
  });
}

function showJson(el, obj) {
  if (!el) {
    return;
  }
  el.textContent = JSON.stringify(obj, null, 2);
}

function setValue(el, value) {
  if (el) {
    el.value = value;
  }
}

function setText(el, value) {
  if (el) {
    el.textContent = value;
  }
}

function toggleHidden(el, shouldHide) {
  if (el) {
    el.classList.toggle("hidden", shouldHide);
  }
}

function currentIndexName() {
  const value = defaultIndexInput?.value.trim();
  return value || runtimeConfig?.default_index || "default-index";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderDocuments(items) {
  if (!documentsList) {
    return;
  }
  if (!items || items.length === 0) {
    documentsList.innerHTML = "<div class='doc-meta'>No documents found for this index.</div>";
    return;
  }
  documentsList.innerHTML = items.map((doc) => `
    <label class="doc-row">
      <input type="checkbox" class="doc-select" value="${escapeHtml(doc.source_name)}">
      <div>
        <div class="doc-name">${escapeHtml(doc.source_name)}</div>
        <div class="doc-meta">Chunks: ${doc.chunk_count} | Latest: ${escapeHtml(doc.latest_created_at || "-")}</div>
      </div>
      <span class="doc-meta">${escapeHtml(doc.index_name)}</span>
    </label>
  `).join("");
}

async function loadDocuments() {
  if (!documentsList) {
    return;
  }
  setStatus(docsStatusDot, docsStatus, "working", "Refreshing documents...");
  const indexName = encodeURIComponent(currentIndexName());
  try {
    const res = await fetch(`/api/documents?index_name=${indexName}`);
    const data = await res.json();
    if (!data.ok) {
      documentsList.innerHTML = `<div class='doc-meta'>Error: ${data.error || "Failed to fetch documents"}</div>`;
      setStatus(docsStatusDot, docsStatus, "error", "Failed to refresh documents");
      return;
    }
    renderDocuments(data.documents || []);
    setStatus(docsStatusDot, docsStatus, "success", "Documents refreshed");
  } catch (err) {
    documentsList.innerHTML = `<div class='doc-meta'>Error: ${err.message || "Failed to fetch documents"}</div>`;
    setStatus(docsStatusDot, docsStatus, "error", "Failed to refresh documents");
  }
}

function selectedSources() {
  return Array.from(document.querySelectorAll(".doc-select:checked")).map((el) => el.value);
}

function applyConfig(config) {
  if (!config) {
    return;
  }
  runtimeConfig = config;
  setValue(defaultIndexInput, config.default_index || "default-index");
  setValue(document.getElementById("azureTier"), config.azure_tier || "FREE");
  setValue(document.getElementById("model"), config.model || "");
  setValue(retrievalTopKInput, config.top_k || 4);
  if (defaultSemanticRankerInput) {
    defaultSemanticRankerInput.checked = Boolean(config.use_semantic_ranker);
  }
  setValue(document.getElementById("chunkSize"), config.chunk_size || 1000);
  setValue(document.getElementById("chunkOverlap"), config.chunk_overlap || 200);
  setValue(document.getElementById("embeddingDimensions"), config.embedding_dimensions || 1536);
  setValue(document.getElementById("inputCost"), config.pricing?.input_per_1k_tokens_usd ?? "");
  setValue(document.getElementById("outputCost"), config.pricing?.output_per_1k_tokens_usd ?? "");
  setValue(document.getElementById("semanticCost"), config.pricing?.semantic_query_cost_usd ?? "");
  if (chatConfigHint) {
    const semanticLabel = config.use_semantic_ranker ? "semantic ranker on" : "semantic ranker off";
    chatConfigHint.textContent = `Current settings: ${config.default_index || "default-index"} · top ${config.top_k || 4} · ${semanticLabel}.`;
  }
}

function renderPendingChat(query) {
  setText(chatPromptText, query);
  toggleHidden(chatPromptCard, false);
  toggleHidden(chatThinking, false);
  toggleHidden(chatAnswerCard, true);
  if (chatAnswerCard) {
    chatAnswerCard.classList.remove("error-state");
  }
}

function renderChatResponse(answer, meta, audioBase64 = null, isError = false) {
    if (chatAnswerText) {
        chatAnswerText.innerHTML = marked.parse(answer);
    }
    setText(chatAnswerMeta, meta);
    toggleHidden(chatThinking, true);
    toggleHidden(chatAnswerCard, false);
    if (chatAnswerCard) {
        chatAnswerCard.classList.toggle("error-state", isError);
    }
    
    if (chatAnswerAudioContainer) {
        chatAnswerAudioContainer.innerHTML = '';
        chatAnswerAudioContainer.classList.add("hidden");
    }
    
    if (chatAnswerAudioContainer && audioBase64 && isSpeakerEnabled) {
        const audioSrc = `data:audio/wav;base64,${audioBase64}`;
        chatAnswerAudioContainer.innerHTML = `<audio src="${audioSrc}" controls></audio>`;
        chatAnswerAudioContainer.classList.remove("hidden");
    }
}

async function loadConfig() {
  setStatus(configStatusDot, configStatus, "working", "Loading configuration...");
  try {
    const res = await fetch("/api/config");
    const data = await res.json();
    applyConfig(data);
    showJson(configResult, data);
    setStatus(configStatusDot, configStatus, "success", "Configuration loaded");
    return data;
  } catch (err) {
    showJson(configResult, { ok: false, error: err.message || "Config load failed" });
    setStatus(configStatusDot, configStatus, "error", "Failed to load configuration");
    return null;
  }
}

if (uploadForm) {
  uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = document.getElementById("uploadFile").files[0];
  if (!file) {
    showJson(uploadResult, { ok: false, error: "Please pick a file" });
    setStatus(docsStatusDot, docsStatus, "error", "Pick a file to upload");
    return;
  }

  setStatus(docsStatusDot, docsStatus, "working", "Uploading and processing document...");
  setBusy([uploadBtn, refreshDocsBtn, deleteDocsBtn], true);

  const fd = new FormData();
  fd.append("file", file);
  fd.append("index_name", currentIndexName());

  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await res.json();
    showJson(uploadResult, data);
    if (data.ok) {
      const chunks = data.chunks_ingested || 0;
      const adiPages =
        data.cost_summary && Number.isFinite(data.cost_summary.adi_pages_used_this_session)
          ? data.cost_summary.adi_pages_used_this_session
          : null;
      const statusText =
        adiPages !== null
          ? `Ingested ${chunks} chunks, ${adiPages} ADI pages`
          : `Ingested ${chunks} chunks`;
      setStatus(docsStatusDot, docsStatus, "success", statusText);
    } else {
      setStatus(
        docsStatusDot,
        docsStatus,
        "error",
        data.error ? `Upload failed: ${data.error}` : "Upload failed"
      );
    }
    await loadDocuments();
  } catch (err) {
    showJson(uploadResult, { ok: false, error: err.message || "Upload failed" });
    setStatus(docsStatusDot, docsStatus, "error", "Upload failed");
  } finally {
    setBusy([uploadBtn, refreshDocsBtn, deleteDocsBtn], false);
  }
  });
}

if (refreshDocsBtn) {
  refreshDocsBtn.addEventListener("click", async () => {
    setBusy([refreshDocsBtn, deleteDocsBtn], true);
    await loadDocuments();
    setBusy([refreshDocsBtn, deleteDocsBtn], false);
  });
}

if (deleteDocsBtn) {
  deleteDocsBtn.addEventListener("click", async () => {
    const sourceNames = selectedSources();
    if (sourceNames.length === 0) {
      showJson(uploadResult, { ok: false, error: "Select at least one document to delete." });
      setStatus(docsStatusDot, docsStatus, "error", "Select documents to delete");
      return;
    }
    setStatus(docsStatusDot, docsStatus, "working", "Deleting selected documents...");
    setBusy([uploadBtn, refreshDocsBtn, deleteDocsBtn], true);
    try {
      const res = await fetch("/api/documents", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          index_name: currentIndexName(),
          source_names: sourceNames,
        }),
      });
      const data = await res.json();
      showJson(uploadResult, data);
      if (data.ok) {
        const deleted = data.delete_summary ? data.delete_summary.deleted_documents : 0;
        setStatus(docsStatusDot, docsStatus, "success", `Deleted ${deleted} chunks`);
      } else {
        setStatus(
          docsStatusDot,
          docsStatus,
          "error",
          data.error ? `Delete failed: ${data.error}` : "Delete failed"
        );
      }
      await loadDocuments();
    } catch (err) {
      showJson(uploadResult, { ok: false, error: err.message || "Delete failed" });
      setStatus(docsStatusDot, docsStatus, "error", "Delete failed");
    } finally {
      setBusy([uploadBtn, refreshDocsBtn, deleteDocsBtn], false);
    }
  });
}

if (chatBtn) {
    chatBtn.addEventListener("click", async () => {
        const query = document.getElementById("chatInput").value.trim();
        if (!query) {
            showJson(chatResult, { ok: false, error: "Question cannot be empty" });
            setStatus(chatStatusDot, chatStatus, "error", "Enter a question");
            return;
        }
        renderPendingChat(query);
        setStatus(chatStatusDot, chatStatus, "working", "Retrieving context and generating answer...");
        setBusy([chatBtn], true);
        const payload = {
            query,
            index_name: runtimeConfig?.default_index || "default-index",
            top_k: Number(runtimeConfig?.top_k ?? 4),
            use_semantic_ranker: Boolean(runtimeConfig?.use_semantic_ranker),
        };
        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            showJson(chatResult, data);
            if (data.ok) {
                const contexts = Array.isArray(data.contexts) ? data.contexts.length : 0;
                const modelName = data.model ? ` · ${data.model}` : "";
                
                renderChatResponse(data.answer || "No answer returned.", `${contexts} contexts${modelName}`, null);
                setStatus(chatStatusDot, chatStatus, "success", `Answer ready (${contexts} contexts)`);
                
                if (isSpeakerEnabled && data.answer) {
                    setStatus(chatStatusDot, chatStatus, "working", "Generating audio...");
                    try {
                        const ttsRes = await fetch("/api/tts-generate", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ text: data.answer })
                        });
                        const ttsData = await ttsRes.json();
                        if (ttsData.ok && ttsData.audio) {
                            updateChatAudio(ttsData.audio);
                            setStatus(chatStatusDot, chatStatus, "success", `Answer ready with audio (${contexts} contexts)`);
                        } else {
                            console.error("TTS generation failed:", ttsData.error);
                            setStatus(chatStatusDot, chatStatus, "success", `Answer ready (${contexts} contexts) - audio failed`);
                        }
                    } catch (ttsErr) {
                        console.error("TTS error:", ttsErr);
                        setStatus(chatStatusDot, chatStatus, "success", `Answer ready (${contexts} contexts)`);
                    }
                }
            } else {
                renderChatResponse(data.error || "Chat failed", "Request failed", true);
                setStatus(
                    chatStatusDot,
                    chatStatus,
                    "error",
                    data.error ? `Chat failed: ${data.error}` : "Chat failed"
                );
            }
        } catch (err) {
            showJson(chatResult, { ok: false, error: err.message || "Chat failed" });
            renderChatResponse(err.message || "Chat failed", "Request failed", true);
            setStatus(chatStatusDot, chatStatus, "error", "Chat failed");
        } finally {
            setBusy([chatBtn], false);
        }
    });
}

function updateChatAudio(audioBase64) {
    if (chatAnswerAudioContainer && audioBase64) {
        const audioSrc = `data:audio/wav;base64,${audioBase64}`;
        chatAnswerAudioContainer.innerHTML = `<audio src="${audioSrc}" controls></audio>`;
        chatAnswerAudioContainer.classList.remove("hidden");
    }
}

if (configForm) {
  configForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    setStatus(configStatusDot, configStatus, "working", "Saving configuration...");
    setBusy([configSaveBtn, resetBtn], true);
    const payload = {
      default_index: currentIndexName(),
      top_k: Number(retrievalTopKInput?.value || 4),
      use_semantic_ranker: Boolean(defaultSemanticRankerInput?.checked),
      azure_tier: document.getElementById("azureTier")?.value,
      model: document.getElementById("model")?.value,
      pricing: {
        input_per_1k_tokens_usd: Number(document.getElementById("inputCost")?.value || 0),
        output_per_1k_tokens_usd: Number(document.getElementById("outputCost")?.value || 0),
        semantic_query_cost_usd: Number(document.getElementById("semanticCost")?.value || 0),
      },
    };
    try {
      const res = await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      showJson(configResult, data);
      if (data.ok) {
        applyConfig(data.config);
        setStatus(configStatusDot, configStatus, "success", "Configuration saved");
        await loadDocuments();
      } else {
        setStatus(
          configStatusDot,
          configStatus,
          "error",
          data.error ? `Save failed: ${data.error}` : "Save failed"
        );
      }
    } catch (err) {
      showJson(configResult, { ok: false, error: err.message || "Save failed" });
      setStatus(configStatusDot, configStatus, "error", "Save failed");
    } finally {
      setBusy([configSaveBtn, resetBtn], false);
    }
  });
}

if (resetBtn) {
  resetBtn.addEventListener("click", async () => {
    setStatus(configStatusDot, configStatus, "working", "Resetting session...");
    setBusy([configSaveBtn, resetBtn], true);
    try {
      const res = await fetch("/api/reset", { method: "POST" });
      const data = await res.json();
      showJson(configResult, data);
      if (data.ok) {
        setStatus(configStatusDot, configStatus, "success", "Session reset");
      } else {
        setStatus(
          configStatusDot,
          configStatus,
          "error",
          data.error ? `Reset failed: ${data.error}` : "Reset failed"
        );
      }
    } catch (err) {
      showJson(configResult, { ok: false, error: err.message || "Reset failed" });
      setStatus(configStatusDot, configStatus, "error", "Reset failed");
    } finally {
      setBusy([configSaveBtn, resetBtn], false);
    }
  });
}

if (defaultIndexInput) {
    defaultIndexInput.addEventListener("change", loadDocuments);
    defaultIndexInput.addEventListener("blur", loadDocuments);
}

async function startRecording() {
    const currentHostname = window.location.hostname;
    const currentProtocol = window.location.protocol;
    const isLocalhost = currentHostname === 'localhost' || currentHostname === '127.0.0.1' || currentHostname === '[::1]';
    const isSecure = window.isSecureContext;
    
    // Check if getUserMedia is supported
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus(chatStatusDot, chatStatus, "error", "Microphone not supported");
        alert("❌ BROWSER NOT SUPPORTED\n\nYour browser doesn't support the Web Audio API.\n\nPlease use:\n• Chrome (recommended)\n• Firefox\n• Edge\n• Safari 11+");
        return;
    }
    
    // Check secure context - but provide helpful message
    if (!isSecure && !isLocalhost) {
        setStatus(chatStatusDot, chatStatus, "error", "Insecure context");
        
        let message = "⚠️ NOT A SECURE CONTEXT\n\n";
        message += "Current URL: " + window.location.href + "\n";
        message += "Protocol: " + currentProtocol + " (UNSECURE)\n";
        message += "Hostname: " + currentHostname + " (NOT localhost)\n\n";
        message += "🔒 WHY THIS HAPPENS:\n";
        message += "Browsers block microphone on HTTP for security.\n";
        message += "Only localhost or HTTPS is allowed.\n\n";
        message += "✅ SOLUTIONS (try in order):\n\n";
        message += "1️⃣ USE LOCALHOST (EASIEST):\n";
        message += "   Open: http://localhost:5000\n\n";
        message += "2️⃣ CHROME FLAG (TEMPORARY):\n";
        message += "   1. Open: chrome://flags/#unsafely-treat-insecure-origin-as-secure\n";
        message += "   2. Search: 'unsafely'\n";
        message += "   3. Enable it\n";
        message += "   4. Add your URL: " + currentProtocol + "//" + currentHostname + ":5000\n";
        message += "   5. Click 'Relaunch'\n\n";
        message += "3️⃣ NGROK (FOR NETWORK):\n";
        message += "   npm install -g ngrok\n";
        message += "   ngrok http 5000\n";
        message += "   Use the https:// URL\n\n";
        
        alert(message);
        return;
    }
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            } 
        });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        mediaRecorder.onstop = () => {
            stream.getTracks().forEach(track => track.stop());
        };
        mediaRecorder.start();
        isRecording = true;
        recordBtn.classList.add("recording");
        recordBtn.textContent = "⏹";
        let secondsLeft = MAX_RECORDING_SECONDS;
        setStatus(chatStatusDot, chatStatus, "working", `Recording... ${secondsLeft}s`);
        recordingTimer = setInterval(() => {
            secondsLeft--;
            if (secondsLeft <= 0) {
                stopRecording();
            } else {
                setStatus(chatStatusDot, chatStatus, "working", `Recording... ${secondsLeft}s`);
            }
        }, 1000);
    } catch (err) {
        let errorMsg = "Microphone access denied. ";
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            errorMsg += "You denied permission. Click the lock icon in address bar to allow.";
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
            errorMsg += "No microphone found. Please connect one.";
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
            errorMsg += "Microphone is busy with another app.";
        } else if (window.location.protocol === 'http:' && window.location.hostname !== 'localhost') {
            errorMsg += "HTTP is not secure. Use localhost or HTTPS.";
        } else {
            errorMsg += err.message;
        }
        setStatus(chatStatusDot, chatStatus, "error", errorMsg);
        alert("Microphone Error:\n" + errorMsg + "\n\nTest at: " + window.location.origin + "/mic-test");
        isRecording = false;
    }
}

async function stopRecording() {
    if (!mediaRecorder || !isRecording) return;
    clearInterval(recordingTimer);
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    isRecording = false;
    recordBtn.classList.remove("recording");
    recordBtn.textContent = "🔴";
    setStatus(chatStatusDot, chatStatus, "success", "Recording complete");
    const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
    await sendVoiceMessage(audioBlob);
}

async function sendVoiceMessage(audioBlob) {
    renderPendingChat("[Voice message]");
    setStatus(chatStatusDot, chatStatus, "working", "Processing voice...");
    setBusy([chatBtn], true);
    const formData = new FormData();
    formData.append("file", audioBlob, "voice.wav");
    formData.append("index_name", runtimeConfig?.default_index || "default-index");
    formData.append("top_k", String(runtimeConfig?.top_k ?? 4));
    formData.append("use_semantic_ranker", String(Boolean(runtimeConfig?.use_semantic_ranker)));
    formData.append("use_vector_search", String(Boolean(runtimeConfig?.use_vector_search)));
    formData.append("temperature", String(runtimeConfig?.chat_temperature ?? 0.2));
    formData.append("max_tokens", String(runtimeConfig?.chat_max_tokens ?? 600));
    formData.append("model", runtimeConfig?.model || "");
    try {
        const res = await fetch("/api/voice-chat", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();
        showJson(chatResult, data);
        if (data.ok) {
            const contexts = Array.isArray(data.contexts) ? data.contexts.length : 0;
            const modelName = data.model ? ` · ${data.model}` : "";
            renderChatResponse(data.answer || "No answer returned.", `${contexts} contexts${modelName}`);
            setStatus(chatStatusDot, chatStatus, "success", `Voice response ready (${contexts} contexts)`);
        } else {
            renderChatResponse(data.error || "Voice chat failed", "Request failed", true);
            setStatus(chatStatusDot, chatStatus, "error", data.error ? `Voice failed: ${data.error}` : "Voice failed");
        }
    } catch (err) {
        showJson(chatResult, { ok: false, error: err.message || "Voice chat failed" });
        renderChatResponse(err.message || "Voice chat failed", "Request failed", true);
        setStatus(chatStatusDot, chatStatus, "error", "Voice chat failed");
    } finally {
        setBusy([chatBtn], false);
    }
}

if (speakerToggleBtn) {
    speakerToggleBtn.addEventListener("click", () => {
        isSpeakerEnabled = !isSpeakerEnabled;
        speakerToggleBtn.classList.toggle("active", isSpeakerEnabled);
        if (isSpeakerEnabled) {
            setStatus(chatStatusDot, chatStatus, "success", "TTS enabled");
        } else {
            setStatus(chatStatusDot, chatStatus, "success", "TTS disabled");
        }
    });
}

if (voiceToggleBtn) {
    voiceToggleBtn.addEventListener("click", () => {
        isVoiceMode = !isVoiceMode;
        voiceToggleBtn.classList.toggle("active", isVoiceMode);
        if (isVoiceMode) {
            recordBtn?.classList.remove("hidden");
            setStatus(chatStatusDot, chatStatus, "success", "Voice mode enabled");
        } else {
            recordBtn?.classList.add("hidden");
            setStatus(chatStatusDot, chatStatus, "success", "Voice mode disabled");
            if (isRecording) {
                stopRecording();
            }
        }
    });
}

if (recordBtn) {
    recordBtn.addEventListener("click", () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });
}

async function initializeApp() {
    await loadConfig();
    if (documentsList) {
        await loadDocuments();
    }
    if (recordBtn) {
        recordBtn.classList.add("hidden");
    }
    if (speakerToggleBtn) {
        speakerToggleBtn.classList.remove("active");
    }
}

initializeApp();

const ttsForm = document.getElementById("ttsForm");
const ttsResult = document.getElementById("ttsResult");
const ttsStatusDot = document.getElementById("ttsStatusDot");
const ttsStatus = document.getElementById("ttsStatus");
const ttsVoice = document.getElementById("ttsVoice");
const ttsSpeed = document.getElementById("ttsSpeed");
const ttsSpeedValue = document.getElementById("ttsSpeedValue");
const ttsPitch = document.getElementById("ttsPitch");
const ttsPitchValue = document.getElementById("ttsPitchValue");
const ttsStyle = document.getElementById("ttsStyle");
const testTtsBtn = document.getElementById("testTtsBtn");

if (ttsForm) {
    fetch("/api/tts-config")
        .then(res => res.json())
        .then(data => {
            if (data.ok && data.settings) {
                if (ttsVoice) ttsVoice.value = data.settings.voice || "en-US-AriaNeural";
                if (ttsSpeed) {
                    ttsSpeed.value = data.settings.speed || 1.0;
                    if (ttsSpeedValue) ttsSpeedValue.textContent = (data.settings.speed || 1.0).toFixed(1) + "x";
                }
                if (ttsPitch) {
                    ttsPitch.value = data.settings.pitch || 0;
                    if (ttsPitchValue) ttsPitchValue.textContent = data.settings.pitch || 0;
                }
                if (ttsStyle) ttsStyle.value = data.settings.style || "default";
            }
        })
        .catch(err => console.error("Failed to load TTS config:", err));

    if (ttsSpeed) {
        ttsSpeed.addEventListener("input", () => {
            if (ttsSpeedValue) ttsSpeedValue.textContent = parseFloat(ttsSpeed.value).toFixed(1) + "x";
        });
    }

    if (ttsPitch) {
        ttsPitch.addEventListener("input", () => {
            if (ttsPitchValue) ttsPitchValue.textContent = ttsPitch.value;
        });
    }

    ttsForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (ttsStatus) {
            ttsStatus.textContent = "Saving TTS settings...";
            if (ttsStatusDot) ttsStatusDot.className = "status-dot working";
        }
        
        const payload = {
            voice: ttsVoice?.value,
            speed: parseFloat(ttsSpeed?.value || 1.0),
            pitch: parseInt(ttsPitch?.value || 0),
            style: ttsStyle?.value
        };

        try {
            const res = await fetch("/api/tts-config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (ttsResult) ttsResult.textContent = JSON.stringify(data, null, 2);
            if (data.ok) {
                if (ttsStatus) ttsStatus.textContent = "TTS settings saved";
                if (ttsStatusDot) ttsStatusDot.className = "status-dot success";
            } else {
                if (ttsStatus) ttsStatus.textContent = "Failed to save settings: " + (data.error || "Unknown error");
                if (ttsStatusDot) ttsStatusDot.className = "status-dot error";
            }
        } catch (err) {
            if (ttsResult) ttsResult.textContent = JSON.stringify({ ok: false, error: err.message });
            if (ttsStatus) ttsStatus.textContent = "Failed to save settings";
            if (ttsStatusDot) ttsStatusDot.className = "status-dot error";
        }
    });
}

if (testTtsBtn) {
    testTtsBtn.addEventListener("click", async () => {
        if (ttsStatus) {
            ttsStatus.textContent = "Testing TTS...";
            if (ttsStatusDot) ttsStatusDot.className = "status-dot working";
        }
        
        try {
            const res = await fetch("/api/tts-test", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: "Hello! This is a test of the text to speech system. Your settings are being applied." })
            });
            const data = await res.json();
            if (ttsResult) ttsResult.textContent = JSON.stringify(data, null, 2);
            
            if (data.ok && data.audio) {
                const audio = new Audio(`data:audio/wav;base64,${data.audio}`);
                audio.play();
                if (ttsStatus) ttsStatus.textContent = "Playing test audio...";
                if (ttsStatusDot) ttsStatusDot.className = "status-dot success";
                
                audio.onended = () => {
                    if (ttsStatus) ttsStatus.textContent = "Test complete";
                };
            } else {
                if (ttsStatus) ttsStatus.textContent = "TTS test failed: " + (data.error || "Unknown error");
                if (ttsStatusDot) ttsStatusDot.className = "status-dot error";
            }
        } catch (err) {
            if (ttsResult) ttsResult.textContent = JSON.stringify({ ok: false, error: err.message });
            if (ttsStatus) ttsStatus.textContent = "TTS test failed";
            if (ttsStatusDot) ttsStatusDot.className = "status-dot error";
        }
    });
}
