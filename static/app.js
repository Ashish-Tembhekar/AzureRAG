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
let audioMimeType = "audio/webm";
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
  const streamToggle = document.getElementById("streamResponses");
  if (streamToggle) {
    streamToggle.checked = Boolean(config.stream_responses !== false);
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
  const stream = getChatStream();
  if (!stream) return;

  let thinking = document.getElementById("chatThinking");
  if (!thinking) {
    thinking = document.createElement("div");
    thinking.id = "chatThinking";
    thinking.className = "message system hidden thinking-row";
    thinking.innerHTML = '<div class="avatar">AI</div><p class="thinking-text">Thinking ...</p>';
    stream.appendChild(thinking);
  }
  thinking.classList.remove("hidden");

  let answerCard = document.getElementById("chatAnswerCard");
  if (answerCard) {
    answerCard.classList.add("hidden");
    answerCard.classList.remove("error-state");
  }
}

function renderChatResponse(answer, meta, audioBase64 = null, isError = false) {
    const stream = getChatStream();
    if (!stream) return;

    const thinking = document.getElementById("chatThinking");
    if (thinking) thinking.classList.add("hidden");

    let answerCard = document.getElementById("chatAnswerCard");
    if (!answerCard) {
      answerCard = document.createElement("div");
      answerCard.id = "chatAnswerCard";
      answerCard.className = "message system hidden";
      answerCard.innerHTML = '<div class="avatar">AI</div><div class="bubble"><p class="tag">Assistant</p><p id="chatAnswerText"></p><p id="chatAnswerMeta" class="bubble-meta"></p></div>';
      stream.appendChild(answerCard);
    }
    
    const chatAnswerText = document.getElementById("chatAnswerText");
    const chatAnswerMeta = document.getElementById("chatAnswerMeta");
    
    if (chatAnswerText) chatAnswerText.innerHTML = marked.parse(answer);
    if (chatAnswerMeta) chatAnswerMeta.textContent = meta || "";
    answerCard.classList.remove("hidden");
    answerCard.classList.toggle("error-state", isError);

    stream.scrollTop = stream.scrollHeight;
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

const chatInput = document.getElementById("chatInput");
if (chatInput) {
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatBtn?.click();
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
        document.getElementById("chatInput").value = "";
        setBusy([chatBtn], true);
        const useStreaming = runtimeConfig?.stream_responses !== false;
        if (isSpeakerEnabled) {
            await sendChatWithTts(query);
        } else if (useStreaming) {
            await sendStreamingChat(query);
        } else {
            await sendNonStreamingChat(query);
        }
        setBusy([chatBtn], false);
    });
}

async function sendStreamingChat(query) {
    renderPendingChat(query);
    setStatus(chatStatusDot, chatStatus, "working", "Retrieving context...");
    const payload = {
        query,
        index_name: runtimeConfig?.default_index || "default-index",
        top_k: Number(runtimeConfig?.top_k ?? 4),
        use_semantic_ranker: Boolean(runtimeConfig?.use_semantic_ranker),
        session_id: activeSessionId || undefined,
    };
    console.log("[Stream] Sending:", payload);
    try {
        const res = await fetch("/api/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!res.ok) {
            throw new Error("HTTP " + res.status + ": " + res.statusText);
        }

        const stream = getChatStream();
        const thinking = document.getElementById("chatThinking");
        if (thinking) thinking.classList.add("hidden");

        const userEl = document.createElement("div");
        userEl.className = "message user";
        userEl.innerHTML = '<div class="bubble user-bubble"><p class="tag">User</p><p>' + escapeHtml(query) + '</p></div><div class="avatar user-avatar">U</div>';
        if (stream) stream.appendChild(userEl);

        const assistantEl = document.createElement("div");
        assistantEl.className = "message system";
        assistantEl.innerHTML = '<div class="avatar">AI</div><div class="bubble"><p class="tag">Assistant</p><p id="streamingText" class="streaming-text"></p></div>';
        if (stream) stream.appendChild(assistantEl);

        const streamingText = document.getElementById("streamingText");

        let fullAnswer = "";
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let finalData = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";
            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    const jsonStr = line.slice(6);
                    try {
                        const data = JSON.parse(jsonStr);
                        if (data.type === "token") {
                            fullAnswer += data.token;
                            if (streamingText) {
                                streamingText.innerHTML = marked.parse(fullAnswer) + '<span class="cursor">▋</span>';
                            }
                            if (stream) stream.scrollTop = stream.scrollHeight;
                        } else if (data.type === "done") {
                            finalData = data;
                        } else if (data.type === "error") {
                            throw new Error(data.error);
                        }
                    } catch (e) {
                        if (e.message && e.message.includes("HTTP")) throw e;
                        console.error("[Stream] Parse error:", e, jsonStr);
                    }
                }
            }
        }

        if (streamingText) {
            streamingText.innerHTML = marked.parse(fullAnswer);
        }

        if (finalData) {
            const session_id = finalData.session_id;
            if (session_id && session_id !== activeSessionId) {
                activeSessionId = session_id;
                localStorage.setItem("rag_session_id", session_id);
                loadSessions();
            }
            currentMessages.push({ role: "user", content: query });
            currentMessages.push({ role: "assistant", content: fullAnswer });

            if (finalData.cost_summary) {
                saveSessionLogs(session_id, finalData.cost_summary);
                const logsHtml = '<details class="trace"><summary>View Logs</summary><pre>' + escapeHtml(JSON.stringify(finalData.cost_summary, null, 2)) + '</pre></details>';
                const bubble = assistantEl.querySelector(".bubble");
                if (bubble) bubble.insertAdjacentHTML("beforeend", logsHtml);
            }
        }

        if (stream) stream.scrollTop = stream.scrollHeight;
        setStatus(chatStatusDot, chatStatus, "success", "Answer ready");
        console.log("[Stream] Complete:", finalData);

        if (isSpeakerEnabled && fullAnswer) {
            generateAndPlayAudio(fullAnswer, stream);
        }
    } catch (err) {
        console.error("[Stream] Error:", err);
        renderChatResponse(err.message || "Chat failed", "Request failed", null, true);
        setStatus(chatStatusDot, chatStatus, "error", "Chat failed: " + (err.message || "Unknown error"));
    }
}

async function sendNonStreamingChat(query) {
    renderPendingChat(query);
    setStatus(chatStatusDot, chatStatus, "working", "Retrieving context and generating answer...");
    const payload = {
        query,
        index_name: runtimeConfig?.default_index || "default-index",
        top_k: Number(runtimeConfig?.top_k ?? 4),
        use_semantic_ranker: Boolean(runtimeConfig?.use_semantic_ranker),
        session_id: activeSessionId || undefined,
    };
    console.log("[Chat] Sending:", payload);
    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        console.log("[Chat] Response status:", res.status);
        if (!res.ok) {
            throw new Error("HTTP " + res.status + ": " + res.statusText);
        }
        const data = await res.json();
        console.log("[Chat] Response data:", data);
        if (data.ok) {
            const contexts = Array.isArray(data.contexts) ? data.contexts.length : 0;
            const modelName = data.model ? " · " + data.model : "";

            if (data.session_id && data.session_id !== activeSessionId) {
                activeSessionId = data.session_id;
                localStorage.setItem("rag_session_id", data.session_id);
                loadSessions();
            }

            currentMessages.push({ role: "user", content: query });
            currentMessages.push({ role: "assistant", content: data.answer || "" });

            if (data.cost_summary) {
                saveSessionLogs(data.session_id, data.cost_summary);
            }

            const stream = getChatStream();
            if (stream) {
                const thinking = document.getElementById("chatThinking");
                if (thinking) thinking.classList.add("hidden");
                const userEl = document.createElement("div");
                userEl.className = "message user";
                userEl.innerHTML = '<div class="bubble user-bubble"><p class="tag">User</p><p>' + escapeHtml(query) + '</p></div><div class="avatar user-avatar">U</div>';
                stream.appendChild(userEl);
            }

            appendMessage("assistant", data.answer || "", data.cost_summary);

            const meta = document.getElementById("chatAnswerMeta");
            if (meta) meta.textContent = contexts + " contexts" + modelName;
            setStatus(chatStatusDot, chatStatus, "success", "Answer ready (" + contexts + " contexts)");

            if (isSpeakerEnabled && data.answer) {
                const stream = getChatStream();
                generateAndPlayAudio(data.answer, stream);
            }
        } else {
            console.error("[Chat] API error:", data.error);
            renderChatResponse(data.error || "Chat failed", "Request failed", null, true);
            setStatus(chatStatusDot, chatStatus, "error", data.error ? "Chat failed: " + data.error : "Chat failed");
        }
    } catch (err) {
        console.error("[Chat] Request error:", err);
        showJson(chatResult, { ok: false, error: err.message || "Chat failed" });
        renderChatResponse(err.message || "Chat failed", "Request failed", null, true);
        setStatus(chatStatusDot, chatStatus, "error", "Chat failed: " + (err.message || "Unknown error"));
    }
}

function playAndRenderAudio(audioBase64, streamEl, shouldAutoplay = true) {
    if (!audioBase64) {
        return;
    }

    const audioSrc = "data:audio/wav;base64," + audioBase64;
    if (shouldAutoplay) {
        const audio = new Audio(audioSrc);
        audio.play().catch(e => console.error("[TTS] Play error:", e));
    }

    if (streamEl) {
        const audioEl = document.createElement("div");
        audioEl.className = "message system";
        audioEl.innerHTML = '<div class="avatar">ðŸ”Š</div><div class="bubble tts-bubble"><p class="tag">Audio Response</p><audio src="' + audioSrc + '" controls></audio></div>';
        streamEl.appendChild(audioEl);
        streamEl.scrollTop = streamEl.scrollHeight;
    }
}

async function sendChatWithTts(query) {
    renderPendingChat(query);
    setStatus(chatStatusDot, chatStatus, "working", "Generating answer and audio...");
    const payload = {
        query,
        index_name: runtimeConfig?.default_index || "default-index",
        top_k: Number(runtimeConfig?.top_k ?? 4),
        use_semantic_ranker: Boolean(runtimeConfig?.use_semantic_ranker),
        session_id: activeSessionId || undefined,
    };
    try {
        const res = await fetch("/api/chat-with-tts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!res.ok) {
            throw new Error("HTTP " + res.status + ": " + res.statusText);
        }

        const data = await res.json();
        showJson(chatResult, data);
        if (!data.ok) {
            throw new Error(data.error || "Chat with TTS failed");
        }

        const contexts = Array.isArray(data.contexts) ? data.contexts.length : 0;
        const modelName = data.model ? " Â· " + data.model : "";

        if (data.session_id && data.session_id !== activeSessionId) {
            activeSessionId = data.session_id;
            localStorage.setItem("rag_session_id", data.session_id);
            loadSessions();
        }

        currentMessages.push({ role: "user", content: query });
        currentMessages.push({ role: "assistant", content: data.answer || "" });

        if (data.cost_summary) {
            saveSessionLogs(data.session_id, data.cost_summary);
        }

        const stream = getChatStream();
        if (stream) {
            const thinking = document.getElementById("chatThinking");
            if (thinking) thinking.classList.add("hidden");
            const userEl = document.createElement("div");
            userEl.className = "message user";
            userEl.innerHTML = '<div class="bubble user-bubble"><p class="tag">User</p><p>' + escapeHtml(query) + '</p></div><div class="avatar user-avatar">U</div>';
            stream.appendChild(userEl);
        }

        appendMessage("assistant", data.answer || "", data.cost_summary);

        const meta = document.getElementById("chatAnswerMeta");
        if (meta) meta.textContent = contexts + " contexts" + modelName;

        playAndRenderAudio(data.audio, stream, true);
        setStatus(chatStatusDot, chatStatus, "success", "Answer and audio ready (" + contexts + " contexts)");
    } catch (err) {
        console.error("[Chat+TTS] Request error:", err);
        showJson(chatResult, { ok: false, error: err.message || "Chat with TTS failed" });
        renderChatResponse(err.message || "Chat with TTS failed", "Request failed", null, true);
        setStatus(chatStatusDot, chatStatus, "error", "Chat with TTS failed: " + (err.message || "Unknown error"));
    }
}

async function generateAndPlayAudio(answerText, streamEl) {
    try {
        setStatus(chatStatusDot, chatStatus, "working", "Generating audio...");
        console.log("[TTS] Requesting audio for answer");
        const ttsRes = await fetch("/api/tts-generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: answerText })
        });
        const ttsData = await ttsRes.json();
        console.log("[TTS] Response:", ttsData);
        if (ttsData.ok && ttsData.audio) {
            const audioSrc = "data:audio/wav;base64," + ttsData.audio;
            const audio = new Audio(audioSrc);
            audio.play().catch(e => console.error("[TTS] Play error:", e));

            if (streamEl) {
                const audioEl = document.createElement("div");
                audioEl.className = "message system";
                audioEl.innerHTML = '<div class="avatar">🔊</div><div class="bubble tts-bubble"><p class="tag">Audio Response</p><audio src="' + audioSrc + '" controls></audio></div>';
                streamEl.appendChild(audioEl);
                streamEl.scrollTop = streamEl.scrollHeight;
            }

            setStatus(chatStatusDot, chatStatus, "success", "Audio playing");
        } else {
            console.error("[TTS] Generation failed:", ttsData.error);
            setStatus(chatStatusDot, chatStatus, "error", "TTS failed: " + (ttsData.error || "Unknown error"));
        }
    } catch (ttsErr) {
        console.error("[TTS] Error:", ttsErr);
        setStatus(chatStatusDot, chatStatus, "error", "TTS error");
    }
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
      stream_responses: Boolean(document.getElementById("streamResponses")?.checked),
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
        if (activeSessionId) {
            clearSessionLogs(activeSessionId);
        }
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
        audioMimeType = mediaRecorder.mimeType || "audio/webm";
        audioChunks = [];
        mediaRecorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) {
                audioChunks.push(e.data);
            }
        };
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
    mediaRecorder.stream.getTracks().forEach(track => track.stop());

    const audioBlob = await new Promise((resolve) => {
        mediaRecorder.onstop = () => {
            const blob = new Blob(audioChunks, { type: audioMimeType });
            resolve(blob);
        };
        mediaRecorder.stop();
    });

    isRecording = false;
    recordBtn.classList.remove("recording");
    recordBtn.textContent = "🔴";
    setStatus(chatStatusDot, chatStatus, "success", "Recording complete");
    console.log("[Voice] Chunks:", audioChunks.length, "MIME:", audioMimeType);
    console.log("[Voice] Blob size:", audioBlob.size, "type:", audioBlob.type);
    await sendVoiceMessage(audioBlob);
}

async function sendVoiceMessage(audioBlob) {
    renderPendingChat("[Voice message]");
    setStatus(chatStatusDot, chatStatus, "working", "Processing voice...");
    setBusy([chatBtn], true);
    const formData = new FormData();
    const mimeToExt = { "webm": "webm", "ogg": "ogg", "mp4": "mp4", "wav": "wav", "mpeg": "mp3" };
    const mimeKey = (audioBlob.type || "").split("/").pop().split(";")[0].trim();
    const ext = mimeToExt[mimeKey] || "webm";
    formData.append("file", audioBlob, "voice." + ext);
    console.log("[Voice] Sending as:", "voice." + ext, "MIME:", audioBlob.type);
    formData.append("index_name", runtimeConfig?.default_index || "default-index");
    formData.append("top_k", String(runtimeConfig?.top_k ?? 4));
    formData.append("use_semantic_ranker", String(Boolean(runtimeConfig?.use_semantic_ranker)));
    formData.append("use_vector_search", String(Boolean(runtimeConfig?.use_vector_search)));
    formData.append("temperature", String(runtimeConfig?.chat_temperature ?? 0.2));
    formData.append("max_tokens", String(runtimeConfig?.chat_max_tokens ?? 600));
    formData.append("model", runtimeConfig?.model || "");
    if (activeSessionId) formData.append("session_id", activeSessionId);
    console.log("[Voice] Sending voice message for session:", activeSessionId);
    try {
        const res = await fetch("/api/voice-chat", {
            method: "POST",
            body: formData,
        });
        console.log("[Voice] Response status:", res.status);
        const data = await res.json();
        console.log("[Voice] Response data:", data);
        showJson(chatResult, data);
        if (data.ok) {
            const contexts = Array.isArray(data.contexts) ? data.contexts.length : 0;
            const modelName = data.model ? " · " + data.model : "";

            if (data.session_id && data.session_id !== activeSessionId) {
                activeSessionId = data.session_id;
                localStorage.setItem("rag_session_id", data.session_id);
                loadSessions();
            }

            const stream = getChatStream();
            if (stream) {
                const thinking = document.getElementById("chatThinking");
                if (thinking) thinking.classList.add("hidden");
            }

            if (data.cost_summary) {
                saveSessionLogs(data.session_id, data.cost_summary);
            }

            appendMessage("assistant", data.answer || "", data.cost_summary);

            const meta = document.getElementById("chatAnswerMeta");
            if (meta) meta.textContent = contexts + " contexts" + modelName;

            setStatus(chatStatusDot, chatStatus, "success", "Voice response ready (" + contexts + " contexts)");
        } else {
            console.error("[Voice] API error:", data.error);
            console.error("[Voice] API error:", data.error);
            renderChatResponse(data.error || "Voice chat failed", "Request failed", null, true);
            setStatus(chatStatusDot, chatStatus, "error", data.error ? "Voice failed: " + data.error : "Voice failed");
        }
    } catch (err) {
        console.error("[Voice] Request error:", err);
        showJson(chatResult, { ok: false, error: err.message || "Voice chat failed" });
        renderChatResponse(err.message || "Voice chat failed", "Request failed", null, true);
        setStatus(chatStatusDot, chatStatus, "error", "Voice chat failed: " + (err.message || "Unknown error"));
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
    if (activeSessionId) {
        await switchSession(activeSessionId);
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

// === SESSION MANAGEMENT ===
let activeSessionId = localStorage.getItem("rag_session_id") || null;
let sessions = [];
let currentMessages = [];

function getSidebar() { return document.getElementById("sessionsSidebar"); }
function getSessionsList() { return document.getElementById("sessionsList"); }
function getChatShell() { return document.getElementById("chatShell"); }
function getChatStream() { return document.querySelector(".chat-stream"); }
function getActiveSessionLabel() { return document.getElementById("activeSessionLabel"); }
function getChatInput() { return document.getElementById("chatInput"); }

function saveSessionLogs(sessionId, costSummary) {
    if (!sessionId || !costSummary) return;
    const key = "rag_logs_" + sessionId;
    let logs = [];
    try {
        logs = JSON.parse(localStorage.getItem(key) || "[]");
    } catch { logs = []; }
    logs.push(costSummary);
    try {
        localStorage.setItem(key, JSON.stringify(logs));
    } catch (e) {
        console.warn("[Logs] localStorage full, clearing old sessions");
        try { localStorage.removeItem(key); localStorage.setItem(key, JSON.stringify(logs)); } catch {}
    }
}

function getSessionLogs(sessionId) {
    if (!sessionId) return [];
    const key = "rag_logs_" + sessionId;
    try {
        return JSON.parse(localStorage.getItem(key) || "[]");
    } catch { return []; }
}

function clearSessionLogs(sessionId) {
    if (!sessionId) return;
    localStorage.removeItem("rag_logs_" + sessionId);
}

async function loadSessions() {
    const list = getSessionsList();
    if (!list) return;
    list.innerHTML = '<div class="sessions-loading">Loading...</div>';
    try {
        const res = await fetch("/api/sessions?limit=50");
        const data = await res.json();
        console.log("[Sessions] List response:", data);
        if (data.ok && Array.isArray(data.sessions)) {
            sessions = data.sessions;
            renderSessionsList();
        } else {
            console.error("[Sessions] Failed to load:", data.error);
            list.innerHTML = '<div class="empty-sessions">Failed to load sessions: ' + (data.error || "Unknown error") + '</div>';
        }
    } catch (err) {
        console.error("[Sessions] Network error:", err);
        list.innerHTML = '<div class="empty-sessions">Network error loading sessions</div>';
    }
}

function renderSessionsList() {
    const list = getSessionsList();
    if (!list) return;
    if (sessions.length === 0) {
        list.innerHTML = '<div class="empty-sessions">No chat history yet.<br>Start a new conversation!</div>';
        return;
    }
    list.innerHTML = sessions.map(s => {
        const isActive = s.session_id === activeSessionId;
        const date = formatSessionDate(s.created_at);
        return `
        <div class="session-item${isActive ? ' active' : ''}" data-session-id="${s.session_id}" title="${escapeHtml(s.title || 'Untitled')}">
            <div class="session-item-info">
                <span class="session-item-title">${escapeHtml(s.title || 'Untitled')}</span>
                <span class="session-item-date">${date}</span>
            </div>
            <div class="session-item-actions">
                <button class="session-action-btn edit" title="Rename" data-action="rename" data-id="${s.session_id}">✎</button>
                <button class="session-action-btn delete" title="Delete" data-action="delete" data-id="${s.session_id}">✕</button>
            </div>
        </div>
        `;
    }).join('');

    list.querySelectorAll(".session-item").forEach(item => {
        item.addEventListener("click", (e) => {
            if (e.target.closest(".session-action-btn")) return;
            const id = item.getAttribute("data-session-id");
            switchSession(id);
        });
    });

    list.querySelectorAll(".session-action-btn").forEach(btn => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            const action = btn.getAttribute("data-action");
            const id = btn.getAttribute("data-id");
            if (action === "delete") {
                if (confirm("Delete this chat? This cannot be undone.")) {
                    deleteSession(id);
                }
            } else if (action === "rename") {
                startRenameSession(id);
            }
        });
    });
}

function formatSessionDate(isoDate) {
    if (!isoDate) return "";
    try {
        const d = new Date(isoDate);
        const now = new Date();
        const diffDays = Math.floor((now - d) / (1000 * 60 * 60 * 24));
        if (diffDays === 0) return "Today";
        if (diffDays === 1) return "Yesterday";
        if (diffDays < 7) return `${diffDays} days ago`;
        return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: d.getFullYear() !== now.getFullYear() ? "numeric" : undefined });
    } catch {
        return "";
    }
}

async function switchSession(sessionId) {
    activeSessionId = sessionId;
    localStorage.setItem("rag_session_id", sessionId);
    const shell = getChatShell();
    if (shell) shell.classList.remove("with-sidebar");

    const stream = getChatStream();
    if (stream) stream.innerHTML = '<div class="sessions-loading">Loading conversation...</div>';

    try {
        const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
        console.log("[SwitchSession] Response status:", res.status);
        if (!res.ok) {
            throw new Error("HTTP " + res.status + ": " + res.statusText);
        }
        const data = await res.json();
        console.log("[SwitchSession] Response data:", data);

        if (!data.ok) {
            throw new Error(data.error || "Failed to load session");
        }

        currentMessages = data.messages || [];
        updateActiveSessionLabel(data.session?.title || "Chat");

        if (stream) {
            stream.innerHTML = '';

            const intro = document.createElement("div");
            intro.className = "message system intro-card";
            intro.innerHTML = `
                <div class="avatar">AI</div>
                <div class="bubble">
                    <p class="tag">System Assistant</p>
                    <p>Use the settings page to upload and manage documents, then ask questions here.</p>
                    <p id="chatConfigHint" class="bubble-meta">Current settings: ${runtimeConfig?.default_index || "default-index"} · top ${runtimeConfig?.top_k || 4} · ${runtimeConfig?.use_semantic_ranker ? "semantic ranker on" : "semantic ranker off"}.</p>
                </div>
            `;
            stream.appendChild(intro);

            const sessionLogs = getSessionLogs(sessionId);
            let assistantIdx = 0;
            currentMessages.forEach(msg => {
                if (msg.role === "user") {
                    const el = document.createElement("div");
                    el.className = "message user";
                    el.innerHTML = `
                        <div class="bubble user-bubble">
                            <p class="tag">User</p>
                            <p>${escapeHtml(msg.content)}</p>
                        </div>
                        <div class="avatar user-avatar">U</div>
                    `;
                    stream.appendChild(el);
                } else if (msg.role === "assistant") {
                    const log = sessionLogs[assistantIdx] || null;
                    assistantIdx++;
                    let logsHtml = "";
                    if (log) {
                        logsHtml = '<details class="trace"><summary>View Logs</summary><pre>' + escapeHtml(JSON.stringify(log, null, 2)) + '</pre></details>';
                    }
                    const el = document.createElement("div");
                    el.className = "message system";
                    el.innerHTML = `
                        <div class="avatar">AI</div>
                        <div class="bubble">
                            <p class="tag">Assistant</p>
                            <p>${marked.parse(msg.content)}</p>
                            ${logsHtml}
                        </div>
                    `;
                    stream.appendChild(el);
                }
            });

            const thinking = document.createElement("div");
            thinking.id = "chatThinking";
            thinking.className = "message system hidden thinking-row";
            thinking.innerHTML = `<div class="avatar">AI</div><p class="thinking-text">Thinking ...</p>`;
            stream.appendChild(thinking);

            const answerCard = document.createElement("div");
            answerCard.id = "chatAnswerCard";
            answerCard.className = "message system hidden";
            answerCard.innerHTML = `
                <div class="avatar">AI</div>
                <div class="bubble">
                    <p class="tag">Assistant</p>
                    <p id="chatAnswerText"></p>
                    <p id="chatAnswerMeta" class="bubble-meta"></p>
                    <div id="chatAnswerAudioContainer" class="tts-audio-container hidden"></div>
                </div>
            `;
            stream.appendChild(answerCard);

            stream.scrollTop = stream.scrollHeight;
        }
    } catch (err) {
        console.error("[SwitchSession] Error:", err);
        if (stream) stream.innerHTML = '<div class="empty-sessions">Failed to load conversation: ' + escapeHtml(err.message || "Unknown error") + '</div>';
    }

    renderSessionsList();
}

function startNewSession() {
    activeSessionId = null;
    currentMessages = [];
    localStorage.removeItem("rag_session_id");
    const shell = getChatShell();
    if (shell) shell.classList.remove("with-sidebar");
    const stream = getChatStream();
    if (stream) {
        const semanticLabel = runtimeConfig?.use_semantic_ranker ? "semantic ranker on" : "semantic ranker off";
        stream.innerHTML = `
            <div class="message system intro-card">
                <div class="avatar">AI</div>
                <div class="bubble">
                    <p class="tag">System Assistant</p>
                    <p>Use the settings page to upload and manage documents, then ask questions here.</p>
                    <p id="chatConfigHint" class="bubble-meta">Current settings: ${runtimeConfig?.default_index || "default-index"} · top ${runtimeConfig?.top_k || 4} · ${semanticLabel}.</p>
                </div>
            </div>
            <div id="chatThinking" class="message system hidden thinking-row">
                <div class="avatar">AI</div>
                <p class="thinking-text">Thinking ...</p>
            </div>
            <div id="chatAnswerCard" class="message system hidden">
                <div class="avatar">AI</div>
                <div class="bubble">
                    <p class="tag">Assistant</p>
                    <p id="chatAnswerText"></p>
                    <p id="chatAnswerMeta" class="bubble-meta"></p>
                    <div id="chatAnswerAudioContainer" class="tts-audio-container hidden"></div>
                </div>
            </div>
        `;
    }
    updateActiveSessionLabel("Active Thread");
    renderSessionsList();
    const input = getChatInput();
    if (input) input.focus();
}

async function deleteSession(sessionId) {
    try {
        const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
        const data = await res.json();
        if (data.ok) {
            clearSessionLogs(sessionId);
            if (sessionId === activeSessionId) {
                startNewSession();
            } else {
                await loadSessions();
            }
        }
    } catch (err) {
        console.error("Failed to delete session:", err);
    }
}

function startRenameSession(sessionId) {
    const session = sessions.find(s => s.session_id === sessionId);
    if (!session) return;
    const item = document.querySelector(`.session-item[data-session-id="${sessionId}"]`);
    if (!item) return;
    const info = item.querySelector(".session-item-info");
    const titleSpan = info.querySelector(".session-item-title");
    const currentTitle = session.title || "";
    titleSpan.innerHTML = `<input type="text" class="session-edit-input" value="${escapeHtml(currentTitle)}" maxlength="100" />`;
    const input = info.querySelector(".session-edit-input");
    input.focus();
    input.select();

    const finish = async (confirmRename) => {
        if (confirmRename) {
            const newTitle = input.value.trim();
            if (newTitle && newTitle !== currentTitle) {
                await renameSession(sessionId, newTitle);
            } else {
                renderSessionsList();
            }
        } else {
            renderSessionsList();
        }
    };

    input.addEventListener("blur", () => finish(true));
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            input.blur();
        } else if (e.key === "Escape") {
            e.preventDefault();
            finish(false);
        }
    });
}

async function renameSession(sessionId, title) {
    try {
        const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title })
        });
        const data = await res.json();
        if (data.ok) {
            await loadSessions();
            if (sessionId === activeSessionId) {
                updateActiveSessionLabel(title);
            }
        }
    } catch (err) {
        console.error("Failed to rename session:", err);
    }
}

function updateActiveSessionLabel(title) {
    const label = getActiveSessionLabel();
    if (label) label.textContent = title || "Active Thread";
}

function toggleSidebar() {
    const sidebar = getSidebar();
    const shell = getChatShell();
    if (!sidebar) return;
    const isHidden = sidebar.classList.contains("hidden");
    sidebar.classList.toggle("hidden");
    if (shell) {
        if (isHidden) {
            shell.classList.add("with-sidebar");
        } else {
            shell.classList.remove("with-sidebar");
        }
    }
}

async function appendMessage(role, content, costSummary) {
    const stream = getChatStream();
    if (!stream) return;

    if (role === "assistant") {
        const el = document.createElement("div");
        el.className = "message system";
        
        let logsHtml = '';
        if (costSummary) {
            logsHtml = `
                <details class="trace">
                    <summary>View Logs</summary>
                    <pre>${escapeHtml(JSON.stringify(costSummary, null, 2))}</pre>
                </details>
            `;
        }

        el.innerHTML = `
            <div class="avatar">AI</div>
            <div class="bubble">
                <p class="tag">Assistant</p>
                <p>${marked.parse(content)}</p>
                ${logsHtml}
            </div>
        `;
        stream.appendChild(el);
        stream.scrollTop = stream.scrollHeight;
    }
}

const sidebarToggleBtn = document.getElementById("sidebarToggleBtn");
if (sidebarToggleBtn) {
    sidebarToggleBtn.addEventListener("click", () => {
        toggleSidebar();
        if (!getSidebar()?.classList.contains("hidden")) {
            loadSessions();
        }
    });
}

const newChatBtn = document.getElementById("newChatBtn");
if (newChatBtn) {
    newChatBtn.addEventListener("click", () => {
        startNewSession();
    });
}

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
