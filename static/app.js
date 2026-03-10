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
const uploadBtn = uploadForm.querySelector("button[type='submit']");
const configSaveBtn = configForm.querySelector("button[type='submit']");

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
  el.textContent = JSON.stringify(obj, null, 2);
}

function currentIndexName() {
  return document.getElementById("uploadIndexName").value.trim() || "default-index";
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

async function loadConfig() {
  setStatus(configStatusDot, configStatus, "working", "Loading configuration...");
  try {
    const res = await fetch("/api/config");
    const data = await res.json();
    document.getElementById("azureTier").value = data.azure_tier || "FREE";
    document.getElementById("model").value = data.model || "";
    document.getElementById("chunkSize").value = data.chunk_size || 1000;
    document.getElementById("chunkOverlap").value = data.chunk_overlap || 200;
    document.getElementById("embeddingDimensions").value = data.embedding_dimensions || 1536;
    document.getElementById("inputCost").value = data.pricing.input_per_1k_tokens_usd;
    document.getElementById("outputCost").value = data.pricing.output_per_1k_tokens_usd;
    document.getElementById("semanticCost").value = data.pricing.semantic_query_cost_usd;
    showJson(configResult, data);
    setStatus(configStatusDot, configStatus, "success", "Configuration loaded");
  } catch (err) {
    showJson(configResult, { ok: false, error: err.message || "Config load failed" });
    setStatus(configStatusDot, configStatus, "error", "Failed to load configuration");
  }
}

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

refreshDocsBtn.addEventListener("click", async () => {
  setBusy([refreshDocsBtn, deleteDocsBtn], true);
  await loadDocuments();
  setBusy([refreshDocsBtn, deleteDocsBtn], false);
});

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

chatBtn.addEventListener("click", async () => {
  const query = document.getElementById("chatInput").value.trim();
  if (!query) {
    showJson(chatResult, { ok: false, error: "Question cannot be empty" });
    setStatus(chatStatusDot, chatStatus, "error", "Enter a question");
    return;
  }
  setStatus(chatStatusDot, chatStatus, "working", "Retrieving context and generating answer...");
  setBusy([chatBtn], true);
  const payload = {
    query,
    index_name: document.getElementById("chatIndexName").value || "default-index",
    top_k: Number(document.getElementById("topK").value || 4),
    use_semantic_ranker: document.getElementById("useSemantic").checked,
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
      setStatus(chatStatusDot, chatStatus, "success", `Answer ready (${contexts} contexts)`);
    } else {
      setStatus(
        chatStatusDot,
        chatStatus,
        "error",
        data.error ? `Chat failed: ${data.error}` : "Chat failed"
      );
    }
  } catch (err) {
    showJson(chatResult, { ok: false, error: err.message || "Chat failed" });
    setStatus(chatStatusDot, chatStatus, "error", "Chat failed");
  } finally {
    setBusy([chatBtn], false);
  }
});

configForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  setStatus(configStatusDot, configStatus, "working", "Saving configuration...");
  setBusy([configSaveBtn, resetBtn], true);
  const payload = {
    azure_tier: document.getElementById("azureTier").value,
    model: document.getElementById("model").value,
    chunk_size: Number(document.getElementById("chunkSize").value),
    chunk_overlap: Number(document.getElementById("chunkOverlap").value),
    embedding_dimensions: Number(document.getElementById("embeddingDimensions").value),
    pricing: {
      input_per_1k_tokens_usd: Number(document.getElementById("inputCost").value),
      output_per_1k_tokens_usd: Number(document.getElementById("outputCost").value),
      semantic_query_cost_usd: Number(document.getElementById("semanticCost").value),
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
      setStatus(configStatusDot, configStatus, "success", "Configuration saved");
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

document.getElementById("uploadIndexName").addEventListener("change", loadDocuments);
document.getElementById("uploadIndexName").addEventListener("blur", loadDocuments);

loadConfig();
loadDocuments();
