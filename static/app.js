const uploadForm = document.getElementById("uploadForm");
const uploadResult = document.getElementById("uploadResult");
const chatBtn = document.getElementById("chatBtn");
const chatResult = document.getElementById("chatResult");
const configForm = document.getElementById("configForm");
const configResult = document.getElementById("configResult");
const resetBtn = document.getElementById("resetBtn");

function showJson(el, obj) {
  el.textContent = JSON.stringify(obj, null, 2);
}

async function loadConfig() {
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
}

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = document.getElementById("uploadFile").files[0];
  if (!file) {
    showJson(uploadResult, { ok: false, error: "Please pick a file" });
    return;
  }

  const fd = new FormData();
  fd.append("file", file);
  fd.append("index_name", document.getElementById("uploadIndexName").value || "default-index");

  const res = await fetch("/api/upload", { method: "POST", body: fd });
  const data = await res.json();
  showJson(uploadResult, data);
});

chatBtn.addEventListener("click", async () => {
  const query = document.getElementById("chatInput").value.trim();
  if (!query) {
    showJson(chatResult, { ok: false, error: "Question cannot be empty" });
    return;
  }
  const payload = {
    query,
    index_name: document.getElementById("chatIndexName").value || "default-index",
    top_k: Number(document.getElementById("topK").value || 4),
    use_semantic_ranker: document.getElementById("useSemantic").checked,
  };
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  showJson(chatResult, data);
});

configForm.addEventListener("submit", async (e) => {
  e.preventDefault();
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
  const res = await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  showJson(configResult, data);
});

resetBtn.addEventListener("click", async () => {
  const res = await fetch("/api/reset", { method: "POST" });
  const data = await res.json();
  showJson(configResult, data);
});

loadConfig();
