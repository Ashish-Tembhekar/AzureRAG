const estimatorStatus = document.getElementById("estimatorStatus");
const estimatorStatusDot = document.getElementById("estimatorStatusDot");
const estimatorForm = document.getElementById("costEstimatorForm");
const estimatorModel = document.getElementById("estimatorModel");
const estimatorRequest = document.getElementById("estimatorRequest");
const estimatorEmptyState = document.getElementById("estimatorEmptyState");
const estimatorResult = document.getElementById("estimatorResult");
const estimateTotalCost = document.getElementById("estimateTotalCost");
const estimateSecondaryTotal = document.getElementById("estimateSecondaryTotal");
const estimateSummaryMeta = document.getElementById("estimateSummaryMeta");
const estimateOverrideBadge = document.getElementById("estimateOverrideBadge");
const estimateServices = document.getElementById("estimateServices");
const estimateBreakdownBody = document.getElementById("estimateBreakdownBody");
const estimateRequestMeta = document.getElementById("estimateRequestMeta");

const USD_TO_INR = 90.955;

function setEstimatorStatus(state, message) {
  if (!estimatorStatus || !estimatorStatusDot) {
    return;
  }
  estimatorStatusDot.classList.remove("idle", "working", "success", "error");
  estimatorStatusDot.classList.add(state);
  estimatorStatus.textContent = message;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function usdToInr(value) {
  return Number(value || 0) * USD_TO_INR;
}

function formatInr(value) {
  return `Rs ${Number(value || 0).toFixed(6)}`;
}

function getDisplayCostInr(item) {
  if (item && item.currency === "INR") {
    return Number(item.cost_inr || 0);
  }
  return usdToInr(item?.cost_usd || 0);
}

function getDisplayRateInr(item) {
  if (item && item.currency === "INR") {
    return Number(item.rate || 0);
  }
  return usdToInr(item?.rate || 0);
}

function formatInrRate(value, label) {
  if (label.includes("Azure AI Search compute")) {
    return `${formatInr(value)} / sec`;
  }
  if (label.includes("characters")) {
    return `${formatInr(value)} / 1M chars`;
  }
  if (label.includes("seconds")) {
    return `${formatInr(value)} / hour`;
  }
  return `${formatInr(value)} / 1K`;
}

function populateModelOptions(models, defaultModel) {
  estimatorModel.innerHTML = "";
  models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = `${model.label} (${formatInr(usdToInr(model.input_per_1k_tokens_usd))} in / ${formatInr(usdToInr(model.output_per_1k_tokens_usd))} out)`;
    if (model.id === defaultModel) {
      option.selected = true;
    }
    estimatorModel.appendChild(option);
  });

  if (!estimatorModel.value && models.length > 0) {
    estimatorModel.value = models[0].id;
  }
}

function populateRequestOptions(requests) {
  estimatorRequest.innerHTML = "";
  requests.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.file_name;
    option.textContent = item.label || item.file_name;
    estimatorRequest.appendChild(option);
  });
}

function renderServices(services) {
  estimateServices.innerHTML = services.map((service) => {
    const details = service.details || {};
    const displayCost = service.cost_inr !== undefined
      ? Number(service.cost_inr || 0)
      : usdToInr(service.cost_usd || 0);
    let detailHtml = "";

    if (service.key === "llm") {
      detailHtml = `
        <p>Model: <strong>${escapeHtml(details.priced_model || "-")}</strong></p>
        <p>Input tokens: ${escapeHtml(details.input_tokens)}</p>
        <p>Output tokens: ${escapeHtml(details.output_tokens)}</p>
        <p>Input cost: ${formatInr(usdToInr(details.input_cost_usd || 0))}</p>
        <p>Output cost: ${formatInr(usdToInr(details.output_cost_usd || 0))}</p>
      `;
    } else if (service.key === "azure_search") {
      detailHtml = `
        <p>Basic tier estimate: ${details.basic_tier_assumed ? "Yes" : "No"}</p>
        <p>Pricing basis: ${escapeHtml(details.search_units)} SU @ ${formatInr(details.basic_price_per_su_month_inr)} / month</p>
        <p>Request duration: ${escapeHtml(details.request_duration_seconds)} sec</p>
        <p>Semantic cost: ${formatInr(usdToInr(details.semantic_cost_usd || 0))}</p>
        <p>Search compute: ${formatInr(details.search_compute_cost_inr)}</p>
        <p>Retrieved text length: ${escapeHtml(details.retrieved_text_length)} chars</p>
        <p>Results returned: ${escapeHtml(details.results_returned)}</p>
      `;
    } else if (service.key === "speech") {
      detailHtml = `
        <p>Pay-as-you-go estimate: ${details.payg_pricing_assumed ? "Yes" : "No"}</p>
        <p>STT seconds: ${escapeHtml(details.stt_seconds)}</p>
        <p>TTS characters: ${escapeHtml(details.tts_characters)}</p>
        <p>STT cost: ${formatInr(details.stt_cost_inr)}</p>
        <p>TTS cost: ${formatInr(details.tts_cost_inr)}</p>
      `;
    } else {
      detailHtml = `
        <p>Request units: ${escapeHtml(details.request_units)}</p>
        <p>Bytes written: ${escapeHtml(details.bytes_written)}</p>
        <p>${escapeHtml(details.pricing_note || "")}</p>
      `;
    }

    return `
      <article class="estimator-service-card">
        <p class="overline">${escapeHtml(service.label)}</p>
        <h3>${formatInr(displayCost)}</h3>
        <div class="estimator-service-copy">${detailHtml}</div>
      </article>
    `;
  }).join("");
}

function renderBreakdown(breakdown) {
  estimateBreakdownBody.innerHTML = breakdown.map((item) => `
    <tr>
      <td>${escapeHtml(item.label)}</td>
      <td>${escapeHtml(item.units)}</td>
      <td>${escapeHtml(formatInrRate(getDisplayRateInr(item), item.label))}</td>
      <td>${escapeHtml(formatInr(getDisplayCostInr(item)))}</td>
    </tr>
  `).join("");
}

function renderEstimate(estimate) {
  estimatorEmptyState.classList.add("hidden");
  estimatorResult.classList.remove("hidden");

  estimateTotalCost.textContent = formatInr(
    usdToInr(estimate.total_cost_usd || 0) + Number(estimate.total_cost_inr || 0)
  );
  if (estimateSecondaryTotal) {
    estimateSecondaryTotal.textContent = `Converted using 1 USD = ${USD_TO_INR} INR`;
  }
  estimateSummaryMeta.textContent = `Recorded model: ${estimate.recorded_model || "Unknown"} | Selected pricing: ${estimate.selected_model}`;
  estimateRequestMeta.textContent = `${estimate.user_query || "Untitled request"} | ${estimate.metrics_file || ""}`;
  estimateOverrideBadge.classList.toggle("hidden", !estimate.pricing_override_applied);

  renderServices(estimate.services || []);
  renderBreakdown(estimate.breakdown || []);
}

async function fetchEstimatorData() {
  setEstimatorStatus("working", "Loading estimator data...");
  try {
    const res = await fetch("/api/cost-estimator/data");
    const data = await res.json();
    if (!data.ok) {
      throw new Error(data.error || "Failed to load estimator data");
    }

    populateModelOptions(data.models || [], data.default_model);
    populateRequestOptions(data.requests || []);
    setEstimatorStatus("success", "Estimator ready");

    if (estimatorRequest.value && estimatorModel.value) {
      await requestEstimate();
    }
  } catch (error) {
    setEstimatorStatus("error", error.message || "Failed to load estimator data");
  }
}

async function requestEstimate() {
  if (!estimatorRequest.value || !estimatorModel.value) {
    return;
  }

  setEstimatorStatus("working", "Estimating request cost...");
  try {
    const res = await fetch("/api/cost-estimator/estimate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        metrics_file: estimatorRequest.value,
        model: estimatorModel.value
      })
    });
    const data = await res.json();
    if (!data.ok) {
      throw new Error(data.error || "Unable to estimate cost");
    }

    renderEstimate(data.estimate);
    setEstimatorStatus("success", "Estimate updated");
  } catch (error) {
    setEstimatorStatus("error", error.message || "Unable to estimate cost");
  }
}

if (estimatorForm) {
  estimatorForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await requestEstimate();
  });

  estimatorModel.addEventListener("change", requestEstimate);
  estimatorRequest.addEventListener("change", requestEstimate);
  fetchEstimatorData();
}
