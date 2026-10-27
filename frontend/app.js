const state = {
  ws: null,
  meta: null,
  latest: null,
  animationStart: 0,
  activeLayer: 0,
  speed: 1,
  resetOnNextInference: true,
  layerNames: ["conv1", "conv2", "conv3", "conv4"],
};

const layoutCanvas = document.getElementById("layoutCanvas");
const networkCanvas = document.getElementById("networkCanvas");
const speedControl = document.getElementById("speedControl");
const speedValue = document.getElementById("speedValue");
const convCanvases = {
  conv1: document.getElementById("conv1Canvas"),
  conv2: document.getElementById("conv2Canvas"),
  conv3: document.getElementById("conv3Canvas"),
  conv4: document.getElementById("conv4Canvas"),
};

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  state.ws = new WebSocket(`${protocol}://${location.host}/ws`);
  state.ws.onmessage = (event) => handleMessage(JSON.parse(event.data));
  state.ws.onclose = () => setTimeout(connect, 800);
}

function handleMessage(message) {
  if (message.type === "meta") {
    state.meta = message;
    renderMeta();
    return;
  }

  if (message.type === "inference") {
    state.latest = message;
    if (state.resetOnNextInference) {
      state.animationStart = performance.now();
      state.activeLayer = 0;
      state.resetOnNextInference = false;
    }
    renderInference(message);
    return;
  }

  if (message.type === "training") {
    renderTraining(message);
    return;
  }

  if (message.type === "model") {
    document.getElementById("modelStatus").textContent = `Model: ${message.status}`;
  }
}

function send(command) {
  if (["prev", "next", "random", "step", "replay"].includes(command)) {
    state.animationStart = performance.now();
    state.activeLayer = 0;
    state.resetOnNextInference = true;
  }
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ command }));
  }
}

function renderMeta() {
  document.getElementById("deviceLabel").textContent = `${state.meta.sample_count} synthetic layouts on ${state.meta.device}`;
  const legend = document.getElementById("elementLegend");
  legend.innerHTML = "";
  Object.entries(state.meta.elements).forEach(([id, name]) => {
    const row = document.createElement("span");
    row.innerHTML = `<i class="swatch" style="background:${ELEMENT_COLORS[id]}"></i>${id} ${name}`;
    legend.appendChild(row);
  });
}

function renderInference(message) {
  drawLayout(layoutCanvas, message.sample.layout);
  document.getElementById("sampleLabel").textContent = `Sample ${message.sample.index}`;
  document.getElementById("trueClass").textContent = `True class: ${message.sample.label_name}`;
  document.getElementById("predictedClass").textContent = message.prediction.class_name;
  document.getElementById("inferenceTime").textContent = `${message.inference_ms} ms`;
  document.getElementById("fps").textContent = message.fps;
  renderBars(message.prediction.confidence);
}

function renderBars(confidence) {
  const root = document.getElementById("confidenceBars");
  root.innerHTML = "";
  Object.entries(confidence).forEach(([name, value]) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span>${name}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${value * 100}%"></div></div>
      <span>${Math.round(value * 100)}%</span>
    `;
    root.appendChild(row);
  });
}

function renderTraining(message) {
  document.getElementById("trainingState").textContent = message.training ? "Running" : "Idle";
  if (message.epoch !== undefined) document.getElementById("epoch").textContent = message.epoch;
  if (message.loss !== undefined) document.getElementById("loss").textContent = message.loss ?? "-";
  if (message.val_accuracy !== undefined) {
    document.getElementById("valAccuracy").textContent =
      message.val_accuracy === null ? "checking" : `${Math.round(message.val_accuracy * 100)}%`;
  }
}

function animate(now) {
  const duration = 3000 / state.speed;
  const elapsed = (now - state.animationStart) % duration;
  const schematicLayerCount = 5;
  const layerSpan = duration / schematicLayerCount;
  state.activeLayer = Math.min(schematicLayerCount - 1, Math.floor(elapsed / layerSpan));
  const progress = Math.min(1, (elapsed % layerSpan) / layerSpan);

  if (state.latest) {
    drawNetworkSchematic(networkCanvas, state.latest, state.activeLayer, progress);
    state.layerNames.forEach((name, index) => {
      const layerPhase = index < state.activeLayer ? 1 : index === state.activeLayer ? progress : 0.18;
      drawFeatureMaps(convCanvases[name], state.latest.activations[name], layerPhase);
    });
    const readableLayer = ["Input -> Conv1", "Conv1 -> Conv2", "Conv2 -> Conv3", "Conv3 -> Conv4", "Conv4 -> Output"][state.activeLayer] || "Conv4 -> Output";
    document.getElementById("animationStatus").textContent = `Activation flow: ${readableLayer}`;
  }
  requestAnimationFrame(animate);
}

document.querySelectorAll("[data-command]").forEach((button) => {
  button.addEventListener("click", () => send(button.dataset.command));
});

speedControl.addEventListener("input", () => {
  state.speed = Number(speedControl.value);
  speedValue.textContent = `${state.speed.toFixed(1)}x`;
});

connect();
requestAnimationFrame(animate);
