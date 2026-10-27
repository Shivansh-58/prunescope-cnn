const ELEMENT_COLORS = [
  "#15181b",
  "#e76f51",
  "#2a9d8f",
  "#f2c14e",
  "#4d96ff",
  "#e6e8eb",
  "#8d99ae",
  "#9b5de5",
  "#f15bb5",
  "#00bbf9",
];

function clearCanvas(ctx, width, height) {
  ctx.clearRect(0, 0, width, height);
}

function drawLayout(canvas, layout) {
  if (!layout) return;
  const ctx = canvas.getContext("2d");
  const cell = canvas.width / 32;
  clearCanvas(ctx, canvas.width, canvas.height);
  for (let y = 0; y < 32; y += 1) {
    for (let x = 0; x < 32; x += 1) {
      const value = layout[y][x];
      ctx.fillStyle = ELEMENT_COLORS[value] || ELEMENT_COLORS[0];
      ctx.fillRect(x * cell, y * cell, Math.ceil(cell), Math.ceil(cell));
    }
  }
}

function drawFeatureMaps(canvas, maps, phase = 1) {
  if (!maps || maps.length === 0) return;
  const ctx = canvas.getContext("2d");
  clearCanvas(ctx, canvas.width, canvas.height);
  const columns = 4;
  const rows = 4;
  const gap = 4;
  const tileWidth = (canvas.width - gap * (columns + 1)) / columns;
  const tileHeight = (canvas.height - gap * (rows + 1)) / rows;

  for (let i = 0; i < Math.min(maps.length, columns * rows); i += 1) {
    const map = maps[i];
    const height = map.length;
    const width = map[0].length;
    const ox = gap + (i % columns) * (tileWidth + gap);
    const oy = gap + Math.floor(i / columns) * (tileHeight + gap);
    const sx = tileWidth / width;
    const sy = tileHeight / height;

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const value = Math.max(0, Math.min(1, map[y][x] * phase));
        const r = Math.floor(value * 79);
        const g = Math.floor(value * 195);
        const b = Math.floor(55 + value * 146);
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(ox + x * sx, oy + y * sy, Math.ceil(sx), Math.ceil(sy));
      }
    }
  }
}

function averageLayoutBlocks(layout, count = 16) {
  if (!layout) return Array(count).fill(0);
  const values = [];
  const blocks = Math.sqrt(count);
  const blockSize = 32 / blocks;
  for (let by = 0; by < blocks; by += 1) {
    for (let bx = 0; bx < blocks; bx += 1) {
      let total = 0;
      let pixels = 0;
      for (let y = by * blockSize; y < (by + 1) * blockSize; y += 1) {
        for (let x = bx * blockSize; x < (bx + 1) * blockSize; x += 1) {
          total += layout[y][x] / 9;
          pixels += 1;
        }
      }
      values.push(total / pixels);
    }
  }
  return values;
}

function sampledHits(values, count) {
  if (!values || values.length === 0) return Array(count).fill(0);
  return Array.from({ length: Math.min(count, values.length) }, (_, index) => {
    const sourceIndex = Math.round((index / Math.max(count - 1, 1)) * (values.length - 1));
    return Math.max(0.04, values[sourceIndex]);
  });
}

function channelHits(maps, count = 16) {
  if (!maps) return Array(count).fill(0);
  return maps.slice(0, count).map((map) => {
    let total = 0;
    let peak = 0;
    let pixels = 0;
    for (const row of map) {
      for (const value of row) {
        total += value;
        peak = Math.max(peak, value);
        pixels += 1;
      }
    }
    return Math.min(1, total / Math.max(pixels, 1) * 0.65 + peak * 0.35);
  });
}

function confidenceHits(confidence) {
  return Object.values(confidence || {}).map((value) => Math.max(0.04, value));
}

function layerNodes(width, height, layerSizes) {
  const left = 74;
  const right = width - 72;
  const top = 62;
  const bottom = height - 76;
  return layerSizes.map((size, layerIndex) => {
    const x = left + ((right - left) * layerIndex) / (layerSizes.length - 1);
    return Array.from({ length: size }, (_, nodeIndex) => {
      const y = top + ((bottom - top) * nodeIndex) / Math.max(size - 1, 1);
      return { x, y };
    });
  });
}

function visualStrength(value) {
  const clamped = Math.max(0, Math.min(1, value));
  if (clamped <= 0.02) return 0;
  return Math.pow(clamped, 1.35);
}

function drawSampledEdges(ctx, fromNodes, toNodes, fromHits, toHits, active, progress) {
  const stride = Math.max(1, Math.ceil((fromNodes.length * toNodes.length) / 110));
  let edgeIndex = 0;
  for (let i = 0; i < fromNodes.length; i += 1) {
    for (let j = 0; j < toNodes.length; j += 1) {
      if ((edgeIndex + i + j) % stride !== 0) {
        edgeIndex += 1;
        continue;
      }
      const source = fromNodes[i];
      const target = toNodes[j];
      const strength = visualStrength((fromHits[i] + toHits[j]) / 2);
      ctx.strokeStyle = active ? `rgba(79, 195, 161, ${0.08 + strength * 0.5})` : "rgba(170, 179, 187, 0.16)";
      ctx.lineWidth = active ? 0.8 + strength * 1.8 : 0.8;
      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.stroke();

      if (active && strength > 0.08) {
        const pulse = Math.max(0, Math.min(1, progress - ((i + j) % 5) * 0.045));
        if (pulse > 0 && pulse < 1) {
          ctx.fillStyle = `rgba(242, 193, 78, ${0.2 + strength * 0.8})`;
          ctx.beginPath();
          ctx.arc(source.x + (target.x - source.x) * pulse, source.y + (target.y - source.y) * pulse, 2.5 + strength * 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      edgeIndex += 1;
    }
  }
}

function drawNetworkSchematic(canvas, payload, activeLayer, progress) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const labels = ["Input", "Conv1", "Conv2", "Conv3", "Conv4", "Output"];
  const fullLayerSizes = [16, 32, 64, 128, 256, 5];
  const hits = [
    averageLayoutBlocks(payload.sample.layout, 16),
    sampledHits(payload.activation_hits?.conv1, 24),
    sampledHits(payload.activation_hits?.conv2, 24),
    sampledHits(payload.activation_hits?.conv3, 24),
    sampledHits(payload.activation_hits?.conv4, 24),
    confidenceHits(payload.prediction.confidence),
  ];
  const nodes = layerNodes(width, height, hits.map((layer) => layer.length));
  clearCanvas(ctx, width, height);

  for (let layerIndex = 0; layerIndex < nodes.length - 1; layerIndex += 1) {
    drawSampledEdges(
      ctx,
      nodes[layerIndex],
      nodes[layerIndex + 1],
      hits[layerIndex],
      hits[layerIndex + 1],
      layerIndex === activeLayer,
      progress
    );
  }

  nodes.forEach((layer, layerIndex) => {
    layer.forEach((node, nodeIndex) => {
      const hit = hits[layerIndex][nodeIndex] || 0;
      const strength = visualStrength(hit);
      const isHit = layerIndex <= activeLayer + 1;
      const pulse = layerIndex === activeLayer + 1 ? 0.72 + progress * 0.5 : 1;
      const radius = (layerIndex === 5 ? 13 : 9) + strength * 11 * pulse;
      ctx.shadowBlur = isHit ? strength * 24 : 0;
      ctx.shadowColor = "#4fc3a1";
      ctx.fillStyle = isHit ? `rgba(${Math.floor(24 + strength * 210)}, ${Math.floor(64 + strength * 160)}, ${Math.floor(76 + strength * 92)}, 1)` : "#20252a";
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = layerIndex === 5 && hit === Math.max(...hits[5]) ? "#f2c14e" : "#dbe1e5";
      ctx.lineWidth = layerIndex === activeLayer + 1 ? 2.2 : 1.2;
      ctx.stroke();
    });

    const labelNode = layer[Math.floor(layer.length / 2)];
    ctx.fillStyle = "#f2f4f5";
    ctx.textAlign = "center";
    ctx.font = "15px system-ui";
    ctx.fillText(labels[layerIndex], labelNode.x, height - 35);
    ctx.fillStyle = "#aab3bb";
    ctx.font = "12px system-ui";
    const detail = layerIndex === 5 ? "5 classes" : `${layer.length} shown / ${fullLayerSizes[layerIndex]} channels`;
    ctx.fillText(detail, labelNode.x, height - 16);
  });
}
