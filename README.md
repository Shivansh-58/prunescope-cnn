# PruneScope CNN

**Real-time CNN activation visualization, built as a foundation for pruning experiments.**

A local educational prototype that runs real PyTorch inference on synthetic webpage layouts and streams activations to a Canvas interface through FastAPI and WebSockets.

> **Status:** visualization and training work. Pruning, Grad-CAM, baseline/pruned comparisons, and exact neuron connectivity are future work, not implemented features. No pretrained weights or generated datasets are included.

## Contents

- [Features](#features)
- [Setup and execution](#setup-and-execution)
- [Using the application](#using-the-application)
- [CNN architecture](#cnn-architecture)
- [Dataset](#dataset)
- [Visualization accuracy](#visualization-accuracy)
- [Project structure](#project-structure)
- [API](#api)
- [Limitations and troubleshooting](#limitations-and-troubleshooting)
- [Pruning roadmap](#pruning-roadmap)

## Features

- Four panels: input layout, CNN visualization, predictions, and training.
- Real forward-hook activation capture from Conv1 through Conv4.
- Live feature-map canvases and an animated, simplified network schematic; no static PNG generation.
- Previous, Next, Random, Play, Pause, Step, Replay, and 0.2x-4x animation speed.
- Class scores, inference time, backend update FPS, loss, epochs, and validation accuracy.
- Local training and checkpoint save/load.
- Automatic CUDA selection when supported by hardware and the installed PyTorch build; CPU fallback.

## Setup and execution

### Requirements

- Python 3.11+ with a compatible PyTorch build. Python 3.12 is a practical starting point for this prototype.
- Git and a modern browser.
- Optional NVIDIA GPU with a compatible driver and CUDA-enabled PyTorch.
- No Node.js, database, frontend build step, or external dataset is needed.

### Windows PowerShell

```powershell
git clone https://github.com/Shivansh-58/prunescope-cnn.git
cd prunescope-cnn
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Replace `py -3.12` with your supported Python interpreter if needed. Calling the virtual-environment Python directly avoids PowerShell activation-policy problems.

### Linux / macOS

```bash
git clone https://github.com/Shivansh-58/prunescope-cnn.git
cd prunescope-cnn
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Keep the terminal open; press `Ctrl+C` to stop.

For development, add `--reload` to the server command. Reloading discards unsaved in-memory model weights.

### IDE setup

1. Open the repository folder in VS Code or PyCharm.
2. Select `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on Linux/macOS as the interpreter.
3. Open the IDE terminal at the repository root and run the server command above.
4. Alternatively, configure a Python module run target: module `uvicorn`, arguments `backend.app:app --host 127.0.0.1 --port 8000`, working directory the repository root.
5. Open the localhost URL in your browser. FastAPI serves the frontend.

## Using the application

1. Startup generates 6,000 layouts in memory and initializes the model with **random weights**. Predictions are not meaningful until trained or loaded.
2. Use **Prev**, **Next**, or **Random** to inspect a sample. **Step** advances one sample, not one individual neuron.
3. **Play** automatically advances samples. **Pause** stops sample changes, but does not freeze the frontend pulse animation.
4. **Replay** reruns inference for the selected sample and restarts its animation. Pause sample playback first to inspect a single sample.
5. **Speed** adjusts illustrative animation timing, not inference execution speed or the backend sample-update interval.
6. Training **Start** runs up to 8 epochs using Adam, learning rate 0.001, batch size 96, and cross-entropy loss. **Stop** requests termination at training-loop checks.
7. **Save** writes `models/layout_cnn.pt`; **Load** restores it if present. After a server restart, loading is manual.

Pause playback while training on slower hardware. Save important weights before stopping or reloading the server.

## CNN architecture

Shapes omit the batch dimension and use `channels x height x width`.

| Stage | Output shape | Operation |
| --- | --- | --- |
| Input | 1 x 32 x 32 | Element IDs divided by 9 |
| Conv1 | 32 x 32 x 32 | 3 x 3 convolution, padding 1 |
| ReLU + MaxPool | 32 x 16 x 16 | 2 x 2 pooling |
| Conv2 | 64 x 16 x 16 | 3 x 3 convolution, padding 1 |
| ReLU + MaxPool | 64 x 8 x 8 | 2 x 2 pooling |
| Conv3 | 128 x 8 x 8 | 3 x 3 convolution, padding 1 |
| ReLU + MaxPool | 128 x 4 x 4 | 2 x 2 pooling |
| Conv4 | 256 x 4 x 4 | 3 x 3 convolution, padding 1 |
| ReLU + MaxPool | 256 x 2 x 2 | 2 x 2 pooling |
| Flatten | 1024 | Flatten spatial dimensions |
| FC1 + ReLU | 256 | Fully connected layer |
| FC2 | 5 | Logits; softmax for display |

Hooks capture convolution outputs **before ReLU and pooling**. Displayed map resolutions are therefore 32 x 32, 16 x 16, 8 x 8, and 4 x 4.

## Dataset

The generator draws rectangular semantic regions on single-channel 32 x 32 grids. These are **not screenshots, HTML documents, or scraped websites**.

| Class ID | Class | Template |
| --- | --- | --- |
| 0 | Blog | Header, image, text rows, footer |
| 1 | Dashboard | Header, sidebar, four cards, button |
| 2 | Login | Header, central form, text fields, button, footer |
| 3 | Ecommerce | Header, navbar, six cards with images/buttons |
| 4 | Landing | Header, large image, text, call-to-action, footer |

| Element ID | Meaning | Element ID | Meaning |
| --- | --- | --- | --- |
| 0 | Background | 5 | Text |
| 1 | Header | 6 | Footer |
| 2 | Navbar | 7 | Sidebar |
| 3 | Button | 8 | Form |
| 4 | Image | 9 | Card |

- Default: **6,000 samples**, 1,200 per class, generator seed 7.
- Small template variations and random element replacement at approximately 1.8% of cells add noise.
- Inputs encode each element as `element_id / 9.0`, not one-hot channels.
- An 82% / 18% split gives 4,920 training and 1,080 validation samples, using split seed 42. The split is not explicitly stratified.
- Samples are generated in memory at startup. No dataset download is required. The generator source is included; dataset artifacts are excluded from Git.

High validation accuracy on these similar templates does not establish generalization to real websites.

## Visualization accuracy

**Inference and captured tensors are real; the large network diagram is an abstraction, not literal neuron wiring.**

| Display | Actual representation |
| --- | --- |
| Input canvas | All 1,024 layout cells |
| Feature-map canvases | First 16 channels per convolution, min-max normalized together per layer and rounded to four decimals |
| Input diagram nodes | 16 averages of 8 x 8 layout blocks, not individual pixels |
| Convolution diagram nodes | 24 sampled channel-summary scores per layer, not all spatial neurons |
| Output nodes | Five softmax class scores |
| Links and traveling pulses | Sampled illustrative links and animation, not actual weights or measured neuron firing times |

The current channel-summary heuristic combines the fraction of values above each channel's 80th-percentile threshold (70%) with a layer-scaled channel peak (30%), then normalizes across channels. Frontend contrast scaling and a small brightness floor also affect appearance. These scores are **not pruning importance metrics**.

Pooling, flattening, and FC256 are omitted from the schematic, but executed by the actual model. Inference finishes before its results are animated. During Play, samples can change before an animation cycle completes.

## Project structure

```text
prunescope-cnn/
|-- backend/
|   |-- __init__.py
|   |-- app.py              # FastAPI routes, WebSockets, shared state
|   |-- model.py            # CNN architecture
|   |-- dataset.py          # Synthetic generation and dataset wrapper
|   |-- train.py            # Training, evaluation, checkpoint save/load
|   |-- inference.py        # Forward pass, predictions, timing, payload
|   `-- hooks.py            # Activation capture and serialization
|-- frontend/
|   |-- index.html          # Four-panel UI
|   |-- style.css           # Styling
|   |-- app.js              # WebSocket client, controls, animation loop
|   `-- visualization.js    # Canvas rendering
|-- data/                   # Local datasets; only .gitkeep tracked
|-- models/                 # Local checkpoints; only .gitkeep tracked
|-- outputs/                # Local outputs/logs; only .gitkeep tracked
|-- .gitignore
|-- README.md
`-- requirements.txt
```

Generated datasets, weights, outputs, virtual environments, caches, credentials, and IDE settings are excluded from version control. Existing local artifacts are not deleted.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Browser interface |
| `GET /api/meta` | Classes, elements, sample count, device |
| `GET /api/sample/{index}` | Raw sample; index wraps around the dataset |
| `WS /ws` | Live messages and commands |
| `GET /docs` | FastAPI HTTP API documentation |

Example WebSocket command:

```json
{"command": "random"}
```

Commands: `prev`, `next`, `random`, `step`, `play`, `pause`, `replay`, `start_training`, `stop_training`, `save_model`, `load_model`.

Server message types: `meta`, `inference`, `training`, `model`. Inference messages contain the sample, predictions, feature maps, channel summaries, inference time, and FPS. Connected tabs share model and playback state.

## Limitations and troubleshooting

- **Local use only:** no authentication, permissive CORS. Keep the server bound to `127.0.0.1`; do not expose it to the internet.
- **Hardware-dependent performance:** backend playback waits approximately 1/24 second between updates in addition to computation/streaming time. Canvas uses `requestAnimationFrame`; 20-60 FPS is a goal, not a guarantee.
- **Training may stutter:** training/inference share the model and event loop. Batch computation and synchronous validation can delay interaction.
- **Approximate metrics:** FPS measures intervals between backend payload builds, not browser rendering FPS. Timing includes hooks and is not a rigorous CUDA benchmark.
- **CPU instead of GPU:** check `python -c "import torch; print(torch.cuda.is_available())"` using your virtual-environment interpreter. Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for an appropriate GPU build.
- **No compatible torch package:** use a supported Python/PyTorch combination and recreate the virtual environment if needed.
- **Port already in use:** stop the existing server or change to `--port 8001` and open that port.
- **No model found:** train and click Save first. Fresh clones intentionally contain no checkpoint.
- **Checkpoint safety:** load only trusted local checkpoints. Restarts discard unsaved weights.
- Dependencies have minimum versions rather than a locked environment; a formal test suite is not yet included.

## Pruning roadmap

Planned directions, **not current functionality**:

- Collect raw activation statistics across many samples.
- Add explicit channel selection, masking, and structured pruning.
- Compare original/pruned accuracy, parameter counts, latency, and activations.
- Support fine-tuning and repeatable experiment metadata.
- Inspect actual receptive fields, signed activations, and kernel connections.
- Add Grad-CAM and more varied or real layout data.

Model, hooks, inference, training, and rendering are separated to support these extensions. Display brightness alone must not be used to decide which channels to prune.
