# Kitless / Archie

**Kitless** is an experimental distributed AI network, and **Archie** is the AI that runs on it.

**Current Release:** **Version 0.1b** *(First Public Prototype)*

---

# The Vision

As a gamer, I've watched GPUs become increasingly difficult to buy and more expensive, and we've also seen RAM shortages and rising memory prices. AI is advancing at an incredible rate, but that progress comes with an ever-increasing demand for expensive hardware and enormous data centres owned by only a handful of companies.

My idea with Kitless is simple.

Instead of building larger and larger AI networks that require massive server farms, why not make use of the computing power that people already own?

Kitless works in a similar way to BitTorrent and distributed computing. Every person who joins the network contributes a small amount of computing power. When you chat with Archie or contribute training, your computer becomes part of the network. One computer alone cannot compete with a dedicated AI cluster, but thousands—or even millions—of computers working together can.

The more people who join, the more computing power the network has available, allowing Archie to continue learning and improving over time.

The long-term goal is to create a worldwide AI that belongs to everyone—built by the community, powered by the community, and improved by the community—instead of relying entirely on massive centralised AI infrastructure.

This is still the very first public prototype, so expect bugs, rough edges, and missing features. I will continue improving Kitless over time, and if you'd like to contribute code, report bugs, suggest ideas, or help develop the project, your support is always welcome.

> **Built by the people. Powered by the people. Improved by the people.**

---

# Running Kitless

Launch:

`START_CLIENT.bat`

When the client starts, it automatically synchronises the latest Archie `model.pt`.
The client launcher starts the GUI without leaving a background CMD window open.
By default, public clients connect to `https://kitless.co.uk`.
HTTPS is handled by nginx. Do not expose the local server port `8799` publicly.

The client allows you to:

* Automatically synchronise the latest Archie model.
* Chat with Archie.
* Use local chat from the synchronised `model.pt`, or enable **SWARM CHAT** so online clients share small inference shards and the server only co-ordinates the reply.
* Contribute local AI training.
* Continuously retrain when enabled.
* Display the number of online clients.
* Display live training graphs in the GUI.

---

# Typical Workflow

1. Start `START_CLIENT.bat`.
2. The latest Archie `model.pt` is synchronised automatically.
3. Chat with Archie or click **START TRAINING** to help improve the model.
4. Click **STOP TRAINING** when you want to pause contribution and talk to Archie.
5. Leave **KEEP RETRAINING** enabled if you want your client to continue training new tasks automatically.

---

# How Kitless Works

Kitless uses distributed AI training.

* Every client automatically synchronises the latest Archie `model.pt`.
* Chat can run locally on the client or through SWARM CHAT, where online clients share small pieces of inference work.
* Training is performed locally on each user's computer.
* Model improvements are contributed back to the network.
* As more people join, Archie gains access to more distributed computing power.
* Updated Archie models are synchronised across all clients as new versions become available.

Every person who joins the network helps Archie become more capable. The aim is to demonstrate that AI does not have to rely entirely on enormous server farms—it can be powered by people all over the world.

---

# Server Access

Public clients should connect through:

`https://kitless.co.uk`

The live server dashboard is available at:

`https://kitless.co.uk/dashboard`

nginx handles HTTPS and proxies requests to the private local server. The local server port `8799` must not be exposed directly to the public internet.

---

# Adaptive Training

Every computer is different, so Kitless automatically adjusts the amount of work each client performs.

Each client begins with a **1.0× training budget**.

During training, Kitless measures how long each task takes to complete and automatically adjusts future workloads.

* Starts with a **1.0×** training budget.
* If training is slower than expected, the client automatically reduces the number of passes and epochs.
* If training is faster than expected, the client automatically increases the number of passes and epochs.
* **Minimum budget:** `0.25×`
* **Maximum budget:** `2.5×`
* **Target task time:** Approximately **18 seconds**.
* The GUI draws a live local loss graph after each completed training task.
* Archie chat stays visible as the main area.
* The `View` menu can show or hide the training graph and terminal log below the chat area.

This means:

* Older or lower-powered systems automatically receive lighter workloads.
* Faster gaming PCs and more powerful GPUs automatically receive larger workloads.
* Every client contributes according to its available performance without requiring manual configuration.

Kitless also records training statistics for every completed task, including:

* Local training time (seconds)
* Training budget multiplier
* Total local training steps
* Loss before and loss after training

These statistics help balance workloads fairly across the distributed network.

The built-in GUI graph is for the current user's client only. It is not the whole server graph.

The graph shows:

* **Green:** loss before local training.
* **Blue:** loss after local training. Blue is the result you want to improve.
* Lower is better. If the blue value is below the green value, the client improved the model on that task.
* Last task time, training budget, and training step count.

---

# Swarm Chat Inference

**SWARM CHAT** is for low-end machines that should not have to answer from `model.pt` alone.

When SWARM CHAT is enabled:

* The asking client sends the message to the Kitless server.
* The server coordinates the chat job without doing the heavy model reading.
* The server creates a short chat job and splits the model's answer space into small output shards.
* Online clients that already synchronised their own local `model.pt` quietly pick up one shard at a time.
* Each helper client reads only its assigned output-weight rows from its own `model.pt`, scores that shard, and sends back its best answer score.
* The server combines those client results and returns the best swarm answer.

The server is the coordinator, not the main thinker. If no helper client answers in time, Archie reports that no swarm helper answered instead of forcing a weak server to read the whole model.

---

# Technology Stack

Kitless is built using modern open-source AI and machine learning technologies.

In this prototype, the core live training path is the PyTorch Archie checkpoint. The wider stack is wired into supporting features, generated assets, monitoring, and future model upgrades.

## Core

### `torch`

The core deep learning framework used to build, train, and run Archie.

### `torch-directml`

Provides GPU acceleration on Windows using AMD, Intel, and supported DirectX 12 graphics hardware, allowing more people to contribute without requiring NVIDIA CUDA.

### `numpy`

Provides fast mathematical operations, numerical computing, and array processing throughout the project.

---

## AI Training

### `transformers`

Included for future transformer-based Archie model upgrades and compatibility with Hugging Face training tools.

### `tokenizers`

Builds `server/server_data/tokenizer.json` from the Kitless training database.

### `sentencepiece`

Included for future sub-word tokenizer support alongside the current BPE tokenizer asset.

### `datasets`

Exports the training database to `server/server_data/hf_dataset` for Hugging Face dataset tooling.

### `accelerate`

Included for future larger CPU/GPU training pipelines. The current GUI already supports CPU, GPU, and BOTH modes directly.

### `peft`

Included for future LoRA/adapter-based training upgrades.

### `safetensors`

Writes `model.safetensors` beside `model.pt` when the dependency is installed.

---

## Evaluation & Monitoring

### `evaluate`

Records loss improvement metrics for accepted training updates.

### `tensorboard`

Stores optional advanced training logs for external graphing. Kitless also includes a built-in GUI graph so users can see training loss without opening TensorBoard.

### `scikit-learn`

Available for future analysis and evaluation tools.

### `tqdm`

Used by dataset tooling and available for longer preprocessing/build jobs.

---

## Distributed Computing

### `deepspeed`

Included for non-Windows future distributed training work. It is skipped on Windows.

### `ray`

Installed for future larger job coordination. Current swarm coordination is handled by the Kitless server.

---

## Memory & Retrieval

### `faiss-cpu`

Builds `server/server_data/memory.faiss` and `memory_meta.json` as a vector-memory index over the training database.

---

## Inference & Export

### `onnxruntime`

Supports exported inference artifacts. Kitless can export the current Archie checkpoint to `server/server_data/model.onnx`.

---

# Generated Training Assets

The developer client includes **BUILD ASSETS** to generate:

* `training_stack.json`
* `tokenizer.json`
* `hf_dataset/`
* `memory.faiss`
* `memory_meta.json`
* `model.safetensors`
* `model.onnx`

The public and developer clients also show a **Training Stack / Assets** status row so users can see which features are ready.

---

# Storage

The client stores its data in:

* `client/client_data/`
* `client/client_checkpoint/`

---

# Version

**Version 0.1b** is the first public prototype of Kitless.

This release demonstrates the core concept of a community-powered distributed AI. New features, performance improvements, bug fixes, and optimisations will continue to be added as the project evolves.

Whether you are chatting with Archie or contributing training, every client strengthens the network and helps shape the future of Kitless.

**Built by the people. Powered by the people. Improved by the people.**

