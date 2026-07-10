from __future__ import annotations

import base64
from io import BytesIO
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import sys
import threading
import tkinter as tk
import urllib.error
import urllib.request
import uuid
from typing import Any
from tkinter import messagebox, scrolledtext, ttk

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_directml  # type: ignore[import-not-found]
except ImportError:
    torch_directml = None

APP_NAME = "Kitless"
APP_DIR = Path(__file__).resolve().parent
CLIENT_DATA_DIR = APP_DIR / "client_data"
CLIENT_CHECKPOINT_DIR = APP_DIR / "client_checkpoint"
CLIENT_ID_FILE = CLIENT_DATA_DIR / "client_id.txt"
API_KEY_FILE = CLIENT_DATA_DIR / "api_key.txt"
SERVER_URL_FILE = CLIENT_DATA_DIR / "server_url.txt"
LATEST_MODEL_FILE = CLIENT_CHECKPOINT_DIR / "model.pt"
DEFAULT_SERVER_URL = "https://kitless.co.uk"
DEFAULT_API_KEY_PARTS = (
    "917530a43cfc3dcf",
    "3a910c64e5a93b30",
    "9c59a9c4da03076b",
    "2fe918ecf15ed619",
)
MODEL_KIND = "kitless-linear-archie-v1"


def default_api_key() -> str:
    return "".join(DEFAULT_API_KEY_PARTS)


def valid_api_key(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", value.strip()))


def parent_process_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    if sys.platform == "win32":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def exit_when_parent_closes() -> None:
    parent_pid = os.getppid()

    def watch() -> None:
        while True:
            threading.Event().wait(1)
            if not parent_process_alive(parent_pid):
                os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def ensure_dirs() -> None:
    CLIENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CLIENT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def get_or_create_client_id(prefix: str = "client") -> str:
    ensure_dirs()
    if CLIENT_ID_FILE.exists():
        return CLIENT_ID_FILE.read_text(encoding="utf-8").strip()
    value = f"{prefix}-{uuid.uuid4().hex[:12]}"
    CLIENT_ID_FILE.write_text(value, encoding="utf-8")
    return value


def new_session_id(prefix: str = "client") -> str:
    return f"{prefix}-session-{uuid.uuid4().hex[:12]}"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def local_api_key() -> str:
    direct = os.environ.get("KITLESS_API_KEY", "").strip()
    if valid_api_key(direct):
        return direct
    if API_KEY_FILE.exists():
        saved = API_KEY_FILE.read_text(encoding="utf-8").strip()
        if valid_api_key(saved):
            return saved
    return default_api_key()


def save_api_key(value: str) -> None:
    API_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    value = value.strip()
    if valid_api_key(value):
        API_KEY_FILE.write_text(value, encoding="utf-8")
    elif value:
        API_KEY_FILE.write_text(default_api_key(), encoding="utf-8")
    elif API_KEY_FILE.exists():
        API_KEY_FILE.unlink()


def local_server_url() -> str:
    if SERVER_URL_FILE.exists():
        saved = SERVER_URL_FILE.read_text(encoding="utf-8").strip()
        if saved:
            return saved
    return DEFAULT_SERVER_URL


def save_server_url(value: str) -> None:
    SERVER_URL_FILE.parent.mkdir(parents=True, exist_ok=True)
    value = value.strip().rstrip("/")
    SERVER_URL_FILE.write_text(value or DEFAULT_SERVER_URL, encoding="utf-8")


def http_json(method: str, url: str, payload: dict | None = None, timeout: int = 10) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "Kitless-Client/1.0")
    key = local_api_key()
    if key:
        req.add_header("X-Kitless-Key", key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read().decode("utf-8"))
        except Exception:
            data = {"error": str(e)}
        raise RuntimeError(data.get("error", str(e))) from e


def word_list(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z']+", text)]


def make_model(vocab_size: int, label_count: int) -> nn.Linear:
    return nn.Linear(max(1, vocab_size), max(1, label_count))


def vectorise(prompt: str, vocab: dict[str, int]) -> torch.Tensor:
    x = torch.zeros(1, max(1, len(vocab)), dtype=torch.float32)
    for word in word_list(prompt):
        if word in vocab:
            x[0, vocab[word]] += 1.0
    if x.sum() > 0:
        x = x / x.sum()
    return x


def gpu_training_device() -> tuple[str, torch.device] | None:
    if torch.cuda.is_available():
        return "GPU-CUDA", torch.device("cuda")
    if torch_directml is not None:
        try:
            return "GPU-DIRECTML", torch_directml.device()
        except Exception:
            return None
    return None


def gpu_status_text() -> str:
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    if torch_directml is not None:
        return "GPU: DirectML available (AMD/Intel/Nvidia)"
    return "GPU: not detected. Install torch-directml for AMD/Intel on Windows."


def available_training_devices(mode: str) -> tuple[list[tuple[str, torch.device]], str | None]:
    want_gpu = mode in {"GPU", "BOTH"}
    gpu = gpu_training_device()
    if mode == "CPU":
        return [("CPU", torch.device("cpu"))], None
    if mode == "GPU":
        if gpu is not None:
            return [gpu], None
        return [("CPU", torch.device("cpu"))], "GPU requested but no supported GPU backend is available; using CPU."
    if want_gpu and gpu is not None:
        return [gpu, ("CPU", torch.device("cpu"))], None
    return [("CPU", torch.device("cpu"))], "Both requested but no supported GPU backend is available; using CPU only."


def package_from_b64(value: str) -> dict:
    return torch.load(BytesIO(base64.b64decode(value.encode("ascii"))), map_location="cpu", weights_only=False)


def package_to_b64(package: dict) -> str:
    buf = BytesIO()
    torch.save(package, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_local_package() -> dict:
    if not LATEST_MODEL_FILE.exists():
        return {"kind": MODEL_KIND, "version": 0, "vocab": {}, "labels": [], "state_dict": None}
    return torch.load(LATEST_MODEL_FILE, map_location="cpu", weights_only=False)


def save_local_package(package: dict) -> None:
    LATEST_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, LATEST_MODEL_FILE)


def loss_for_package(package: dict, prompt: str, target: str) -> float:
    labels = package.get("labels") or []
    if target not in labels:
        return float("inf")
    model = make_model(len(package["vocab"]), len(labels))
    model.load_state_dict(package["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(vectorise(prompt, package["vocab"]))
        y = torch.tensor([labels.index(target)], dtype=torch.long)
        return float(F.cross_entropy(logits, y).item())


class ClientApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Kitless Client - Archie")
        self.geometry("980x720")
        self.server_url = tk.StringVar(value=local_server_url())
        self.api_key = tk.StringVar(value=local_api_key())
        save_api_key(self.api_key.get())
        save_server_url(self.server_url.get())
        self.install_id = get_or_create_client_id("client")
        self.client_id = new_session_id("client")
        self.connected = False
        self.contributing = False
        self.keep_training = tk.BooleanVar(value=True)
        self.compute_mode = tk.StringVar(value="BOTH")
        self.local_model = load_local_package()
        self._build()
        self.after(1500, self._heartbeat_loop)

    def _build(self) -> None:
        self.configure(bg="#101623")
        hero = tk.Frame(self, bg="#101623", padx=18, pady=14)
        hero.pack(fill=tk.X)
        tk.Label(hero, text="Kitless", fg="#ffffff", bg="#101623", font=("Segoe UI", 24, "bold")).pack(anchor=tk.W)
        tk.Label(hero, text="Public Client - chat with Archie and contribute training power", fg="#9fb3d1", bg="#101623", font=("Segoe UI", 11)).pack(anchor=tk.W)

        card = ttk.Frame(self, padding=12)
        card.pack(fill=tk.X, padx=14, pady=8)
        ttk.Label(card, text="Server").pack(side=tk.LEFT)
        ttk.Entry(card, textvariable=self.server_url, width=34).pack(side=tk.LEFT, padx=8)
        ttk.Label(card, text="API key").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Entry(card, textvariable=self.api_key, width=18, show="*").pack(side=tk.LEFT, padx=6)
        ttk.Button(card, text="SAVE", command=lambda: self.save_connection_settings(silent=False)).pack(side=tk.LEFT, padx=4)
        ttk.Button(card, text="CONNECT", command=self.connect).pack(side=tk.LEFT, padx=4)
        ttk.Button(card, text="SYNC MODEL", command=self.sync_model).pack(side=tk.LEFT, padx=4)
        ttk.Button(card, text="CONTRIBUTE TRAINING", command=self.toggle_contribute).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(card, text="KEEP RETRAINING", variable=self.keep_training).pack(side=tk.LEFT, padx=4)
        self.status = ttk.Label(card, text="Offline")
        self.status.pack(side=tk.RIGHT)

        compute = ttk.LabelFrame(self, text="Training Power", padding=10)
        compute.pack(fill=tk.X, padx=14, pady=4)
        ttk.Label(compute, text="Use").pack(side=tk.LEFT, padx=(0, 6))
        for label in ("CPU", "GPU", "BOTH"):
            ttk.Radiobutton(compute, text=label, value=label, variable=self.compute_mode).pack(side=tk.LEFT, padx=8)
        ttk.Label(compute, text=gpu_status_text()).pack(side=tk.LEFT, padx=16)

        stats = ttk.LabelFrame(self, text="Network", padding=12)
        stats.pack(fill=tk.X, padx=14, pady=8)
        self.online_lbl = ttk.Label(stats, text="Online: 0")
        self.online_lbl.pack(side=tk.LEFT, padx=10)
        self.version_lbl = ttk.Label(stats, text="Model build: 0")
        self.version_lbl.pack(side=tk.LEFT, padx=10)
        self.steps_lbl = ttk.Label(stats, text="Training steps: 0")
        self.steps_lbl.pack(side=tk.LEFT, padx=10)
        self.weights_lbl = ttk.Label(stats, text="Weights: 0")
        self.weights_lbl.pack(side=tk.LEFT, padx=10)
        self.tasks_lbl = ttk.Label(stats, text="Open tasks: 0")
        self.tasks_lbl.pack(side=tk.LEFT, padx=10)
        self.packs_lbl = ttk.Label(stats, text="Teaching packs: 0")
        self.packs_lbl.pack(side=tk.LEFT, padx=10)
        self.done_lbl = ttk.Label(stats, text="Completed: 0")
        self.done_lbl.pack(side=tk.LEFT, padx=10)
        self.contrib_lbl = ttk.Label(stats, text="Contributions: 0")
        self.contrib_lbl.pack(side=tk.LEFT, padx=10)
        self.round_lbl = ttk.Label(stats, text="Round: 0")
        self.round_lbl.pack(side=tk.LEFT, padx=10)

        self.log = scrolledtext.ScrolledText(self, height=8, font=("Consolas", 10))
        self.log.pack(fill=tk.X, padx=14, pady=8)

        chat_box = ttk.LabelFrame(self, text="Talk To Archie", padding=10)
        chat_box.pack(fill=tk.BOTH, expand=True, padx=14, pady=8)
        self.chat = scrolledtext.ScrolledText(chat_box, font=("Segoe UI", 11), height=12, wrap=tk.WORD)
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.insert(
            tk.END,
            "Archie chat is here.\n"
            "1. CONNECT to the server.\n"
            "2. Click SYNC MODEL after some training has completed.\n"
            "3. Type below and press SEND TO ARCHIE.\n\n",
        )

        row = ttk.Frame(self, padding=12)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Message Archie").pack(side=tk.LEFT, padx=(0, 8))
        self.entry = ttk.Entry(row, font=("Segoe UI", 12))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.entry.bind("<Return>", lambda _e: self.send_chat())
        ttk.Button(row, text="SEND TO ARCHIE", command=self.send_chat).pack(side=tk.RIGHT)

    def log_line(self, text: str) -> None:
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)

    def save_connection_settings(self, silent: bool = True) -> None:
        self.server_url.set((self.server_url.get().strip() or DEFAULT_SERVER_URL).rstrip("/"))
        save_api_key(self.api_key.get())
        if not valid_api_key(self.api_key.get()):
            self.api_key.set(local_api_key())
        save_server_url(self.server_url.get())
        self.log_line("Connection settings saved locally.")
        if not silent:
            messagebox.showinfo(APP_NAME, "Connection settings saved.")

    def connect(self) -> None:
        try:
            self.save_connection_settings(silent=True)
            data = http_json(
                "POST",
                self.server_url.get() + "/connect",
                {"client_id": self.client_id, "install_id": self.install_id, "name": socket.gethostname(), "mode": "client"},
            )
            self.connected = True
            self.status.configure(text="Connected")
            self.apply_status(data["status"])
            self.log_line("Connected to Kitless Server.")
        except Exception as e:
            messagebox.showerror(APP_NAME, str(e))

    def apply_status(self, s: dict) -> None:
        self.online_lbl.configure(text=f"Online: {s.get('online', 0)}")
        self.version_lbl.configure(text=f"Model build: {s.get('model_version', 0)}")
        self.steps_lbl.configure(text=f"Training steps: {s.get('training_steps', s.get('contributions', 0))}")
        self.weights_lbl.configure(text=f"Weights: {s.get('model_weights', 0)}")
        self.tasks_lbl.configure(text=f"Open tasks: {s.get('open_tasks', 0)}")
        self.packs_lbl.configure(text=f"Teaching packs: {s.get('packs_received', 0)}")
        self.done_lbl.configure(text=f"Completed: {s.get('completed_tasks', 0)}")
        self.contrib_lbl.configure(text=f"Contributions: {s.get('contributions', 0)}")
        self.round_lbl.configure(text=f"Round: {s.get('training_round', 0)}")

    def _heartbeat_loop(self) -> None:
        if self.connected:
            try:
                data = http_json("POST", self.server_url.get() + "/heartbeat", {"client_id": self.client_id}, timeout=4)
                self.apply_status(data["status"])
            except Exception:
                self.status.configure(text="Disconnected")
                self.connected = False
        self.after(3000, self._heartbeat_loop)

    def toggle_contribute(self) -> None:
        if not self.connected:
            self.connect()
        self.contributing = not self.contributing
        self.log_line("Training contribution started." if self.contributing else "Training contribution stopped.")
        if self.contributing:
            threading.Thread(target=self.contribution_loop, daemon=True).start()

    def contribution_loop(self) -> None:
        while self.contributing and self.connected:
            try:
                data = http_json(
                    "POST",
                    self.server_url.get() + "/task",
                    {"client_id": self.client_id, "keep_training": self.keep_training.get()},
                    timeout=8,
                )
                task = data.get("task")
                if not task:
                    reason = data.get("reason") or "no training task available"
                    self.after(0, self.log_line, f"Waiting: {reason}.")
                    threading.Event().wait(5)
                    continue
                result = self.process_training_task(task, data["model_package"])
                submit = http_json(
                    "POST",
                    self.server_url.get() + "/submit",
                    {"client_id": self.client_id, "task_id": task["id"], "result": result},
                    timeout=10,
                )
                if submit.get("accepted"):
                    swarm = submit.get("swarm") or {}
                    self.after(
                        0,
                        self.log_line,
                        f"Shared task {task['id'][:8]} accepted. Swarm {swarm.get('contributors', 0)}/{swarm.get('target_contributions', 1)}. Step {submit.get('training_steps')}. Build {submit.get('model_version')}.",
                    )
                    self.sync_model(silent=True)
                    threading.Event().wait(1)
                else:
                    self.after(0, self.log_line, f"Task {task['id'][:8]} rejected: {submit.get('reason')}")
            except Exception as e:
                self.after(0, self.log_line, f"Training contribution error: {e}")
                threading.Event().wait(5)

    def process_training_task(self, task: dict, model_package: str) -> dict:
        prompt = task["prompt"]
        target = task["target"]
        package = package_from_b64(model_package)
        if package.get("kind") != MODEL_KIND:
            raise RuntimeError("Server sent unknown model kind.")
        labels = package["labels"]
        if target not in labels:
            raise RuntimeError("Training target is not in model labels.")
        devices, warning = available_training_devices(self.compute_mode.get())
        if warning:
            self.after(0, self.log_line, warning)
        best_state = None
        best_before = float("inf")
        best_after = float("inf")
        used_devices: list[str] = []
        for device_name, device in devices:
            model = make_model(len(package["vocab"]), len(labels)).to(device)
            model.load_state_dict(package["state_dict"])
            model.train()
            x = vectorise(prompt, package["vocab"]).to(device)
            y = torch.tensor([labels.index(target)], dtype=torch.long, device=device)
            with torch.no_grad():
                loss_before = float(F.cross_entropy(model(x), y).item())
            opt = torch.optim.AdamW(model.parameters(), lr=0.05)
            for _ in range(60):
                loss = F.cross_entropy(model(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                loss_after = float(F.cross_entropy(model(x), y).item())
            used_devices.append(device_name)
            if loss_after < best_after:
                best_before = loss_before
                best_after = loss_after
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if best_state is None:
            raise RuntimeError("No training device was available.")
        loss_before = best_before
        loss_after = best_after
        if loss_after >= loss_before:
            raise RuntimeError(f"Local training did not improve loss ({loss_before:.4f} -> {loss_after:.4f}).")
        self.after(0, self.log_line, f"Training used {', '.join(used_devices)}. Best loss {loss_before:.4f} -> {loss_after:.4f}.")
        package["state_dict"] = best_state
        words = re.findall(r"[a-zA-Z']+", f"{prompt} {target}".lower())
        stop = {"the", "and", "you", "are", "with", "that", "this", "from", "what", "who", "how", "why", "for"}
        keywords: list[str] = []
        for word in words:
            if len(word) < 3 or word in stop or word in keywords:
                continue
            keywords.append(word)
        return {
            "prompt": prompt,
            "target": target,
            "digest": sha256(f"{prompt}\n{target}".encode("utf-8")).hexdigest(),
            "keywords": keywords[:20],
            "loss_before": loss_before,
            "loss_after": loss_after,
            "model_package": package_to_b64(package),
        }

    def send_chat(self) -> None:
        msg = self.entry.get().strip()
        if not msg:
            return
        self.entry.delete(0, tk.END)
        self.chat.insert(tk.END, f"You: {msg}\n")
        self.chat.insert(tk.END, f"Archie: {self.answer_from_synced_model(msg)}\n\n")
        self.chat.see(tk.END)

    def sync_model(self, silent: bool = False) -> None:
        if not self.connected:
            self.connect()
        try:
            data = http_json("GET", self.server_url.get() + "/model", timeout=20)
            package = package_from_b64(data["model_package"])
            save_local_package(package)
            self.local_model = package
            def update_ui() -> None:
                self.version_lbl.configure(text=f"Model build: {data.get('model_version', 0)}")
                self.steps_lbl.configure(text=f"Training steps: {data.get('training_steps', 0)}")
                self.weights_lbl.configure(text=f"Weights: {data.get('model_weights', 0)}")
                if silent:
                    self.log_line("Checkpoint synced to shared server model.pt.")
                else:
                    self.log_line(
                        f"Downloaded Archie model.pt build {data.get('model_version', 0)} with {len(package.get('labels', []))} learned answers and {data.get('model_weights', 0)} weights."
                    )
            self.after(0, update_ui)
        except Exception as e:
            if silent:
                self.after(0, self.log_line, f"Checkpoint sync error: {e}")
            else:
                messagebox.showerror(APP_NAME, str(e))

    def answer_from_synced_model(self, msg: str) -> str:
        labels = self.local_model.get("labels") or []
        if not labels or self.local_model.get("state_dict") is None:
            return "No Archie model.pt synced yet. Click SYNC MODEL after clients train tasks."
        try:
            model = make_model(len(self.local_model["vocab"]), len(labels))
            model.load_state_dict(self.local_model["state_dict"])
            model.eval()
            with torch.no_grad():
                idx = int(torch.argmax(model(vectorise(msg, self.local_model["vocab"])), dim=-1).item())
            return labels[idx]
        except Exception:
            return "I need a synced Kitless model before I can answer properly."


if __name__ == "__main__":
    exit_when_parent_closes()
    ClientApp().mainloop()
