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
import time
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
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.pool_chat import (
    brain_ready_local,
    handle_pool_chat_job,
    poll_pool_chat_work,
    submit_pool_chat_result,
    sync_brain_from_server,
)
from shared.archie_brain import is_archie_brain_file

from shared.chat_reply import chat_answer_from_label, direct_chat_reply
from shared.help_docs import load_doc, show_help_window
from shared.settings import TARGET_LOSS, MASTERED_LOSS, chat_mode_label, training_loss_accepted
CLIENT_DATA_DIR = APP_DIR / "client_data"
CLIENT_CHECKPOINT_DIR = APP_DIR / "client_checkpoint"
CLIENT_ID_FILE = CLIENT_DATA_DIR / "client_id.txt"
API_KEY_FILE = CLIENT_DATA_DIR / "api_key.txt"
SERVER_URL_FILE = CLIENT_DATA_DIR / "server_url.txt"
DISCLAIMER_ACCEPTED_FILE = CLIENT_DATA_DIR / "disclaimer_accepted.txt"
LATEST_MODEL_FILE = CLIENT_CHECKPOINT_DIR / "model.pt"
DEFAULT_SERVER_URL = "https://kitless.co.uk"
DEFAULT_API_KEY_PARTS = (
    "917530a43cfc3dcf",
    "3a910c64e5a93b30",
    "9c59a9c4da03076b",
    "2fe918ecf15ed619",
)
MODEL_KIND = "kitless-linear-archie-v1"
CHAT_WELCOME = (
    "Archie chat is here.\n"
    "The server hosts model.pt — your PC (and the pool) runs the brain when you talk to Archie.\n"
    "Stay connected so pool chat can use your CPU/GPU. Press START TRAINING to add network teaching.\n\n"
)


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
        raise RuntimeError(f"{url}: {data.get('error', str(e))}") from e


def is_stale_swarm_chat_response(data: dict) -> bool:
    stale_modes = frozenset({"swarm", "swarm-timeout", "direct", "quick", "linear", "network"})
    mode = str(data.get("mode", "")).strip().lower()
    if mode in stale_modes:
        return True
    answer = str(data.get("answer", "")).strip()
    prefixes = (
        "No swarm helper answered",
        "Leave more clients connected with synced model.pt",
    )
    return any(answer.startswith(prefix) for prefix in prefixes)


def is_network_server_outdated(status: dict | None) -> bool:
    if not status:
        return False
    return status.get("chat_backend") != "pool"


STALE_SERVER_CHAT_MESSAGE = (
    "kitless.co.uk is not running pool chat yet. "
    "Upload updated app.py + shared/ to the server and restart kitless. "
    "Until then use http://127.0.0.1:8799 with START_SERVER.bat."
)


def request_archie_chat(server_url: str, client_id: str, message: str, timeout: int = 900) -> dict:
    base = server_url.rstrip("/")
    payload = {"client_id": client_id, "message": message}
    last_error: Exception | None = None
    for path in ("/chat", "/chat_swarm"):
        url = f"{base}{path}"
        try:
            data = http_json("POST", url, payload, timeout=timeout)
        except RuntimeError as e:
            last_error = e
            text = str(e).lower()
            if path == "/chat" and ("404" in text or "not found" in text):
                continue
            raise
        if is_stale_swarm_chat_response(data):
            raise RuntimeError(STALE_SERVER_CHAT_MESSAGE)
        return data
    raise RuntimeError(
        f"Server has no working chat endpoint ({last_error}). "
        "Deploy updated server/app.py and add nginx /chat proxy."
    )


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
        self.withdraw()
        ensure_dirs()
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
        self.pool_size = 1
        self.pool_mode = "solo"
        self.loss_history: list[tuple[float, float]] = []
        self.graph_visible = tk.BooleanVar(value=False)
        self.log_visible = tk.BooleanVar(value=False)
        self._build()
        if not self.show_startup_disclaimer():
            return
        self.deiconify()
        self._build_menu()
        self.apply_view_menu()
        self.after(500, self.auto_startup_connect)
        self.after(1500, self._heartbeat_loop)
        threading.Thread(target=self._pool_chat_loop, daemon=True).start()

    def auto_startup_connect(self) -> None:
        threading.Thread(target=lambda: self.connect(silent=True), daemon=True).start()

    def show_startup_disclaimer(self) -> bool:
        if DISCLAIMER_ACCEPTED_FILE.exists():
            return True

        accepted = {"ok": False}
        win = tk.Toplevel(self)
        win.title("Kitless Disclaimer")
        win.geometry("780x580")
        win.minsize(660, 480)
        win.configure(bg="#120b1d")
        win.grab_set()
        win.lift()
        win.focus_force()
        try:
            win.attributes("-topmost", True)
            win.after(750, lambda: win.attributes("-topmost", False))
        except tk.TclError:
            pass
        win.update_idletasks()

        def open_help_window(action) -> None:
            win.grab_release()
            action()
            win.lift()

        def close_app() -> None:
            accepted["ok"] = False
            win.destroy()
            self.destroy()

        def accept() -> None:
            DISCLAIMER_ACCEPTED_FILE.parent.mkdir(parents=True, exist_ok=True)
            DISCLAIMER_ACCEPTED_FILE.write_text("accepted=8/07/2026\n", encoding="utf-8")
            accepted["ok"] = True
            win.destroy()

        header = tk.Frame(win, bg="#211332", padx=18, pady=14)
        header.pack(fill=tk.X)
        tk.Label(header, text="Before using Kitless", fg="#ffffff", bg="#211332", font=("Segoe UI", 20, "bold")).pack(anchor=tk.W)
        tk.Label(header, text="Please read and accept the important notices", fg="#d0b7ff", bg="#211332", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        tk.Frame(win, bg="#a371f7", height=3).pack(fill=tk.X)

        body = tk.Frame(win, bg="#120b1d", padx=18, pady=16)
        body.pack(fill=tk.BOTH, expand=True)
        msg = (
            "Kitless is open-source distributed AI software. By using it, you accept that you are responsible for your own device, server configuration, data, messages, contributed compute, and use of AI output.\n\n"
            "Archie can be wrong. Do not rely on Archie for medical, legal, financial, emergency, safety-critical, or other high-risk decisions.\n\n"
            "When training is enabled, this computer may use CPU, GPU, memory, storage, network bandwidth, and electrical power.\n\n"
            "Please read the Licence, Privacy Policy, and Terms of Service before continuing."
        )
        tk.Label(body, text=msg, fg="#f2e7ff", bg="#120b1d", font=("Segoe UI", 11), justify=tk.LEFT, wraplength=720).pack(fill=tk.X, anchor=tk.W)

        links = ttk.Frame(body)
        links.pack(fill=tk.X, pady=16)
        ttk.Button(links, text="Read Licence", command=lambda: open_help_window(self.show_licence)).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(links, text="Read Privacy Policy", command=lambda: open_help_window(self.show_privacy_policy)).pack(side=tk.LEFT, padx=8)
        ttk.Button(links, text="Read Terms of Service", command=lambda: open_help_window(self.show_terms_of_service)).pack(side=tk.LEFT, padx=8)

        ticked = tk.BooleanVar(value=False)
        footer = ttk.Frame(body)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        accept_button = ttk.Button(footer, text="Accept and continue", command=accept, state=tk.DISABLED)

        def update_accept_state() -> None:
            accept_button.configure(state=tk.NORMAL if ticked.get() else tk.DISABLED)

        ttk.Checkbutton(
            body,
            text="I have read and accept the Licence, Privacy Policy, Terms of Service, and disclaimer.",
            variable=ticked,
            command=update_accept_state,
        ).pack(anchor=tk.W, pady=(8, 16))

        ttk.Button(footer, text="Exit", command=close_app).pack(side=tk.RIGHT, padx=(8, 0))
        accept_button.pack(side=tk.RIGHT)

        win.protocol("WM_DELETE_WINDOW", close_app)
        self.wait_window(win)
        return accepted["ok"]

    def _build(self) -> None:
        self._build_menu()
        self.configure(bg="#101623")
        hero = tk.Frame(self, bg="#101623", padx=18, pady=14)
        hero.pack(fill=tk.X)
        tk.Label(hero, text="Kitless", fg="#ffffff", bg="#101623", font=("Segoe UI", 24, "bold")).pack(anchor=tk.W)
        tk.Label(hero, text="Public Client — pool chat and shared training power across the network", fg="#9fb3d1", bg="#101623", font=("Segoe UI", 11)).pack(anchor=tk.W)

        card = ttk.Frame(self, padding=12)
        card.pack(fill=tk.X, padx=14, pady=8)
        ttk.Label(card, text="Server").pack(side=tk.LEFT)
        ttk.Entry(card, textvariable=self.server_url, width=34).pack(side=tk.LEFT, padx=8)
        ttk.Label(card, text="API key").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Entry(card, textvariable=self.api_key, width=18, show="*").pack(side=tk.LEFT, padx=6)
        ttk.Button(card, text="SAVE", command=lambda: self.save_connection_settings(silent=False)).pack(side=tk.LEFT, padx=4)
        ttk.Button(card, text="CONNECT", command=self.connect).pack(side=tk.LEFT, padx=4)
        self.training_btn = ttk.Button(card, text="START TRAINING", command=self.toggle_contribute)
        self.training_btn.pack(side=tk.LEFT, padx=4)
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
        self.online_lbl = ttk.Label(stats, text="Pool: 0")
        self.online_lbl.pack(side=tk.LEFT, padx=10)
        self.version_lbl = ttk.Label(stats, text="Model build: 0")
        self.version_lbl.pack(side=tk.LEFT, padx=10)
        self.steps_lbl = ttk.Label(stats, text="Training steps: 0")
        self.steps_lbl.pack(side=tk.LEFT, padx=10)
        self.weights_lbl = ttk.Label(stats, text="Weights: 0")
        self.weights_lbl.pack(side=tk.LEFT, padx=10)
        self.learned_lbl = ttk.Label(stats, text="Learned: 0")
        self.learned_lbl.pack(side=tk.LEFT, padx=10)
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

        self.chat_box = ttk.LabelFrame(self, text="Talk To Archie", padding=10)
        self.chat_box.pack(fill=tk.BOTH, expand=True, padx=14, pady=8)
        self.chat = scrolledtext.ScrolledText(self.chat_box, font=("Segoe UI", 11), height=12, wrap=tk.WORD)
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.insert(tk.END, CHAT_WELCOME)

        self.message_row = ttk.Frame(self.chat_box, padding=(0, 8, 0, 0))
        self.message_row.pack(fill=tk.X)
        self.entry = ttk.Entry(self.message_row, font=("Segoe UI", 12))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.entry.bind("<Return>", lambda _e: self.send_chat())
        chat_actions = ttk.Frame(self.message_row)
        chat_actions.pack(side=tk.RIGHT)
        ttk.Button(chat_actions, text="CLEAR CHAT", command=self.clear_chat).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(chat_actions, text="SEND TO ARCHIE", command=self.send_chat).pack(side=tk.LEFT)

        self.graph_box = ttk.LabelFrame(self, text="Training Graph (local client only)", padding=10)
        self.graph_canvas = tk.Canvas(self.graph_box, height=150, bg="#0d1420", highlightthickness=0)
        self.graph_canvas.pack(fill=tk.X)
        self.graph_info = ttk.Label(
            self.graph_box,
            text="Waiting for local training data. Green = before training. Blue = after training, and blue is the better result when it is lower.",
        )
        self.graph_info.pack(anchor=tk.W, pady=(6, 0))
        self.graph_canvas.bind("<Configure>", lambda _event: self.draw_loss_graph())

        self.log = scrolledtext.ScrolledText(self, height=8, font=("Consolas", 10))
        self.apply_view_menu()

    def _build_menu(self) -> None:
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Exit", command=self.destroy)
        menu.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menu, tearoff=False)
        view_menu.add_checkbutton(label="Training Graph", variable=self.graph_visible, command=self.apply_view_menu)
        view_menu.add_checkbutton(label="Terminal Log", variable=self.log_visible, command=self.apply_view_menu)
        view_menu.add_separator()
        view_menu.add_command(label="Clear Chat", command=self.clear_chat)
        menu.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="FAQ", command=self.show_faq)
        help_menu.add_command(label="Licence", command=self.show_licence)
        help_menu.add_command(label="Privacy Policy", command=self.show_privacy_policy)
        help_menu.add_command(label="Terms of Service", command=self.show_terms_of_service)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)
        menu.add_cascade(label="Help", menu=help_menu)
        self.configure(menu=menu)

    def apply_view_menu(self) -> None:
        self.graph_box.pack_forget()
        self.log.pack_forget()
        if self.graph_visible.get():
            self.graph_box.pack(fill=tk.X, padx=14, pady=6, after=self.chat_box)
            self.draw_loss_graph()
        if self.log_visible.get():
            after_widget = self.graph_box if self.graph_visible.get() else self.chat_box
            self.log.pack(fill=tk.X, padx=14, pady=8, after=after_widget)

    def record_training_graph(self, loss_before: float, loss_after: float, elapsed_seconds: float) -> None:
        self.loss_history.append((loss_before, loss_after))
        self.loss_history = self.loss_history[-60:]
        self.graph_info.configure(
            text=f"Green before {loss_before:.6f}, blue after {loss_after:.6f}. If blue is lower than green, Archie improved. Task time {elapsed_seconds:.1f}s"
        )
        self.draw_loss_graph()

    def draw_loss_graph(self) -> None:
        canvas = self.graph_canvas
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        pad = 18
        canvas.create_text(pad, 12, anchor=tk.W, fill="#9fb3d1", text="Local training loss - blue lower than green is better")
        canvas.create_line(pad, height - pad, width - pad, height - pad, fill="#2a3a52")
        canvas.create_line(pad, pad, pad, height - pad, fill="#2a3a52")
        if not self.loss_history:
            canvas.create_text(width // 2, height // 2, fill="#9fb3d1", text="Start training to draw live loss graphs")
            return
        values = [value for pair in self.loss_history for value in pair]
        lo = min(values)
        hi = max(values)
        span = hi - lo or 1.0

        def point(index: int, value: float) -> tuple[float, float]:
            x = pad if len(self.loss_history) == 1 else pad + (index / (len(self.loss_history) - 1)) * (width - (pad * 2))
            y = height - pad - ((value - lo) / span) * (height - (pad * 2))
            return x, y

        before_points = [point(index, pair[0]) for index, pair in enumerate(self.loss_history)]
        after_points = [point(index, pair[1]) for index, pair in enumerate(self.loss_history)]
        if len(before_points) > 1:
            canvas.create_line(*[coord for point_pair in before_points for coord in point_pair], fill="#7ee787", width=2)
        if len(after_points) > 1:
            canvas.create_line(*[coord for point_pair in after_points for coord in point_pair], fill="#58a6ff", width=2)
        for x, y in before_points:
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#7ee787", outline="")
        for x, y in after_points:
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#58a6ff", outline="")

    def show_faq(self) -> None:
        show_help_window(
            self,
            "Kitless Client FAQ",
            (
                "Kitless Public Client FAQ\n\n"
                "Talk To Archie\n"
                "Type a message and press SEND TO ARCHIE. Online clients share CPU to score answer shards; the server combines the best swarm reply.\n\n"
                "CONNECT\n"
                "Connects this client to the Kitless server.\n\n"
                "START TRAINING / STOP TRAINING\n"
                "Join network power — your CPU/GPU helps train the shared model on the server.\n\n"
                "KEEP RETRAINING\n"
                "Keeps asking the server for more training work after a task finishes.\n\n"
                "Training Power: CPU / GPU / BOTH\n"
                "Chooses which hardware can train. BOTH uses GPU first when available, then CPU adds more work.\n\n"
                "Network Status\n"
                "Shows online clients, model build, training steps, weights, open tasks, completed tasks, contributions, and training round.\n\n"
                "View Menu\n"
                "The Training Graph shows local loss before and after training. The Terminal Log shows client events and errors. Clear Chat wipes the chat window on this PC only.\n\n"
                "Clear Chat\n"
                "Use CLEAR CHAT or View -> Clear Chat when you are finished. This only clears the chat display on your computer. It does not delete server training data.\n\n"
                "Join Power\n"
                "Each client syncs the shared network model, helps answer swarm chat shards while connected, and can press START TRAINING to improve the model.\n"
                "Network status (online count, open tasks, model build) refreshes about every 3 seconds while connected.\n\n"
                "Graph Colours\n"
                "Green is loss before training. Blue is loss after training. Lower blue means the task improved.\n"
            ),
            subtitle="Kitless Client Help Centre",
        )

    def show_about(self) -> None:
        readme = None
        for base in (PROJECT_ROOT, APP_DIR):
            candidate = base / "README.md"
            if candidate.exists():
                readme = candidate
                break
        text = readme.read_text(encoding="utf-8") if readme else "Kitless README.md was not found."
        show_help_window(self, "About Kitless", text, subtitle="Kitless Client Help Centre")

    def show_licence(self) -> None:
        show_help_window(
            self,
            "Kitless Licence",
            load_doc("LICENCE.txt", "Kitless licence file was not found."),
            subtitle="Kitless Client Help Centre",
        )

    def show_privacy_policy(self) -> None:
        show_help_window(
            self,
            "Kitless Privacy Policy",
            load_doc("PRIVACY.txt", "Kitless privacy policy file was not found."),
            subtitle="Kitless Client Help Centre",
        )

    def show_terms_of_service(self) -> None:
        show_help_window(
            self,
            "Kitless Terms of Service",
            load_doc("TERMS.txt", "Kitless terms of service file was not found."),
            subtitle="Kitless Client Help Centre",
        )

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

    def connect(self, silent: bool = False) -> bool:
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
            self.sync_model_from_server(silent=True)
            threading.Thread(
                target=lambda: self.sync_brain_from_server(silent=not is_archie_brain_file()),
                daemon=True,
            ).start()
            pool_size = int((data.get("status") or {}).get("pool_size", (data.get("status") or {}).get("online", 0)) or 0)
            self.log_line(f"Connected to Kitless Server. Pool: {pool_size}. Pool chat runs on connected PCs.")
            return True
        except Exception as e:
            if silent:
                self.log_line(f"Auto-connect waiting for server: {e}")
            else:
                messagebox.showerror(APP_NAME, str(e))
            return False

    def apply_status(self, s: dict) -> None:
        pool_size = int(s.get("pool_size", s.get("online", 0)) or 0)
        pool_mode = str(s.get("pool_mode", "solo" if pool_size <= 1 else "swarm"))
        if pool_size != self.pool_size or pool_mode != self.pool_mode:
            if pool_size <= 1:
                self.log_line(f"Pool: {pool_size} client online.")
            else:
                self.log_line(f"Pool: {pool_size} clients online — training power shared.")
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        mode_label = "swarm" if pool_mode == "swarm" else "solo"
        self.online_lbl.configure(text=f"Pool: {pool_size} ({mode_label})")
        self.version_lbl.configure(text=f"Model build: {s.get('model_version', 0)}")
        self.steps_lbl.configure(text=f"Training steps: {s.get('training_steps', s.get('contributions', 0))}")
        self.weights_lbl.configure(text=f"Weights: {s.get('model_weights', 0)}")
        self.learned_lbl.configure(text=f"Learned: {s.get('learned_answers', 0)}")
        self.tasks_lbl.configure(text=f"Open tasks: {s.get('open_tasks', 0)}")
        self.packs_lbl.configure(text=f"Teaching packs: {s.get('packs_received', 0)}")
        self.done_lbl.configure(text=f"Completed: {s.get('completed_tasks', 0)}")
        self.contrib_lbl.configure(text=f"Contributions: {s.get('contributions', 0)}")
        self.round_lbl.configure(text=f"Round: {s.get('training_round', 0)}")
        if s.get("chat_backend") != "pool":
            if not getattr(self, "_warned_stale_server", False):
                self._warned_stale_server = True
                self.log_line(
                    "Network server outdated: pool chat is not live on kitless.co.uk. "
                    "Upload updated app.py + shared/, restart kitless."
                )
        elif s.get("brain_ready") is False and not getattr(self, "_warned_no_brain", False):
            self._warned_no_brain = True
            self.log_line("Warning: Archie brain (model.pt) is not hosted on the server yet.")
        self.maybe_sync_model(s.get("model_version", 0))

    def sync_model_from_server(self, silent: bool = False) -> bool:
        try:
            data = http_json("GET", self.server_url.get() + "/model", timeout=60)
            package = package_from_b64(data["model_package"])
            if package.get("kind") != MODEL_KIND:
                raise RuntimeError("Server sent unknown model kind.")
            package["version"] = data.get("model_version", package.get("version", 0))
            package["training_steps"] = data.get("training_steps", package.get("training_steps", 0))
            save_local_package(package)
            self.local_model = package
            if not silent:
                self.log_line(f"Synchronised network model build {package['version']}.")
            return True
        except Exception as e:
            if not silent:
                self.log_line(f"Model sync failed: {e}")
            return False

    def maybe_sync_model(self, remote_version: int) -> None:
        if int(remote_version or 0) != int(self.local_model.get("version", 0) or 0):
            threading.Thread(target=lambda: self.sync_model_from_server(silent=True), daemon=True).start()

    def sync_brain_from_server(self, silent: bool = False) -> bool:
        try:
            return sync_brain_from_server(
                http_json,
                self.server_url.get(),
                api_key=local_api_key(),
                log=None if silent else lambda text: self.after(0, self.log_line, text),
            )
        except Exception as e:
            if not silent:
                self.after(0, self.log_line, f"Brain sync failed: {e}")
            return False

    def _pool_chat_loop(self) -> None:
        while True:
            if not self.connected:
                time.sleep(2)
                continue
            try:
                work = poll_pool_chat_work(
                    http_json,
                    self.server_url.get(),
                    self.client_id,
                    timeout=15,
                )
                if not work:
                    time.sleep(0.25)
                    continue
                job_id = str(work["job_id"])
                mode_hint = str(work.get("mode_hint") or "brain")
                answer, mode = handle_pool_chat_job(
                    work,
                    http_json,
                    self.server_url.get(),
                    api_key=local_api_key(),
                    log=lambda text: self.after(0, self.log_line, text),
                )
                if mode == "none":
                    mode = mode_hint
                submit_pool_chat_result(
                    http_json,
                    self.server_url.get(),
                    self.client_id,
                    job_id,
                    answer,
                    mode,
                    timeout=60,
                )
                self.after(0, self.log_line, f"Pool chat answered on this PC (job {job_id[:8]}).")
            except Exception as e:
                if self.connected:
                    self.after(0, self.log_line, f"Pool chat worker: {e}")
                time.sleep(2)

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
        if self.contributing:
            self.training_btn.configure(text="STOP TRAINING")
            self.status.configure(text="Training")
            self.log_line("Training contribution started. Press STOP TRAINING to pause and chat.")
        else:
            self.training_btn.configure(text="START TRAINING")
            self.status.configure(text="Connected" if self.connected else "Offline")
            self.log_line("Training contribution stopped. You can talk to Archie, then press START TRAINING again.")
        if self.contributing:
            threading.Thread(target=self.contribution_loop, daemon=True).start()

    def contribution_loop(self) -> None:
        while self.contributing and self.connected:
            try:
                data = http_json(
                    "POST",
                    self.server_url.get() + "/task",
                    {"client_id": self.client_id, "keep_training": self.keep_training.get()},
                    timeout=60,
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
                    timeout=120,
                )
                if submit.get("accepted"):
                    swarm = submit.get("swarm") or {}
                    self.after(
                        0,
                        self.log_line,
                        f"Shared task {task['id'][:8]} accepted. Swarm {swarm.get('contributors', 0)}/{swarm.get('target_contributions', 1)}. Step {submit.get('training_steps')}. Build {submit.get('model_version')}.",
                    )
                    threading.Event().wait(1)
                else:
                    self.after(0, self.log_line, f"Task {task['id'][:8]} rejected: {submit.get('reason')}")
            except Exception as e:
                self.after(0, self.log_line, f"Training contribution error: {e}")
                threading.Event().wait(5)

    def process_training_task(self, task: dict, model_package: str) -> dict:
        started = time.time()
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
        if not training_loss_accepted(loss_before, loss_after):
            raise RuntimeError(f"Local training did not improve loss ({loss_before:.4f} -> {loss_after:.4f}).")
        elapsed_seconds = time.time() - started
        if loss_before <= MASTERED_LOSS and loss_after <= MASTERED_LOSS:
            self.after(0, self.log_line, f"Task already learned by shared model (loss {loss_before:.6f}). Confirming completion.")
        else:
            self.after(0, self.log_line, f"Training used {', '.join(used_devices)}. Best loss {loss_before:.4f} -> {loss_after:.4f}.")
        self.after(0, self.record_training_graph, loss_before, loss_after, elapsed_seconds)
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

    def clear_chat(self) -> None:
        self.chat.delete("1.0", tk.END)
        self.chat.insert(tk.END, CHAT_WELCOME)
        self.chat.see(tk.END)

    def send_chat(self) -> None:
        msg = self.entry.get().strip()
        if not msg:
            return
        self.entry.delete(0, tk.END)
        self.chat.insert(tk.END, f"You: {msg}\n")
        if not brain_ready_local():
            self.chat.insert(tk.END, "Archie: downloading brain (model.pt) to this PC first — then answering...\n")
        else:
            self.chat.insert(tk.END, "Archie: thinking...\n")
        self.chat.see(tk.END)
        threading.Thread(target=self._chat_worker, args=(msg,), daemon=True).start()

    def _chat_worker(self, msg: str) -> None:
        chat_mode = "server"
        if not self.connected:
            self.connect(silent=True)
        if not brain_ready_local():
            self.sync_brain_from_server(silent=False)
        try:
            self.save_connection_settings(silent=True)
            data = request_archie_chat(self.server_url.get(), self.client_id, msg, timeout=900)
            answer = str(data.get("answer", "")).strip() or "Archie did not return an answer."
            chat_mode = chat_mode_label(str(data.get("mode", "")))
            if not self.connected:
                self.connected = True
                self.after(0, lambda: self.status.configure(text="Connected"))
        except Exception as e:
            answer = f"Cannot reach Archie on the server ({e}). Check CONNECT and that the server is running."
            chat_mode = "offline"

        def show_answer() -> None:
            text = self.chat.get("1.0", tk.END)
            if "Archie: thinking..." in text or "Archie: downloading brain" in text:
                text = text.replace("Archie: thinking...\n", "")
                text = text.replace("Archie: downloading brain (model.pt) to this PC first — then answering...\n", "")
                self.chat.delete("1.0", tk.END)
                self.chat.insert(tk.END, text)
            self.chat.insert(tk.END, f"Archie ({chat_mode}): {answer}\n\n")
            self.chat.see(tk.END)

        self.after(0, show_answer)
        if chat_mode == "training":
            self.after(0, self.log_line, "Answer from network training.")
        elif chat_mode == "brain+training":
            self.after(0, self.log_line, "Answer from Archie brain (model.pt) with network training facts.")
        elif chat_mode == "brain":
            self.after(0, self.log_line, "Answer from pool brain (model.pt on a connected PC).")
        elif chat_mode == "pool-timeout":
            self.after(0, self.log_line, "No pool client answered in time — stay connected with synced brain.")


if __name__ == "__main__":
    if not sys.executable.lower().endswith("pythonw.exe"):
        exit_when_parent_closes()
    ClientApp().mainloop()
