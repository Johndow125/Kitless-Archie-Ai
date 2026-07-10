from __future__ import annotations

import re
import tkinter as tk
from tkinter import scrolledtext, ttk
from pathlib import Path

CLIENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CLIENT_DIR.parent


def load_doc(filename: str, fallback: str = "") -> str:
    for base in (CLIENT_DIR, PROJECT_ROOT):
        path = base / filename
        if path.exists():
            return path.read_text(encoding="utf-8")
    return fallback


def show_help_window(parent: tk.Misc, title: str, text: str, subtitle: str = "Kitless Help Centre") -> None:
    win = tk.Toplevel(parent)
    win.title(title)
    win.geometry("920x700")
    win.minsize(720, 520)
    win.configure(bg="#120b1d")

    header = tk.Frame(win, bg="#211332", padx=18, pady=14)
    header.pack(fill=tk.X)
    tk.Label(header, text=title, fg="#ffffff", bg="#211332", font=("Segoe UI", 20, "bold")).pack(anchor=tk.W)
    tk.Label(header, text=subtitle, fg="#d0b7ff", bg="#211332", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)

    accent = tk.Frame(win, bg="#a371f7", height=3)
    accent.pack(fill=tk.X)

    body = tk.Frame(win, bg="#120b1d", padx=16, pady=14)
    body.pack(fill=tk.BOTH, expand=True)
    box = scrolledtext.ScrolledText(
        body,
        wrap=tk.WORD,
        font=("Segoe UI", 10),
        bg="#1b1029",
        fg="#f2e7ff",
        insertbackground="#ffffff",
        relief=tk.FLAT,
        borderwidth=0,
        padx=16,
        pady=14,
    )
    box.pack(fill=tk.BOTH, expand=True)
    box.tag_configure("title", foreground="#ffffff", font=("Segoe UI", 16, "bold"), spacing3=10)
    box.tag_configure("section", foreground="#d0b7ff", font=("Segoe UI", 12, "bold"), spacing1=8, spacing3=4)
    box.tag_configure("highlight", foreground="#7ee787", font=("Segoe UI", 10, "bold"))
    box.tag_configure("bullet", foreground="#ffd166", lmargin1=18, lmargin2=34)
    box.tag_configure("body", foreground="#f2e7ff", spacing3=6)

    heading_words = {
        "Talk To Archie",
        "CONNECT",
        "SYNC MODEL",
        "START TRAINING / STOP TRAINING",
        "KEEP RETRAINING",
        "Training Power: CPU / GPU / BOTH",
        "Network Status",
        "View Menu",
        "Clear Chat",
        "How Sync Works",
        "Graph Colours",
        "No warranty",
        "No responsibility",
        "Operator responsibility",
    }
    for idx, line in enumerate(text.splitlines(), start=1):
        tag = "body"
        stripped = line.strip()
        if idx == 1 and stripped:
            tag = "title"
        elif stripped.startswith("-"):
            tag = "bullet"
        elif stripped.startswith("Effective date:"):
            tag = "highlight"
        elif re.match(r"^\d+\.\s+", stripped) or stripped in heading_words:
            tag = "section"
        box.insert(tk.END, line + "\n", tag)
    box.configure(state=tk.DISABLED)

    footer = tk.Frame(win, bg="#120b1d", padx=16, pady=(0, 14))
    footer.pack(fill=tk.X)
    ttk.Button(footer, text="Close", command=win.destroy).pack(side=tk.RIGHT)
