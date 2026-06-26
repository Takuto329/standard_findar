#!/usr/bin/env python3
"""
refstar_gui.py — Main launcher for refstar_planner GUI.

Opens a tabbed window with two modes:
  固定モード  : fixed RA/Dec target  (reference stars for any star/field)
  可変モード  : asteroid mode via Horizons API (time-series reference star check)

Run:  python3.12 refstar_gui.py
"""

from __future__ import annotations

# matplotlib backend must be set before any pyplot import
import matplotlib
matplotlib.use("TkAgg")

import platform as _platform
if _platform.system() == "Darwin":
    matplotlib.rcParams["font.family"] = [
        "Hiragino Sans", "Hiragino Maru Gothic Pro", "DejaVu Sans"
    ]

import tkinter as tk
from tkinter import ttk

from refstar_gui_fixed    import FixedModePanel
from refstar_gui_variable import VariableModePanel


def main() -> None:
    root = tk.Tk()
    root.title("refstar_planner — 参照星プランナー")
    root.minsize(1200, 750)

    try:
        root.tk.call("::tk::unsupported::MacWindowStyle", "style", root._w, "document", "")
    except Exception:
        pass

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=2, pady=2)

    fixed_panel    = FixedModePanel(nb)
    variable_panel = VariableModePanel(nb)

    nb.add(fixed_panel,    text="  📍 固定モード  ")
    nb.add(variable_panel, text="  ☄  可変モード (小惑星)  ")

    root.mainloop()


if __name__ == "__main__":
    main()
