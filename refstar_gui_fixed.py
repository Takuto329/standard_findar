#!/usr/bin/env python3
"""
refstar_gui_fixed.py — Fixed-target mode panel for refstar_planner GUI.

A ttk.Frame subclass that can be embedded in a Notebook tab or run standalone.
"""

from __future__ import annotations

import datetime
import threading
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")

import platform as _platform
if _platform.system() == "Darwin":
    matplotlib.rcParams["font.family"] = [
        "Hiragino Sans", "Hiragino Maru Gothic Pro", "DejaVu Sans"
    ]

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from refstar_planner import (
    OUTPUT_COLUMNS,
    BAND_LABELS,
    add_separation,
    apply_quality_filters,
    circumscribed_radius,
    filter_rectangular_fov,
    parse_coord,
    query_simbad,
    query_target_magnitude,
    query_vizier_catalog,
    summarize,
)

# ---------------------------------------------------------------------------
# Telescope presets
# ---------------------------------------------------------------------------

TELESCOPE_PRESETS: list[tuple[str, Optional[tuple[float, float]]]] = [
    ("すばる 8.2m / HSC",            (90.0, 90.0)),
    ("すばる 8.2m / FOCAS",          (6.0,  6.0)),
    ("すばる 8.2m / IRCS (no AO)",   (0.9,  0.9)),
    ("京大 Seimei 3.8m / TriCCS",    (8.3,  8.3)),
    ("OAO 188cm / MuSCAT",            (6.1,  6.1)),
    ("Nayuta 2m / MSI",               (9.4,  9.4)),
    ("TNG 3.6m / MuSCAT2",            (7.4,  7.4)),
    ("Haleakala 2m / MuSCAT3",        (9.1,  9.1)),
    ("Keck 10m / LRIS",               (6.0,  7.8)),
    ("Keck 10m / DEIMOS",             (16.7, 5.0)),
    ("VLT 8.2m / FORS2",              (6.8,  6.8)),
    ("Gemini 8.1m / GMOS",            (5.5,  5.5)),
    ("ksirius",                        (3.7, 2.9)),
    ("Custom",                         None),
]

PRESET_MAP: dict[str, Optional[tuple[float, float]]] = {
    name: fov for name, fov in TELESCOPE_PRESETS
}
PRESET_NAMES = [name for name, _ in TELESCOPE_PRESETS]
DEFAULT_TELESCOPE = "OAO 188cm / MuSCAT"

ASSESS_COLORS = {
    "GOOD":     "#1a9641",
    "OK":       "#78c679",
    "MARGINAL": "#d9a800",
    "POOR":     "#e05a00",
    "BAD":      "#d7191c",
}


# ---------------------------------------------------------------------------
# Fixed-mode panel
# ---------------------------------------------------------------------------

class FixedModePanel(ttk.Frame):
    """Fixed RA/Dec target mode: query reference stars at one sky position."""

    def __init__(self, parent) -> None:
        super().__init__(parent)

        self._result_df: Optional[pd.DataFrame] = None
        self._summary:   Optional[dict] = None
        self._running    = False

        # FoV drag / optimize state
        self._all_df: Optional[pd.DataFrame] = None  # all stars in search circle, annotated
        self._query_center  = None   # SkyCoord of target
        self._query_params: Optional[dict] = None
        self._fov_cx = 0.0  # FoV centre x-offset from target (arcmin, rotated frame)
        self._fov_cy = 0.0  # FoV centre y-offset from target (arcmin, rotated frame)
        self._dragging       = False
        self._drag_anchor_x  = 0.0
        self._drag_anchor_y  = 0.0
        self._drag_fov_x0    = 0.0
        self._drag_fov_y0    = 0.0
        self._mpl_cids: list = []

        # ── Tkinter variables ──────────────────────────────────────────────
        self.var_name       = tk.StringVar()
        self.var_ra         = tk.StringVar()
        self.var_dec        = tk.StringVar()
        self.var_tele       = tk.StringVar(value=DEFAULT_TELESCOPE)
        self.var_w_arcmin   = tk.StringVar(value="6")
        self.var_w_arcsec   = tk.StringVar(value="6")
        self.var_h_arcmin   = tk.StringVar(value="6")
        self.var_h_arcsec   = tk.StringVar(value="6")
        self.var_pa         = tk.StringVar(value="0")
        self.var_catalog    = tk.StringVar(value="panstarrs")
        self.var_target_mag = tk.StringVar()
        self.var_delta_mag  = tk.StringVar(value="3.0")
        self.var_mag_err    = tk.StringVar(value="0.05")
        self.var_min_sep    = tk.StringVar(value="5")
        self.var_thr_good   = tk.StringVar(value="30")
        self.var_thr_ok     = tk.StringVar(value="10")
        self.var_thr_marg   = tk.StringVar(value="5")
        self._last_center   = None  # SkyCoord of last name-searched star

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_ui()
        self._on_telescope_change()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── PanedWindow (left input | right output) ─────────────────────────
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.grid(row=0, column=0, sticky="nsew")

        left_outer = ttk.Frame(paned, width=360)
        left_outer.columnconfigure(0, weight=1)
        left_outer.rowconfigure(0, weight=1)
        left_outer.pack_propagate(False)
        paned.add(left_outer, weight=0)

        left_canvas = tk.Canvas(left_outer, highlightthickness=0)
        vsb = ttk.Scrollbar(left_outer, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=vsb.set)
        left_canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        left = ttk.Frame(left_canvas, padding=(10, 10, 6, 10))
        _win_id = left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _on_canvas_resize(event):
            left_canvas.itemconfig(_win_id, width=event.width)
            self.lbl_status.config(wraplength=max(100, event.width - 20))
        left_canvas.bind("<Configure>", _on_canvas_resize)
        left.bind("<Configure>",
                  lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))

        def _on_scroll_fixed(event):
            try:
                lx, ly = left_outer.winfo_rootx(), left_outer.winfo_rooty()
                lw, lh = left_outer.winfo_width(), left_outer.winfo_height()
                if lx <= event.x_root < lx + lw and ly <= event.y_root < ly + lh:
                    left_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
            except Exception:
                pass
        self.bind_all("<MouseWheel>", _on_scroll_fixed, add="+")

        r = 0

        # 座標
        fc = ttk.LabelFrame(left, text="📍 座標 (ICRS)", padding=8)
        fc.grid(row=r, column=0, sticky="ew", pady=(0, 8)); r += 1
        fc.columnconfigure(1, weight=1)

        ttk.Label(fc, text="天体名").grid(row=0, column=0, sticky="e", padx=(0, 6))
        name_row = ttk.Frame(fc)
        name_row.grid(row=0, column=1, sticky="ew")
        name_row.columnconfigure(0, weight=1)
        self.ent_name = ttk.Entry(name_row, textvariable=self.var_name, width=17)
        self.ent_name.grid(row=0, column=0, sticky="ew")
        self.ent_name.bind("<Return>", lambda _: self._on_name_search())
        self.btn_name_search = ttk.Button(
            name_row, text="検索", width=5, command=self._on_name_search)
        self.btn_name_search.grid(row=0, column=1, padx=(4, 0))

        self.lbl_name_status = ttk.Label(
            fc, text='例: "GJ 1214"  "Vega"  "HD 189733"',
            foreground="#888888", font=("", 9))
        self.lbl_name_status.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 6))

        ttk.Separator(fc, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        ttk.Label(fc, text="RA").grid(row=3, column=0, sticky="e", padx=(0, 6))
        ttk.Entry(fc, textvariable=self.var_ra, width=24).grid(row=3, column=1, sticky="ew")
        ttk.Label(fc, text="Dec").grid(row=4, column=0, sticky="e", padx=(0, 6), pady=(4, 0))
        ttk.Entry(fc, textvariable=self.var_dec, width=24).grid(
            row=4, column=1, sticky="ew", pady=(4, 0))
        ttk.Label(fc, text='直接入力も可: "123.456" または "12:34:56.7"',
                  foreground="#888888", font=("", 9)).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # 視野 / FoV
        ff = ttk.LabelFrame(left, text="🔭 視野 / FoV", padding=8)
        ff.grid(row=r, column=0, sticky="ew", pady=(0, 8)); r += 1

        ttk.Label(ff, text="望遠鏡").grid(row=0, column=0, sticky="e", padx=(0, 6))
        self.cb_telescope = ttk.Combobox(
            ff, textvariable=self.var_tele,
            values=PRESET_NAMES, state="readonly", width=26)
        self.cb_telescope.grid(row=0, column=1, columnspan=5, sticky="ew")
        self.cb_telescope.bind("<<ComboboxSelected>>", lambda _: self._on_telescope_change())

        self.lbl_fov_info = ttk.Label(ff, text="", foreground="#555555", font=("", 9))
        self.lbl_fov_info.grid(row=1, column=0, columnspan=6, sticky="w", pady=(3, 6))

        ttk.Label(ff, text="幅 (W)").grid(row=2, column=0, sticky="e", padx=(0, 6))
        self.ent_w_arcmin = ttk.Entry(ff, textvariable=self.var_w_arcmin, width=5)
        self.ent_w_arcmin.grid(row=2, column=1)
        ttk.Label(ff, text="'", foreground="#666").grid(row=2, column=2, padx=(2, 8))
        self.ent_w_arcsec = ttk.Entry(ff, textvariable=self.var_w_arcsec, width=5)
        self.ent_w_arcsec.grid(row=2, column=3)
        ttk.Label(ff, text='"', foreground="#666").grid(row=2, column=4, padx=(2, 0))

        ttk.Label(ff, text="高さ (H)").grid(
            row=3, column=0, sticky="e", padx=(0, 6), pady=(4, 0))
        self.ent_h_arcmin = ttk.Entry(ff, textvariable=self.var_h_arcmin, width=5)
        self.ent_h_arcmin.grid(row=3, column=1, pady=(4, 0))
        ttk.Label(ff, text="'", foreground="#666").grid(row=3, column=2, padx=(2, 8))
        self.ent_h_arcsec = ttk.Entry(ff, textvariable=self.var_h_arcsec, width=5)
        self.ent_h_arcsec.grid(row=3, column=3, pady=(4, 0))
        ttk.Label(ff, text='"', foreground="#666").grid(row=3, column=4, padx=(2, 0))

        ttk.Label(ff, text="PA").grid(row=4, column=0, sticky="e", padx=(0, 6), pady=(4, 0))
        ttk.Entry(ff, textvariable=self.var_pa, width=7).grid(row=4, column=1, pady=(4, 0))
        ttk.Label(ff, text="°", foreground="#666").grid(row=4, column=2, padx=(2, 0))

        # カタログ
        fcat = ttk.LabelFrame(left, text="📚 カタログ", padding=8)
        fcat.grid(row=r, column=0, sticky="ew", pady=(0, 8)); r += 1

        for i, (label, val) in enumerate([
            ("Pan-STARRS (推奨)", "panstarrs"),
            ("Gaia DR3",          "gaia"),
            ("2MASS",             "2mass"),
            ("SIMBAD  ⚠",        "simbad"),
        ]):
            ttk.Radiobutton(fcat, text=label, variable=self.var_catalog, value=val).grid(
                row=i // 2, column=i % 2, sticky="w", padx=6, pady=1)
        ttk.Label(fcat, text="⚠ SIMBAD は測光較正用カタログではありません",
                  foreground="#888888", font=("", 9)).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # 等級条件
        fmag = ttk.LabelFrame(left, text="🌟 等級条件", padding=8)
        fmag.grid(row=r, column=0, sticky="ew", pady=(0, 8)); r += 1
        fmag.columnconfigure(1, weight=1)

        ttk.Label(fmag, text="目標等級").grid(row=0, column=0, sticky="e", padx=(0, 6))
        tmag_row = ttk.Frame(fmag)
        tmag_row.grid(row=0, column=1, sticky="ew")
        ttk.Entry(tmag_row, textvariable=self.var_target_mag, width=8).grid(row=0, column=0)
        self.lbl_target_band = ttk.Label(
            tmag_row, text="mag", foreground="#555555", font=("", 9))
        self.lbl_target_band.grid(row=0, column=1, padx=(4, 0))

        self.lbl_target_hint = ttk.Label(
            fmag, text="天体名検索後に自動入力されます",
            foreground="#888888", font=("", 9))
        self.lbl_target_hint.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 4))

        ttk.Label(fmag, text="± ΔMag").grid(row=2, column=0, sticky="e", padx=(0, 6))
        delta_row = ttk.Frame(fmag)
        delta_row.grid(row=2, column=1, sticky="w")
        ttk.Entry(delta_row, textvariable=self.var_delta_mag, width=6).grid(row=0, column=0)
        ttk.Label(delta_row, text="mag", foreground="#666").grid(row=0, column=1, padx=(4, 0))

        self.lbl_eff_range = ttk.Label(
            fmag, text="有効範囲: —", foreground="#888888", font=("", 9))
        self.lbl_eff_range.grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 6))

        ttk.Separator(fmag, orient="horizontal").grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        ttk.Label(fmag, text="等級誤差 ≤").grid(row=5, column=0, sticky="e", padx=(0, 6))
        ttk.Entry(fmag, textvariable=self.var_mag_err, width=7).grid(row=5, column=1, sticky="w")

        ttk.Label(fmag, text="最小離角").grid(
            row=6, column=0, sticky="e", padx=(0, 6), pady=(4, 0))
        sep_row = ttk.Frame(fmag)
        sep_row.grid(row=6, column=1, sticky="w", pady=(4, 0))
        ttk.Entry(sep_row, textvariable=self.var_min_sep, width=7).grid(row=0, column=0)
        ttk.Label(sep_row, text='"', foreground="#666").grid(row=0, column=1, padx=(2, 0))

        self.var_target_mag.trace_add("write", lambda *_: self._update_eff_range())
        self.var_delta_mag.trace_add("write",  lambda *_: self._update_eff_range())
        self.var_catalog.trace_add("write", lambda *_: self._on_catalog_change())

        # 判定閾値
        fthr = ttk.LabelFrame(left, text="📊 判定閾値", padding=8)
        fthr.grid(row=r, column=0, sticky="ew", pady=(0, 8)); r += 1

        for i, (label, var) in enumerate([
            ("GOOD ≥",     self.var_thr_good),
            ("OK ≥",       self.var_thr_ok),
            ("MARGINAL ≥", self.var_thr_marg),
        ]):
            col = (i % 2) * 2
            ttk.Label(fthr, text=label).grid(row=i // 2, column=col, sticky="e", padx=(0, 4))
            ttk.Entry(fthr, textvariable=var, width=5).grid(
                row=i // 2, column=col + 1, sticky="w", padx=(0, 12))

        # ── Buttons ────────────────────────────────────────────────────────
        fbtn = ttk.Frame(left)
        fbtn.grid(row=r, column=0, sticky="ew"); r += 1

        self.btn_run = ttk.Button(fbtn, text="▶  実行", command=self._on_run, width=12)
        self.btn_run.pack(side="left", padx=(0, 8))
        self.btn_optimize = ttk.Button(fbtn, text="🎯 最適化", command=self._on_optimize,
                                       state="disabled", width=12)
        self.btn_optimize.pack(side="left", padx=(0, 8))
        self.btn_save = ttk.Button(fbtn, text="💾 CSV 保存", command=self._on_save,
                                   state="disabled", width=12)
        self.btn_save.pack(side="left")

        self.lbl_status = ttk.Label(left, text="", font=("", 9), justify="left")
        self.lbl_status.grid(row=r, column=0, sticky="w", pady=(6, 0))

        # ── Right panel ─────────────────────────────────────────────────────
        right = ttk.Frame(paned, padding=(4, 10, 10, 10))
        paned.add(right, weight=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(7, 5.2))
        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(right)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        fsum = ttk.LabelFrame(right, text="📈 サマリ", padding=8)
        fsum.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        fsum.columnconfigure(0, weight=1)
        self.lbl_summary = ttk.Label(
            fsum, text="— まだ実行していません —",
            font=("", 10), wraplength=700, justify="left")
        self.lbl_summary.grid(row=0, column=0, sticky="w")

    # ── Telescope preset ──────────────────────────────────────────────────────

    def _on_telescope_change(self) -> None:
        name = self.var_tele.get()
        fov  = PRESET_MAP.get(name)

        if fov is None:
            for ent in (self.ent_w_arcmin, self.ent_w_arcsec,
                        self.ent_h_arcmin, self.ent_h_arcsec):
                ent.config(state="normal")
            self.lbl_fov_info.config(text="↑ 視野を直接入力してください")
        else:
            w, h = fov
            wm, ws = int(w), round((w - int(w)) * 60)
            hm, hs = int(h), round((h - int(h)) * 60)
            for var, val in [(self.var_w_arcmin, wm), (self.var_w_arcsec, ws),
                             (self.var_h_arcmin, hm), (self.var_h_arcsec, hs)]:
                var.set(str(val))
            for ent in (self.ent_w_arcmin, self.ent_w_arcsec,
                        self.ent_h_arcmin, self.ent_h_arcsec):
                ent.config(state="disabled")
            self.lbl_fov_info.config(
                text=f"視野: {w:.1f}′ × {h:.1f}′  ({w * 60:.0f}″ × {h * 60:.0f}″)")

        fov_w, fov_h = self._get_fov()
        self._draw_plot(df=None, fov_w=fov_w, fov_h=fov_h)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_fov(self) -> tuple[float, float]:
        try:
            wm = float(self.var_w_arcmin.get() or 0)
            ws = float(self.var_w_arcsec.get() or 0)
            hm = float(self.var_h_arcmin.get() or 0)
            hs = float(self.var_h_arcsec.get() or 0)
            return wm + ws / 60.0, hm + hs / 60.0
        except ValueError:
            return 6.0, 6.0

    def _get_float(self, var: tk.StringVar, default: float) -> float:
        try:
            return float(var.get())
        except ValueError:
            return default

    # ── Plot ──────────────────────────────────────────────────────────────────

    def _draw_plot(self, df: Optional[pd.DataFrame], fov_w: float, fov_h: float,
                   cx: float = 0.0, cy: float = 0.0) -> None:
        self.ax.clear()
        half_w = fov_w / 2
        half_h = fov_h / 2
        R = circumscribed_radius(fov_w, fov_h)

        # Search circle (query region)
        if self._all_df is not None:
            circ = plt.Circle((0, 0), R, fill=False, linestyle=":",
                               edgecolor="#999999", linewidth=1.2, alpha=0.7,
                               label=f"FoV移動範囲 (R={R:.1f}′)", zorder=1)
            self.ax.add_patch(circ)

        # FoV rectangle (at current offset)
        rect = mpatches.Rectangle(
            (cx - half_w, cy - half_h), fov_w, fov_h,
            linewidth=1.8, edgecolor="#2255aa", facecolor="#e8f0ff",
            linestyle="--", alpha=0.25, label="FoV boundary", zorder=2)
        self.ax.add_patch(rect)
        rect_border = mpatches.Rectangle(
            (cx - half_w, cy - half_h), fov_w, fov_h,
            linewidth=1.8, edgecolor="#2255aa", facecolor="none",
            linestyle="--", zorder=3)
        self.ax.add_patch(rect_border)

        # Target star (always at origin)
        in_fov = abs(cx) <= half_w and abs(cy) <= half_h
        self.ax.scatter([0], [0], s=250, c="gold", marker="*", zorder=7,
                        label="Target" + (" (FoV内)" if in_fov else " (FoV外)"),
                        edgecolors="#cc8800", linewidths=0.8)
        if not in_fov:
            self.ax.scatter([0], [0], s=350, c="none", marker="o",
                            edgecolors="red", linewidths=1.5, zorder=8)

        # Stars outside the current FoV (from full search cache)
        if self._all_df is not None and "x_arcmin" in self._all_df.columns:
            outside_mask = ((self._all_df["x_arcmin"] - cx).abs() > half_w) | \
                           ((self._all_df["y_arcmin"] - cy).abs() > half_h)
            outside = self._all_df[outside_mask]
            if not outside.empty:
                self.ax.scatter(
                    outside["x_arcmin"], outside["y_arcmin"],
                    s=25, c="none", marker="o", alpha=0.45,
                    zorder=3, label=f"FoV外  ({len(outside)})",
                    edgecolors="#cc4444", linewidths=0.8)

        n_usable = 0
        if df is not None and not df.empty and "x_arcmin" in df.columns:
            usable   = df[df["usable"]]
            rejected = df[~df["usable"]]
            n_usable = len(usable)

            if not usable.empty:
                mags  = usable["mag"].fillna(15.0).clip(lower=8, upper=22)
                sizes = np.clip(400 - mags * 16, 15, 400)
                self.ax.scatter(
                    usable["x_arcmin"], usable["y_arcmin"],
                    s=sizes, c="deepskyblue", marker="o", alpha=0.85,
                    zorder=5, label=f"Usable  ({len(usable)})",
                    edgecolors="steelblue", linewidths=0.6)

            if not rejected.empty:
                self.ax.scatter(
                    rejected["x_arcmin"], rejected["y_arcmin"],
                    s=40, c="tomato", marker="x", alpha=0.7,
                    zorder=4, label=f"Rejected  ({len(rejected)})",
                    linewidths=1.2)

        elif df is None:
            self.ax.text(0, 0, "RA / Dec を入力して\n[実行] を押してください",
                         ha="center", va="center", color="#888888",
                         fontsize=12, style="italic")

        # Axis limits: show full search circle + margin
        margin = R * 0.18 if self._all_df is not None else max(fov_w, fov_h) * 0.18
        lim = R + margin if self._all_df is not None else half_w + margin
        self.ax.set_xlim(lim, -lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("ΔRA  [arcmin]  (East →←)", fontsize=10)
        self.ax.set_ylabel("ΔDec  [arcmin]", fontsize=10)

        if df is not None and "usable" in df.columns:
            assessment = (self._summary or {}).get("assessment", "")
            color = ASSESS_COLORS.get(assessment, "black")
            offset_str = f"  offset ({cx:+.1f}′, {cy:+.1f}′)" if (cx != 0 or cy != 0) else ""
            self.ax.set_title(
                f"Reference Stars  [{assessment}]  ({n_usable} usable){offset_str}",
                fontsize=11, color=color, fontweight="bold")
        else:
            self.ax.set_title("Reference Star Preview  (FoVをドラッグして移動可)", fontsize=11)

        if self._all_df is not None:
            self.ax.text(0.01, 0.01,
                         "FoV をドラッグして移動  |  [🎯 最適化] で自動位置決め",
                         transform=self.ax.transAxes, fontsize=8,
                         color="#666666", va="bottom")

        self.ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
                       ncol=3, fontsize=9, framealpha=0.9)
        self.ax.grid(True, alpha=0.25, linestyle=":")
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    # ── Name search ───────────────────────────────────────────────────────────

    def _on_name_search(self) -> None:
        name = self.var_name.get().strip()
        if not name:
            return
        self.btn_name_search.config(state="disabled", text="検索中...")
        self.lbl_name_status.config(text="SIMBAD を検索中...", foreground="#888888")
        threading.Thread(target=self._do_name_search, args=(name,), daemon=True).start()

    def _do_name_search(self, name: str) -> None:
        try:
            from astroquery.simbad import Simbad
            simbad = Simbad()
            simbad.add_votable_fields("otype")
            simbad.TIMEOUT = 30
            result = simbad.query_object(name)

            if result is None or len(result) == 0:
                self.after(0, self._name_search_done, None,
                           f"'{name}' は SIMBAD で見つかりませんでした")
                return

            row     = result[0]
            ra_col  = "ra"  if "ra"  in result.colnames else "RA"
            dec_col = "dec" if "dec" in result.colnames else "DEC"
            ra_deg  = float(row[ra_col])
            dec_deg = float(row[dec_col])

            otype_col = "otype" if "otype" in result.colnames else (
                        "OTYPE" if "OTYPE" in result.colnames else None)
            otype = str(row[otype_col]).strip() if otype_col else ""

            msg = f"{name}  →  RA={ra_deg:.6f}°, Dec={dec_deg:+.6f}°"
            if otype and otype not in ("--", "nan", ""):
                msg += f"  [{otype}]"

            self.after(0, self._name_search_done, (ra_deg, dec_deg), msg)

        except Exception as exc:
            self.after(0, self._name_search_done, None, f"エラー: {exc}")

    def _name_search_done(self, coords: Optional[tuple[float, float]], message: str) -> None:
        self.btn_name_search.config(state="normal", text="検索")
        if coords is None:
            self.lbl_name_status.config(text=message, foreground="#cc0000")
        else:
            ra_deg, dec_deg = coords
            self.var_ra.set(f"{ra_deg:.7f}")
            self.var_dec.set(f"{dec_deg:.7f}")
            self.lbl_name_status.config(text=message, foreground="#1a7a1a")

            from astropy.coordinates import SkyCoord as _SC
            import astropy.units as _u
            center = _SC(ra=ra_deg * _u.deg, dec=dec_deg * _u.deg, frame="icrs")
            self._last_center = center
            self.lbl_target_band.config(text="取得中...")
            self.lbl_target_hint.config(
                text="カタログから目標等級を取得中...", foreground="#888888")
            threading.Thread(
                target=self._fetch_target_mag, args=(center,), daemon=True).start()

    # ── Target magnitude helpers ───────────────────────────────────────────────

    def _update_eff_range(self) -> None:
        try:
            t = float(self.var_target_mag.get())
            d = float(self.var_delta_mag.get())
            self.lbl_eff_range.config(
                text=f"有効範囲: {t - d:.1f} 〜 {t + d:.1f} mag", foreground="#1a5580")
        except ValueError:
            self.lbl_eff_range.config(
                text="有効範囲: 目標等級を入力してください", foreground="#888888")

    def _on_catalog_change(self) -> None:
        if self._last_center is not None:
            self.lbl_target_band.config(text="更新中...")
            self.lbl_target_hint.config(
                text="カタログ変更: 目標等級を再取得中...", foreground="#888888")
            threading.Thread(
                target=self._fetch_target_mag, args=(self._last_center,),
                daemon=True).start()

    def _fetch_target_mag(self, center) -> None:
        catalog    = self.var_catalog.get()
        band_label = BAND_LABELS.get(catalog, "")
        if not band_label:
            self.after(0, self._finish_target_mag, None, "")
            return
        mag = query_target_magnitude(center, catalog)
        self.after(0, self._finish_target_mag, mag, band_label)

    def _finish_target_mag(self, mag: Optional[float], band_label: str) -> None:
        if not band_label:
            self.lbl_target_band.config(text="mag")
            self.lbl_target_hint.config(
                text="SIMBAD モード: 目標等級を手動入力してください",
                foreground="#cc6600")
            return
        self.lbl_target_band.config(text=f"mag  ({band_label})")
        if mag is not None:
            self.var_target_mag.set(f"{mag:.2f}")
            self.lbl_target_hint.config(
                text=f"自動取得: {band_label} = {mag:.2f} mag", foreground="#1a7a1a")
        else:
            self.lbl_target_hint.config(
                text=f"{band_label} で見つかりませんでした。手動入力してください。",
                foreground="#cc6600")

    # ── Run query ─────────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        if self._running:
            return

        ra  = self.var_ra.get().strip()
        dec = self.var_dec.get().strip()
        if not ra or not dec:
            messagebox.showwarning("入力エラー", "RA と Dec を入力してください。")
            return

        target_mag_str = self.var_target_mag.get().strip()
        if not target_mag_str:
            messagebox.showwarning(
                "入力エラー",
                "目標等級を入力してください。\n天体名検索後に自動入力されます。")
            return
        try:
            target_mag = float(target_mag_str)
            delta_mag  = float(self.var_delta_mag.get() or "3.0")
        except ValueError:
            messagebox.showwarning("入力エラー", "目標等級・ΔMag の値が不正です。")
            return

        mag_min = target_mag - delta_mag
        mag_max = target_mag + delta_mag

        fov_w, fov_h = self._get_fov()
        if fov_w <= 0 or fov_h <= 0:
            messagebox.showwarning("入力エラー", "視野サイズを正しく入力してください。")
            return

        params = {
            "ra": ra, "dec": dec,
            "catalog":  self.var_catalog.get(),
            "mag_min":  mag_min,
            "mag_max":  mag_max,
            "max_err":  self._get_float(self.var_mag_err,  0.05),
            "min_sep":  self._get_float(self.var_min_sep,  5.0),
            "thr_good": int(self._get_float(self.var_thr_good, 30)),
            "thr_ok":   int(self._get_float(self.var_thr_ok,   10)),
            "thr_marg": int(self._get_float(self.var_thr_marg,  5)),
            "fov_w": fov_w, "fov_h": fov_h,
            "pa_deg": self._get_float(self.var_pa, 0.0),
        }

        self._running = True
        self.btn_run.config(state="disabled", text="⏳ 検索中...")
        self.btn_save.config(state="disabled")
        self.lbl_status.config(text="カタログへ問い合わせ中... (数秒〜1分)", foreground="#888888")
        self.lbl_summary.config(text="— 処理中 —", foreground="black")

        threading.Thread(target=self._run_query, args=(params,), daemon=True).start()

    def _annotate_stars(self, raw_df: pd.DataFrame, center, pa_deg: float) -> pd.DataFrame:
        """Project all stars to PA-rotated tangent plane and add separation column."""
        import astropy.units as _u
        from astropy.coordinates import SkyCoord as _SC
        stars = _SC(ra=raw_df["ra_deg"].values * _u.deg,
                    dec=raw_df["dec_deg"].values * _u.deg, frame="icrs")
        dra  = ((stars.ra - center.ra).wrap_at(180 * _u.deg).deg
                * np.cos(np.deg2rad(center.dec.deg)))
        ddec = (stars.dec - center.dec).deg
        dra_am, ddec_am = dra * 60.0, ddec * 60.0
        pa_rad = np.deg2rad(pa_deg)
        c, s = np.cos(pa_rad), np.sin(pa_rad)
        df = raw_df.copy()
        df["x_arcmin"] = dra_am * c - ddec_am * s
        df["y_arcmin"] = dra_am * s + ddec_am * c
        return add_separation(df, center)

    def _filter_at_offset(self, cx: float, cy: float) -> pd.DataFrame:
        """Return quality-filtered df for FoV centred at (cx, cy) offset in arcmin."""
        fov_w, fov_h = self._get_fov()
        p = self._query_params
        df = self._all_df
        inside = ((df["x_arcmin"] - cx).abs() <= fov_w / 2) & \
                 ((df["y_arcmin"] - cy).abs() <= fov_h / 2)
        return apply_quality_filters(
            df[inside].copy(), p["catalog"], p["mag_min"], p["mag_max"],
            p["max_err"], p["min_sep"])

    def _run_query(self, params: dict) -> None:
        try:
            center = parse_coord(params["ra"], params["dec"])
        except Exception as exc:
            self.after(0, self._finish_error, f"座標解析失敗: {exc}")
            return

        catalog  = params["catalog"]
        mag_min  = params["mag_min"]
        mag_max  = params["mag_max"]
        fov_w    = params["fov_w"]
        fov_h    = params["fov_h"]
        pa_deg   = params["pa_deg"]
        max_err  = params["max_err"]
        min_sep  = params["min_sep"]
        thr_good = params["thr_good"]
        thr_ok   = params["thr_ok"]
        thr_marg = params["thr_marg"]

        # Query the full diagonal so stars remain in coverage when FoV is shifted
        # to place the target at the corner (max FoV-corner distance = 2 × R_circ).
        radius = 2 * circumscribed_radius(fov_w, fov_h)

        try:
            if catalog == "simbad":
                raw_df = query_simbad(center, radius)
            else:
                raw_df = query_vizier_catalog(center, radius, catalog, mag_min, mag_max)
        except Exception as exc:
            self.after(0, self._finish_error, f"クエリ失敗: {exc}")
            return

        n_raw = len(raw_df)

        if raw_df.empty:
            empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
            summ  = summarize(empty, 0, 0, thr_good, thr_ok, thr_marg)
            self.after(0, self._finish_ok, empty, summ, fov_w, fov_h, center, params, None)
            return

        # Annotate ALL stars in the search circle (x_arcmin, y_arcmin, separation)
        all_df = self._annotate_stars(raw_df, center, pa_deg)

        # Initial FoV filter at centre (0, 0)
        fov_df      = all_df[(all_df["x_arcmin"].abs() <= fov_w / 2) &
                             (all_df["y_arcmin"].abs() <= fov_h / 2)].copy()
        n_fov       = len(fov_df)
        filtered_df = apply_quality_filters(fov_df, catalog, mag_min, mag_max, max_err, min_sep)
        summ        = summarize(filtered_df, n_raw, n_fov, thr_good, thr_ok, thr_marg)

        self.after(0, self._finish_ok, filtered_df, summ, fov_w, fov_h, center, params, all_df)

    def _finish_error(self, msg: str) -> None:
        self._running = False
        self.btn_run.config(state="normal", text="▶  実行")
        self.lbl_status.config(text=f"エラー: {msg}", foreground="#cc0000")

    def _finish_ok(self, df: pd.DataFrame, summary: dict,
                   fov_w: float, fov_h: float,
                   center=None, params: Optional[dict] = None,
                   all_df: Optional[pd.DataFrame] = None) -> None:
        self._result_df    = df
        self._summary      = summary
        self._running      = False
        self._all_df       = all_df
        self._query_center = center
        self._query_params = params
        self._fov_cx       = 0.0
        self._fov_cy       = 0.0

        self.btn_run.config(state="normal", text="▶  実行")
        self.btn_save.config(state="normal")
        drag_ok = all_df is not None and not all_df.empty
        self.btn_optimize.config(state="normal" if drag_ok else "disabled")
        self.lbl_status.config(
            text="完了  — FoVをドラッグして移動できます" if drag_ok else "完了",
            foreground="#1a7a1a")

        self._draw_plot(df, fov_w, fov_h, cx=0.0, cy=0.0)
        self._update_summary(summary)
        if drag_ok:
            self._connect_drag()

    def _update_summary(self, summary: dict) -> None:
        assessment    = summary.get("assessment", "?")
        n_usable      = summary.get("n_usable", 0)
        reject_counts = summary.get("reject_counts", {})
        cx, cy = self._fov_cx, self._fov_cy
        offset_str = (f"  |  FoV中心 ({cx:+.2f}′, {cy:+.2f}′)"
                      if (cx != 0 or cy != 0) else "")
        parts = [
            f"Raw (外接円): {summary.get('n_raw', 0)}",
            f"FoV 内: {summary.get('n_fov', 0)}",
            f"近傍除外: {summary.get('n_rejected_near_asteroid', 0)}",
            f"使用可能: {n_usable} 星",
            f"判定: {assessment}{offset_str}",
        ]
        if reject_counts:
            detail = "  ".join(
                f"{r}: {c}"
                for r, c in sorted(reject_counts.items(), key=lambda x: -x[1]))
            parts.append(f"除外理由 → {detail}")
        color = ASSESS_COLORS.get(assessment, "#000000")
        self.lbl_summary.config(text="     ".join(parts), foreground=color)

    # ── FoV drag ─────────────────────────────────────────────────────────────

    def _connect_drag(self) -> None:
        for cid in self._mpl_cids:
            self.canvas.mpl_disconnect(cid)
        self._mpl_cids = [
            self.canvas.mpl_connect("button_press_event",   self._on_drag_start),
            self.canvas.mpl_connect("motion_notify_event",  self._on_drag_motion),
            self.canvas.mpl_connect("button_release_event", self._on_drag_end),
        ]

    def _on_drag_start(self, event) -> None:
        if (event.inaxes != self.ax or event.button != 1
                or self._all_df is None
                or getattr(self.toolbar, "mode", "") != ""):
            return
        fov_w, fov_h = self._get_fov()
        cx, cy = self._fov_cx, self._fov_cy
        if (abs(event.xdata - cx) <= fov_w / 2 and
                abs(event.ydata - cy) <= fov_h / 2):
            self._dragging      = True
            self._drag_anchor_x = event.xdata
            self._drag_anchor_y = event.ydata
            self._drag_fov_x0   = cx
            self._drag_fov_y0   = cy

    @staticmethod
    def _clamp_fov_center(cx: float, cy: float,
                          half_w: float, half_h: float, R: float) -> tuple[float, float]:
        """Clamp (cx, cy) so target stays in FoV and centre stays within coverage circle."""
        cx = float(np.clip(cx, -half_w, half_w))
        cy = float(np.clip(cy, -half_h, half_h))
        dist = np.hypot(cx, cy)
        if dist > R:
            cx *= R / dist
            cy *= R / dist
        return cx, cy

    def _on_drag_motion(self, event) -> None:
        if not self._dragging or event.inaxes != self.ax or event.xdata is None:
            return
        fov_w, fov_h = self._get_fov()
        R = circumscribed_radius(fov_w, fov_h)
        raw_cx = self._drag_fov_x0 + (event.xdata - self._drag_anchor_x)
        raw_cy = self._drag_fov_y0 + (event.ydata - self._drag_anchor_y)
        new_cx, new_cy = self._clamp_fov_center(raw_cx, raw_cy, fov_w / 2, fov_h / 2, R)
        self._fov_cx = new_cx
        self._fov_cy = new_cy
        filtered = self._filter_at_offset(new_cx, new_cy)
        self._draw_plot(filtered, fov_w, fov_h, cx=new_cx, cy=new_cy)

    def _on_drag_end(self, event) -> None:
        if not self._dragging:
            return
        self._dragging  = False
        cx, cy = self._fov_cx, self._fov_cy
        fov_w, fov_h = self._get_fov()
        filtered = self._filter_at_offset(cx, cy)
        self._result_df = filtered
        p = self._query_params
        thr_good = p["thr_good"]
        thr_ok   = p["thr_ok"]
        thr_marg = p["thr_marg"]
        n_fov = int(((filtered["x_arcmin"].abs() <= fov_w / 2) &
                     (filtered["y_arcmin"].abs() <= fov_h / 2)).sum())
        summ = summarize(filtered, len(self._all_df), n_fov, thr_good, thr_ok, thr_marg)
        self._summary = summ
        self._update_summary(summ)

    # ── Optimize ─────────────────────────────────────────────────────────────

    def _on_optimize(self) -> None:
        if self._all_df is None:
            return
        self.btn_optimize.config(state="disabled", text="⏳ 最適化中...")
        self.btn_run.config(state="disabled")
        threading.Thread(target=self._do_optimize, daemon=True).start()

    def _do_optimize(self) -> None:
        fov_w, fov_h = self._get_fov()
        R = circumscribed_radius(fov_w, fov_h)
        n_grid = 40
        xs = np.linspace(-R, R, n_grid)
        ys = np.linspace(-R, R, n_grid)
        best_cx, best_cy, best_n = 0.0, 0.0, -1
        df = self._all_df
        half_w, half_h = fov_w / 2, fov_h / 2
        p = self._query_params

        for cx in xs:
            for cy in ys:
                # target must stay in FoV AND centre within coverage circle
                if abs(cx) > half_w or abs(cy) > half_h:
                    continue
                if cx ** 2 + cy ** 2 > R ** 2:
                    continue
                inside = ((df["x_arcmin"] - cx).abs() <= half_w) & \
                         ((df["y_arcmin"] - cy).abs() <= half_h)
                sub = df[inside]
                if sub.empty:
                    continue
                filtered = apply_quality_filters(
                    sub.copy(), p["catalog"], p["mag_min"], p["mag_max"],
                    p["max_err"], p["min_sep"])
                n = int(filtered["usable"].sum())
                if n > best_n:
                    best_n, best_cx, best_cy = n, cx, cy

        self.after(0, self._finish_optimize, best_cx, best_cy)

    def _finish_optimize(self, cx: float, cy: float) -> None:
        self._fov_cx = cx
        self._fov_cy = cy
        fov_w, fov_h = self._get_fov()
        filtered = self._filter_at_offset(cx, cy)
        self._result_df = filtered
        p = self._query_params
        n_fov = len(filtered)
        summ = summarize(filtered, len(self._all_df), n_fov,
                         p["thr_good"], p["thr_ok"], p["thr_marg"])
        self._summary = summ
        self._draw_plot(filtered, fov_w, fov_h, cx=cx, cy=cy)
        self._update_summary(summ)
        self.btn_optimize.config(state="normal", text="🎯 最適化")
        self.btn_run.config(state="normal")

    # ── Save ─────────────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        if self._result_df is None or self._result_df.empty:
            messagebox.showinfo("保存", "データがありません。先に実行してください。")
            return
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV ファイル", "*.csv"), ("すべてのファイル", "*.*")],
            initialfile=f"refs_{ts}.csv")
        if not path:
            return
        df = self._result_df.copy()
        for col in OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df["time"] = ""
        df[OUTPUT_COLUMNS].to_csv(path, index=False)
        self.lbl_status.config(text=f"保存: {path}", foreground="#0055cc")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    root.title("refstar_planner — 固定モード")
    root.minsize(1100, 700)
    panel = FixedModePanel(root)
    panel.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
