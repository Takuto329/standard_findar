#!/usr/bin/env python3
"""
refstar_gui_variable.py — Variable-mode panel for refstar_planner GUI.

Asteroid mode: queries Horizons API for ephemeris over a time range,
then checks reference star availability at each epoch via VizieR
for five photometric bands simultaneously:
  optical : Pan-STARRS g, i
  NIR     : 2MASS J, H, Ks

Solar (neutral-reflection) colour offsets derived from Mamajek (2022)
G2V row and Jester et al. (2005):
  g = V + 0.286   i = V − 0.300
  J = V − 1.198   H = V − 1.491   Ks = V − 1.564
"""

from __future__ import annotations

import datetime
import sys
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
    add_separation,
    apply_quality_filters,
    circumscribed_radius,
    filter_rectangular_fov,
    parse_coord,
    summarize,
)

# ---------------------------------------------------------------------------
# Multi-band configuration
# ---------------------------------------------------------------------------

# Solar colour offsets: band_mag = V + offset  (neutral reflection = G2V colours)
# g, i: from Jester (2005) + Mamajek (2022) G2V g−r=0.476 + SDSS solar i
# J, H, Ks: Mamajek (2022) G2V: V−Ks=1.564, H−Ks=0.073, J−H=0.293
SOLAR_OFFSETS: dict[str, float] = {
    "g":  +0.286,
    "i":  -0.300,
    "J":  -1.198,
    "H":  -1.491,
    "Ks": -1.564,
}
BANDS = ["g", "i", "J", "H", "Ks"]
BAND_COLORS = {
    "g":  "#2166ac",
    "i":  "#4dac26",
    "J":  "#f4a441",
    "H":  "#d6604d",
    "Ks": "#9b59b6",
}
BAND_LINESTYLE = {"g": "-", "i": "-", "J": "--", "H": "--", "Ks": ":"}
# (wide_df column name for mag, column name for mag_err)
BAND_MAG_COLS: dict[str, tuple[str, str]] = {
    "g":  ("gmag",  "e_gmag"),
    "i":  ("imag",  "e_imag"),
    "J":  ("Jmag",  "e_Jmag"),
    "H":  ("Hmag",  "e_Hmag"),
    "Ks": ("Kmag",  "e_Kmag"),
}

# ---------------------------------------------------------------------------
# Observatory database  (code, lat_deg, lon_deg, alt_m, name_en, name_ja)
# ---------------------------------------------------------------------------

_OBS_DATA: list[tuple] = [
    ("W84",  34.576,  133.594,  372,
     "Okayama Astrophys. Obs. OAO (188cm)",    "岡山天体物理観測所 OAO"),
    ("D74",  34.690,  133.540,  386,
     "Bisei Spaceguard Center",                 "美星スペースガードセンター"),
    ("T16",  35.026,  134.337,  449,
     "Nishi-Harima Astro. Obs. (Nayuta 2m)",   "西はりま天文台 (Nayuta)"),
    ("Z26",  34.577,  133.594,  373,
     "Seimei Telescope 3.8m (Kyoto Univ.)",    "京大 Seimei 3.8m"),
    ("381",  35.944,  138.472, 1350,
     "Nobeyama Radio Obs.",                     "野辺山電波観測所"),
    ("372",  35.674,  139.539,   58,
     "NAOJ Mitaka",                             "国立天文台 三鷹"),
    ("568",  19.826, -155.472, 4213,
     "Mauna Kea (Subaru / Keck / CFHT)",       "マウナケア天文台群"),
    ("T09",  20.707, -156.258, 3052,
     "Haleakala (MuSCAT3 / Pan-STARRS)",       "ハレアカラ天文台"),
    ("F65",  20.707, -156.258, 3065,
     "Haleakala (Pan-STARRS 1)",               "ハレアカラ PS1"),
    ("695",  31.963, -111.600, 2064,
     "Kitt Peak National Obs. (KPNO)",         "キットピーク"),
    ("675",  33.356, -116.864, 1706,
     "Palomar Mountain Observatory",           "パロマー天文台"),
    ("291",  31.681, -110.879, 2350,
     "F.L. Whipple Obs. (FLWO / SAO)",        "フレッドロレンスウィップル"),
    ("G96",  32.443, -110.789, 2791,
     "Mt. Lemmon Survey (CSS)",               "マウント・レモン"),
    ("309", -24.625,  -70.403, 2635,
     "Cerro Paranal (VLT / ESO)",             "パラナル (VLT)"),
    ("304", -29.257,  -70.730, 2347,
     "La Silla (ESO / NTT / HARPS)",         "ラ・シヤ (ESO)"),
    ("807", -30.169,  -70.806, 2215,
     "Cerro Tololo (CTIO / DECam)",           "セロ・トロロ (CTIO)"),
    ("I11", -30.240,  -70.736, 2722,
     "Gemini South",                          "ジェミニ南"),
    ("950",  28.760,  -17.890, 2326,
     "La Palma (Roque de los Muchachos)",     "ラ・パルマ"),
    ("J04",  28.754,  -17.889, 2370,
     "Telescopio Nazionale Galileo (TNG)",   "TNG 3.6m"),
    ("413", -31.273,  149.071, 1165,
     "Siding Spring Obs. (AAT)",             "サイディングスプリング"),
    ("074", -32.380,   20.811, 1760,
     "SAAO Sutherland",                      "南アフリカ天文台"),
    ("500",   0.000,    0.000,    0,
     "Geocenter (no parallax correction)",   "地心 (視差補正なし)"),
]

_OBS_BY_CODE: dict[str, tuple] = {row[0]: row for row in _OBS_DATA}


def _search_obs(query: str, max_results: int = 10) -> list[str]:
    q = query.strip().lower()
    if not q:
        return [f"{r[0]}  {r[4]}" for r in _OBS_DATA[:max_results]]
    out = []
    for r in _OBS_DATA:
        code, _, _, _, name_en, name_ja = r
        if q in code.lower() or q in name_en.lower() or q in name_ja:
            out.append(f"{code}  {name_en}")
            if len(out) >= max_results:
                break
    return out


# ---------------------------------------------------------------------------
# Telescope presets
# ---------------------------------------------------------------------------

from refstar_gui_fixed import (
    TELESCOPE_PRESETS, PRESET_MAP, PRESET_NAMES, DEFAULT_TELESCOPE, ASSESS_COLORS
)

# ---------------------------------------------------------------------------
# Wide VizieR query helpers (module-level, called from background thread)
# ---------------------------------------------------------------------------

def _resolve_col(tbl: pd.DataFrame, name: str) -> Optional[str]:
    if name in tbl.columns:
        return name
    lm = {c.lower(): c for c in tbl.columns}
    return lm.get(name.lower())


def _query_wide_ps(center, radius_arcmin: float,
                   mag_min: float, mag_max: float) -> pd.DataFrame:
    """Query Pan-STARRS DR2 for g and i bands in one VizieR call."""
    try:
        from astroquery.vizier import Vizier
        from astropy import units as u
    except ImportError:
        return pd.DataFrame()

    viz = Vizier(
        columns=["RAJ2000", "DEJ2000", "objID", "gmag", "e_gmag", "imag", "e_imag"],
        column_filters={"gmag": f"{mag_min:.2f}..{mag_max:.2f}"},
        row_limit=-1,
    )
    viz.TIMEOUT = 120
    try:
        result = viz.query_region(center, radius=radius_arcmin * u.arcmin,
                                  catalog="II/349/ps1")
    except Exception as exc:
        print(f"[WARN] PS wide query: {exc}", file=sys.stderr)
        return pd.DataFrame()

    if not result or len(result) == 0:
        return pd.DataFrame()

    tbl = result[0].to_pandas()
    ra_c = _resolve_col(tbl, "RAJ2000")
    dc_c = _resolve_col(tbl, "DEJ2000")
    if ra_c is None or dc_c is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["ra_deg"]  = pd.to_numeric(tbl[ra_c], errors="coerce")
    out["dec_deg"] = pd.to_numeric(tbl[dc_c], errors="coerce")
    sid = _resolve_col(tbl, "objID")
    out["source_id"] = (tbl[sid].astype(str) if sid else
                        [f"ps_{i}" for i in range(len(tbl))])
    out["catalog"] = "panstarrs"
    out["object_type"] = ""
    out["color"] = np.nan

    for band_col, err_col in [("gmag", "e_gmag"), ("imag", "e_imag")]:
        bc = _resolve_col(tbl, band_col)
        ec = _resolve_col(tbl, err_col)
        out[band_col] = pd.to_numeric(tbl[bc], errors="coerce") if bc else np.nan
        out[err_col]  = pd.to_numeric(tbl[ec], errors="coerce") if ec else np.nan

    return out.dropna(subset=["ra_deg", "dec_deg"]).reset_index(drop=True)


def _query_wide_2m(center, radius_arcmin: float,
                   mag_min: float, mag_max: float) -> pd.DataFrame:
    """Query 2MASS PSC for J, H, Ks bands in one VizieR call."""
    try:
        from astroquery.vizier import Vizier
        from astropy import units as u
    except ImportError:
        return pd.DataFrame()

    viz = Vizier(
        columns=["RAJ2000", "DEJ2000", "2MASS",
                 "Jmag", "e_Jmag", "Hmag", "e_Hmag", "Kmag", "e_Kmag"],
        column_filters={"Jmag": f"{mag_min:.2f}..{mag_max:.2f}"},
        row_limit=-1,
    )
    viz.TIMEOUT = 120
    try:
        result = viz.query_region(center, radius=radius_arcmin * u.arcmin,
                                  catalog="II/246/out")
    except Exception as exc:
        print(f"[WARN] 2MASS wide query: {exc}", file=sys.stderr)
        return pd.DataFrame()

    if not result or len(result) == 0:
        return pd.DataFrame()

    tbl = result[0].to_pandas()
    ra_c = _resolve_col(tbl, "RAJ2000")
    dc_c = _resolve_col(tbl, "DEJ2000")
    if ra_c is None or dc_c is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["ra_deg"]  = pd.to_numeric(tbl[ra_c], errors="coerce")
    out["dec_deg"] = pd.to_numeric(tbl[dc_c], errors="coerce")
    sid = _resolve_col(tbl, "2MASS")
    out["source_id"] = (tbl[sid].astype(str) if sid else
                        [f"2m_{i}" for i in range(len(tbl))])
    out["catalog"] = "2mass"
    out["object_type"] = ""
    out["color"] = np.nan

    for band_col, err_col in [("Jmag", "e_Jmag"), ("Hmag", "e_Hmag"), ("Kmag", "e_Kmag")]:
        bc = _resolve_col(tbl, band_col)
        ec = _resolve_col(tbl, err_col)
        out[band_col] = pd.to_numeric(tbl[bc], errors="coerce") if bc else np.nan
        out[err_col]  = pd.to_numeric(tbl[ec], errors="coerce") if ec else np.nan

    return out.dropna(subset=["ra_deg", "dec_deg"]).reset_index(drop=True)


def _band_view(wide_df: pd.DataFrame, band: str) -> pd.DataFrame:
    """Return a copy of wide_df with 'mag'/'mag_err' set to the requested band."""
    mag_col, err_col = BAND_MAG_COLS[band]
    if wide_df is None or wide_df.empty or mag_col not in wide_df.columns:
        return pd.DataFrame()
    df = wide_df.copy()
    df["mag"]     = df[mag_col]
    df["mag_err"] = df[err_col] if err_col in df.columns else np.nan
    return df


# ---------------------------------------------------------------------------
# Variable-mode panel
# ---------------------------------------------------------------------------

class VariableModePanel(ttk.Frame):
    """Asteroid tracking mode: Horizons ephemeris + multi-band reference star check."""

    def __init__(self, parent) -> None:
        super().__init__(parent)

        self._running   = False
        self._ephem: Optional[pd.DataFrame] = None
        self._cache: list[dict] = []
        self._display_indices: list[int] = []          # filtered view into _cache
        self._wide_ps: Optional[pd.DataFrame] = None
        self._wide_2m: Optional[pd.DataFrame] = None
        self._cursor_line = None
        self._step_idx    = 0

        self.var_target     = tk.StringVar()
        self.var_obs_search = tk.StringVar()
        self.var_mpc_code   = tk.StringVar(value="568")
        self.var_lat        = tk.StringVar(value="19.826")
        self.var_lon        = tk.StringVar(value="-155.472")
        self.var_alt        = tk.StringVar(value="4213")
        self.var_start      = tk.StringVar(value="2026-07-01 20:00")
        self.var_end        = tk.StringVar(value="2026-07-04 06:00")
        self.var_step_val   = tk.StringVar(value="1")
        self.var_step_unit  = tk.StringVar(value="h")
        self.var_tele       = tk.StringVar(value=DEFAULT_TELESCOPE)
        self.var_w_arcmin   = tk.StringVar(value="6")
        self.var_w_arcsec   = tk.StringVar(value="6")
        self.var_h_arcmin   = tk.StringVar(value="6")
        self.var_h_arcsec   = tk.StringVar(value="6")
        self.var_pa         = tk.StringVar(value="0")
        self.var_delta_mag    = tk.StringVar(value="3.0")
        self.var_filter_start = tk.StringVar()
        self.var_filter_end   = tk.StringVar()
        self.var_tz_jst       = tk.BooleanVar(value=False)
        self.var_mag_err    = tk.StringVar(value="0.05")
        self.var_min_sep    = tk.StringVar(value="5")
        self.var_thr_good   = tk.StringVar(value="30")
        self.var_thr_ok     = tk.StringVar(value="10")
        self.var_thr_marg   = tk.StringVar(value="5")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_ui()
        self._on_telescope_change()

        self.var_obs_search.trace_add("write", lambda *_: self._on_obs_search_change())

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
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
                  lambda e: left_canvas.configure(
                      scrollregion=left_canvas.bbox("all")))

        def _on_scroll_var(event):
            try:
                lx, ly = left_outer.winfo_rootx(), left_outer.winfo_rooty()
                lw, lh = left_outer.winfo_width(), left_outer.winfo_height()
                if lx <= event.x_root < lx + lw and ly <= event.y_root < ly + lh:
                    left_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
            except Exception:
                pass
        self.bind_all("<MouseWheel>", _on_scroll_var, add="+")

        r = 0

        # ── 小惑星 ──────────────────────────────────────────────────────────
        fast = ttk.LabelFrame(left, text="☄  小惑星", padding=8)
        fast.grid(row=r, column=0, sticky="ew", pady=(0, 8)); r += 1
        fast.columnconfigure(1, weight=1)

        ttk.Label(fast, text="Horizons 名称").grid(
            row=0, column=0, sticky="e", padx=(0, 6))
        tgt_row = ttk.Frame(fast)
        tgt_row.grid(row=0, column=1, sticky="ew")
        tgt_row.columnconfigure(0, weight=1)
        self.cb_target = ttk.Combobox(tgt_row, textvariable=self.var_target,
                                      state="normal", width=16)
        self.cb_target.grid(row=0, column=0, sticky="ew")
        self.cb_target.bind("<<ComboboxSelected>>", self._on_target_selected)
        self._cand_map: dict[str, str] = {}
        self.btn_verify = ttk.Button(tgt_row, text="確認", width=5,
                                     command=self._on_verify_target)
        self.btn_verify.grid(row=0, column=1, padx=(4, 0))

        self.lbl_target_status = ttk.Label(
            fast, text='例: "Ceres"  "99942"  "2026 BU"',
            foreground="#888888", font=("", 9))
        self.lbl_target_status.grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))

        # ── 観測地 ──────────────────────────────────────────────────────────
        floc = ttk.LabelFrame(left, text="📡 観測地", padding=8)
        floc.grid(row=r, column=0, sticky="ew", pady=(0, 8)); r += 1
        floc.columnconfigure(1, weight=1)

        ttk.Label(floc, text="観測地検索").grid(
            row=0, column=0, sticky="e", padx=(0, 6))
        self.cb_obs = ttk.Combobox(
            floc, textvariable=self.var_obs_search,
            values=_search_obs(""), state="normal", width=22)
        self.cb_obs.grid(row=0, column=1, sticky="ew")
        self.cb_obs.bind("<<ComboboxSelected>>", self._on_obs_selected)

        ttk.Label(floc, text='天文台名・MPC コードで検索',
                  foreground="#888888", font=("", 9)).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(2, 4))

        ttk.Separator(floc, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        ttk.Label(floc, text="MPC コード").grid(
            row=3, column=0, sticky="e", padx=(0, 6))
        mpc_row = ttk.Frame(floc)
        mpc_row.grid(row=3, column=1, sticky="ew")
        ttk.Entry(mpc_row, textvariable=self.var_mpc_code, width=6).grid(
            row=0, column=0)
        ttk.Label(mpc_row, text="(未入力なら下の緯度経度を使用)",
                  foreground="#888888", font=("", 9)).grid(
            row=0, column=1, padx=(6, 0))

        ttk.Label(floc, text="緯度").grid(
            row=4, column=0, sticky="e", padx=(0, 6), pady=(4, 0))
        loc_row = ttk.Frame(floc)
        loc_row.grid(row=4, column=1, sticky="ew", pady=(4, 0))
        ttk.Entry(loc_row, textvariable=self.var_lat, width=9).grid(row=0, column=0)
        ttk.Label(loc_row, text="°N").grid(row=0, column=1, padx=(2, 10))
        ttk.Label(loc_row, text="経度").grid(row=0, column=2)
        ttk.Entry(loc_row, textvariable=self.var_lon, width=9).grid(
            row=0, column=3, padx=(4, 0))
        ttk.Label(loc_row, text="°E").grid(row=0, column=4, padx=(2, 0))

        ttk.Label(floc, text="標高").grid(
            row=5, column=0, sticky="e", padx=(0, 6), pady=(4, 0))
        alt_row = ttk.Frame(floc)
        alt_row.grid(row=5, column=1, sticky="w", pady=(4, 0))
        ttk.Entry(alt_row, textvariable=self.var_alt, width=7).grid(row=0, column=0)
        ttk.Label(alt_row, text="m", foreground="#666").grid(
            row=0, column=1, padx=(4, 0))

        # ── 観測期間 ─────────────────────────────────────────────────────────
        _ftime_lbl = ttk.Frame(left)
        ttk.Label(_ftime_lbl, text="⏱ 観測期間").pack(side="left")
        ttk.Checkbutton(
            _ftime_lbl, text="JST", variable=self.var_tz_jst,
            command=self._on_tz_toggle,
        ).pack(side="left", padx=(6, 0))
        self.ftime = ttk.LabelFrame(left, labelwidget=_ftime_lbl, padding=8)
        ftime = self.ftime
        ftime.grid(row=r, column=0, sticky="ew", pady=(0, 8)); r += 1
        ftime.columnconfigure(1, weight=1)

        ttk.Label(ftime, text="開始").grid(row=0, column=0, sticky="e", padx=(0, 6))
        ttk.Entry(ftime, textvariable=self.var_start, width=18).grid(
            row=0, column=1, sticky="ew")

        ttk.Label(ftime, text="終了").grid(
            row=1, column=0, sticky="e", padx=(0, 6), pady=(4, 0))
        ttk.Entry(ftime, textvariable=self.var_end, width=18).grid(
            row=1, column=1, sticky="ew", pady=(4, 0))

        ttk.Label(ftime, text="ステップ").grid(
            row=2, column=0, sticky="e", padx=(0, 6), pady=(4, 0))
        step_row = ttk.Frame(ftime)
        step_row.grid(row=2, column=1, sticky="w", pady=(4, 0))
        ttk.Entry(step_row, textvariable=self.var_step_val, width=5).grid(
            row=0, column=0)
        ttk.Combobox(
            step_row, textvariable=self.var_step_unit,
            values=["m", "h", "d"], state="readonly", width=4,
        ).grid(row=0, column=1, padx=(4, 0))
        ttk.Label(step_row, text="(m=分 h=時間 d=日)",
                  foreground="#888888", font=("", 9)).grid(
            row=0, column=2, padx=(6, 0))

        self.lbl_step_info = ttk.Label(
            ftime, text="", foreground="#555555", font=("", 9))
        self.lbl_step_info.grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        for v in (self.var_start, self.var_end, self.var_step_val, self.var_step_unit):
            v.trace_add("write", lambda *_: self._update_step_info())

        # ── 視野 / FoV ───────────────────────────────────────────────────────
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

        # ── 等級条件 (中性反射を仮定して全バンドに同一条件を適用) ──────────────
        fmag = ttk.LabelFrame(left, text="🌟 等級条件 (全バンド共通)", padding=8)
        fmag.grid(row=r, column=0, sticky="ew", pady=(0, 8)); r += 1
        fmag.columnconfigure(1, weight=1)

        # V mag info label (populated from Horizons)
        self.lbl_vmag_range = ttk.Label(
            fmag, text="小惑星 V 等級: — (Horizons から自動表示)",
            foreground="#888888", font=("", 9))
        self.lbl_vmag_range.grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        # Band offsets info
        ttk.Label(fmag,
                  text="中性反射仮定:  g=V+0.29  i=V−0.30  J=V−1.20  H=V−1.49  Ks=V−1.56",
                  foreground="#666666", font=("", 8)).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 6))

        ttk.Separator(fmag, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        ttk.Label(fmag, text="± ΔMag").grid(row=3, column=0, sticky="e", padx=(0, 6))
        dmag_row = ttk.Frame(fmag)
        dmag_row.grid(row=3, column=1, sticky="w")
        ttk.Entry(dmag_row, textvariable=self.var_delta_mag, width=7).grid(row=0, column=0)
        ttk.Label(dmag_row, text="mag  (各バンド中心 ±)", foreground="#666",
                  font=("", 9)).grid(row=0, column=1, padx=(4, 0))

        ttk.Separator(fmag, orient="horizontal").grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(6, 6))

        ttk.Label(fmag, text="等級誤差 ≤").grid(row=5, column=0, sticky="e", padx=(0, 6))
        ttk.Entry(fmag, textvariable=self.var_mag_err, width=7).grid(
            row=5, column=1, sticky="w")

        ttk.Label(fmag, text="最小離角").grid(
            row=6, column=0, sticky="e", padx=(0, 6), pady=(4, 0))
        sep_row = ttk.Frame(fmag)
        sep_row.grid(row=6, column=1, sticky="w", pady=(4, 0))
        ttk.Entry(sep_row, textvariable=self.var_min_sep, width=7).grid(row=0, column=0)
        ttk.Label(sep_row, text='"', foreground="#666").grid(
            row=0, column=1, padx=(2, 0))

        # ── 判定閾値 ─────────────────────────────────────────────────────────
        fthr = ttk.LabelFrame(left, text="📊 判定閾値", padding=8)
        fthr.grid(row=r, column=0, sticky="ew", pady=(0, 8)); r += 1

        for i, (label, var) in enumerate([
            ("GOOD ≥",     self.var_thr_good),
            ("OK ≥",       self.var_thr_ok),
            ("MARGINAL ≥", self.var_thr_marg),
        ]):
            col = (i % 2) * 2
            ttk.Label(fthr, text=label).grid(
                row=i // 2, column=col, sticky="e", padx=(0, 4))
            ttk.Entry(fthr, textvariable=var, width=5).grid(
                row=i // 2, column=col + 1, sticky="w", padx=(0, 12))

        # ── Buttons ──────────────────────────────────────────────────────────
        fbtn = ttk.Frame(left)
        fbtn.grid(row=r, column=0, sticky="ew"); r += 1

        self.btn_run = ttk.Button(
            fbtn, text="▶  計算", command=self._on_run, width=12)
        self.btn_run.pack(side="left", padx=(0, 8))
        self.btn_save = ttk.Button(
            fbtn, text="💾 CSV 保存", command=self._on_save,
            state="disabled", width=12)
        self.btn_save.pack(side="left")

        self.lbl_status = ttk.Label(left, text="", font=("", 9), wraplength=320,
                                    justify="left")
        self.lbl_status.grid(row=r, column=0, sticky="w", pady=(6, 0))

        # ── Right panel ──────────────────────────────────────────────────────
        right = ttk.Frame(paned, padding=(4, 10, 10, 10))
        paned.add(right, weight=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        # Figure: top=time graph (full width), bottom=5 FoV panels
        self.fig = plt.figure(figsize=(12, 8), layout="constrained")
        gs = self.fig.add_gridspec(2, 5, height_ratios=[1, 2])
        self.ax_time = self.fig.add_subplot(gs[0, :])
        self.ax_bands: dict[str, plt.Axes] = {
            band: self.fig.add_subplot(gs[1, i])
            for i, band in enumerate(BANDS)
        }

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", rowspan=3)

        # ── Time slider + navigation ─────────────────────────────────────────
        slider_frame = ttk.Frame(right)
        slider_frame.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        slider_frame.columnconfigure(2, weight=1)

        self.btn_prev = ttk.Button(slider_frame, text="◀", width=3,
                                   command=self._on_prev, state="disabled")
        self.btn_prev.grid(row=0, column=0, padx=(0, 4))

        self.btn_next = ttk.Button(slider_frame, text="▶", width=3,
                                   command=self._on_next, state="disabled")
        self.btn_next.grid(row=0, column=1, padx=(0, 8))

        self.lbl_cur_time = ttk.Label(
            slider_frame, text="—", font=("", 9), foreground="#333")
        self.lbl_cur_time.grid(row=0, column=2, sticky="w")

        ttk.Label(slider_frame, text="← → キーでも移動可",
                  foreground="#999", font=("", 8)).grid(
            row=0, column=3, sticky="e", padx=(8, 0))

        self.slider = ttk.Scale(
            slider_frame, from_=0, to=1, orient="horizontal",
            command=self._on_slider)
        self.slider.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(2, 0))
        self.slider.state(["disabled"])

        # ── Display filter ───────────────────────────────────────────────────
        ffilter = ttk.LabelFrame(right, text="🔍 表示期間フィルタ", padding=(6, 4))
        ffilter.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        ffilter.columnconfigure(1, weight=1)
        ffilter.columnconfigure(3, weight=1)

        ttk.Label(ffilter, text="開始").grid(row=0, column=0, padx=(0, 4))
        ttk.Entry(ffilter, textvariable=self.var_filter_start,
                  width=16).grid(row=0, column=1, sticky="ew")
        ttk.Label(ffilter, text="〜  終了").grid(row=0, column=2, padx=(8, 4))
        ttk.Entry(ffilter, textvariable=self.var_filter_end,
                  width=16).grid(row=0, column=3, sticky="ew")
        ttk.Button(ffilter, text="適用", width=5,
                   command=self._on_filter_apply).grid(row=0, column=4, padx=(8, 0))
        ttk.Button(ffilter, text="リセット", width=7,
                   command=self._on_filter_reset).grid(row=0, column=5, padx=(4, 0))
        ttk.Label(ffilter, text="形式: YYYY-MM-DD HH:MM (空欄=制限なし)",
                  foreground="#999", font=("", 8)).grid(
            row=1, column=0, columnspan=6, sticky="w", pady=(2, 0))

        # Navigation toolbar
        toolbar_frame = ttk.Frame(right)
        toolbar_frame.grid(row=5, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Summary
        fsum = ttk.LabelFrame(right, text="📈 サマリ (現在のエポック)", padding=8)
        fsum.grid(row=6, column=0, sticky="ew", pady=(8, 0))
        fsum.columnconfigure(0, weight=1)
        _txt_bg = ttk.Style().lookup("TFrame", "background") or "systemWindowBackgroundColor"
        self.txt_summary = tk.Text(
            fsum, height=4, font=("Menlo", 10), relief="flat",
            background=_txt_bg, state="disabled", cursor="arrow", wrap="word",
            borderwidth=0)
        self.txt_summary.grid(row=0, column=0, sticky="ew")
        self.txt_summary.tag_configure("colored", foreground="#333333")
        self.txt_summary.tag_configure("hint", foreground="#aaaaaa")

        self._draw_empty_plots()

        # Arrow key navigation (skip if focus is on a text-input widget)
        def _on_key(event):
            if isinstance(self.focus_get(), (ttk.Entry, tk.Entry, ttk.Combobox)):
                return
            if event.keysym == "Left":
                self._on_prev()
            elif event.keysym == "Right":
                self._on_next()
        self.bind_all("<Left>",  _on_key, add="+")
        self.bind_all("<Right>", _on_key, add="+")

    # ── Observatory search ────────────────────────────────────────────────────

    def _on_obs_search_change(self) -> None:
        q = self.var_obs_search.get()
        matches = _search_obs(q)
        self.cb_obs["values"] = matches

    def _on_obs_selected(self, _event=None) -> None:
        sel = self.var_obs_search.get()
        code = sel.split()[0] if sel else ""
        if code in _OBS_BY_CODE:
            _, lat, lon, alt, name_en, _ = _OBS_BY_CODE[code]
            self.var_mpc_code.set(code)
            self.var_lat.set(str(lat))
            self.var_lon.set(str(lon))
            self.var_alt.set(str(alt))

    # ── Step-count estimate ───────────────────────────────────────────────────

    def _update_step_info(self) -> None:
        try:
            start = datetime.datetime.strptime(self.var_start.get().strip(), "%Y-%m-%d %H:%M")
            end   = datetime.datetime.strptime(self.var_end.get().strip(),   "%Y-%m-%d %H:%M")
            val   = float(self.var_step_val.get())
            unit  = self.var_step_unit.get()
            step_min = val * (1 if unit == "m" else 60 if unit == "h" else 1440)
            n = max(1, int((end - start).total_seconds() / 60 / step_min) + 1)
            self.lbl_step_info.config(
                text=f"計: {n} エポック  (推定 1〜3 分)",
                foreground="#555555")
        except (ValueError, ZeroDivisionError):
            self.lbl_step_info.config(text="日付形式: YYYY-MM-DD HH:MM", foreground="#888888")

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

    def _horizons_location(self) -> object:
        mpc = self.var_mpc_code.get().strip()
        try:
            lat = float(self.var_lat.get())
            lon = float(self.var_lon.get())
            alt = float(self.var_alt.get())
            return {"lat": lat, "lon": lon, "elevation": alt / 1000.0}
        except ValueError:
            if mpc:
                return mpc
            raise ValueError("観測地を設定してください（MPC コードまたは緯度経度）")

    # ── Target verify ─────────────────────────────────────────────────────────

    def _on_verify_target(self) -> None:
        name = self.var_target.get().strip()
        if not name:
            messagebox.showwarning("入力エラー", "Horizons 名称を入力してください。")
            return
        self.btn_verify.config(state="disabled", text="確認中...")
        self.lbl_target_status.config(text="Horizons に接続中...", foreground="#888888")
        threading.Thread(target=self._do_verify_target, args=(name,), daemon=True).start()

    def _do_verify_target(self, name: str) -> None:
        try:
            from astroquery.jplhorizons import Horizons
            loc = self._horizons_location()
            obj = Horizons(id=name, location=loc,
                           epochs={"start": "2000-01-01", "stop": "2000-01-01 01:00",
                                   "step": "1h"})
            eph = obj.ephemerides(quantities="1")
            label = str(eph["targetname"][0]) if "targetname" in eph.colnames else name
            self.after(0, self._verify_done, True, f"確認: {label}")
        except Exception as exc:
            msg = str(exc)
            if "ambiguous" in msg.lower():
                candidates = self._parse_ambiguous_candidates(msg)
                try:
                    import re as _re
                    from astroquery.jplhorizons import Horizons as _H
                    _loc = self._horizons_location()
                    _eph = _H(id=name, id_type="smallbody", location=_loc,
                              epochs={"start": "2000-01-01",
                                      "stop": "2000-01-01 01:00",
                                      "step": "1h"}).ephemerides(quantities="1")
                    tname = str(_eph["targetname"][0]) if "targetname" in _eph.colnames else ""
                    if tname:
                        m = _re.match(r"^(\d+)\s+", tname)
                        sb_id = m.group(1) if m else name
                        if sb_id not in {c[0] for c in candidates}:
                            candidates.append((sb_id, tname))
                except Exception:
                    pass
                if candidates:
                    self.after(0, self._verify_done, False, "")
                    self.after(0, self._show_candidate_picker, candidates)
                    return
            self.after(0, self._verify_done, False, f"エラー: {exc}")

    @staticmethod
    def _parse_ambiguous_candidates(msg: str) -> list[tuple[str, str]]:
        import re
        candidates = []
        past_dashes = False
        for line in msg.split("\n"):
            stripped = line.strip()
            if stripped and re.match(r"[-]+(\s+[-]+)+$", stripped):
                past_dashes = True
                continue
            if not past_dashes or not stripped:
                continue
            m = re.match(r"\s*(-?\d+)\s+(.+)", line)
            if m:
                id_str = m.group(1).strip()
                name = re.sub(r"\s{3,}\S.*$", "", m.group(2)).strip()
                if id_str and name:
                    candidates.append((id_str, name))
        return candidates

    def _show_candidate_picker(self, candidates: list[tuple[str, str]]) -> None:
        self._cand_map = {f"{id_str}   {name}": id_str for id_str, name in candidates}
        self.cb_target["values"] = list(self._cand_map.keys())
        self.lbl_target_status.config(
            text="複数候補が見つかりました。▼から選択してください",
            foreground="#cc6600")
        self.cb_target.event_generate("<Down>")

    def _on_target_selected(self, _event=None) -> None:
        display = self.cb_target.get()
        id_str = self._cand_map.get(display, display)
        self.var_target.set(id_str)
        self.cb_target["values"] = []
        self._cand_map = {}
        self._on_verify_target()

    def _verify_done(self, ok: bool, msg: str) -> None:
        self.btn_verify.config(state="normal", text="確認")
        color = "#1a7a1a" if ok else "#cc0000"
        if msg:
            self.lbl_target_status.config(text=msg, foreground=color)

    # ── Run ───────────────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        if self._running:
            return

        target = self.var_target.get().strip()
        if not target:
            messagebox.showwarning("入力エラー", "Horizons 名称を入力してください。")
            return

        try:
            loc = self._horizons_location()
        except ValueError as exc:
            messagebox.showwarning("入力エラー", str(exc))
            return

        start = self.var_start.get().strip()
        end   = self.var_end.get().strip()
        try:
            datetime.datetime.strptime(start, "%Y-%m-%d %H:%M")
            datetime.datetime.strptime(end,   "%Y-%m-%d %H:%M")
        except ValueError:
            messagebox.showwarning("入力エラー", "日時形式が不正です。\n形式: YYYY-MM-DD HH:MM")
            return
        start = self._input_to_utc(start)
        end   = self._input_to_utc(end)

        step = self.var_step_val.get().strip() + self.var_step_unit.get().strip()

        fov_w, fov_h = self._get_fov()
        if fov_w <= 0 or fov_h <= 0:
            messagebox.showwarning("入力エラー", "視野サイズを正しく入力してください。")
            return

        params = {
            "target":   target,
            "loc":      loc,
            "start":    start,
            "end":      end,
            "step":     step,
            "delta_mag": self._get_float(self.var_delta_mag, 3.0),
            "max_err":  self._get_float(self.var_mag_err,  0.05),
            "min_sep":  self._get_float(self.var_min_sep,  5.0),
            "thr_good": int(self._get_float(self.var_thr_good, 30)),
            "thr_ok":   int(self._get_float(self.var_thr_ok,   10)),
            "thr_marg": int(self._get_float(self.var_thr_marg,  5)),
            "fov_w":    fov_w,
            "fov_h":    fov_h,
            "pa_deg":   self._get_float(self.var_pa, 0.0),
        }

        self._running = True
        self._cache   = []
        self._ephem   = None
        self._wide_ps = None
        self._wide_2m = None
        self.btn_run.config(state="disabled", text="⏳ 計算中...")
        self.btn_save.config(state="disabled")
        self.slider.state(["disabled"])
        self.lbl_status.config(
            text="Step 1/3: Horizons から軌道暦を取得中...", foreground="#888888")

        threading.Thread(target=self._run_all, args=(params,), daemon=True).start()

    # ── Background computation ─────────────────────────────────────────────────

    def _run_all(self, params: dict) -> None:
        try:
            self._run_horizons(params)
        except Exception as exc:
            self.after(0, self._finish_error, f"Horizons エラー: {exc}")

    def _run_horizons(self, params: dict) -> None:
        from astroquery.jplhorizons import Horizons

        obj = Horizons(
            id=params["target"],
            location=params["loc"],
            epochs={"start": params["start"], "stop": params["end"],
                    "step":  params["step"]},
        )
        eph = obj.ephemerides(quantities="1,9")
        df  = eph.to_pandas()
        df.columns = [c.lower() for c in df.columns]

        required = {"ra", "dec"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"Horizons 結果に RA/Dec 列がありません: {list(df.columns)}")

        df["ra"]  = pd.to_numeric(df["ra"],  errors="coerce")
        df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
        df["v"]   = pd.to_numeric(df["v"],   errors="coerce") if "v" in df.columns else np.nan

        if "datetime_str" in df.columns:
            df["time_label"] = df["datetime_str"].astype(str)
        else:
            df["time_label"] = [f"t{i}" for i in range(len(df))]

        self._ephem = df.dropna(subset=["ra", "dec"]).reset_index(drop=True)
        n = len(self._ephem)

        v_min = self._ephem["v"].min()
        v_max = self._ephem["v"].max()
        self.after(0, self._show_vmag_range, v_min, v_max)
        self.after(0, lambda: self.lbl_status.config(
            text=f"Step 2/3: VizieR を広域クエリ中 ({n} エポック)...",
            foreground="#888888"))

        try:
            self._run_wide_query(params)
        except Exception as exc:
            self.after(0, self._finish_error, f"VizieR エラー: {exc}")

    def _run_wide_query(self, params: dict) -> None:
        from astropy.coordinates import SkyCoord
        from astropy import units as u

        eph   = self._ephem
        fov_w = params["fov_w"]
        fov_h = params["fov_h"]

        ra_vals  = eph["ra"].values
        dec_vals = eph["dec"].values
        mean_ra  = float(np.mean(ra_vals))
        mean_dec = float(np.mean(dec_vals))
        centroid = SkyCoord(ra=mean_ra * u.deg, dec=mean_dec * u.deg, frame="icrs")

        stars   = SkyCoord(ra=ra_vals * u.deg, dec=dec_vals * u.deg, frame="icrs")
        max_sep = centroid.separation(stars).to(u.arcmin).max().value

        fov_r   = circumscribed_radius(fov_w, fov_h)
        query_r = max_sep + fov_r + 2.0

        delta   = params["delta_mag"]
        v_vals  = self._ephem["v"].dropna()
        if len(v_vals) > 0:
            v_mid   = float(v_vals.mean())
            v_range = float(v_vals.max() - v_vals.min())
        else:
            v_mid   = 15.0   # fallback when Horizons has no V mag
            v_range = 0.0

        buf = v_range / 2 + 0.5   # extra margin covering V variation + edge buffer

        g_cen  = v_mid + SOLAR_OFFSETS["g"]
        j_cen  = v_mid + SOLAR_OFFSETS["J"]
        self._wide_ps = _query_wide_ps(centroid, query_r,
                                        g_cen - delta - buf, g_cen + delta + buf)
        self._wide_2m = _query_wide_2m(centroid, query_r,
                                        j_cen - delta - buf, j_cen + delta + buf)

        n_ps = len(self._wide_ps) if self._wide_ps is not None else 0
        n_2m = len(self._wide_2m) if self._wide_2m is not None else 0
        delta = params["delta_mag"]
        self.after(0, lambda: self.lbl_status.config(
            text=f"Step 3/3: {len(eph)} エポック × 5 バンドをフィルタ中... "
                 f"(±{delta} mag, PS={n_ps}星, 2MASS={n_2m}星)",
            foreground="#888888"))

        try:
            self._run_filter_all(params)
        except Exception as exc:
            self.after(0, self._finish_error, f"フィルタ失敗: {exc}")

    def _run_filter_all(self, params: dict) -> None:
        from astropy.coordinates import SkyCoord
        from astropy import units as u

        eph      = self._ephem
        fov_w     = params["fov_w"]
        fov_h     = params["fov_h"]
        pa_deg    = params["pa_deg"]
        delta_mag = params["delta_mag"]
        max_err   = params["max_err"]

        v_vals = self._ephem["v"].dropna()
        v_fallback = float(v_vals.mean()) if len(v_vals) > 0 else 15.0
        min_sep  = params["min_sep"]
        thr_good = params["thr_good"]
        thr_ok   = params["thr_ok"]
        thr_marg = params["thr_marg"]

        # Pre-compute band-specific wide DataFrames (set mag/mag_err columns)
        wide: dict[str, pd.DataFrame] = {}
        for band in BANDS:
            src = self._wide_ps if band in ("g", "i") else self._wide_2m
            wide[band] = _band_view(src, band)

        cache = []
        n = len(eph)

        for i, row in enumerate(eph.itertuples(index=False)):
            center = SkyCoord(ra=row.ra * u.deg, dec=row.dec * u.deg, frame="icrs")
            v_mag  = float(row.v) if not np.isnan(row.v) else None

            v = v_mag if v_mag is not None else v_fallback

            bands_data: dict[str, dict] = {}
            for band in BANDS:
                wdf     = wide[band]
                cat     = "panstarrs" if band in ("g", "i") else "2mass"
                b_cen   = v + SOLAR_OFFSETS[band]
                mag_min = b_cen - delta_mag
                mag_max = b_cen + delta_mag

                if wdf.empty:
                    fov_df   = pd.DataFrame()
                    filtered = pd.DataFrame()
                    summ     = {"n_usable": 0, "assessment": "BAD",
                                "n_fov": 0, "n_rejected_near_asteroid": 0,
                                "reject_counts": {}}
                else:
                    fov_df = filter_rectangular_fov(wdf, center, fov_w, fov_h, pa_deg)
                    if not fov_df.empty:
                        fov_df   = add_separation(fov_df, center)
                        filtered = apply_quality_filters(
                            fov_df, cat, mag_min, mag_max, max_err, min_sep)
                    else:
                        filtered = fov_df
                    n_raw = len(fov_df)
                    summ  = summarize(filtered, n_raw, n_raw,
                                      thr_good, thr_ok, thr_marg)

                bands_data[band] = {
                    "df":         filtered,
                    "summary":    summ,
                    "n_usable":   summ.get("n_usable", 0),
                    "assessment": summ.get("assessment", "BAD"),
                }

            cache.append({
                "time":    row.time_label,
                "ra":      row.ra,
                "dec":     row.dec,
                "v_mag":   v_mag,
                "bands":   bands_data,
            })

            if (i + 1) % max(1, n // 20) == 0 or i == n - 1:
                pct = int((i + 1) / n * 100)
                self.after(0, lambda p=pct: self.lbl_status.config(
                    text=f"フィルタ処理中... {p}%", foreground="#888888"))

        self._cache = cache
        self.after(0, self._finish_ok)

    # ── Callbacks (main thread) ────────────────────────────────────────────────

    def _show_vmag_range(self, v_min: float, v_max: float) -> None:
        if np.isnan(v_min):
            self.lbl_vmag_range.config(
                text="小惑星 V 等級: なし", foreground="#cc6600")
        else:
            self.lbl_vmag_range.config(
                text=f"小惑星 V 等級: {v_min:.1f} 〜 {v_max:.1f} mag (Horizons 取得値)",
                foreground="#1a5580")

    def _finish_error(self, msg: str) -> None:
        self._running = False
        self.btn_run.config(state="normal", text="▶  計算")
        self.lbl_status.config(text=f"エラー: {msg}", foreground="#cc0000")

    def _finish_ok(self) -> None:
        self._running = False
        self.btn_run.config(state="normal", text="▶  計算")
        self.btn_save.config(state="normal")
        self.lbl_status.config(
            text=f"完了 — {len(self._cache)} エポック × 5 バンド計算済み",
            foreground="#1a7a1a")

        n = len(self._cache)
        if n == 0:
            return

        self._display_indices = list(range(n))
        self.var_filter_start.set("")
        self.var_filter_end.set("")
        self._step_idx = 0

        self.slider.config(to=n - 1)
        self.slider.set(0)
        self.slider.state(["!disabled"])
        self.btn_prev.config(state="normal")
        self.btn_next.config(state="normal")

        self._draw_time_graph()
        self._update_epoch_view(0)

    # ── Time slider ───────────────────────────────────────────────────────────

    def _on_slider(self, value: str) -> None:
        idx = int(float(value))
        if idx != self._step_idx and self._display_indices:
            self._step_idx = idx
            self._update_cursor(idx)
            self._update_epoch_view(idx)

    def _on_prev(self, _event=None) -> None:
        if not self._display_indices:
            return
        new_idx = max(0, self._step_idx - 1)
        if new_idx != self._step_idx:
            self._step_idx = new_idx
            self.slider.set(new_idx)
            self._update_cursor(new_idx)
            self._update_epoch_view(new_idx)

    def _on_next(self, _event=None) -> None:
        if not self._display_indices:
            return
        new_idx = min(len(self._display_indices) - 1, self._step_idx + 1)
        if new_idx != self._step_idx:
            self._step_idx = new_idx
            self.slider.set(new_idx)
            self._update_cursor(new_idx)
            self._update_epoch_view(new_idx)

    def _update_epoch_view(self, idx: int) -> None:
        if not self._display_indices or idx >= len(self._display_indices):
            return
        entry = self._cache[self._display_indices[idx]]
        self.lbl_cur_time.config(text=self._fmt_label(entry["time"]))
        self._draw_star_fields(entry)
        self._update_summary(entry)

    def _update_cursor(self, idx: int) -> None:
        if self._cursor_line is not None:
            self._cursor_line.set_xdata([idx])
            self.canvas.draw_idle()

    # ── Timezone helpers ──────────────────────────────────────────────────────

    _JST_OFFSET = datetime.timedelta(hours=9)

    def _input_to_utc(self, val: str) -> str:
        """Convert an input time string to UTC. Subtracts 9h when JST mode is on."""
        if not self.var_tz_jst.get():
            return val
        dt = datetime.datetime.strptime(val.strip(), "%Y-%m-%d %H:%M")
        return (dt - self._JST_OFFSET).strftime("%Y-%m-%d %H:%M")

    def _fmt_label(self, time_label: str) -> str:
        """Return time_label converted to JST if the toggle is on, else as-is."""
        if not self.var_tz_jst.get():
            return time_label
        dt = self._parse_epoch_dt(time_label)
        if dt is None:
            return time_label
        dt_jst = dt + self._JST_OFFSET
        return dt_jst.strftime("%Y-%b-%d %H:%M JST")

    def _on_tz_toggle(self) -> None:
        jst_on = self.var_tz_jst.get()
        delta = self._JST_OFFSET if jst_on else -self._JST_OFFSET
        for var in (self.var_start, self.var_end):
            val = var.get().strip()
            try:
                dt = datetime.datetime.strptime(val, "%Y-%m-%d %H:%M")
                var.set((dt + delta).strftime("%Y-%m-%d %H:%M"))
            except ValueError:
                pass
        if self._cache:
            self._draw_time_graph()
            if self._display_indices:
                self._update_epoch_view(self._step_idx)

    # ── Filter helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_epoch_dt(time_label: str) -> "Optional[datetime.datetime]":
        """Parse a Horizons time label string to a datetime for filtering."""
        import re
        s = re.sub(r"^A\.D\.\s*", "", time_label.strip())
        s = re.sub(r"\s+TDB.*$", "", s)
        s = re.sub(r"\.\d+$", "", s)
        for fmt in ("%Y-%b-%d %H:%M:%S", "%Y-%b-%d %H:%M",
                    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.datetime.strptime(s, fmt)
            except ValueError:
                pass
        return None

    def _on_filter_apply(self) -> None:
        if not self._cache:
            return
        start_str = self.var_filter_start.get().strip()
        end_str   = self.var_filter_end.get().strip()
        try:
            start_dt = (datetime.datetime.strptime(start_str, "%Y-%m-%d %H:%M")
                        if start_str else None)
            end_dt   = (datetime.datetime.strptime(end_str,   "%Y-%m-%d %H:%M")
                        if end_str else None)
        except ValueError:
            messagebox.showwarning("フィルタエラー", "日時形式: YYYY-MM-DD HH:MM")
            return

        indices = []
        for i, entry in enumerate(self._cache):
            dt = self._parse_epoch_dt(entry["time"])
            if dt is None:
                indices.append(i)
                continue
            if start_dt and dt < start_dt:
                continue
            if end_dt and dt > end_dt:
                continue
            indices.append(i)

        if not indices:
            messagebox.showinfo("フィルタ", "該当するエポックがありません。\nフィルタをリセットします。")
            self._on_filter_reset()
            return

        self._display_indices = indices
        self._step_idx = 0
        self.slider.config(to=max(0, len(indices) - 1))
        self.slider.set(0)
        self._draw_time_graph()
        self._update_epoch_view(0)

    def _on_filter_reset(self) -> None:
        if not self._cache:
            return
        self._display_indices = list(range(len(self._cache)))
        self.var_filter_start.set("")
        self.var_filter_end.set("")
        self._step_idx = 0
        self.slider.config(to=max(0, len(self._cache) - 1))
        self.slider.set(0)
        self._draw_time_graph()
        self._update_epoch_view(0)

    # ── Twilight helpers ─────────────────────────────────────────────────────

    def _sun_altitudes(self, time_labels: list[str]) -> "np.ndarray | None":
        try:
            import re
            from datetime import datetime
            from astropy.coordinates import get_sun, AltAz, EarthLocation
            from astropy.time import Time
            import astropy.units as u
            lat = float(self.var_lat.get())
            lon = float(self.var_lon.get())
            alt = float(self.var_alt.get())
            location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)

            def _parse(s: str):
                s = re.sub(r"^A\.D\.\s*", "", s.strip())
                s = re.sub(r"\s+TDB.*$", "", s)
                s = re.sub(r"\.\d+$", "", s)
                for fmt in ("%Y-%b-%d %H:%M:%S", "%Y-%b-%d %H:%M",
                            "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                    try:
                        return datetime.strptime(s, fmt)
                    except ValueError:
                        pass
                return None

            dts = [_parse(t) for t in time_labels]
            if any(d is None for d in dts):
                return None
            times = Time(dts, format='datetime', scale='utc')
            frame = AltAz(obstime=times, location=location)
            sun   = get_sun(times).transform_to(frame)
            return sun.alt.deg
        except Exception:
            return None

    @staticmethod
    def _shade_daytime(ax: plt.Axes, is_day: "np.ndarray") -> None:
        n = len(is_day)
        in_day = False
        start  = 0
        for i, day in enumerate(is_day):
            if day and not in_day:
                start  = i
                in_day = True
            elif not day and in_day:
                ax.axvspan(start - 0.5, i - 0.5,
                           color="#aaaaaa", alpha=0.22, zorder=0.5, linewidth=0)
                in_day = False
        if in_day:
            ax.axvspan(start - 0.5, n - 0.5,
                       color="#aaaaaa", alpha=0.22, zorder=0.5, linewidth=0)

    # ── Time graph ───────────────────────────────────────────────────────────

    def _draw_time_graph(self) -> None:
        if not self._display_indices:
            return
        cache = [self._cache[i] for i in self._display_indices]

        thr_good = self._get_float(self.var_thr_good, 30)
        thr_ok   = self._get_float(self.var_thr_ok,   10)
        thr_marg = self._get_float(self.var_thr_marg,  5)
        x        = list(range(len(cache)))

        self.ax_time.clear()

        # Twilight shading
        labels   = [e["time"] for e in cache]
        sun_alts = self._sun_altitudes(labels)
        if sun_alts is not None:
            self._shade_daytime(self.ax_time, sun_alts > -18)

        # Threshold bands and lines
        all_n = [e["bands"][b]["n_usable"] for e in cache for b in BANDS]
        ymax  = max(max(all_n) * 1.15 if all_n else 1, thr_good * 1.2, 5)
        self.ax_time.axhspan(thr_good, ymax,     alpha=0.07, color="#1a9641", zorder=0)
        self.ax_time.axhspan(thr_ok,   thr_good, alpha=0.07, color="#78c679", zorder=0)
        self.ax_time.axhspan(thr_marg, thr_ok,   alpha=0.07, color="#d9a800", zorder=0)
        self.ax_time.axhspan(0,        thr_marg, alpha=0.07, color="#d7191c", zorder=0)

        for y, lbl, col in [(thr_good, "GOOD",     "#1a9641"),
                             (thr_ok,   "OK",       "#78c679"),
                             (thr_marg, "MARGINAL", "#d9a800")]:
            self.ax_time.axhline(y=y, color=col, linewidth=0.8,
                                 linestyle="--", alpha=0.7, zorder=1)
            self.ax_time.text(len(x) - 0.5, y + 0.4, lbl,
                              color=col, fontsize=7, va="bottom", ha="right")

        # One line per band
        for band in BANDS:
            n_list = [e["bands"][band]["n_usable"] for e in cache]
            self.ax_time.plot(x, n_list,
                              color=BAND_COLORS[band],
                              linestyle=BAND_LINESTYLE[band],
                              linewidth=1.6, label=band, zorder=3)

        self.ax_time.legend(loc="upper left", fontsize=8, ncol=5, framealpha=0.85)

        # X-axis ticks
        step = max(1, len(x) // 8)
        display_labels = [self._fmt_label(lb) for lb in labels]
        self.ax_time.set_xticks(x[::step])
        self.ax_time.set_xticklabels(display_labels[::step], rotation=25, ha="right", fontsize=7)
        self.ax_time.set_xlim(-0.5, len(x) - 0.5)
        self.ax_time.set_ylim(0, ymax)
        self.ax_time.set_ylabel("使用可能参照星数", fontsize=9)
        self.ax_time.set_title("時刻別 参照星数  (g・i: PS1   J・H・Ks: 2MASS)", fontsize=10)
        self.ax_time.grid(True, alpha=0.2, linestyle=":")

        self._cursor_line = self.ax_time.axvline(
            x=0, color="#cc3300", linewidth=1.5, linestyle="-", alpha=0.7, zorder=5)

        self.canvas.draw()

    # ── Star field panels ────────────────────────────────────────────────────

    def _draw_star_fields(self, entry: dict) -> None:
        fov_w, fov_h = self._get_fov()
        half_w = fov_w / 2
        half_h = fov_h / 2
        margin = max(fov_w, fov_h) * 0.18

        for band, ax in self.ax_bands.items():
            ax.clear()
            bdata = entry["bands"][band]
            df    = bdata["df"]
            n_u   = bdata["n_usable"]
            ass   = bdata["assessment"]

            rect = mpatches.Rectangle(
                (-half_w, -half_h), fov_w, fov_h,
                linewidth=1.2, edgecolor=BAND_COLORS[band],
                facecolor="none", linestyle="--")
            ax.add_patch(rect)

            ax.scatter([0], [0], s=120, c="gold", marker="*",
                       zorder=6, edgecolors="#cc8800", linewidths=0.6)

            if df is not None and not df.empty and "x_arcmin" in df.columns:
                usable   = df[df["usable"]]
                rejected = df[~df["usable"]]

                if not usable.empty:
                    mags  = usable["mag"].fillna(15.0).clip(lower=8, upper=22)
                    sizes = np.clip(200 - mags * 8, 8, 200)
                    ax.scatter(usable["x_arcmin"], usable["y_arcmin"],
                               s=sizes, c=BAND_COLORS[band], alpha=0.75,
                               zorder=5, edgecolors="none")

                if not rejected.empty:
                    ax.scatter(rejected["x_arcmin"], rejected["y_arcmin"],
                               s=18, c="tomato", marker="x", alpha=0.6,
                               zorder=4, linewidths=0.8)

            ax.set_xlim( half_w + margin, -half_w - margin)
            ax.set_ylim(-half_h - margin,  half_h + margin)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("ΔRA [']", fontsize=7)
            if band == "g":
                ax.set_ylabel("ΔDec [']", fontsize=7)
            ax.tick_params(labelsize=6)

            color = ASSESS_COLORS.get(ass, "black")
            ax.set_title(f"{band}   {n_u}★  [{ass}]",
                         fontsize=8, color=color, fontweight="bold")
            ax.grid(True, alpha=0.2, linestyle=":")

        self.canvas.draw()

    def _draw_star_fields_empty(self) -> None:
        fov_w, fov_h = self._get_fov()
        half_w = fov_w / 2
        half_h = fov_h / 2
        margin = max(fov_w, fov_h) * 0.18

        for band, ax in self.ax_bands.items():
            ax.clear()
            rect = mpatches.Rectangle(
                (-half_w, -half_h), fov_w, fov_h,
                linewidth=1.2, edgecolor=BAND_COLORS[band], facecolor="none",
                linestyle="--")
            ax.add_patch(rect)
            if band == "g":
                ax.text(0, 0, "計算後に表示", ha="center", va="center",
                        color="#888888", fontsize=8, style="italic")
            ax.set_xlim( half_w + margin, -half_w - margin)
            ax.set_ylim(-half_h - margin,  half_h + margin)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("ΔRA [']", fontsize=7)
            if band == "g":
                ax.set_ylabel("ΔDec [']", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_title(band, fontsize=8, color=BAND_COLORS[band])
            ax.grid(True, alpha=0.2, linestyle=":")

    def _draw_empty_plots(self) -> None:
        self.ax_time.clear()
        self.ax_time.text(0.5, 0.5, "計算後に参照星数グラフが表示されます",
                          ha="center", va="center", transform=self.ax_time.transAxes,
                          color="#888888", fontsize=9, style="italic")
        self.ax_time.set_title(
            "時刻別 参照星数  (g・i: PS1   J・H・Ks: 2MASS)", fontsize=10)
        self._draw_star_fields_empty()
        self.canvas.draw()
        self.txt_summary.config(state="normal")
        self.txt_summary.delete("1.0", "end")
        self.txt_summary.insert("1.0", "— まだ計算していません —", "hint")
        self.txt_summary.config(state="disabled")

    # ── Summary ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_radec(ra_deg: float, dec_deg: float) -> str:
        from astropy.coordinates import Angle
        import astropy.units as u
        ra_str  = Angle(ra_deg  * u.deg).to_string(unit=u.hour, sep=":", precision=2, pad=True)
        dec_str = Angle(dec_deg * u.deg).to_string(unit=u.deg,  sep=":", precision=1,
                                                    alwayssign=True, pad=True)
        return f"RA {ra_str}  Dec {dec_str}"

    def _update_summary(self, entry: dict) -> None:
        time_str = self._fmt_label(entry.get("time", ""))
        v_str    = (f"  V={entry['v_mag']:.2f}" if entry.get("v_mag") is not None else "")
        ra       = entry.get("ra")
        dec      = entry.get("dec")
        radec    = ("  " + self._fmt_radec(ra, dec)
                    if ra is not None and dec is not None else "")

        band_parts = []
        for band in BANDS:
            bd  = entry["bands"][band]
            ass = bd["assessment"]
            col_marker = {"GOOD": "◎", "OK": "○", "MARGINAL": "△",
                          "POOR": "▲", "BAD": "✕"}.get(ass, "?")
            band_parts.append(f"{band}: {bd['n_usable']}★ {col_marker}[{ass}]")

        lines = [
            f"時刻: {time_str}{v_str}",
            f"座標:{radec}",
            "   ".join(band_parts),
        ]
        text = "\n".join(lines)

        # Best assessment across optical bands
        best = max(
            (entry["bands"][b]["n_usable"] for b in ("g", "i")),
            default=0
        )
        thr_good = self._get_float(self.var_thr_good, 30)
        thr_ok   = self._get_float(self.var_thr_ok,   10)
        thr_marg = self._get_float(self.var_thr_marg,  5)
        if best >= thr_good:
            overall = "GOOD"
        elif best >= thr_ok:
            overall = "OK"
        elif best >= thr_marg:
            overall = "MARGINAL"
        elif best >= 1:
            overall = "POOR"
        else:
            overall = "BAD"

        color = ASSESS_COLORS.get(overall, "#000000")
        self.txt_summary.tag_configure("colored", foreground=color)
        self.txt_summary.config(state="normal")
        self.txt_summary.delete("1.0", "end")
        self.txt_summary.insert("1.0", text, "colored")
        self.txt_summary.config(state="disabled")

    # ── Save ─────────────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        if not self._cache:
            messagebox.showinfo("保存", "データがありません。先に計算してください。")
            return

        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV ファイル", "*.csv"), ("すべてのファイル", "*.*")],
            initialfile=f"asteroid_{ts}.csv")
        if not path:
            return

        rows = []
        for entry in self._cache:
            for band in BANDS:
                bd = entry["bands"][band]
                df = bd.get("df")
                if df is None or df.empty:
                    continue
                df = df.copy()
                df["time"]       = entry["time"]
                df["v_mag"]      = entry.get("v_mag")
                df["band"]       = band
                df["assessment"] = bd["assessment"]
                for col in OUTPUT_COLUMNS:
                    if col not in df.columns:
                        df[col] = ""
                rows.append(df)

        if rows:
            out = pd.concat(rows, ignore_index=True)
        else:
            out = pd.DataFrame(columns=OUTPUT_COLUMNS + ["band"])

        out.to_csv(path, index=False)
        self.lbl_status.config(text=f"保存: {path}", foreground="#0055cc")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    root.title("refstar_planner — 可変モード (小惑星)")
    root.minsize(1200, 750)
    panel = VariableModePanel(root)
    panel.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
