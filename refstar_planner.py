#!/usr/bin/env python3
"""
refstar_planner.py — Reference star planner for asteroid photometry.

Queries Pan-STARRS / Gaia DR3 / 2MASS (via VizieR) or SIMBAD to count
photometrically usable stars within a rectangular instrument FoV centred
on a given RA/Dec.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning

warnings.filterwarnings("ignore", category=AstropyWarning)

# ---------------------------------------------------------------------------
# Catalog column mappings
# ---------------------------------------------------------------------------

CATALOG_CONFIG: dict[str, dict] = {
    "panstarrs": {
        "vizier_id": "II/349/ps1",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "source_id_col": "objID",
        "mag_col": "rmag",
        "mag_err_col": "e_rmag",
        "mag_col2": "gmag",
        "color_formula": ("gmag", "rmag"),
        "extra_cols": ["gmag", "imag", "e_gmag", "e_imag", "Qual"],
        "column_filters_key": "rmag",
        "extended_flag_col": None,
    },
    "gaia": {
        "vizier_id": "I/355/gaiadr3",
        "ra_col": "RA_ICRS",
        "dec_col": "DE_ICRS",
        "source_id_col": "Source",
        "mag_col": "Gmag",
        "mag_err_col": "e_Gmag",
        "mag_col2": "BPmag",
        "color_formula": ("BPmag", "RPmag"),
        "extra_cols": ["BPmag", "RPmag", "e_BPmag", "e_RPmag", "VarFlag", "NSS"],
        "column_filters_key": "Gmag",
        "extended_flag_col": None,
    },
    "2mass": {
        "vizier_id": "II/246/out",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "source_id_col": "2MASS",
        "mag_col": "Jmag",
        "mag_err_col": "e_Jmag",
        "mag_col2": "Kmag",
        "color_formula": ("Jmag", "Kmag"),
        "extra_cols": ["Hmag", "Kmag", "e_Hmag", "e_Kmag", "Qflg"],
        "column_filters_key": "Jmag",
        "extended_flag_col": None,
    },
}

BAND_TO_COL: dict[str, dict[str, str]] = {
    "2mass": {
        "J": ("Jmag", "e_Jmag"),
        "H": ("Hmag", "e_Hmag"),
        "Ks": ("Kmag", "e_Kmag"),
    }
}

ASSESSMENT_LABELS = {
    "good": "GOOD",
    "ok": "OK",
    "marginal": "MARGINAL",
    "poor": "POOR",
    "bad": "BAD",
}

# Human-readable band labels shown in the GUI
BAND_LABELS: dict[str, str] = {
    "panstarrs": "PS1 r",
    "gaia":      "Gaia G",
    "2mass":     "2MASS J",
}


# ---------------------------------------------------------------------------
# Target-star magnitude lookup (VizieR point-source match)
# ---------------------------------------------------------------------------

def query_target_magnitude(
    center: SkyCoord,
    catalog_name: str,
    band: Optional[str] = None,
    search_radius_arcsec: float = 15.0,
) -> Optional[float]:
    """Return the catalog magnitude of the target star closest to *center*.

    Queries VizieR within *search_radius_arcsec* and picks the nearest match.
    Returns None if the star is not found or the magnitude is masked/NaN.
    """
    try:
        from astroquery.vizier import Vizier
    except ImportError:
        return None

    cfg = CATALOG_CONFIG.get(catalog_name)
    if cfg is None:
        return None

    mag_col     = cfg["mag_col"]
    mag_err_col = cfg["mag_err_col"]
    if catalog_name == "2mass" and band and band in BAND_TO_COL["2mass"]:
        mag_col, mag_err_col = BAND_TO_COL["2mass"][band]

    ra_col  = cfg["ra_col"]
    dec_col = cfg["dec_col"]

    vizier = Vizier(columns=[ra_col, dec_col, mag_col], row_limit=10)
    vizier.TIMEOUT = 30

    try:
        result = vizier.query_region(
            center,
            radius=search_radius_arcsec * u.arcsec,
            catalog=cfg["vizier_id"],
        )
    except Exception as exc:
        print(f"[WARN] query_target_magnitude ({catalog_name}): {exc}", file=sys.stderr)
        return None

    if result is None or len(result) == 0:
        return None

    tbl = result[0]
    if len(tbl) == 0:
        return None

    # Resolve column names flexibly (some VizieR responses add suffixes)
    def _col(name: str) -> Optional[str]:
        if name in tbl.colnames:
            return name
        lm = {c.lower(): c for c in tbl.colnames}
        return lm.get(name.lower())

    ra_c  = _col(ra_col)
    dec_c = _col(dec_col)
    mag_c = _col(mag_col)

    if ra_c is None or dec_c is None or mag_c is None:
        return None

    try:
        df = tbl.to_pandas()
        ra_vals  = pd.to_numeric(df[ra_c],  errors="coerce").values
        dec_vals = pd.to_numeric(df[dec_c], errors="coerce").values
        mag_vals = pd.to_numeric(df[mag_c], errors="coerce").values

        valid = ~(np.isnan(ra_vals) | np.isnan(dec_vals) | np.isnan(mag_vals))
        if not valid.any():
            return None

        stars = SkyCoord(
            ra=ra_vals[valid] * u.deg,
            dec=dec_vals[valid] * u.deg,
            frame="icrs",
        )
        sep = center.separation(stars).to(u.arcsec).value
        idx = int(np.argmin(sep))
        if sep[idx] > search_radius_arcsec:
            return None
        return float(mag_vals[valid][idx])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Coordinate parsing
# ---------------------------------------------------------------------------

def parse_coord(ra: str | float, dec: str | float) -> SkyCoord:
    """Parse RA/Dec (sexagesimal or decimal degrees) into SkyCoord (ICRS)."""
    try:
        ra_f = float(ra)
        dec_f = float(dec)
        return SkyCoord(ra=ra_f * u.deg, dec=dec_f * u.deg, frame="icrs")
    except (ValueError, TypeError):
        ra_str = str(ra).strip()
        dec_str = str(dec).strip()
        # hms / dms format
        if ":" in ra_str:
            return SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg), frame="icrs")
        else:
            return SkyCoord(ra=ra_str, dec=dec_str, unit=(u.deg, u.deg), frame="icrs")


# ---------------------------------------------------------------------------
# FoV geometry
# ---------------------------------------------------------------------------

def circumscribed_radius(width_arcmin: float, height_arcmin: float) -> float:
    """Return circumscribed circle radius in arcmin for a rectangular FoV."""
    return 0.5 * np.sqrt(width_arcmin**2 + height_arcmin**2)


def filter_rectangular_fov(
    df: pd.DataFrame,
    center: SkyCoord,
    width_arcmin: float,
    height_arcmin: float,
    pa_deg: float,
) -> pd.DataFrame:
    """
    Keep only rows whose (ra_deg, dec_deg) fall inside the rectangular FoV.

    Uses tangent-plane projection then rotation by PA (East of North).
    """
    if df.empty:
        return df.copy()

    stars = SkyCoord(
        ra=df["ra_deg"].values * u.deg,
        dec=df["dec_deg"].values * u.deg,
        frame="icrs",
    )

    # Tangent-plane offsets in arcmin (dRA projected, dDec)
    dra = (stars.ra - center.ra).wrap_at(180 * u.deg).deg * np.cos(np.deg2rad(center.dec.deg))
    ddec = (stars.dec - center.dec).deg
    dra_arcmin = dra * 60.0
    ddec_arcmin = ddec * 60.0

    # Rotate by PA (position angle, degrees East of North)
    pa_rad = np.deg2rad(pa_deg)
    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)
    # x = along detector width axis, y = along detector height axis
    x_arcmin = dra_arcmin * cos_pa - ddec_arcmin * sin_pa
    y_arcmin = dra_arcmin * sin_pa + ddec_arcmin * cos_pa

    df = df.copy()
    df["x_arcmin"] = x_arcmin
    df["y_arcmin"] = y_arcmin

    inside = (np.abs(x_arcmin) <= width_arcmin / 2) & (np.abs(y_arcmin) <= height_arcmin / 2)
    return df[inside].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Catalog queries
# ---------------------------------------------------------------------------

def query_simbad(center: SkyCoord, radius_arcmin: float, verbose: bool = False) -> pd.DataFrame:
    """
    Query SIMBAD for objects within radius_arcmin of center.

    NOTE: SIMBAD is NOT a photometric calibration catalog.
    """
    try:
        from astroquery.simbad import Simbad
    except ImportError:
        print("[ERROR] astroquery is not installed.", file=sys.stderr)
        return pd.DataFrame()

    simbad = Simbad()
    simbad.add_votable_fields("otype", "flux(V)", "flux_error(V)", "flux(R)", "flux_error(R)")
    simbad.TIMEOUT = 60

    radius = radius_arcmin * u.arcmin
    try:
        result = simbad.query_region(center, radius=radius)
    except Exception as exc:
        print(f"[WARN] SIMBAD query failed: {exc}", file=sys.stderr)
        return pd.DataFrame()

    if result is None or len(result) == 0:
        return pd.DataFrame()

    tbl = result.to_pandas()
    if verbose:
        print(f"  SIMBAD raw columns: {list(tbl.columns)}")

    # Map columns robustly (astroquery >= 0.4.11 uses lowercase "ra"/"dec" in degrees)
    # Priority: exact lowercase → legacy RA_d/DEC_d → any RA/DEC prefix
    def _find_col(candidates: list[str], cols: list[str]) -> str | None:
        for c in candidates:
            if c in cols:
                return c
        for c in cols:
            if c.upper().startswith("RA") and "RA" in candidates[0].upper():
                return c
            if c.upper().startswith("DEC") and "DEC" in candidates[0].upper():
                return c
        return None

    ra_col  = _find_col(["ra", "RA_d", "RA"], list(tbl.columns))
    dec_col = _find_col(["dec", "DEC_d", "DEC"], list(tbl.columns))

    if ra_col is None or dec_col is None:
        print(f"[ERROR] Cannot identify RA/Dec columns in SIMBAD result. Columns: {list(tbl.columns)}", file=sys.stderr)
        return pd.DataFrame()

    mag_col = next((c for c in tbl.columns if c.upper() in ("FLUX_V", "FLUX_R", "V", "R")), None)
    mag_err_col = next((c for c in tbl.columns if "ERROR" in c.upper() and ("V" in c.upper() or "R" in c.upper())), None)
    otype_col = next((c for c in tbl.columns if c.lower() == "otype"), None)
    main_id_col = next((c for c in tbl.columns if c.lower() in ("main_id", "main-id")), None)

    out = pd.DataFrame()
    out["ra_deg"] = pd.to_numeric(tbl[ra_col], errors="coerce")
    out["dec_deg"] = pd.to_numeric(tbl[dec_col], errors="coerce")
    out["source_id"] = tbl[main_id_col].astype(str) if main_id_col else [f"simbad_{i}" for i in range(len(tbl))]
    out["mag"] = pd.to_numeric(tbl[mag_col], errors="coerce") if mag_col else np.nan
    out["mag_err"] = pd.to_numeric(tbl[mag_err_col], errors="coerce") if mag_err_col else np.nan
    out["color"] = np.nan
    out["object_type"] = tbl[otype_col].astype(str) if otype_col else ""
    out["catalog"] = "simbad"

    return out.dropna(subset=["ra_deg", "dec_deg"]).reset_index(drop=True)


def query_vizier_catalog(
    center: SkyCoord,
    radius_arcmin: float,
    catalog_name: str,
    mag_min: float,
    mag_max: float,
    band: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Query Pan-STARRS / Gaia DR3 / 2MASS via VizieR.

    Returns a DataFrame with standardised columns.
    """
    try:
        from astroquery.vizier import Vizier
    except ImportError:
        print("[ERROR] astroquery is not installed.", file=sys.stderr)
        return pd.DataFrame()

    cfg = CATALOG_CONFIG.get(catalog_name)
    if cfg is None:
        print(f"[ERROR] Unknown catalog: {catalog_name}", file=sys.stderr)
        return pd.DataFrame()

    # Optionally switch band for 2MASS
    mag_col = cfg["mag_col"]
    mag_err_col = cfg["mag_err_col"]
    if catalog_name == "2mass" and band and band in BAND_TO_COL["2mass"]:
        mag_col, mag_err_col = BAND_TO_COL["2mass"][band]

    columns = [cfg["ra_col"], cfg["dec_col"], cfg["source_id_col"], mag_col, mag_err_col]
    columns += [c for c in cfg["extra_cols"] if c not in columns]
    columns = list(dict.fromkeys(columns))  # deduplicate, preserve order

    col_filters = {cfg["column_filters_key"]: f"{mag_min}..{mag_max}"}

    vizier = Vizier(
        columns=columns,
        column_filters=col_filters,
        row_limit=-1,
    )
    vizier.TIMEOUT = 120

    radius = radius_arcmin * u.arcmin
    try:
        result = vizier.query_region(center, radius=radius, catalog=cfg["vizier_id"])
    except Exception as exc:
        print(f"[WARN] VizieR query failed for {catalog_name}: {exc}", file=sys.stderr)
        return pd.DataFrame()

    if result is None or len(result) == 0:
        return pd.DataFrame()

    tbl = result[0].to_pandas()

    if verbose:
        print(f"  VizieR ({catalog_name}) raw columns: {list(tbl.columns)}")

    # Resolve column names (VizieR may add prefixes)
    def resolve(col: str) -> Optional[str]:
        if col in tbl.columns:
            return col
        # try case-insensitive match
        lower_map = {c.lower(): c for c in tbl.columns}
        return lower_map.get(col.lower())

    ra_actual = resolve(cfg["ra_col"])
    dec_actual = resolve(cfg["dec_col"])
    sid_actual = resolve(cfg["source_id_col"])
    mag_actual = resolve(mag_col)
    mag_err_actual = resolve(mag_err_col)

    missing = [n for n, a in [("ra", ra_actual), ("dec", dec_actual), ("mag", mag_actual)] if a is None]
    if missing:
        print(
            f"[ERROR] Missing expected columns {missing} in {catalog_name} result. "
            f"Available: {list(tbl.columns)}",
            file=sys.stderr,
        )
        return pd.DataFrame()

    out = pd.DataFrame()
    out["ra_deg"] = pd.to_numeric(tbl[ra_actual], errors="coerce")
    out["dec_deg"] = pd.to_numeric(tbl[dec_actual], errors="coerce")
    out["source_id"] = tbl[sid_actual].astype(str) if sid_actual else [f"{catalog_name}_{i}" for i in range(len(tbl))]
    out["mag"] = pd.to_numeric(tbl[mag_actual], errors="coerce")
    out["mag_err"] = pd.to_numeric(tbl[mag_err_actual], errors="coerce") if mag_err_actual else np.nan
    out["catalog"] = catalog_name

    # Compute color
    c1, c2 = cfg["color_formula"]
    c1_actual, c2_actual = resolve(c1), resolve(c2)
    if c1_actual and c2_actual:
        out["color"] = pd.to_numeric(tbl[c1_actual], errors="coerce") - pd.to_numeric(tbl[c2_actual], errors="coerce")
    else:
        out["color"] = np.nan

    # Object type
    if catalog_name == "gaia":
        vf = resolve("VarFlag")
        out["object_type"] = tbl[vf].astype(str).str.strip() if vf else ""
    elif catalog_name == "panstarrs":
        out["object_type"] = ""
    elif catalog_name == "2mass":
        qf = resolve("Qflg")
        out["object_type"] = tbl[qf].astype(str).str.strip() if qf else ""
    else:
        out["object_type"] = ""

    return out.dropna(subset=["ra_deg", "dec_deg"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------

STAR_OTYPES = {
    "*", "Star", "**", "V*", "PM*", "HB*", "RG*", "WD*", "XB*", "LP*",
    "Be*", "BS*", "s*b", "s*r", "s*y",
}

REJECT_OTYPES = {"G", "GiG", "GiC", "Cl*", "QSO", "AGN", "SN", "SNR", "ISM", "Neb"}


def apply_quality_filters(
    df: pd.DataFrame,
    catalog: str,
    mag_min: float,
    mag_max: float,
    max_mag_err: float,
    min_separation_arcsec: float,
) -> pd.DataFrame:
    """
    Apply photometric quality filters and flag each row with usable/reject_reason.

    Filters applied:
    - magnitude in [mag_min, mag_max]
    - mag_err <= max_mag_err (if available)
    - separation from asteroid > min_separation_arcsec
    - no NaN in ra_deg, dec_deg, mag
    - exclude likely galaxies / QSOs / variables (catalog-specific)
    """
    df = df.copy()
    df["usable"] = True
    df["reject_reason"] = ""

    def reject(mask: pd.Series, reason: str) -> None:
        newly = mask & df["usable"]
        df.loc[newly, "usable"] = False
        df.loc[newly, "reject_reason"] = reason

    # NaN in critical columns
    reject(df["ra_deg"].isna() | df["dec_deg"].isna() | df["mag"].isna(), "missing_coord_or_mag")

    # Magnitude range
    reject(df["mag"] < mag_min, "too_bright")
    reject(df["mag"] > mag_max, "too_faint")

    # Magnitude error
    if "mag_err" in df.columns:
        reject(df["mag_err"].notna() & (df["mag_err"] > max_mag_err), "mag_err_too_large")

    # Separation from asteroid center (already in df if filter_rectangular_fov ran)
    if "separation_from_center_arcsec" in df.columns:
        reject(
            df["separation_from_center_arcsec"].notna()
            & (df["separation_from_center_arcsec"] < min_separation_arcsec),
            "too_close_to_asteroid",
        )

    # Object type filtering (catalog-specific)
    if catalog == "simbad" and "object_type" in df.columns:
        non_star = df["object_type"].apply(
            lambda ot: ot.strip() in REJECT_OTYPES or (ot.strip() and ot.strip() not in STAR_OTYPES)
        )
        reject(non_star & df["object_type"].ne(""), "non_stellar_otype")

    if catalog == "gaia" and "object_type" in df.columns:
        # VarFlag != "" indicates variable
        reject(df["object_type"].notna() & df["object_type"].ne("") & df["object_type"].ne("nan"), "gaia_variable_flag")

    if catalog == "panstarrs" and "object_type" in df.columns:
        pass  # No extended-source flag available in basic PS1 query

    return df


# ---------------------------------------------------------------------------
# Separation calculation
# ---------------------------------------------------------------------------

def add_separation(df: pd.DataFrame, center: SkyCoord) -> pd.DataFrame:
    """Add separation_from_center_arcsec column."""
    if df.empty:
        df = df.copy()
        df["separation_from_center_arcsec"] = pd.Series(dtype=float)
        return df
    stars = SkyCoord(
        ra=df["ra_deg"].values * u.deg,
        dec=df["dec_deg"].values * u.deg,
        frame="icrs",
    )
    sep = center.separation(stars).to(u.arcsec).value
    df = df.copy()
    df["separation_from_center_arcsec"] = sep
    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def assess(n_usable: int, min_good: int, min_ok: int, min_marginal: int) -> str:
    """Return assessment label based on usable star count."""
    if n_usable >= min_good:
        return ASSESSMENT_LABELS["good"]
    elif n_usable >= min_ok:
        return ASSESSMENT_LABELS["ok"]
    elif n_usable >= min_marginal:
        return ASSESSMENT_LABELS["marginal"]
    elif n_usable >= 1:
        return ASSESSMENT_LABELS["poor"]
    else:
        return ASSESSMENT_LABELS["bad"]


def summarize(
    df: pd.DataFrame,
    n_raw: int,
    n_fov: int,
    min_good: int,
    min_ok: int,
    min_marginal: int,
) -> dict:
    """Summarise results into a dict."""
    n_usable = int(df["usable"].sum()) if "usable" in df.columns else 0
    n_rejected_near = int((df.get("reject_reason", pd.Series()) == "too_close_to_asteroid").sum())
    n_quality_rejected = n_fov - n_usable

    reject_counts: dict[str, int] = {}
    if "reject_reason" in df.columns:
        counts = df[df["reject_reason"] != ""]["reject_reason"].value_counts().to_dict()
        reject_counts = {str(k): int(v) for k, v in counts.items()}

    return {
        "n_raw": n_raw,
        "n_fov": n_fov,
        "n_quality_rejected": n_quality_rejected,
        "n_rejected_near_asteroid": n_rejected_near,
        "n_usable": n_usable,
        "reject_counts": reject_counts,
        "assessment": assess(n_usable, min_good, min_ok, min_marginal),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "time",
    "catalog",
    "source_id",
    "ra_deg",
    "dec_deg",
    "x_arcmin",
    "y_arcmin",
    "separation_from_center_arcsec",
    "mag",
    "mag_err",
    "color",
    "object_type",
    "usable",
    "reject_reason",
]


def save_outputs(
    df: pd.DataFrame,
    summary: dict,
    output_path: Optional[str],
    summary_output_path: Optional[str] = None,
    time_label: str = "",
) -> None:
    """Save star list CSV and optional summary CSV."""
    if output_path:
        # Ensure all output columns exist
        for col in OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = "" if col in ("time", "reject_reason", "object_type") else np.nan
        df["time"] = time_label
        out = df[OUTPUT_COLUMNS]
        out.to_csv(output_path, index=False)
        print(f"Saved star list to {output_path}")

    if summary_output_path and summary:
        row = {
            "time": time_label,
            "n_raw": summary["n_raw"],
            "n_fov": summary["n_fov"],
            "n_usable": summary["n_usable"],
            "n_rejected_near_asteroid": summary["n_rejected_near_asteroid"],
            "assessment": summary["assessment"],
        }
        sdf = pd.DataFrame([row])
        write_header = not Path(summary_output_path).exists()
        sdf.to_csv(summary_output_path, mode="a", index=False, header=write_header)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(
    df: pd.DataFrame,
    width_arcmin: float,
    height_arcmin: float,
    output_path: str,
    title: str = "",
) -> None:
    """Plot FoV with usable and rejected stars."""
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skipping plot.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    if not df.empty and "usable" in df.columns:
        usable = df[df["usable"]]
        rejected = df[~df["usable"]]

        if not usable.empty:
            # Scale marker size by brightness (brighter = larger)
            sizes = np.clip(200 - usable["mag"] * 8, 10, 200)
            ax.scatter(
                usable["x_arcmin"], usable["y_arcmin"],
                s=sizes, c="steelblue", marker="o", alpha=0.8, label="Usable",
                edgecolors="navy", linewidths=0.5,
            )

        if not rejected.empty:
            ax.scatter(
                rejected["x_arcmin"], rejected["y_arcmin"],
                s=30, c="tomato", marker="x", alpha=0.7, label="Rejected",
            )

    # Asteroid position
    ax.scatter([0], [0], s=120, c="gold", marker="*", zorder=5, label="Asteroid (center)")

    # FoV rectangle
    rect = mpatches.Rectangle(
        (-width_arcmin / 2, -height_arcmin / 2),
        width_arcmin, height_arcmin,
        linewidth=1.5, edgecolor="black", facecolor="none", linestyle="--", label="FoV",
    )
    ax.add_patch(rect)

    ax.set_xlabel("ΔRA offset [arcmin] (East ←)")
    ax.set_ylabel("ΔDec offset [arcmin]")
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title or "Reference Star Distribution", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_single(
    center: SkyCoord,
    width_arcmin: float,
    height_arcmin: float,
    pa_deg: float,
    catalog: str,
    mag_min: float,
    mag_max: float,
    max_mag_err: float,
    min_separation_arcsec: float,
    min_good: int,
    min_ok: int,
    min_marginal: int,
    band: Optional[str] = None,
    verbose: bool = False,
    time_label: str = "",
) -> tuple[pd.DataFrame, dict]:
    """Run the full pipeline for a single pointing. Returns (df, summary)."""
    radius_arcmin = circumscribed_radius(width_arcmin, height_arcmin)

    if verbose:
        print(f"  Query radius: {radius_arcmin:.2f} arcmin")

    # Query
    if catalog == "simbad":
        raw_df = query_simbad(center, radius_arcmin, verbose=verbose)
    else:
        raw_df = query_vizier_catalog(
            center, radius_arcmin, catalog, mag_min, mag_max, band=band, verbose=verbose
        )

    n_raw = len(raw_df)

    if raw_df.empty:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        return empty, summarize(empty, 0, 0, min_good, min_ok, min_marginal)

    # FoV filter
    fov_df = filter_rectangular_fov(raw_df, center, width_arcmin, height_arcmin, pa_deg)
    n_fov = len(fov_df)

    # Separation from asteroid center
    fov_df = add_separation(fov_df, center)

    # Quality filters
    filtered_df = apply_quality_filters(
        fov_df, catalog, mag_min, mag_max, max_mag_err, min_separation_arcsec
    )

    summary = summarize(filtered_df, n_raw, n_fov, min_good, min_ok, min_marginal)
    return filtered_df, summary


def print_results(
    center: SkyCoord,
    width_arcmin: float,
    height_arcmin: float,
    pa_deg: float,
    catalog: str,
    mag_min: float,
    mag_max: float,
    summary: dict,
    time_label: str = "",
) -> None:
    """Print formatted summary to stdout."""
    if time_label:
        print(f"\nTime: {time_label}")
    print(f"Center: RA={center.ra.deg:.6f} deg, Dec={center.dec.deg:.6f} deg")
    print(f"FoV: {width_arcmin:.2f} x {height_arcmin:.2f} arcmin, PA={pa_deg:.1f} deg")
    print(f"Catalog: {catalog}")
    print(f"Magnitude range: {mag_min:.1f} <= mag <= {mag_max:.1f}")
    print()
    print(f"Raw objects from query: {summary['n_raw']}")
    print(f"Inside rectangular FoV: {summary['n_fov']}")
    print(f"After quality filters: {summary['n_usable']}")
    print(f"Objects rejected near asteroid: {summary['n_rejected_near_asteroid']}")
    print()
    print(f"Usable reference stars: {summary['n_usable']}")
    print(f"Assessment: {summary['assessment']}")
    if summary["reject_counts"]:
        print("Rejection reasons:")
        for reason, count in sorted(summary["reject_counts"].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reference star planner for asteroid photometry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--ra", help="Right Ascension (deg or hh:mm:ss.s)")
    group.add_argument("--input", metavar="CSV", help="Input CSV with columns: time,ra,dec")

    p.add_argument("--dec", help="Declination (deg or +dd:mm:ss)")
    p.add_argument("--width-arcmin", type=float, required=True, help="FoV width in arcmin")
    p.add_argument("--height-arcmin", type=float, required=True, help="FoV height in arcmin")
    p.add_argument("--pa-deg", type=float, default=0.0, help="Position angle (deg E of N)")
    p.add_argument(
        "--catalog",
        choices=["simbad", "panstarrs", "gaia", "2mass"],
        default="panstarrs",
        help="Catalog to query",
    )
    p.add_argument("--band", choices=["J", "H", "Ks"], default=None, help="NIR band for 2MASS")
    p.add_argument("--mag-min", type=float, default=12.0, help="Minimum magnitude")
    p.add_argument("--mag-max", type=float, default=18.0, help="Maximum magnitude")
    p.add_argument("--max-mag-err", type=float, default=0.05, help="Maximum allowed magnitude error")
    p.add_argument("--min-separation-arcsec", type=float, default=5.0, help="Min separation from asteroid (arcsec)")
    p.add_argument("--min-good-stars", type=int, default=30, help="Threshold for GOOD assessment")
    p.add_argument("--min-ok-stars", type=int, default=10, help="Threshold for OK assessment")
    p.add_argument("--min-marginal-stars", type=int, default=5, help="Threshold for MARGINAL assessment")
    p.add_argument("--output", metavar="CSV", help="Output star list CSV path")
    p.add_argument("--summary-output", metavar="CSV", help="Summary CSV path (multi-epoch mode)")
    p.add_argument("--plot", metavar="PNG", help="Output plot path (e.g. refs.png)")
    p.add_argument("--verbose", action="store_true", help="Print debug info")
    p.add_argument("--sleep-sec", type=float, default=2.0, help="Sleep between queries (multi-epoch mode)")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.ra and not args.dec:
        parser.error("--dec is required when --ra is given")

    if args.catalog == "simbad":
        print(
            "[NOTE] SIMBAD is NOT a photometric calibration catalog. "
            "Use --catalog panstarrs/gaia/2mass for reliable photometry.",
            file=sys.stderr,
        )

    # Clear summary output if starting fresh
    if args.summary_output and Path(args.summary_output).exists():
        Path(args.summary_output).unlink()

    # ----- Single RA/Dec mode -----
    if args.ra:
        center = parse_coord(args.ra, args.dec)
        df, summary = run_single(
            center=center,
            width_arcmin=args.width_arcmin,
            height_arcmin=args.height_arcmin,
            pa_deg=args.pa_deg,
            catalog=args.catalog,
            mag_min=args.mag_min,
            mag_max=args.mag_max,
            max_mag_err=args.max_mag_err,
            min_separation_arcsec=args.min_separation_arcsec,
            min_good=args.min_good_stars,
            min_ok=args.min_ok_stars,
            min_marginal=args.min_marginal_stars,
            band=args.band,
            verbose=args.verbose,
        )
        print_results(
            center, args.width_arcmin, args.height_arcmin, args.pa_deg,
            args.catalog, args.mag_min, args.mag_max, summary,
        )
        save_outputs(df, summary, args.output, args.summary_output, time_label="")
        if args.plot:
            make_plot(
                df, args.width_arcmin, args.height_arcmin, args.plot,
                title=f"RA={center.ra.deg:.4f} Dec={center.dec.deg:.4f} | {args.catalog} | {summary['assessment']}",
            )

    # ----- Multi-epoch CSV mode -----
    else:
        try:
            positions = pd.read_csv(args.input)
        except Exception as exc:
            print(f"[ERROR] Cannot read input CSV: {exc}", file=sys.stderr)
            sys.exit(1)

        required_cols = {"ra", "dec"}
        if not required_cols.issubset(set(positions.columns)):
            print(
                f"[ERROR] Input CSV must contain columns: ra, dec (got {list(positions.columns)})",
                file=sys.stderr,
            )
            sys.exit(1)

        all_dfs: list[pd.DataFrame] = []
        for i, row in positions.iterrows():
            time_label = str(row.get("time", "")) if "time" in positions.columns else ""
            print(f"\n[{i + 1}/{len(positions)}] {time_label or 'row ' + str(i)}", end="  ")

            try:
                center = parse_coord(row["ra"], row["dec"])
            except Exception as exc:
                print(f"[WARN] Cannot parse coord at row {i}: {exc}", file=sys.stderr)
                continue

            df, summary = run_single(
                center=center,
                width_arcmin=args.width_arcmin,
                height_arcmin=args.height_arcmin,
                pa_deg=args.pa_deg,
                catalog=args.catalog,
                mag_min=args.mag_min,
                mag_max=args.mag_max,
                max_mag_err=args.max_mag_err,
                min_separation_arcsec=args.min_separation_arcsec,
                min_good=args.min_good_stars,
                min_ok=args.min_ok_stars,
                min_marginal=args.min_marginal_stars,
                band=args.band,
                verbose=args.verbose,
                time_label=time_label,
            )

            print_results(
                center, args.width_arcmin, args.height_arcmin, args.pa_deg,
                args.catalog, args.mag_min, args.mag_max, summary, time_label=time_label,
            )

            save_outputs(df, summary, output_path=None, summary_output_path=args.summary_output, time_label=time_label)
            df["time"] = time_label
            all_dfs.append(df)

            if i < len(positions) - 1:
                time.sleep(args.sleep_sec)

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
        else:
            combined = pd.DataFrame(columns=OUTPUT_COLUMNS)

        if args.output:
            for col in OUTPUT_COLUMNS:
                if col not in combined.columns:
                    combined[col] = ""
            combined[OUTPUT_COLUMNS].to_csv(args.output, index=False)
            print(f"\nSaved combined star list to {args.output}")

        if args.plot and not combined.empty:
            make_plot(
                combined, args.width_arcmin, args.height_arcmin, args.plot,
                title=f"All epochs | {args.catalog}",
            )


if __name__ == "__main__":
    main()
