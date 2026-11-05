#!/usr/bin/env python3
"""Coarsen real V4r6 initial Theta/Salt (from the production LLC90 pickup
file) into a 15-level LLC30 initial condition, replacing this config's
current flat analytic IC (tRef=15*20., sRef=15*35., no hydrogThetaFile/
hydrogSaltFile at all).

Two-stage design (vertical remap FIRST, horizontal coarsening SECOND --
see the "Initial condition" section of README.md for the full rationale):

  Stage 1: at full LLC90 horizontal resolution, remap the 50 production
  vertical levels down to LLC30's EXISTING 15 levels (parsed from
  ../input/data's &PARM04, not re-hardcoded) via a depth-weighted overlap-
  conservative average, masked by a FINE 50-level wet mask (built from the
  real LLC90 bathymetry + production's 50-level delR). This must happen
  before any horizontal averaging: the real fine bathymetry is the only
  place per-column seafloor depth is still precise enough to decide which
  of the 50 source levels are physically meaningful for THIS column.

  Stage 2: horizontal 3:1 coarsening of the now-15-level, still-LLC90-
  resolution fields, using a SECOND, independently-computed wet mask (same
  fine bathymetry, now evaluated against LLC30's 15-level delR) and a new
  wet-cell-only mean (neither of coarsen_llc90_to_llc30.py's two existing
  rules -- coarsen_bathy_block's min-magnitude-over-wet, or
  coarsen_forcing_facet_mean's plain unweighted mean -- is correct for T/S).

Binary-layout assumptions (confirmed by direct byte-level read-back of the
real pickup file, not just its .meta header -- see README.md):

  pickup.0000000001.data: raw, unformatted, big-endian real*8, no record
  markers, 453 stacked (COMPACT_NY=1170, COMPACT_NX=90) records (field
  order Uvel,Vvel,Theta,Salt,GuNm1,GuNm2,GvNm1,GvNm2,AddMass [50 levels
  each] then EtaN,dEtaHdt,EtaH [2-D]; 8*50 + 50 + 3 = 453, matching both
  the .meta file and the actual file size exactly). Theta = records
  100-149, Salt = records 150-199, k=0 is the surface (confirmed: level-0
  Theta min/max ~ -2.54/32.39C, physically sane SST; level-49 ~
  -0.73/1.86C, physically sane abyssal temp -- NOT reversed/duplicated).

  BATHY_ICE_SHELF_CAVITY_PLUS_ICE_FRONT_LLC_0090.bin: real*4 big-endian,
  same (1170,90) compact convention as coarsen_llc90_to_llc30.py's
  BATHY_SRC -- this is the REAL fine bathymetry, distinct from (and the
  original source of) the already-coarsened input/BATHY_LLC30.bin.

Run:
    export PATH=$HOME/micromamba/envs/podaac/bin:$PATH
    cd MITgcm/verification/global_ocean.llc30x15/preprocess
    python3 coarsen_initial_ts_to_llc30.py
"""

import os
import re
import sys

import numpy as np

import coarsen_llc90_to_llc30 as base

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PICKUP_SRC = (
    "/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/"
    "ancillary_data/ancillary_data_orig/input_init/pickup.0000000001.data"
)
FINE_BATHY_SRC = (
    "/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/"
    "ancillary_data/ancillary_data_orig/input_init/"
    "BATHY_ICE_SHELF_CAVITY_PLUS_ICE_FRONT_LLC_0090.bin"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLC30_DATA_PATH = os.path.join(SCRIPT_DIR, "..", "input", "data")
LLC30_BATHY_PATH = os.path.join(SCRIPT_DIR, "..", "input", "BATHY_LLC30.bin")
MITGRID_DIR = os.path.join(SCRIPT_DIR, "..", "input")

OUT_DIR = os.path.join(SCRIPT_DIR, "output")
OUT_IC_DIR = os.path.join(OUT_DIR, "initial_condition")

COMPACT_NX = base.COMPACT_NX          # 90
COMPACT_NY = base.COMPACT_NY          # 1170
COMPACT_COARSE_NX = base.COMPACT_COARSE_NX   # 30
COMPACT_COARSE_NY = base.COMPACT_COARSE_NY   # 390
FACTOR = base.FACTOR                  # 3

# ---------------------------------------------------------------------------
# Pickup file layout (field order confirmed against the real file -- see
# module docstring)
# ---------------------------------------------------------------------------

PICKUP_NLEV = 50
PICKUP_FIELD_ORDER = [
    "Uvel", "Vvel", "Theta", "Salt", "GuNm1", "GuNm2", "GvNm1", "GvNm2",
    "AddMass",  # 9 fields x 50 levels = 450 records
]
PICKUP_2D_FIELDS = ["EtaN", "dEtaHdt", "EtaH"]  # 3 more records = 453 total
PICKUP_EXPECTED_RECORDS = PICKUP_NLEV * len(PICKUP_FIELD_ORDER) + len(PICKUP_2D_FIELDS)
assert PICKUP_EXPECTED_RECORDS == 453, PICKUP_EXPECTED_RECORDS

THETA_FIELD_INDEX0 = PICKUP_FIELD_ORDER.index("Theta")  # 2
SALT_FIELD_INDEX0 = PICKUP_FIELD_ORDER.index("Salt")    # 3

# ---------------------------------------------------------------------------
# Production V4r6's 50-level delR / tRef / sRef, verbatim from
# Time-invariant-adjoint/v4r6/input.V4r6.102/data's &PARM01/&PARM04. No
# local file to parse these from in this repo -- guarded by explicit
# length/sum asserts below against a hand-transcription typo (mirroring
# coarsen_llc90_to_llc30.py's MONTH_HOURS_1992 leap-year-sum assert style).
# ---------------------------------------------------------------------------

PRODUCTION_DELR_50 = np.array([
    10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.01,
    10.03, 10.11, 10.32, 10.80, 11.76, 13.42, 16.04, 19.82, 24.85,
    31.10, 38.42, 46.50, 55.00, 63.50, 71.58, 78.90, 85.15, 90.18,
    93.96, 96.58, 98.25, 99.25, 100.01, 101.33, 104.56, 111.33, 122.83,
    139.09, 158.94, 180.83, 203.55, 226.50, 249.50, 272.50, 295.50, 318.50,
    341.50, 364.50, 387.50, 410.50, 433.50, 456.50,
])
assert PRODUCTION_DELR_50.shape == (50,)
assert abs(PRODUCTION_DELR_50.sum() - 6134.5) < 1e-6, PRODUCTION_DELR_50.sum()


def _expand_repeat_count(fortran_literal):
    """Expand a Fortran namelist repeat-count literal like '3*23.,2*20.,19.'
    into a flat list of floats. Used for production tRef/sRef below.
    """
    vals = []
    for tok in fortran_literal.strip().rstrip(",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "*" in tok:
            n, v = tok.split("*")
            vals.extend([float(v)] * int(n))
        else:
            vals.append(float(tok))
    return vals


PRODUCTION_TREF_50 = np.array(_expand_repeat_count(
    "3*23.,3*22.,21.,2*20.,19.,2*18.,17.,2*16.,15.,14.,13.,"
    "12.,11.,2*9.,8.,7.,2*6.,2*5.,3*4.,3*3.,4*2.,12*1.,"
))
assert PRODUCTION_TREF_50.shape == (50,), PRODUCTION_TREF_50.shape

PRODUCTION_SREF_50 = np.full(50, 34.5)


def parse_llc30_delR(data_path=LLC30_DATA_PATH):
    """Parse LLC30's EXISTING 15-level delR directly out of input/data's
    &PARM04 text (NOT re-hardcoded -- input/data is the actual source of
    truth this script must stay consistent with; a second hand-transcribed
    copy could silently drift out of sync with a future edit there)."""
    with open(data_path) as fh:
        text = fh.read()
    m = re.search(r"delR\s*=\s*([0-9.,\s]+?)\n\s*&", text)
    if m is None:
        raise ValueError(f"{data_path}: could not find a 'delR=...' block in &PARM04")
    vals = [float(v) for v in m.group(1).replace("\n", " ").split(",") if v.strip()]
    delR = np.array(vals)
    assert delR.shape == (15,), (
        f"{data_path}: parsed {delR.shape[0]} delR values, expected 15 -- "
        f"regex may have mismatched input/data's current formatting"
    )
    return delR


# ===========================================================================
# Vertical-grid helpers
# ===========================================================================

def build_depth_interfaces(delR):
    """delR: 1-D array of positive layer thicknesses, length N.
    Returns rF: 1-D array length N+1, rF[0]=0.0 (surface), rF[k] more
    negative with depth -- MITgcm's own rF sign/ordering convention.
    """
    delR = np.asarray(delR, dtype=np.float64)
    rF = np.zeros(delR.shape[0] + 1)
    rF[1:] = -np.cumsum(delR)
    return rF


def build_overlap_matrix(rF_target, rF_source):
    """rF_target (Nt+1,), rF_source (Ns+1,) -> W (Nt,Ns) overlap-thickness
    matrix. Depends ONLY on the two vertical grids, not on any column's
    bathymetry -- computed once, reused for every column and for the 1-D
    tRef/sRef remap.
    """
    Nt = rF_target.shape[0] - 1
    Ns = rF_source.shape[0] - 1
    top = np.minimum(rF_target[:-1, None], rF_source[None, :-1])
    bot = np.maximum(rF_target[1:, None], rF_source[None, 1:])
    W = np.maximum(0.0, top - bot)
    assert W.shape == (Nt, Ns)
    return W


def build_fine_wet_mask(fine_bathy_compact, delR):
    """fine_bathy_compact: (COMPACT_NY, COMPACT_NX) real depth (negative =
    ocean, 0 = land). delR: either production's 50-level delR (Stage 1
    mask) or LLC30's existing 15-level delR (Stage 2 mask) -- same
    function, different delR. Returns bool (len(delR), COMPACT_NY,
    COMPACT_NX): level k wet at column (j,i) iff rF[k] (the layer's TOP
    interface) is strictly shallower than the local seafloor, and the
    column itself is not land.
    """
    rF = build_depth_interfaces(delR)
    land = fine_bathy_compact == 0.0
    seafloor = fine_bathy_compact  # negative depth, 0 at land
    nlev = len(delR)
    wet = np.empty((nlev, COMPACT_NY, COMPACT_NX), dtype=bool)
    for k in range(nlev):
        wet[k] = (rF[k] > seafloor) & (~land)
    return wet


def vertical_remap_overlap_weighted(source_values, rF_target, rF_source,
                                     wet_mask=None, fill_value=0.0):
    """Generic depth-weighted conservative vertical remap, Ns -> Nt levels.

    source_values: (Ns, N) (N = flattened spatial/column count; use N=1
    for the 1-D tRef/sRef profile remap).
    wet_mask: None (every source level treated as wet -- tRef/sRef) or
    (Ns, N) bool (Stage 1's fine 50-level mask, for the 3-D T/S fields).
    Returns (Nt, N). fill_value=0.0 (not NaN) for any target level/column
    with zero overlap-weighted wet thickness.
    """
    W = build_overlap_matrix(rF_target, rF_source)  # (Nt, Ns)
    if wet_mask is None:
        wm = np.ones_like(source_values, dtype=np.float64)
    else:
        wm = wet_mask.astype(np.float64)
    masked_vals = source_values * wm
    numerator = W @ masked_vals          # (Nt, N)
    denom = W @ wm                       # (Nt, N)
    out = np.where(denom > 0, numerator / np.maximum(denom, 1e-30), fill_value)
    return out, W


# ===========================================================================
# Pickup file I/O
# ===========================================================================

def read_pickup_field_levels(path, field_index0, nlev=PICKUP_NLEV):
    """Read `nlev` consecutive (COMPACT_NY, COMPACT_NX) big-endian real*8
    records starting at `field_index0`'s first (surface) level.
    """
    rec_elems = COMPACT_NX * COMPACT_NY
    rec_bytes = rec_elems * 8
    start_record0 = field_index0 * nlev
    with open(path, "rb") as fh:
        fh.seek(start_record0 * rec_bytes)
        raw = np.fromfile(fh, dtype=">f8", count=nlev * rec_elems)
    assert raw.size == nlev * rec_elems, (
        f"{path}: read {raw.size} values for field_index0={field_index0}, "
        f"expected {nlev * rec_elems}"
    )
    return raw.reshape(nlev, COMPACT_NY, COMPACT_NX)


# ===========================================================================
# Stage 2: wet-aware horizontal 3:1 coarsening (new rule -- NOT
# coarsen_bathy_block's min-magnitude-over-wet, NOT coarsen_forcing_
# facet_mean's plain unweighted mean)
# ===========================================================================

def coarsen_ts_facet_wet_mean(arr, wet):
    """arr: (ny,nx) one level's T or S (post Stage-1 remap), still LLC90
    horizontal resolution. wet: (ny,nx) bool, Stage 2's fine wet mask, SAME
    level. Returns (value_coarse, wet_frac_coarse), each (ny//3, nx//3).
    """
    ny, nx = arr.shape
    factor = FACTOR
    cny, cnx = ny // factor, nx // factor
    blocks = arr.reshape(cny, factor, cnx, factor).transpose(0, 2, 1, 3).reshape(cny, cnx, factor * factor)
    wblocks = wet.reshape(cny, factor, cnx, factor).transpose(0, 2, 1, 3).reshape(cny, cnx, factor * factor)
    wblocks_f = wblocks.astype(np.float64)
    wet_frac_coarse = wblocks_f.mean(axis=-1)
    denom = wblocks_f.sum(axis=-1)
    numerator = (blocks * wblocks_f).sum(axis=-1)
    value_coarse = np.where(wblocks.any(axis=-1), numerator / np.maximum(denom, 1.0), 0.0)
    return value_coarse, wet_frac_coarse


def coarsen_field_horizontally(field_15_fine, wet15):
    """field_15_fine: (15, COMPACT_NY, COMPACT_NX) post-Stage-1 field.
    wet15: (15, COMPACT_NY, COMPACT_NX) Stage-2 wet mask.
    Returns (value_coarse, wet_frac_coarse), each
    (15, COMPACT_COARSE_NY, COMPACT_COARSE_NX).
    """
    nlev = field_15_fine.shape[0]
    value_levels = []
    wetfrac_levels = []
    for k in range(nlev):
        fine_facets = base.split_compact_into_facets(field_15_fine[k])
        wet_facets = base.split_compact_into_facets(wet15[k])
        coarse_facets = {}
        wf_facets = {}
        for facet in range(1, 6):
            coarse_facets[facet], wf_facets[facet] = coarsen_ts_facet_wet_mean(
                fine_facets[facet], wet_facets[facet]
            )
        value_levels.append(base.assemble_facets_into_compact(coarse_facets))
        wetfrac_levels.append(base.assemble_facets_into_compact(wf_facets))
    return np.stack(value_levels, axis=0), np.stack(wetfrac_levels, axis=0)


# ===========================================================================
# Correctness checks
# ===========================================================================

def spot_check_pickup_fill_values(source50, wet50, fine_bathy_compact,
                                   n_samples=20):
    """Before trusting wet50's exclusion, directly inspect raw pickup
    values at a handful of known-shallow fine columns, at levels wet50
    marks DRY, and assert they are small-magnitude / not a suspicious
    sentinel. This does not change the computation (those levels are
    already excluded) -- it validates that wet50's criterion actually
    lines up with where the real pickup data stops being physically
    meaningful, catching an off-by-one that bounds-preserved/byte-count
    checks would NOT catch (a quiet, not a loud, failure mode).
    """
    land = fine_bathy_compact == 0.0
    shallow_wet = (~land) & (fine_bathy_compact > -500.0) & (fine_bathy_compact < -10.0)
    js, iis = np.nonzero(shallow_wet)
    if js.size == 0:
        print("  (no shallow-wet columns found for fill-value spot-check -- skipping)")
        return
    rng = np.random.default_rng(0)
    idx = rng.choice(js.size, size=min(n_samples, js.size), replace=False)
    n_checked = 0
    for s in idx:
        j, i = js[s], iis[s]
        dry_levels = np.nonzero(~wet50[:, j, i])[0]
        if dry_levels.size == 0:
            continue
        k = dry_levels[0]  # shallowest dry level for this column
        raw_val = source50[k, j, i]
        n_checked += 1
        assert abs(raw_val) < 50.0, (
            f"Pickup fill-value sanity check FAILED at column (j={j},i={i}), "
            f"level {k}: raw value {raw_val} is implausibly large for a "
            f"'below local seafloor' cell -- possible off-by-one in "
            f"build_fine_wet_mask's rF[k] > seafloor comparison"
        )
        assert abs(abs(raw_val) - 999.0) > 1.0 and abs(abs(raw_val) - 9999.0) > 1.0, (
            f"Pickup fill-value sanity check FAILED at column (j={j},i={i}), "
            f"level {k}: raw value {raw_val} looks like a sentinel fill "
            f"value, not a benign small residual"
        )
    print(f"  Checked {n_checked} shallow-wet columns' shallowest-dry-level "
          f"raw value: all small-magnitude, no sentinel fill values. PASSED.")


def verify_ts_bounds_preserved(coarse_levels, fine_levels, wet_coarse, name):
    """Per level, coarse (wet) T/S must stay within that level's fine
    wet-cell min/max -- guaranteed by construction (nested weighted
    means), asserted anyway since a masking bug would show up here as an
    out-of-range value.
    """
    nlev = coarse_levels.shape[0]
    for k in range(nlev):
        fine_wet_vals = fine_levels[k][fine_levels[k] != 0.0]
        if fine_wet_vals.size == 0:
            continue
        fine_min, fine_max = fine_wet_vals.min(), fine_wet_vals.max()
        coarse_wet_mask = wet_coarse[k] > 0.0
        if not coarse_wet_mask.any():
            continue
        coarse_wet_vals = coarse_levels[k][coarse_wet_mask]
        assert coarse_wet_vals.min() >= fine_min - 1e-6, (
            f"{name} level {k}: coarse min {coarse_wet_vals.min():.4f} below "
            f"fine min {fine_min:.4f}"
        )
        assert coarse_wet_vals.max() <= fine_max + 1e-6, (
            f"{name} level {k}: coarse max {coarse_wet_vals.max():.4f} above "
            f"fine max {fine_max:.4f}"
        )
    print(f"  {name}: bounds-preserved check PASSED for all {nlev} levels.")


def verify_surface_wetfrac_vs_bathy_llc30(wet_frac_surface, llc30_bathy_path=LLC30_BATHY_PATH):
    """Compare global land fraction implied by this script's own surface
    wet mask against the already-committed input/BATHY_LLC30.bin. Different
    aggregation rules (wet-cell-count-of-9 vs. min-magnitude-over-wet) so
    exact agreement isn't expected -- assert within 5 percentage points,
    reported per-facet.
    """
    raw = np.fromfile(llc30_bathy_path, dtype=">f4")
    expected = COMPACT_COARSE_NX * COMPACT_COARSE_NY
    assert raw.size == expected, (llc30_bathy_path, raw.size, expected)
    bathy_compact = raw.reshape(COMPACT_COARSE_NY, COMPACT_COARSE_NX)
    bathy_land_frac_global = (bathy_compact == 0.0).mean()
    ts_land_frac_global = (wet_frac_surface == 0.0).mean()
    print(f"  global land fraction: BATHY_LLC30.bin={bathy_land_frac_global:.4f}  "
          f"this script's surface wet mask={ts_land_frac_global:.4f}")

    worst_diff = 0.0

    # split_compact_into_facets is written for LLC90-resolution
    # FACET_ROW_RANGES/LLC90_FACET_DIMS; at LLC30 coarse resolution the row
    # ranges and dims must be scaled down by FACTOR, mirroring
    # assemble_facets_into_compact's own scaling -- NOT a call to
    # split_compact_into_facets itself, which would use the wrong
    # (fine-resolution) dims.
    def _split_coarse(compact_2d):
        flat = compact_2d.reshape(-1)
        out = {}
        for facet in range(1, 6):
            r0, r1 = base.FACET_ROW_RANGES[facet]
            r0c, r1c = r0 // FACTOR, r1 // FACTOR
            nx30, ny30 = base.LLC30_FACET_DIMS[facet]
            start = r0c * COMPACT_COARSE_NX
            n = (r1c - r0c) * COMPACT_COARSE_NX
            out[facet] = flat[start:start + n].reshape(ny30, nx30)
        return out

    bathy_facets = _split_coarse(bathy_compact)
    ts_facets = _split_coarse(wet_frac_surface)
    for facet in range(1, 6):
        bf = (bathy_facets[facet] == 0.0).mean()
        tf = (ts_facets[facet] == 0.0).mean()
        diff = abs(bf - tf)
        worst_diff = max(worst_diff, diff)
        print(f"    facet{facet}: BATHY_LLC30 land_frac={bf:.4f}  "
              f"T/S surface land_frac={tf:.4f}  diff={diff:.4f}")
    assert worst_diff < 0.05, (
        f"Surface wet-fraction check FAILED: worst per-facet land-fraction "
        f"diff {worst_diff:.4f} exceeds 0.05 -- possible wet-mask bug"
    )
    print(f"  Surface wet-fraction-vs-BATHY_LLC30 check PASSED "
          f"(worst per-facet diff={worst_diff:.4f}).")


def spot_check_known_locations(theta_surface_coarse):
    """Use the already-coarsened tile00X.mitgrid's XC/YC (same lookup the
    plotting scripts already do) to find the LLC30 cell nearest a couple
    of known locations and assert surface Theta is in a physically sane
    range. Not rigorous validation, just a sanity check in the same spirit
    as the rest of this codebase's spot-checks.

    theta_surface_coarse is the FLAT compact (COMPACT_COARSE_NY,
    COMPACT_COARSE_NX) array -- for the wide facets (4,5, native nx30=90)
    a global (row,col) index into that flat array does NOT correspond to
    (j,i) within the facet's own (ny30,nx30) shape (same compact-layout
    gotcha as everywhere else in this codebase); split into facets first,
    then index with the facet's own local (j,i).
    """
    targets = [
        ("equatorial Pacific (0N,160W)", 0.0, -160.0, (24.0, 30.0)),
        ("Weddell Sea (-65N,-45E)", -65.0, -45.0, (-2.0, 4.0)),
    ]
    LLC30_FACET_DIMS = base.LLC30_FACET_DIMS

    def _split_coarse(compact_2d):
        flat = compact_2d.reshape(-1)
        out = {}
        for facet in range(1, 6):
            r0, r1 = base.FACET_ROW_RANGES[facet]
            r0c, r1c = r0 // FACTOR, r1 // FACTOR
            nx30, ny30 = LLC30_FACET_DIMS[facet]
            start = r0c * COMPACT_COARSE_NX
            n = (r1c - r0c) * COMPACT_COARSE_NX
            out[facet] = flat[start:start + n].reshape(ny30, nx30)
        return out

    theta_facets = _split_coarse(theta_surface_coarse)

    for label, lat0, lon0, (lo, hi) in targets:
        best = None
        for facet in range(1, 6):
            nx30, ny30 = LLC30_FACET_DIMS[facet]
            path = os.path.join(MITGRID_DIR, f"tile{facet:03d}.mitgrid")
            fields = base.read_mitgrid_tile(path, nx30, ny30)
            xc = fields["XC"][:ny30, :nx30]
            yc = fields["YC"][:ny30, :nx30]
            dist = (yc - lat0) ** 2 + (((xc - lon0 + 180) % 360 - 180)) ** 2
            j, i = np.unravel_index(np.argmin(dist), dist.shape)
            d = dist[j, i]
            if best is None or d < best[0]:
                best = (d, facet, j, i)
        _, facet, j, i = best
        val = theta_facets[facet][j, i]
        print(f"  {label}: nearest LLC30 cell (facet {facet}, j={j}, i={i}) "
              f"surface Theta = {val:.2f} C  (expected [{lo},{hi}])")
        assert lo <= val <= hi, (
            f"Spot-check FAILED for {label}: Theta={val:.2f}C not in "
            f"expected range [{lo},{hi}]"
        )
    print("  Known-location spot-checks PASSED.")


# ===========================================================================
# main
# ===========================================================================

def main():
    print("=" * 70)
    print("Step 1/6: vertical grids + overlap matrix")
    print("=" * 70)
    delR15 = parse_llc30_delR()
    print(f"  LLC30 delR (15 levels, from input/data): {delR15.tolist()}")
    rF50 = build_depth_interfaces(PRODUCTION_DELR_50)
    rF15 = build_depth_interfaces(delR15)
    W = build_overlap_matrix(rF15, rF50)
    row_sums = W.sum(axis=1)
    assert np.allclose(row_sums, delR15, rtol=1e-12), (
        "Overlap matrix row-sum check FAILED: W.sum(axis=1) does not equal "
        "delR15 -- interval-overlap arithmetic bug"
    )
    print(f"  Overlap matrix row-sum check PASSED (W.sum(axis=1) == delR15).")

    print()
    print("=" * 70)
    print("Step 2/6: read pickup Theta/Salt (records 100-149 / 150-199)")
    print("=" * 70)
    observed_in = os.path.getsize(PICKUP_SRC)
    expected_in = PICKUP_EXPECTED_RECORDS * COMPACT_NX * COMPACT_NY * 8
    assert observed_in == expected_in, (
        f"{PICKUP_SRC}: {observed_in} bytes, expected {expected_in} "
        f"({PICKUP_EXPECTED_RECORDS}*{COMPACT_NX}*{COMPACT_NY}*8)"
    )
    theta50 = read_pickup_field_levels(PICKUP_SRC, THETA_FIELD_INDEX0)
    salt50 = read_pickup_field_levels(PICKUP_SRC, SALT_FIELD_INDEX0)
    print(f"  Theta: level-0 min/max = [{theta50[0].min():.2f}, {theta50[0].max():.2f}] C "
          f"(expect physically sane SST)")
    print(f"  Theta: level-49 min/max = [{theta50[-1].min():.2f}, {theta50[-1].max():.2f}] C "
          f"(expect physically sane abyssal temp, different from level 0)")

    print()
    print("=" * 70)
    print("Step 3/6: fine bathymetry + 50-level wet mask + fill-value sanity check")
    print("=" * 70)
    observed_bathy = os.path.getsize(FINE_BATHY_SRC)
    expected_bathy = COMPACT_NX * COMPACT_NY * 4
    assert observed_bathy == expected_bathy, (
        f"{FINE_BATHY_SRC}: {observed_bathy} bytes, expected {expected_bathy}"
    )
    fine_bathy = np.fromfile(FINE_BATHY_SRC, dtype=">f4").astype(np.float64).reshape(
        COMPACT_NY, COMPACT_NX
    )
    wet50 = build_fine_wet_mask(fine_bathy, PRODUCTION_DELR_50)
    spot_check_pickup_fill_values(theta50, wet50, fine_bathy)

    print()
    print("=" * 70)
    print("Step 4/6: Stage 1 -- vertical remap, 50 -> 15 levels (still LLC90 horizontal res)")
    print("=" * 70)
    N = COMPACT_NY * COMPACT_NX
    theta15_fine, _ = vertical_remap_overlap_weighted(
        theta50.reshape(PICKUP_NLEV, N), rF15, rF50,
        wet_mask=wet50.reshape(PICKUP_NLEV, N),
    )
    salt15_fine, _ = vertical_remap_overlap_weighted(
        salt50.reshape(PICKUP_NLEV, N), rF15, rF50,
        wet_mask=wet50.reshape(PICKUP_NLEV, N),
    )
    theta15_fine = theta15_fine.reshape(15, COMPACT_NY, COMPACT_NX)
    salt15_fine = salt15_fine.reshape(15, COMPACT_NY, COMPACT_NX)
    print(f"  Stage-1 vertical remap done for Theta and Salt (15 levels, "
          f"still ({COMPACT_NY},{COMPACT_NX}) horizontal resolution).")

    print()
    print("=" * 70)
    print("Step 5/6: Stage 2 -- fresh 15-level wet mask + wet-aware horizontal coarsening")
    print("=" * 70)
    wet15 = build_fine_wet_mask(fine_bathy, delR15)  # FRESH mask, NOT wet50
    theta_coarse, theta_wetfrac = coarsen_field_horizontally(theta15_fine, wet15)
    salt_coarse, salt_wetfrac = coarsen_field_horizontally(salt15_fine, wet15)
    print(f"  Stage-2 horizontal coarsening done: Theta/Salt now "
          f"(15,{COMPACT_COARSE_NY},{COMPACT_COARSE_NX}).")

    print()
    print("=" * 70)
    print("Step 6/6: write binaries, remap tRef/sRef, correctness checks")
    print("=" * 70)
    os.makedirs(OUT_IC_DIR, exist_ok=True)

    def _write(arr, name):
        out_path = os.path.join(OUT_IC_DIR, name)
        expected_out = 15 * COMPACT_COARSE_NY * COMPACT_COARSE_NX * 4
        out_bytes = arr.astype(">f4").tobytes()
        assert len(out_bytes) == expected_out, (
            f"{name}: assembled {len(out_bytes)} bytes, expected {expected_out}"
        )
        with open(out_path, "wb") as fh:
            fh.write(out_bytes)
        observed_out = os.path.getsize(out_path)
        assert observed_out == expected_out, (
            f"{name}: written {observed_out} bytes, expected {expected_out}"
        )
        print(f"  {name}: {observed_out} bytes OK (15 x {COMPACT_COARSE_NY}x{COMPACT_COARSE_NX} records)")

    _write(theta_coarse, "THETA_LLC30_INIT.bin")
    _write(salt_coarse, "SALT_LLC30_INIT.bin")

    tref15, _ = vertical_remap_overlap_weighted(
        PRODUCTION_TREF_50.reshape(50, 1), rF15, rF50
    )
    sref15, _ = vertical_remap_overlap_weighted(
        PRODUCTION_SREF_50.reshape(50, 1), rF15, rF50
    )
    tref15 = tref15.reshape(15)
    sref15 = sref15.reshape(15)
    assert np.allclose(sref15, 34.5), (
        f"sRef regression check FAILED: expected flat 34.5, got {sref15}"
    )
    print(f"  sRef trivial-regression check PASSED (flat 34.5 preserved under remap).")

    tref_str = "tRef= " + ",".join(f"{v:.4f}" for v in tref15) + ",\n"
    sref_str = "sRef= " + "15*34.5,\n"
    txt_path = os.path.join(OUT_IC_DIR, "tRef_sRef_llc30.txt")
    with open(txt_path, "w") as fh:
        fh.write(tref_str)
        fh.write(sref_str)
    print(f"  Wrote paste-ready {txt_path}:")
    print(f"    {tref_str.strip()}")
    print(f"    {sref_str.strip()}")

    print()
    verify_ts_bounds_preserved(theta_coarse, theta15_fine, theta_wetfrac, "Theta")
    verify_ts_bounds_preserved(salt_coarse, salt15_fine, salt_wetfrac, "Salt")
    verify_surface_wetfrac_vs_bathy_llc30(theta_wetfrac[0])
    spot_check_known_locations(theta_coarse[0])

    print()
    print("ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
