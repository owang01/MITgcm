#!/usr/bin/env python3
"""Build a from-scratch, GLOBAL (single-file, not per-tile), 15-level,
named-field LLC30 pickup file (`pickup.0000000000.data/.meta`), so this
config's cold start can switch from `hydrogThetaFile`/`hydrogSaltFile`/
`pSurfInitFile` over to MITgcm's real `pickupSuff` mechanism -- the user
explicitly chose this over the simpler `pSurfInitFile`-only route (see
HANDOFF.md's "Pickup-file research" section) specifically to exercise
MITgcm's pickup-read code path, and explicitly wants every field (not
just Theta/Salt/Eta) to match real LLC90/V4r6 as closely as possible --
i.e. NOT a cold start with zero velocity.

Writes a SINGLE global file per pickup (matching real V4r6's own
pickup.0000000001.data/.meta convention exactly, confirmed byte-for-byte
against that real file's .meta), not 13 per-tile files -- this config's
`input/data` now sets `useSingleCpuIO=.TRUE.` (also matching real V4r6),
which requires every mdsio-read file (including pickups) to be a global
file; there is no tiled-file fallback once that flag is set (confirmed
in pkg/mdsio/mdsio_read_field.F's `useSingleCPUIO .AND. globFile.GE.0`
branch, which only ever opens `dataFName` -- the global name -- and
STOPs if that specific file is missing, with no per-tile retry). See
HANDOFF.md's "global-file I/O" section for the full investigation.

Required fields (confirmed by reading model/src/read_pickup.F/
check_pickup.F for this config's exact CPP/runtime flags -- see
preprocess/README.md's "Pickup-file" section for the full derivation):
  Uvel, Vvel, Theta, Salt, GuNm1, GvNm1, EtaN, dEtaHdt, EtaH
(no GtNm1/GsNm1: AdamsBashforthGt/Gs are both FALSE for this config's
tempAdvScheme=2/doAB_onGtGs=.TRUE. combination; no AddMass/PhiHyd/etc:
all gated off by CPP flags or runtime params this config never sets).

Source data (read-only, never modified) -- same V4r6 production pickup
Theta/Salt/EtaN already coarsened from, reused here for EVERY field:
  /ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/ancillary_data/
  ancillary_data_orig/input_init/pickup.0000000001.data
Field order (PICKUP_FIELD_ORDER, from coarsen_initial_ts_to_llc30.py,
confirmed against the real file): Uvel,Vvel,Theta,Salt,GuNm1,GuNm2,GvNm1,
GvNm2,AddMass (50 levels each) then EtaN,dEtaHdt,EtaH (2-D) -- Uvel=
records 0-49, Vvel=50-99, GuNm1=200-249, GvNm1=300-349, EtaN=450,
dEtaHdt=451, EtaH=452.

Theta/Salt are NOT recomputed here -- read back the already-coarsened,
already-verified ../input/THETA_LLC30_INIT.bin/SALT_LLC30_INIT.bin
(produced by coarsen_initial_ts_to_llc30.py) and reshape into tiles.

Uvel/Vvel/GuNm1/GvNm1 are grid-relative, staggered-grid VECTOR components
-- unlike every scalar field this pipeline has coarsened so far, these
need a rotate/average/rotate-back procedure (see module docstring of
remap_velocity_pair_llc90_to_llc30 below) mirroring MITgcm's own
model/src/rotate_uv2en.F, with a facet-seam-aware fix for the cells
directly abutting an internal facet connection (mirroring
pkg/exch2/exch2_uv_cgrid_3d_r8.F's useCubedSphereExchange=.TRUE. U<->V
swap+sign-flip at a "rotated" seam -- the extra one-cell index shift that
routine also applies is deliberately NOT replicated here, a user-approved
scope decision: that shift only affects the single row/column of fine
cells directly on a seam, and its exact direction is hard to transcribe
correctly from Fortran array-index conventions without a live MITgcm run
to validate against).

An explicit land/ocean-mask safeguard (user's explicit ask, since
MITgcm's own READ_PICKUP/CHECK_PICKUP do ZERO masking/consistency
checking on a pickup file) derives this config's own hFacC/hFacW/hFacS
from ../input/BATHY_LLC30.bin + ../input/data's delR, cross-checks them
against this config's own already-committed ../run/hFacC.*.001.data
(etc.) model output, and explicitly zeros + asserts zero land-cell
leakage before writing.

Run:
    export PATH=$HOME/micromamba/envs/podaac/bin:$PATH
    cd MITgcm/verification/global_ocean.llc30x15/preprocess
    python3 build_llc30_pickup.py
"""

import os
import re
import sys

import numpy as np

import coarsen_llc90_to_llc30 as base
import coarsen_initial_ts_to_llc30 as ts

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PICKUP_SRC = ts.PICKUP_SRC
V4R6_GRID_DIR = (
    "/mnt/extraid/ifraid8/data07/owang/Ecco_data/v4/JPL/input/grid/V4r6"
)
LLC90_ANGLECS_PATH = os.path.join(V4R6_GRID_DIR, "AngleCS.data")
LLC90_ANGLESN_PATH = os.path.join(V4R6_GRID_DIR, "AngleSN.data")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "..", "input")
RUN_DIR = os.path.join(SCRIPT_DIR, "..", "run")
LLC30_DATA_PATH = os.path.join(INPUT_DIR, "data")
LLC30_BATHY_PATH = os.path.join(INPUT_DIR, "BATHY_LLC30.bin")
ICE_DRAFT_PATH = os.path.join(INPUT_DIR, "ICE_DRAFT_LLC30.bin")
THETA_LLC30_PATH = os.path.join(INPUT_DIR, "THETA_LLC30_INIT.bin")
SALT_LLC30_PATH = os.path.join(INPUT_DIR, "SALT_LLC30_INIT.bin")

OUT_DIR = os.path.join(SCRIPT_DIR, "output")
OUT_PICKUP_DIR = os.path.join(OUT_DIR, "pickup")

COMPACT_NX = base.COMPACT_NX                  # 90
COMPACT_NY = base.COMPACT_NY                  # 1170
COMPACT_COARSE_NX = base.COMPACT_COARSE_NX    # 30
COMPACT_COARSE_NY = base.COMPACT_COARSE_NY    # 390
FACTOR = base.FACTOR                          # 3
NLEV30 = 15

# ---------------------------------------------------------------------------
# Pickup field record indices (reuse ts.PICKUP_FIELD_ORDER, add EtaN/
# dEtaHdt/EtaH indices which coarsen_initial_ts_to_llc30.py didn't need)
# ---------------------------------------------------------------------------

UVEL_IDX0 = ts.PICKUP_FIELD_ORDER.index("Uvel")     # 0
VVEL_IDX0 = ts.PICKUP_FIELD_ORDER.index("Vvel")     # 1
GUNM1_IDX0 = ts.PICKUP_FIELD_ORDER.index("GuNm1")   # 4
GVNM1_IDX0 = ts.PICKUP_FIELD_ORDER.index("GvNm1")   # 6

PICKUP_NLEV = ts.PICKUP_NLEV  # 50
N3D_FIELDS = len(ts.PICKUP_FIELD_ORDER)  # 9 (Uvel..AddMass)
ETAN_RECORD = PICKUP_NLEV * N3D_FIELDS        # 450
DETAHDT_RECORD = ETAN_RECORD + 1              # 451
ETAH_RECORD = ETAN_RECORD + 2                 # 452

# ---------------------------------------------------------------------------
# Tile <-> facet mapping (per plots/plot_llc_tile_layout.py's own comment
# block -- facet1->tiles1-3, facet2->tiles4-6, facet3->tile7,
# facet4->tiles8-10, facet5->tiles11-13). Facets 1/2 (native (30,90) at
# LLC30 res) stack 3 tiles along Y; facets 4/5 (native (90,30)) stack 3
# tiles along X; facet 3 (native (30,30)) is already exactly one tile.
# ---------------------------------------------------------------------------

FACET_TILE_RANGE = {1: (1, 3), 2: (4, 6), 3: (7, 7), 4: (8, 10), 5: (11, 13)}
FACET_STACK_AXIS = {1: "y", 2: "y", 3: None, 4: "x", 5: "x"}

TILE_NY, TILE_NX = 30, 30

# ---------------------------------------------------------------------------
# Output fields for the pickup, in write order (per model/src/write_pickup.F
# for this config's exact CPP flags -- see module docstring)
# ---------------------------------------------------------------------------

PICKUP_OUT_FIELDS_3D = ["Uvel", "Vvel", "Theta", "Salt", "GuNm1", "GvNm1"]
PICKUP_OUT_FIELDS_2D = ["EtaN", "dEtaHdt", "EtaH"]
PICKUP_OUT_FIELDS = PICKUP_OUT_FIELDS_3D + PICKUP_OUT_FIELDS_2D
N_RECORDS_PER_TILE = len(PICKUP_OUT_FIELDS_3D) * NLEV30 + len(PICKUP_OUT_FIELDS_2D)
assert N_RECORDS_PER_TILE == 93, N_RECORDS_PER_TILE


def split_facet_into_tiles(facet_arr, facet):
    """facet_arr: (ny30,nx30) (or (nlev,ny30,nx30)) at LLC30 facet native
    resolution. Returns dict tile_number -> same-shape-minus-facet-dims
    (30,30) (or (nlev,30,30)) block, per FACET_TILE_RANGE/FACET_STACK_AXIS.
    Confirmed against this config's own already-committed run/XC.<tile>.
    001.data output (see build_llc30_pickup's module docstring / plan).
    """
    t0, t1 = FACET_TILE_RANGE[facet]
    axis = FACET_STACK_AXIS[facet]
    out = {}
    if axis is None:
        assert t0 == t1
        out[t0] = facet_arr
        return out
    n_tiles = t1 - t0 + 1
    if axis == "y":
        # facet_arr's last-2 dims: (ny30, nx30) = (n_tiles*30, 30)
        chunks = np.split(facet_arr, n_tiles, axis=-2)
    else:
        # axis == "x": (ny30, nx30) = (30, n_tiles*30)
        chunks = np.split(facet_arr, n_tiles, axis=-1)
    for i, tile in enumerate(range(t0, t1 + 1)):
        out[tile] = chunks[i]
    return out


def split_compact_coarse_into_facets(arr2d):
    """arr2d: (COMPACT_COARSE_NY, COMPACT_COARSE_NX). Returns dict
    facet(1..5) -> (ny30,nx30) block, at LLC30 (coarse) resolution --
    base.split_compact_into_facets is hardcoded to LLC90 (fine)
    dimensions and cannot be reused directly for a coarse-resolution
    array (confirmed: raises a reshape error if you try). Same
    FACET_ROW_RANGES-scaled-by-FACTOR convention as
    coarsen_initial_ts_to_llc30.py's own
    verify_surface_wetfrac_vs_bathy_llc30._split_coarse helper and this
    module's compact_coarse_to_tiles (both independently needed this same
    logic -- factored out here as the single shared implementation).
    """
    flat = arr2d.reshape(-1)
    out = {}
    for facet in range(1, 6):
        r0, r1 = base.FACET_ROW_RANGES[facet]
        r0c, r1c = r0 // FACTOR, r1 // FACTOR
        nx30, ny30 = base.LLC30_FACET_DIMS[facet]
        start = r0c * COMPACT_COARSE_NX
        n = (r1c - r0c) * COMPACT_COARSE_NX
        out[facet] = flat[start:start + n].reshape(ny30, nx30)
    return out


def compact_coarse_to_tiles(compact_arr):
    """compact_arr: (nlev,COMPACT_COARSE_NY,COMPACT_COARSE_NX) or
    (COMPACT_COARSE_NY,COMPACT_COARSE_NX). Returns dict tile(1..13) ->
    (nlev,30,30) or (30,30) block, via split_compact_coarse_into_facets
    then split_facet_into_tiles.
    """
    is_3d = compact_arr.ndim == 3
    nlev = compact_arr.shape[0] if is_3d else 1
    arrs = compact_arr if is_3d else compact_arr[None]

    tiles = {t: [] for t in range(1, 14)}
    for k in range(nlev):
        facets_k = split_compact_coarse_into_facets(arrs[k])
        for facet in range(1, 6):
            tile_blocks = split_facet_into_tiles(facets_k[facet], facet)
            for tile, block in tile_blocks.items():
                tiles[tile].append(block)
    if is_3d:
        return {t: np.stack(blocks, axis=0) for t, blocks in tiles.items()}
    return {t: blocks[0] for t, blocks in tiles.items()}


def tiles_to_compact_coarse(tiles_dict):
    """Inverse of compact_coarse_to_tiles: dict tile(1..13) -> (30,30) or
    (nlev,30,30) block -> (COMPACT_COARSE_NY,COMPACT_COARSE_NX) or
    (nlev,COMPACT_COARSE_NY,COMPACT_COARSE_NX) compact array. Generalizes
    the inline tile-reassembly logic originally written just for
    AngleCS/AngleSN (_llc30_angle_cos_sin_compact's old _tiles_to_compact
    helper) to any tiled field, now needed for the global-file pickup
    writer too (see module docstring)."""
    sample = tiles_dict[1]
    is_3d = sample.ndim == 3

    def _tiles_to_facets(nlev_index=None):
        facets = {}
        for facet in range(1, 6):
            t0, t1 = FACET_TILE_RANGE[facet]
            axis = FACET_STACK_AXIS[facet]

            def _get(t):
                block = tiles_dict[t]
                return block[nlev_index] if nlev_index is not None else block

            if axis is None:
                facets[facet] = _get(t0)
            elif axis == "y":
                facets[facet] = np.concatenate(
                    [_get(t) for t in range(t0, t1 + 1)], axis=0
                )
            else:
                facets[facet] = np.concatenate(
                    [_get(t) for t in range(t0, t1 + 1)], axis=1
                )
        return facets

    if not is_3d:
        facets = _tiles_to_facets()
        return base.assemble_facets_into_compact(facets)

    nlev = sample.shape[0]
    levels = []
    for k in range(nlev):
        facets_k = _tiles_to_facets(nlev_index=k)
        levels.append(base.assemble_facets_into_compact(facets_k))
    return np.stack(levels, axis=0)


# ===========================================================================
# Scalar (Theta/Salt/EtaN/dEtaHdt/EtaH) loading
# ===========================================================================

def load_theta_salt_tiles():
    """Read back the already-coarsened, already-verified
    THETA_LLC30_INIT.bin/SALT_LLC30_INIT.bin (real*4 big-endian, compact
    (15,390,30)) and reshape into 13 tiles x 15 levels x 30x30."""
    expected = NLEV30 * COMPACT_COARSE_NY * COMPACT_COARSE_NX
    theta_raw = np.fromfile(THETA_LLC30_PATH, dtype=">f4").astype(np.float64)
    salt_raw = np.fromfile(SALT_LLC30_PATH, dtype=">f4").astype(np.float64)
    assert theta_raw.size == expected, (THETA_LLC30_PATH, theta_raw.size, expected)
    assert salt_raw.size == expected, (SALT_LLC30_PATH, salt_raw.size, expected)
    theta_compact = theta_raw.reshape(NLEV30, COMPACT_COARSE_NY, COMPACT_COARSE_NX)
    salt_compact = salt_raw.reshape(NLEV30, COMPACT_COARSE_NY, COMPACT_COARSE_NX)
    return compact_coarse_to_tiles(theta_compact), compact_coarse_to_tiles(salt_compact)


def coarsen_scalar_2d_field(record_index):
    """Read one 2-D record (EtaN=450/dEtaHdt=451/EtaH=452) from the real
    pickup, coarsen horizontally 3:1 the SAME way EtaN was already done
    (coarsen_ts_facet_wet_mean/coarsen_field_horizontally with nlev=1),
    and return dict tile -> (30,30) coarse array.
    """
    rec_elems = COMPACT_NX * COMPACT_NY
    rec_bytes = rec_elems * 8
    with open(PICKUP_SRC, "rb") as fh:
        fh.seek(record_index * rec_bytes)
        raw = np.fromfile(fh, dtype=">f8", count=rec_elems)
    assert raw.size == rec_elems, (PICKUP_SRC, record_index, raw.size, rec_elems)
    fine = raw.reshape(1, COMPACT_NY, COMPACT_NX)

    fine_bathy = _load_fine_bathy()
    delR15 = ts.parse_llc30_delR()
    wet15 = ts.build_fine_wet_mask(fine_bathy, delR15)  # (15, NY, NX)
    wet_surface = wet15[0:1]  # (1, NY, NX), reuse level-0 wet mask (Eta has no depth)

    coarse, _ = ts.coarsen_field_horizontally(fine, wet_surface)  # (1, 390, 30)
    return compact_coarse_to_tiles(coarse[0])  # dict tile -> (30,30)


_fine_bathy_cache = None


def _load_fine_bathy():
    global _fine_bathy_cache
    if _fine_bathy_cache is None:
        observed = os.path.getsize(ts.FINE_BATHY_SRC)
        expected = COMPACT_NX * COMPACT_NY * 4
        assert observed == expected, (ts.FINE_BATHY_SRC, observed, expected)
        _fine_bathy_cache = np.fromfile(ts.FINE_BATHY_SRC, dtype=">f4").astype(
            np.float64
        ).reshape(COMPACT_NY, COMPACT_NX)
    return _fine_bathy_cache


# ===========================================================================
# LLC90 AngleCS/AngleSN (real, already-computed -- see reference memory
# reference_v4r6_llc90_grid_files.md; NOT re-derived from mitgrid files)
# ===========================================================================

def load_llc90_angle_cos_sin():
    """Returns (angleCosC, angleSinC), each (COMPACT_NY, COMPACT_NX) real*4
    big-endian compact arrays, read directly from the user-provided real
    V4r6 LLC90 grid file set."""
    expected = COMPACT_NX * COMPACT_NY * 4
    for path in (LLC90_ANGLECS_PATH, LLC90_ANGLESN_PATH):
        observed = os.path.getsize(path)
        assert observed == expected, (path, observed, expected)
    cos_c = np.fromfile(LLC90_ANGLECS_PATH, dtype=">f4").astype(np.float64).reshape(
        COMPACT_NY, COMPACT_NX
    )
    sin_c = np.fromfile(LLC90_ANGLESN_PATH, dtype=">f4").astype(np.float64).reshape(
        COMPACT_NY, COMPACT_NX
    )
    assert cos_c.min() >= -1.0001 and cos_c.max() <= 1.0001, (cos_c.min(), cos_c.max())
    assert sin_c.min() >= -1.0001 and sin_c.max() <= 1.0001, (sin_c.min(), sin_c.max())
    return cos_c, sin_c


def load_llc30_angle_cos_sin_compact():
    """Returns (cos_compact, sin_compact), each (COMPACT_COARSE_NY,
    COMPACT_COARSE_NX) real*4 array, read directly from this config's own
    already-committed GLOBAL run/AngleCS.data / AngleSN.data model
    output (useSingleCpuIO=.TRUE. -- see module docstring)."""
    cpath = os.path.join(RUN_DIR, "AngleCS.data")
    spath = os.path.join(RUN_DIR, "AngleSN.data")
    expected = COMPACT_COARSE_NY * COMPACT_COARSE_NX
    cos_raw = np.fromfile(cpath, dtype=">f4").astype(np.float64)
    sin_raw = np.fromfile(spath, dtype=">f4").astype(np.float64)
    assert cos_raw.size == expected, (cpath, cos_raw.size, expected)
    assert sin_raw.size == expected, (spath, sin_raw.size, expected)
    return (
        cos_raw.reshape(COMPACT_COARSE_NY, COMPACT_COARSE_NX),
        sin_raw.reshape(COMPACT_COARSE_NY, COMPACT_COARSE_NX),
    )


# ===========================================================================
# Un-staggering (SAME-FACET approximation -- mid-implementation
# correction, 2026-11-08)
#
# MITgcm convention: Uvel(i) is tracer cell i's WEST face, Vvel(j) is
# tracer cell j's SOUTH face -- so cell (j,i)'s own two bounding U-faces
# are Uvel(i) and Uvel(i+1); only the very last column (i=nx-1, 0-indexed)
# is missing its "i+1" (EAST) face within this facet's own array (same
# for V's missing NORTH face at the last row). Confirmed against
# rotate_uv2en.F's own un-stagger formula (`tmpU(i,j) =
# 0.5*(uFldX(i+1,j)+uFldX(i,j))`, which needs i+1 -- this cell's own
# east-side face).
#
# The original plan called for a facet-seam-aware fix mirroring
# pkg/exch2/exch2_uv_cgrid_3d_r8.F's U<->V-swap-with-sign-flip at
# "unlike-oriented" (reversed_=True) seam connections. TWO independent
# implementation attempts at the exact swap/sign logic each introduced a
# real bug, caught by a synthetic correctness test (feed in a KNOWN
# uniform true-eastward flow, rotate+stagger it onto the real LLC90
# grid, run it through this function, rotate back, check the output
# recovers the known input) -- the second attempt's "fix" was actually
# WORSE than no fix at all (mean error 0.0037 vs. 0.0005 for the plain
# same-facet approximation; max error 1.22 vs. 0.54). The user,
# presented with this evidence, chose to drop the swap-based fix
# entirely (see plan) rather than continue debugging it -- the LAST
# row/column of every facet now simply uses a one-sided value, exactly
# like a true open domain boundary, at EVERY seam (not just genuinely
# open ones). Error is bounded to the single row/column of fine cells
# directly on a seam (confirmed by the synthetic test: ~0.2% of all
# cells have >0.05 error).
# ===========================================================================

def unstagger_uv_to_center_llc90(uvel_fine, vvel_fine):
    """uvel_fine, vvel_fine: (nlev, COMPACT_NY, COMPACT_NX) real LLC90
    pickup Uvel/Vvel. Returns (uC, vC), same shape, at cell centers, using
    the same-facet (one-sided at every facet edge) approximation -- see
    module section docstring above for why this replaced the originally-
    planned facet-seam swap fix.
    """
    nlev = uvel_fine.shape[0]
    u_by_facet = {f: np.stack(
        [base.split_compact_into_facets(uvel_fine[k])[f] for k in range(nlev)], axis=0
    ) for f in range(1, 6)}
    v_by_facet = {f: np.stack(
        [base.split_compact_into_facets(vvel_fine[k])[f] for k in range(nlev)], axis=0
    ) for f in range(1, 6)}

    uc_by_facet = {}
    vc_by_facet = {}
    for facet in range(1, 6):
        u = u_by_facet[facet]  # (nlev, ny, nx)
        v = v_by_facet[facet]

        # Interior un-stagger: uC[...,i] = 0.5*(u[...,i] + u[...,i+1]) for
        # i < nx-1; vC[...,j,:] = 0.5*(v[...,j,:] + v[...,j+1,:]) for
        # j < ny-1. The LAST column/row (i=nx-1 / j=ny-1) is missing its
        # "+1" neighbor within this facet; use the one-sided value there
        # (same-facet approximation, see module docstring).
        uC = np.empty_like(u)
        vC = np.empty_like(v)
        uC[:, :, :-1] = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
        uC[:, :, -1] = u[:, :, -1]
        vC[:, :-1, :] = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
        vC[:, -1, :] = v[:, -1, :]

        uc_by_facet[facet] = uC
        vc_by_facet[facet] = vC

    uC_levels = []
    vC_levels = []
    for k in range(nlev):
        uC_levels.append(_assemble_fine_facets_into_compact(
            {f: uc_by_facet[f][k] for f in range(1, 6)}
        ))
        vC_levels.append(_assemble_fine_facets_into_compact(
            {f: vc_by_facet[f][k] for f in range(1, 6)}
        ))
    return np.stack(uC_levels, axis=0), np.stack(vC_levels, axis=0)


def _assemble_fine_facets_into_compact(facets_fine):
    """Inverse of base.split_compact_into_facets, at LLC90 FINE resolution
    (base.assemble_facets_into_compact is written for LLC30-coarse-
    resolution dims only -- this is the fine-resolution analogue)."""
    out_flat = np.empty(COMPACT_NY * COMPACT_NX, dtype=facets_fine[1].dtype)
    for facet in range(1, 6):
        nx, ny = base.LLC90_FACET_DIMS[facet]
        r0, r1 = base.FACET_ROW_RANGES[facet]
        n = (r1 - r0) * COMPACT_NX
        assert n == nx * ny, (facet, n, nx, ny)
        chunk = facets_fine[facet]
        assert chunk.shape == (ny, nx), (facet, chunk.shape, (ny, nx))
        start = r0 * COMPACT_NX
        out_flat[start:start + n] = chunk.reshape(-1)
    return out_flat.reshape(COMPACT_NY, COMPACT_NX)


# ===========================================================================
# Rotation to/from east-north (model/src/rotate_uv2en.F's exact formulas)
# ===========================================================================

def rotate_xy_to_en(uC, vC, cos_c, sin_c):
    """uC,vC: (nlev,NY,NX) grid-relative cell-center components. cos_c,
    sin_c: (NY,NX) angleCosC/angleSinC. Returns (uE,vN) -- EXACT formula
    from rotate_uv2en.F's xy2en=.TRUE. branch:
      uFldE = angleCosC*tmpU - angleSinC*tmpV
      vFldN = angleSinC*tmpU + angleCosC*tmpV
    """
    uE = cos_c[None, :, :] * uC - sin_c[None, :, :] * vC
    vN = sin_c[None, :, :] * uC + cos_c[None, :, :] * vC
    return uE, vN


def rotate_en_to_xy(uE, vN, cos_c, sin_c):
    """Inverse of rotate_xy_to_en -- EXACT formula from rotate_uv2en.F's
    xy2en=.FALSE. branch:
      tmpU =  angleCosC*uFldE + angleSinC*vFldN
      tmpV = -angleSinC*uFldE + angleCosC*vFldN
    """
    uC = cos_c[None, :, :] * uE + sin_c[None, :, :] * vN
    vC = -sin_c[None, :, :] * uE + cos_c[None, :, :] * vN
    return uC, vC


def restagger_center_to_uv_llc30(uC30, vC30):
    """uC30,vC30: (nlev,COMPACT_COARSE_NY,COMPACT_COARSE_NX) LLC30
    grid-relative cell-center components (post rotate-back). Returns
    (Uvel30,Vvel30), same shape, at LLC30's own U/V face points -- mirror
    of unstagger_uv_to_center_llc90's un-stagger, exact formula from
    rotate_uv2en.F's switchGrid=.TRUE., xy2en=.FALSE. branch:
      uFldX(i,j) = 0.5*(tmpU(i-1,j) + tmpU(i,j))
      vFldY(i,j) = 0.5*(tmpV(i,j-1) + tmpV(i,j))
    i.e. Uvel(i) (this cell's WEST face) = avg of THIS cell's center and
    the WEST-neighbor's center. Same-facet (one-sided at the FIRST
    column/row of each facet) approximation -- see
    unstagger_uv_to_center_llc90's module section docstring for why the
    originally-planned facet-seam swap fix was dropped in favor of this
    simpler, verified-better approach.
    """
    nlev = uC30.shape[0]
    uC_by_facet = {f: np.stack(
        [split_compact_coarse_into_facets(uC30[k])[f] for k in range(nlev)], axis=0
    ) for f in range(1, 6)}
    vC_by_facet = {f: np.stack(
        [split_compact_coarse_into_facets(vC30[k])[f] for k in range(nlev)], axis=0
    ) for f in range(1, 6)}

    u_out_by_facet = {}
    v_out_by_facet = {}
    for facet in range(1, 6):
        uC = uC_by_facet[facet]  # (nlev, ny, nx)
        vC = vC_by_facet[facet]

        Uvel = np.empty_like(uC)
        Vvel = np.empty_like(vC)
        # Interior: Uvel(i) = 0.5*(uC(i-1) + uC(i)) for i >= 1. The FIRST
        # column/row (i=0/j=0) is missing its "-1" neighbor; use the
        # one-sided value there (same-facet approximation).
        Uvel[:, :, 1:] = 0.5 * (uC[:, :, :-1] + uC[:, :, 1:])
        Uvel[:, :, 0] = uC[:, :, 0]
        Vvel[:, 1:, :] = 0.5 * (vC[:, :-1, :] + vC[:, 1:, :])
        Vvel[:, 0, :] = vC[:, 0, :]

        u_out_by_facet[facet] = Uvel
        v_out_by_facet[facet] = Vvel

    u_levels = []
    v_levels = []
    for k in range(nlev):
        u_levels.append(base.assemble_facets_into_compact(
            {f: u_out_by_facet[f][k] for f in range(1, 6)}
        ))
        v_levels.append(base.assemble_facets_into_compact(
            {f: v_out_by_facet[f][k] for f in range(1, 6)}
        ))
    return np.stack(u_levels, axis=0), np.stack(v_levels, axis=0)


# ===========================================================================
# Full Uvel/Vvel (and GuNm1/GvNm1) remap: un-stagger -> rotate to E/N ->
# vertical remap 50->15 -> horizontal 3:1 coarsen -> rotate back ->
# re-stagger onto LLC30's own U/V points
# ===========================================================================

def remap_velocity_pair_llc90_to_llc30(uvel_idx0, vvel_idx0):
    """uvel_idx0/vvel_idx0: PICKUP_FIELD_ORDER index0 for the U/V-pair to
    remap (0/1 for Uvel/Vvel; 4/6 for GuNm1/GvNm1). Returns dict tile ->
    (Uvel30, Vvel30), each (15,30,30) LLC30-resolution, grid-relative,
    properly-staggered arrays, masked at dry hFacW/hFacS cells by the
    caller (not here)."""
    uvel_fine = ts.read_pickup_field_levels(PICKUP_SRC, uvel_idx0)
    vvel_fine = ts.read_pickup_field_levels(PICKUP_SRC, vvel_idx0)

    uC, vC = unstagger_uv_to_center_llc90(uvel_fine, vvel_fine)

    cos90, sin90 = load_llc90_angle_cos_sin()
    uE, vN = rotate_xy_to_en(uC, vC, cos90, sin90)

    fine_bathy = _load_fine_bathy()
    wet50 = ts.build_fine_wet_mask(fine_bathy, ts.PRODUCTION_DELR_50)
    delR15 = ts.parse_llc30_delR()
    rF50 = ts.build_depth_interfaces(ts.PRODUCTION_DELR_50)
    rF15 = ts.build_depth_interfaces(delR15)

    N = COMPACT_NY * COMPACT_NX
    uE15_fine, _ = ts.vertical_remap_overlap_weighted(
        uE.reshape(PICKUP_NLEV, N), rF15, rF50, wet_mask=wet50.reshape(PICKUP_NLEV, N)
    )
    vN15_fine, _ = ts.vertical_remap_overlap_weighted(
        vN.reshape(PICKUP_NLEV, N), rF15, rF50, wet_mask=wet50.reshape(PICKUP_NLEV, N)
    )
    uE15_fine = uE15_fine.reshape(NLEV30, COMPACT_NY, COMPACT_NX)
    vN15_fine = vN15_fine.reshape(NLEV30, COMPACT_NY, COMPACT_NX)

    wet15 = ts.build_fine_wet_mask(fine_bathy, delR15)
    uE_coarse, _ = ts.coarsen_field_horizontally(uE15_fine, wet15)
    vN_coarse, _ = ts.coarsen_field_horizontally(vN15_fine, wet15)

    cos30_compact, sin30_compact = _llc30_angle_cos_sin_compact()
    uC30, vC30 = rotate_en_to_xy(uE_coarse, vN_coarse, cos30_compact, sin30_compact)

    Uvel30, Vvel30 = restagger_center_to_uv_llc30(uC30, vC30)

    return compact_coarse_to_tiles(Uvel30), compact_coarse_to_tiles(Vvel30), (
        uE_coarse, vN_coarse,
    )


def _llc30_angle_cos_sin_compact():
    """LLC30's own AngleCS/AngleSN in compact (COMPACT_COARSE_NY,
    COMPACT_COARSE_NX) form, for use by rotate_en_to_xy (which operates
    on whole-compact arrays like every other scalar field in this
    pipeline). Thin wrapper kept for call-site compatibility -- the
    global run/AngleCS.data/AngleSN.data read already returns compact
    form directly, no tile-reassembly needed anymore."""
    return load_llc30_angle_cos_sin_compact()


# ===========================================================================
# Land/ocean mask safeguard (user's explicit ask -- MITgcm's own
# READ_PICKUP/CHECK_PICKUP do ZERO masking/consistency checking on a
# pickup file, confirmed in HANDOFF.md's "Pickup-file research" section)
# ===========================================================================

def _parse_hfac_params():
    """Parse hFacMin/hFacMinDr out of input/data's &PARM01 text (NOT
    re-hardcoded -- same "stay consistent with the actual source of
    truth" convention as parse_llc30_delR)."""
    with open(LLC30_DATA_PATH) as fh:
        text = fh.read()
    m_min = re.search(r"hFacMin\s*=\s*([\d.]+)", text)
    m_dr = re.search(r"hFacMinDr\s*=\s*([\d.]+)", text)
    assert m_min is not None and m_dr is not None, (
        "input/data: could not find hFacMin/hFacMinDr in &PARM01"
    )
    return float(m_min.group(1)), float(m_dr.group(1))


def build_llc30_hfac_masks():
    """Derive LLC30's own hFacC/hFacW/hFacS BOOLEAN (wet/dry) masks from
    ../input/BATHY_LLC30.bin + ../input/ICE_DRAFT_LLC30.bin +
    ../input/data's delR/hFacMin/hFacMinDr, replicating
    model/src/ini_masks_etc.F's EXACT wet-cell criterion in Python (an
    INDEPENDENT check, not calling the Fortran) -- including the
    hFacMin/hFacMinDr partial-cell-thinning step (hFac_loc <
    hFacMnSz*0.5 -> dry), which a naive `rF[k] > seafloor` test (as used
    by coarsen_initial_ts_to_llc30.py's build_fine_wet_mask, a
    deliberately simpler mask for THAT script's different purpose) does
    NOT apply.

    Phase A update (adjoint-experiment plan, see HANDOFF.md): this
    config now compiles pkg/shelfice/pkg/steep_icecavity with a real
    nonzero ICE_DRAFT_LLC30.bin, so Ro_surf is NO LONGER 0 everywhere --
    `model/src/ini_depths.F` initializes Ro_surf=rF(1)=0, then
    `SHELFICE_INIT_DEPTHS` lowers it to the ice draft wherever
    SHELFICEtopoFile is nonzero (`rSurfC = MIN(rSurfC, R_shelfIce)`,
    confirmed in shelfice_init_depths.F). The SECOND (Ro_surf-based)
    hFacC pass `ini_masks_etc.F` applies -- previously skipped here as
    a no-op -- is now REQUIRED; implemented below exactly per that
    file's own two-pass algorithm (first pass vs. R_low/bathymetry,
    re-derive R_low from the thinned hFacC, second pass vs. Ro_surf/ice
    draft, re-derive Ro_surf from the re-thinned hFacC). Confirmed by
    direct cross-check against this config's own real GLOBAL
    run/hFacC.data output (see cross_check_hfac_against_model_output).

    hFacW/hFacS use the simpler min-of-adjacent-hFacC convention
    (useMin4hFacEdges=.TRUE. style) rather than this config's actual
    useMin4hFacEdges=.FALSE. rSurfW/rLowW method -- a conservative,
    boolean-only safeguard, not a bit-exact hFac reproduction (see plan).

    Returns dict tile(1..13) -> (hFacC, hFacW, hFacS), each (15,30,30)
    bool arrays.
    """
    bathy_raw = np.fromfile(LLC30_BATHY_PATH, dtype=">f4").astype(np.float64)
    expected = COMPACT_COARSE_NY * COMPACT_COARSE_NX
    assert bathy_raw.size == expected, (LLC30_BATHY_PATH, bathy_raw.size, expected)
    bathy_compact = bathy_raw.reshape(COMPACT_COARSE_NY, COMPACT_COARSE_NX)

    ice_draft_raw = np.fromfile(ICE_DRAFT_PATH, dtype=">f4").astype(np.float64)
    assert ice_draft_raw.size == expected, (ICE_DRAFT_PATH, ice_draft_raw.size, expected)
    ice_draft_compact = ice_draft_raw.reshape(COMPACT_COARSE_NY, COMPACT_COARSE_NX)
    # Ro_surf: rF(1)=0 everywhere, lowered to the (already-negative) ice
    # draft wherever nonzero -- matches shelfice_init_depths.F's
    # `rSurfC = MIN(rSurfC, R_shelfIce)` exactly (both are <= 0, so MIN
    # picks the ice draft whenever it's present).
    ro_surf0 = np.where(ice_draft_compact != 0.0, ice_draft_compact, 0.0)

    delR15 = ts.parse_llc30_delR()
    rF15 = ts.build_depth_interfaces(delR15)
    hFacMin, hFacMinDr = _parse_hfac_params()
    land = bathy_compact == 0.0

    # --- Pass 1 (vs. R_low/bathymetry) ---
    hfacC_frac = np.zeros((NLEV30, COMPACT_COARSE_NY, COMPACT_COARSE_NX))
    for k in range(NLEV30):
        drF_k = delR15[k]
        hFacMnSz = max(hFacMin, min(hFacMinDr / drF_k, 1.0))
        hfac_loc = (rF15[k] - bathy_compact) / drF_k
        hfac_loc = np.clip(hfac_loc, 0.0, 1.0)
        # ini_masks_etc.F's pass-1 ALSO closes a cell if R_low>=Ro_surf
        # (a column entirely above the ice draft / entirely dry).
        wet = (hfac_loc >= hFacMnSz * 0.5) & (~land) & (bathy_compact < ro_surf0)
        hfacC_frac[k] = np.where(wet, np.maximum(hfac_loc, hFacMnSz), 0.0)

    # Re-derive R_low from the pass-1-thinned hFacC (ini_masks_etc.F:
    # "Re-calculate lower-R Boundary position, taking into account hFacC").
    r_low_adj = rF15[0] - np.sum(delR15[:, None, None] * hfacC_frac, axis=0)

    # --- Pass 2 (vs. Ro_surf/ice draft): remove the part ABOVE Ro_surf ---
    hfacC_frac2 = np.zeros_like(hfacC_frac)
    for k in range(NLEV30):
        drF_k = delR15[k]
        hFacMnSz = max(hFacMin, min(hFacMinDr / drF_k, 1.0))
        hfac_loc = (rF15[k] - ro_surf0) / drF_k
        hfac_loc = hfacC_frac[k] - np.maximum(hfac_loc, 0.0)
        hfac_loc = np.maximum(hfac_loc, 0.0)
        hfacC_frac2[k] = np.where(
            hfac_loc < hFacMnSz * 0.5, 0.0, np.maximum(hfac_loc, hFacMnSz)
        )
    hfacC_frac = hfacC_frac2
    hfacC_compact = hfacC_frac > 0.0

    # Re-derive Ro_surf from the pass-2-thinned hFacC, for use in the
    # rSurfW/rSurfS formula below (ini_masks_etc.F: "Re-calculate
    # Reference surface position, taking into account hFacC").
    ro_surf_adj = r_low_adj + np.sum(delR15[:, None, None] * hfacC_frac, axis=0)

    # rLowW/rSurfW: MAX(R_low(i-1),R_low(i)) / MIN(Ro_surf(i-1),Ro_surf(i))
    # at U points. West-face (i-1 neighbor) needs the facet-seam-aware
    # neighbor column for tile/facet boundary columns -- same topology
    # as the velocity un-stagger fix, but a scalar value (no rotation
    # needed, just a borrow).
    r_low_w, r_low_s = _rlow_w_s_with_seam_fix(r_low_adj)
    r_surf_w, r_surf_s = _rsurf_w_s_with_seam_fix(ro_surf_adj)

    hfacC_tiles = compact_coarse_to_tiles(hfacC_compact)
    hfacW_tiles, hfacS_tiles = {}, {}
    rlowW_tiles = compact_coarse_to_tiles(r_low_w)
    rlowS_tiles = compact_coarse_to_tiles(r_low_s)
    rsurfW_tiles = compact_coarse_to_tiles(r_surf_w)
    rsurfS_tiles = compact_coarse_to_tiles(r_surf_s)
    for tile in range(1, 14):
        hfacW = np.zeros((NLEV30, TILE_NY, TILE_NX), dtype=bool)
        hfacS = np.zeros((NLEV30, TILE_NY, TILE_NX), dtype=bool)
        rLowW_tile = rlowW_tiles[tile]  # (30,30)
        rLowS_tile = rlowS_tiles[tile]
        rSurfW_tile = rsurfW_tiles[tile]
        rSurfS_tile = rsurfS_tiles[tile]
        for k in range(NLEV30):
            drF_k = delR15[k]
            hFacMnSz = max(hFacMin, min(hFacMinDr / drF_k, 1.0))
            hfac1 = np.clip((rF15[k] - rLowW_tile) / drF_k, None, 1.0)
            hfac1 = np.where(
                (hfac1 < hFacMnSz * 0.5) | (rLowW_tile >= rSurfW_tile), 0.0,
                np.maximum(hfac1, hFacMnSz)
            )
            # Remove the part above rSurfW (mirrors hfacC's pass 2).
            above = np.maximum((rF15[k] - rSurfW_tile) / drF_k, 0.0)
            hfac1 = np.maximum(hfac1 - above, 0.0)
            hfac1 = np.where(hfac1 < hFacMnSz * 0.5, 0.0, hfac1)
            hfacW[k] = hfac1 > 0.0

            hfac1s = np.clip((rF15[k] - rLowS_tile) / drF_k, None, 1.0)
            hfac1s = np.where(
                (hfac1s < hFacMnSz * 0.5) | (rLowS_tile >= rSurfS_tile), 0.0,
                np.maximum(hfac1s, hFacMnSz)
            )
            above_s = np.maximum((rF15[k] - rSurfS_tile) / drF_k, 0.0)
            hfac1s = np.maximum(hfac1s - above_s, 0.0)
            hfac1s = np.where(hfac1s < hFacMnSz * 0.5, 0.0, hfac1s)
            hfacS[k] = hfac1s > 0.0
        hfacW_tiles[tile] = hfacW
        hfacS_tiles[tile] = hfacS
    return {tile: (hfacC_tiles[tile], hfacW_tiles[tile], hfacS_tiles[tile])
            for tile in range(1, 14)}


def _rsurf_w_s_with_seam_fix(ro_surf_adj_compact):
    """Same facet-seam-aware MIN-of-adjacent borrow as
    _rlow_w_s_with_seam_fix, but for rSurfW/rSurfS = MIN (not MAX) of
    the two adjacent Ro_surf values -- per ini_masks_etc.F's exact
    rSurfW/rSurfS formula."""
    by_facet = split_compact_coarse_into_facets(ro_surf_adj_compact)

    def _neighbor_edge_adjacent(nf, ne):
        arr = by_facet[nf]
        if ne == "N":
            return arr[-1, :]
        elif ne == "S":
            return arr[0, :]
        elif ne == "E":
            return arr[:, -1]
        elif ne == "W":
            return arr[:, 0]
        raise ValueError(ne)

    w_by_facet = {}
    s_by_facet = {}
    for facet in range(1, 6):
        arr = by_facet[facet]
        rSurfW = np.empty_like(arr)
        rSurfS = np.empty_like(arr)
        rSurfW[:, 1:] = np.minimum(arr[:, :-1], arr[:, 1:])
        rSurfS[1:, :] = np.minimum(arr[:-1, :], arr[1:, :])

        res_w = base._facet_neighbor(facet, "W")
        if res_w is None:
            rSurfW[:, 0] = arr[:, 0]
        else:
            nf, ne, rev = res_w
            line = _neighbor_edge_adjacent(nf, ne)
            line = line[::-1] if rev else line
            rSurfW[:, 0] = np.minimum(line, arr[:, 0])

        res_s = base._facet_neighbor(facet, "S")
        if res_s is None:
            rSurfS[0, :] = arr[0, :]
        else:
            nf, ne, rev = res_s
            line = _neighbor_edge_adjacent(nf, ne)
            line = line[::-1] if rev else line
            rSurfS[0, :] = np.minimum(line, arr[0, :])

        w_by_facet[facet] = rSurfW
        s_by_facet[facet] = rSurfS

    return (
        base.assemble_facets_into_compact(w_by_facet),
        base.assemble_facets_into_compact(s_by_facet),
    )


def _rlow_w_s_with_seam_fix(r_low_adj_compact):
    """r_low_adj_compact: (COMPACT_COARSE_NY, COMPACT_COARSE_NX), the
    hFacC-adjusted R_low (always <= 0, negative depth). Returns
    (rLowW, rLowS), each same shape: rLowW(i,j) = MAX(R_low(i-1,j),
    R_low(i,j)), rLowS(i,j) = MAX(R_low(i,j-1), R_low(i,j)) -- per
    ini_masks_etc.F's exact formula, with a facet-seam-aware fix (borrow
    the true neighbor facet's own edge-adjacent column/row, not a
    one-sided value) at the first column/row of each facet, same
    topology-walk mechanism as the velocity un-stagger fix (here a plain
    scalar borrow, no rotation needed since R_low isn't a vector)."""
    by_facet = split_compact_coarse_into_facets(r_low_adj_compact)

    def _neighbor_edge_adjacent(nf, ne):
        arr = by_facet[nf]
        if ne == "N":
            return arr[-1, :]
        elif ne == "S":
            return arr[0, :]
        elif ne == "E":
            return arr[:, -1]
        elif ne == "W":
            return arr[:, 0]
        raise ValueError(ne)

    w_by_facet = {}
    s_by_facet = {}
    for facet in range(1, 6):
        arr = by_facet[facet]
        ny30, nx30 = arr.shape
        rLowW = np.empty_like(arr)
        rLowS = np.empty_like(arr)
        rLowW[:, 1:] = np.maximum(arr[:, :-1], arr[:, 1:])
        rLowS[1:, :] = np.maximum(arr[:-1, :], arr[1:, :])

        res_w = base._facet_neighbor(facet, "W")
        if res_w is None:
            rLowW[:, 0] = arr[:, 0]  # open boundary: one-sided (rEmpty-like)
        else:
            nf, ne, rev = res_w
            line = _neighbor_edge_adjacent(nf, ne)
            line = line[::-1] if rev else line
            rLowW[:, 0] = np.maximum(line, arr[:, 0])

        res_s = base._facet_neighbor(facet, "S")
        if res_s is None:
            rLowS[0, :] = arr[0, :]
        else:
            nf, ne, rev = res_s
            line = _neighbor_edge_adjacent(nf, ne)
            line = line[::-1] if rev else line
            rLowS[0, :] = np.maximum(line, arr[0, :])

        w_by_facet[facet] = rLowW
        s_by_facet[facet] = rLowS

    return (
        base.assemble_facets_into_compact(w_by_facet),
        base.assemble_facets_into_compact(s_by_facet),
    )


def cross_check_hfac_against_model_output(masks):
    """Compare this script's own independently-derived hFacC/hFacW/hFacS
    boolean masks against this config's own already-committed GLOBAL
    run/hFacC.data (etc.) model output -- the model's own ACTUAL masks
    (written by a prior mitgcmuv run with useSingleCpuIO=.TRUE., see
    module docstring), not an aggregated summary. Asserts EXACT boolean
    (>0 vs ==0) agreement per the plan (a stronger bar than the
    5%-tolerance wet-fraction check elsewhere in this pipeline, since
    these are the literal ground truth, not an independently-aggregated
    comparison)."""
    print("Cross-checking derived hFac masks against run/hFac{C,W,S}.data ...")
    expected = NLEV30 * COMPACT_COARSE_NY * COMPACT_COARSE_NX
    for name, idx in (("hFacC", 0), ("hFacW", 1), ("hFacS", 2)):
        path = os.path.join(RUN_DIR, f"{name}.data")
        real = np.fromfile(path, dtype=">f4")
        assert real.size == expected, (path, real.size, expected)
        real = real.reshape(NLEV30, COMPACT_COARSE_NY, COMPACT_COARSE_NX)
        real_wet_tiles = compact_coarse_to_tiles(real > 0.0)
        worst_mismatch = 0
        for tile in range(1, 14):
            real_wet = real_wet_tiles[tile]
            derived_wet = masks[tile][idx]
            mismatch = int(np.sum(real_wet != derived_wet))
            worst_mismatch = max(worst_mismatch, mismatch)
            assert mismatch == 0, (
                f"{name} tile {tile}: {mismatch} cells disagree with "
                f"run/{name}.data's real wet/dry mask -- derived mask "
                f"does not match the model's own ACTUAL mask"
            )
        print(f"  {name}: EXACT agreement across all 13 tiles (worst_mismatch={worst_mismatch}).")


def apply_mask_and_assert(field_tiles, hfac_tiles, mask_key, field_name):
    """field_tiles: dict tile -> array (any shape matching hfac_tiles[tile][mask_key]).
    Zeros every dry cell, then asserts zero land-leakage remains (mirrors
    verify_bathy_correctness's "compute, then explicitly re-check" style)."""
    for tile in range(1, 14):
        wet = hfac_tiles[tile][mask_key]
        field_tiles[tile] = np.where(wet, field_tiles[tile], 0.0)
        leak = int(np.sum((~wet) & (field_tiles[tile] != 0.0)))
        assert leak == 0, (
            f"{field_name} tile {tile}: {leak} dry cells still nonzero after "
            f"masking -- land-cell leakage safeguard FAILED"
        )
    return field_tiles


# ===========================================================================
# pkg/seaice's own real pickup fields (Phase A, see HANDOFF.md) --
# siTICE/siAREA/siHEFF/siHSNOW (plain 2-D scalars) and siUICE/siVICE
# (grid-relative vector components on the SAME C-grid U/V convention as
# Uvel/Vvel, confirmed in pkg/seaice/SEAICE.h: "UICE ... at ... C-grid
# U point", "VICE ... at ... C-grid V point"), real, from the same
# pickup_seaice.0000000001.data source identified in the plan. Real
# source field order (confirmed earlier this session):
#   siTICE, siAREA, siHEFF, siHSNOW, siUICE, siVICE
# (6 stacked 2-D records, compact LLC90 (1170,90), real*8 big-endian).
# ===========================================================================

SEAICE_PICKUP_SRC = (
    "/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/"
    "ancillary_data/ancillary_data_orig/input_init/"
    "pickup_seaice.0000000001.data"
)
SEAICE_FIELD_ORDER = ["siTICE", "siAREA", "siHEFF", "siHSNOW", "siUICE", "siVICE"]
SI_TICE_REC, SI_AREA_REC, SI_HEFF_REC, SI_HSNOW_REC, SI_UICE_REC, SI_VICE_REC = range(6)


def read_seaice_2d_record(record_index):
    """Read one 2-D record from the real pickup_seaice source (compact
    LLC90 (1170,90), real*8 big-endian). Returns (COMPACT_NY,COMPACT_NX)."""
    rec_elems = COMPACT_NX * COMPACT_NY
    rec_bytes = rec_elems * 8
    with open(SEAICE_PICKUP_SRC, "rb") as fh:
        fh.seek(record_index * rec_bytes)
        raw = np.fromfile(fh, dtype=">f8", count=rec_elems)
    assert raw.size == rec_elems, (SEAICE_PICKUP_SRC, record_index, raw.size, rec_elems)
    return raw.reshape(COMPACT_NY, COMPACT_NX)


def coarsen_seaice_scalar_2d(record_index):
    """Horizontal-only 3:1 coarsen of a real seaice scalar record (siTICE/
    siAREA/siHEFF/siHSNOW), same treatment as EtaN/dEtaHdt/EtaH's own
    coarsen_scalar_2d_field -- reuses the SAME wet-cell-only horizontal
    averaging, no vertical remap needed (these are all 2-D fields).
    Returns dict tile -> (30,30)."""
    fine = read_seaice_2d_record(record_index)[None]  # (1, NY, NX)
    fine_bathy = _load_fine_bathy()
    delR15 = ts.parse_llc30_delR()
    wet15 = ts.build_fine_wet_mask(fine_bathy, delR15)
    wet_surface = wet15[0:1]
    coarse, _ = ts.coarsen_field_horizontally(fine, wet_surface)  # (1, 390, 30)
    return compact_coarse_to_tiles(coarse[0])


def remap_seaice_velocity_pair():
    """siUICE/siVICE: grid-relative vector components on sea-ice's OWN
    staggered grid -- same category of problem as Uvel/Vvel (confirmed
    same C-grid west/south-face convention, see module section
    docstring above), so reuse remap_velocity_pair_llc90_to_llc30's
    exact same unstagger/rotate/remap/rotate-back/restagger pipeline
    directly, just reading from the seaice source file's own records
    (50-level ocean-velocity machinery is overkill for a single-level
    2-D field, so this is a thin 1-level-only variant of that pipeline,
    not a reuse of remap_velocity_pair_llc90_to_llc30 itself, which
    hard-codes PICKUP_NLEV=50 vertical remap -- a single level needs no
    vertical remap at all, only the horizontal rotate/coarsen steps).
    Returns dict tile -> (Uvel30, Vvel30), each (30,30) LLC30-resolution.
    """
    uice_fine = read_seaice_2d_record(SI_UICE_REC)[None]  # (1, NY, NX)
    vice_fine = read_seaice_2d_record(SI_VICE_REC)[None]

    uC, vC = unstagger_uv_to_center_llc90(uice_fine, vice_fine)

    cos90, sin90 = load_llc90_angle_cos_sin()
    uE, vN = rotate_xy_to_en(uC, vC, cos90, sin90)

    fine_bathy = _load_fine_bathy()
    delR15 = ts.parse_llc30_delR()
    wet15 = ts.build_fine_wet_mask(fine_bathy, delR15)
    uE_coarse, _ = ts.coarsen_field_horizontally(uE, wet15[0:1])
    vN_coarse, _ = ts.coarsen_field_horizontally(vN, wet15[0:1])

    cos30_compact, sin30_compact = _llc30_angle_cos_sin_compact()
    uC30, vC30 = rotate_en_to_xy(uE_coarse, vN_coarse, cos30_compact, sin30_compact)

    Uvel30, Vvel30 = restagger_center_to_uv_llc30(uC30, vC30)
    return compact_coarse_to_tiles(Uvel30[0]), compact_coarse_to_tiles(Vvel30[0])


# ===========================================================================
# Pickup file writer
# ===========================================================================

PICKUP_FLDLIST = [f"{name:<8s}" for name in PICKUP_OUT_FIELDS]
assert all(len(f) == 8 for f in PICKUP_FLDLIST)


def write_ggl90_pickup_global(ggl90tke_tiles, out_dir):
    """Write a SINGLE GLOBAL pickup_ggl90.0000000000.data/.meta -- a
    SEPARATE pickup file pkg/ggl90 reads via its own GGL90_READ_PICKUP
    (confirmed: unconditionally triggered whenever pickupSuff.NE.' ',
    per ggl90_init_varia.F's `IF (nIter0.NE.0 .OR. pickupSuff.NE.' ')` --
    a real runtime dependency discovered only when the first end-to-end
    mitgcmuv run aborted with a missing-file STOP, not foreseen in the
    original plan). Single-field (GGL90TKE, 15 levels), "old way"
    READ_REC_3D_RL dispatch via MDS_READ_FIELD, which -- now that
    useSingleCpuIO=.TRUE. -- REQUIRES a global file (confirmed: that
    routine's useSingleCPUIO branch only ever opens the global filename,
    no per-tile fallback, see module docstring). dimList is the FULL
    global compact extent (no per-tile row range, no map2glob), matching
    real V4r6's own global-pickup .meta convention exactly.
    ggl90tke_tiles: dict tile(1..13) -> (15,30,30) array, already masked
    at dry hFacC cells by the caller."""
    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(out_dir, "pickup_ggl90.0000000000.data")
    meta_path = os.path.join(out_dir, "pickup_ggl90.0000000000.meta")

    compact = tiles_to_compact_coarse(ggl90tke_tiles)
    assert compact.shape == (NLEV30, COMPACT_COARSE_NY, COMPACT_COARSE_NX), compact.shape
    blob = compact.astype(">f8").tobytes()
    expected_bytes = NLEV30 * COMPACT_COARSE_NY * COMPACT_COARSE_NX * 8
    assert len(blob) == expected_bytes, (len(blob), expected_bytes)
    with open(data_path, "wb") as fh:
        fh.write(blob)
    observed = os.path.getsize(data_path)
    assert observed == expected_bytes, (data_path, observed, expected_bytes)

    lines = [
        " nDims = [   3 ];",
        " dimList = [",
        f"    {COMPACT_COARSE_NX},    1,   {COMPACT_COARSE_NX},",
        f"   {COMPACT_COARSE_NY},    1,  {COMPACT_COARSE_NY},",
        f"    {NLEV30},    1,   {NLEV30}",
        " ];",
        " dataprec = [ 'float64' ];",
        " nrecords = [          1 ];",
        " timeStepNumber = [          0 ];",
    ]
    with open(meta_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


SEAICE_FLDLIST = ["siTICE  ", "siAREA  ", "siHEFF  ", "siHSNOW ",
                   "siUICE  ", "siVICE  "]
assert all(len(f) == 8 for f in SEAICE_FLDLIST)


def write_seaice_pickup_global(seaice_tiles, out_dir):
    """seaice_tiles: dict field_name -> dict tile(1..13) -> (30,30) array
    (all 6 seaice pickup fields are single-level 2-D). Writes a SINGLE
    GLOBAL pickup_seaice.0000000000.data/.meta, using pkg/seaice's own
    "new way" fldList convention (confirmed in
    pkg/seaice/seaice_read_pickup.F: READ_MFLDS_3D_RL dispatch keyed by
    field name, in the order siTICE/siAREA/siHEFF/siHSNOW/siUICE/
    siVICE) -- a proper fldList meta avoids the fragile "old way"
    fixed-record-order fallback entirely (confirmed: `nbFields.LE.0`
    triggers only a WARNING, not a STOP, when pickupStrictlyMatch is
    .FALSE., but still risks silently misreading fields in the wrong
    order without one)."""
    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(out_dir, "pickup_seaice.0000000000.data")
    meta_path = os.path.join(out_dir, "pickup_seaice.0000000000.meta")

    parts = []
    for name in SEAICE_FIELD_ORDER:
        compact = tiles_to_compact_coarse(seaice_tiles[name])
        assert compact.shape == (COMPACT_COARSE_NY, COMPACT_COARSE_NX), (
            name, compact.shape
        )
        parts.append(compact.astype(">f8").tobytes())
    blob = b"".join(parts)
    expected_bytes = len(SEAICE_FIELD_ORDER) * COMPACT_COARSE_NY * COMPACT_COARSE_NX * 8
    assert len(blob) == expected_bytes, (len(blob), expected_bytes)
    with open(data_path, "wb") as fh:
        fh.write(blob)
    observed = os.path.getsize(data_path)
    assert observed == expected_bytes, (data_path, observed, expected_bytes)

    fld_list_str = " ".join(f"'{f}'" for f in SEAICE_FLDLIST)
    lines = [
        " nDims = [   2 ];",
        " dimList = [",
        f"    {COMPACT_COARSE_NX},    1,   {COMPACT_COARSE_NX},",
        f"   {COMPACT_COARSE_NY},    1,  {COMPACT_COARSE_NY}",
        " ];",
        " dataprec = [ 'float64' ];",
        f" nrecords = [{len(SEAICE_FIELD_ORDER):11d} ];",
        " timeStepNumber = [          0 ];",
        f" nFlds = [{len(SEAICE_FIELD_ORDER):4d} ];",
        " fldList = {",
        f" {fld_list_str}",
        " };",
    ]
    with open(meta_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_pickup_global(field_arrays_tiles, out_dir):
    """field_arrays_tiles: dict field_name -> dict tile(1..13) -> array
    ((15,30,30) for the 6 3-D fields, (30,30) for the 3 2-D fields).
    Writes a SINGLE GLOBAL pickup.0000000000.data/.meta per
    model/src/write_pickup.F's exact record order (PICKUP_OUT_FIELDS)
    and big-endian real*8 precision (pickup files are always
    precFloat64, NOT this config's readBinaryPrec=32). dimList/fldList
    format confirmed byte-for-byte against real V4r6's own GLOBAL
    pickup.0000000001.meta (nDims=2, full compact extent, no per-tile
    row range, no map2glob -- see module docstring)."""
    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(out_dir, "pickup.0000000000.data")
    meta_path = os.path.join(out_dir, "pickup.0000000000.meta")

    parts = []
    for name in PICKUP_OUT_FIELDS_3D:
        compact = tiles_to_compact_coarse(field_arrays_tiles[name])
        assert compact.shape == (NLEV30, COMPACT_COARSE_NY, COMPACT_COARSE_NX), (
            name, compact.shape
        )
        parts.append(compact.astype(">f8").tobytes())
    for name in PICKUP_OUT_FIELDS_2D:
        compact = tiles_to_compact_coarse(field_arrays_tiles[name])
        assert compact.shape == (COMPACT_COARSE_NY, COMPACT_COARSE_NX), (
            name, compact.shape
        )
        parts.append(compact.astype(">f8").tobytes())
    blob = b"".join(parts)
    expected_bytes = N_RECORDS_PER_TILE * COMPACT_COARSE_NY * COMPACT_COARSE_NX * 8
    assert len(blob) == expected_bytes, (len(blob), expected_bytes)
    with open(data_path, "wb") as fh:
        fh.write(blob)
    observed = os.path.getsize(data_path)
    assert observed == expected_bytes, (data_path, observed, expected_bytes)

    # nDims=2 (x,y ONLY -- no z row), matching real V4r6's own global
    # pickup.0000000001.meta exactly: a pickup's mixed 3-D/2-D record
    # layout is interpreted by READ_MFLDS_SET via its own
    # nFl3D = (nRecords-nFlds)/(thirdDim-1) formula (thirdDim=Nr passed
    # separately as an argument, NOT encoded in dimList).
    fld_list_str = " ".join(f"'{f}'" for f in PICKUP_FLDLIST)
    lines = [
        " nDims = [   2 ];",
        " dimList = [",
        f"    {COMPACT_COARSE_NX},    1,   {COMPACT_COARSE_NX},",
        f"   {COMPACT_COARSE_NY},    1,  {COMPACT_COARSE_NY}",
        " ];",
        " dataprec = [ 'float64' ];",
        f" nrecords = [{N_RECORDS_PER_TILE:11d} ];",
        " timeStepNumber = [          0 ];",
        f" nFlds = [{len(PICKUP_OUT_FIELDS):4d} ];",
        " fldList = {",
        f" {fld_list_str}",
        " };",
    ]
    with open(meta_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def verify_unstagger_with_synthetic_uniform_flow():
    """Correctness check for unstagger_uv_to_center_llc90's same-facet
    approximation (the check that replaced the originally-planned facet-
    seam swap fix, after that fix was found to be a net regression --
    see this module's module-level docstring and plan): construct a
    KNOWN uniform true-eastward flow (uE=1, vN=0 everywhere), rotate +
    stagger it onto the real LLC90 grid, run it through
    unstagger_uv_to_center_llc90, rotate back to east/north, and assert
    the output recovers the known input to within a small tolerance
    almost everywhere (bounded error only at the single row/column of
    cells directly on a facet seam, per the same-facet approximation).
    """
    cos90, sin90 = load_llc90_angle_cos_sin()
    uE_true = np.ones((1, COMPACT_NY, COMPACT_NX))
    vN_true = np.zeros((1, COMPACT_NY, COMPACT_NX))
    uC_true, vC_true = rotate_en_to_xy(uE_true, vN_true, cos90, sin90)

    u_by_facet = {f: base.split_compact_into_facets(uC_true[0])[f] for f in range(1, 6)}
    v_by_facet = {f: base.split_compact_into_facets(vC_true[0])[f] for f in range(1, 6)}
    uvel_by_facet, vvel_by_facet = {}, {}
    for facet in range(1, 6):
        uC, vC = u_by_facet[facet], v_by_facet[facet]
        Uvel, Vvel = np.empty_like(uC), np.empty_like(vC)
        Uvel[:, 1:] = 0.5 * (uC[:, :-1] + uC[:, 1:])
        Uvel[:, 0] = uC[:, 0]
        Vvel[1:, :] = 0.5 * (vC[:-1, :] + vC[1:, :])
        Vvel[0, :] = vC[0, :]
        uvel_by_facet[facet] = Uvel
        vvel_by_facet[facet] = Vvel
    uvel_fine = _assemble_fine_facets_into_compact(uvel_by_facet)[None]
    vvel_fine = _assemble_fine_facets_into_compact(vvel_by_facet)[None]

    uC_rec, vC_rec = unstagger_uv_to_center_llc90(uvel_fine, vvel_fine)
    uE_rec, vN_rec = rotate_xy_to_en(uC_rec, vC_rec, cos90, sin90)
    err = np.abs(uE_rec[0] - 1.0) + np.abs(vN_rec[0] - 0.0)

    frac_bad = (err > 0.05).mean()
    assert err.mean() < 0.01, (
        f"unstagger_uv_to_center_llc90 synthetic round-trip check FAILED: "
        f"mean error {err.mean():.5f} too large for the same-facet "
        f"approximation (expected ~0.0005)"
    )
    assert frac_bad < 0.01, (
        f"unstagger_uv_to_center_llc90 synthetic round-trip check FAILED: "
        f"{frac_bad:.4f} fraction of cells have >0.05 error, expected "
        f"<1% (bounded to the single row/column of cells on a facet seam)"
    )
    print(f"  Synthetic uniform-flow round-trip check PASSED "
          f"(mean err={err.mean():.5f}, frac>0.05={frac_bad:.4f}).")


# ===========================================================================
# main
# ===========================================================================

def main():
    print("=" * 70)
    print("Step 0/6: synthetic correctness check for velocity un-staggering")
    print("=" * 70)
    verify_unstagger_with_synthetic_uniform_flow()

    print()
    print("=" * 70)
    print("Step 1/6: Theta/Salt (reuse already-coarsened THETA/SALT_LLC30_INIT.bin)")
    print("=" * 70)
    theta_tiles, salt_tiles = load_theta_salt_tiles()
    for tile in (1, 7, 13):
        print(f"  tile {tile}: Theta level-0 min/max "
              f"[{theta_tiles[tile][0].min():.2f},{theta_tiles[tile][0].max():.2f}]")

    print()
    print("=" * 70)
    print("Step 2/6: EtaN/dEtaHdt/EtaH (coarsen real pickup records 450/451/452)")
    print("=" * 70)
    etan_tiles = coarsen_scalar_2d_field(ETAN_RECORD)
    detahdt_tiles = coarsen_scalar_2d_field(DETAHDT_RECORD)
    etah_tiles = coarsen_scalar_2d_field(ETAH_RECORD)
    print(f"  EtaN tile 1 min/max [{etan_tiles[1].min():.4f},{etan_tiles[1].max():.4f}]")

    print()
    print("=" * 70)
    print("Step 3/6: Uvel/Vvel and GuNm1/GvNm1 (full vector-aware remap)")
    print("=" * 70)
    uvel_tiles, vvel_tiles, (uE_c, vN_c) = remap_velocity_pair_llc90_to_llc30(
        UVEL_IDX0, VVEL_IDX0
    )
    gunm1_tiles, gvnm1_tiles, _ = remap_velocity_pair_llc90_to_llc30(
        GUNM1_IDX0, GVNM1_IDX0
    )
    print(f"  Uvel tile 1 level-0 min/max "
          f"[{uvel_tiles[1][0].min():.4f},{uvel_tiles[1][0].max():.4f}]")
    print(f"  Vvel tile 1 level-0 min/max "
          f"[{vvel_tiles[1][0].min():.4f},{vvel_tiles[1][0].max():.4f}]")

    print()
    print("=" * 70)
    print("Step 4/6: land/ocean mask safeguard -- derive + cross-check hFacC/W/S")
    print("=" * 70)
    masks = build_llc30_hfac_masks()
    cross_check_hfac_against_model_output(masks)

    print()
    print("=" * 70)
    print("Step 5/6: zero every field at dry cells, assert zero leakage")
    print("=" * 70)
    theta_tiles = apply_mask_and_assert(theta_tiles, masks, 0, "Theta")
    salt_tiles = apply_mask_and_assert(salt_tiles, masks, 0, "Salt")
    uvel_tiles = apply_mask_and_assert(uvel_tiles, masks, 1, "Uvel")
    vvel_tiles = apply_mask_and_assert(vvel_tiles, masks, 2, "Vvel")
    gunm1_tiles = apply_mask_and_assert(gunm1_tiles, masks, 1, "GuNm1")
    gvnm1_tiles = apply_mask_and_assert(gvnm1_tiles, masks, 2, "GvNm1")
    hfacC_surface_tiles = {t: masks[t][0][0] for t in range(1, 14)}
    etan_tiles = apply_mask_and_assert(etan_tiles, {t: (hfacC_surface_tiles[t],) for t in range(1, 14)}, 0, "EtaN")
    detahdt_tiles = apply_mask_and_assert(detahdt_tiles, {t: (hfacC_surface_tiles[t],) for t in range(1, 14)}, 0, "dEtaHdt")
    etah_tiles = apply_mask_and_assert(etah_tiles, {t: (hfacC_surface_tiles[t],) for t in range(1, 14)}, 0, "EtaH")
    print("  All fields masked, zero land-leakage confirmed.")

    print()
    print("=" * 70)
    print("Step 6/6: write GLOBAL pickup.0000000000.{data,meta} (single file)")
    print("=" * 70)
    field_arrays_tiles = {
        "Uvel": uvel_tiles,
        "Vvel": vvel_tiles,
        "Theta": theta_tiles,
        "Salt": salt_tiles,
        "GuNm1": gunm1_tiles,
        "GvNm1": gvnm1_tiles,
        "EtaN": etan_tiles,
        "dEtaHdt": detahdt_tiles,
        "EtaH": etah_tiles,
    }
    write_pickup_global(field_arrays_tiles, OUT_PICKUP_DIR)
    print(f"  Wrote global pickup to {OUT_PICKUP_DIR}")

    print()
    print("=" * 70)
    print("Step 6b/6: write pkg/ggl90's own separate GLOBAL pickup_ggl90.* file")
    print("=" * 70)
    GGL90_TKE_MIN = 1.0e-11  # ggl90_readparms.F's own GGL90TKEmin default
    ggl90tke_tiles = {
        tile: np.where(masks[tile][0], GGL90_TKE_MIN, 0.0)
        for tile in range(1, 14)
    }
    write_ggl90_pickup_global(ggl90tke_tiles, OUT_PICKUP_DIR)
    print(f"  Wrote global pickup_ggl90 to {OUT_PICKUP_DIR}")

    print()
    print("=" * 70)
    print("Step 6c/6: write pkg/seaice's own separate GLOBAL pickup_seaice.* file")
    print("=" * 70)
    sitice_tiles = coarsen_seaice_scalar_2d(SI_TICE_REC)
    siarea_tiles = coarsen_seaice_scalar_2d(SI_AREA_REC)
    siheff_tiles = coarsen_seaice_scalar_2d(SI_HEFF_REC)
    sihsnow_tiles = coarsen_seaice_scalar_2d(SI_HSNOW_REC)
    siuice_tiles, sivice_tiles = remap_seaice_velocity_pair()

    hfacC_surface_tiles2 = {t: masks[t][0][0] for t in range(1, 14)}
    hfacW_surface_tiles = {t: masks[t][1][0] for t in range(1, 14)}
    hfacS_surface_tiles = {t: masks[t][2][0] for t in range(1, 14)}
    sitice_tiles = apply_mask_and_assert(
        sitice_tiles, {t: (hfacC_surface_tiles2[t],) for t in range(1, 14)}, 0, "siTICE"
    )
    siarea_tiles = apply_mask_and_assert(
        siarea_tiles, {t: (hfacC_surface_tiles2[t],) for t in range(1, 14)}, 0, "siAREA"
    )
    siheff_tiles = apply_mask_and_assert(
        siheff_tiles, {t: (hfacC_surface_tiles2[t],) for t in range(1, 14)}, 0, "siHEFF"
    )
    sihsnow_tiles = apply_mask_and_assert(
        sihsnow_tiles, {t: (hfacC_surface_tiles2[t],) for t in range(1, 14)}, 0, "siHSNOW"
    )
    siuice_tiles = apply_mask_and_assert(
        siuice_tiles, {t: (hfacW_surface_tiles[t],) for t in range(1, 14)}, 0, "siUICE"
    )
    sivice_tiles = apply_mask_and_assert(
        sivice_tiles, {t: (hfacS_surface_tiles[t],) for t in range(1, 14)}, 0, "siVICE"
    )
    seaice_tiles = {
        "siTICE": sitice_tiles,
        "siAREA": siarea_tiles,
        "siHEFF": siheff_tiles,
        "siHSNOW": sihsnow_tiles,
        "siUICE": siuice_tiles,
        "siVICE": sivice_tiles,
    }
    write_seaice_pickup_global(seaice_tiles, OUT_PICKUP_DIR)
    print(f"  Wrote global pickup_seaice to {OUT_PICKUP_DIR}")

    print()
    print("ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
