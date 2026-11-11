#!/usr/bin/env python3
"""Coarsen the REAL satellite SST product and its real per-cell error
file -- the data real V4r6 itself uses for its 'sst-reynoldsmean' gencost
term (input.V4r6.102/data.ecco) -- into LLC30 inputs for this config's
Phase B adjoint experiment's own SST cost term.

An earlier planning pass proposed substituting a WOA18 monthly
climatology, reasoning no real satellite product existed for this
project. That reasoning was WRONG (caught before implementation, see
HANDOFF.md / docs/prompts): the real AVHRR_OI-NCEI-L4-GLOB product,
including the 1992 file matching this config's own data.cal/forcing
start year, and the real sigma_half_maskpolynya_areascaled.bin error
file, both exist locally. This script coarsens those real files instead.

Source data (read-only, never modified):
  AVHRR SST: /ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/
    ancillary_data/ancillary_data_orig/data_constraints/input_sst/
    AVHRR_OI-NCEI-L4-GLOB_1992
  real*4 big-endian, compact LLC90 (1170,90), 12 monthly records
  (confirmed: 5,054,400 bytes = 1170*90*4*12 exactly). UNLIKE every other
  field this preprocessing pipeline has coarsened so far, this file's
  land/missing convention is a large sentinel fill value (9.96921e+36,
  confirmed the single unique value above 1e30), NOT zero -- masked
  explicitly before averaging, not inherited from BATHY_LLC30.bin's own
  ocean/land mask.

  SST error: .../ancillary_data_orig/data_constraints/data_error/scaled/
    sigma_half_maskpolynya_areascaled.bin
  real*4 big-endian, compact LLC90 (1170,90), single STATIC 2-D record
  (confirmed: 421,200 bytes = 1170*90*4 exactly, same shape/convention as
  BATHY_ICE_SHELF_CAVITY_PLUS_ICE_FRONT_LLC_0090.bin). Unlike the AVHRR
  file, this one uses plain 0.0 as its land/masked value (confirmed:
  6.07% exactly-zero cells, zero cells with the AVHRR-style large fill
  value, nonzero range [0.5, 130.67]) -- same land-value convention as
  BATHY_LLC30.bin, so it reuses coarsen_bathy_facet/coarsen_bathy_block's
  wet-cell-only rule directly, NOT the AVHRR fill-aware masking below
  (two different conventions, two different coarsening paths).

Run:
    export PATH=$HOME/micromamba/envs/podaac/bin:$PATH
    cd MITgcm/verification/global_ocean.llc30x15/preprocess
    python3 coarsen_sst_obs_to_llc30.py
"""

import os

import numpy as np

import coarsen_llc90_to_llc30 as base
import coarsen_initial_ts_to_llc30 as ts

AVHRR_SRC = (
    "/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/"
    "ancillary_data/ancillary_data_orig/data_constraints/input_sst/"
    "AVHRR_OI-NCEI-L4-GLOB_1992"
)
SIGMA_SST_SRC = (
    "/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/"
    "ancillary_data/ancillary_data_orig/data_constraints/data_error/scaled/"
    "sigma_half_maskpolynya_areascaled.bin"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "input_ad")

COMPACT_NX = base.COMPACT_NX          # 90
COMPACT_NY = base.COMPACT_NY          # 1170
COMPACT_COARSE_NX = base.COMPACT_COARSE_NX   # 30
COMPACT_COARSE_NY = base.COMPACT_COARSE_NY   # 390

AVHRR_FILL = np.float32(9.96921e+36)
AVHRR_N_MONTHS = 12

AVHRR_OUT_PATH = os.path.join(OUT_DIR, "AVHRR_LLC30_MONTHLY_1992")
SIGMA_SST_OUT_PATH = os.path.join(OUT_DIR, "SIGMA_SST_LLC30.bin")


# ===========================================================================
# AVHRR SST: 12 monthly records, fill-value-masked, wet-mean coarsened
# (reuses coarsen_initial_ts_to_llc30.coarsen_field_horizontally, which
# already accepts an externally-supplied per-field wet mask -- here the
# AVHRR fill mask, not BATHY_LLC30.bin's ocean/land mask)
# ===========================================================================

def load_avhrr_monthly():
    expected_bytes = COMPACT_NX * COMPACT_NY * 4 * AVHRR_N_MONTHS
    observed_bytes = os.path.getsize(AVHRR_SRC)
    assert observed_bytes == expected_bytes, (
        AVHRR_SRC, observed_bytes, expected_bytes
    )
    raw = np.fromfile(AVHRR_SRC, dtype=">f4")
    data = raw.reshape(AVHRR_N_MONTHS, COMPACT_NY, COMPACT_NX)
    fill = data > 1.0e30
    frac_fill = fill.mean()
    print(f"  AVHRR SST: {AVHRR_N_MONTHS} monthly records, "
          f"{frac_fill:.4f} fraction fill/land (month-0 check: "
          f"{fill[0].mean():.4f})")
    return data.astype(np.float64), ~fill


def coarsen_avhrr_sst():
    sst_fine, wet_fine = load_avhrr_monthly()
    value_months = []
    for m in range(AVHRR_N_MONTHS):
        value_coarse, wet_frac_coarse = ts.coarsen_field_horizontally(
            sst_fine[m:m + 1], wet_fine[m:m + 1]
        )
        value_months.append(value_coarse[0])
    out = np.stack(value_months, axis=0)  # (12, 390, 30)
    assert out.shape == (AVHRR_N_MONTHS, COMPACT_COARSE_NY, COMPACT_COARSE_NX)
    return out


# ===========================================================================
# SST error: single static record, bathymetry-style wet-cell-only
# coarsening (land convention is plain 0.0, like BATHY_LLC30.bin)
# ===========================================================================

def load_sigma_sst():
    expected_bytes = COMPACT_NX * COMPACT_NY * 4
    observed_bytes = os.path.getsize(SIGMA_SST_SRC)
    assert observed_bytes == expected_bytes, (
        SIGMA_SST_SRC, observed_bytes, expected_bytes
    )
    raw = np.fromfile(SIGMA_SST_SRC, dtype=">f4").astype(np.float64)
    data = raw.reshape(COMPACT_NY, COMPACT_NX)
    frac_zero = (data == 0.0).mean()
    nonzero = data[data != 0.0]
    print(f"  SST error: single static record, {frac_zero:.4f} fraction "
          f"zero/land, nonzero range [{nonzero.min():.4f}, "
          f"{nonzero.max():.4f}]")
    return data


def coarsen_sigma_sst():
    data = load_sigma_sst()
    facets_fine = base.split_compact_into_facets(data)
    facets_coarse = {}
    for facet in range(1, 6):
        facets_coarse[facet] = base.coarsen_bathy_facet(facets_fine[facet])
    return base.assemble_facets_into_compact(facets_coarse)  # (390, 30)


# ===========================================================================
# Verification
# ===========================================================================

def verify_avhrr_bounds(coarse):
    """Coarse SST (wet cells only) must stay within realistic ocean SST
    bounds -- guaranteed by construction (wet-cell-only mean of real
    fine-resolution values), asserted anyway as a masking-bug tripwire."""
    wet = coarse != 0.0
    vals = coarse[wet]
    assert vals.size > 0, "coarsened AVHRR SST has no wet cells at all"
    lo, hi = vals.min(), vals.max()
    assert -3.0 <= lo and hi <= 36.0, (
        f"coarsened AVHRR SST out of physically sane range: [{lo}, {hi}]"
    )
    print(f"  AVHRR bounds check PASSED: wet-cell range [{lo:.2f}, "
          f"{hi:.2f}] C across all 12 months.")


def verify_sigma_bounds(coarse):
    wet = coarse != 0.0
    vals = coarse[wet]
    assert vals.size > 0, "coarsened SST error field has no wet cells at all"
    lo, hi = vals.min(), vals.max()
    assert lo > 0.0, "SST error field has a non-positive wet-cell value"
    print(f"  SST error bounds check PASSED: wet-cell range "
          f"[{lo:.4f}, {hi:.4f}].")


def main():
    print("=" * 70)
    print("AVHRR satellite SST -> LLC30")
    print("=" * 70)
    avhrr_coarse = coarsen_avhrr_sst()
    verify_avhrr_bounds(avhrr_coarse)
    os.makedirs(OUT_DIR, exist_ok=True)
    avhrr_coarse.astype(">f4").tofile(AVHRR_OUT_PATH)
    expected_bytes = AVHRR_N_MONTHS * COMPACT_COARSE_NY * COMPACT_COARSE_NX * 4
    observed_bytes = os.path.getsize(AVHRR_OUT_PATH)
    assert observed_bytes == expected_bytes, (
        AVHRR_OUT_PATH, observed_bytes, expected_bytes
    )
    print(f"  Wrote {AVHRR_OUT_PATH} ({observed_bytes} bytes).")

    print("=" * 70)
    print("Real SST error field -> LLC30")
    print("=" * 70)
    sigma_coarse = coarsen_sigma_sst()
    verify_sigma_bounds(sigma_coarse)
    sigma_coarse.astype(">f4").tofile(SIGMA_SST_OUT_PATH)
    expected_bytes = COMPACT_COARSE_NY * COMPACT_COARSE_NX * 4
    observed_bytes = os.path.getsize(SIGMA_SST_OUT_PATH)
    assert observed_bytes == expected_bytes, (
        SIGMA_SST_OUT_PATH, observed_bytes, expected_bytes
    )
    print(f"  Wrote {SIGMA_SST_OUT_PATH} ({observed_bytes} bytes).")

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
