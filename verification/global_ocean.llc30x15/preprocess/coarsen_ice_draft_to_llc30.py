#!/usr/bin/env python3
"""Coarsen the real V4r6 ice-shelf/ice-front draft file 3:1 into LLC30,
for use as pkg/steep_icecavity's (STIC) STICdepthFile and
pkg/shelfice's SHELFICEtopoFile (same source file drives both -- see
input/data.stic and input/data.shelfice's comments, and HANDOFF.md's
STIC rationale).

Source (read-only, never modified): real*4 big-endian compact LLC90
(1170,90), same ancillary_data_orig/input_init/ directory as the
bathymetry source this project already coarsens
(coarsen_llc90_to_llc30.py's BATHY_SRC):
  /ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/ancillary_data/
  ancillary_data_orig/input_init/
  ICE_DRAFT_ICE_SHELF_CAVITY_PLUS_ICE_FRONT_LLC_0090.bin
Confirmed size: 90*1170*4 = 421200 bytes, matching BATHY_SRC's own
convention exactly.

Coarsening rule: reuses coarsen_llc90_to_llc30.py's own
coarsen_bathy_facet/coarsen_bathy_block (3x3-block min-magnitude-over-
wet-cells, land only if ALL 9 constituents are 0) UNCHANGED -- the ice
draft field has the exact same "0 = no ice shelf, negative = draft
depth" convention as bathymetry's "0 = land, negative = ocean depth",
so the identical rule applies without modification.

Output: ICE_DRAFT_LLC30.bin, real*4 compact (390,30), same 46800-byte
size as BATHY_LLC30.bin, committed directly into input/ (matching this
project's existing binary-commit convention -- not a gitignored
preprocess/output/ scratch file).

Run:
    export PATH=$HOME/micromamba/envs/podaac/bin:$PATH
    cd MITgcm/verification/global_ocean.llc30x15/preprocess
    python3 coarsen_ice_draft_to_llc30.py
"""

import os

import numpy as np

import coarsen_llc90_to_llc30 as base

ICE_DRAFT_SRC = (
    "/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/"
    "ancillary_data/ancillary_data_orig/input_init/"
    "ICE_DRAFT_ICE_SHELF_CAVITY_PLUS_ICE_FRONT_LLC_0090.bin"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "..", "input")
OUT_PATH = os.path.join(INPUT_DIR, "ICE_DRAFT_LLC30.bin")
BATHY_PATH = os.path.join(INPUT_DIR, "BATHY_LLC30.bin")

COMPACT_NX = base.COMPACT_NX                  # 90
COMPACT_NY = base.COMPACT_NY                  # 1170
COMPACT_COARSE_NX = base.COMPACT_COARSE_NX    # 30
COMPACT_COARSE_NY = base.COMPACT_COARSE_NY    # 390


def verify_ice_draft_correctness(fine_compact, coarse_compact, coarse_bathy):
    """Land(no-ice)-fraction-preserved-within-tolerance and depth-bounds-
    preserved checks, same style as coarsen_llc90_to_llc30.py's own
    verify_bathy_correctness. Also spot-checks that the ice-draft
    footprint is a SUBSET of BATHY_LLC30.bin's wet mask -- an ice shelf
    cannot exist over dry land.
    """
    fine_noice_frac = (fine_compact == 0.0).mean()
    coarse_noice_frac = (coarse_compact == 0.0).mean()
    print("\nIce-draft correctness check:")
    print(f"  fine no-ice fraction   = {fine_noice_frac:.4f}")
    print(f"  coarse no-ice fraction = {coarse_noice_frac:.4f}")
    assert coarse_noice_frac <= fine_noice_frac + 1e-9, (
        f"Ice-draft check FAILED: coarse no-ice fraction "
        f"({coarse_noice_frac:.4f}) exceeds fine no-ice fraction "
        f"({fine_noice_frac:.4f})"
    )

    fine_ice = fine_compact[fine_compact != 0.0]
    coarse_ice = coarse_compact[coarse_compact != 0.0]
    if fine_ice.size and coarse_ice.size:
        assert coarse_ice.min() >= fine_ice.min() - 1e-6, (
            coarse_ice.min(), fine_ice.min()
        )
        assert coarse_ice.max() <= fine_ice.max() + 1e-6, (
            coarse_ice.max(), fine_ice.max()
        )
        print(f"  fine ice-draft range   = [{fine_ice.min():.2f}, {fine_ice.max():.2f}]")
        print(f"  coarse ice-draft range = [{coarse_ice.min():.2f}, {coarse_ice.max():.2f}]")
    print("  depth-bounds check PASSED.")

    ice_mask = coarse_compact != 0.0
    wet_mask = coarse_bathy != 0.0
    leak = int(np.sum(ice_mask & ~wet_mask))
    assert leak == 0, (
        f"Ice-draft check FAILED: {leak} coarse cells have a nonzero ice "
        f"draft over dry land (BATHY_LLC30.bin == 0) -- an ice shelf "
        f"cannot exist over dry land"
    )
    print(f"  subset-of-wet-mask check PASSED (ice footprint ⊆ bathymetry wet mask).")


def main():
    observed_in = os.path.getsize(ICE_DRAFT_SRC)
    expected_in = COMPACT_NX * COMPACT_NY * 4
    assert observed_in == expected_in, (
        f"ICE_DRAFT source: {observed_in} bytes, expected {expected_in} (90*1170*4)"
    )

    raw = np.fromfile(ICE_DRAFT_SRC, dtype=">f4").astype(np.float64)
    compact = raw.reshape(COMPACT_NY, COMPACT_NX)

    fine_facets = base.split_compact_into_facets(compact)
    coarse_facets = {}
    for facet in range(1, 6):
        coarse_facets[facet] = base.coarsen_bathy_facet(fine_facets[facet])
    coarse_compact = base.assemble_facets_into_compact(coarse_facets)

    expected_out = COMPACT_COARSE_NX * COMPACT_COARSE_NY * 4
    out_bytes = coarse_compact.astype(">f4").tobytes()
    assert len(out_bytes) == expected_out, (
        f"ICE_DRAFT_LLC30.bin: assembled {len(out_bytes)} bytes, expected {expected_out}"
    )
    with open(OUT_PATH, "wb") as fh:
        fh.write(out_bytes)
    observed_out = os.path.getsize(OUT_PATH)
    assert observed_out == expected_out, (
        f"ICE_DRAFT_LLC30.bin: written {observed_out} bytes, expected {expected_out}"
    )
    print(f"ICE_DRAFT_LLC30.bin: {observed_out} bytes OK "
          f"(compact shape {COMPACT_COARSE_NY}x{COMPACT_COARSE_NX})")

    bathy_observed = os.path.getsize(BATHY_PATH)
    assert bathy_observed == expected_out, (BATHY_PATH, bathy_observed, expected_out)
    coarse_bathy = np.fromfile(BATHY_PATH, dtype=">f4").astype(np.float64).reshape(
        COMPACT_COARSE_NY, COMPACT_COARSE_NX
    )
    verify_ice_draft_correctness(compact, coarse_compact, coarse_bathy)


if __name__ == "__main__":
    main()
