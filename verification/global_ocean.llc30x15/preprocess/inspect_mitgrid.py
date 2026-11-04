#!/usr/bin/env python3
"""Task 1: Inspect and confirm the exact binary layout of LLC90 .mitgrid files.

CONFIRMED FIELD ORDER AND SHAPE (do not change without re-verifying):

  MITgcm's native-grid ".mitgrid" files are raw, unformatted, big-endian
  real*8 (float64) binary files with NO Fortran record markers -- i.e. a
  flat array of 16 stacked 2-D fields, each field exactly
  (ny+1) rows x (nx+1) columns (one extra row/column of corner points
  per facet), stored in row-major (C) order within each field.

  Field order (confirmed independently from two sources in this MITgcm
  checkout, both of which must agree -- and do):
    1. model/src/ini_curvilinear_grid.F (the multi-tile / OLD_GRID_IO-off
       path, lines ~294-336) calls MDS_FACEF_READ_RS with the explicit
       1-based record/field index:
         1 xC   2 yC   3 dxF  4 dyF  5 rA   6 xG   7 yG   8 dxV
         9 dyU  10 rAz 11 dxC 12 dyC 13 rAw 14 rAs 15 dxG 16 dyG
    2. pkg/mdsio/mdsio_facef_read.F shows each "record" irec is read as
       one (dNy+1)-row x (dNx+1)-column block (loop "DO jj=1+tBy,sNy+1+tBy
       ... READ(dUnit,rec=jj+jBase) (ioBuf4(i),i=1,dNx+1)"), i.e. row-major,
       y (row) outer index, x (column) inner index -- confirming the
       (ny+1, nx+1) numpy reshape used below (numpy's default C/row-major
       order matches Fortran's column-record layout exactly because MDSIO
       reads it one row of length (nx+1) at a time).

  So: byte offset of field k (1-based) for a facet of shape (nx,ny) is
      k_offset = (k-1) * (ny+1) * (nx+1) * 8 bytes
  and each field reshapes as arr.reshape(ny+1, nx+1) from a big-endian
  float64 buffer.

  Expected total file size for a facet (nx,ny):  16 * (nx+1) * (ny+1) * 8

This script performs exactly the checks required by Task 1 of the LLC30
lightweight-verification-config plan: open all 5 tile*.mitgrid files,
reshape per the formula above, confirm the byte-count formula matches the
observed file sizes exactly, and sanity-check XC/YC ranges.
"""

import os
import sys

import numpy as np

SRC_DIR = (
    "/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/"
    "ancillary_data/ancillary_data_orig/native_grid_files"
)

FIELD_NAMES = [
    "XC", "YC", "DXF", "DYF", "RAC", "XG", "YG", "DXV",
    "DYU", "RAZ", "DXC", "DYC", "RAW", "RAS", "DXG", "DYG",
]
N_FIELDS = 16

# facet number -> (nx, ny), per the LLC90 standard facet layout (Context
# section of the plan): facets 1,2 tall (90,270), facet 3 square (90,90),
# facets 4,5 wide (270,90).
FACET_DIMS = {
    1: (90, 270),
    2: (90, 270),
    3: (90, 90),
    4: (270, 90),
    5: (270, 90),
}

# Observed file sizes (bytes), from `ls -la` on the source directory --
# used as the independent ground truth the formula must reproduce exactly.
EXPECTED_SIZES = {
    1: 3156608,
    2: 3156608,
    3: 1059968,
    4: 3156608,
    5: 3156608,
}


def mitgrid_byte_formula(nx, ny):
    """16 fields * (nx+1)*(ny+1) corner points * 8 bytes/real*8."""
    return N_FIELDS * (nx + 1) * (ny + 1) * 8


def read_mitgrid_tile(path, nx, ny):
    """Read one .mitgrid file and return dict of 16 fields, each (ny+1, nx+1)."""
    raw = np.fromfile(path, dtype=">f8")
    expected_n = N_FIELDS * (ny + 1) * (nx + 1)
    if raw.size != expected_n:
        raise ValueError(
            f"{path}: read {raw.size} float64 values, expected {expected_n} "
            f"(16 * {ny + 1} * {nx + 1})"
        )
    fields = {}
    for k, name in enumerate(FIELD_NAMES):
        chunk = raw[k * (ny + 1) * (nx + 1):(k + 1) * (ny + 1) * (nx + 1)]
        fields[name] = chunk.reshape(ny + 1, nx + 1)
    return fields


def main():
    print(f"Source directory: {SRC_DIR}")
    print(f"{'tile':<10}{'(nx,ny)':<12}{'formula bytes':<16}"
          f"{'observed bytes':<16}{'match':<8}")

    all_ok = True
    for facet in range(1, 6):
        nx, ny = FACET_DIMS[facet]
        tile_path = os.path.join(SRC_DIR, f"tile{facet:03d}.mitgrid")
        observed = os.path.getsize(tile_path)
        formula = mitgrid_byte_formula(nx, ny)
        expected = EXPECTED_SIZES[facet]
        match = (formula == observed == expected)
        all_ok = all_ok and match
        print(f"tile{facet:03d}   ({nx},{ny})     {formula:<16}"
              f"{observed:<16}{'OK' if match else 'MISMATCH'}")

    if not all_ok:
        print("\nBYTE-COUNT FORMULA MISMATCH -- blocker, do not proceed to Task 2.")
        sys.exit(1)

    print("\nAll 5 tiles match the 16-field / +1-corner / real*8 formula exactly.")

    print(f"\n{'tile':<10}{'field':<6}{'min':>14}{'max':>14}{'mean':>14}")
    range_ok = True
    for facet in range(1, 6):
        nx, ny = FACET_DIMS[facet]
        tile_path = os.path.join(SRC_DIR, f"tile{facet:03d}.mitgrid")
        fields = read_mitgrid_tile(tile_path, nx, ny)
        for name in ("XC", "YC"):
            arr = fields[name]
            lo, hi, mn = arr.min(), arr.max(), arr.mean()
            print(f"tile{facet:03d}   {name:<6}{lo:14.4f}{hi:14.4f}{mn:14.4f}")
            if name == "XC" and not (-360.0 <= lo <= 360.0 and -360.0 <= hi <= 360.0):
                range_ok = False
            if name == "YC" and not (-90.0 <= lo <= 90.0 and -90.0 <= hi <= 90.0):
                range_ok = False

    if not range_ok:
        print("\nXC/YC OUT OF VALID LON/LAT RANGE -- field-order or byte-order "
              "assumption is likely wrong. Blocker.")
        sys.exit(1)

    print("\nXC/YC ranges are physically valid longitude/latitude values for all "
          "5 tiles.")
    print("Task 1 PASSED: field order/shape/byte-count assumptions confirmed.")


if __name__ == "__main__":
    main()
