# LLC90 -> LLC30 preprocessing

This directory contains the offline, one-time preprocessing scripts that
coarsen real ECCO V4r6 LLC90 geometry, bathymetry, and 1992 forcing data by
a factor of 3 (per linear dimension) into the LLC30 inputs consumed by this
`global_ocean.llc30x15` verification config. See the plan
`docs/plans/20261103-llc30-lightweight-verification-config.md` (Task 1/2 and
the "Technical Details" section) for the full derivation history.

## Scripts

- `inspect_mitgrid.py` (Task 1) — confirms the `.mitgrid` binary layout
  (field order, shape, byte-count formula) against the source files before
  any coarsening code was written. Run this first if you ever doubt the
  layout assumptions below.
- `coarsen_llc90_to_llc30.py` (Task 2) — does all the actual coarsening:
  grid tiles, bathymetry, and the 9 combforc forcing variables. Produces
  everything under `output/`.

## Source data (read-only, never modified)

- Grid (5 LLC90 face-geometry `.mitgrid` tiles, real\*8, big-endian):
  `/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/ancillary_data/ancillary_data_orig/native_grid_files/tile00{1,2,3,4,5}.mitgrid`
- Bathymetry (real\*4, big-endian, compact shape `(90,1170)`):
  `/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/ancillary_data/ancillary_data_orig/input_init/BATHY_ICE_SHELF_CAVITY_PLUS_ICE_FRONT_LLC_0090.bin`
- Forcing (9 combforc variables, real\*4, big-endian, hourly, 1992, compact
  shape `(90,1170)` per record, 8784 records):
  `/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/ancillary_data/ancillary_data_orig/input_forcing/combforc/eccov4r6_{dlw,dsw,pres,rain,spfh2m,tmp2m,ustr,vstr,wspeed}_1992`

## Run

```
export PATH=$HOME/micromamba/envs/podaac/bin:$PATH   # has numpy; the
                                                        # mitgcm_ad_v4r6 env
                                                        # does NOT (Fortran/
                                                        # MPI only)
cd MITgcm/verification/global_ocean.llc30x15/preprocess
python3 inspect_mitgrid.py            # Task 1 sanity check (optional re-run)
python3 coarsen_llc90_to_llc30.py     # Task 2 full preprocessing
```

Takes roughly 6-7 minutes end-to-end (dominated by reading the 9 x 3.7 GB
combforc files for the temporal/monthly reduction).

## .mitgrid binary layout (confirmed, see `inspect_mitgrid.py`)

Raw, unformatted, big-endian real\*8, no Fortran record markers. 16 fields
in this exact order, each shape `(ny+1, nx+1)` per facet (row-major, y
outer / x inner):

```
XC, YC, DXF, DYF, RAC, XG, YG, DXV, DYU, RAZ, DXC, DYC, RAW, RAS, DXG, DYG
```

Expected file size for a facet `(nx,ny)`: `16*(nx+1)*(ny+1)*8` bytes.

Per-facet LLC90 -> LLC30 dims (facets 1,2 tall; 3 square/Arctic cap; 4,5
wide):

| Facet | LLC90 (nx,ny) | LLC30 (nx,ny) | LLC90 bytes | LLC30 bytes |
|---|---|---|---|---|
| 1 | 90,270 | 30,90 | 3,156,608 | 361,088 |
| 2 | 90,270 | 30,90 | 3,156,608 | 361,088 |
| 3 | 90,90 | 30,30 | 1,059,968 | 123,008 |
| 4 | 270,90 | 90,30 | 3,156,608 | 361,088 |
| 5 | 270,90 | 90,30 | 3,156,608 | 361,088 |

Note facet 3's LLC30 output is 123,008 bytes, **not** 361,088 like the
other four — this is asserted explicitly in `write_mitgrid_tile()`.

A non-corner (center-type) field's extra padding row/column in the fine
`.mitgrid` files (the `+1` beyond `ny`/`nx`) is **zero-filled**, not a
duplicate of the last real row/column (confirmed empirically on
`tile001.mitgrid`'s `RAC`/`DXF`/`DYF`/`XC` fields — their last column is
all zeros). The coarsening code reproduces this exact zero-padding
convention when writing the LLC30 tiles.

## The six coarsening rules (and five padding patterns)

An earlier draft of this preprocessing used a simple "center-mean vs.
corner-subsample" split, which is **wrong** for lengths and areas (it
silently produces 1/3-scale lengths and 1/9-scale areas). A second-generation
fix correctly separated lengths/areas by sum vs. mean, but still
mis-classified `DXV`/`DYU`/`RAZ` as plain corner-position fields
(subsampled like `XG`/`YG`) — this is ALSO wrong, and was caught by a real
numerical blow-up (see "Corner-indexed length/area fields" below). The
actual, verified-correct rule set has 6 rules, one independently-implemented
function per rule:

1. **`coarsen_coord_lonlat`** — `XC`, `YC` (cell-center lon/lat): block-mean
   of each 3x3 block. `YC` uses a plain arithmetic mean; `XC` uses
   `atan2(mean(sin(lon)), mean(cos(lon)))` (circular/dateline-safe mean) —
   confirmed necessary because facet 4's data has 3x3 blocks straddling the
   +/-180 deg dateline, where a naive mean would give ~20 deg instead of the
   correct ~180 deg.
2. **`coarsen_corner`** — `XG`, `YG` (cell-corner lon/lat, POSITIONS): subsample
   every 3rd corner point directly, no averaging (coarse-cell corners
   coincide with a subset of fine-grid corners; averaging would displace
   the cell boundary).
3. **`coarsen_xlength_sum`** — `DXF`, `DXC`, `DXG` (x-direction
   lengths): sum of 3 consecutive fine values along x (a coarse cell's
   width is the sum, not the average, of its 3 constituent fine widths);
   the orthogonal (y) sub-rows are then combined with a mean.
4. **`coarsen_ylength_sum`** — `DYF`, `DYC`, `DYG` (y-direction
   lengths): sum of 3 consecutive fine values along y, mirroring rule 3;
   orthogonal (x) sub-columns combined with a mean.
5. **`coarsen_area_sum`** — `RAC`, `RAW`, `RAS` (areas): sum of the
   9 constituent fine-cell areas in each 3x3 block (not a mean).
6. **`coarsen_corner_indexed_symmetric`** — `DXV`, `DYU`, `RAZ`:
   corner-INDEXED (full `(ny+1,nx+1)` extent, no padding, same shape as
   `XG`/`YG`) but are length/area **quantities**, not positions. Subsampling
   them (as an earlier version of this fix did, reasoning "same shape as
   `XG`/`YG` so same rule") is WRONG: it picks one fine value where the
   coarse value should represent the sum of the ~3 (or ~9, for `RAZ`) fine
   values it spans, deflating `DXV`/`DYU` by ~3x and `RAZ` by ~9x. This was
   caught empirically, not just reasoned about: classifying `RAZ` as
   "corner" (subsample) during this fix's own development caused a real
   smoke-test crash (`uvel_max` exploding from ~0.1 to >100 within 9
   timesteps, aborting in `CALC_R_STAR`) traced to `mom_vecinv`'s vorticity
   term dividing by a `rAz` that was ~9x too small. The windowed-sum rule:
   each coarse index sums (`DXV`: along x; `DYU`: along y; `RAZ`: along
   both axes) the fine values its coarse cell spans, with any non-summed
   axis (`DXV`, `DYU` only) combined by mean. Verified: this gives `RAZ`
   the same order of magnitude as the coarse `RAC` (ratio ≈0.98 on
   tile001, expected since they are offset corner-vs-center cell areas),
   vs. ≈1/9 of that under the (wrong) subsample rule.

   **Facet-seam symmetry (second, independent fix, on top of the
   windowed-sum rule above):** a plain windowed sum, computed
   independently per facet using only that facet's OWN one-sided
   fine-grid neighborhood, produced boundary values that disagreed
   between adjacent facets by up to 13% at facet-to-facet seams — this
   was initially (incorrectly) attributed to a "geometric singularity"
   at triple-facet-junction corners, but a decisive re-check disproved
   that: raw fine-resolution `RAZ`/`DYU` values pulled directly from two
   adjacent `.mitgrid` files along their shared boundary are EXACTLY
   identical (`0.0` difference) at source resolution — there is no
   singularity in the actual grid data anywhere. The real cause: each
   facet's boundary coarse cell = [the shared boundary fine cell, which
   genuinely IS identical between facets, like `XG`/`YG`'s shared
   corner] + [that facet's OWN next-interior-neighbor fine cell, which is
   a genuinely DIFFERENT physical cell from the other facet's — nothing
   requires them to agree, and empirically they don't, by up to ~25% in
   `RAZ` at the worst-case connection]. `coarsen_corner_indexed_symmetric`
   fixes this structurally rather than tolerating the residual: for every
   internal (connected, non-open) seam, each facet's boundary window
   borrows one ghost cell from its neighbor's own interior data (so both
   sides' windows cover the IDENTICAL set of physical fine cells), and
   the handful of true multi-facet corner points — where even a
   ghost-cell pair is still an exact tie between 2 valid-but-not-
   bit-identical float64 summation orders — are resolved by a final
   corner-group averaging pass (RAZ: mean of all facets sharing that
   physical corner; DXV/DYU: pairwise mean of each connection's 2
   endpoints). Result: every one of the 16 `facetEdgeLink` connections
   now agrees to ~1e-15 relative (pure float64 rounding), not a 13%
   "tolerance" — see `check_facet_boundary_continuity_quantity`'s
   `rel_tol=1e-9`, six orders of magnitude of margin above the measured
   noise floor and far below either previous bug class (13%–97%).

## Padding patterns (separate from the coarsening rule above)

A SECOND, independent classification determines which of a field's extra
"+1" row/column (beyond the true `(ny,nx)` cell-quantity extent) is
zero-padding vs. real data. An earlier version of this script assumed a
single blanket pattern — every non-`XG`/`YG` field's extra row AND column
are zero-padding — which is empirically false for 9 of the 14 remaining
fields (confirmed by direct inspection of `tile001.mitgrid`: compare each
field's last-row/last-column mean magnitude against its interior mean
magnitude). The correct classification has 5 patterns:

- **`both_pad`** — `XC`, `YC`, `DXF`, `DYF`, `RAC`: last ROW **and** last
  COLUMN are zero-padding.
- **`y_pad_x_real`** — `DXC`, `DYG`, `RAW`: only the last ROW is
  zero-padding; the last COLUMN holds real, non-zero boundary data of the
  same magnitude as the interior and is coarsened (reduced along its
  remaining fine axis), not discarded.
- **`x_pad_y_real`** — `DYC`, `DXG`, `RAS`: mirror of the above — only the
  last COLUMN is zero-padding, the last ROW is real.
- **`corner`** — `XG`, `YG`: neither row nor column is padding — both are
  real corner-position data over the full extent, subsampled (rule 2).
- **`corner_indexed_real`** — `DXV`, `DYU`, `RAZ`: also neither row nor
  column is padding (full real extent, like `XG`/`YG`), but handled by the
  windowed-sum rule (rule 6), not subsampling.

## Compact-grid `(90,1170)` facet layout — IMPORTANT CORRECTION

The plan's Technical Details section describes the compact `(90,1170)`
array as stackable-by-ROW-RANGE across all 5 facets, treating a facet's
"natural" slice as a uniform `(Δrows, 90)` chunk. **This is not what the
actual on-disk data looks like for the wide facets (4,5), and the row-range
description was corrected empirically while implementing this task:**

The compact file is 5 facet blocks concatenated in the *flat* buffer, each
stored in **that facet's own native `(ny,nx)` row-major order** — not a
uniform `(1170,90)` 2-D array that can be row-sliced for every facet. For
the narrow facets (1,2,3; `nx=90`) this is indistinguishable from
row-slicing a `(1170,90)` matrix, since their native `nx` already equals
the nominal compact width of 90. For the **wide** facets (4,5; `nx=270`) it
is **not**: naively slicing 90 rows out of a `(*,90)`-shaped 2-D array and
then transposing/reshaping to `(90,270)` produces a field that is ~15-20x
rougher (checked via a finite-difference roughness metric) and has a
visibly wrong land/ocean mask compared to reshaping that facet's flat byte
range directly to its native `(ny=90, nx=270)` shape.

The **flat element ranges** in the plan's table are still correct (facet 1:
elements `[0, 270*90)`; facet 2: `[270*90, 540*90)`; facet 3: `[540*90,
630*90)`; facet 4: `[630*90, 900*90)`; facet 5: `[900*90, 1170*90)` — i.e.
`(r1-r0)*90` elements per facet using the plan's row-range numbers) — what
differs is the **reshape target**: reshape each facet's flat slice directly
to its own native `(ny,nx)`, not to `(r1-r0, 90)`.

`split_compact_into_facets()` / `assemble_facets_into_compact()` in
`coarsen_llc90_to_llc30.py` implement this corrected convention, with the
reasoning and empirical check documented in their docstrings/comments.

### Decisive orientation verification (resolves a real reviewer disagreement)

The checks above (RAC conservation, per-facet forcing spot-check) are all
invariant to within-facet reordering/transpose — a wrong facet-4/5 reshape
would still pass them. The actual decisive test,
`check_facet_boundary_continuity()`, directly compares `XG`/`YG`
corner-point values across every `facetEdgeLink`-implied facet-to-facet
connection (16 connections total across the 5 facets, using the
edge-connectivity convention from `pkg/exch2/w2_set_f2f_index.F`): two
genuinely adjacent facets must share *identical* corner coordinates along
their common edge, to within floating-point rounding, since `XG`/`YG` are
exact (subsampled, not averaged) positions — any orientation bug (wrong
transpose, reversed index order, wrong facet pairing) breaks this exactly,
not approximately. Run on both the LLC90 fine source grid and the LLC30
coarsened output, **every one of the 16 connections passed with max
residual ~1e-13 degrees** (pure floating-point rounding noise) in both
cases — decisively confirming the facet-4/5 orientation described above is
correct, not merely "better than one wrong alternative." This check runs
every time the script executes (not a one-off), so a future regression
would be caught immediately.

An analogous check, `check_facet_boundary_continuity_quantity()`, does the
same for `DXV`/`DYU`/`RAZ` (see rule 6's "Facet-seam symmetry" writeup
above) — **every one of the 16 connections now agrees to ~1e-15 relative**
(float64 rounding), checked against a `rel_tol=1e-9` tolerance.

## Bathymetry coarsening

Real\*4, big-endian, compact shape `(90,1170)`, confirmed via
`90*1170*4 = 421,200` bytes (exact match to the source file). Land cells
are exactly `0.0`; ocean cells are negative depth values (confirmed by
inspection: ~41% of the domain is land/zero).

Coarsening rule: 3x3 block **minimum depth magnitude over wet cells only**
(`coarsen_bathy_facet`/`coarsen_bathy_block`) — i.e. the shallowest
(safest/most conservative) of the wet constituent cells, preserving its
sign. A coarse cell is land (`0.0`) only if **all 9** constituent cells are
land. Output shape `(30,390)`, size `30*390*4 = 46,800` bytes (verified on
write). Two correctness checks run on every execution
(`verify_bathy_correctness()`): (1) land-fraction-preserved — the coarse
land fraction must be <= the fine land fraction (coarsening can only turn
land into ocean, never the reverse) and not implausibly far below it; (2)
depth-bounds-preserved — every coarse wet depth must fall within the fine
field's `[min,max]` range, since min-magnitude-over-wet-cells selection
cannot manufacture a depth outside the source range.

## Forcing reduction

Real\*4, big-endian, compact shape `(90,1170)` per hourly record, 8784
records (1992, a leap year — Feb has 29 days; month lengths used are
`31,29,31,30,31,30,31,31,30,31,30,31` days x 24h, verified to sum to 8784).

Two-stage reduction per variable:

1. **Temporal**: arithmetic mean per calendar month over all hourly records
   -> 12 records of shape `(90,1170)`.
2. **Spatial**: 3x3 block-**average** (mean, not min/sum — these are
   flux/state fields, not bathymetry or geometric lengths) of each monthly
   record, using the same corrected facet-layout convention as bathymetry,
   -> `(30,390)`.

Output: one file per variable, 12 monthly records of shape `(30,390)`
real\*4 big-endian, `12*30*390*4 = 561,600` bytes (verified on write),
named `eccov4r6_<var>_llc30_clim` (e.g. `eccov4r6_dlw_llc30_clim`).

### Vector-field / dateline caveat (documented, not silently ignored)

`ustr`/`vstr` are wind-stress vector components on a curvilinear grid.
Facets 4/5 and the Arctic cap (facet 3) are rotated relative to facets 1/2,
so naive per-component 3x3 block-averaging mixes u/v components
differently across facet seams. A fully correct treatment would rotate
vectors to a common reference frame before averaging and rotate back after.
**This script does not do that** — it applies the same naive
per-component block-mean used for all other (scalar) forcing fields. This
is accepted as a known simplification for this smoke-test config (no
scientific validation is claimed anywhere in this plan — see the plan's
Non-Goals section). The dateline/circular-mean issue identified for
`XC`/`XG` above does **not** apply here: forcing fields have no longitude
value to average, only flux/stress magnitudes.

## Mandatory verification results (this run)

- **RAC conservation** (catches a mean-vs-sum area bug, but is
  orientation-invariant — see the boundary-continuity check below for the
  orientation-sensitive verification): fine-grid global `RAC` total =
  `5.099044e+14 m^2`; coarse-grid global `RAC` total = `5.099044e+14 m^2`;
  relative difference = `0.0` (bit-identical in this run); ratio to Earth's
  surface area (5.1e14 m^2) = `0.9998` — **PASSED**.
- **Facet-boundary-continuity check** (decisive, orientation-sensitive —
  see the "Decisive orientation verification" section above): all 16
  `facetEdgeLink`-implied connections passed, both for the LLC90 fine
  source grid and the LLC30 coarsened output, with max residual `~1e-13`
  degrees in `XG` and `YG` at every connection — **PASSED**.
- **Facet-boundary-continuity check, `DXV`/`DYU`/`RAZ`** (fixes the
  facet-seam symmetry bug described under rule 6 above): all 16
  `facetEdgeLink`-implied connections passed on the LLC30 coarsened
  output, with max residual `5.680e-15` relative (float64 rounding,
  tolerance `1e-9`) — **PASSED**. The previous worst-case connection
  (`facet1.E <-> facet2.W`) independently re-verified by reading the
  written `output/grid/tile001.mitgrid`/`tile002.mitgrid` files directly
  off disk: `RAZ` max relative diff `2.161e-16`, `DYU` max relative diff
  `1.609e-16`.
- **Byte-size assertions**: every one of the 5 grid tiles, the bathymetry
  file, and the 9 forcing climatology files matched its expected size
  exactly (see per-facet table above and the `561,600`/`46,800` figures) —
  **PASSED**.
- **Bathymetry correctness** (land-fraction-preserved and
  depth-bounds-preserved): fine land fraction `0.4079`, coarse land
  fraction `0.3644` (coarse <= fine, as expected); fine wet-depth range
  `[-5997.99, -19.99]` m, coarse wet-depth range `[-5959.98, -19.99]` m
  (within the fine range, as expected) — **PASSED**.
- **Spot-check** (`dlw`, January mean, PER FACET): all 5 facets passed with
  relative difference at machine-precision level (worst `1.873e-16`) —
  **PASSED** (well within the "a few percent" tolerance; exact agreement
  here is expected since a pure block-mean commutes exactly with a prior
  temporal mean for this reduction order). Note this check is per-facet
  but still mean-based, so it verifies arithmetic + facet identity, not
  within-facet orientation — see the boundary-continuity check above for
  that.

## Output

- `output/grid/tile00{1,2,3,4,5}.mitgrid` — LLC30 geometry tiles.
- `output/bathy/BATHY_LLC30.bin` — LLC30 bathymetry, compact shape
  `(30,390)`.
- `output/forcing/eccov4r6_<var>_llc30_clim` — 9 files, 12-month LLC30
  forcing climatology each, compact shape `(30,390)` per record.

`output/` is a generated/intermediate build-artifact directory (Task 5 of
the plan adds a `.gitignore` covering this at the config-root level).
