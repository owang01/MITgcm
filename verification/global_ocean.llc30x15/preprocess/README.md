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
  everything under `output/` (a scratch directory — see "Output" below for
  why the committed files actually live in `../input/`).

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

`output/` is this script's gitignored scratch/regeneration directory, not
the config's actual input location. The committed, authoritative copies
of all 15 files live directly in `../input/` (same convention as every
other `verification/` experiment's own grid/forcing binaries, e.g.
`tutorial_held_suarez_cs/input/grid_cs32.face00?.bin`) — after running
this script, copy `output/{grid,bathy,forcing}/*` into `../input/` to
update them.

## Initial condition (Theta/Salt) — `coarsen_initial_ts_to_llc30.py`

A separate script, `coarsen_initial_ts_to_llc30.py`, coarsens real V4r6
initial Theta/Salt into a 15-level LLC30 initial condition, replacing this
config's original flat analytic IC (`tRef=15*20., sRef=15*35.`, no
`hydrogThetaFile`/`hydrogSaltFile` at all). It is a separate script from
`coarsen_llc90_to_llc30.py` because it solves a genuinely different
problem — a **vertical** remap (50 production levels → LLC30's own 15
levels), not just the horizontal 3:1 coarsening every other field here
needs — but it imports and reuses `coarsen_llc90_to_llc30.py`'s
`split_compact_into_facets()`/`assemble_facets_into_compact()` directly
(same compact-array/facet-orientation convention, unchanged).

### Source data (read-only, never modified)

- Real V4r6 initial Theta/Salt: a production pickup/restart file,
  `/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/ancillary_data/ancillary_data_orig/input_init/pickup.0000000001.data`
  — confirmed (by direct byte-level read-back, not just its `.meta`
  header) to be raw, unformatted, big-endian real\*8, no record markers,
  453 stacked `(1170,90)` records exactly matching the file's actual size
  (`453*1170*90*8 = 381,607,200` bytes). Field order (per the `.meta`
  file's `fldList`): `Uvel, Vvel, Theta, Salt, GuNm1, GuNm2, GvNm1, GvNm2,
  AddMass` (each 50 levels = 450 records), then `EtaN, dEtaHdt, EtaH`
  (2-D, 3 more records) — `8*50 + 50 + 3 = 453`. **Theta occupies records
  100–149, Salt occupies records 150–199**, k=0 is the surface (confirmed
  empirically: level-0 Theta min/max ≈ −2.54/32.39°C, physically sane SST;
  level-49 ≈ −0.73/1.86°C, physically sane abyssal temperature — not
  reversed or duplicated).
- Real fine (LLC90) bathymetry:
  `/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/ancillary_data/ancillary_data_orig/input_init/BATHY_ICE_SHELF_CAVITY_PLUS_ICE_FRONT_LLC_0090.bin`
  — real\*4 big-endian, same `(1170,90)` compact convention as this
  script's `BATHY_SRC` (confirmed: `90*1170*4 = 421,200` bytes matches the
  file size exactly). **This is deliberately NOT the same file as the
  already-coarsened `../input/BATHY_LLC30.bin`** — the real fine
  bathymetry is the only place per-column seafloor depth is still precise
  enough to build a correct wet/dry mask; `BATHY_LLC30.bin` has already
  lost that precision via its own min-magnitude-over-wet-cells rule.
- Production V4r6's 50-level `delR`, `tRef`, and `sRef` — verbatim from
  `Time-invariant-adjoint/v4r6/input.V4r6.102/data`'s `&PARM01`/`&PARM04`.
  There is no local file in this repo to parse these from, so they are
  hardcoded in `coarsen_initial_ts_to_llc30.py`, guarded by explicit
  length/sum asserts (`delR` sums to 6134.5 m over 50 levels; `tRef`'s
  Fortran repeat-count literal expands to exactly 50 values) against a
  hand-transcription error, mirroring this directory's own
  `MONTH_HOURS_1992` leap-year-sum assert pattern.

### Design: two-stage remap (vertical first, then horizontal)

T/S coarsening needs a **vertical** remap that nothing else in this
preprocessing pipeline needs (every other field — grid metrics,
bathymetry, forcing — is purely 2-D/horizontal). The two stages, in
order, and why the order matters:

1. **Vertical remap, at full LLC90 horizontal resolution.** For each
   LLC90 fine column, remap the 50 production levels down to LLC30's
   *existing* 15 levels (parsed directly out of `../input/data`'s
   `&PARM04` at script load time — NOT re-hardcoded a second time, since
   that `input/data` text is the actual source of truth this script must
   stay consistent with) via a depth-weighted overlap-conservative
   average, masked by a **fine 50-level wet mask** built from the real
   fine bathymetry and production's 50-level `delR`.
2. **Horizontal 3:1 coarsening**, on the now-15-level, still-LLC90-
   resolution fields, using a **second, independently-computed** wet mask
   (same fine bathymetry, now evaluated against LLC30's own 15-level
   `delR`) and a new wet-cell-only mean.

Vertical must come first: the real fine bathymetry is the only place
per-column seafloor depth is still precise, so it's the only correct
basis for deciding which source levels are physically meaningful for a
given column. Doing horizontal-first would force deciding wet/dry on an
already-horizontally-blurred depth, and would need extra wet masks at
every stage instead of exactly two. The two wet masks are intentionally
kept as two separate, independently-named variables (`wet50` for Stage 1,
`wet15` for Stage 2) — reusing one for the other's purpose is the most
likely copy-paste bug given how similar they are (same function, same
bathymetry, different `delR` argument).

### The new coarsening rule: wet-cell-only mean

Neither existing rule in `coarsen_llc90_to_llc30.py` is correct for T/S:
`coarsen_bathy_block`'s min-magnitude-over-wet-cells rule is right for
bathymetry (pick the shallowest/safest depth) but wrong for a state
variable like temperature; `coarsen_forcing_facet_mean`'s plain
unweighted mean is right for 2-D flux/state forcing fields (every
constituent cell is always wet) but wrong for T/S, which must exclude dry
(below-seafloor) constituent cells from the average or the result would
be biased toward whatever fill/stale value sits below the real seafloor.
`coarsen_ts_facet_wet_mean` (new, in `coarsen_initial_ts_to_llc30.py`)
follows `coarsen_bathy_facet`'s vectorized reshape/transpose style but
averages only over the wet constituents of each 3×3 block, with a
`wet_frac_coarse` fraction returned alongside the value — used by the
surface-wet-fraction correctness check (see below), not fed back into the
model.

### Verification

- **Byte-count assertions** at every read/write (pickup source:
  `381,607,200` bytes; each output file: `15*390*30*4 = 702,000` bytes).
- **Overlap-matrix row-sum check**: the vertical-remap weight matrix's
  row sums must exactly equal LLC30's 15-level `delR` — the vertical-
  remap analogue of this directory's own `verify_rac_conservation`.
- **Bounds-preserved per level**: coarse T/S (wet cells only) must stay
  within that level's Stage-1 fine min/max — guaranteed by construction
  (nested weighted means), asserted anyway since a masking bug would show
  up here as an out-of-range value.
- **Surface wet-fraction vs. `../input/BATHY_LLC30.bin`**: this script's
  own surface wet-cell-count land fraction is compared, per facet,
  against the already-committed `BATHY_LLC30.bin`'s min-magnitude-based
  land fraction. Different aggregation rules, so exact agreement isn't
  guaranteed in principle — in practice, on this data, every facet
  matched to 4 decimal places (worst per-facet diff `0.0000`).
- **Mandatory pickup fill-value sanity check** (the one check that can't
  be skipped): directly inspects raw pickup Theta/Salt values at a
  handful of known-shallow fine columns, at levels the 50-level wet mask
  marks dry, and asserts they're small-magnitude, not a suspicious
  sentinel (e.g. not ±999/±9999). This exists because an off-by-one in
  the wet-mask's `rF[k] > seafloor` comparison (`>` vs. `>=`, or the wrong
  interface) would NOT necessarily be caught by the bounds-preserved or
  byte-count checks if the leaked fill value happened to look physically
  plausible — a quiet failure mode, unlike the loud numerical blow-ups
  the original grid-coarsening review cycle caught (see the module
  docstring in `coarsen_llc90_to_llc30.py` for that history).
- **Known-location spot-checks**: looks up the LLC30 cell nearest the
  equatorial Pacific and the Weddell Sea (via the same `tile00X.mitgrid`
  XC/YC lookup this repo's plotting scripts use) and asserts surface
  Theta is in a physically sane range for each (confirmed: 29.2°C and
  −1.9°C respectively).
- **End-to-end**: build + `mpiexec -np 1 ./mitgcmuv` the 10-step smoke
  test with the new IC linked in. Confirmed `STOP NORMAL END`, no NaN, no
  `ABNORMAL END` from `model/src/ini_theta.F`'s own `checkIniTemp` check
  (left at its default `.TRUE.` — a wet cell with `theta==0.0` would
  trigger a hard stop there, a free built-in regression check against
  exactly the masking-bug class this design is most at risk of). The
  model's own `dynstat_theta_max/min/mean` monitor output stayed
  physically sane and stable across all 10 steps.

### Output

- `output/initial_condition/THETA_LLC30_INIT.bin` and
  `SALT_LLC30_INIT.bin` — real\*4 big-endian, 15 stacked `(390,30)`
  compact records each (`15*390*30*4 = 702,000` bytes).
- `output/initial_condition/tRef_sRef_llc30.txt` — paste-ready text
  (`tRef= v1,...,v15,` / `sRef= 15*34.5,`) for manual insertion into
  `../input/data`'s `&PARM01` — namelist literals aren't something this
  script can write into `data` directly. `sRef` remaps to exactly flat
  `34.5` since production `sRef` is itself flat (also a useful trivial
  regression check on the remap function).

Same manual-copy convention as grid/bathy/forcing: `output/` is this
script's own gitignored scratch/regeneration directory, not the config's
actual input location — after running, copy the two `.bin` files into
`../input/` by hand (nothing automates this), and manually paste the
printed `tRef`/`sRef` values into `../input/data`.
