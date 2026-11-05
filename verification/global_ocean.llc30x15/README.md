LLC30 lightweight LLC/exch2 verification config, 30x30x15 (5 facets, 13 tiles)
================================================================================

This is a small, laptop-friendly MITgcm `verification/` config that exercises
the real LLC (lat-lon-cap) face/`pkg/exch2` topology: 5 facets, 13 `30x30`
tiles, 15 vertical levels. It exists because no LLC/exch2 config previously
existed anywhere in this `verification/` tree -- the two other "lightweight
global ocean" configs, `global_ocean.cs32x15` (cubed-sphere exchange) and
`global_ocean.90x40x15` (plain lat-lon), both deliberately avoid exch2/LLC,
and the only other LLC config in this project (`global_oce_llc90`, in a
separate contrib repo) is full production scale (595M, 12+ cores) and far too
heavy to build/run casually.

The grid, bathymetry, forcing, and initial Theta/Salt are not idealized/
analytic: they are real ECCO V4r6 LLC90 fields, coarsened 3:1 per linear
dimension (horizontally) and, for the initial condition, also remapped
vertically from V4r6's 50 levels to this config's own 15 (see "Data
provenance and coarsening method" below, and
`preprocess/README.md`'s "Initial condition" section). This makes the config
a general LLC
dev sandbox -- useful on its own for testing `pkg/exch2`/tiling/core-MITgcm
changes -- and a possible future foundation for adding `pkg/ecco`/`pkg/ctrl`/
adjoint support (not attempted here; see Non-Goals).

Build
-----

```
cd build
export PATH=$HOME/micromamba/envs/mitgcm_ad_v4r6/bin:$PATH   # gfortran, OpenMPI, netcdf-fortran
../../../tools/genmake2 -mods=../code -optfile=../code/linux_amd64_gfortran+mpi_local -mpi
make depend
make
cd ..
```

This produces `build/mitgcmuv`. The optfile pins `gfortran`+OpenMPI
(`mpicc`/`mpifort`) and forces big-endian binary I/O (`-fconvert=big-endian`)
-- every binary input file in this config (mitgrid tiles, bathymetry, forcing
climatology) is big-endian, so a different toolchain/optfile without that
flag will misread them.

Run
---

```
mkdir run
cd run
ln -s ../input/* .
ln -s ../build/mitgcmuv .
mpiexec -np 1 ./mitgcmuv > output.txt
cd ..
```

The grid (`tile00{1..5}.mitgrid`), bathymetry (`BATHY_LLC30.bin`), and
forcing climatology (`eccov4r6_*_llc30_clim`) files are committed directly
in `input/`, same as every other `verification/` experiment's own grid/
forcing binaries (e.g. `tutorial_held_suarez_cs/input/grid_cs32.face00?.bin`)
-- no `prepare_run`/separate symlink step is needed. They were generated
once by `preprocess/coarsen_llc90_to_llc30.py` (see "Data provenance and
coarsening method" below); re-run that script only if you need to
regenerate them from the source LLC90 data.

`nPx=1` (all 13 tiles run on a single process, per `code/SIZE.h`), so
`mpiexec -np 1` is correct and sufficient here -- do not guess at a
different `-np`; the tile/process decomposition is baked into `SIZE.h` at
build time (see Technical Details in the plan this config was built from).

This is a short (10-step) forward-only smoke test, not a scientifically
meaningful integration.

Comparison output
------------------

There is comparison/proof output in:

    results/output.txt

This is the `STDOUT.0000` of a successful run: ends in
`PROGRAM MAIN: Execution ended Normally` with `CONFIG_CHECK : Normal End`,
reaches `time_tsnumber = 10` (the full `nIter0(0)+nTimeSteps(10)` requested
in `input/data` PARM03), and reports
`W2_E2SETUP: Total number of Facets = 5` /
`W2_E2SETUP: Total number of Tiles = 13` from the `pkg/exch2` setup banner.

Comments
--------

* This config uses `pkg/exch2` with an explicit (`preDefTopol=0`) 5-facet
  LLC topology (`input/data.exch2`) -- the standard ECCO LLC
  `facetEdgeLink` connectivity, carried over unchanged from the production
  LLC90 reference, with `dimsFacets` scaled down 3:1.
* `useCubedSphereExchange=.TRUE.` in `input/eedata` is required even though
  this is not a true cubed-sphere grid: `pkg/exch2` reads this flag itself to
  select vector component/sign rotation across facet seams on any cube-like
  topology, LLC included.
* Forcing comes entirely through `pkg/exf` (see `input/data.exf`): 9 fields
  (`ustr`,`vstr`,`tmp2m`,`spfh2m`,`rain`,`wspeed`,`dsw`,`dlw`,`pres`), each a
  12-month climatology, no interannual variability.
* `useExfCheckRange=.FALSE.` is set deliberately, not as an oversight --
  see "Known limitations" below.
* `nSx=13, nSy=1, nPx=1, nPy=1` (all 13 tiles single-process) is a
  laptop-friendly choice, not a tuned decomposition; revisit if genuine
  multi-process `pkg/exch2` exchange testing becomes a goal.

Build gotchas
-------------

Three runtime namelist/CPP-option requirements were discovered the hard way
while first bringing this config up (each caused a hard `ABNORMAL END` on an
otherwise-clean build) -- if you ever rebuild from scratch and hit a similar
stop, the source-of-truth explanation for each is the inline `Task 6 fix`
comment at its own setting, not this README:

* `input/data.ggl90`'s `GGL90TKEmin` (see the `Task 6 fix` comment there).
* `input/data`'s `PARM01` `implicitViscosity=.TRUE.` (see the `Task 6 fix`
  comment there).
* `code/CPP_OPTIONS.h`'s `#define NONLIN_FRSURF` (see the `Task 6 fix`
  comment there) -- this one requires a full `genmake2`/`make depend`/`make`
  rebuild, not just a namelist edit, since it's a compile-time flag.

Data provenance and coarsening method
--------------------------------------

This config's grid, bathymetry, and forcing are derived from real ECCO V4r6
LLC90 fields by the scripts in `preprocess/` (see `preprocess/README.md` for
the full derivation, including every byte-size/conservation check). Exact
source paths (local, read-only, never modified by this config or its
preprocessing scripts):

* **Grid** (5 LLC90 face-geometry `.mitgrid` tiles, real\*8, big-endian):
  `/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/ancillary_data/ancillary_data_orig/native_grid_files/tile00{1,2,3,4,5}.mitgrid`
* **Bathymetry** (real\*4, big-endian, compact shape `(90,1170)`):
  `/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/ancillary_data/ancillary_data_orig/input_init/BATHY_ICE_SHELF_CAVITY_PLUS_ICE_FRONT_LLC_0090.bin`
* **Forcing** (9 combforc variables, real\*4, big-endian, hourly, 1992,
  compact shape `(90,1170)` per record, 8784 records):
  `/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/ancillary_data/ancillary_data_orig/input_forcing/combforc/eccov4r6_{dlw,dsw,pres,rain,spfh2m,tmp2m,ustr,vstr,wspeed}_1992`

Coarsening is 3:1 per linear dimension (LLC90 -> LLC30), applying three
distinct, field-appropriate rules: (1) the grid's 16 `.mitgrid` fields are
each coarsened by one of 6 physically-motivated rules depending on field
type -- coordinate fields block-mean (with a dateline-safe circular mean for
longitude), corner-position fields (`XG`,`YG`) subsample every 3rd point,
x/y-length fields sum 3 consecutive fine values along their axis (with the
orthogonal axis mean-combined), area fields sum the 9 constituent fine-cell
areas, and corner-INDEXED length/area fields (`DXV`,`DYU`,`RAZ` -- full
extent, no padding, like `XG`/`YG`, but a length/area quantity rather than a
position) use a dedicated windowed-sum rule rather than the corner-position
subsample rule, which would deflate them by ~factor-squared -- never a plain
block-mean for lengths/areas, which would silently produce 1/3-scale lengths
and 1/9-scale areas; (2) bathymetry is coarsened by 3x3-block minimum depth
magnitude over wet cells only (a coarse cell is land only if all 9
constituents are land), the most conservative choice that avoids
manufacturing dry-side artifacts, with a land-fraction-preserved and
depth-bounds-preserved check run on every execution; (3) forcing is reduced
in two stages -- first a 1992 monthly climatology (arithmetic mean per
calendar month, 8784 hourly records -> 12 monthly records), then a 3x3
spatial block-mean of each monthly field. The mandatory RAC-conservation
check (coarse-grid vs. fine-grid global cell-area sum) passed with a
relative difference of 0.0, and a decisive, non-permutation-invariant
facet-boundary-continuity check (comparing `XG`/`YG` corner values across
every `facetEdgeLink`-implied facet connection) passed with residuals at
floating-point-rounding level (~1e-13 degrees) at all 16 connections, in
the run that produced the committed `input/` grid/bathymetry/forcing
files; see `preprocess/README.md` for that and all other verification
results in full.

Non-Goals / known limitations
------------------------------

Read this section before using this config for anything beyond what it
claims:

* **No adjoint, `pkg/ecco`, or `pkg/ctrl` support.** This is a forward-only
  config. No `code_ad`/`input_ad` siblings exist (unlike
  `global_ocean.90x40x15`'s pattern) -- adding them is explicitly deferred
  future work, not attempted here.
* **No scientific validation of the physics.** "Done" for this config means
  "builds and completes a short forward integration without crashing or
  producing NaN/Infinity" -- nothing more. There is no claim that the
  resulting circulation, heat/freshwater budgets, or any other diagnostic is
  physically meaningful. In particular:
  - `useExfCheckRange=.FALSE.` is set in `input/data.exf` because the
    coarsened forcing produces some out-of-range values (e.g. `hflux` up to
    ~2300 W/m^2 against `pkg/exf`'s hardcoded 1600 W/m^2 sanity cap) that are
    a direct, expected consequence of monthly-mean + naive 3x3 block-average
    coarsening, not a sign of a correct, validated flux field. Note this
    flag also globally suppresses `ustress`/`vstress` range checks, which
    could otherwise independently catch a future unit/sign error in wind-
    stress coarsening -- there is no narrower escape hatch in `pkg/exf` to
    disable only the `hflux` check, so this is an accepted trade-off, not
    an oversight.
  - A warming/drift trend tied to the `hflux` outliers this
    `useExfCheckRange=.FALSE.` setting permits through is plausible over a
    longer integration than the 10-step smoke test exercises here; this has
    not been root-caused (per this config's Non-Goals). (An earlier version
    of this note measured `dynstat_theta_max` drift against this config's
    original flat `tRef=15*20.` initial condition -- that measurement is
    stale now that the IC is real V4r6-derived data, see "Initial
    condition" above; a future reader extending the integration length
    should re-measure drift against the current IC rather than reuse the
    old number.)
  - The `ustr`/`vstr` wind-stress vector components are coarsened by naive
    per-component block-averaging, without rotating to a common reference
    frame first, even though facets 4/5 and the Arctic cap are rotated
    relative to facets 1/2. This mixes vector components incorrectly across
    facet seams. Documented in `preprocess/README.md`, not fixed here.
  - Isolated single-cell bathymetry artifacts from block-coarsening (e.g. at
    coastlines) are accepted as-is; no manual smoothing or visual QA of the
    coarsened coastlines has been done.
* **Not a production-ready LLC30 setup.** This is a dev/test sandbox and a
  smoke test, not a configuration anyone should run multi-year integrations
  with, publish results from, or treat as a reduced-resolution substitute
  for production LLC90 ECCO V4r6. Multi-year/interannual forcing, parallel-
  decomposition tuning, and any reconciliation with NAS/Pleiades production
  output are all out of scope and were never attempted.
