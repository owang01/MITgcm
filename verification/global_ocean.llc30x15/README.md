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
changes, and a real testbed for ECCO V4r6's adjoint machinery: a full
`code_ad`/`input_ad` adjoint setup (real V4r6 package list -- `gmredi`,
`seaice`, `shelfice`, `steep_icecavity`, `salt_plume`, plus the adjoint
stack `autodiff cost ctrl grdchk ecco profiles smooth`) exists alongside
the forward config -- see "Adjoint build and run" below.

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

This is a short (10-step) forward smoke test, not a scientifically
meaningful integration.

Adjoint build and run
----------------------

The adjoint experiment (`code_ad`/`input_ad`) builds on top of the forward
config above with the full real-V4r6 adjoint package stack (`autodiff cost
ctrl grdchk ecco profiles smooth`). It requires TAF (FastOpt's source-to-
source AD tool) -- this project uses the remote `staf` service (point
`PATH` at a directory containing a `staf` wrapper script, e.g. `~/bin/`).

**Via the Docker harness** (the primary, documented way to build/run this
-- works from any machine with Docker, does not require TAF/MPI/NetCDF
installed locally beyond what the container provides):

```
./experiment_compile.sh global_ocean.llc30x15 -mpi -adm -taf_dir ~/bin/ -clean
./experiment_run_no_compile.sh global_ocean.llc30x15 -adm -mpi 13
```

(`experiment_compile.sh`/`experiment_run_no_compile.sh` are symlinked in
from the sibling `MITgcm_verification_docker` repo -- see that repo's own
README if these scripts are missing.) This reaches `PROGRAM MAIN:
Execution ended Normally` with a real non-zero cost function and writes
output into `output_docker_adm/`.

**Directly via `genmake2`/`mpiexec`** (faster, no Docker required, but
needs TAF reachable, plus `gfortran`+OpenMPI+NetCDF-Fortran -- e.g. the
`mitgcm_ad_v4r6` micromamba env this config was developed against):

```
cd build_ad
cp ../code_ad/SIZE.h_mpi ./SIZE.h.mpi    # SEE WARNING BELOW
ln -s SIZE.h.mpi SIZE.h
export PATH=$HOME/micromamba/envs/mitgcm_ad_v4r6/bin:$HOME/bin:$PATH  # gfortran, OpenMPI, netcdf-fortran, staf
../../../tools/genmake2 -mods=../code_ad -optfile=../code/linux_amd64_gfortran+mpi_local -mpi
make depend
make adall
cd ../run_ad
mkdir -p diags   # pkg/diagnostics writes here; not created automatically
mpiexec --oversubscribe -np 13 ./mitgcmuv_ad
```

**WARNING, read before using `-mpi` on ANY variant of this config**:
`genmake2`'s default `SIZE.h` (from `code/` or `code_ad/`) is the
single-process decomposition (`nPx=1, nSx=13` -- all 13 tiles on one
process, the forward config's own documented laptop-friendly default, see
"Run" above). The genuine 13-rank MPI decomposition (`nPx=13, nSx=1`)
lives in a SEPARATE file, `SIZE.h_mpi`, which must be staged as
`SIZE.h.mpi` + a `SIZE.h` symlink pointing to it (as shown above) BEFORE
calling `genmake2 -mpi` -- `genmake2 -mpi` alone does NOT do this
automatically; it only adds `-DALLOW_USE_MPI` to the preprocessor defines,
nothing else. Skipping this step, or passing `-mpi` without first staging
`SIZE.h.mpi`, produces a binary that either (a) silently runs as 13
independent single-process copies if `ALLOW_USE_MPI` was never defined at
all (each one racing to create/delete an identical `scratch1.000000000`
temp file -- `Fortran runtime error: File cannot be deleted`), or (b)
correctly detects 13 MPI ranks but then fails `EEBOOT_MINIMAL: No. of
procs=13 not equal to nPx*nPy=1` if `ALLOW_USE_MPI` was defined but
`SIZE.h` was never switched to the MPI variant. (MITgcm's own
`verification/testreport` script, which the Docker harness above uses
internally, stages this automatically for `-MPI=N` builds -- a bare
`genmake2 -mpi` invocation does not replicate that step.)

Comparison output
------------------

There is comparison/proof output in:

    results/output.txt

This is the `STDOUT.0000` of a successful forward run: ends in
`PROGRAM MAIN: Execution ended Normally` with `CONFIG_CHECK : Normal End`,
reaches `time_tsnumber = 10` (the full `nIter0(0)+nTimeSteps(10)` requested
in `input/data` PARM03), and reports
`W2_E2SETUP: Total number of Facets = 5` /
`W2_E2SETUP: Total number of Tiles = 13` from the `pkg/exch2` setup banner.

The adjoint's own comparison/proof output is in:

    results/output_adm.txt

Same `STDOUT.0000`-of-a-successful-run convention, for the `-adm` build:
ends in `PROGRAM MAIN: Execution ended Normally`, `STOP NORMAL END`, with
a real non-zero `global fc` cost-function value.

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

* **Adjoint (`pkg/ecco`/`pkg/ctrl`/TAF) support DOES exist** -- see
  "Adjoint build and run" above; this note previously said otherwise and
  was stale (adjoint support was added after this README was first
  written). 4 time-varying 2-D forcing controls (`xx_atemp`/`xx_precip`/
  `xx_tauu`/`xx_tauv`), a real-SST `pkg/ecco` cost term, and `pkg/profiles`
  CTD-profile ingestion are wired up; `pkg/grdchk`'s gradient check passes
  cleanly at the seed cell/control/`grdchk_eps` combination documented in
  `input_ad/data.grdchk`'s own header comment, which also documents a
  per-control `grdchk_eps` reference table -- a single `grdchk_eps` value
  does NOT give a clean check for every control at the same cell, this is
  expected and documented, not a bug (see `input_ad/data.grdchk` for the
  full explanation if the gradient check looks wrong for a control/cell
  combination not already covered there). No `pkg/ctrl`-driven optimization
  (actually adjusting controls to minimize the cost function) has been
  attempted -- this is gradient-check-level adjoint verification only.
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
