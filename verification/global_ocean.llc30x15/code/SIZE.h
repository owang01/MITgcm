CBOP
C    !ROUTINE: SIZE.h
C    !INTERFACE:
C    include SIZE.h
C    !DESCRIPTION: \bv
C     *==========================================================*
C     | SIZE.h Declare size of underlying computational grid.
C     *==========================================================*
C     | The design here supports a three-dimensional model grid
C     | with indices I,J and K. The three-dimensional domain
C     | is comprised of nPx*nSx blocks (or tiles) of size sNx
C     | along the first (left-most index) axis, nPy*nSy blocks
C     | of size sNy along the second axis and one block of size
C     | Nr along the vertical (third) axis.
C     | Blocks/tiles have overlap regions of size OLx and OLy
C     | along the dimensions that are subdivided.
C     *==========================================================*
C     \ev
C
C     Voodoo numbers controlling data layout:
C     sNx :: Number of X points in tile.
C     sNy :: Number of Y points in tile.
C     OLx :: Tile overlap extent in X.
C     OLy :: Tile overlap extent in Y.
C     nSx :: Number of tiles per process in X.
C     nSy :: Number of tiles per process in Y.
C     nPx :: Number of processes to use in X.
C     nPy :: Number of processes to use in Y.
C     Nx  :: Number of points in X for the full domain.
C     Ny  :: Number of points in Y for the full domain.
C     Nr  :: Number of points in vertical direction.
CEOP
      INTEGER sNx
      INTEGER sNy
      INTEGER OLx
      INTEGER OLy
      INTEGER nSx
      INTEGER nSy
      INTEGER nPx
      INTEGER nPy
      INTEGER Nx
      INTEGER Ny
      INTEGER Nr
C-- LLC30: a 1/3-linear-resolution coarsening of the production LLC90 grid
C   (see verification/global_ocean.llc30x15/preprocess/ and
C   docs/plans/20261103-llc30-lightweight-verification-config.md). LLC90
C   production itself tiles in 30x30 blocks (code12m_.../SIZE.h: sNx=sNy=30,
C   nPx=113), so LLC30's 5 facets, each exactly 1 or 3 tiles wide/tall at
C   this same 30x30 tile size, need no sub-30 tiling and no blankList:
C     facet 1 (30 x  90): 1 wide x 3 tall = 3 tiles
C     facet 2 (30 x  90): 1 wide x 3 tall = 3 tiles
C     facet 3 (30 x  30): 1 wide x 1 tall = 1 tile
C     facet 4 (90 x  30): 3 wide x 1 tall = 3 tiles
C     facet 5 (90 x  30): 3 wide x 1 tall = 3 tiles
C                                   total = 3+3+1+3+3 = 13 tiles
C   All 13 tiles are kept on a single process (nPx=nPy=1, nSx=13, nSy=1) --
C   single-core-friendly for a laptop smoke test; revisit nPx=13,nSx=1 only
C   if single-process memory becomes an issue.
      PARAMETER (
     &           sNx =  30,
     &           sNy =  30,
     &           OLx =   4,
     &           OLy =   4,
     &           nSx =  13,
     &           nSy =   1,
     &           nPx =   1,
     &           nPy =   1,
     &           Nx  = sNx*nSx*nPx,
     &           Ny  = sNy*nSy*nPy,
     &           Nr  =  15)

C     MAX_OLX :: Set to the maximum overlap region size of any array
C     MAX_OLY    that will be exchanged. Controls the sizing of exch
C                routine buffers.
      INTEGER MAX_OLX
      INTEGER MAX_OLY
      PARAMETER ( MAX_OLX = OLx,
     &            MAX_OLY = OLy )
