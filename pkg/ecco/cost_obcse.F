#include "ECCO_OPTIONS.h"
#ifdef ALLOW_CTRL
# include "CTRL_OPTIONS.h"
#endif

CBOP
C     !ROUTINE: COST_OBCSE
C     !INTERFACE:
      subroutine cost_obcse(
     I                       startrec, endrec,
     I                       myTime, myIter, myThid )

C     !DESCRIPTION: \bv
c     ==================================================================
c     SUBROUTINE cost_obcse
c     ==================================================================
c
c     o cost function contribution obc
c
c     ==================================================================
c     SUBROUTINE cost_obcse
c     ==================================================================
C     \ev

C     !USES:

      implicit none

c     == global variables ==
#include "EEPARAMS.h"
#include "SIZE.h"
#include "PARAMS.h"
#include "GRID.h"
c#ifdef ALLOW_OBCS
c# include "OBCS_GRID.h"
c#endif

#ifdef ALLOW_CAL
# include "cal.h"
#endif
#ifdef ALLOW_CTRL
# include "CTRL_SIZE.h"
# include "CTRL.h"
# include "CTRL_DUMMY.h"
# include "OPTIMCYCLE.h"
# include "CTRL_OBCS.h"
#endif

C     !INPUT/OUTPUT PARAMETERS:
      integer startrec, endrec
      _RL     myTime
      integer myIter
      integer myThid

#if (defined (ALLOW_CTRL) && defined (ALLOW_OBCS))

#ifdef ALLOW_OBCSE_COST_CONTRIBUTION
C     !FUNCTIONS:
      integer  ilnblnk
      external ilnblnk

C     !LOCAL VARIABLES:
      integer bi,bj
      integer j,k
      integer jmin,jmax
      integer irec
      integer iobcs
      integer nrec
      integer ilfld
      integer igg
      _RL fctile
      _RL fcthread
      _RL dummy
      _RL gg
      _RL tmpx
cgg(
      _RL tmpfield (1-OLy:sNy+OLy,Nr,nSx,nSy)
      _RL maskyz   (1-OLy:sNy+OLy,Nr,nSx,nSy)
      character*(MAX_LEN_FNAM) fnamefld
      logical doglobalread
      logical ladinit
#ifdef ECCO_VERBOSE
      character*(MAX_LEN_MBUF) msgbuf
#endif
CEOP

      jmin = 1
      jmax = sNy

c--   Read tiled data.
      doglobalread = .false.
      ladinit      = .false.

c     Number of records to be used.
      nrec = endrec-startrec+1

c     ip1 = 1
      fcthread = 0. _d 0

#ifdef ECCO_VERBOSE
      _BEGIN_MASTER( myThid )
      write(msgbuf,'(a)') ' '
      call print_message( msgbuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, myThid )
      write(msgbuf,'(a)') ' '
      call print_message( msgbuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, myThid )
      write(msgbuf,'(a,i9.8)')
     &  ' cost_obcse: number of records to process: ',nrec
      call print_message( msgbuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, myThid )
      write(msgbuf,'(a)') ' '
      call print_message( msgbuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, myThid )
      _END_MASTER( myThid )
#endif

      if (optimcycle .ge. 0) then
        ilfld=ilnblnk( xx_obcse_file )
        write(fnamefld,'(2a,i10.10)')
     &       xx_obcse_file(1:ilfld), '.', optimcycle
      endif

c--   Loop over records.
      do irec = 1,nrec

#ifdef ALLOW_AUTODIFF
        call active_read_yz( fnamefld, tmpfield, irec,
     &                       doglobalread, ladinit, optimcycle,
     &                       myThid, xx_obcse_dummy )
#else
        CALL READ_REC_YZ_RL( fnamefld, ctrlprec, Nr,
     &                       tmpfield, irec, 1, myThid )
#endif

cgg     Need to solve for iobcs would have been.
        gg    = (irec-1)/nobcs
        igg   = int(gg)
        iobcs = irec - igg*nobcs

#ifdef ALLOW_AUTODIFF
        call active_read_yz( 'maskobcse', maskyz, iobcs,
     &                       doglobalread, ladinit, 0,
     &                       myThid, dummy )
#else
        CALL READ_REC_YZ_RL( 'maskobcse', ctrlprec, Nr,
     &                       maskyz, iobcs, 1, myThid )
#endif

c--     Loop over this thread s tiles.
        do bj = myByLo(myThid), myByHi(myThid)
          do bi = myBxLo(myThid), myBxHi(myThid)

c--         Determine the weights to be used.
            fctile = 0. _d 0

            do k = 1, Nr
              do j = jmin,jmax
c               i = OB_Iw(j,bi,bj)
cgg                if (maskW(i+ip1,j,k,bi,bj) .ne. 0.) then
                  tmpx = tmpfield(j,k,bi,bj)
CMM                  fctile = fctile + wobcse2(j,k,bi,bj,iobcs)
                  fctile = fctile + wobcse(k,iobcs)
     &                        *tmpx*tmpx*maskyz(j,k,bi,bj)
cgg                endif
CMM                  if (wobcsw2(j,k,bi,bj,iobcs)*maskyz(j,k,bi,bj).ne.0.)
                  if (wobcse(k,iobcs)*maskyz(j,k,bi,bj).ne.0.)
     &                    num_obcse(bi,bj) = num_obcse(bi,bj) + 1. _d 0
              enddo
            enddo

            objf_obcse(bi,bj) = objf_obcse(bi,bj) + fctile
            fcthread         = fcthread + fctile
          enddo
        enddo

#ifdef ECCO_VERBOSE
c--     Print cost function for all tiles.
        _GLOBAL_SUM_RL( fcthread , myThid )
        write(msgbuf,'(a)') ' '
        call print_message( msgbuf, standardMessageUnit,
     &                      SQUEEZE_RIGHT, myThid )
        write(msgbuf,'(a,i8.8)')
     &    ' cost_obcse: irec = ',irec
        call print_message( msgbuf, standardMessageUnit,
     &                      SQUEEZE_RIGHT, myThid )
        write(msgbuf,'(a,a,d22.15)')
     &    ' global cost function value',
     &    ' (obcse) = ',fcthread
        call print_message( msgbuf, standardMessageUnit,
     &                      SQUEEZE_RIGHT, myThid )
        write(msgbuf,'(a)') ' '
        call print_message( msgbuf, standardMessageUnit,
     &                      SQUEEZE_RIGHT, myThid )
#endif

      enddo
c--   End of loop over records.

#endif /* ALLOW_OBCSE_COST_CONTRIBUTION */

#endif /* ALLOW_CTRL and ALLOW_OBCS */

      return
      end
