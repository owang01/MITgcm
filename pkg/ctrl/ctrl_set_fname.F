C
#include "CTRL_OPTIONS.h"

      subroutine ctrl_set_fname( xx_fname, fname, mythid )

c     ==================================================================
c     SUBROUTINE ctrl_set_fname
c     ==================================================================
c
c     o get filename for control variable and adjoint thereof
c     o initialise adjoint field
c
c     started: heimbach@mit.edu, 16-Aug-2001
c
c     ==================================================================

      implicit none

c     == global variables ==

#include "EEPARAMS.h"
#include "SIZE.h"
#include "PARAMS.h"
#include "GRID.h"

#include "ctrl.h"
#include "optim.h"

c     == routine arguments ==

      character(len=*) xx_fname
      character*( 80)   fname(3)
!      character(len=MAX_LEN_FNAM)   fname(3)
      integer mythid

c     == local variables ==

      integer ii
      integer il,ibin,imask,ir2,iopt
      CHARACTER*(MAX_LEN_MBUF) msgBuf
      CHARACTER(len=10) str

c     == external ==

      integer  ilnblnk
      external ilnblnk

c     == end of interface ==

c--   Only the master thread will do I/O.
      _BEGIN_MASTER( mythid )

      il=len_trim(xx_fname)

      write(fname(1)(1:80),'(80a)') ' '
      write(fname(2)(1:80),'(80a)') ' '
      write(fname(3)(1:80),'(80a)') ' '

#ifdef PLEIADES_LUSTRE_OPT3
      write(str,'(i10.10)') optimcycle
      iopt = (index(xx_fname,str) - 1 )
      ibin = (index(xx_fname,'.bin') - 1 )
      imask = (index(xx_fname,'mask') - 1 )
      ir2 = (index(xx_fname,'r2.') - 1 )

#ifdef DSK_DEBUG
        WRITE(msgBuf,'(A,A,4I4)') 'ctrl_set_fname: ',
     &     trim(xx_fname), len_trim(xx_fname), il, ibin, imask
      CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                        SQUEEZE_RIGHT, myThid )
#endif

! Don't add optimcycle to files that already have it or those matching mask* or *.bin or r2.*
! -------------------------------------------------------------------------------------------
      if ((ibin .GT. 0 .OR. imask .EQ. 0
     & .OR. ir2 .EQ. 0 .OR. iopt .GT. 0)
     &  .AND. il .LE. 80) then
        write(fname(1),'(2a)') trim(tmpDir), trim(xx_fname)
        write(fname(2),'(3a)') trim(tmpDir), yadprefix, trim(xx_fname)
        write(fname(3),'(3a)') trim(tmpDir), 'hn', trim(xx_fname)

      elseif ( il .GT. 0 .AND. il .LE. 80 ) then
         write(fname(1)(1:80),'(3a,i10.10)')
     &      trim(tmpDir), xx_fname(1:il) ,'.', optimcycle
         write(fname(2)(1:80),'(4a,i10.10)')
     &      trim(tmpDir), yadprefix, xx_fname(1:il), '.', optimcycle
         write(fname(3)(1:80),'(4a,i10.10)')
     &      trim(tmpDir), 'hn', xx_fname(1:il), '.', optimcycle
#else
      if ( il .GT. 0 .AND. il .LE. 80 ) then
         write(fname(1)(1:80),'(2a,i10.10)')
     &        xx_fname(1:il) ,'.', optimcycle
         write(fname(2)(1:80),'(3a,i10.10)')
     &        yadprefix, xx_fname(1:il), '.', optimcycle
         write(fname(3)(1:80),'(3a,i10.10)')
     &        'hn', xx_fname(1:il), '.', optimcycle
#endif
      else
        WRITE(msgBuf,'(A,I6)')
     & 'ctrl_set_fname: Input name out of range ',il
        CALL PRINT_MESSAGE( msgBuf, errorMessageUnit,
     &                        SQUEEZE_RIGHT, myThid )
         stop
      endif

#ifdef DSK_DEBUG
      WRITE(msgBuf,'(8A)')
     & 'ctrl_set_fname: ', trim(xx_fname),' ', trim(fname(1)), ' ',
     &                     trim(fname(2)), ' ', trim(fname(3))
      CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                        SQUEEZE_RIGHT, myThid )
#endif
      _END_MASTER( mythid )

      end
