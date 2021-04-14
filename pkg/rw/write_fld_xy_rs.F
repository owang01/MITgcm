#include "RW_OPTIONS.h"

      SUBROUTINE WRITE_FLD_XY_RS( pref,suff,field,myIter,myThid )
C WRITE_FLD_XY_RS is a "front-end" interface to the low-level I/O
C routines. It assumes single record files.
      IMPLICIT NONE
C Global
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
C Arguments
      CHARACTER*(*) pref,suff
      _RS field(1-Olx:sNx+Olx,1-Oly:sNy+Oly,nSx,nSy)
      INTEGER myIter
      INTEGER myThid
C Functions
      INTEGER ILNBLNK,IFNBLNK
C Common
      COMMON /RD_WR_FLD/ globalFile
      LOGICAL globalFile
C Local
      LOGICAL useCurrentDir
      _RL dummyRL(1)
      CHARACTER*(2) fType
      INTEGER nNz
      INTEGER iRec
      INTEGER s1Lo,s1Hi,s2Lo,s2Hi
      CHARACTER*(MAX_LEN_FNAM) fullName
#if PLEIADES_LUSTRE_OPT
      INTEGER il
      CHARACTER*(MAX_LEN_MBUF) msgBuf
      character*(80) fname
      character*(80) fnames(3)
#endif
C
C--   Build file name
C     Name has form 'prefix.suffix'
C     e.g. U.0000000100
      s1Lo = IFNBLNK(pref)
      s1Hi = ILNBLNK(pref)
      IF ( suff .EQ. ' ' ) THEN
       WRITE( fullName, '(A)' ) pref(s1Lo:s1Hi)
      ELSEIF ( suff .EQ. 'I10' ) THEN
       WRITE( fullName, '(A,A,I10.10)' ) pref(s1Lo:s1Hi),'.',myIter
      ELSE
       s2Lo = IFNBLNK(suff)
       s2Hi = ILNBLNK(suff)
       WRITE( fullName, '(A,A)' ) pref(s1Lo:s1Hi),suff(s2Lo:s2Hi)
      ENDIF

      useCurrentDir = .FALSE.
      fType='RS'
      nNz=1
      iRec=1
#ifdef ALLOW_MDSIO
#if PLEIADES_LUSTRE_OPT
      il = len_trim(fullName)
      WRITE(fname(1:80),'(A)') ' '
      WRITE(fname(1:il),'(A)') trim(fullName)
      call ctrl_set_fname( fname, fnames, myThid )
#ifdef DSK_DEBUG
      WRITE(msgBuf,'(A,I4,6A)')
     &'WRITE_FLD_XY_RS: ',iRec,' (', trim(fullName),')',
     & ' (', trim(fnames(1)),')'
      CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                        SQUEEZE_RIGHT, myThid )
#endif
      CALL MDS_WRITE_FIELD(
     I                      trim(fnames(1)), writeBinaryPrec,
     I                      globalFile, useCurrentDir,
     I                      fType, nNz, 1, nNz, dummyRL, field,
     I                      iRec, myIter, myThid )
#else
      CALL MDS_WRITE_FIELD(
     I                      fullName, writeBinaryPrec,
     I                      globalFile, useCurrentDir,
     I                      fType, nNz, 1, nNz, dummyRL, field,
     I                      iRec, myIter, myThid )
#endif
#endif
      RETURN
      END
