c***********************************************************************
c     Routines to test struct_ls fortran interfaces
c***********************************************************************

c***********************************************************************
c     fhypre_structbicgstabcreate
c***********************************************************************
      subroutine fhypre_structbicgstabcreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_StructBiCGSTABCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabcreate: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structbicgstabdestroy
c***********************************************************************
      subroutine fhypre_structbicgstabdestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructBiCGSTABDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabdestroy: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structbicgstabsetup
c***********************************************************************
      subroutine fhypre_structbicgstabsetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructBiCGSTABSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabsetup: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structbicgstabsolve
c***********************************************************************
      subroutine fhypre_structbicgstabsolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructBiCGSTABSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabsolve: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structbicgstabsettol
c***********************************************************************
      subroutine fhypre_structbicgstabsettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      real ftol

      call HYPRE_StructBiCGSTABSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabsettol: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structbicgstabsetmaxiter
c***********************************************************************
      subroutine fhypre_structbicgstabsetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_StructBiCGSTABSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabsetmaxiter: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structbicgstabsetprecond
c***********************************************************************
      subroutine fhypre_structbicgstabsetprecond(fsolver, fprecond_id,
     1                                           fprecond_solver)
      integer ierr
      integer*8 fsolver
      integer*8 fprecond_id
      integer*8 fprecond_solver

      call HYPRE_StructBiCGSTABSetPrecond(fsolver, fprecond_id,
     1                                    fprecond_solver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabsetprecond: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structbicgstabsetlogging
c***********************************************************************
      subroutine fhypre_structbicgstabsetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call HYPRE_StructBiCGSTABSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabsetlogging: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structbicgstabsetprintlevel
c***********************************************************************
      subroutine fhypre_structbicgstabsetprintle(fsolver, fprintlev)
      integer ierr
      integer fprintlev
      integer*8 fsolver

      call HYPRE_StructBiCGSTABSetPrintLev(fsolver, fprintlev, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabsetprintle: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structbicgstabgetnumiterations
c***********************************************************************
      subroutine fhypre_structbicgstabgetnumiter(fsolver, fnumiter)
      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_StructBiCGSTABGetNumItera(fsolver, fnumiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabgetnumiter: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structbicgstabgetresidual
c***********************************************************************
      subroutine fhypre_structbicgstabgetresidua(fsolver, fresidual)
      integer ierr
      integer*8 fsolver
      real fresidual

      call HYPRE_StructBiCGSTABGetResidual(fsolver, fresidual, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabgetresidua: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structbicgstabgetfinalrelativeresidualnorm
c***********************************************************************
      subroutine fhypre_structbicgstabgetfinalre(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      real fnorm

      call HYPRE_StructBiCGSTABGetFinalRel(fsolver, fnorm)
      if (ierr .ne. 0) then
         print *, 'fhypre_structbicgstabgetfinalre: err = ', ierr
      endif

      return
      end



c***********************************************************************
c     fhypre_structgmrescreate
c***********************************************************************
      subroutine fhypre_structgmrescreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_StructGMRESCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmrescreate: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structgmresdestroy
c***********************************************************************
      subroutine fhypre_structgmresdestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructGMRESDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmresdestroy: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structgmressetup
c***********************************************************************
      subroutine fhypre_structgmressetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructGMRESSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmressetup: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structgmressolve
c***********************************************************************
      subroutine fhypre_structgmressolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructGMRESSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmressolve: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structgmressettol
c***********************************************************************
      subroutine fhypre_structgmressettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      real ftol

      call HYPRE_StructGMRESSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmressettol: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structgmressetmaxiter
c***********************************************************************
      subroutine fhypre_structgmressetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_StructGMRESSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmressetmaxiter: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structgmressetprecond
c***********************************************************************
      subroutine fhypre_structgmressetprecond(fsolver, fprecond_id,
     1                                        fprecond_solver)
      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond_solver

      call HYPRE_StructGMRESSetPrecond(fsolver, fprecond_id,
     1                                 fprecond_solver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmressetprecond: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structgmressetlogging
c***********************************************************************
      subroutine fhypre_structgmressetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call HYPRE_StructGMRESSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmressetlogging: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structgmressetprintlevel
c***********************************************************************
      subroutine fhypre_structgmressetprintlevel(fsolver, fprintlevel)
      integer ierr
      integer fprintlevel
      integer*8 fsolver

      call HYPRE_StructGMRESSetPrintLevel(fsolver, fprint_level, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmressetprintlevel: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structgmresgetnumiterations
c***********************************************************************
      subroutine fhypre_structgmresgetnumiterati(fsolver, fnumiters)
      integer ierr
      integer fnumiters
      integer*8 fsolver

      call HYPRE_StructGMRESGetNumIteratio(fsolver, fnumiters, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmresgetnumiterati: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structgmresgetfinalrelativeresidualnorm
c***********************************************************************
      subroutine fhypre_structgmresgetfinalrelat(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      real fnorm

      call HYPRE_StructGMRESGetFinalRelati(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structgmresgetfinalrelat: err = ', ierr
      endif

      return
      end



c***********************************************************************
c     fhypre_structhybridcreate
c***********************************************************************
      subroutine fhypre_structhybridcreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_StructHybridCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridcreate: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybriddestroy
c***********************************************************************
      subroutine fhypre_structhybriddestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructHybridDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybriddestroy: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridsetup
c***********************************************************************
      subroutine fhypre_structhybridsetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructHybridSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetup: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridsolve
c***********************************************************************
      subroutine fhypre_structhybridsolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructHybridSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsolve: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridsetsolvertype
c***********************************************************************
      subroutine fhypre_structhybridsetsolvertyp(fsolver, fsolver_typ)
      integer ierr
      integer fsolver_typ
      integer*8 fsolver

      call HYPRE_StructHybridSetSolverType(fsolver, fsolver_typ, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetsolvertyp: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridsetstopcrit
c***********************************************************************
      subroutine fhypre_structhybridsetstopcrit(fsolver, fstop_crit)
      integer ierr
      integer fstop_crit
      integer*8 fsolver

      call HYPRE_StructHybridSetStopCrit(fsolver, fstop_crit, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetstopcrit: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridsetkdim
c***********************************************************************
      subroutine fhypre_structhybridsetkdim(fsolver, fkdim)
      integer ierr
      integer fkdim
      integer*8 fsolver

      call HYPRE_StructHybridSetKDim(fsolver, fkdim, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetkdim: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridsettol
c***********************************************************************
      subroutine fhypre_structhybridsettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      real ftol

      call HYPRE_StructHybridSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsettol: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridsetconvergencetol
c***********************************************************************
      subroutine fhypre_structhybridsetconvergen(fsolver, fcftol)
      integer ierr
      integer*8 fsolver
      real fcftol

      call HYPRE_StructHybridSetConvergenc(fsolver, fcftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetconvergen: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridsetpcgabsolutetolfactor
c***********************************************************************
      subroutine fhypre_structhybridsetpcgabsolu(fsolver, fpcgtol)
      integer ierr
      integer*8 fsolver
      real fpcgtol

      call HYPRE_StructHybridSetPCGAbsolut(fsolver, fpcgtol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetpcgabsolu: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridsetdscgmaxiter
c***********************************************************************
      subroutine fhypre_structhybridsetdscgmaxit(fsolver, fdscgmaxitr)
      integer ierr
      integer fdscgmaxitr
      integer*8 fsolver

      call HYPRE_StructHybridSetDSCGMaxIte(fsolver, fdscgmaxitr, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetdscgmaxit: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridsetpcgmaxiter
c***********************************************************************
      subroutine fhypre_structhybridsetpcgmaxite(fsolver, fpcgmaxitr)
      integer ierr
      integer fpcgmaxitr
      integer*8 fsolver

      call HYPRE_StructHybridSetPCGMaxIter(fsolver, fpcgmaxitr, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetpcgmaxite: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridsettwonorm
c***********************************************************************
      subroutine fhypre_structhybridsettwonorm(fsolver, ftwonorm)
      integer ierr
      integer ftwonorm
      integer*8 fsolver

      call HYPRE_StructHybridSetTwoNorm(fsolver, ftwonorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsettwonorm: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridsetrelchange
c***********************************************************************
      subroutine fhypre_structhybridsetrelchange(fsolver, frelchng)
      integer ierr
      integer frelchng
      integer*8 fsolver

      call HYPRE_StructHybridSetRelChange(fsolver, frelchng, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetrelchange: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridsetprecond
c***********************************************************************
      subroutine fhypre_structhybridsetprecond(fsolver, fprecond_id,
     1                                         fprecond)
      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_StructHybridSetPrecond(fsolver, fprecond_id, fprecond,
     1                                  ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetprecond: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridsetlogging
c***********************************************************************
      subroutine fhypre_structhybridsetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call HYPRE_StructHybridSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetlogging: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridsetprintlevel
c***********************************************************************
      subroutine fhypre_structhybridsetprintleve(fsolver, fprntlvl)
      integer ierr
      integer fprntlvl
      integer*8 fsolver

      call HYPRE_StructHybridSetPrintLevel(fsolver, fprntlvl, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridsetprintleve: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridgetnumiterations
c***********************************************************************
      subroutine fhypre_structhybridgetnumiterat(fsolver, fnumits)
      integer ierr
      integer fnumits
      integer*8 fsolver

      call HYPRE_StructHybridGetNumIterati(fsolver, fnumits, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridgetnumiterat: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridgetdscgnumiterations
c***********************************************************************
      subroutine fhypre_structhybridgetdscgnumit(fsolver, fdscgnumits)
      integer ierr
      integer fdscgnumits
      integer*8 fsolver

      call HYPRE_StructHybridGetDSCGNumIte(fsolver, fdscgnumits, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridgetdscgnumit: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridgetpcgnumiterations
c***********************************************************************
      subroutine fhypre_structhybridgetpcgnumite(fsolver, fpcgnumits)
      integer ierr
      integer fpcgnumits
      integer*8 fsolver

      call HYPRE_StructHybridGetPCGNumIter(fsolver, fpcgnumits, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridgetpcgnumite: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structhybridgetfinalrelativeresidualnorm
c***********************************************************************
      subroutine fhypre_structhybridgetfinalrela(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      real fnorm

      call HYPRE_StructHybridGetFinalRelat(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structhybridgetfinalrela: err = ', ierr
      endif

      return
      end



c***********************************************************************
c     fhypre_structjacobicreate
c***********************************************************************
      subroutine fhypre_structjacobicreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_StructJacobiCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobicreate: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structjacobidestroy
c***********************************************************************
      subroutine fhypre_structjacobidestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructJacobiDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobidestroy: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structjacobisetup
c***********************************************************************
      subroutine fhypre_structjacobisetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructJacobiSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobisetup: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structjacobisolve
c***********************************************************************
      subroutine fhypre_structjacobisolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructJacobiSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobisolve: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structjacobisettol
c***********************************************************************
      subroutine fhypre_structjacobisettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      real ftol

      call HYPRE_StructJacobiSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobisettol: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structjacobisetmaxiter
c***********************************************************************
      subroutine fhypre_structjacobisetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_StructJacobiSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobisetmaxiter: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structjacobisetzeroguess
c***********************************************************************
      subroutine fhypre_structjacobisetzeroguess(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructJacobiSetZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobisetzeroguess: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structjacobisetnonzeroguess
c***********************************************************************
      subroutine fhypre_structjacobisetnonzerogu(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructJacobiSetNonZeroGue(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobisetnonzerogu: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structjacobigetnumiterations
c***********************************************************************
      subroutine fhypre_structjacobigetnumiterat(fsolver, fnumiters)
      integer ierr
      integer fnumiters
      integer*8 fsolver

      call HYPRE_StructJacobiGetNumIterati(fsolver, fnumiters, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobigetnumiterat: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structjacobigetfinalrelativeresidualnorm
c***********************************************************************
      subroutine fhypre_structjacobigetfinalrela(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      real fnorm

      call HYPRE_StructJacobiGetFinalRelat(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobigetfinalrela: err = ', ierr
      endif

      return
      end



c***********************************************************************
c     fhypre_structpcgcreate
c***********************************************************************
      subroutine fhypre_structpcgcreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_StructPCGCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgcreate: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpcgdestroy
c***********************************************************************
      subroutine fhypre_structpcgdestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructPCGDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgdestroy: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpcgsetup
c***********************************************************************
      subroutine fhypre_structpcgsetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructPCGSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgsetup: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpcgsolve
c***********************************************************************
      subroutine fhypre_structpcgsolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructPCGSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgsolve: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpcgsettol
c***********************************************************************
      subroutine fhypre_structpcgsettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      real ftol

      call HYPRE_StructPCGSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgsettol: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpcgsetmaxiter
c***********************************************************************
      subroutine fhypre_structpcgsetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_StructPCGSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgsetmaxiter: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpcgsettwonorm
c***********************************************************************
      subroutine fhypre_structpcgsettwonorm(fsolver, ftwonorm)
      integer ierr
      integer ftwonorm
      integer*8 fsolver

      call HYPRE_StructPCGSetTwoNorm(fsolver, ftwonorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgsettwonorm: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpcgsetrelchange
c***********************************************************************
      subroutine fhypre_structpcgsetrelchange(fsolver, frelchng)
      integer ierr
      integer frelchng
      integer*8 fsolver

      call HYPRE_StructPCGSetRelChange(fsolver, frelchng, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgsetrelchange: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpcgsetprecond
c***********************************************************************
      subroutine fhypre_structpcgsetprecond(fsolver, fprecond_id, 
     1                                      fprecond)
      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_StructPCGSetPrecond(fsolver, fprecond_id, fprecond,
     1                               ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgsetprecond: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpcgsetlogging
c***********************************************************************
      subroutine fhypre_structpcgsetlogging(fsolver, flogging) 
      integer ierr
      integer flogging
      integer*8 fsolver

      call HYPRE_StructPCGSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgsetlogging: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpcgsetprintlevel
c***********************************************************************
      subroutine fhypre_structpcgsetprintlevel(fsolver, fprntlvl) 
      integer ierr
      integer fprntlvl
      integer*8 fsolver

      call HYPRE_StructPCGSetPrintLevel(fsolver, fprntlvl, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcgsetprintlevel: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpcggetnumiterations
c***********************************************************************
      subroutine fhypre_structpcggetnumiteration(fsolver, fnumiters)
      integer ierr
      integer fnumiters
      integer*8 fsolver

      call HYPRE_StructPCGGetNumIterations(fsolver, fnumiters, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpcggetnumiteration: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpcggetfinalrelativeresidualnorm
c***********************************************************************
      subroutine fhypre_structpcggetfinalrelativ(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      real fnorm

      call HYPRE_StructPCGGetFinalRelative(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structjacobigetfinalrelativ: err = ', ierr
      endif

      return
      end



c***********************************************************************
c     fhypre_structdiagscalesetup
c***********************************************************************
      subroutine fhypre_structdiagscalesetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructDiagScaleSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structdiagscalesetup: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structdiagscalesolve
c***********************************************************************
      subroutine fhypre_structdiagscalesolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructDiagScale(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structdiagscalesolve: err = ', ierr
      endif

      return
      end



c***********************************************************************
c     fhypre_structpfmgcreate
c***********************************************************************
      subroutine fhypre_structpfmgcreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_StructPFMGCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgcreate: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpfmgdestroy
c***********************************************************************
      subroutine fhypre_structpfmgdestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructPFMGDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgdestroy: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpfmgsetup
c***********************************************************************
      subroutine fhypre_structpfmgsetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructPFMGSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetup: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpfmgsolve
c***********************************************************************
      subroutine fhypre_structpfmgsolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructPFMGSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsolve: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpfmgsettol
c***********************************************************************
      subroutine fhypre_structpfmgsettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      real ftol

      call HYPRE_StructPFMGSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsettol: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpfmgsetmaxiter
c***********************************************************************
      subroutine fhypre_structpfmgsetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_StructPFMGSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetmaxiter: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpfmgsetrelchange
c***********************************************************************
      subroutine fhypre_structpfmgsetrelchange(fsolver, frelchange)
      integer ierr
      integer frelchange
      integer*8 fsolver

      call HYPRE_StructPFMGSetRelChange(fsolver, frelchange, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetrelchange: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpfmgsetzeroguess
c***********************************************************************
      subroutine fhypre_structpfmgsetzeroguess(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructPFMGSetZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetzeroguess: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpfmgsetnonzeroguess
c***********************************************************************
      subroutine fhypre_structpfmgsetnonzerogues(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructPFMGSetNonZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetnonzerogues: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpfmggetnumiterations
c***********************************************************************
      subroutine fhypre_structpfmggetnumiteratio(fsolver, fnumiters)
      integer ierr
      integer fnumiters
      integer*8 fsolver

      call HYPRE_StructPFMGGetNumIteration(fsolver, fnumiters, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmggetnumiteratio: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpfmggetfinalrelativeresidualnorm
c***********************************************************************
      subroutine fhypre_structpfmggetfinalrelati(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      real fnorm

      call HYPRE_StructPFMGGetFinalRelativ(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmggetfinalrelati: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpfmgsetskiprelax
c***********************************************************************
      subroutine fhypre_structpfmgsetskiprelax(fsolver, fskiprelax)
      integer ierr
      integer fskiprelax
      integer*8 fsolver

      call HYPRE_StructPFMGSetSkipRelax(fsolver, fskiprelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetskiprelax: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpfmgsetrelaxtype
c***********************************************************************
      subroutine fhypre_structpfmgsetrelaxtype(fsolver, frelaxtype)
      integer ierr
      integer frelaxtype
      integer*8 fsolver

      call HYPRE_StructPFMGSetRelaxType(fsolver, frelaxtype, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetrelaxtype: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpfmgsetraptype
c***********************************************************************
      subroutine fhypre_structpfmgsetraptype(fsolver, fraptype)
      integer ierr
      integer fraptype
      integer*8 fsolver

      call HYPRE_StructPFMGSetRAPType(fsolver, fraptype, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetraptype: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpfmgsetnumprerelax
c***********************************************************************
      subroutine fhypre_structpfmgsetnumprerelax(fsolver,
     1                                             fnumprerelax)
      integer ierr
      integer fnumprerelax
      integer*8 fsolver

      call HYPRE_StructPFMGSetNumPreRelax(fsolver, fnumprerelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetnumprerelax: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpfmgsetnumpostrelax
c***********************************************************************
      subroutine fhypre_structpfmgsetnumpostrela(fsolver,
     1                                             fnumpostrelax)
      integer ierr
      integer fnumpostrelax
      integer*8 fsolver

      call HYPRE_StructPFMGSetNumPostRelax(fsolver, fnumpostrelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetnumpostrela: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpfmgsetdxyz
c***********************************************************************
      subroutine fhypre_structpfmgsetdxyz(fsolver, fdxyz)
      integer ierr
      integer*8 fsolver
      real fdxyz

      call HYPRE_StructPFMGSetDxyz(fsolver, fdxyz, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetdxyz: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpfmgsetlogging
c***********************************************************************
      subroutine fhypre_structpfmgsetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call HYPRE_StructPFMGSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetlogging: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structpfmgsetprintlevel
c***********************************************************************
      subroutine fhypre_structpfmgsetprintlevel(fsolver, fprintlevel)
      integer ierr
      integer fprintlevel
      integer*8 fsolver

      call HYPRE_StructPFMGSetPrintLevel(fsolver, fprintlevel, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structpfmgsetprintlevel: err = ', ierr
      endif

      return
      end



c***********************************************************************
c     fhypre_structsmgcreate
c***********************************************************************
      subroutine fhypre_structsmgcreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_StructSMGCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgcreate: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsmgdestroy
c***********************************************************************
      subroutine fhypre_structsmgdestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructSMGDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgdestroy: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsmgsetup
c***********************************************************************
      subroutine fhypre_structsmgsetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructSMGSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetup: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsmgsolve
c***********************************************************************
      subroutine fhypre_structsmgsolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructSMGSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsolve: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsmgsetmemoryuse
c***********************************************************************
      subroutine fhypre_structsmgsetmemoryuse(fsolver, fmemuse)
      integer ierr
      integer fmemuse
      integer*8 fsolver

      call HYPRE_StructSMGSetMemoryUse(fsolver, fmemuse, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetmemoryuse: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsmgsettol
c***********************************************************************
      subroutine fhypre_structsmgsettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      real ftol

      call HYPRE_StructSMGSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsettol: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsmgsetmaxiter
c***********************************************************************
      subroutine fhypre_structsmgsetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_StructSMGSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetmaxiter: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsmgsetrelchange
c***********************************************************************
      subroutine fhypre_structsmgsetrelchange(fsolver, frelchange)
      integer ierr
      integer frelchange
      integer*8 fsolver

      call HYPRE_StructSMGSetRelChange(fsolver, frelchange, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetrelchange: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsmgsetzeroguess
c***********************************************************************
      subroutine fhypre_structsmgsetzeroguess(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructSMGSetZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetzeroguess: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsmgsetnonzeroguess
c***********************************************************************
      subroutine fhypre_structsmgsetnonzeroguess(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructSMGSetNonZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetnonzeroguess: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsmggetnumiterations
c***********************************************************************
      subroutine fhypre_structsmggetnumiteration(fsolver, fnumiters)
      integer ierr
      integer fnumiters
      integer*8 fsolver

      call HYPRE_StructSMGGetNumIterations(fsolver, fnumiters, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmggetnumiteration: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsmggetfinalrelativeresidualnorm
c***********************************************************************
      subroutine fhypre_structsmggetfinalrelativ(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      real fnorm

      call HYPRE_StructSMGGetFinalRelative(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmggetfinalrelativ: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsmgsetnumprerelax
c***********************************************************************
      subroutine fhypre_structsmgsetnumprerelax(fsolver, fnumprerelax)
      integer ierr
      integer fnumprerelax
      integer*8 fsolver

      call HYPRE_StructSMGSetNumPreRelax(fsolver, fnumprerelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetnumprerelax: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsmgsetnumpostrelax
c***********************************************************************
      subroutine fhypre_structsmgsetnumpostrelax(fsolver, fnumpstrlx)
      integer ierr
      integer fnumpstrlx
      integer*8 fsolver

      call HYPRE_StructSMGSetNumPostRelax(fsolver, fnumpstrlx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetnumpostrelax: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsmgsetlogging
c***********************************************************************
      subroutine fhypre_structsmgsetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call HYPRE_StructSMGSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetlogging: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsmgsetprintlevel
c***********************************************************************
      subroutine fhypre_structsmgsetprintlevel(fsolver, fprintlevel)
      integer ierr
      integer fprintlevel
      integer*8 fsolver

      call HYPRE_StructSMGSetPrintLevel(fsolver, fprintlevel, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsmgsetprintlevel: err = ', ierr
      endif

      return
      end



c***********************************************************************
c     fhypre_structsparsemsgcreate
c***********************************************************************
      subroutine fhypre_structsparsemsgcreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_StructSparseMSGCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgcreate: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsparsemsgdestroy
c***********************************************************************
      subroutine fhypre_structsparsemsgdestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructSparseMSGDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgdestroy: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsparsemsgsetup
c***********************************************************************
      subroutine fhypre_structsparsemsgsetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructSparseMSGSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetup: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsparsemsgsolve
c***********************************************************************
      subroutine fhypre_structsparsemsgsolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_StructSparseMSGSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsolve: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsparsemsgsetjump
c***********************************************************************
      subroutine fhypre_structsparsemsgsetjump(fsolver, fjump)
      integer ierr
      integer fjump
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetJump(fsolver, fjump, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetjump: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsparsemsgsettol
c***********************************************************************
      subroutine fhypre_structsparsemsgsettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      real ftol

      call HYPRE_StructSparseMSGSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsettol: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsparsemsgsetmaxiter
c***********************************************************************
      subroutine fhypre_structsparsemsgsetmaxite(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetmaxite: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsparsemsgsetrelchange
c***********************************************************************
      subroutine fhypre_structsparsemsgsetrelcha(fsolver, frelchange)
      integer ierr
      integer frelchange
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetRelChan(fsolver, frelchange, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetrelcha: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsparsemsgsetzeroguess
c***********************************************************************
      subroutine fhypre_structsparsemsgsetzerogu(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetZeroGue(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetzerogu: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsparsemsgsetnonzeroguess
c***********************************************************************
      subroutine fhypre_structsparsemsgsetnonzer(fsolver)
      integer ierr
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetNonZero(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetnonzer: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsparsemsggetnumiterations
c***********************************************************************
      subroutine fhypre_structsparsemsggetnumite(fsolver, fniters)
      integer ierr
      integer fniters
      integer*8 fsolver

      call HYPRE_StructSparseMSGGetNumIter(fsolver, fniters, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsggetnumite: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsparsemsggetfinalrelativeresidualnorm
c***********************************************************************
      subroutine fhypre_structsparsemsggetfinalr(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      real fnorm

      call HYPRE_StructSparseMSGGetFinalRe(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsggetfinalr: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsparsemsgsetrelaxtype
c***********************************************************************
      subroutine fhypre_structsparsemsgsetrelaxt(fsolver, frelaxtype)
      integer ierr
      integer frelaxtype
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetRelaxTy(fsolver, frelaxtype, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetrelaxt: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsparsemsgsetnumprerelax
c***********************************************************************
      subroutine fhypre_structsparsemsgsetnumpre(fsolver, fnprelax)
      integer ierr
      integer fnprelax
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetNumPreR(fsolver, fnprelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetnumpre: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsparsemsgsetnumpostrelax
c***********************************************************************
      subroutine fhypre_structsparsemsgsetnumpos(fsolver, fnpstrlx)
      integer ierr
      integer fnpstrlx
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetNumPost(fsolver, fnpstrlx, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetnumpos: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsparsemsgsetnumfinerelax
c***********************************************************************
      subroutine fhypre_structsparsemsgsetnumfin(fsolver, fnfine)
      integer ierr
      integer fnfine
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetNumFine(fsolver, fnfine, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetnumfin: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsparsemsgsetlogging
c***********************************************************************
      subroutine fhypre_structsparsemsgsetloggin(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetloggin: err = ', ierr
      endif

      return
      end

c***********************************************************************
c     fhypre_structsparsemsgsetprintlevel
c***********************************************************************
      subroutine fhypre_structsparsemsgsetprintl(fsolver, fprntlvl)
      integer ierr
      integer fprntlvl
      integer*8 fsolver

      call HYPRE_StructSparseMSGSetPrintLe(fsolver, fprntlvl, ierr)
      if (ierr .ne. 0) then
         print *, 'fhypre_structsparsemsgsetprintl: err = ', ierr
      endif

      return
      end
