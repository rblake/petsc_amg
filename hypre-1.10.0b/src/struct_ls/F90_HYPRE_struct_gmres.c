/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.1 $
 *********************************************************************EHEADER*/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgmrescreate, HYPRE_STRUCTGMRESCREATE)( int      *comm,
                                            long int *solver,
                                            int      *ierr   )

{
   *ierr = (int)
      ( HYPRE_StructGMRESCreate( (MPI_Comm)             *comm,
                               (HYPRE_StructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structgmresdestroy, HYPRE_STRUCTGMRESDESTROY)( long int *solver,
                                          int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructGMRESDestroy( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structgmressetup, HYPRE_STRUCTGMRESSETUP)( long int *solver,
                                       long int *A,
                                       long int *b,
                                       long int *x,
                                       int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructGMRESSetup( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structgmressolve, HYPRE_STRUCTGMRESSOLVE)( long int *solver,
                                       long int *A,
                                       long int *b,
                                       long int *x,
                                       int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructGMRESSolve( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgmressettol, HYPRE_STRUCTGMRESSETTOL)( long int *solver,
                                        double   *tol,
                                        int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructGMRESSetTol( (HYPRE_StructSolver) *solver,
                                          (double)             *tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgmressetmaxiter, HYPRE_STRUCTGMRESSETMAXITER)( long int *solver,
                                            int      *max_iter,
                                            int      *ierr     )
{
   *ierr = (int)
      ( HYPRE_StructGMRESSetMaxIter( (HYPRE_StructSolver) *solver,
                                   (int)                *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgmressetprecond, HYPRE_STRUCTGMRESSETPRECOND)( long int *solver,
                                            int      *precond_id,
                                            long int *precond_solver,
                                            int      *ierr           )
{

   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 0 - setup a smg preconditioner
    * 1 - setup a pfmg preconditioner
    * 8 - setup a ds preconditioner
    * 9 - dont setup a preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = (int)
         ( HYPRE_StructGMRESSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructSMGSolve,
                                      HYPRE_StructSMGSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else if (*precond_id == 1)
   {
      *ierr = (int)
         ( HYPRE_StructGMRESSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructPFMGSolve,
                                      HYPRE_StructPFMGSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else if (*precond_id == 5)
   {
      *ierr = (int)
         ( HYPRE_StructGMRESSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructSparseMSGSolve,
                                      HYPRE_StructSparseMSGSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else if (*precond_id == 6)
   {
      *ierr = (int)
         ( HYPRE_StructGMRESSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructJacobiSolve,
                                      HYPRE_StructJacobiSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else if (*precond_id == 8)
   {
      *ierr = (int)
         ( HYPRE_StructGMRESSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructDiagScale,
                                      HYPRE_StructDiagScaleSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else if (*precond_id == 9)
   {
      *ierr = 0;
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgmressetlogging, HYPRE_STRUCTGMRESSETLOGGING)( long int *solver,
                                            int      *logging,
                                            int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructGMRESSetLogging( (HYPRE_StructSolver) *solver,
                                   (int)                *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgmressetprintlevel, HYPRE_STRUCTGMRESSETPRINTLEVEL)( long int *solver,
                                            int      *print_level,
                                            int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructGMRESSetPrintLevel( (HYPRE_StructSolver) *solver,
                                   (int)                *print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgmresgetnumiteratio, HYPRE_STRUCTGMRESGETNUMITERATIO)( long int *solver,
                                                  int      *num_iterations,
                                                  int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructGMRESGetNumIterations(
         (HYPRE_StructSolver) *solver,
         (int *)              num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgmresgetfinalrelati, HYPRE_STRUCTGMRESGETFINALRELATI)( long int *solver,
                                                  double   *norm,
                                                  int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructGMRESGetFinalRelativeResidualNorm(
         (HYPRE_StructSolver) *solver,
         (double *)           norm ) );
}
