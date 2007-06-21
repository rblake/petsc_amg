/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_ParCSRGMRES Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmrescreate, HYPRE_PARCSRGMRESCREATE)( int      *comm,
                                          long int *solver,
                                          int      *ierr    )

{
   *ierr = (int) ( HYPRE_ParCSRGMRESCreate( (MPI_Comm)      *comm,
                                                (HYPRE_Solver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrgmresdestroy, HYPRE_PARCSRGMRESDESTROY)( long int *solver,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrgmressetup, HYPRE_PARCSRGMRESSETUP)( long int *solver,
                                         long int *A,
                                         long int *b,
                                         long int *x,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESSetup( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrgmressolve, HYPRE_PARCSRGMRESSOLVE)( long int *solver,
                                         long int *A,
                                         long int *b,
                                         long int *x,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESSolve( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetkdim, HYPRE_PARCSRGMRESSETKDIM)( long int *solver,
                                           int      *kdim,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESSetKDim( (HYPRE_Solver) *solver,
                                             (int)          *kdim    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressettol, HYPRE_PARCSRGMRESSETTOL)( long int *solver,
                                          double   *tol,
                                          int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESSetTol( (HYPRE_Solver) *solver,
                                            (double)       *tol     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetminiter, HYPRE_PARCSRGMRESSETMINITER)( long int *solver,
                                              int      *min_iter,
                                              int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESSetMinIter( (HYPRE_Solver) *solver,
                                                (int)          *min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetmaxiter, HYPRE_PARCSRGMRESSETMAXITER)( long int *solver,
                                              int      *max_iter,
                                              int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESSetMaxIter( (HYPRE_Solver) *solver,
                                                (int)          *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetprecond, HYPRE_PARCSRGMRESSETPRECOND)( long int *solver,
                                              int      *precond_id,
                                              long int *precond_solver,
                                              int      *ierr          )
{
   /*------------------------------------------------------------
    * The precond_id flags mean :
    *  0 - no preconditioner
    *  1 - set up a ds preconditioner
    *  2 - set up an amg preconditioner
    *  3 - set up a pilut preconditioner
    *  4 - set up a parasails preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = 0;
   }
   else if (*precond_id == 1)
   {
      *ierr = (int)
              ( HYPRE_ParCSRGMRESSetPrecond( (HYPRE_Solver) *solver,
                                             HYPRE_ParCSRDiagScale,
                                             HYPRE_ParCSRDiagScaleSetup,
                                             NULL                        ) );
   }
   else if (*precond_id == 2)
   {

   *ierr = (int) ( HYPRE_ParCSRGMRESSetPrecond( (HYPRE_Solver) *solver,
                                                HYPRE_BoomerAMGSolve,
                                                HYPRE_BoomerAMGSetup,
                                                (void *)       *precond_solver ) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (int)
              ( HYPRE_ParCSRGMRESSetPrecond( (HYPRE_Solver) *solver,
                                             HYPRE_ParCSRPilutSolve,
                                             HYPRE_ParCSRPilutSetup,
                                             (void *)       *precond_solver ) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (int)
              ( HYPRE_ParCSRGMRESSetPrecond( (HYPRE_Solver) *solver,
                                             HYPRE_ParCSRParaSailsSolve,
                                             HYPRE_ParCSRParaSailsSetup,
                                             (void *)       *precond_solver ) );
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmresgetprecond, HYPRE_PARCSRGMRESGETPRECOND)( long int *solver,
                                              long int *precond_solver_ptr,
                                              int      *ierr                )
{
    *ierr = (int)
            ( HYPRE_ParCSRGMRESGetPrecond( (HYPRE_Solver)   *solver,
                                           (HYPRE_Solver *)  precond_solver_ptr ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetlogging, HYPRE_PARCSRGMRESSETLOGGING)( long int *solver,
                                              int      *logging,
                                              int      *ierr     )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESSetLogging( (HYPRE_Solver) *solver,
                                                (int)          *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetNumIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmresgetnumiteratio, HYPRE_PARCSRGMRESGETNUMITERATIO)( long int *solver,
                                                  int      *num_iterations,
                                                  int      *ierr            )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESGetNumIterations(
                            (HYPRE_Solver) *solver,
                            (int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetFinalRelati
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmresgetfinalrelati, HYPRE_PARCSRGMRESGETFINALRELATI)( long int *solver,
                                                  double   *norm,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(
                            (HYPRE_Solver) *solver,
                            (double *)      norm    ) );
}
