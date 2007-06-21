/*BHEADER**********************************************************************
 * (c) 1998-2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_ParCSRGMRES interface
 *
 *****************************************************************************/
#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   hypre_GMRESFunctions * gmres_functions =
      hypre_GMRESFunctionsCreate(
         hypre_CAlloc, hypre_ParKrylovFree, hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovCreateVectorArray,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );

   *solver = ( (HYPRE_Solver) hypre_GMRESCreate( gmres_functions ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRGMRESDestroy( HYPRE_Solver solver )
{
   return( hypre_GMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRGMRESSetup( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return( HYPRE_GMRESSetup( solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRGMRESSolve( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return( HYPRE_GMRESSolve( solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetKDim
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESSetKDim( HYPRE_Solver solver,
                          int             k_dim    )
{
   return( HYPRE_GMRESSetKDim( solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESSetTol( HYPRE_Solver solver,
                         double             tol    )
{
   return( HYPRE_GMRESSetTol( solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetMinIter
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESSetMinIter( HYPRE_Solver solver,
                             int          min_iter )
{
   return( HYPRE_GMRESSetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESSetMaxIter( HYPRE_Solver solver,
                             int          max_iter )
{
   return( HYPRE_GMRESSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetStopCrit
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESSetStopCrit( HYPRE_Solver solver,
                              int          stop_crit )
{
   return( HYPRE_GMRESSetStopCrit( solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESSetPrecond( HYPRE_Solver          solver,
                             HYPRE_PtrToParSolverFcn  precond,
                             HYPRE_PtrToParSolverFcn  precond_setup,
                             HYPRE_Solver          precond_solver )
{
   return( HYPRE_GMRESSetPrecond( solver,
                                  (HYPRE_PtrToSolverFcn) precond,
                                  (HYPRE_PtrToSolverFcn) precond_setup,
                                  precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESGetPrecond( HYPRE_Solver  solver,
                             HYPRE_Solver *precond_data_ptr )
{
   return( HYPRE_GMRESGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESSetLogging( HYPRE_Solver solver,
                             int logging)
{
   return( HYPRE_GMRESSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESSetPrintLevel( HYPRE_Solver solver,
                             int print_level)
{
   return( HYPRE_GMRESSetPrintLevel( solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESGetNumIterations( HYPRE_Solver  solver,
                                   int                *num_iterations )
{
   return( HYPRE_GMRESGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               double             *norm   )
{
   return( HYPRE_GMRESGetFinalRelativeResidualNorm( solver, norm ) );
}
