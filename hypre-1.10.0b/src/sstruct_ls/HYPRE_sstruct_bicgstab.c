/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_SStructBiCGSTAB interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructBiCGSTABCreate( MPI_Comm             comm,
                          HYPRE_SStructSolver *solver )
{
   hypre_BiCGSTABFunctions * bicgstab_functions =
      hypre_BiCGSTABFunctionsCreate(
         hypre_SStructKrylovCreateVector,
         hypre_SStructKrylovDestroyVector, hypre_SStructKrylovMatvecCreate,
         hypre_SStructKrylovMatvec, hypre_SStructKrylovMatvecDestroy,
         hypre_SStructKrylovInnerProd, hypre_SStructKrylovCopyVector,
         hypre_SStructKrylovScaleVector, hypre_SStructKrylovAxpy,
	 hypre_SStructKrylovCommInfo,
         hypre_SStructKrylovIdentitySetup, hypre_SStructKrylovIdentity );

   *solver = ( (HYPRE_SStructSolver) hypre_BiCGSTABCreate( bicgstab_functions ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructBiCGSTABDestroy( HYPRE_SStructSolver solver )
{
   return( hypre_BiCGSTABDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructBiCGSTABSetup( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   return( HYPRE_BiCGSTABSetup( (HYPRE_Solver) solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructBiCGSTABSolve( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   return( HYPRE_BiCGSTABSolve( (HYPRE_Solver) solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructBiCGSTABSetTol( HYPRE_SStructSolver solver,
                          double              tol )
{
   return( HYPRE_BiCGSTABSetTol( (HYPRE_Solver) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructBiCGSTABSetMinIter( HYPRE_SStructSolver solver,
                              int                 min_iter )
{
   return( HYPRE_BiCGSTABSetMinIter( (HYPRE_Solver) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructBiCGSTABSetMaxIter( HYPRE_SStructSolver solver,
                              int                 max_iter )
{
   return( HYPRE_BiCGSTABSetMaxIter( (HYPRE_Solver) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructBiCGSTABSetStopCrit( HYPRE_SStructSolver solver,
                               int                 stop_crit )
{
   return( HYPRE_BiCGSTABSetStopCrit( (HYPRE_Solver) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructBiCGSTABSetPrecond( HYPRE_SStructSolver          solver,
                              HYPRE_PtrToSStructSolverFcn  precond,
                              HYPRE_PtrToSStructSolverFcn  precond_setup,
                              void *          precond_data )
{
   return( HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                  (HYPRE_PtrToSolverFcn) precond,
                                  (HYPRE_PtrToSolverFcn) precond_setup,
                                  (HYPRE_Solver) precond_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructBiCGSTABSetLogging( HYPRE_SStructSolver solver,
                              int                 logging )
{
   return( HYPRE_BiCGSTABSetLogging( (HYPRE_Solver) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructBiCGSTABSetPrintLevel( HYPRE_SStructSolver solver,
                              int                 print_level )
{
   return( HYPRE_BiCGSTABSetPrintLevel( (HYPRE_Solver) solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructBiCGSTABGetNumIterations( HYPRE_SStructSolver  solver,
                                    int                 *num_iterations )
{
   return( HYPRE_BiCGSTABGetNumIterations( (HYPRE_Solver) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                                double              *norm )
{
   return( HYPRE_BiCGSTABGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABGetResidual
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructBiCGSTABGetResidual( HYPRE_SStructSolver  solver,
                                  void 			**residual)
{
   return( HYPRE_BiCGSTABGetResidual( (HYPRE_Solver) solver, residual ) );
}
