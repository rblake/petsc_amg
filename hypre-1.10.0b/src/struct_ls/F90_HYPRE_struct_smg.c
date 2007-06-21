/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_StructSMG Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgcreate, HYPRE_STRUCTSMGCREATE)( int      *comm,
                                        long int *solver,
                                        int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGCreate( (MPI_Comm)             *comm,
                                          (HYPRE_StructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgdestroy, HYPRE_STRUCTSMGDESTROY)( long int *solver,
                                         int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGDestroy( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structsmgsetup, HYPRE_STRUCTSMGSETUP)( long int *solver,
                                       long int *A,
                                       long int *b,
                                       long int *x,
                                       int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGSetup( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structsmgsolve, HYPRE_STRUCTSMGSOLVE)( long int *solver,
                                       long int *A,
                                       long int *b,
                                       long int *x,
                                       int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGSolve( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetMemoryUse
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetmemoryuse, HYPRE_STRUCTSMGSETMEMORYUSE)( long int *solver,
                                              int      *memory_use,
                                              int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructSMGSetMemoryUse( (HYPRE_StructSolver) *solver,
                                     (int)                *memory_use ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsettol, HYPRE_STRUCTSMGSETTOL)( long int *solver,
                                        double   *tol,
                                        int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGSetTol( (HYPRE_StructSolver) *solver,
                                          (double)             *tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetmaxiter, HYPRE_STRUCTSMGSETMAXITER)( long int *solver,
                                            int      *max_iter,
                                            int      *ierr     )
{
   *ierr = (int)
      ( HYPRE_StructSMGSetMaxIter( (HYPRE_StructSolver) *solver,
                                   (int)                *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetrelchange, HYPRE_STRUCTSMGSETRELCHANGE)( long int *solver,
                                              int      *rel_change,
                                              int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructSMGSetRelChange( (HYPRE_StructSolver) *solver,
                                     (int)                *rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structsmgsetzeroguess, HYPRE_STRUCTSMGSETZEROGUESS)( long int *solver,
                                              int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructSMGSetZeroGuess( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structsmgsetnonzeroguess, HYPRE_STRUCTSMGSETNONZEROGUESS)( long int *solver,
                                                 int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructSMGSetNonZeroGuess( (HYPRE_StructSolver) *solver ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNumPreRelax
 *
 * Note that we require at least 1 pre-relax sweep. 
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetnumprerelax, HYPRE_STRUCTSMGSETNUMPRERELAX)( long int *solver,
                                                int      *num_pre_relax,
                                                int      *ierr         )
{
   *ierr = (int)
      ( HYPRE_StructSMGSetNumPreRelax( (HYPRE_StructSolver) *solver,
                                       (int)                *num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetnumpostrelax, HYPRE_STRUCTSMGSETNUMPOSTRELAX)( long int *solver,
                                                 int      *num_post_relax,
                                                 int      *ierr           )
{
   *ierr = (int)
      ( HYPRE_StructSMGSetNumPostRelax(
         (HYPRE_StructSolver) *solver,
         (int)                *num_post_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetlogging, HYPRE_STRUCTSMGSETLOGGING)( long int *solver,
                                            int      *logging,
                                            int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructSMGSetLogging( (HYPRE_StructSolver) *solver,
                                   (int)                *logging) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetprintlevel, HYPRE_STRUCTSMGSETPRINTLEVEL)( long int *solver,
                                            int      *print_level,
                                            int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructSMGSetPrintLevel( (HYPRE_StructSolver) *solver,
                                   (int)                *print_level) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmggetnumiterations, HYPRE_STRUCTSMGGETNUMITERATIONS)( long int *solver,
                                                  int      *num_iterations,
                                                  int      *ierr           )
{
   *ierr = (int)
      ( HYPRE_StructSMGGetNumIterations(
         (HYPRE_StructSolver) *solver,
         (int *)              num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmggetfinalrelative, HYPRE_STRUCTSMGGETFINALRELATIVE)( long int *solver,
                                                  double   *norm,
                                                  int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructSMGGetFinalRelativeResidualNorm(
         (HYPRE_StructSolver) *solver,
         (double *)           norm ) );
}
