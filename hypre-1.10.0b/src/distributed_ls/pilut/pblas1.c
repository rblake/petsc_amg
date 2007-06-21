/*
 * pblas1.c
 *
 * This file contains functions that implement various distributed
 * level 1 BLAS routines
 *
 * Started 11/28/95
 * George
 *
 * $Id: pblas1.c,v 2.2 2004/03/05 21:24:37 treadway Exp $
 *
 */

#include "ilu.h"
#include "DistributedMatrixPilutSolver.h"


/*************************************************************************
* This function computes the 2 norm of a vector. The result is returned
* at all the processors
**************************************************************************/
double hypre_p_dnrm2(DataDistType *ddist, double *x, hypre_PilutSolverGlobals *globals)
{
  int incx=1;
  double sum;

  sum = SNRM2(&(ddist->ddist_lnrows), x, &incx);
  return sqrt(hypre_GlobalSESumDouble(sum*sum, pilut_comm));
}


/*************************************************************************
* This function computes the dot product of 2 vectors. 
* The result is returned at all the processors
**************************************************************************/
double hypre_p_ddot(DataDistType *ddist, double *x, double *y,
              hypre_PilutSolverGlobals *globals)
{
  int incx=1;

  return hypre_GlobalSESumDouble(SDOT(&(ddist->ddist_lnrows), x, &incx, y, &incx), 
         pilut_comm );
}


/*************************************************************************
* This function performs y = alpha*x, where alpha resides on pe 0
**************************************************************************/
void hypre_p_daxy(DataDistType *ddist, double alpha, double *x, double *y)
{
  int i, local_lnrows=ddist->ddist_lnrows;

  for (i=0; i<local_lnrows; i++)
    y[i] = alpha*x[i];
}


/*************************************************************************
* This function performs y = alpha*x+y, where alpha resides on pe 0
**************************************************************************/
void hypre_p_daxpy(DataDistType *ddist, double alpha, double *x, double *y)
{
  int i, local_lnrows=ddist->ddist_lnrows;

  for (i=0; i<local_lnrows; i++)
    y[i] += alpha*x[i];
}



/*************************************************************************
* This function performs z = alpha*x+beta*y, where alpha resides on pe 0
**************************************************************************/
void hypre_p_daxbyz(DataDistType *ddist, double alpha, double *x, double beta, 
              double *y, double *z)
{
  int i, local_lnrows=ddist->ddist_lnrows;

  for (i=0; i<local_lnrows; i++)
    z[i] = alpha*x[i] + beta*y[i];
}

/*************************************************************************
* This function prints a vector
**************************************************************************/
int hypre_p_vprintf(DataDistType *ddist, double *x,
                    hypre_PilutSolverGlobals *globals )
{
  int pe, i;

  for (pe=0; pe<npes; pe++) {
    if (mype == pe) {
      for (i=0; i<ddist->ddist_lnrows; i++)
        printf("%d:%f, ", ddist->ddist_rowdist[mype]+i, x[i]);
      if (pe == npes-1)
        printf("\n");
    }
    MPI_Barrier( pilut_comm );
  }
  fflush(stdout);
  MPI_Barrier( pilut_comm );

  return 0;
}
