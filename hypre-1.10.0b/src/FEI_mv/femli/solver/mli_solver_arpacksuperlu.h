/*BHEADER**********************************************************************
 * (c) 2002   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifdef MLI_SUPERLU

#ifndef __MLI_SOLVER_ARPACKSUPERLU_H__
#define __MLI_SOLVER_ARPACKSUPERLU_H__

#include <stdio.h>
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"
#include "matrix/mli_matrix.h"

/******************************************************************************
 * data structure for the SuperLU in ARPACK shift-and-invert procedure
 *---------------------------------------------------------------------------*/

class MLI_Solver_ARPACKSuperLU : public MLI_Solver
{
   MLI_Matrix *Amat_;
   int        nRecvs_;
   int        *recvLengs_;
   int        *recvProcs_;
   int        nSends_;
   int        *sendLengs_;
   int        *sendProcs_;
   int        *sendMap_;
   int        nSendMap_;
   int        nNodes_;
   int        *ANodeEqnList_;
   int        *SNodeEqnList_;
   int        blockSize_;

public :

   MLI_Solver_ARPACKSuperLU(char *name);
   ~MLI_Solver_ARPACKSuperLU();
   int setup(MLI_Matrix *mat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int setParams(char *paramString, int argc, char **argv);
};

#endif

#else
   int bogus;
#endif

