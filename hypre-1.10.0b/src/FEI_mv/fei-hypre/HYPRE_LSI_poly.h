/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_POLY interface
 *
 *****************************************************************************/

#ifndef __HYPRE_POLY__
#define __HYPRE_POLY__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

#include "utilities/utilities.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "parcsr_mv/parcsr_mv.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern int HYPRE_LSI_PolyCreate( MPI_Comm comm, HYPRE_Solver *solver );
extern int HYPRE_LSI_PolyDestroy( HYPRE_Solver solver );
extern int HYPRE_LSI_PolySetOrder( HYPRE_Solver solver, int order);
extern int HYPRE_LSI_PolySetOutputLevel( HYPRE_Solver solver, int level);
extern int HYPRE_LSI_PolySolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                HYPRE_ParVector b,   HYPRE_ParVector x );
extern int HYPRE_LSI_PolySetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                HYPRE_ParVector b,   HYPRE_ParVector x );
#ifdef __cplusplus
}
#endif

#endif

