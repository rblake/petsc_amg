/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.2 $
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Relaxation scheme
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * hypre_BoomerAMGRelax
 *--------------------------------------------------------------------------*/

int  hypre_BoomerAMGRelaxIF( hypre_ParCSRMatrix *A,
                             hypre_ParVector    *f,
                             int                *cf_marker,
                             int                 relax_type,
                             int                 relax_order,
                             int                 cycle_type,
                             double              relax_weight,
                             double              omega,
                             hypre_ParVector    *u,
                             hypre_ParVector    *Vtemp )
{
   int i, Solve_err_flag = 0;
   int relax_points[2];
   if (relax_order == 1 && cycle_type < 3)
   {
      if (cycle_type < 2)
      {
         relax_points[0] = 1;
	 relax_points[1] = -1;
      }
      else
      {
	 relax_points[0] = -1;
	 relax_points[1] = 1;
      }
/*      if (relax_type == 6)
      {
         Solve_err_flag = hypre_BoomerAMGRelax(A,
                                            f,
                                            cf_marker,
                                            3,
                                            relax_points[0],
                                            relax_weight,
                                            omega,
                                            u,
                                            Vtemp); 
         Solve_err_flag = hypre_BoomerAMGRelax(A,
                                            f,
                                            cf_marker,
                                            4,
                                            relax_points[0],
                                            relax_weight,
                                            omega,
                                            u,
                                            Vtemp); 
         Solve_err_flag = hypre_BoomerAMGRelax(A,
                                            f,
                                            cf_marker,
                                            4,
                                            relax_points[1],
                                            relax_weight,
                                            omega,
                                            u,
                                            Vtemp); 
         Solve_err_flag = hypre_BoomerAMGRelax(A,
                                            f,
                                            cf_marker,
                                            3,
                                            relax_points[1],
                                            relax_weight,
                                            omega,
                                            u,
                                            Vtemp); 
      }
      else */
      {
         for (i=0; i < 2; i++)
            Solve_err_flag = hypre_BoomerAMGRelax(A,
                                            f,
                                            cf_marker,
                                            relax_type,
                                            relax_points[i],
                                            relax_weight,
                                            omega,
                                            u,
                                            Vtemp); 
      }
   }
   else
   {
      Solve_err_flag = hypre_BoomerAMGRelax(A,
                                            f,
                                            cf_marker,
                                            relax_type,
                                            0,
                                            relax_weight,
                                            omega,
                                            u,
                                            Vtemp); 
   }

   return Solve_err_flag;
}
