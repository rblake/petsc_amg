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
 *****************************************************************************/

/* following should be in a header file */


#include "headers.h"



/*==========================================================================*/
/*==========================================================================*/
/**
  Generates global coarse_size and dof_func for next coarser level

  Notes:
  \begin{itemize}
  \item The routine returns the following:
  \begin{itemize}
  \item an integer array containing the
  function values for the local coarse points
  \item the global number of coarse points
  \end{itemize}
  \end{itemize}

  {\bf Input files:}
  headers.h

  @return Error code.
  
  @param comm [IN]
  MPI Communicator
  @param local_num_variables [IN]
  number of points on local processor
  @param dof_func [IN]
  array that contains the function numbers for all local points
  @param CF_marker [IN]
  marker array for coarse points 
  @param coarse_dof_func_ptr [OUT]
  pointer to array which contains the function numbers for local coarse points
  @param coarse_pnts_global_ptr [OUT]
  pointer to array which contains the number of the first coarse point on each  processor and the total number of coarse points in its last element
  
  @see */
/*--------------------------------------------------------------------------*/

int
hypre_BoomerAMGCoarseParms(MPI_Comm comm,
		           int      local_num_variables,
		           int      num_functions,
		           int     *dof_func,
		           int     *CF_marker, 
                      	   int    **coarse_dof_func_ptr, 
                      	   int    **coarse_pnts_global_ptr) 
{
   int            i;
   int            ierr = 0;
   int		  num_procs;
   int            local_coarse_size = 0;

   int	 *coarse_dof_func;
   int	 *coarse_pnts_global;

   /*--------------------------------------------------------------
    *----------------------------------------------------------------*/

   MPI_Comm_size(comm,&num_procs);

   for (i=0; i < local_num_variables; i++)
   {
      if (CF_marker[i] == 1) local_coarse_size++;
   }
   if (num_functions > 1)
   {
      coarse_dof_func = hypre_CTAlloc(int,local_coarse_size);

      local_coarse_size = 0;
      for (i=0; i < local_num_variables; i++)
      {
         if (CF_marker[i] == 1)
            coarse_dof_func[local_coarse_size++] = dof_func[i];
      }
      *coarse_dof_func_ptr    = coarse_dof_func;
   }


#ifdef HYPRE_NO_GLOBAL_PARTITION
   {
      int scan_recv;
      
      coarse_pnts_global = hypre_CTAlloc(int,2);
      MPI_Scan(&local_coarse_size, &scan_recv, 1, MPI_INT, MPI_SUM, comm);
      /* first point in my range */ 
      coarse_pnts_global[0] = scan_recv - local_coarse_size;
      /* first point in next proc's range */
      coarse_pnts_global[1] = scan_recv;

   }
      

#else
   coarse_pnts_global = hypre_CTAlloc(int,num_procs+1);

   MPI_Allgather(&local_coarse_size,1,MPI_INT,&coarse_pnts_global[1],
		1,MPI_INT,comm);

   for (i=2; i < num_procs+1; i++)
      coarse_pnts_global[i] += coarse_pnts_global[i-1];
#endif




   *coarse_pnts_global_ptr = coarse_pnts_global;

   return (ierr);
}
