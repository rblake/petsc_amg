/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.3 $
 *********************************************************************EHEADER*/

#include "headers.h"                                                                                                           
#include "utilities.h"
#include "parcsr_mv.h"

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockCommHandleCreate
 *--------------------------------------------------------------------------*/

hypre_ParCSRCommHandle *
hypre_ParCSRBlockCommHandleCreate(int job, int bnnz, hypre_ParCSRCommPkg *comm_pkg,
                                  void *send_data, void *recv_data )
{
   int      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int      num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   MPI_Comm comm      = hypre_ParCSRCommPkgComm(comm_pkg);
   hypre_ParCSRCommHandle *comm_handle;
   int         num_requests;
   MPI_Request *requests;
   int    i, j, my_id, num_procs, ip, vec_start, vec_len;
   double *d_send_data = (double *) send_data;
   double *d_recv_data = (double *) recv_data;
                  
/*---------------------------------------------------------------------------
    * job = 1 : is used to initialize communication exchange for the parts
    *		of vector needed to perform a Matvec,  it requires send_data 
    *		and recv_data to be doubles, recv_vec_starts and 
    *		send_map_starts need to be set in comm_pkg.
    * job = 2 : is used to initialize communication exchange for the parts
    *		of vector needed to perform a MatvecT,  it requires send_data 
    *		and recv_data to be doubles, recv_vec_starts and 
    *		send_map_starts need to be set in comm_pkg.
    *------------------------------------------------------------------------*/


   num_requests = num_sends + num_recvs;
   requests = hypre_CTAlloc(MPI_Request, num_requests);
 
   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   j = 0;

   switch (job)
   {
      case  1:
      {
         for (i = 0; i < num_recvs; i++)
         {
            ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i); 
            vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i);
            vec_len = (hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i+1)-vec_start)*bnnz;
            MPI_Irecv(&d_recv_data[vec_start*bnnz], vec_len, MPI_DOUBLE,
                      ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_sends; i++)
         {
            vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len = (hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1)-vec_start)*bnnz;
            ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i); 
            MPI_Isend(&d_send_data[vec_start*bnnz], vec_len, MPI_DOUBLE, ip, 0, comm, 
                      &requests[j++]);
         }
         break;
      }
      case  2:
      {

         for (i = 0; i < num_sends; i++)
         {
	    vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	    vec_len = (hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1) - vec_start)*bnnz;
      	    ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i); 
   	    MPI_Irecv(&d_recv_data[vec_start*bnnz], vec_len, MPI_DOUBLE,
                      ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_recvs; i++)
         {
            ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i); 
            vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i);
            vec_len = (hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i+1)-vec_start)*bnnz;
            MPI_Isend(&d_send_data[vec_start*bnnz], vec_len, MPI_DOUBLE,
                      ip, 0, comm, &requests[j++]);
         }
         break;
      }
   }
   

  /*--------------------------------------------------------------------
   * set up comm_handle and return
   *--------------------------------------------------------------------*/

   comm_handle = hypre_CTAlloc(hypre_ParCSRCommHandle, 1);

   hypre_ParCSRCommHandleCommPkg(comm_handle)     = comm_pkg;
   hypre_ParCSRCommHandleSendData(comm_handle)    = send_data;
   hypre_ParCSRCommHandleRecvData(comm_handle)    = recv_data;
   hypre_ParCSRCommHandleNumRequests(comm_handle) = num_requests;
   hypre_ParCSRCommHandleRequests(comm_handle)    = requests;
   return ( comm_handle );
}

int
hypre_ParCSRBlockCommHandleDestroy(hypre_ParCSRCommHandle *comm_handle)
{
   MPI_Status          *status0;
   int			ierr = 0;

   if ( comm_handle==NULL ) return ierr;
   if (hypre_ParCSRCommHandleNumRequests(comm_handle))
   {
      status0 = hypre_CTAlloc(MPI_Status,
                       hypre_ParCSRCommHandleNumRequests(comm_handle));
      MPI_Waitall(hypre_ParCSRCommHandleNumRequests(comm_handle),
                  hypre_ParCSRCommHandleRequests(comm_handle), status0);
      hypre_TFree(status0);
   }

   hypre_TFree(hypre_ParCSRCommHandleRequests(comm_handle));
   hypre_TFree(comm_handle);

   return ierr;
}


/* ----------------------------------------------------------------------
 * hypre_BlockNewCommPkgCreate
 * ---------------------------------------------------------------------*/

int
hypre_BlockNewCommPkgCreate(hypre_ParCSRBlockMatrix *A)
{
 
   int        row_start=0, row_end=0, col_start = 0, col_end = 0;
   int        num_recvs, *recv_procs, *recv_vec_starts;

   int        num_sends, *send_procs, *send_map_starts;
   int        *send_map_elements;

   int        num_cols_off_d; 
   int       *col_map_off_d; 

   int        first_col_diag;
   int        global_num_cols;

   int        ierr = 0;

   MPI_Comm   comm;

   hypre_ParCSRCommPkg	 *comm_pkg;

   
   /*-----------------------------------------------------------
    * get parcsr_A information 
    *----------------------------------------------------------*/

   row_start = hypre_ParCSRBlockMatrixFirstRowIndex(A);
   row_end = hypre_ParCSRBlockMatrixLastRowIndex(A);
   col_start =  hypre_ParCSRBlockMatrixFirstColDiag(A);
   col_end =  hypre_ParCSRBlockMatrixLastColDiag(A);
   
   col_map_off_d =  hypre_ParCSRBlockMatrixColMapOffd(A);
   num_cols_off_d = hypre_CSRBlockMatrixNumCols(hypre_ParCSRBlockMatrixOffd(A));
   
   global_num_cols = hypre_ParCSRBlockMatrixGlobalNumCols(A); 
   
   comm = hypre_ParCSRBlockMatrixComm(A);

   first_col_diag = hypre_ParCSRBlockMatrixFirstColDiag(A);

   /*-----------------------------------------------------------
    * get commpkg info information 
    *----------------------------------------------------------*/

   hypre_NewCommPkgCreate_core( comm, col_map_off_d, first_col_diag, 
                                col_start, col_end, 
                                num_cols_off_d, global_num_cols,
                                &num_recvs, &recv_procs, &recv_vec_starts,
                                &num_sends, &send_procs, &send_map_starts, 
                                &send_map_elements);



   if (!num_recvs)
   {
      hypre_TFree(recv_procs);
      recv_procs = NULL;
   }
   if (!num_sends)
   {
      hypre_TFree(send_procs);
      hypre_TFree(send_map_elements);
      send_procs = NULL;
      send_map_elements = NULL;
   }

  /*-----------------------------------------------------------
   * setup commpkg
   *----------------------------------------------------------*/

   comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg, 1);

   hypre_ParCSRCommPkgComm(comm_pkg) = comm;

   hypre_ParCSRCommPkgNumRecvs(comm_pkg) = num_recvs;
   hypre_ParCSRCommPkgRecvProcs(comm_pkg) = recv_procs;
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg) = recv_vec_starts;

   hypre_ParCSRCommPkgNumSends(comm_pkg) = num_sends;
   hypre_ParCSRCommPkgSendProcs(comm_pkg) = send_procs;
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg) = send_map_starts;
   hypre_ParCSRCommPkgSendMapElmts(comm_pkg) = send_map_elements;

   hypre_ParCSRBlockMatrixCommPkg(A) = comm_pkg;

   return ierr;
}
