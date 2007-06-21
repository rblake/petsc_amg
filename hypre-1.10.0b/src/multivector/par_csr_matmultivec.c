/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.2 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "par_csr_multimatvec.h"

#include "parcsr_mv.h"

#include "seq_multivector.h"
#include "par_multivector.h"
#include "par_csr_pmvcomm.h"
#include "csr_multimatvec.h"

#include <assert.h>

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMultiMatvec
 *
 *   Performs y <- alpha * A * x + beta * y
 *
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRMatrixMatMultiVec(double alpha, hypre_ParCSRMatrix *A,
                              hypre_ParMultivector *x, double beta,
                              hypre_ParMultivector *y)
{
   hypre_ParCSRCommMultiHandle	*comm_handle;
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix     *diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix     *offd   = hypre_ParCSRMatrixOffd(A);
   hypre_Multivector  *x_local  = hypre_ParMultivectorLocalVector(x);   
   hypre_Multivector  *y_local  = hypre_ParMultivectorLocalVector(y);   
   int                 num_rows = hypre_CSRMatrixNumRows(diag);
   int                 num_cols = hypre_CSRMatrixNumCols(diag);
   int                 *x_active_ind = x->active_indices;
   int                 *y_active_ind = y->active_indices;

   hypre_Multivector   *x_tmp;
   int        x_size = hypre_MultivectorSize(x_local);
   int        y_size = hypre_MultivectorSize(y_local);
   int        num_vectors = hypre_MultivectorNumVectors(x_local);
   int	      num_cols_offd = hypre_CSRMatrixNumCols(offd);
   int        ierr = 0, send_leng, num_vec_sends, endp1;
   int	      num_sends, i, j, jj, index, start, offset, length, jv;
   int        num_active_vectors;

   double     *x_tmp_data, *x_buf_data;
   double     *x_local_data = hypre_MultivectorData(x_local);
 
   /*---------------------------------------------------------------------
    * count the number of active vectors -> num_vec_sends
    *--------------------------------------------------------------------*/

   num_active_vectors = x->num_active_vectors;
   hypre_assert(num_active_vectors == y->num_active_vectors);
   if (x_active_ind == NULL) num_vec_sends = num_vectors;
   else                      num_vec_sends = x->num_active_vectors;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  ParMatvec returns ierr = 11 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 12 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 13 if both are true.
    *
    *  Because temporary vectors are often used in ParMatvec, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
 
   if (num_cols != x_size) ierr = 11;
   if (num_rows != y_size) ierr = 12;
   if (num_cols != x_size && num_rows != y_size) ierr = 13;

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   send_leng = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);

   /*---------------------------------------------------------------------
    * allocate temporary and send buffers and communication handle
    *--------------------------------------------------------------------*/

   x_buf_data = hypre_CTAlloc(double, num_vec_sends*send_leng);
   x_tmp = hypre_SeqMultivectorCreate( num_cols_offd, num_vectors );
   hypre_SeqMultivectorInitialize(x_tmp);
   x_tmp_data = hypre_MultivectorData(x_tmp);
   comm_handle = hypre_CTAlloc(hypre_ParCSRCommMultiHandle, 1);

   /*---------------------------------------------------------------------
    * put the send data into the send buffer
    *--------------------------------------------------------------------*/
   
   offset = 0;
   for ( jv = 0; jv < num_active_vectors; ++jv )
   {
      jj = x_active_ind[jv];
      for (i = 0; i < num_sends; i++)
      {
         start  = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         endp1  = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1);
         length = endp1 - start;
         for (j = start; j < endp1; j++)
         {
            index = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            x_buf_data[offset+j] = x_local_data[jj*x_size + index];
         }
      }
      offset += hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   }

   /*---------------------------------------------------------------------
    * initiate sending data
    *--------------------------------------------------------------------*/

   comm_handle = hypre_ParCSRCommMultiHandleCreate(1,comm_pkg,x_buf_data, 
                                   x_tmp_data, num_vec_sends);

   hypre_CSRMatrixMatMultivec(alpha, diag, x_local, beta, y_local);
   
   hypre_ParCSRCommMultiHandleDestroy(comm_handle);
   comm_handle = NULL;
   hypre_TFree(comm_handle);

   if (num_cols_offd)
      hypre_CSRMatrixMultiMatvec(alpha, offd, x_tmp, 1.0, y_local);    

   hypre_SeqMultivectorDestroy(x_tmp);
   x_tmp = NULL;
   hypre_TFree(x_buf_data);
  
   return ierr;
}

/*--------------------------------------------------------------------------
 *           hypre_ParCSRMatrixMultiMatvecT 
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRMatrixMultiMatVecT(double alpha, hypre_ParCSRMatrix *A,
                               hypre_ParMultivector *x, double beta,
                               hypre_ParMultivector *y)
{
   hypre_ParCSRCommMultiHandle	*comm_handle;
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix     *diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix     *offd   = hypre_ParCSRMatrixOffd(A);
   hypre_Multivector   *x_local  = hypre_ParMultivectorLocalVector(x);   
   hypre_Multivector   *y_local  = hypre_ParMultivectorLocalVector(y);   
   int                 num_rows = hypre_CSRMatrixNumRows(diag);
   int                 num_cols = hypre_CSRMatrixNumCols(diag);
   int                 *x_active_ind = x->active_indices;

   hypre_Multivector   *y_tmp;
   int        x_size = hypre_MultivectorSize(x_local);
   int        y_size = hypre_MultivectorSize(y_local);
   int        num_vectors = hypre_MultivectorNumVectors(x_local);
   int	      num_cols_offd = hypre_CSRMatrixNumCols(offd);
   int        ierr = 0, send_leng, num_vec_sends, endp1;
   int	      num_sends, i, j, jj, index, start, offset, length, jv;
   int        num_active_vectors;

   double     *y_tmp_data, *y_buf_data;
   double     *y_local_data = hypre_MultivectorData(y_local);

   /*---------------------------------------------------------------------
    * count the number of active vectors -> num_vec_sends
    *--------------------------------------------------------------------*/

   num_active_vectors = x->num_active_vectors;
   hypre_assert(num_active_vectors == y->num_active_vectors);
   if (x_active_ind == NULL) num_vec_sends = num_vectors;
   else                      num_vec_sends = x->num_active_vectors;
    
   /*---------------------------------------------------------------------
    *  Check for size compatibility.  MatvecT returns ierr = 1 if
    *  length of X doesn't equal the number of rows of A,
    *  ierr = 2 if the length of Y doesn't equal the number of 
    *  columns of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in MatvecT, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
 
    if (num_rows != x_size)  ierr = 1;
    if (num_cols != y_size)  ierr = 2;
    if (num_rows != x_size && num_cols != y_size)  ierr = 3;
    
    /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   send_leng = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   
    /*---------------------------------------------------------------------
    * allocate temporary and send buffers and communication handle
    *--------------------------------------------------------------------*/

   y_buf_data = hypre_CTAlloc(double, num_vec_sends*send_leng);
   y_tmp = hypre_SeqMultivectorCreate( num_cols_offd, num_vectors );
   hypre_SeqMultivectorInitialize(y_tmp);
   y_tmp_data = hypre_MultivectorData(y_tmp);
   comm_handle = hypre_CTAlloc(hypre_ParCSRCommMultiHandle, 1);
   
   /*---------------------------------------------------------------------
    * put the send data into the send buffer
    *--------------------------------------------------------------------*/
   
   offset = 0;
   for ( jv = 0; jv < num_vectors; ++jv )
   {
      jj = x_active_ind[jv];
      for (i = 0; i < num_sends; i++)
      {
         start  = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         endp1  = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1);
         length = endp1 - start;
         for (j = start; j < endp1; j++)
         {
            index = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            y_buf_data[offset+j] = y_local_data[jj*y_size + index];
         }
      }
      offset += hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   }

   /*---------------------------------------------------------------------
    * initiate sending data
    *--------------------------------------------------------------------*/

   comm_handle = hypre_ParCSRCommMultiHandleCreate(1, comm_pkg, 
		            y_buf_data, y_tmp_data, num_vec_sends);

   hypre_CSRMatrixMultiMatvecT(alpha, diag, x_local, beta, y_local);
   
   hypre_ParCSRCommMultiHandleDestroy(comm_handle);
   comm_handle = NULL;
   hypre_TFree(comm_handle);

   if (num_cols_offd)
      hypre_CSRMatrixMultiMatvecT(alpha, offd, y_tmp, 1.0, y_local);    

   hypre_SeqMultivectorDestroy(y_tmp);
   y_tmp = NULL;
   hypre_TFree(y_buf_data);
  
   return ierr;
}
   
