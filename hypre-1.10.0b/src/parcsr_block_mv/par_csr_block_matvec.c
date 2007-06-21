/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.3 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/



#include "headers.h"

#include "HYPRE.h"
#include "parcsr_mv/parcsr_mv.h"
#include "seq_mv/seq_mv.h"
#include <assert.h>


/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvec
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRBlockMatrixMatvec(double alpha, hypre_ParCSRBlockMatrix *A,
                              hypre_ParVector *x, double beta,
                              hypre_ParVector *y)
{
   hypre_ParCSRCommHandle *comm_handle;
   hypre_ParCSRCommPkg	  *comm_pkg;
   hypre_CSRBlockMatrix   *diag, *offd;
   hypre_Vector           *x_local, *y_local, *x_tmp;
   int                    i, j, k, index, num_rows, num_cols;
   int                    blk_size, x_size, y_size, size;
   int	                  num_cols_offd, start, finish, elem;
   int                    ierr = 0, nprocs, num_sends, mypid;
   double                 *x_tmp_data, *x_buf_data, *x_local_data;

   MPI_Comm_size(hypre_ParCSRBlockMatrixComm(A), &nprocs);
   MPI_Comm_rank(hypre_ParCSRBlockMatrixComm(A), &mypid);
   comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   num_rows = hypre_ParCSRBlockMatrixGlobalNumRows(A);
   num_cols = hypre_ParCSRBlockMatrixGlobalNumCols(A);
   blk_size = hypre_ParCSRBlockMatrixBlockSize(A);
   diag   = hypre_ParCSRBlockMatrixDiag(A);
   offd   = hypre_ParCSRBlockMatrixOffd(A);
   num_cols_offd = hypre_CSRBlockMatrixNumCols(offd);
   x_local  = hypre_ParVectorLocalVector(x);   
   y_local  = hypre_ParVectorLocalVector(y);   
   x_size = hypre_ParVectorGlobalSize(x);
   y_size = hypre_ParVectorGlobalSize(y);
   x_local_data = hypre_VectorData(x_local);

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  
    *--------------------------------------------------------------------*/
 
   if (num_cols*blk_size != x_size) ierr = 11;
   if (num_rows*blk_size != y_size) ierr = 12;
   if (num_cols*blk_size != x_size && num_rows*blk_size != y_size) ierr = 13;

   if (nprocs > 1)
   {
      x_tmp = hypre_SeqVectorCreate(num_cols_offd*blk_size);
      hypre_SeqVectorInitialize(x_tmp);
      x_tmp_data = hypre_VectorData(x_tmp);

      if (!comm_pkg)
      {

#ifdef HYPRE_NO_GLOBAL_PARTITION
        hypre_BlockNewCommPkgCreate(A); 
#else
        hypre_BlockMatvecCommPkgCreate(A);
#endif
         comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A); 
      }
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      size = hypre_ParCSRCommPkgSendMapStart(comm_pkg,num_sends)*blk_size;
      x_buf_data = hypre_CTAlloc(double, size);
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         finish = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1);
         for (j = start; j < finish; j++)
         {
            elem = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)*blk_size;
            for (k = 0; k < blk_size; k++)
               x_buf_data[index++] = x_local_data[elem++];
         }
      }
      comm_handle = hypre_ParCSRBlockCommHandleCreate(1, blk_size,comm_pkg,
                                 x_buf_data, x_tmp_data);
   }
   hypre_CSRBlockMatrixMatvec(alpha, diag, x_local, beta, y_local);
   if (nprocs > 1)
   {
      hypre_ParCSRBlockCommHandleDestroy(comm_handle);
      comm_handle = NULL;
      if (num_cols_offd) 
         hypre_CSRBlockMatrixMatvec(alpha,offd,x_tmp,1.0,y_local);    
      hypre_SeqVectorDestroy(x_tmp);
      x_tmp = NULL;
      hypre_TFree(x_buf_data);
   }
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRBlockMatrixMatvecT( double           alpha,
                  hypre_ParCSRBlockMatrix *A,
                  hypre_ParVector    *x,
                  double           beta,
                  hypre_ParVector    *y     )
{
   hypre_ParCSRCommHandle	*comm_handle;
   hypre_ParCSRCommPkg	*comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   hypre_CSRBlockMatrix *diag = hypre_ParCSRBlockMatrixDiag(A);
   hypre_CSRBlockMatrix *offd = hypre_ParCSRBlockMatrixOffd(A);
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
   hypre_Vector *y_tmp;

   double       *y_local_data;
   int         blk_size = hypre_ParCSRBlockMatrixBlockSize(A);
   int         x_size = hypre_ParVectorGlobalSize(x);
   int         y_size = hypre_ParVectorGlobalSize(y);
   double       *y_tmp_data, *y_buf_data;
   

   int         num_rows  = hypre_ParCSRBlockMatrixGlobalNumRows(A);
   int         num_cols  = hypre_ParCSRBlockMatrixGlobalNumCols(A);
   int	       num_cols_offd = hypre_CSRBlockMatrixNumCols(offd);


   int         i, j, index, start, finish, elem, num_sends;
   int         size, k;
   

   int         ierr  = 0;

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
 
    if (num_rows*blk_size != x_size)
              ierr = 1;

    if (num_cols*blk_size != y_size)
              ierr = 2;

    if (num_rows*blk_size != x_size && num_cols*blk_size != y_size)
              ierr = 3;
   /*-----------------------------------------------------------------------
    *-----------------------------------------------------------------------*/


    y_tmp = hypre_SeqVectorCreate(num_cols_offd*blk_size);
    hypre_SeqVectorInitialize(y_tmp);


   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
      hypre_BlockNewCommPkgCreate(A);
#else
      hypre_BlockMatvecCommPkgCreate(A);
#endif
      comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   size = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends)*blk_size;
   y_buf_data = hypre_CTAlloc(double, size);

   y_tmp_data = hypre_VectorData(y_tmp);
   y_local_data = hypre_VectorData(y_local);
  
   if (num_cols_offd) hypre_CSRBlockMatrixMatvecT(alpha, offd, x_local, 0.0, y_tmp);

   comm_handle = hypre_ParCSRBlockCommHandleCreate
      ( 2, blk_size, comm_pkg, y_tmp_data, y_buf_data);

  
   hypre_CSRBlockMatrixMatvecT(alpha, diag, x_local, beta, y_local);


   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      finish = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1);
      
      for (j = start; j < finish; j++)
      {
         elem =  hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)*blk_size;
         for (k = 0; k < blk_size; k++)
         {
            y_local_data[elem++]
               += y_buf_data[index++];
         }
      }
   }
   
   hypre_TFree(y_buf_data);

	
   hypre_SeqVectorDestroy(y_tmp);
   y_tmp = NULL;
   
   return ierr;
}
