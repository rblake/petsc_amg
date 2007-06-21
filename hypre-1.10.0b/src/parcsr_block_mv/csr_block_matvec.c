/*BHEADER**********************************************************************
 * (c) 2005   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.2 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Matvec functions for hypre_CSRBlockMatrix class.
 *
 *****************************************************************************/

#include "csr_block_matrix.h"
#include "../seq_mv/seq_mv.h"
#include <assert.h>

/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixMatvec
 *--------------------------------------------------------------------------*/

int
hypre_CSRBlockMatrixMatvec(double alpha, hypre_CSRBlockMatrix *A,
                           hypre_Vector *x, double beta, hypre_Vector *y)
{
   double     *A_data   = hypre_CSRBlockMatrixData(A);
   int        *A_i      = hypre_CSRBlockMatrixI(A);
   int        *A_j      = hypre_CSRBlockMatrixJ(A);
   int         num_rows = hypre_CSRBlockMatrixNumRows(A);
   int         num_cols = hypre_CSRBlockMatrixNumCols(A);
   int         blk_size = hypre_CSRBlockMatrixBlockSize(A);

   double     *x_data = hypre_VectorData(x);
   double     *y_data = hypre_VectorData(y);
   int         x_size = hypre_VectorSize(x);
   int         y_size = hypre_VectorSize(y);

   int         i, b1, b2, jj, bnnz=blk_size*blk_size;
   int         ierr = 0;
   double      temp;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  Matvec returns ierr = 1 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 2 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in Matvec, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
 
    if (num_cols*blk_size != x_size) ierr = 1;
    if (num_rows*blk_size != y_size) ierr = 2;
    if (num_cols*blk_size != x_size && num_rows*blk_size != y_size) ierr = 3;

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
      for (i = 0; i < num_rows*blk_size; i++) y_data[i] *= beta;

      return ierr;
   }

   /*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/
   
   temp = beta / alpha;
   
   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
	 for (i = 0; i < num_rows*blk_size; i++)
	    y_data[i] = 0.0;
      }
      else
      {
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
	 for (i = 0; i < num_rows*blk_size; i++)
            y_data[i] *= temp;
      }
   }

   /*-----------------------------------------------------------------
    * y += A*x
    *-----------------------------------------------------------------*/

#define HYPRE_SMP_PRIVATE i,jj
#include "../utilities/hypre_smp_forloop.h"

   for (i = 0; i < num_rows; i++)
   {
      for (jj = A_i[i]; jj < A_i[i+1]; jj++)
      {
         for (b1 = 0; b1 < blk_size; b1++)
         {
            temp = y_data[i*blk_size+b1];
            for (b2 = 0; b2 < blk_size; b2++)
               temp += A_data[jj*bnnz+b1*blk_size+b2] * x_data[A_j[jj]*blk_size+b2];
            y_data[i*blk_size+b1] = temp;
         }
      }
   }

   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/

   if (alpha != 1.0)
   {
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
      for (i = 0; i < num_rows*blk_size; i++)
	 y_data[i] *= alpha;
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *   From Van Henson's modification of hypre_CSRMatrixMatvec.
 *--------------------------------------------------------------------------*/

int
hypre_CSRBlockMatrixMatvecT( double               alpha,
                             hypre_CSRBlockMatrix *A,
                             hypre_Vector         *x,
                             double                beta,
                             hypre_Vector          *y     )
{
   double     *A_data    = hypre_CSRBlockMatrixData(A);
   int        *A_i       = hypre_CSRBlockMatrixI(A);
   int        *A_j       = hypre_CSRBlockMatrixJ(A);
   int         num_rows  = hypre_CSRBlockMatrixNumRows(A);
   int         num_cols  = hypre_CSRBlockMatrixNumCols(A);

   double     *x_data = hypre_VectorData(x);
   double     *y_data = hypre_VectorData(y);
   int         x_size = hypre_VectorSize(x);
   int         y_size = hypre_VectorSize(y);

   double      temp;
   
   int         i, j, jj;
   int         ierr  = 0;
   int         b1, b2;
   
   int         blk_size = hypre_CSRBlockMatrixBlockSize(A);
   int         bnnz=blk_size*blk_size;

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
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
      for (i = 0; i < num_cols*blk_size; i++)
	 y_data[i] *= beta;

      return ierr;
   }

   /*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/

   temp = beta / alpha;
   
   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
	 for (i = 0; i < num_cols*blk_size; i++)
	    y_data[i] = 0.0;
      }
      else
      {
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
	 for (i = 0; i < num_cols*blk_size; i++)
	    y_data[i] *= temp;
      }
   }

   /*-----------------------------------------------------------------
    * y += A^T*x
    *-----------------------------------------------------------------*/
 
#define HYPRE_SMP_PRIVATE i, i1,jj,j,ns,ne,size,rest
#include "../utilities/hypre_smp_forloop.h"
    
   for (i = 0; i < num_rows; i++)
   {
      for (jj = A_i[i]; jj < A_i[i+1]; jj++) /*each nonzero in that row*/
      {
         for (b1 = 0; b1 < blk_size; b1++) /*row */
         {
            for (b2 = 0; b2 < blk_size; b2++) /*col*/
            {
               j = A_j[jj]; /*col */
               y_data[j*blk_size+b2] += A_data[jj*bnnz+b1*blk_size+b2] * x_data[i*blk_size + b1];
            }
         }
      }
   }
      
   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/

   if (alpha != 1.0)
   {
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
      for (i = 0; i < num_cols*blk_size; i++)
	 y_data[i] *= alpha;
   }

   return ierr;
}

