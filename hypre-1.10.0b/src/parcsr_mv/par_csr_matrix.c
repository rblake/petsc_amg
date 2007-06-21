/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.12 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Member functions for hypre_ParCSRMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

#include "../seq_mv/HYPRE_seq_mv.h"
/* In addition to publically accessible interface in HYPRE_mv.h, the implementation
   in this file uses accessor macros into the sequential matrix structure, and
   so includes the .h that defines that structure. Should those accessor functions
   become proper functions at some later date, this will not be necessary. AJC 4/99
*/
#include "../seq_mv/csr_matrix.h"


#ifdef HYPRE_NO_GLOBAL_PARTITION
int hypre_FillResponseParToCSRMatrix(void*, int, int, void*, MPI_Comm, void**, int*);
#endif

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixCreate
 *--------------------------------------------------------------------------*/

/* If create is called for HYPRE_NO_GLOBAL_PARTITION and row_starts and col_starts are NOT null,
   then it is assumed that they are array of length 2 containing the start row of 
   the calling processor followed by the start row of the next processor - AHB 6/05 */

hypre_ParCSRMatrix *
hypre_ParCSRMatrixCreate( MPI_Comm comm,
                          int global_num_rows,
                          int global_num_cols,
                          int *row_starts,
                          int *col_starts,
                          int num_cols_offd,
                          int num_nonzeros_diag,
                          int num_nonzeros_offd)
{
   hypre_ParCSRMatrix  *matrix;
   int  num_procs, my_id;
   int local_num_rows, local_num_cols;
   int first_row_index, first_col_diag;
   
   matrix = hypre_CTAlloc(hypre_ParCSRMatrix, 1);

   MPI_Comm_rank(comm,&my_id);
   MPI_Comm_size(comm,&num_procs);

   if (!row_starts)
   {
    
#ifdef HYPRE_NO_GLOBAL_PARTITION  
      hypre_GenerateLocalPartitioning(global_num_rows, num_procs, my_id, &row_starts);
#else
      hypre_GeneratePartitioning(global_num_rows,num_procs,&row_starts);
#endif
   }
   if (!col_starts)
   {
      if (global_num_rows == global_num_cols)
      {
        col_starts = row_starts;
      }
      else
      {
#ifdef HYPRE_NO_GLOBAL_PARTITION   
      hypre_GenerateLocalPartitioning(global_num_cols, num_procs, my_id, &col_starts);
#else
      hypre_GeneratePartitioning(global_num_cols,num_procs,&col_starts);
#endif
      }
   }

#ifdef HYPRE_NO_GLOBAL_PARTITION
   /* row_starts[0] is start of local rows.  row_starts[1] is start of next 
      processor's rows */
   first_row_index = row_starts[0];
   local_num_rows = row_starts[1]-first_row_index ;
   first_col_diag = col_starts[0];
   local_num_cols = col_starts[1]-first_col_diag;
#else
   first_row_index = row_starts[my_id];
   local_num_rows = row_starts[my_id+1]-first_row_index;
   first_col_diag = col_starts[my_id];
   local_num_cols = col_starts[my_id+1]-first_col_diag;
#endif


   hypre_ParCSRMatrixComm(matrix) = comm;
   hypre_ParCSRMatrixDiag(matrix) = hypre_CSRMatrixCreate(local_num_rows,
                local_num_cols,num_nonzeros_diag);
   hypre_ParCSRMatrixOffd(matrix) = hypre_CSRMatrixCreate(local_num_rows,
                num_cols_offd,num_nonzeros_offd);
   hypre_ParCSRMatrixGlobalNumRows(matrix) = global_num_rows;
   hypre_ParCSRMatrixGlobalNumCols(matrix) = global_num_cols;
   hypre_ParCSRMatrixFirstRowIndex(matrix) = first_row_index;
   hypre_ParCSRMatrixFirstColDiag(matrix) = first_col_diag;
 
   hypre_ParCSRMatrixLastRowIndex(matrix) = first_row_index + local_num_rows - 1;
   hypre_ParCSRMatrixLastColDiag(matrix) = first_col_diag + local_num_cols - 1;

   hypre_ParCSRMatrixColMapOffd(matrix) = NULL;

   /* When NO_GLOBAL_PARTITION is set we could make these null, instead
      of leaving the range.  If that change is made, then when this create
      is called from functions like the matrix-matrix multiply, be careful
      not to generate a new partition */

   hypre_ParCSRMatrixRowStarts(matrix) = row_starts;
   hypre_ParCSRMatrixColStarts(matrix) = col_starts;

   hypre_ParCSRMatrixCommPkg(matrix) = NULL;
   hypre_ParCSRMatrixCommPkgT(matrix) = NULL;

   /* set defaults */
   hypre_ParCSRMatrixOwnsData(matrix) = 1;
   hypre_ParCSRMatrixOwnsRowStarts(matrix) = 1;
   hypre_ParCSRMatrixOwnsColStarts(matrix) = 1;
   if (row_starts == col_starts)
        hypre_ParCSRMatrixOwnsColStarts(matrix) = 0;
   hypre_ParCSRMatrixRowindices(matrix) = NULL;
   hypre_ParCSRMatrixRowvalues(matrix) = NULL;
   hypre_ParCSRMatrixGetrowactive(matrix) = 0;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_ParCSRMatrixDestroy( hypre_ParCSRMatrix *matrix )
{
   int  ierr=0;

   if (matrix)
   {
      if ( hypre_ParCSRMatrixOwnsData(matrix) )
      {
         hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(matrix));
         hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(matrix));
         if (hypre_ParCSRMatrixColMapOffd(matrix))
              hypre_TFree(hypre_ParCSRMatrixColMapOffd(matrix));
         if (hypre_ParCSRMatrixCommPkg(matrix))
              hypre_MatvecCommPkgDestroy(hypre_ParCSRMatrixCommPkg(matrix));
         if (hypre_ParCSRMatrixCommPkgT(matrix))
              hypre_MatvecCommPkgDestroy(hypre_ParCSRMatrixCommPkgT(matrix));
      }
      if ( hypre_ParCSRMatrixOwnsRowStarts(matrix) )
              hypre_TFree(hypre_ParCSRMatrixRowStarts(matrix));
      if ( hypre_ParCSRMatrixOwnsColStarts(matrix) )
              hypre_TFree(hypre_ParCSRMatrixColStarts(matrix));

      hypre_TFree(hypre_ParCSRMatrixRowindices(matrix));
      hypre_TFree(hypre_ParCSRMatrixRowvalues(matrix));

      hypre_TFree(matrix);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_ParCSRMatrixInitialize( hypre_ParCSRMatrix *matrix )
{
   int  ierr=0;

   hypre_CSRMatrixInitialize(hypre_ParCSRMatrixDiag(matrix));
   hypre_CSRMatrixInitialize(hypre_ParCSRMatrixOffd(matrix));
   hypre_ParCSRMatrixColMapOffd(matrix) = 
                hypre_CTAlloc(int,hypre_CSRMatrixNumCols(
                hypre_ParCSRMatrixOffd(matrix)));
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixSetNumNonzeros
 *--------------------------------------------------------------------------*/

int 
hypre_ParCSRMatrixSetNumNonzeros( hypre_ParCSRMatrix *matrix)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(matrix);
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(matrix);
   int *diag_i = hypre_CSRMatrixI(diag);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(matrix);
   int *offd_i = hypre_CSRMatrixI(offd);
   int local_num_rows = hypre_CSRMatrixNumRows(diag);
   int total_num_nonzeros;
   int local_num_nonzeros;
   int ierr = 0;

   local_num_nonzeros = diag_i[local_num_rows] + offd_i[local_num_rows];
   MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1, MPI_INT,
        MPI_SUM, comm);
   hypre_ParCSRMatrixNumNonzeros(matrix) = total_num_nonzeros;
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixSetDNumNonzeros
 *--------------------------------------------------------------------------*/

int 
hypre_ParCSRMatrixSetDNumNonzeros( hypre_ParCSRMatrix *matrix)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(matrix);
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(matrix);
   int *diag_i = hypre_CSRMatrixI(diag);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(matrix);
   int *offd_i = hypre_CSRMatrixI(offd);
   int local_num_rows = hypre_CSRMatrixNumRows(diag);
   double total_num_nonzeros;
   double local_num_nonzeros;
   int ierr = 0;

   local_num_nonzeros = (double) diag_i[local_num_rows] 
		+ (double) offd_i[local_num_rows];
   MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1, MPI_DOUBLE,
        MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matrix) = total_num_nonzeros;
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

int 
hypre_ParCSRMatrixSetDataOwner( hypre_ParCSRMatrix *matrix,
                                int              owns_data )
{
   int    ierr=0;

   hypre_ParCSRMatrixOwnsData(matrix) = owns_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixSetRowStartsOwner
 *--------------------------------------------------------------------------*/

int 
hypre_ParCSRMatrixSetRowStartsOwner( hypre_ParCSRMatrix *matrix,
                                     int owns_row_starts )
{
   int    ierr=0;

   hypre_ParCSRMatrixOwnsRowStarts(matrix) = owns_row_starts;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixSetColStartsOwner
 *--------------------------------------------------------------------------*/

int 
hypre_ParCSRMatrixSetColStartsOwner( hypre_ParCSRMatrix *matrix,
                                     int owns_col_starts )
{
   int    ierr=0;

   hypre_ParCSRMatrixOwnsColStarts(matrix) = owns_col_starts;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixRead
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix *
hypre_ParCSRMatrixRead( MPI_Comm    comm,
                        const char *file_name )
{
   hypre_ParCSRMatrix  *matrix;
   hypre_CSRMatrix  *diag;
   hypre_CSRMatrix  *offd;
   int  my_id, i, num_procs;
   char new_file_d[80], new_file_o[80], new_file_info[80];
   int  global_num_rows, global_num_cols, num_cols_offd;
   int  local_num_rows;
   int  *row_starts;
   int  *col_starts;
   int  *col_map_offd;
   FILE *fp;
   int   equal = 1;
#ifdef HYPRE_NO_GLOBAL_PARTITION
   int   row_s, row_e, col_s, col_e;
#endif
   

   MPI_Comm_rank(comm,&my_id);
   MPI_Comm_size(comm,&num_procs);
#ifdef HYPRE_NO_GLOBAL_PARTITION
   row_starts = hypre_CTAlloc(int, 2);
   col_starts =  hypre_CTAlloc(int, 2);
#else
   row_starts = hypre_CTAlloc(int, num_procs+1);
   col_starts = hypre_CTAlloc(int, num_procs+1);
#endif
   sprintf(new_file_d,"%s.D.%d",file_name,my_id);
   sprintf(new_file_o,"%s.O.%d",file_name,my_id);
   sprintf(new_file_info,"%s.INFO.%d",file_name,my_id);
   fp = fopen(new_file_info, "r");
   fscanf(fp, "%d", &global_num_rows);
   fscanf(fp, "%d", &global_num_cols);
   fscanf(fp, "%d", &num_cols_offd);
#ifdef HYPRE_NO_GLOBAL_PARTITION
   /* the bgl input file should only contain the EXACT range for local processor */
   fscanf(fp, "%d %d %d %d", &row_s, &row_e, &col_s, &col_e);
   row_starts[0] = row_s;
   row_starts[1] = row_e;
   col_starts[0] = col_s;
   col_starts[1] = col_e;
   
#else
   for (i=0; i < num_procs; i++)
           fscanf(fp, "%d %d", &row_starts[i], &col_starts[i]);
   row_starts[num_procs] = global_num_rows;
   col_starts[num_procs] = global_num_cols;
#endif

   col_map_offd = hypre_CTAlloc(int, num_cols_offd);

   for (i=0; i < num_cols_offd; i++)
        fscanf(fp, "%d", &col_map_offd[i]);
        
   fclose(fp);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   for (i=1; i >= 0; i--)
      if (row_starts[i] != col_starts[i])
      {
         equal = 0;
         break;
      }
#else
   for (i=num_procs; i >= 0; i--)
        if (row_starts[i] != col_starts[i])
        {
                equal = 0;
                break;
        }
#endif
   if (equal)
   {
        hypre_TFree(col_starts);
        col_starts = row_starts;
   }




   diag = hypre_CSRMatrixRead(new_file_d);
   local_num_rows = hypre_CSRMatrixNumRows(diag);

   if (num_cols_offd)
   {
        offd = hypre_CSRMatrixRead(new_file_o);
   }
   else
   {
        offd = hypre_CSRMatrixCreate(local_num_rows,0,0);
        hypre_CSRMatrixInitialize(offd);
   }

        
   matrix = hypre_CTAlloc(hypre_ParCSRMatrix, 1);
   
   hypre_ParCSRMatrixComm(matrix) = comm;
   hypre_ParCSRMatrixGlobalNumRows(matrix) = global_num_rows;
   hypre_ParCSRMatrixGlobalNumCols(matrix) = global_num_cols;
#ifdef HYPRE_NO_GLOBAL_PARTITION
   hypre_ParCSRMatrixFirstRowIndex(matrix) = row_s;
   hypre_ParCSRMatrixFirstColDiag(matrix) = col_s;
   hypre_ParCSRMatrixLastRowIndex(matrix) = row_e - 1;
   hypre_ParCSRMatrixLastColDiag(matrix) = col_e - 1;
#else
   hypre_ParCSRMatrixFirstRowIndex(matrix) = row_starts[my_id];
   hypre_ParCSRMatrixFirstColDiag(matrix) = col_starts[my_id];
   hypre_ParCSRMatrixLastRowIndex(matrix) = row_starts[my_id+1]-1;
   hypre_ParCSRMatrixLastColDiag(matrix) = col_starts[my_id+1]-1;
#endif

   hypre_ParCSRMatrixRowStarts(matrix) = row_starts;
   hypre_ParCSRMatrixColStarts(matrix) = col_starts;
   hypre_ParCSRMatrixCommPkg(matrix) = NULL;

   /* set defaults */
   hypre_ParCSRMatrixOwnsData(matrix) = 1;
   hypre_ParCSRMatrixOwnsRowStarts(matrix) = 1;
   hypre_ParCSRMatrixOwnsColStarts(matrix) = 1;
   if (row_starts == col_starts)
        hypre_ParCSRMatrixOwnsColStarts(matrix) = 0;

   hypre_ParCSRMatrixDiag(matrix) = diag;
   hypre_ParCSRMatrixOffd(matrix) = offd;
   if (num_cols_offd)
        hypre_ParCSRMatrixColMapOffd(matrix) = col_map_offd;
   else
        hypre_ParCSRMatrixColMapOffd(matrix) = NULL;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixPrint
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRMatrixPrint( hypre_ParCSRMatrix *matrix, 
                         const char         *file_name )
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(matrix);
   int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(matrix);
   int global_num_cols = hypre_ParCSRMatrixGlobalNumCols(matrix);
   int *col_map_offd = hypre_ParCSRMatrixColMapOffd(matrix);
#ifndef HYPRE_NO_GLOBAL_PARTITION
   int *row_starts = hypre_ParCSRMatrixRowStarts(matrix);
   int *col_starts = hypre_ParCSRMatrixColStarts(matrix);
#endif
   int  my_id, i, num_procs;
   char   new_file_d[80], new_file_o[80], new_file_info[80];
   int  ierr = 0;
   FILE *fp;
   int num_cols_offd = 0;
#ifdef HYPRE_NO_GLOBAL_PARTITION
   int row_s, row_e, col_s, col_e;
#endif
   if (hypre_ParCSRMatrixOffd(matrix))
        num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(matrix));

   MPI_Comm_rank(comm, &my_id);
   MPI_Comm_size(comm, &num_procs);
   
   sprintf(new_file_d,"%s.D.%d",file_name,my_id);
   sprintf(new_file_o,"%s.O.%d",file_name,my_id);
   sprintf(new_file_info,"%s.INFO.%d",file_name,my_id);
   hypre_CSRMatrixPrint(hypre_ParCSRMatrixDiag(matrix),new_file_d);
   if (num_cols_offd != 0)
        hypre_CSRMatrixPrint(hypre_ParCSRMatrixOffd(matrix),new_file_o);
  
   fp = fopen(new_file_info, "w");
   fprintf(fp, "%d\n", global_num_rows);
   fprintf(fp, "%d\n", global_num_cols);
   fprintf(fp, "%d\n", num_cols_offd);
#ifdef HYPRE_NO_GLOBAL_PARTITION
   row_s = hypre_ParCSRMatrixFirstRowIndex(matrix);
   row_e = hypre_ParCSRMatrixLastRowIndex(matrix);
   col_s =  hypre_ParCSRMatrixFirstColDiag(matrix);
   col_e =  hypre_ParCSRMatrixLastColDiag(matrix);
   /* add 1 to the ends because this is a starts partition */
   fprintf(fp, "%d %d %d %d\n", row_s, row_e + 1, col_s, col_e + 1);
#else
   for (i=0; i < num_procs; i++)
        fprintf(fp, "%d %d\n", row_starts[i], col_starts[i]);
#endif
   for (i=0; i < num_cols_offd; i++)
        fprintf(fp, "%d\n", col_map_offd[i]);
   fclose(fp);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixPrintIJ
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRMatrixPrintIJ( hypre_ParCSRMatrix *matrix, 
                           int		       base_i,
                           int		       base_j,
                           const char         *filename )
{
   int ierr = 0;
   MPI_Comm          comm = hypre_ParCSRMatrixComm(matrix);
   int               global_num_rows = hypre_ParCSRMatrixGlobalNumRows(matrix);
   int               global_num_cols = hypre_ParCSRMatrixGlobalNumCols(matrix);
   int               first_row_index = hypre_ParCSRMatrixFirstRowIndex(matrix);
   int               first_col_diag  = hypre_ParCSRMatrixFirstColDiag(matrix);
   hypre_CSRMatrix  *diag            = hypre_ParCSRMatrixDiag(matrix);
   hypre_CSRMatrix  *offd            = hypre_ParCSRMatrixOffd(matrix);
   int              *col_map_offd    = hypre_ParCSRMatrixColMapOffd(matrix);
   int               num_rows        = hypre_ParCSRMatrixNumRows(matrix);
   int              *row_starts      = hypre_ParCSRMatrixRowStarts(matrix);
   int              *col_starts      = hypre_ParCSRMatrixColStarts(matrix);
   double           *diag_data;
   int              *diag_i;
   int              *diag_j;
   double           *offd_data;
   int              *offd_i;
   int              *offd_j;
   int               myid, num_procs, i, j, I, J;
   char              new_filename[255];
   FILE             *file;
   int num_cols_offd, num_nonzeros_diag, num_nonzeros_offd;
   int num_cols, row, col;

   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);
   
   sprintf(new_filename,"%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   num_cols = hypre_CSRMatrixNumCols(diag);
   num_cols_offd = hypre_CSRMatrixNumCols(offd);
   num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(offd);
   num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(diag);

   diag_data = hypre_CSRMatrixData(diag);
   diag_i    = hypre_CSRMatrixI(diag);
   diag_j    = hypre_CSRMatrixJ(diag);
   offd_i    = hypre_CSRMatrixI(offd);
   if (num_nonzeros_offd)
   {
      offd_data = hypre_CSRMatrixData(offd);
      offd_j    = hypre_CSRMatrixJ(offd);
   }

   fprintf(file, "%d %d\n", global_num_rows, global_num_cols);
   fprintf(file, "%d %d %d\n", num_rows, num_cols, num_cols_offd);
   fprintf(file, "%d %d\n", num_nonzeros_diag, num_nonzeros_offd);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   for (i=0; i <= 1; i++)
   {
	row = row_starts[i]+base_i;
	col = col_starts[i]+base_j;
        fprintf(file, "%d %d\n", row, col);
   }
#else
   for (i=0; i <= num_procs; i++)
   {
	row = row_starts[i]+base_i;
	col = col_starts[i]+base_j;
        fprintf(file, "%d %d\n", row, col);
   }
#endif
   for (i = 0; i < num_rows; i++)
   {
      I = first_row_index + i + base_i;

      /* print diag columns */
      for (j = diag_i[i]; j < diag_i[i+1]; j++)
      {
         J = first_col_diag + diag_j[j] + base_j;
         fprintf(file, "%d %d %e\n", I, J, diag_data[j]);
      }

      /* print offd columns */
      if ( num_nonzeros_offd)
      {
         for (j = offd_i[i]; j < offd_i[i+1]; j++)
         {
            J = col_map_offd[offd_j[j]] + base_j;
            fprintf(file, "%d %d %e\n", I, J, offd_data[j]);
         }
      }
   }

   fclose(file);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixReadIJ
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRMatrixReadIJ( MPI_Comm 	       comm,
			  const char          *filename,
			  int		      *base_i_ptr,
			  int		      *base_j_ptr,
			  hypre_ParCSRMatrix **matrix_ptr) 
{
   int ierr = 0;
   int               global_num_rows;
   int               global_num_cols;
   int               first_row_index;
   int               first_col_diag;
   int               last_col_diag;
   hypre_ParCSRMatrix *matrix;
   hypre_CSRMatrix  *diag;
   hypre_CSRMatrix  *offd;
   int              *col_map_offd;
   int              *row_starts;
   int              *col_starts;
   int               num_rows;
   int               base_i, base_j;
   double           *diag_data;
   int              *diag_i;
   int              *diag_j;
   double           *offd_data;
   int              *offd_i;
   int              *offd_j;
   int              *aux_offd_j;
   int               myid, num_procs, i, j, I, J;
   char              new_filename[255];
   FILE             *file;
   int num_cols_offd, num_nonzeros_diag, num_nonzeros_offd;
   int equal, i_col, num_cols;
   int diag_cnt, offd_cnt, row_cnt;
   double data;

   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);
   
   sprintf(new_filename,"%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "r")) == NULL)
   {
      printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   fscanf(file, "%d %d", &global_num_rows, &global_num_cols);
   fscanf(file, "%d %d %d", &num_rows, &num_cols, &num_cols_offd);
   fscanf(file, "%d %d", &num_nonzeros_diag, &num_nonzeros_offd);

   row_starts = hypre_CTAlloc(int,num_procs+1);
   col_starts = hypre_CTAlloc(int,num_procs+1);

   for (i = 0; i <= num_procs; i++)
      fscanf(file, "%d %d", &row_starts[i], &col_starts[i]);

   base_i = row_starts[0];
   base_j = col_starts[0];

   equal = 1;
   for (i = 0; i <= num_procs; i++)
   {
      row_starts[i] -= base_i;      
      col_starts[i] -= base_j;
      if (row_starts[i] != col_starts[i]) equal = 0;
   }

   if (equal)
   {
      hypre_TFree(col_starts);
      col_starts = row_starts;
   }
   matrix = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
				     row_starts, col_starts,
				     num_cols_offd,
				     num_nonzeros_diag, num_nonzeros_offd);
   hypre_ParCSRMatrixInitialize(matrix);
 
   diag = hypre_ParCSRMatrixDiag(matrix);
   offd = hypre_ParCSRMatrixOffd(matrix);

   diag_data = hypre_CSRMatrixData(diag);
   diag_i    = hypre_CSRMatrixI(diag);
   diag_j    = hypre_CSRMatrixJ(diag);

   offd_i    = hypre_CSRMatrixI(offd);
   if (num_nonzeros_offd)
   {
      offd_data = hypre_CSRMatrixData(offd);
      offd_j    = hypre_CSRMatrixJ(offd);
   }

   first_row_index = hypre_ParCSRMatrixFirstRowIndex(matrix);
   first_col_diag = hypre_ParCSRMatrixFirstColDiag(matrix);
   last_col_diag = first_col_diag+num_cols-1;

   diag_cnt = 0;
   offd_cnt = 0;
   row_cnt = 0;
   for (i = 0; i < num_nonzeros_diag+num_nonzeros_offd; i++)
   {
      /* read values */
      fscanf(file, "%d %d %le", &I, &J, &data);
      I = I-base_i-first_row_index;       
      J -= base_j;
      if (I > row_cnt)
      {
	 diag_i[I] = diag_cnt;
	 offd_i[I] = offd_cnt;
	 row_cnt++;
      }
      if (J < first_col_diag || J > last_col_diag)
      {
	 offd_j[offd_cnt] = J;       
	 offd_data[offd_cnt++] = data;
      }
      else       
      {
	 diag_j[diag_cnt] = J - first_col_diag;       
	 diag_data[diag_cnt++] = data;
      }
   }
   diag_i[num_rows] = diag_cnt;
   offd_i[num_rows] = offd_cnt;

   fclose(file);

   /*  generate col_map_offd */
   if (num_nonzeros_offd)
   {
      aux_offd_j = hypre_CTAlloc(int, num_nonzeros_offd);
      for (i=0; i < num_nonzeros_offd; i++)
         aux_offd_j[i] = offd_j[i];
      qsort0(aux_offd_j,0,num_nonzeros_offd-1);
      col_map_offd = hypre_ParCSRMatrixColMapOffd(matrix);
      col_map_offd[0] = aux_offd_j[0];
      offd_cnt = 0;
      for (i=1; i < num_nonzeros_offd; i++)
      {
         if (aux_offd_j[i] > col_map_offd[offd_cnt])
            col_map_offd[++offd_cnt] = aux_offd_j[i];
      }
      for (i=0; i < num_nonzeros_offd; i++)
      {
	 offd_j[i] = hypre_BinarySearch(col_map_offd, offd_j[i], num_cols_offd);
      }
      hypre_TFree(aux_offd_j);
   }

   /* move diagonal element in first position in each row */
   for (i=0; i < num_rows; i++)
   {
      i_col = diag_i[i];
      for (j=i_col; j < diag_i[i+1]; j++)
      {
	 if (diag_j[j] == i)
	 {
	    diag_j[j] = diag_j[i_col];
	    data = diag_data[j];
	    diag_data[j] = diag_data[i_col];
	    diag_data[i_col] = data;
	    diag_j[i_col] = i;
      	    break;
	 }
      }
   }
	  
   *base_i_ptr = base_i;
   *base_j_ptr = base_j;
   *matrix_ptr = matrix;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixGetLocalRange
 * returns the row numbers of the rows stored on this processor.
 * "End" is actually the row number of the last row on this processor.
 *--------------------------------------------------------------------------*/

int 
hypre_ParCSRMatrixGetLocalRange( hypre_ParCSRMatrix *matrix,
                         int               *row_start,
                         int               *row_end,
                         int               *col_start,
                         int               *col_end )
{  
   int ierr=0;
   int my_id;

   MPI_Comm_rank( hypre_ParCSRMatrixComm(matrix), &my_id );

#ifdef HYPRE_NO_GLOBAL_PARTITION
  *row_start = hypre_ParCSRMatrixFirstRowIndex(matrix);
  *row_end = hypre_ParCSRMatrixLastRowIndex(matrix);
  *col_start =  hypre_ParCSRMatrixFirstColDiag(matrix);
  *col_end =  hypre_ParCSRMatrixLastColDiag(matrix);
#else
   *row_start = hypre_ParCSRMatrixRowStarts(matrix)[ my_id ];
   *row_end = hypre_ParCSRMatrixRowStarts(matrix)[ my_id + 1 ]-1;
   *col_start = hypre_ParCSRMatrixColStarts(matrix)[ my_id ];
   *col_end = hypre_ParCSRMatrixColStarts(matrix)[ my_id + 1 ]-1;
#endif

   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixGetRow
 * Returns global column indices and/or values for a given row in the global
 * matrix. Global row number is used, but the row must be stored locally or
 * an error is returned. This implementation copies from the two matrices that
 * store the local data, storing them in the hypre_ParCSRMatrix structure.
 * Only a single row can be accessed via this function at any one time; the
 * corresponding RestoreRow function must be called, to avoid bleeding memory,
 * and to be able to look at another row.
 * Either one of col_ind and values can be left null, and those values will
 * not be returned.
 * All indices are returned in 0-based indexing, no matter what is used under
 * the hood. EXCEPTION: currently this only works if the local CSR matrices
 * use 0-based indexing.
 * This code, semantics, implementation, etc., are all based on PETSc's MPI_AIJ
 * matrix code, adjusted for our data and software structures.
 * AJC 4/99.
 *--------------------------------------------------------------------------*/

int 
hypre_ParCSRMatrixGetRow( hypre_ParCSRMatrix  *mat,
                         int                row,
                         int               *size,
                         int               **col_ind,
                         double            **values )
{  
   int ierr=0;
   int my_id;
   int row_start, row_end;
   hypre_CSRMatrix *Aa = (hypre_CSRMatrix *) hypre_ParCSRMatrixDiag(mat);
   hypre_CSRMatrix *Ba = (hypre_CSRMatrix *) hypre_ParCSRMatrixOffd(mat);
   

   if (hypre_ParCSRMatrixGetrowactive(mat)) return(-1);

   MPI_Comm_rank( hypre_ParCSRMatrixComm(mat), &my_id );

   hypre_ParCSRMatrixGetrowactive(mat) = 1;
#ifdef HYPRE_NO_GLOBAL_PARTITION
   row_start = hypre_ParCSRMatrixFirstRowIndex(mat);
   row_end = hypre_ParCSRMatrixLastRowIndex(mat) + 1;
#else
   row_end = hypre_ParCSRMatrixRowStarts(mat)[ my_id + 1 ];
   row_start = hypre_ParCSRMatrixRowStarts(mat)[ my_id ];
#endif
   if (row < row_start || row >= row_end) return(-1);

   /* if buffer is not allocated and some information is requested,
      allocate buffer */
   if (!hypre_ParCSRMatrixRowvalues(mat) && ( col_ind || values )) 
   {
    /*
        allocate enough space to hold information from the longest row.
    */
     int     max = 1,tmp;
     int i;
     int     m = row_end-row_start;

     for ( i=0; i<m; i++ ) {
       tmp = hypre_CSRMatrixI(Aa)[i+1] - hypre_CSRMatrixI(Aa)[i] + 
             hypre_CSRMatrixI(Ba)[i+1] - hypre_CSRMatrixI(Ba)[i];
       if (max < tmp) { max = tmp; }
     }

     hypre_ParCSRMatrixRowvalues(mat) = (double *) hypre_CTAlloc
                             ( double, max ); 
     hypre_ParCSRMatrixRowindices(mat) = (int *) hypre_CTAlloc
                             ( int, max ); 
   }

   /* Copy from dual sequential matrices into buffer */
   {
   double     *vworkA, *vworkB, *v_p;
   int        i, *cworkA, *cworkB, 
              cstart = hypre_ParCSRMatrixFirstColDiag(mat);
   int        nztot, nzA, nzB, lrow=row-row_start;
   int        *cmap, *idx_p;

   nzA = hypre_CSRMatrixI(Aa)[lrow+1]-hypre_CSRMatrixI(Aa)[lrow];
   cworkA = &( hypre_CSRMatrixJ(Aa)[ hypre_CSRMatrixI(Aa)[lrow] ] );
   vworkA = &( hypre_CSRMatrixData(Aa)[ hypre_CSRMatrixI(Aa)[lrow] ] );

   nzB = hypre_CSRMatrixI(Ba)[lrow+1]-hypre_CSRMatrixI(Ba)[lrow];
   cworkB = &( hypre_CSRMatrixJ(Ba)[ hypre_CSRMatrixI(Ba)[lrow] ] );
   vworkB = &( hypre_CSRMatrixData(Ba)[ hypre_CSRMatrixI(Ba)[lrow] ] );

   nztot = nzA + nzB;

   cmap  = hypre_ParCSRMatrixColMapOffd(mat);

   if (values  || col_ind) {
     if (nztot) {
       /* Sort by increasing column numbers, assuming A and B already sorted */
       int imark = -1;
       if (values) {
         *values = v_p = hypre_ParCSRMatrixRowvalues(mat);
         for ( i=0; i<nzB; i++ ) {
           if (cmap[cworkB[i]] < cstart)   v_p[i] = vworkB[i];
           else break;
         }
         imark = i;
         for ( i=0; i<nzA; i++ )     v_p[imark+i] = vworkA[i];
         for ( i=imark; i<nzB; i++ ) v_p[nzA+i]   = vworkB[i];
       }
       if (col_ind) {
         *col_ind = idx_p = hypre_ParCSRMatrixRowindices(mat);
         if (imark > -1) {
           for ( i=0; i<imark; i++ ) {
             idx_p[i] = cmap[cworkB[i]];
           }
         } else {
           for ( i=0; i<nzB; i++ ) {
             if (cmap[cworkB[i]] < cstart)   idx_p[i] = cmap[cworkB[i]];
             else break;
           }
           imark = i;
         }
         for ( i=0; i<nzA; i++ )     idx_p[imark+i] = cstart + cworkA[i];
         for ( i=imark; i<nzB; i++ ) idx_p[nzA+i]   = cmap[cworkB[i]];
       } 
     } 
     else {
       if (col_ind) *col_ind = 0; 
       if (values)   *values   = 0;
     }
   }
   *size = nztot;

   } /* End of copy */


   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixRestoreRow
 *--------------------------------------------------------------------------*/

int 
hypre_ParCSRMatrixRestoreRow( hypre_ParCSRMatrix *matrix,
                         int                row,
                         int               *size,
                         int               **col_ind,
                         double            **values )
{  
   int ierr=0;

  if (!hypre_ParCSRMatrixGetrowactive(matrix)) {
    return( -1 );
  }

  hypre_ParCSRMatrixGetrowactive(matrix)=0;

   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixToParCSRMatrix:
 * generates a ParCSRMatrix distributed across the processors in comm
 * from a CSRMatrix on proc 0 .
 *
 * This shouldn't be used with the HYPRE_NO_GLOBAL_PARTITON option 
 *
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix *
hypre_CSRMatrixToParCSRMatrix( MPI_Comm comm, hypre_CSRMatrix *A,
                               int *row_starts,
                               int *col_starts )
{
   int          *global_data;
   int          global_size;
   int          global_num_rows;
   int          global_num_cols;
   int          *local_num_rows;

   int          num_procs, my_id;
   int          *local_num_nonzeros;
   int          num_nonzeros;
  
   double       *a_data;
   int          *a_i;
   int          *a_j;
  
   hypre_CSRMatrix *local_A;

   MPI_Request  *requests;
   MPI_Status   *status, status0;
   MPI_Datatype *csr_matrix_datatypes;

   hypre_ParCSRMatrix *par_matrix;

   int          first_col_diag;
   int          last_col_diag;
 
   int i, j, ind;

   MPI_Comm_rank(comm, &my_id);
   MPI_Comm_size(comm, &num_procs);

   global_data = hypre_CTAlloc(int, 2*num_procs+6);
   if (my_id == 0) 
   {
      global_size = 3;
      if (row_starts) 
      {
	 if (col_starts)
	 {
	    if (col_starts != row_starts)
	    {
            /* contains code for what to expect, 
		 if 0:  row_starts = col_starts, only row_starts given
		 if 1: only row_starts given, col_starts = NULL
		 if 2: both row_starts and col_starts given 
		 if 3: only col_starts given, row_starts = NULL */
               global_data[3] = 2;
	       global_size = 2*num_procs+6;
	       for (i=0; i < num_procs+1; i++)
	    	  global_data[i+4] = row_starts[i];
	       for (i=0; i < num_procs+1; i++)
		  global_data[i+num_procs+5] = col_starts[i];
	    }
	    else
	    {
               global_data[3] = 0;
	       global_size = num_procs+5;
	       for (i=0; i < num_procs+1; i++)
		  global_data[i+4] = row_starts[i];
	    }
	 }
	 else
	 {
            global_data[3] = 1;
	    global_size = num_procs+5;
	    for (i=0; i < num_procs+1; i++)
	       global_data[i+4] = row_starts[i];
	 }
      }
      else 
      {
	 if (col_starts)
	 {
            global_data[3] = 3;
	    global_size = num_procs+5;
	    for (i=0; i < num_procs+1; i++)
	       global_data[i+4] = col_starts[i];
	 }
      }
      global_data[0] = hypre_CSRMatrixNumRows(A);
      global_data[1] = hypre_CSRMatrixNumCols(A);
      global_data[2] = global_size;
      a_data = hypre_CSRMatrixData(A);
      a_i = hypre_CSRMatrixI(A);
      a_j = hypre_CSRMatrixJ(A);
   }
   MPI_Bcast(global_data,3,MPI_INT,0,comm);
   global_num_rows = global_data[0];
   global_num_cols = global_data[1];

   global_size = global_data[2];
   if (global_size > 3)
   {
      MPI_Bcast(&global_data[3],global_size-3,MPI_INT,0,comm);
      if (my_id > 0)
      {
	 if (global_data[3] < 3)
	 {
	    row_starts = hypre_CTAlloc(int, num_procs+1);
	    for (i=0; i< num_procs+1; i++)
	    {
	       row_starts[i] = global_data[i+4];
	    }
	    if (global_data[3] == 0)
	       col_starts = row_starts;
	    if (global_data[3] == 2)
	    {
	       col_starts = hypre_CTAlloc(int, num_procs+1);
	       for (i=0; i < num_procs+1; i++)
	       {
	          col_starts[i] = global_data[i+num_procs+5];
	       }
	    }
	 }
	 else
	 {
	    col_starts = hypre_CTAlloc(int, num_procs+1);
	    for (i=0; i< num_procs+1; i++)
	    {
	       col_starts[i] = global_data[i+4];
	    }
	 }
      }
   }
   hypre_TFree(global_data);

   local_num_rows = hypre_CTAlloc(int, num_procs);
   csr_matrix_datatypes = hypre_CTAlloc(MPI_Datatype, num_procs);

   par_matrix = hypre_ParCSRMatrixCreate (comm, global_num_rows,
        global_num_cols,row_starts,col_starts,0,0,0);

   row_starts = hypre_ParCSRMatrixRowStarts(par_matrix);
   col_starts = hypre_ParCSRMatrixColStarts(par_matrix);

   for (i=0; i < num_procs; i++)
         local_num_rows[i] = row_starts[i+1] - row_starts[i];

   if (my_id == 0)
   {
        local_num_nonzeros = hypre_CTAlloc(int, num_procs);
        for (i=0; i < num_procs-1; i++)
                local_num_nonzeros[i] = a_i[row_starts[i+1]] 
                                - a_i[row_starts[i]];
        local_num_nonzeros[num_procs-1] = a_i[global_num_rows] 
                                - a_i[row_starts[num_procs-1]];
   }
   MPI_Scatter(local_num_nonzeros,1,MPI_INT,&num_nonzeros,1,MPI_INT,0,comm);

   if (my_id == 0) num_nonzeros = local_num_nonzeros[0];

   local_A = hypre_CSRMatrixCreate(local_num_rows[my_id], global_num_cols,
                num_nonzeros);
   if (my_id == 0)
   {
        requests = hypre_CTAlloc (MPI_Request, num_procs-1);
        status = hypre_CTAlloc(MPI_Status, num_procs-1);
        j=0;
        for (i=1; i < num_procs; i++)
        {
                ind = a_i[row_starts[i]];
                hypre_BuildCSRMatrixMPIDataType(local_num_nonzeros[i], 
                        local_num_rows[i],
                        &a_data[ind],
                        &a_i[row_starts[i]],
                        &a_j[ind],
                        &csr_matrix_datatypes[i]);
                MPI_Isend(MPI_BOTTOM, 1, csr_matrix_datatypes[i], i, 0, comm,
                        &requests[j++]);
                MPI_Type_free(&csr_matrix_datatypes[i]);
        }
        hypre_CSRMatrixData(local_A) = a_data;
        hypre_CSRMatrixI(local_A) = a_i;
        hypre_CSRMatrixJ(local_A) = a_j;
        hypre_CSRMatrixOwnsData(local_A) = 0;
        MPI_Waitall(num_procs-1,requests,status);
        hypre_TFree(requests);
        hypre_TFree(status);
        hypre_TFree(local_num_nonzeros);
    }
   else
   {
        hypre_CSRMatrixInitialize(local_A);
        hypre_BuildCSRMatrixMPIDataType(num_nonzeros, 
                        local_num_rows[my_id],
                        hypre_CSRMatrixData(local_A),
                        hypre_CSRMatrixI(local_A),
                        hypre_CSRMatrixJ(local_A),
                        csr_matrix_datatypes);
        MPI_Recv(MPI_BOTTOM,1,csr_matrix_datatypes[0],0,0,comm,&status0);
        MPI_Type_free(csr_matrix_datatypes);
   }

   first_col_diag = col_starts[my_id];
   last_col_diag = col_starts[my_id+1]-1;

   GenerateDiagAndOffd(local_A, par_matrix, first_col_diag, last_col_diag);

   /* set pointers back to NULL before destroying */
   if (my_id == 0)
   {      
      hypre_CSRMatrixData(local_A) = NULL;
      hypre_CSRMatrixI(local_A) = NULL;
      hypre_CSRMatrixJ(local_A) = NULL; 
   }      
   hypre_CSRMatrixDestroy(local_A);
   hypre_TFree(local_num_rows);
   hypre_TFree(csr_matrix_datatypes);

   return par_matrix;
}

int
GenerateDiagAndOffd(hypre_CSRMatrix *A,
                    hypre_ParCSRMatrix *matrix,
                    int first_col_diag,
                    int last_col_diag)
{
   int  i, j;
   int  jo, jd;
   int  ierr = 0;
   int  num_rows = hypre_CSRMatrixNumRows(A);
   int  num_cols = hypre_CSRMatrixNumCols(A);
   double *a_data = hypre_CSRMatrixData(A);
   int *a_i = hypre_CSRMatrixI(A);
   int *a_j = hypre_CSRMatrixJ(A);

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(matrix);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(matrix);

   int  *col_map_offd;

   double *diag_data, *offd_data;
   int  *diag_i, *offd_i;
   int  *diag_j, *offd_j;
   int  *marker;
   int num_cols_diag, num_cols_offd;
   int first_elmt = a_i[0];
   int num_nonzeros = a_i[num_rows]-first_elmt;
   int counter;

   num_cols_diag = last_col_diag - first_col_diag +1;
   num_cols_offd = 0;

   if (num_cols - num_cols_diag)
   {
        hypre_CSRMatrixInitialize(diag);
        diag_i = hypre_CSRMatrixI(diag);

        hypre_CSRMatrixInitialize(offd);
        offd_i = hypre_CSRMatrixI(offd);
        marker = hypre_CTAlloc(int,num_cols);

        for (i=0; i < num_cols; i++)
                marker[i] = 0;
        
        jo = 0;
        jd = 0;
        for (i=0; i < num_rows; i++)
        {
            offd_i[i] = jo;
            diag_i[i] = jd;
   
            for (j=a_i[i]-first_elmt; j < a_i[i+1]-first_elmt; j++)
                if (a_j[j] < first_col_diag || a_j[j] > last_col_diag)
                {
                        if (!marker[a_j[j]])
                        {
                                marker[a_j[j]] = 1;
                                num_cols_offd++;
                        }
                        jo++;
                }
                else
                {
                        jd++;
                }
        }
        offd_i[num_rows] = jo;
        diag_i[num_rows] = jd;

        hypre_ParCSRMatrixColMapOffd(matrix) = hypre_CTAlloc(int,num_cols_offd);
        col_map_offd = hypre_ParCSRMatrixColMapOffd(matrix);

        counter = 0;
        for (i=0; i < num_cols; i++)
                if (marker[i])
                {
                        col_map_offd[counter] = i;
                        marker[i] = counter;
                        counter++;
                }

        hypre_CSRMatrixNumNonzeros(diag) = jd;
        hypre_CSRMatrixInitialize(diag);
        diag_data = hypre_CSRMatrixData(diag);
        diag_j = hypre_CSRMatrixJ(diag);

        hypre_CSRMatrixNumNonzeros(offd) = jo;
        hypre_CSRMatrixNumCols(offd) = num_cols_offd;
        hypre_CSRMatrixInitialize(offd);
        offd_data = hypre_CSRMatrixData(offd);
        offd_j = hypre_CSRMatrixJ(offd);

        jo = 0;
        jd = 0;
        for (i=0; i < num_rows; i++)
        {
            for (j=a_i[i]-first_elmt; j < a_i[i+1]-first_elmt; j++)
                if (a_j[j] < first_col_diag || a_j[j] > last_col_diag)
                {
                        offd_data[jo] = a_data[j];
                        offd_j[jo++] = marker[a_j[j]];
                }
                else
                {
                        diag_data[jd] = a_data[j];
                        diag_j[jd++] = a_j[j]-first_col_diag;
                }
        }
        hypre_TFree(marker);
   }
   else 
   {
        hypre_CSRMatrixNumNonzeros(diag) = num_nonzeros;
        hypre_CSRMatrixInitialize(diag);
        diag_data = hypre_CSRMatrixData(diag);
        diag_i = hypre_CSRMatrixI(diag);
        diag_j = hypre_CSRMatrixJ(diag);

        for (i=0; i < num_nonzeros; i++)
        {
                diag_data[i] = a_data[i];
                diag_j[i] = a_j[i];
        }
        offd_i = hypre_CTAlloc(int, num_rows+1);

        for (i=0; i < num_rows+1; i++)
        {
                diag_i[i] = a_i[i];
                offd_i[i] = 0;
        }

        hypre_CSRMatrixNumCols(offd) = 0;
        hypre_CSRMatrixI(offd) = offd_i;
   }
   
   return ierr;
}

hypre_CSRMatrix *
hypre_MergeDiagAndOffd(hypre_ParCSRMatrix *par_matrix)
{
   hypre_CSRMatrix  *diag = hypre_ParCSRMatrixDiag(par_matrix);
   hypre_CSRMatrix  *offd = hypre_ParCSRMatrixOffd(par_matrix);
   hypre_CSRMatrix  *matrix;

   int          num_cols = hypre_ParCSRMatrixGlobalNumCols(par_matrix);
   int          first_col_diag = hypre_ParCSRMatrixFirstColDiag(par_matrix);
   int          *col_map_offd = hypre_ParCSRMatrixColMapOffd(par_matrix);
   int          num_rows = hypre_CSRMatrixNumRows(diag);

   int          *diag_i = hypre_CSRMatrixI(diag);
   int          *diag_j = hypre_CSRMatrixJ(diag);
   double       *diag_data = hypre_CSRMatrixData(diag);
   int          *offd_i = hypre_CSRMatrixI(offd);
   int          *offd_j = hypre_CSRMatrixJ(offd);
   double       *offd_data = hypre_CSRMatrixData(offd);

   int          *matrix_i;
   int          *matrix_j;
   double       *matrix_data;

   int          num_nonzeros, i, j;
   int          count;

/*
   if (!num_cols_offd)
   {
        matrix = hypre_CSRMatrixCreate(num_rows,num_cols,diag_i[num_rows]);
        hypre_CSRMatrixOwnsData(matrix) = 0;
        hypre_CSRMatrixI(matrix) = diag_i;
        hypre_CSRMatrixJ(matrix) = diag_j;
        hypre_CSRMatrixData(matrix) = diag_data;
        return matrix;
   }
*/

   num_nonzeros = diag_i[num_rows] + offd_i[num_rows];

   matrix = hypre_CSRMatrixCreate(num_rows,num_cols,num_nonzeros);
   hypre_CSRMatrixInitialize(matrix);

   matrix_i = hypre_CSRMatrixI(matrix);
   matrix_j = hypre_CSRMatrixJ(matrix);
   matrix_data = hypre_CSRMatrixData(matrix);

   count = 0;
   matrix_i[0] = 0;
   for (i=0; i < num_rows; i++)
   {
        for (j=diag_i[i]; j < diag_i[i+1]; j++)
        {
                matrix_data[count] = diag_data[j];
                matrix_j[count++] = diag_j[j]+first_col_diag;
        }
        for (j=offd_i[i]; j < offd_i[i+1]; j++)
        {
                matrix_data[count] = offd_data[j];
                matrix_j[count++] = col_map_offd[offd_j[j]];
        }
        matrix_i[i+1] = count;
   }

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixToCSRMatrixAll:
 * generates a CSRMatrix from a ParCSRMatrix on all processors that have
 * parts of the ParCSRMatrix
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_ParCSRMatrixToCSRMatrixAll(hypre_ParCSRMatrix *par_matrix)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(par_matrix);
   hypre_CSRMatrix *matrix;
   hypre_CSRMatrix *local_matrix;
   int num_rows = hypre_ParCSRMatrixGlobalNumRows(par_matrix);
   int num_cols = hypre_ParCSRMatrixGlobalNumCols(par_matrix);
#ifndef HYPRE_NO_GLOBAL_PARTITION
   int *row_starts = hypre_ParCSRMatrixRowStarts(par_matrix);
#endif
   int *matrix_i;
   int *matrix_j;
   double *matrix_data;
  
   int *local_matrix_i;
   int *local_matrix_j;
   double *local_matrix_data;
  
   int i, j;
   int local_num_rows;
   int local_num_nonzeros;
   int num_nonzeros;
   int num_data;
   int num_requests;
   int vec_len, offset;
   int start_index;
   int proc_id;
   int num_procs, my_id;
   int num_types;
   int *used_procs;

   MPI_Request *requests;
   MPI_Status *status;
   /* MPI_Datatype *data_type; */


#ifdef HYPRE_NO_GLOBAL_PARTITION

   int *new_vec_starts;
   
   int num_contacts;
   int contact_proc_list[1];
   int contact_send_buf[1];
   int contact_send_buf_starts[2];
   int max_response_size;
   int *response_recv_buf=NULL;
   int *response_recv_buf_starts = NULL;
   hypre_DataExchangeResponse response_obj;
   hypre_ProcListElements send_proc_obj;
   
   int *send_info = NULL;
   MPI_Status  status1;
   int count, tag1 = 11112, tag2 = 22223, tag3 = 33334;
   int start;
   
#endif



   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);

#ifdef HYPRE_NO_GLOBAL_PARTITION

   local_num_rows = hypre_ParCSRMatrixLastRowIndex(par_matrix)  - 
                    hypre_ParCSRMatrixFirstRowIndex(par_matrix) + 1;
   

   local_matrix = hypre_MergeDiagAndOffd(par_matrix); /* creates matrix */
   local_matrix_i = hypre_CSRMatrixI(local_matrix);
   local_matrix_j = hypre_CSRMatrixJ(local_matrix);
   local_matrix_data = hypre_CSRMatrixData(local_matrix);

 
/* determine procs that have vector data and store their ids in used_procs */
/* we need to do an exchange data for this.  If I own row then I will contact
   processor 0 with the endpoint of my local range */

   if (local_num_rows > 0)
   {
      num_contacts = 1;
      contact_proc_list[0] = 0;
      contact_send_buf[0] =  hypre_ParCSRMatrixLastRowIndex(par_matrix);
      contact_send_buf_starts[0] = 0;
      contact_send_buf_starts[1] = 1;
      
   }
   else
   {
      num_contacts = 0;
      contact_send_buf_starts[0] = 0;
      contact_send_buf_starts[1] = 0;
   }
  /*build the response object*/
   /*send_proc_obj will  be for saving info from contacts */
   send_proc_obj.length = 0;
   send_proc_obj.storage_length = 10;
   send_proc_obj.id = hypre_CTAlloc(int, send_proc_obj.storage_length);
   send_proc_obj.vec_starts = hypre_CTAlloc(int, send_proc_obj.storage_length + 1); 
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = 10;
   send_proc_obj.elements = hypre_CTAlloc(int, send_proc_obj.element_storage_length);

   max_response_size = 0; /* each response is null */
   response_obj.fill_response = hypre_FillResponseParToCSRMatrix;
   response_obj.data1 = NULL;
   response_obj.data2 = &send_proc_obj; /*this is where we keep info from contacts*/
  
   
   hypre_DataExchangeList(num_contacts, 
                          contact_proc_list, contact_send_buf, 
                          contact_send_buf_starts, sizeof(int), 
                          sizeof(int), &response_obj, 
                          max_response_size, 1,
                          comm, (void**) &response_recv_buf,	   
                          &response_recv_buf_starts);
   



   /* now processor 0 should have a list of ranges for processors that have rows -
      these are in send_proc_obj - it needs to create the new list of processors
      and also an array of vec starts - and send to those who own row*/
   if (my_id)
   {
      if (local_num_rows)      
      {
         /* look for a message from processor 0 */         
         MPI_Probe(0, tag1, comm, &status1);
         MPI_Get_count(&status1, MPI_INT, &count);
         
         send_info = hypre_CTAlloc(int, count);
         MPI_Recv(send_info, count, MPI_INT, 0, tag1, comm, &status1);

         /* now unpack */  
         num_types = send_info[0];
         used_procs =  hypre_CTAlloc(int, num_types);  
         new_vec_starts = hypre_CTAlloc(int, num_types+1);

         for (i=1; i<= num_types; i++)
         {
            used_procs[i-1] = send_info[i];
         }
         for (i=num_types+1; i< count; i++)
         {
            new_vec_starts[i-num_types-1] = send_info[i] ;
         }
      }
      else /* clean up and exit */
      {
         hypre_TFree(send_proc_obj.vec_starts);
         hypre_TFree(send_proc_obj.id);
         hypre_TFree(send_proc_obj.elements);
         if(response_recv_buf)        hypre_TFree(response_recv_buf);
         if(response_recv_buf_starts) hypre_TFree(response_recv_buf_starts);


         if (hypre_CSRMatrixOwnsData(local_matrix))
            hypre_CSRMatrixDestroy(local_matrix);
         else
            hypre_TFree(local_matrix);


         return NULL;
      }
   }
   else /* my_id ==0 */
   {
      num_types = send_proc_obj.length;
      used_procs =  hypre_CTAlloc(int, num_types);  
      new_vec_starts = hypre_CTAlloc(int, num_types+1);
      
      new_vec_starts[0] = 0;
      for (i=0; i< num_types; i++)
      {
         used_procs[i] = send_proc_obj.id[i];
         new_vec_starts[i+1] = send_proc_obj.elements[i]+1;
      }
      qsort0(used_procs, 0, num_types-1);
      qsort0(new_vec_starts, 0, num_types);
      /*now we need to put into an array to send */
      count =  2*num_types+2;
      send_info = hypre_CTAlloc(int, count);
      send_info[0] = num_types;
      for (i=1; i<= num_types; i++)
      {
         send_info[i] = used_procs[i-1];
      }
      for (i=num_types+1; i< count; i++)
      {
         send_info[i] = new_vec_starts[i-num_types-1];
      }
      requests = hypre_CTAlloc(MPI_Request, num_types);
      status =  hypre_CTAlloc(MPI_Status, num_types);

      /* don't send to myself  - these are sorted so my id would be first*/
      start = 0;
      if (used_procs[0] == 0)
      {
         start = 1;
      }
   
      
      for (i=start; i < num_types; i++)
      {
         MPI_Isend(send_info, count, MPI_INT, used_procs[i], tag1, comm, &requests[i-start]);
      }
      MPI_Waitall(num_types-start, requests, status);

      hypre_TFree(status);
      hypre_TFree(requests);
      

   }
   /* clean up */
   hypre_TFree(send_proc_obj.vec_starts);
   hypre_TFree(send_proc_obj.id);
   hypre_TFree(send_proc_obj.elements);
   hypre_TFree(send_info);
   if(response_recv_buf)        hypre_TFree(response_recv_buf);
   if(response_recv_buf_starts) hypre_TFree(response_recv_buf_starts);

   /* now proc 0 can exit if it has no rows */
   if (!local_num_rows) 
   { 
      if (hypre_CSRMatrixOwnsData(local_matrix))
         hypre_CSRMatrixDestroy(local_matrix);
      else
         hypre_TFree(local_matrix);

      hypre_TFree(new_vec_starts);
      hypre_TFree(used_procs);

      return NULL;
   }
   

   /* everyone left has rows and knows: new_vec_starts, num_types, and used_procs */

  /* this matrix should be rather small */
   matrix_i = hypre_CTAlloc(int, num_rows+1);


   num_requests = 4*num_types;
   requests = hypre_CTAlloc(MPI_Request, num_requests);
   status = hypre_CTAlloc(MPI_Status, num_requests);


   /* exchange contents of local_matrix_i - here we are sending to ourself also*/

   j = 0;
   for (i = 0; i < num_types; i++)
   {
        proc_id = used_procs[i];
        vec_len = new_vec_starts[i+1] - new_vec_starts[i];
        MPI_Irecv(&matrix_i[new_vec_starts[i]+1], vec_len, MPI_INT,
                                proc_id, tag2, comm, &requests[j++]);
   }
   for (i = 0; i < num_types; i++)
   {
        proc_id = used_procs[i];
        MPI_Isend(&local_matrix_i[1], local_num_rows, MPI_INT,
                                proc_id, tag2, comm, &requests[j++]);
   }

   MPI_Waitall(j, requests, status);



/* generate matrix_i from received data */
/* global numbering?*/
   offset = matrix_i[new_vec_starts[1]];
   for (i=1; i < num_types; i++)
   {
        for (j = new_vec_starts[i]; j < new_vec_starts[i+1]; j++)
           matrix_i[j+1] += offset;
        offset = matrix_i[new_vec_starts[i+1]];
   }

   num_nonzeros = matrix_i[num_rows];

   matrix = hypre_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   hypre_CSRMatrixI(matrix) = matrix_i;
   hypre_CSRMatrixInitialize(matrix);
   matrix_j = hypre_CSRMatrixJ(matrix);
   matrix_data = hypre_CSRMatrixData(matrix);

/* generate datatypes for further data exchange and exchange remaining
   data, i.e. column info and actual data */

   j = 0;
   for (i = 0; i < num_types; i++)
   {
        proc_id = used_procs[i];
        start_index = matrix_i[new_vec_starts[i]];
        num_data = matrix_i[new_vec_starts[i+1]] - start_index; 
        MPI_Irecv(&matrix_data[start_index], num_data, MPI_DOUBLE,
                        used_procs[i], tag1, comm, &requests[j++]);
        MPI_Irecv(&matrix_j[start_index], num_data, MPI_INT,
                        used_procs[i], tag3, comm, &requests[j++]);
   }
   local_num_nonzeros = local_matrix_i[local_num_rows];
   for (i=0; i < num_types; i++)
   {
        MPI_Isend(local_matrix_data, local_num_nonzeros, MPI_DOUBLE,
                        used_procs[i], tag1, comm, &requests[j++]);
        MPI_Isend(local_matrix_j, local_num_nonzeros, MPI_INT,
                        used_procs[i], tag3, comm, &requests[j++]);
   }


   MPI_Waitall(num_requests, requests, status);

   hypre_TFree(new_vec_starts);
   
#else
   local_num_rows = row_starts[my_id+1] - row_starts[my_id];

/* if my_id contains no data, return NULL */
 
   if (!local_num_rows)
        return NULL;
 
   local_matrix = hypre_MergeDiagAndOffd(par_matrix);
   local_matrix_i = hypre_CSRMatrixI(local_matrix);
   local_matrix_j = hypre_CSRMatrixJ(local_matrix);
   local_matrix_data = hypre_CSRMatrixData(local_matrix);

   matrix_i = hypre_CTAlloc(int, num_rows+1);

/* determine procs that have vector data and store their ids in used_procs */

   num_types = 0;
   for (i=0; i < num_procs; i++)
        if (row_starts[i+1]-row_starts[i] && i-my_id)
                num_types++;
   num_requests = 4*num_types;

   used_procs = hypre_CTAlloc(int, num_types);
   j = 0;
   for (i=0; i < num_procs; i++)
        if (row_starts[i+1]-row_starts[i] && i-my_id)
                used_procs[j++] = i;

   requests = hypre_CTAlloc(MPI_Request, num_requests);
   status = hypre_CTAlloc(MPI_Status, num_requests);
   /* data_type = hypre_CTAlloc(MPI_Datatype, num_types+1); */

/* exchange contents of local_matrix_i */

   j = 0;
   for (i = 0; i < num_types; i++)
   {
        proc_id = used_procs[i];
        vec_len = row_starts[proc_id+1] - row_starts[proc_id];
        MPI_Irecv(&matrix_i[row_starts[proc_id]+1], vec_len, MPI_INT,
                                proc_id, 0, comm, &requests[j++]);
   }
   for (i = 0; i < num_types; i++)
   {
        proc_id = used_procs[i];
        MPI_Isend(&local_matrix_i[1], local_num_rows, MPI_INT,
                                proc_id, 0, comm, &requests[j++]);
   }

   vec_len = row_starts[my_id+1] - row_starts[my_id];
   for (i=1; i <= vec_len; i++)
        matrix_i[row_starts[my_id]+i] = local_matrix_i[i];

   MPI_Waitall(j, requests, status);

/* generate matrix_i from received data */

   offset = matrix_i[row_starts[1]];
   for (i=1; i < num_procs; i++)
   {
        for (j = row_starts[i]; j < row_starts[i+1]; j++)
                matrix_i[j+1] += offset;
        offset = matrix_i[row_starts[i+1]];
   }

   num_nonzeros = matrix_i[num_rows];

   matrix = hypre_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   hypre_CSRMatrixI(matrix) = matrix_i;
   hypre_CSRMatrixInitialize(matrix);
   matrix_j = hypre_CSRMatrixJ(matrix);
   matrix_data = hypre_CSRMatrixData(matrix);

/* generate datatypes for further data exchange and exchange remaining
   data, i.e. column info and actual data */

   j = 0;
   for (i = 0; i < num_types; i++)
   {
        proc_id = used_procs[i];
        start_index = matrix_i[row_starts[proc_id]];
        num_data = matrix_i[row_starts[proc_id+1]] - start_index; 
        MPI_Irecv(&matrix_data[start_index], num_data, MPI_DOUBLE,
                        used_procs[i], 0, comm, &requests[j++]);
        MPI_Irecv(&matrix_j[start_index], num_data, MPI_INT,
                        used_procs[i], 0, comm, &requests[j++]);
   }
   local_num_nonzeros = local_matrix_i[local_num_rows];
   for (i=0; i < num_types; i++)
   {
        MPI_Isend(local_matrix_data, local_num_nonzeros, MPI_DOUBLE,
                        used_procs[i], 0, comm, &requests[j++]);
        MPI_Isend(local_matrix_j, local_num_nonzeros, MPI_INT,
                        used_procs[i], 0, comm, &requests[j++]);
   }

   start_index = matrix_i[row_starts[my_id]];
   for (i=0; i < local_num_nonzeros; i++)
   {
        matrix_j[start_index+i] = local_matrix_j[i];
        matrix_data[start_index+i] = local_matrix_data[i];
   }
   MPI_Waitall(num_requests, requests, status);
/*   for (i = 0; i < num_types; i++)
   {
        proc_id = used_procs[i];
        start_index = matrix_i[row_starts[proc_id]];
        num_data = matrix_i[row_starts[proc_id+1]] - start_index; 
        hypre_BuildCSRJDataType(num_data,
                          &matrix_data[start_index],
                          &matrix_j[start_index],
                          &data_type[i]);
   }
   local_num_nonzeros = local_matrix_i[local_num_rows];
   hypre_BuildCSRJDataType(local_num_nonzeros,
                     local_matrix_data,
                     local_matrix_j,
                     &data_type[num_types]);
   j = 0;
   for (i=0; i < num_types; i++)
        MPI_Irecv(MPI_BOTTOM, 1, data_type[i],
                        used_procs[i], 0, comm, &requests[j++]);
   for (i=0; i < num_types; i++)
        MPI_Isend(MPI_BOTTOM, 1, data_type[num_types],
                        used_procs[i], 0, comm, &requests[j++]);
*/
   start_index = matrix_i[row_starts[my_id]];
   for (i=0; i < local_num_nonzeros; i++)
   {
        matrix_j[start_index+i] = local_matrix_j[i];
        matrix_data[start_index+i] = local_matrix_data[i];
   }
   MPI_Waitall(num_requests, requests, status);
/*
   for (i=0; i <= num_types; i++)
        MPI_Type_free(&data_type[i]);
*/

#endif

   if (hypre_CSRMatrixOwnsData(local_matrix))
        hypre_CSRMatrixDestroy(local_matrix);
   else
        hypre_TFree(local_matrix);

   if (num_requests)
   {
        hypre_TFree(requests);
        hypre_TFree(status);
        hypre_TFree(used_procs);
   }
   /* hypre_TFree(data_type); */

   return matrix;
}
    
/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixCopy,
 * copies B to A,
 * if copy_data = 0, only the structure of A is copied to B
 * the routine does not check whether the dimensions of A and B are compatible
 *--------------------------------------------------------------------------*/

int 
hypre_ParCSRMatrixCopy( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B, 
                        int copy_data )
{
   int  ierr=0;
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   int *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);
   hypre_CSRMatrix *B_diag = hypre_ParCSRMatrixDiag(B);
   hypre_CSRMatrix *B_offd = hypre_ParCSRMatrixOffd(B);
   int *col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B);
   int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   int i;

   hypre_CSRMatrixCopy(A_diag, B_diag, copy_data);
   hypre_CSRMatrixCopy(A_offd, B_offd, copy_data);
   if (num_cols_offd && col_map_offd_B == NULL)
   {
      col_map_offd_B = hypre_CTAlloc(int,num_cols_offd);
      hypre_ParCSRMatrixColMapOffd(B) = col_map_offd_B;
   }
   for (i = 0; i < num_cols_offd; i++)
        col_map_offd_B[i] = col_map_offd_A[i];
        
   return ierr;
}
/*--------------------------------------------------------------------
 * hypre_FillResponseParToCSRMatrix
 * Fill response function for determining the send processors
 * data exchange
 *--------------------------------------------------------------------*/

int
hypre_FillResponseParToCSRMatrix(void *p_recv_contact_buf, 
                                 int contact_size, int contact_proc, void *ro, 
                                 MPI_Comm comm, void **p_send_response_buf, 
                                 int *response_message_size )
{
   int    myid;
   int    i, index, count, elength;

   int    *recv_contact_buf = (int * ) p_recv_contact_buf;

   hypre_DataExchangeResponse  *response_obj = ro;  

   hypre_ProcListElements      *send_proc_obj = response_obj->data2;   


   MPI_Comm_rank(comm, &myid );


   /*check to see if we need to allocate more space in send_proc_obj for ids*/
   if (send_proc_obj->length == send_proc_obj->storage_length)
   {
      send_proc_obj->storage_length +=10; /*add space for 10 more processors*/
      send_proc_obj->id = hypre_TReAlloc(send_proc_obj->id,int, 
					 send_proc_obj->storage_length);
      send_proc_obj->vec_starts = hypre_TReAlloc(send_proc_obj->vec_starts,int, 
                                  send_proc_obj->storage_length + 1);
   }
  
   /*initialize*/ 
   count = send_proc_obj->length;
   index = send_proc_obj->vec_starts[count]; /*this is the number of elements*/

   /*send proc*/ 
   send_proc_obj->id[count] = contact_proc; 

   /*do we need more storage for the elements?*/
     if (send_proc_obj->element_storage_length < index + contact_size)
   {
      elength = hypre_max(contact_size, 10);   
      elength += index;
      send_proc_obj->elements = hypre_TReAlloc(send_proc_obj->elements, 
					       int, elength);
      send_proc_obj->element_storage_length = elength; 
   }
   /*populate send_proc_obj*/
   for (i=0; i< contact_size; i++) 
   { 
      send_proc_obj->elements[index++] = recv_contact_buf[i];
   }
   send_proc_obj->vec_starts[count+1] = index;
   send_proc_obj->length++;
   

  /*output - no message to return (confirmation) */
   *response_message_size = 0; 
  
   
   return(0);

}
