/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.2 $
*********************************************************************EHEADER*/
 
#include "headers.h"
 
/*--------------------------------------------------------------------------
 * hypre_GenerateLaplacian9pt
 *--------------------------------------------------------------------------*/

HYPRE_ParCSRMatrix 
GenerateLaplacian9pt( MPI_Comm comm,
                      int      nx,
                      int      ny,
                      int      P,
                      int      Q,
                      int      p,
                      int      q,
                      double  *value )
{
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;

   int    *diag_i;
   int    *diag_j;
   double *diag_data;

   int    *offd_i;
   int    *offd_j;
   double *offd_data;

   int *global_part;
   int ix, iy;
   int cnt, o_cnt;
   int local_num_rows; 
   int *col_map_offd;
   int *work;
   int row_index;
   int i,j;

   int nx_local, ny_local;
   int nx_size, ny_size;
   int num_cols_offd;
   int grid_size;

   int *nx_part;
   int *ny_part;

   int num_procs, my_id;
   int P_busy, Q_busy;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   grid_size = nx*ny;

   hypre_GeneratePartitioning(nx,P,&nx_part);
   hypre_GeneratePartitioning(ny,Q,&ny_part);

   global_part = hypre_CTAlloc(int,P*Q+1);

   global_part[0] = 0;
   cnt = 1;
   for (iy = 0; iy < Q; iy++)
   {
      ny_size = ny_part[iy+1]-ny_part[iy];
      for (ix = 0; ix < P; ix++)
      {
         nx_size = nx_part[ix+1] - nx_part[ix];
         global_part[cnt] = global_part[cnt-1];
         global_part[cnt++] += nx_size*ny_size;
      }
   }

   nx_local = nx_part[p+1] - nx_part[p];
   ny_local = ny_part[q+1] - ny_part[q];

   my_id = q*P + p;
   num_procs = P*Q;

   local_num_rows = nx_local*ny_local;
   diag_i = hypre_CTAlloc(int, local_num_rows+1);
   offd_i = hypre_CTAlloc(int, local_num_rows+1);

   P_busy = hypre_min(nx,P);
   Q_busy = hypre_min(ny,Q);

   num_cols_offd = 0;
   if (p) num_cols_offd += ny_local;
   if (p < P_busy-1) num_cols_offd += ny_local;
   if (q) num_cols_offd += nx_local;
   if (q < Q_busy-1) num_cols_offd += nx_local;
   if (p && q) num_cols_offd++;
   if (p && q < Q_busy-1 ) num_cols_offd++;
   if (p < P_busy-1 && q ) num_cols_offd++;
   if (p < P_busy-1 && q < Q_busy-1 ) num_cols_offd++;

   if (!local_num_rows) num_cols_offd = 0;

   col_map_offd = hypre_CTAlloc(int, num_cols_offd);

   cnt = 0;
   o_cnt = 0;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
   {
      for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
      {
         cnt++;
         o_cnt++;
         diag_i[cnt] = diag_i[cnt-1];
         offd_i[o_cnt] = offd_i[o_cnt-1];
         diag_i[cnt]++;
         if (iy > ny_part[q]) 
         {
            diag_i[cnt]++;
	    if (ix > nx_part[p])
	    {
	       diag_i[cnt]++;
	    }
	    else
	    {
	       if (ix) 
		  offd_i[o_cnt]++;
	    }
	    if (ix < nx_part[p+1]-1)
	    {
	       diag_i[cnt]++;
	    }
	    else
	    {
	       if (ix+1 < nx) 
		  offd_i[o_cnt]++;
	    }
         }
         else
         {
            if (iy) 
            {
               offd_i[o_cnt]++;
	       if (ix > nx_part[p])
	       {
	          offd_i[o_cnt]++;
	       }
	       else if (ix)
	       {
	          offd_i[o_cnt]++;
	       }
	       if (ix < nx_part[p+1]-1)
	       {
	          offd_i[o_cnt]++;
	       }
	       else if (ix < nx-1)
	       {
	          offd_i[o_cnt]++;
	       }
            }
         }
         if (ix > nx_part[p]) 
            diag_i[cnt]++;
         else
         {
            if (ix) 
            {
               offd_i[o_cnt]++; 
            }
         }
         if (ix+1 < nx_part[p+1]) 
            diag_i[cnt]++;
         else
         {
            if (ix+1 < nx) 
            {
               offd_i[o_cnt]++; 
            }
         }
         if (iy+1 < ny_part[q+1]) 
         {
            diag_i[cnt]++;
	    if (ix > nx_part[p])
	    {
	       diag_i[cnt]++;
	    }
	    else
	    {
	       if (ix) 
		  offd_i[o_cnt]++;
	    }
	    if (ix < nx_part[p+1]-1)
	    {
	       diag_i[cnt]++;
	    }
	    else
	    {
	       if (ix+1 < nx) 
		  offd_i[o_cnt]++;
	    }
         }
         else
         {
            if (iy+1 < ny) 
            {
               offd_i[o_cnt]++;
	       if (ix > nx_part[p])
	       {
	          offd_i[o_cnt]++;
	       }
	       else if (ix)
	       {
	          offd_i[o_cnt]++;
	       }
	       if (ix < nx_part[p+1]-1)
	       {
	          offd_i[o_cnt]++;
	       }
	       else if (ix < nx-1)
	       {
	          offd_i[o_cnt]++;
	       }
            }
         }
      }
   }

   diag_j = hypre_CTAlloc(int, diag_i[local_num_rows]);
   diag_data = hypre_CTAlloc(double, diag_i[local_num_rows]);

   if (num_procs > 1)
   {
      offd_j = hypre_CTAlloc(int, offd_i[local_num_rows]);
      offd_data = hypre_CTAlloc(double, offd_i[local_num_rows]);
   }

   row_index = 0;
   cnt = 0;
   o_cnt = 0;
   for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
   {
      for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
      {
         diag_j[cnt] = row_index;
         diag_data[cnt++] = value[0];
         if (iy > ny_part[q]) 
         {
	    if (ix > nx_part[p])
	    {
	       diag_j[cnt] = row_index-nx_local-1 ;
               diag_data[cnt++] = value[1];
	    }
	    else
	    {
	       if (ix) 
	       { 
                  offd_j[o_cnt] = hypre_map2(ix-1,iy-1,p-1,q,P,Q,
                                   nx_part,ny_part,global_part);
                  offd_data[o_cnt++] = value[1];
	       } 
	    }
            diag_j[cnt] = row_index-nx_local;
            diag_data[cnt++] = value[1];
	    if (ix < nx_part[p+1]-1)
	    {
	       diag_j[cnt] = row_index-nx_local+1 ;
               diag_data[cnt++] = value[1];
	    }
	    else
	    {
	       if (ix+1 < nx)
	       { 
		  offd_j[o_cnt] = hypre_map2(ix+1,iy-1,p+1,q,P,Q,
                                   nx_part,ny_part,global_part);
                  offd_data[o_cnt++] = value[1];
	       } 
	    }
         }
         else
         {
            if (iy) 
            {
	       if (ix > nx_part[p])
	       {
                  offd_j[o_cnt] = hypre_map2(ix-1,iy-1,p,q-1,P,Q,
                                   nx_part,ny_part,global_part);
                  offd_data[o_cnt++] = value[1];
	       }
	       else if (ix)
	       {
                  offd_j[o_cnt] = hypre_map2(ix-1,iy-1,p-1,q-1,P,Q,
                                   nx_part,ny_part,global_part);
                  offd_data[o_cnt++] = value[1];
	       }
               offd_j[o_cnt] = hypre_map2(ix,iy-1,p,q-1,P,Q,
                                   nx_part,ny_part,global_part);
               offd_data[o_cnt++] = value[1];
	       if (ix < nx_part[p+1]-1)
	       {
                  offd_j[o_cnt] = hypre_map2(ix+1,iy-1,p,q-1,P,Q,
                                   nx_part,ny_part,global_part);
                  offd_data[o_cnt++] = value[1];
	       }
	       else if (ix+1 < nx)
	       {
                  offd_j[o_cnt] = hypre_map2(ix+1,iy-1,p+1,q-1,P,Q,
                                   nx_part,ny_part,global_part);
                  offd_data[o_cnt++] = value[1];
	       }
            }
         }
         if (ix > nx_part[p]) 
         {
            diag_j[cnt] = row_index-1;
            diag_data[cnt++] = value[1];
         }
         else
         {
            if (ix) 
            {
               offd_j[o_cnt] = hypre_map2(ix-1,iy,p-1,q,P,Q,
                                   nx_part,ny_part,global_part);
               offd_data[o_cnt++] = value[1];
            }
         }
         if (ix+1 < nx_part[p+1]) 
         {
            diag_j[cnt] = row_index+1;
            diag_data[cnt++] = value[1];
         }
         else
         {
            if (ix+1 < nx) 
            {
               offd_j[o_cnt] = hypre_map2(ix+1,iy,p+1,q,P,Q,
                                   nx_part,ny_part,global_part);
               offd_data[o_cnt++] = value[1];
            }
         }
         if (iy+1 < ny_part[q+1]) 
         {
	    if (ix > nx_part[p])
	    {
	       diag_j[cnt] = row_index+nx_local-1 ;
               diag_data[cnt++] = value[1];
	    }
	    else
	    {
	       if (ix) 
               {
                  offd_j[o_cnt] = hypre_map2(ix-1,iy+1,p-1,q,P,Q,
                                   nx_part,ny_part,global_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
            diag_j[cnt] = row_index+nx_local;
            diag_data[cnt++] = value[1];
	    if (ix < nx_part[p+1]-1)
	    {
	       diag_j[cnt] = row_index+nx_local+1 ;
               diag_data[cnt++] = value[1];
	    }
	    else
	    {
	       if (ix+1 < nx)
	       { 
                  offd_j[o_cnt] = hypre_map2(ix+1,iy+1,p+1,q,P,Q,
                                   nx_part,ny_part,global_part);
                  offd_data[o_cnt++] = value[1];
	       } 
	    }
         }
         else
         {
            if (iy+1 < ny) 
            {
	       if (ix > nx_part[p])
	       {
                  offd_j[o_cnt] = hypre_map2(ix-1,iy+1,p,q+1,P,Q,
                                   nx_part,ny_part,global_part);
                  offd_data[o_cnt++] = value[1];
	       }
	       else if (ix)
	       {
                  offd_j[o_cnt] = hypre_map2(ix-1,iy+1,p-1,q+1,P,Q,
                                   nx_part,ny_part,global_part);
                  offd_data[o_cnt++] = value[1];
	       }
               offd_j[o_cnt] = hypre_map2(ix,iy+1,p,q+1,P,Q,
                                   nx_part,ny_part,global_part);
               offd_data[o_cnt++] = value[1];
	       if (ix < nx_part[p+1]-1)
	       {
                  offd_j[o_cnt] = hypre_map2(ix+1,iy+1,p,q+1,P,Q,
                                   nx_part,ny_part,global_part);
                  offd_data[o_cnt++] = value[1];
	       }
	       else if (ix < nx-1)
	       {
                  offd_j[o_cnt] = hypre_map2(ix+1,iy+1,p+1,q+1,P,Q,
                                   nx_part,ny_part,global_part);
                  offd_data[o_cnt++] = value[1];
	       }
            }
         }
         row_index++;
      }
   }

   if (num_procs > 1)
   {
      work = hypre_CTAlloc(int,o_cnt);

      for (i=0; i < o_cnt; i++)
         work[i] = offd_j[i];

      qsort0(work, 0, o_cnt-1);

      col_map_offd[0] = work[0];
      cnt = 0;
      for (i=0; i < o_cnt; i++)
      {
         if (work[i] > col_map_offd[cnt])
         {
            cnt++;
            col_map_offd[cnt] = work[i];
         }
      }

      for (i=0; i < o_cnt; i++)
      {
         for (j=0; j < num_cols_offd; j++)
         {
            if (offd_j[i] == col_map_offd[j])
            {
               offd_j[i] = j;
               break;
            }
         }
      }

      hypre_TFree(work);
   }



#ifdef HYPRE_NO_GLOBAL_PARTITION
/* ideally we would use less storage earlier in this function, but this is fine
   for testing */
   {
      int tmp1, tmp2;
      tmp1 = global_part[my_id];
      tmp2 = global_part[my_id + 1];
      hypre_TFree(global_part);
      global_part = hypre_CTAlloc(int, 2);
      global_part[0] = tmp1;
      global_part[1] = tmp2;
   }
#endif

   A = hypre_ParCSRMatrixCreate(comm, grid_size, grid_size,
                                global_part, global_part, num_cols_offd,
                                diag_i[local_num_rows],
                                offd_i[local_num_rows]);

   hypre_ParCSRMatrixColMapOffd(A) = col_map_offd;

   diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_data;

   offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrixI(offd) = offd_i;
   if (num_cols_offd)
   {
      hypre_CSRMatrixJ(offd) = offd_j;
      hypre_CSRMatrixData(offd) = offd_data;
   }

   hypre_TFree(nx_part);
   hypre_TFree(ny_part);

   return (HYPRE_ParCSRMatrix) A;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_map2( int  ix,
      int  iy,
      int  p,
      int  q,
      int  P,
      int  Q,
      int *nx_part,
      int *ny_part,
      int *global_part )
{
   int nx_local;
   int ix_local;
   int iy_local;
   int global_index;
   int proc_num;
 
   proc_num = q*P + p;
   nx_local = nx_part[p+1] - nx_part[p];
   ix_local = ix - nx_part[p];
   iy_local = iy - ny_part[q];
   global_index = global_part[proc_num] 
      + iy_local*nx_local + ix_local;

   return global_index;
}
