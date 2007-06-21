/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface (IJ_matrix interface).
 * This is the version which uses the Babel interface.
 * Do `driver -help' for usage info.
 * This driver started from the driver for parcsr_linear_solvers, and it
 * works by first building a parcsr matrix as before and then "copying"
 * that matrix row-by-row into the IJMatrix interface. AJC 7/99.
 *--------------------------------------------------------------------------*/
/* As of October 2005, the solvers implemented are AMG, ParaSails, PCG, GMRES, diagonal
   scaling, and combinations thereof.  The babel (bHYPRE) interface is used exclusively.
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"

#include "bHYPRE.h"
#include "bHYPRE_IJParCSRMatrix_Impl.h"
#include "bHYPRE_IJParCSRVector_Impl.h"
#include "bHYPRE_ParCSRDiagScale_Impl.h"
#include "bHYPRE_Schwarz_Impl.h"
#include "bHYPRE_MPICommunicator_Impl.h"

int BuildParFromFile (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParLaplacian (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int bBuildParLaplacian( int argc, char *argv[], int arg_index, bHYPRE_MPICommunicator bmpi_comm,
                        bHYPRE_IJParCSRMatrix  *bA_ptr );
int BuildParDifConv (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParFromOneFile (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildFuncsFromFiles (int argc , char *argv [], int arg_index , bHYPRE_IJParCSRMatrix A , int **dof_func_ptr );
int BuildFuncsFromOneFile (int argc , char *argv [], int arg_index , bHYPRE_IJParCSRMatrix A , int **dof_func_ptr );
int BuildRhsParFromOneFile_ (int argc , char *argv [], int arg_index , int *partitioning , HYPRE_ParVector *b_ptr );
int BuildParLaplacian9pt (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParLaplacian27pt (int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );

int
main( int   argc,
      char *argv[] )
{
   int                 arg_index;
   int                 print_usage;
   int                 sparsity_known = 0;
   int                 build_matrix_type;
   int                 build_matrix_arg_index;
   int                 build_rhs_type;
   int                 build_rhs_arg_index;
   int                 build_src_type;
   int                 build_src_arg_index;
   int                 build_funcs_type;
   int                 build_funcs_arg_index;
   int                 solver_id;
   int                 hpcg = 0;
   int                 ioutdat;
   int                 poutdat;
   int                 log_level;
   int                 debug_flag;
   int                 ierr = 0;
   int                 i,j,k; 
   int                 indx, rest, tms;
   int                 max_levels = 25;
   int                 num_iterations; 
   /*double              norm;*/
   double tmp;
   double              final_res_norm;

   HYPRE_ParCSRMatrix    parcsr_A;/* only for timing computation */
   bHYPRE_MPICommunicator bmpicomm;
   bHYPRE_IJParCSRMatrix  bHYPRE_parcsr_A;
   bHYPRE_Operator        bHYPRE_op_A;
   bHYPRE_IJParCSRVector  bHYPRE_b;
   bHYPRE_IJParCSRVector  bHYPRE_x;
   bHYPRE_IJParCSRVector  bHYPRE_y;
   bHYPRE_IJParCSRVector  bHYPRE_y2;
   bHYPRE_Vector          y,bHYPRE_Vector_x, bHYPRE_Vector_b;

   bHYPRE_BoomerAMG        bHYPRE_AMG;
   bHYPRE_PCG           bHYPRE_PCG;
   bHYPRE_HPCG          bHYPRE_HPCG;
   bHYPRE_GMRES         bHYPRE_GMRES;
   bHYPRE_HGMRES        bHYPRE_HGMRES;
   bHYPRE_BiCGSTAB      bHYPRE_BiCGSTAB;
   bHYPRE_CGNR          bHYPRE_CGNR;
   bHYPRE_ParCSRDiagScale  bHYPRE_ParCSRDiagScale;
   bHYPRE_ParaSails     bHYPRE_ParaSails;
   bHYPRE_Euclid        bHYPRE_Euclid;
   bHYPRE_Solver        bHYPRE_SolverPC;
   bHYPRE_Schwarz bHYPRE_Schwarz;

   int                 num_procs, myid;
   int                *rows;
   int                *ncols;
   int                *col_inds;
   int                *dof_func;
   int		       num_functions = 1;

   int		       time_index;
   MPI_Comm            mpi_comm = MPI_COMM_WORLD;
   int M, N;
   int first_local_row, last_local_row, local_num_rows;
   int first_local_col, last_local_col, local_num_cols;
   int local_num_vars;
   int variant, overlap, domain_type;
   double schwarz_rlx_weight;
   double *values, val;
   int *indices;
   struct sidl_int__array* bHYPRE_grid_relax_points=NULL;

   int dimsl[2], dimsu[2];

   const double dt_inf = 1.e40;
   double dt = dt_inf;

   /* parameters for BoomerAMG */
   double   strong_threshold;
   double   trunc_factor;
   int      cycle_type;
   int      coarsen_type = 6;
   int      hybrid = 1;
   int      measure_type = 0;
   int     *num_grid_sweeps = NULL;  
   int     *grid_relax_type = NULL;   
   int    **grid_relax_points = NULL;
   int	    smooth_type = 6;
   int	    smooth_num_levels = 0;
   int      relax_default;
   int      smooth_num_sweep = 1;
   int      num_sweep = 1;
   double  *relax_weight = NULL; 
   double  *omega;
   double   tol = 1.e-8;
   double   pc_tol = 0.;
   double   max_row_sum = 1.;

   /* parameters for ParaSAILS */
   double   sai_threshold = 0.1;
   double   sai_filter = 0.1;

   /* parameters for PILUT */
   double   drop_tol = -1;
   int      nonzeros_to_keep = -1;

   /* parameters for GMRES */
   int	    k_dim;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size( mpi_comm, &num_procs );
   MPI_Comm_rank( mpi_comm, &myid );
   bmpicomm = bHYPRE_MPICommunicator_CreateC( (void *)(&mpi_comm) );
/*
  hypre_InitMemoryDebug(myid);
*/
   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   build_matrix_type = 2;
   build_matrix_arg_index = argc;
   build_rhs_type = 2;
   build_rhs_arg_index = argc;
   build_src_type = -1;
   build_src_arg_index = argc;
   build_funcs_type = 0;
   build_funcs_arg_index = argc;
   relax_default = 3;
   debug_flag = 0;

   solver_id = 0;

   ioutdat = 3;
   poutdat = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
 
   print_usage = 0;
   arg_index = 1;

   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-fromijfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = -1;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromparcsrfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 0;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromonecsrfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 1;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-laplacian") == 0 )
      {
         arg_index++;
         build_matrix_type      = 2;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-9pt") == 0 )
      {
         arg_index++;
         build_matrix_type      = 3;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-27pt") == 0 )
      {
         arg_index++;
         build_matrix_type      = 4;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-difconv") == 0 )
      {
         arg_index++;
         build_matrix_type      = 5;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-funcsfromonefile") == 0 )
      {
         arg_index++;
         build_funcs_type      = 1;
         build_funcs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-funcsfromfile") == 0 )
      {
         arg_index++;
         build_funcs_type      = 2;
         build_funcs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-exact_size") == 0 )
      {
         arg_index++;
         sparsity_known = 1;
      }
      else if ( strcmp(argv[arg_index], "-storage_low") == 0 )
      {
         arg_index++;
         sparsity_known = 2;
      }
      else if ( strcmp(argv[arg_index], "-concrete_parcsr") == 0 )
      {
         arg_index++;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
         solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-hpcg") == 0 )
      {
         arg_index++;
         hpcg = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rhsfromfile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 0;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromonefile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 1;
         build_rhs_arg_index = arg_index;
      }      
      else if ( strcmp(argv[arg_index], "-rhsisone") == 0 )
      {
         arg_index++;
         build_rhs_type      = 2;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsrand") == 0 )
      {
         arg_index++;
         build_rhs_type      = 3;
         build_rhs_arg_index = arg_index;
      }    
      else if ( strcmp(argv[arg_index], "-xisone") == 0 )
      {
         arg_index++;
         build_rhs_type      = 4;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhszero") == 0 )
      {
         arg_index++;
         build_rhs_type      = 5;
         build_rhs_arg_index = arg_index;
      }    
      else if ( strcmp(argv[arg_index], "-srcfromfile") == 0 )
      {
         arg_index++;
         build_src_type      = 0;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcfromonefile") == 0 )
      {
         arg_index++;
         build_src_type      = 1;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcisone") == 0 )
      {
         arg_index++;
         build_src_type      = 2;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcrand") == 0 )
      {
         arg_index++;
         build_src_type      = 3;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srczero") == 0 )
      {
         arg_index++;
         build_src_type      = 4;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-cljp") == 0 )
      {
         arg_index++;
         coarsen_type      = 0;
      }    
      else if ( strcmp(argv[arg_index], "-ruge") == 0 )
      {
         arg_index++;
         coarsen_type      = 1;
      }    
      else if ( strcmp(argv[arg_index], "-ruge2b") == 0 )
      {
         arg_index++;
         coarsen_type      = 2;
      }    
      else if ( strcmp(argv[arg_index], "-ruge3") == 0 )
      {
         arg_index++;
         coarsen_type      = 3;
      }    
      else if ( strcmp(argv[arg_index], "-ruge3c") == 0 )
      {
         arg_index++;
         coarsen_type      = 4;
      }    
      else if ( strcmp(argv[arg_index], "-rugerlx") == 0 )
      {
         arg_index++;
         coarsen_type      = 5;
      }    
      else if ( strcmp(argv[arg_index], "-falgout") == 0 )
      {
         arg_index++;
         coarsen_type      = 6;
      }    
      else if ( strcmp(argv[arg_index], "-nohybrid") == 0 )
      {
         arg_index++;
         hybrid      = -1;
      }    
      else if ( strcmp(argv[arg_index], "-gm") == 0 )
      {
         arg_index++;
         measure_type      = 1;
      }    
      else if ( strcmp(argv[arg_index], "-rlx") == 0 )
      {
         arg_index++;
         relax_default = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-smtype") == 0 )
      {
         arg_index++;
         smooth_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-smlv") == 0 )
      {
         arg_index++;
         smooth_num_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mxl") == 0 )
      {
         arg_index++;
         max_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dbg") == 0 )
      {
         arg_index++;
         debug_flag = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nf") == 0 )
      {
         arg_index++;
         num_functions = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ns") == 0 )
      {
         arg_index++;
         num_sweep = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sns") == 0 )
      {
         arg_index++;
         smooth_num_sweep = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dt") == 0 )
      {
         arg_index++;
         dt = atof(argv[arg_index++]);
         build_rhs_type = -1;
         if ( build_src_type == -1 ) build_src_type = 2;
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
      }
      else
      {
         arg_index++;
      }
   }

   if (solver_id == 8 || solver_id == 18)
   {
      max_levels = 1;
   }

   /* defaults for BoomerAMG */
   if (solver_id == 0 || solver_id == 1 || solver_id == 3 || solver_id == 5
	|| solver_id == 9 || solver_id == 13 || solver_id == 14
 	|| solver_id == 15 || solver_id == 20)
   {
      strong_threshold = 0.25;
      trunc_factor = 0.;
      cycle_type = 1;

      num_grid_sweeps   = hypre_CTAlloc(int,4);
      grid_relax_type   = hypre_CTAlloc(int,4);
      grid_relax_points = hypre_CTAlloc(int *,4);
      relax_weight      = hypre_CTAlloc(double, max_levels);
      omega      = hypre_CTAlloc(double, max_levels);

      for (i=0; i < max_levels; i++)
      {
         relax_weight[i] = 1.;
         omega[i] = 1.;
      }

         /* for CGNR preconditioned with Boomeramg, only relaxation scheme 0 is
      implemented, i.e. Jacobi relaxation */
      if (solver_id == 5) 
      {
         /* fine grid */
         relax_default = 7;
         grid_relax_type[0] = relax_default; 
         num_grid_sweeps[0] = num_sweep;
         grid_relax_points[0] = hypre_CTAlloc(int, num_sweep); 
         for (i=0; i<num_sweep; i++)
         {
            grid_relax_points[0][i] = 0;
         } 
         /* down cycle */
         grid_relax_type[1] = relax_default; 
         num_grid_sweeps[1] = num_sweep;
         grid_relax_points[1] = hypre_CTAlloc(int, num_sweep); 
         for (i=0; i<num_sweep; i++)
         {
            grid_relax_points[1][i] = 0;
         } 
         /* up cycle */
         grid_relax_type[2] = relax_default; 
         num_grid_sweeps[2] = num_sweep;
         grid_relax_points[2] = hypre_CTAlloc(int, num_sweep); 
         for (i=0; i<num_sweep; i++)
         {
            grid_relax_points[2][i] = 0;
         } 
      }
      else if (coarsen_type == 5)
      {
         /* fine grid */
         num_grid_sweeps[0] = 3;
         grid_relax_type[0] = relax_default; 
         grid_relax_points[0] = hypre_CTAlloc(int, 3); 
         grid_relax_points[0][0] = -2;
         grid_relax_points[0][1] = -1;
         grid_relax_points[0][2] = 1;
   
         /* down cycle */
         num_grid_sweeps[1] = 4;
         grid_relax_type[1] = relax_default; 
         grid_relax_points[1] = hypre_CTAlloc(int, 4); 
         grid_relax_points[1][0] = -1;
         grid_relax_points[1][1] = 1;
         grid_relax_points[1][2] = -2;
         grid_relax_points[1][3] = -2;
   
         /* up cycle */
         num_grid_sweeps[2] = 4;
         grid_relax_type[2] = relax_default; 
         grid_relax_points[2] = hypre_CTAlloc(int, 4); 
         grid_relax_points[2][0] = -2;
         grid_relax_points[2][1] = -2;
         grid_relax_points[2][2] = 1;
         grid_relax_points[2][3] = -1;
      }
      else
      {   
         /* fine grid */
         num_grid_sweeps[0] = 2*num_sweep;
         grid_relax_type[0] = relax_default; 
         grid_relax_points[0] = hypre_CTAlloc(int, 2*num_sweep); 
         for (i=0; i<2*num_sweep; i+=2)
         {
            grid_relax_points[0][i] = 1;
            grid_relax_points[0][i+1] = -1;
         }

         /* down cycle */
         num_grid_sweeps[1] = 2*num_sweep;
         grid_relax_type[1] = relax_default; 
         grid_relax_points[1] = hypre_CTAlloc(int, 2*num_sweep); 
         for (i=0; i<2*num_sweep; i+=2)
         {
            grid_relax_points[1][i] = 1;
            grid_relax_points[1][i+1] = -1;
         }

         /* up cycle */
         num_grid_sweeps[2] = 2*num_sweep;
         grid_relax_type[2] = relax_default; 
         grid_relax_points[2] = hypre_CTAlloc(int, 2*num_sweep); 
         for (i=0; i<2*num_sweep; i+=2)
         {
            grid_relax_points[2][i] = -1;
            grid_relax_points[2][i+1] = 1;
         }
      }

      /* coarsest grid */
      num_grid_sweeps[3] = 1;
      grid_relax_type[3] = 9;
      grid_relax_points[3] = hypre_CTAlloc(int, 1);
      grid_relax_points[3][0] = 0;
   }

   /* defaults for Schwarz */

   variant = 0;  /* multiplicative */
   overlap = 1;  /* 1 layer overlap */
   domain_type = 2; /* through agglomeration */
   schwarz_rlx_weight = 1.;

   /* defaults for GMRES */

   k_dim = 5;

   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-k") == 0 )
      {
         arg_index++;
         k_dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-w") == 0 )
      {
         arg_index++;
         if (solver_id == 0 || solver_id == 1 || solver_id == 3 
             || solver_id == 5 )
         {
            relax_weight[0] = atof(argv[arg_index++]);
            for (i=1; i < max_levels; i++)
            {
               relax_weight[i] = relax_weight[0];
            }
         }
      }
      else if ( strcmp(argv[arg_index], "-sw") == 0 )
      {
         arg_index++;
         schwarz_rlx_weight = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-th") == 0 )
      {
         arg_index++;
         strong_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mxrs") == 0 )
      {
         arg_index++;
         max_row_sum  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sai_th") == 0 )
      {
         arg_index++;
         sai_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sai_filt") == 0 )
      {
         arg_index++;
         sai_filter  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-drop_tol") == 0 )
      {
         arg_index++;
         drop_tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nonzeros_to_keep") == 0 )
      {
         arg_index++;
         nonzeros_to_keep  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tr") == 0 )
      {
         arg_index++;
         trunc_factor  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-iout") == 0 )
      {
         arg_index++;
         ioutdat  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pout") == 0 )
      {
         arg_index++;
         poutdat  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-var") == 0 )
      {
         arg_index++;
         variant  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ov") == 0 )
      {
         arg_index++;
         overlap  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dom") == 0 )
      {
         arg_index++;
         domain_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mu") == 0 )
      {
         arg_index++;
         cycle_type  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/
 
   if ( (print_usage) && (myid == 0) )
   {
      printf("\n");
      printf("Usage: %s [<options>]\n", argv[0]);
      printf("\n");
      printf("  -fromijfile <filename>     : ");
      printf("matrix read in IJ format from distributed files\n");
      printf("  -fromparcsrfile <filename> : ");
      printf("matrix read in ParCSR format from distributed files\n");
      printf("  -fromonecsrfile <filename> : ");
      printf("matrix read in CSR format from a file on one processor\n");
      printf("\n");
      printf("  -laplacian [<options>] : build 5pt 2D laplacian problem (default) \n");
      printf(" only the default is supported at present\n" );
      printf("  -9pt [<opts>]          : build 9pt 2D laplacian problem\n");
      printf("  -27pt [<opts>]         : build 27pt 3D laplacian problem\n");
      printf("  -difconv [<opts>]      : build convection-diffusion problem\n");
      printf("    -n <nx> <ny> <nz>    : total problem size \n");
      printf("    -P <Px> <Py> <Pz>    : processor topology\n");
      printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
      printf("    -a <ax> <ay> <az>    : convection coefficients\n");
      printf("\n");
      printf("  -exact_size            : inserts immediately into ParCSR structure\n");
      printf("  -storage_low           : allocates not enough storage for aux struct\n");
      printf("  -concrete_parcsr       : use parcsr matrix type as concrete type\n");
      printf("\n");
      printf("  -rhsfromfile           : rhs read in IJ form from distributed files\n");
      printf("  -rhsfromonefile        : rhs read from a file one one processor\n");
      printf("  -rhsrand               : rhs is random vector\n");
      printf("  -rhsisone              : rhs is vector with unit components (default)\n");
      printf(" only the default is supported at present\n" );
      printf("  -xisone                : solution of all ones\n");
      printf("  -rhszero               : rhs is zero vector\n");
      printf("\n");
      printf(" the backward Euler and src options are not supported yet\n");
#ifdef DO_THIS_LATER
      printf("  -dt <val>              : specify finite backward Euler time step\n");
      printf("                         :    -rhsfromfile, -rhsfromonefile, -rhsrand,\n");
      printf("                         :    -rhsrand, or -xisone will be ignored\n");
      printf("  -srcfromfile           : backward Euler source read in IJ form from distributed files\n");
      printf("  -srcfromonefile        : ");
      printf("backward Euler source read from a file on one processor\n");
      printf("  -srcrand               : ");
      printf("backward Euler source is random vector with components in range 0 - 1\n");
      printf("  -srcisone              : ");
      printf("backward Euler source is vector with unit components (default)\n");
      printf("  -srczero               : ");
      printf("backward Euler source is zero-vector\n");
      printf("\n");
#endif /* DO_THIS_LATER */
      printf("  -solver <ID>           : solver ID\n");
      printf("        0=AMG                1=AMG-PCG        \n");
      printf("        2=DS-PCG             3=AMG-GMRES      \n");
      printf("        4=DS-GMRES           5=AMG-CGNR       \n");     
      printf("        6=DS-CGNR            7*=PILUT-GMRES    \n");     
      printf("        8=ParaSails-PCG      9*=AMG-BiCGSTAB   \n");
      printf("       10=DS-BiCGSTAB       11*=PILUT-BiCGSTAB \n");
      printf("       12=Schwarz-PCG      18=ParaSails-GMRES\n");     
      printf("        43=Euclid-PCG       44*=Euclid-GMRES   \n");
      printf("       45*=Euclid-BICGSTAB\n");
      printf("Solvers marked with '*' have not yet been implemented.\n");
      printf("   -hpcg 1               : for HYPRE-interface version of PCG or GMRES solver\n");
      printf("\n");
      printf("   -cljp                 : CLJP coarsening \n");
      printf("   -ruge                 : Ruge coarsening (local)\n");
      printf("   -ruge3                : third pass on boundary\n");
      printf("   -ruge3c               : third pass on boundary, keep c-points\n");
      printf("   -ruge2b               : 2nd pass is global\n");
      printf("   -rugerlx              : relaxes special points\n");
      printf("   -falgout              : local ruge followed by LJP\n");
      printf("   -nohybrid             : no switch in coarsening\n");
      printf("   -gm                   : use global measures\n");
      printf("\n");
      printf("  -rlx  <val>            : relaxation type\n");
      printf("       0=Weighted Jacobi  \n");
      printf("       1=Gauss-Seidel (very slow!)  \n");
      printf("       3=Hybrid Jacobi/Gauss-Seidel  \n");
      printf("  -ns <val>              : Use <val> sweeps on each level\n");
      printf("                           (default C/F down, F/C up, F/C fine\n");
      printf("\n"); 
      printf("  -mu   <val>            : set AMG cycles (1=V, 2=W, etc.)\n"); 
      printf("  -th   <val>            : set AMG threshold Theta = val \n");
      printf("  -tr   <val>            : set AMG interpolation truncation factor = val \n");
      printf("  -mxrs <val>            : set AMG maximum row sum threshold for dependency weakening \n");
      printf("  -nf <val>              : set number of functions for systems AMG\n");
     
      printf("  -w   <val>             : set Jacobi relax weight = val\n");
      printf("  -k   <val>             : dimension Krylov space for GMRES\n");
      printf("  -mxl  <val>            : maximum number of levels (AMG, ParaSAILS)\n");
      printf("  -tol  <val>            : set solver convergence tolerance = val\n");
      printf("\n");
      printf("  -sai_th   <val>        : set ParaSAILS threshold = val \n");
      printf("  -sai_filt <val>        : set ParaSAILS filter = val \n");
      printf("\n");  
      printf("  -drop_tol  <val>       : set threshold for dropping in PILUT\n");
      printf("  -nonzeros_to_keep <val>: number of nonzeros in each row to keep\n");
      printf("\n");  
      printf("  -iout <val>            : set output flag\n");
      printf("       0=no output    1=matrix stats\n"); 
      printf("       2=cycle stats  3=matrix & cycle stats\n"); 
      printf("\n");  
      printf("  -dbg <val>             : set debug flag\n");
      printf("       0=no debugging\n       1=internal timing\n       2=interpolation truncation\n       3=more detailed timing in coarsening routine\n");

      bHYPRE_MPICommunicator_deleteRef( bmpicomm );
      MPI_Finalize();
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("Running with these driver parameters:\n");
      printf("  solver ID    = %d\n\n", solver_id);
   }

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/

   if ( myid == 0 && dt != dt_inf)
   {
      printf("  Backward Euler time step with dt = %e\n", dt);
      printf("  Dirichlet 0 BCs are implicit in the spatial operator\n");
   }

   /*
   bBuildParLaplacian and the function it calls, bHYPRE_IJMatrix_GenerateLaplacian,
   include part of what is (and should be) called "matrix setup" in ij.c,
   but they also include matrix values computation which is not considered part
   of "setup".  So here we do the values computation alone, just so we'll know the non-setup
   part of the setup timing computation done below.  Hypre timing functions
   don't have a subtraction feature, so you the "user" will have to do it yourself.
   */

   time_index = hypre_InitializeTiming("LaplacianComputation");
   hypre_BeginTiming(time_index);
   if ( build_matrix_type == 2 )
   {
      BuildParLaplacian(argc, argv, build_matrix_arg_index, &parcsr_A);
      HYPRE_ParCSRMatrixDestroy( parcsr_A );
   }
   else
   {
      printf("timing only correct for build_matrix_type==2\n");
   }
   hypre_EndTiming(time_index);
   hypre_PrintTiming( "Laplacian Computation, deduct from Matrix Setup", mpi_comm );
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   time_index = hypre_InitializeTiming("Spatial operator");
   hypre_BeginTiming(time_index);

   if ( build_matrix_type == -1 )

   {
#ifdef DO_THIS_LATER
      HYPRE_IJMatrixRead( argv[build_matrix_arg_index], comm,
                          HYPRE_PARCSR, &ij_A );
#endif /* DO_THIS_LATER */
   }
   else if ( build_matrix_type == 0 )
   {
#ifdef DO_THIS_LATER
      BuildParFromFile(argc, argv, build_matrix_arg_index, &parcsr_A);
#endif /* DO_THIS_LATER */
   }
   else if ( build_matrix_type == 1 )
   {
#ifdef DO_THIS_LATER
      BuildParFromOneFile(argc, argv, build_matrix_arg_index, &parcsr_A);
#endif /* DO_THIS_LATER */
   }
   else if ( build_matrix_type == 2 )
   {
      bBuildParLaplacian(argc, argv, build_matrix_arg_index, bmpicomm, &bHYPRE_parcsr_A);
   }
   else if ( build_matrix_type == 3 )
   {
#ifdef DO_THIS_LATER
      BuildParLaplacian9pt(argc, argv, build_matrix_arg_index, &parcsr_A);
#endif /* DO_THIS_LATER */
   }
   else if ( build_matrix_type == 4 )
   {
#ifdef DO_THIS_LATER
      BuildParLaplacian27pt(argc, argv, build_matrix_arg_index, &parcsr_A);
#endif /* DO_THIS_LATER */
   }
   else if ( build_matrix_type == 5 )
   {
#ifdef DO_THIS_LATER
      BuildParDifConv(argc, argv, build_matrix_arg_index, &parcsr_A);
#endif /* DO_THIS_LATER */
   }
   else
   {
      printf("You have asked for an unsupported problem with\n");
      printf("build_matrix_type = %d.\n", build_matrix_type);
      return(-1);
   }

   ierr += bHYPRE_IJParCSRMatrix_GetLocalRange(
      bHYPRE_parcsr_A, &first_local_row, &last_local_row,
      &first_local_col, &last_local_col );
   local_num_rows = last_local_row - first_local_row + 1;
   local_num_cols = last_local_col - first_local_col +1;
   ierr += bHYPRE_IJParCSRMatrix_GetIntValue( bHYPRE_parcsr_A,
                                              "GlobalNumRows", &M );
   ierr += bHYPRE_IJParCSRMatrix_GetIntValue( bHYPRE_parcsr_A,
                                              "GlobalNumCols", &N );

   ierr += bHYPRE_IJParCSRMatrix_Assemble( bHYPRE_parcsr_A );


   hypre_EndTiming(time_index);
   hypre_PrintTiming("IJ Matrix Setup", MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   if (ierr)
   {
      printf("Error in driver building IJMatrix from parcsr matrix. \n");
      return(-1);
   }

   /* This is to emphasize that one can IJMatrixAddToValues after an
      IJMatrixRead or an IJMatrixAssemble.  After an IJMatrixRead,
      assembly is unnecessary if the sparsity pattern of the matrix is
      not changed somehow.  If one has not used IJMatrixRead, one has
      the opportunity to IJMatrixAddTo before a IJMatrixAssemble. */

   ncols    = hypre_CTAlloc(int, last_local_row - first_local_row + 1);
   rows     = hypre_CTAlloc(int, last_local_row - first_local_row + 1);
   col_inds = hypre_CTAlloc(int, last_local_row - first_local_row + 1);
   values   = hypre_CTAlloc(double, last_local_row - first_local_row + 1);
   
   if (dt < dt_inf)
      val = 1./dt;
   else 
      val = 0.;  /* Use zero to avoid unintentional loss of significance */

   for (i = first_local_row; i <= last_local_row; i++)
   {
      j = i - first_local_row;
      rows[j] = i;
      ncols[j] = 1;
      col_inds[j] = i;
      values[j] = val;
   }
      
   ierr += bHYPRE_IJParCSRMatrix_AddToValues
      ( bHYPRE_parcsr_A, local_num_rows, ncols, rows, col_inds, values, local_num_rows );

   hypre_TFree(values);
   hypre_TFree(col_inds);
   hypre_TFree(rows);
   hypre_TFree(ncols);


   /* If sparsity pattern is not changed since last IJMatrixAssemble call,
      this should be a no-op */

   ierr += bHYPRE_IJParCSRMatrix_Assemble( bHYPRE_parcsr_A );

   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("RHS and Initial Guess");
   hypre_BeginTiming(time_index);

   if ( build_rhs_type == 0 )
   {
#ifdef DO_THIS_LATER
      if (myid == 0)
      {
         printf("  RHS vector read from file %s\n", argv[build_rhs_arg_index]);
         printf("  Initial guess is 0\n");
      }

/* RHS */
      ierr = HYPRE_IJVectorRead( argv[build_rhs_arg_index], mpi_comm, 
                                 HYPRE_PARCSR, &ij_b );
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

/* Initial guess */
      HYPRE_IJVectorCreate(mpi_comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
#endif /* DO_THIS_LATER */
}
   else if ( build_rhs_type == 1 )
   {
      printf("build_rhs_type == 1 not currently implemented\n");
      return(-1);
#if 0
/* RHS */
      BuildRhsParFromOneFile(argc, argv, build_rhs_arg_index, part_b, &b);
#endif
   }
   else if ( build_rhs_type == 2 )
   {
      if (myid == 0)
      {
         printf("  RHS vector has unit components\n");
         printf("  Initial guess is 0\n");
      }

/* RHS */
      bHYPRE_b = bHYPRE_IJParCSRVector_Create( bmpicomm,
                                               first_local_row,
                                               last_local_row );

      ierr += bHYPRE_IJParCSRVector_Initialize( bHYPRE_b );


      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      bHYPRE_IJParCSRVector_SetValues( bHYPRE_b, local_num_rows, NULL, values );
      hypre_TFree(values);

      ierr += bHYPRE_IJParCSRVector_Assemble( bHYPRE_b );


/* Initial guess */
      bHYPRE_x = bHYPRE_IJParCSRVector_Create( bmpicomm,
                                               first_local_row,
                                               last_local_row );

      ierr += bHYPRE_IJParCSRVector_Initialize( bHYPRE_x );

      values = hypre_CTAlloc(double, local_num_cols);
      for ( i=0; i<local_num_cols; ++i )
         values[i] = 0.;
      bHYPRE_IJParCSRVector_SetValues( bHYPRE_x, local_num_cols,
                                       NULL, values );
      hypre_TFree(values);

      ierr += bHYPRE_IJParCSRVector_Assemble( bHYPRE_x );

   }
   else if ( build_rhs_type == 3 )
   {
#ifdef DO_THIS_LATER
      if (myid == 0)
      {
         printf("  RHS vector has random components and unit 2-norm\n");
         printf("  Initial guess is 0\n");
      }

/* RHS */
      HYPRE_IJVectorCreate(mpi_comm, first_local_row, last_local_row, &ij_b); 
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

/* For purposes of this test, HYPRE_ParVector functions are used, but these are 
   not necessary.  For a clean use of the interface, the user "should"
   modify components of ij_x by using functions HYPRE_IJVectorSetValues or
   HYPRE_IJVectorAddToValues */

      HYPRE_ParVectorSetRandomValues(b, 22775);
      HYPRE_ParVectorInnerProd(b,b,&norm);
      norm = 1./sqrt(norm);
      ierr = HYPRE_ParVectorScale(norm, b);      

/* Initial guess */
      HYPRE_IJVectorCreate(mpi_comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
#endif /* DO_THIS_LATER */
   }
   else if ( build_rhs_type == 4 )
   {
#ifdef DO_THIS_LATER
      if (myid == 0)
      {
         printf("  RHS vector set for solution with unit components\n");
         printf("  Initial guess is 0\n");
      }

/* Temporary use of solution vector */
      HYPRE_IJVectorCreate(mpi_comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 1.;
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;

/* RHS */
      HYPRE_IJVectorCreate(mpi_comm, first_local_row, last_local_row, &ij_b); 
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      HYPRE_ParCSRMatrixMatvec(1.,parcsr_A,x,0.,b);

/* Initial guess */
      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);
#endif /* DO_THIS_LATER */
   }
   else if ( build_rhs_type == 5 )
   {
#ifdef DO_THIS_LATER
      if (myid == 0)
      {
         printf("  RHS vector is 0\n");
         printf("  Initial guess has unit components\n");
      }

/* RHS */
      HYPRE_IJVectorCreate(mpi_comm, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);

      values = hypre_CTAlloc(double, local_num_rows);
      for (i = 0; i < local_num_rows; i++)
      {
         values[i] = 0.;
      }
      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

/* Initial guess */
      HYPRE_IJVectorCreate(mpi_comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 1.;
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
#endif /* DO_THIS_LATER */
   }

#ifdef DO_THIS_LATER
   if ( build_src_type == 0 )
   {
#if 0
/* RHS */
      BuildRhsParFromFile(argc, argv, build_src_arg_index, &b);
#endif

      if (myid == 0)
      {
         printf("  Source vector read from file %s\n", argv[build_src_arg_index]);
         printf("  Initial unknown vector in evolution is 0\n");
      }

      ierr = HYPRE_IJVectorRead( argv[build_src_arg_index], mpi_comm, 
                                 HYPRE_PARCSR, &ij_b );

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

/* Initial unknown vector */
      HYPRE_IJVectorCreate(mpi_comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_src_type == 1 )
   {
      printf("build_src_type == 1 not currently implemented\n");
      return(-1);

#if 0
      BuildRhsParFromOneFile(argc, argv, build_src_arg_index, part_b, &b);
#endif
   }
   else if ( build_src_type == 2 )
   {
      if (myid == 0)
      {
         printf("  Source vector has unit components\n");
         printf("  Initial unknown vector is 0\n");
      }

/* RHS */
      HYPRE_IJVectorCreate(mpi_comm, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);

      values = hypre_CTAlloc(double, local_num_rows);
      for (i = 0; i < local_num_rows; i++)
      {
         values[i] = 1.;
      }
      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

/* Initial guess */
      HYPRE_IJVectorCreate(mpi_comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

/* For backward Euler the previous backward Euler iterate (assumed
   0 here) is usually used as the initial guess */
      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_src_type == 3 )
   {
      if (myid == 0)
      {
         printf("  Source vector has random components in range 0 - 1\n");
         printf("  Initial unknown vector is 0\n");
      }

/* RHS */
      HYPRE_IJVectorCreate(mpi_comm, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);
      values = hypre_CTAlloc(double, local_num_rows);

      hypre_SeedRand(myid);
      for (i = 0; i < local_num_rows; i++)
      {
         values[i] = hypre_Rand();
      }

      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

/* Initial guess */
      HYPRE_IJVectorCreate(mpi_comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

/* For backward Euler the previous backward Euler iterate (assumed
   0 here) is usually used as the initial guess */
      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_src_type == 4 )
   {
      if (myid == 0)
      {
         printf("  Source vector is 0 \n");
         printf("  Initial unknown vector has random components in range 0 - 1\n");
      }

/* RHS */
      HYPRE_IJVectorCreate(mpi_comm, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);

      values = hypre_CTAlloc(double, local_num_rows);
      hypre_SeedRand(myid);
      for (i = 0; i < local_num_rows; i++)
      {
         values[i] = hypre_Rand()/dt;
      }
      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

/* Initial guess */
      HYPRE_IJVectorCreate(mpi_comm, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

/* For backward Euler the previous backward Euler iterate (assumed
   random in 0 - 1 here) is usually used as the initial guess */
      values = hypre_CTAlloc(double, local_num_cols);
      hypre_SeedRand(myid);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = hypre_Rand();
      }
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
#endif /* DO_THIS_LATER */

   hypre_EndTiming(time_index);
   hypre_PrintTiming("IJ Vector Setup", mpi_comm);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   
   bHYPRE_IJParCSRMatrix_Print( bHYPRE_parcsr_A, "driver.out.HA");
   bHYPRE_IJParCSRVector_Print( bHYPRE_b, "driver.out.Hb0");
   bHYPRE_IJParCSRVector_Print( bHYPRE_x, "driver.out.Hx0");

   if (num_functions > 1)
   {
      dof_func = NULL;
      if (build_funcs_type == 1)
      {
	 BuildFuncsFromOneFile(argc, argv, build_funcs_arg_index, bHYPRE_parcsr_A, &dof_func);
      }
      else if (build_funcs_type == 2)
      {
	 BuildFuncsFromFiles(argc, argv, build_funcs_arg_index, bHYPRE_parcsr_A, &dof_func);
      }
      else
      {
         local_num_vars = local_num_rows;
         dof_func = hypre_CTAlloc(int,local_num_vars);
         if (myid == 0)
	    printf (" Number of unknown functions = %d \n", num_functions);
         rest = first_local_row-((first_local_row/num_functions)*num_functions);
         indx = num_functions-rest;
         if (rest == 0) indx = 0;
         k = num_functions - 1;
         for (j = indx-1; j > -1; j--)
         {
	    dof_func[j] = k--;
         }
         tms = local_num_vars/num_functions;
         if (tms*num_functions+indx > local_num_vars) tms--;
         for (j=0; j < tms; j++)
         {
	    for (k=0; k < num_functions; k++)
            {
	       dof_func[indx++] = k;
            }
         }
         k = 0;
         while (indx < local_num_vars)
	    dof_func[indx++] = k++;
      }
   }
 

   /*-----------------------------------------------------------
    * Matrix-Vector and Vector Operation Debugging code begun by adapting
    * from Rob Falgout's sstruct tests
    *-----------------------------------------------------------*/

#define HYPRE_IJMV_DEBUG 1
   {
#if HYPRE_IJMV_DEBUG
                       
      /*  Apply, y=A*b: result is 1's on the interior of the grid */
      bHYPRE_y = bHYPRE_IJParCSRVector_Create( bmpicomm,
                                               first_local_col,
                                               last_local_col );
      ierr += bHYPRE_IJParCSRVector_Initialize( bHYPRE_y );
      y = bHYPRE_Vector__cast( bHYPRE_y );

      bHYPRE_IJParCSRMatrix_Apply( bHYPRE_parcsr_A,
                                   bHYPRE_Vector__cast( bHYPRE_b ),
                                   &y );

      bHYPRE_IJParCSRMatrix_Print( bHYPRE_parcsr_A, "test.A" );
      bHYPRE_IJParCSRVector_Print( bHYPRE_y, "test.apply" );
      bHYPRE_Vector_deleteRef( y );

      /* SetValues, x=1; result is all 1's */
      indices = hypre_CTAlloc(int, local_num_cols);
      values = hypre_CTAlloc(double, local_num_cols);
      for ( i=0; i<local_num_cols; ++i )
      {
         indices[i] = i+first_local_col;
         values[i] = 1.0;
      }
      bHYPRE_IJParCSRVector_SetValues( bHYPRE_x, local_num_cols, indices, values );
      hypre_TFree(indices);
      hypre_TFree(values);
      bHYPRE_IJParCSRVector_Print( bHYPRE_x, "test.setvalues" );

      /* Copy, b=x; result is all 1's */
      bHYPRE_Vector_x = bHYPRE_Vector__cast( bHYPRE_x );
      bHYPRE_IJParCSRVector_Copy( bHYPRE_b, bHYPRE_Vector_x );
      bHYPRE_IJParCSRVector_Print( bHYPRE_b, "test.copy" );

      /* Clone y=b; result is all 1's */
      bHYPRE_IJParCSRVector_Clone( bHYPRE_b, &y );
      bHYPRE_y = bHYPRE_IJParCSRVector__cast( y );
      bHYPRE_IJParCSRVector_Print( bHYPRE_y, "test.clone" );
      bHYPRE_Vector_deleteRef( y );

      /* Read y2=y; result is all 1's */
      bHYPRE_y2 = bHYPRE_IJParCSRVector_Create( bmpicomm,
                                                first_local_col,
                                                last_local_col );
      ierr += bHYPRE_IJParCSRVector_Initialize( bHYPRE_y2 );
      bHYPRE_IJParCSRVector_Read( bHYPRE_y2, "test.clone", bmpicomm );
      bHYPRE_IJParCSRVector_Print( bHYPRE_y2, "test.read" );

      bHYPRE_IJParCSRVector_deleteRef( bHYPRE_y2 );

      /* Scale, x=2*x; result is all 2's */
      bHYPRE_IJParCSRVector_Scale( bHYPRE_x, 2.0 );
      bHYPRE_IJParCSRVector_Print( bHYPRE_x, "test.scale" );

      /* Dot, tmp = b.x; at this point all b[i]==1, all x[i]==2 */
      bHYPRE_IJParCSRVector_Dot( bHYPRE_b, bHYPRE_Vector_x, &tmp );
      hypre_assert( tmp==2*N );

      /* Axpy, b=b-0.5*x; result is all 0's */
      bHYPRE_IJParCSRVector_Axpy( bHYPRE_b, -0.5, bHYPRE_Vector_x );
      bHYPRE_IJParCSRVector_Print( bHYPRE_b, "test.axpy" );

      /* tested by other parts of this driver program: ParCSRVector_GetObject */

      /* Clear and AddToValues, b=1, which restores its initial value of 1 */
      indices = hypre_CTAlloc(int, local_num_cols);
      values = hypre_CTAlloc(double, local_num_cols);
      for ( i=0; i<local_num_cols; ++i )
      {
         indices[i] = i+first_local_col;
         values[i] = 1.0;
      }
      bHYPRE_IJParCSRVector_Clear( bHYPRE_b );
      bHYPRE_IJParCSRVector_AddToValues
         ( bHYPRE_b, local_num_cols, indices, values );
      hypre_TFree(indices);
      hypre_TFree(values);
      bHYPRE_IJParCSRVector_Print( bHYPRE_b, "test.addtovalues" );

      /* Clear,x=0, which restores its initial value of 0 */
      bHYPRE_IJParCSRVector_Clear( bHYPRE_x );
      bHYPRE_IJParCSRVector_Print( bHYPRE_x, "test.clear" );
#endif
   }


   /*-----------------------------------------------------------
    * Solve the system using AMG
    *-----------------------------------------------------------*/

   if (solver_id == 0)
   {
      if (myid == 0) printf("Solver:  AMG\n");
      time_index = hypre_InitializeTiming("BoomerAMG Setup");
      hypre_BeginTiming(time_index);


      /* To call a bHYPRE solver:
         create, set comm, set operator, set other parameters,
         Setup (noop in this case), Apply */
      bHYPRE_op_A = bHYPRE_Operator__cast( bHYPRE_parcsr_A );
      bHYPRE_AMG = bHYPRE_BoomerAMG_Create( bmpicomm, bHYPRE_parcsr_A );
      bHYPRE_Vector_b = bHYPRE_Vector__cast( bHYPRE_b );
      bHYPRE_Vector_x = bHYPRE_Vector__cast( bHYPRE_x );

      bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "Tolerance", tol);
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "PrintLevel", ioutdat ); 

      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CoarsenType",
                                        (hybrid*coarsen_type));
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MeasureType",
                                        measure_type);
      bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "StrongThreshold",
                                           strong_threshold);
      bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "TruncFactor",
                                           trunc_factor);
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CycleType", cycle_type);
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0NumSweeps",
                                        num_grid_sweeps[0] );
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1NumSweeps",
                                        num_grid_sweeps[1] );
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2NumSweeps",
                                        num_grid_sweeps[2] );
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3NumSweeps",
                                        num_grid_sweeps[3] );
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0RelaxType",
                                        grid_relax_type[0] );
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1RelaxType",
                                        grid_relax_type[1] );
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2RelaxType",
                                        grid_relax_type[2] );
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3RelaxType",
                                        grid_relax_type[3] );
      for ( i=0; i<max_levels; ++i )
      {
         bHYPRE_BoomerAMG_SetLevelRelaxWt( bHYPRE_AMG, relax_weight[i], i );
      }
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothType",
                                        smooth_type );
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothNumSweeps",
                                        smooth_num_sweep);
      dimsl[0] = 0;   dimsl[1] = 0;   dimsu[0] = 4;   dimsu[1] = 4;
      bHYPRE_grid_relax_points = sidl_int__array_createCol( 2, dimsl, dimsu );
      for ( i=0; i<4; ++i )
      {
         for ( j=0; j<num_grid_sweeps[i]; ++j )
         {
            sidl_int__array_set2( bHYPRE_grid_relax_points, i, j,
                                  grid_relax_points[i][j] );
         }
      }
      bHYPRE_BoomerAMG_SetIntArray2Parameter( bHYPRE_AMG, "GridRelaxPoints",
                                              bHYPRE_grid_relax_points );
      sidl_int__array_deleteRef( bHYPRE_grid_relax_points );


      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MaxLevels", max_levels);
      bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "MaxRowSum",
                                           max_row_sum);
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "DebugFlag", debug_flag);
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Variant", variant);
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Overlap", overlap);
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "DomainType", domain_type);
      bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG,
                                           "SchwarzRlxWeight",
                                           schwarz_rlx_weight);
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "NumFunctions",
                                        num_functions);
      if (num_functions > 1)
      {
	 bHYPRE_BoomerAMG_SetIntArray1Parameter( bHYPRE_AMG, "DOFFunc",
                                                 dof_func, num_functions );
      }
      log_level = 3;
      bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Logging", log_level );

      ierr += bHYPRE_BoomerAMG_Setup( bHYPRE_AMG, bHYPRE_Vector_b,
                                      bHYPRE_Vector_x );
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      time_index = hypre_InitializeTiming("BoomerAMG Solve");
      hypre_BeginTiming(time_index);

      ierr += bHYPRE_BoomerAMG_Apply( bHYPRE_AMG, bHYPRE_Vector_b, &bHYPRE_Vector_x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      bHYPRE_BoomerAMG_deleteRef( bHYPRE_AMG );

   }

   /*-----------------------------------------------------------
    * Solve the system using PCG 
    *-----------------------------------------------------------*/

   if (solver_id == 1 || solver_id == 2 || solver_id == 8 || 
       solver_id == 12 || solver_id == 43)
      if ( hpcg==0 )
   {

      ioutdat = 2;
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);
 
      bHYPRE_op_A = bHYPRE_Operator__cast( bHYPRE_parcsr_A );
      bHYPRE_PCG = bHYPRE_PCG_Create( bmpicomm, bHYPRE_op_A );
      bHYPRE_Vector_b = bHYPRE_Vector__cast( bHYPRE_b );
      bHYPRE_Vector_x = bHYPRE_Vector__cast( bHYPRE_x );

      bHYPRE_PCG_SetIntParameter( bHYPRE_PCG, "MaxIterations", 500);
      bHYPRE_PCG_SetDoubleParameter( bHYPRE_PCG, "Tolerance", tol);
      bHYPRE_PCG_SetIntParameter( bHYPRE_PCG, "TwoNorm", 1 );
      bHYPRE_PCG_SetIntParameter( bHYPRE_PCG, "RelChange", 0 );
      bHYPRE_PCG_SetIntParameter( bHYPRE_PCG, "PrintLevel", ioutdat );

      if (solver_id == 1)
      {
         /* use BoomerAMG as preconditioner */
	 ioutdat = 1;
         if (myid == 0) printf("Solver: AMG-PCG\n");
         bHYPRE_AMG = bHYPRE_BoomerAMG_Create( bmpicomm, bHYPRE_parcsr_A );
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "Tolerance", pc_tol);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CoarsenType",
                                        (hybrid*coarsen_type));
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MeasureType",
                                           measure_type);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "StrongThreshold",
                                              strong_threshold);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "TruncFactor",
                                              trunc_factor);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "PrintLevel", poutdat );
         bHYPRE_BoomerAMG_SetStringParameter( bHYPRE_AMG, "PrintFileName",
                                              "driver.out.log" );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MaxIter", 1 );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CycleType", cycle_type );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0NumSweeps",
                                           num_grid_sweeps[0] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1NumSweeps",
                                           num_grid_sweeps[1] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2NumSweeps",
                                           num_grid_sweeps[2] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3NumSweeps",
                                           num_grid_sweeps[3] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0RelaxType",
                                           grid_relax_type[0] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1RelaxType",
                                           grid_relax_type[1] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2RelaxType",
                                           grid_relax_type[2] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3RelaxType",
                                           grid_relax_type[3] );
         for ( i=0; i<max_levels; ++i )
         {
            bHYPRE_BoomerAMG_SetLevelRelaxWt( bHYPRE_AMG, relax_weight[i], i );
         }
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothType",
                                           smooth_type );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothNumSweeps",
                                           smooth_num_sweep);

         dimsl[0] = 0;   dimsl[1] = 0;   dimsu[0] = 4;   dimsu[1] = 4;
         bHYPRE_grid_relax_points = sidl_int__array_createCol( 2, dimsl, dimsu );
         for ( i=0; i<4; ++i )
         {
            for ( j=0; j<num_grid_sweeps[i]; ++j )
            {
               sidl_int__array_set2( bHYPRE_grid_relax_points, i, j,
                                     grid_relax_points[i][j] );
            }
         }
         bHYPRE_BoomerAMG_SetIntArray2Parameter( bHYPRE_AMG, "GridRelaxPoints",
                                                 bHYPRE_grid_relax_points );
         sidl_int__array_deleteRef( bHYPRE_grid_relax_points );

         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MaxLevels", max_levels);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "MaxRowSum",
                                              max_row_sum);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "NumFunctions",
                                           num_functions);
         if (num_functions > 1)
         {
            bHYPRE_BoomerAMG_SetIntArray1Parameter( bHYPRE_AMG, "DOFFunc",
                                                    dof_func, num_functions );
         }
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Variant", variant);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Overlap", overlap);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "DomainType", domain_type);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG,
                                              "SchwarzRlxWeight",
                                              schwarz_rlx_weight);
         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_AMG );
         ierr += bHYPRE_PCG_SetPreconditioner( bHYPRE_PCG, bHYPRE_SolverPC );
         ierr += bHYPRE_PCG_Setup( bHYPRE_PCG, bHYPRE_Vector_b, bHYPRE_Vector_x );

      }

      else if (solver_id == 2)
      {
         /* use diagonal scaling as preconditioner */

         /* To call a bHYPRE solver:
            create, set comm, set operator, set other parameters,
            Setup (noop in this case), Apply */
         bHYPRE_ParCSRDiagScale = bHYPRE_ParCSRDiagScale_Create(
            bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParCSRDiagScale_Setup( bHYPRE_ParCSRDiagScale,
                                               bHYPRE_Vector_b, bHYPRE_Vector_x );
         bHYPRE_SolverPC =
            bHYPRE_Solver__cast( bHYPRE_ParCSRDiagScale );
         ierr += bHYPRE_PCG_SetPreconditioner( bHYPRE_PCG, bHYPRE_SolverPC );
         ierr += bHYPRE_PCG_Setup( bHYPRE_PCG, bHYPRE_Vector_b, bHYPRE_Vector_x );

      }
      else if (solver_id == 8)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) printf("Solver: ParaSails-PCG\n");

         bHYPRE_ParaSails = bHYPRE_ParaSails_Create( bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParaSails_SetDoubleParameter( bHYPRE_ParaSails, "Thresh",
                                                      sai_threshold );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Nlevels",
                                                   max_levels );
         ierr += bHYPRE_ParaSails_SetDoubleParameter( bHYPRE_ParaSails, "Filter",
                                                      sai_filter );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Logging",
                                                   ioutdat );
         hypre_assert( ierr==0 );
         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_ParaSails );
         ierr += bHYPRE_PCG_SetPreconditioner( bHYPRE_PCG, bHYPRE_SolverPC );
         ierr += bHYPRE_PCG_Setup( bHYPRE_PCG, bHYPRE_Vector_b, bHYPRE_Vector_x );

      }
      else if (solver_id == 12)
      {
         /* use Schwarz preconditioner */
         if (myid == 0) printf("Solver: Schwarz-PCG\n");
         bHYPRE_Schwarz = bHYPRE_Schwarz_Create( bHYPRE_parcsr_A );
         ierr += bHYPRE_Schwarz_SetIntParameter(
            bHYPRE_Schwarz, "Variant", variant );
         ierr += bHYPRE_Schwarz_SetIntParameter(
            bHYPRE_Schwarz, "Overlap", overlap );
         ierr += bHYPRE_Schwarz_SetIntParameter(
            bHYPRE_Schwarz, "DomainType", domain_type );
         ierr += bHYPRE_Schwarz_SetDoubleParameter(
            bHYPRE_Schwarz, "RelaxWeight", schwarz_rlx_weight );
         hypre_assert( ierr==0 );
         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_Schwarz );
         ierr += bHYPRE_PCG_SetPreconditioner( bHYPRE_PCG, bHYPRE_SolverPC );
         ierr += bHYPRE_PCG_Setup( bHYPRE_PCG, bHYPRE_Vector_b, bHYPRE_Vector_x );
      }
      else if (solver_id == 43)
      {
         /* use Euclid preconditioning */
         if (myid == 0) printf("Solver: Euclid-PCG\n");

         bHYPRE_Euclid = bHYPRE_Euclid_Create( bmpicomm, bHYPRE_parcsr_A );

         /* note: There are three three methods of setting run-time 
            parameters for Euclid: (see HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally 
            parse the command line.
         */
         /*ierr += bHYPRE_Euclid_SetIntParameter( bHYPRE_Euclid, "-eu_stats", 1 );*/
         ierr += bHYPRE_Euclid_SetParameters( bHYPRE_Euclid, argc, argv );

         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_Euclid );
         ierr += bHYPRE_PCG_SetPreconditioner( bHYPRE_PCG, bHYPRE_SolverPC );
         ierr += bHYPRE_PCG_Setup( bHYPRE_PCG, bHYPRE_Vector_b, bHYPRE_Vector_x );
      }
 

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      ierr += bHYPRE_PCG_Apply( bHYPRE_PCG, bHYPRE_Vector_b, &bHYPRE_Vector_x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      ierr += bHYPRE_PCG_GetIntValue( bHYPRE_PCG, "NumIterations",
                                      &num_iterations );
      ierr += bHYPRE_PCG_GetDoubleValue( bHYPRE_PCG, "Final Relative Residual Norm",
                                         &final_res_norm );

      bHYPRE_PCG_deleteRef( bHYPRE_PCG );
      if ( solver_id == 1 )
      {
         bHYPRE_BoomerAMG_deleteRef( bHYPRE_AMG );
      }
      else if ( solver_id == 2 )
      {
         bHYPRE_ParCSRDiagScale_deleteRef( bHYPRE_ParCSRDiagScale );
      }
      else if (solver_id == 8)
      {
         bHYPRE_ParaSails_deleteRef( bHYPRE_ParaSails );
      }
      else if (solver_id == 12)
      {
         bHYPRE_Schwarz_deleteRef( bHYPRE_Schwarz );
      }
      else if (solver_id == 43)
      {
         bHYPRE_Euclid_deleteRef( bHYPRE_Euclid );
      }

      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
 
   }

   /*-----------------------------------------------------------
    * Solve the system using original PCG 
    *-----------------------------------------------------------*/

   if (solver_id == 1 || solver_id == 2 || solver_id == 8 || 
       solver_id == 12 || solver_id == 43)
      if ( hpcg!=0 )
   {

      time_index = hypre_InitializeTiming("HPCG Setup");
      hypre_BeginTiming(time_index);
 
      bHYPRE_HPCG = bHYPRE_HPCG_Create( bmpicomm );
      bHYPRE_Vector_b = bHYPRE_Vector__cast( bHYPRE_b );
      bHYPRE_Vector_x = bHYPRE_Vector__cast( bHYPRE_x );
      bHYPRE_Vector_Dot( bHYPRE_Vector_b, bHYPRE_Vector_b, &tmp );
      bHYPRE_Vector_Dot( bHYPRE_Vector_x, bHYPRE_Vector_x, &tmp );

      bHYPRE_op_A = bHYPRE_Operator__cast( bHYPRE_parcsr_A );
      bHYPRE_HPCG_SetOperator( bHYPRE_HPCG, bHYPRE_op_A );
      bHYPRE_HPCG_SetIntParameter( bHYPRE_HPCG, "MaxIterations", 500);
      bHYPRE_HPCG_SetDoubleParameter( bHYPRE_HPCG, "Tolerance", tol);
      bHYPRE_HPCG_SetIntParameter( bHYPRE_HPCG, "TwoNorm", 1 );
      bHYPRE_HPCG_SetIntParameter( bHYPRE_HPCG, "RelChange", 0 );
      bHYPRE_HPCG_SetIntParameter( bHYPRE_HPCG, "PrintLevel", ioutdat );

      if (solver_id == 1)
      {
         /* use BoomerAMG as preconditioner */
	 ioutdat = 1;
         if (myid == 0) printf("Solver: AMG-HPCG\n");
         bHYPRE_AMG = bHYPRE_BoomerAMG_Create( bmpicomm, bHYPRE_parcsr_A );
         bHYPRE_BoomerAMG_SetOperator( bHYPRE_AMG, bHYPRE_op_A );
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "Tolerance", pc_tol);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CoarsenType",
                                        (hybrid*coarsen_type));
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MeasureType",
                                           measure_type);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "StrongThreshold",
                                              strong_threshold);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "TruncFactor",
                                              trunc_factor);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "PrintLevel", poutdat );
         bHYPRE_BoomerAMG_SetStringParameter( bHYPRE_AMG, "PrintFileName",
                                              "driver.out.log" );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MaxIter", 1 );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CycleType", cycle_type );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0NumSweeps",
                                           num_grid_sweeps[0] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1NumSweeps",
                                           num_grid_sweeps[1] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2NumSweeps",
                                           num_grid_sweeps[2] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3NumSweeps",
                                           num_grid_sweeps[3] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0RelaxType",
                                           grid_relax_type[0] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1RelaxType",
                                           grid_relax_type[1] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2RelaxType",
                                           grid_relax_type[2] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3RelaxType",
                                           grid_relax_type[3] );
         for ( i=0; i<max_levels; ++i )
         {
            bHYPRE_BoomerAMG_SetLevelRelaxWt( bHYPRE_AMG, relax_weight[i], i );
         }
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothType",
                                           smooth_type );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothNumSweeps",
                                           smooth_num_sweep);

         dimsl[0] = 0;   dimsl[1] = 0;   dimsu[0] = 4;   dimsu[1] = 4;
         bHYPRE_grid_relax_points = sidl_int__array_createCol( 2, dimsl, dimsu );
         for ( i=0; i<4; ++i )
         {
            for ( j=0; j<num_grid_sweeps[i]; ++j )
            {
               sidl_int__array_set2( bHYPRE_grid_relax_points, i, j,
                                     grid_relax_points[i][j] );
            }
         }
         bHYPRE_BoomerAMG_SetIntArray2Parameter( bHYPRE_AMG, "GridRelaxPoints",
                                                 bHYPRE_grid_relax_points );
         sidl_int__array_deleteRef( bHYPRE_grid_relax_points );

         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MaxLevels", max_levels);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "MaxRowSum",
                                              max_row_sum);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "NumFunctions",
                                           num_functions);
         if (num_functions > 1)
         {
            bHYPRE_BoomerAMG_SetIntArray1Parameter( bHYPRE_AMG, "DOFFunc",
                                                    dof_func, num_functions );
         }
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Variant", variant);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Overlap", overlap);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "DomainType", domain_type);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG,
                                              "SchwarzRlxWeight",
                                              schwarz_rlx_weight);

         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_AMG );
         ierr += bHYPRE_HPCG_SetPreconditioner( bHYPRE_HPCG, bHYPRE_SolverPC );
         ierr += bHYPRE_HPCG_Setup( bHYPRE_HPCG, bHYPRE_Vector_b, bHYPRE_Vector_x );

      }

      else if (solver_id == 2)
      {
         /* use diagonal scaling as preconditioner */

         /* To call a bHYPRE solver:
            create, set comm, set operator, set other parameters,
            Setup (noop in this case), Apply */
         bHYPRE_ParCSRDiagScale = bHYPRE_ParCSRDiagScale_Create(
            bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParCSRDiagScale_Setup( bHYPRE_ParCSRDiagScale,
                                               bHYPRE_Vector_b, bHYPRE_Vector_x );
         bHYPRE_SolverPC =
            bHYPRE_Solver__cast( bHYPRE_ParCSRDiagScale );
         ierr += bHYPRE_HPCG_SetPreconditioner( bHYPRE_HPCG, bHYPRE_SolverPC );
         ierr += bHYPRE_HPCG_Setup( bHYPRE_HPCG, bHYPRE_Vector_b, bHYPRE_Vector_x );

      }
      else if (solver_id == 8)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) printf("Solver: ParaSails-HPCG\n");

         bHYPRE_ParaSails = bHYPRE_ParaSails_Create( bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParaSails_SetDoubleParameter( bHYPRE_ParaSails, "Thresh",
                                                      sai_threshold );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Nlevels",
                                                   max_levels );
         ierr += bHYPRE_ParaSails_SetDoubleParameter( bHYPRE_ParaSails, "Filter",
                                                      sai_filter );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Logging",
                                                   ioutdat );
         hypre_assert( ierr==0 );
         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_ParaSails );
         ierr += bHYPRE_HPCG_SetPreconditioner( bHYPRE_HPCG, bHYPRE_SolverPC );
         ierr += bHYPRE_HPCG_Setup( bHYPRE_HPCG, bHYPRE_Vector_b, bHYPRE_Vector_x );

      }
      else if (solver_id == 12)
      {
#ifdef DO_THIS_LATER
         /* use Schwarz preconditioner */
         if (myid == 0) printf("Solver: Schwarz-HPCG\n");

	 HYPRE_SchwarzCreate(&pcg_precond);
	 HYPRE_SchwarzSetVariant(pcg_precond, variant);
	 HYPRE_SchwarzSetOverlap(pcg_precond, overlap);
	 HYPRE_SchwarzSetDomainType(pcg_precond, domain_type);
         HYPRE_SchwarzSetRelaxWeight(pcg_precond, schwarz_rlx_weight);

         HYPRE_HPCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_SchwarzSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_SchwarzSetup,
                             pcg_precond);
#endif  /*DO_THIS_LATER*/
      }
      else if (solver_id == 43)
      {
#ifdef DO_THIS_LATER
         /* use Euclid preconditioning */
         if (myid == 0) printf("Solver: Euclid-HPCG\n");

         HYPRE_EuclidCreate(mpi_comm, &pcg_precond);

         /* note: There are three three methods of setting run-time 
            parameters for Euclid: (see HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally 
            parse the command line.
         */   
         HYPRE_EuclidSetParams(pcg_precond, argc, argv);

         HYPRE_HPCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                             pcg_precond);
#endif  /*DO_THIS_LATER*/
      }
 

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      time_index = hypre_InitializeTiming("HPCG Solve");
      hypre_BeginTiming(time_index);

      ierr += bHYPRE_HPCG_Apply( bHYPRE_HPCG, bHYPRE_Vector_b, &bHYPRE_Vector_x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      ierr += bHYPRE_HPCG_GetIntValue( bHYPRE_HPCG, "NumIterations",
                                      &num_iterations );
      ierr += bHYPRE_HPCG_GetDoubleValue( bHYPRE_HPCG, "Final Relative Residual Norm",
                                         &final_res_norm );

      bHYPRE_HPCG_deleteRef( bHYPRE_HPCG );
      if ( solver_id == 1 )
      {
         bHYPRE_BoomerAMG_deleteRef( bHYPRE_AMG );
      }
      else if ( solver_id == 2 )
      {
         bHYPRE_ParCSRDiagScale_deleteRef( bHYPRE_ParCSRDiagScale );
      }
      else if (solver_id == 8)
      {
         bHYPRE_ParaSails_deleteRef( bHYPRE_ParaSails );
      }
#ifdef DO_THIS_LATER
   else if (solver_id == 12)
   {
   HYPRE_SchwarzDestroy(pcg_precond);
   }
   else if (solver_id == 43)
   {
   / * HYPRE_EuclidPrintParams(pcg_precond); * /
   HYPRE_EuclidDestroy(pcg_precond);
   }
#endif  /*DO_THIS_LATER*/

      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
 
   }

   /*-----------------------------------------------------------
    * Solve the system using GMRES, pure Babel-interface version
    *-----------------------------------------------------------*/

   if (solver_id == 3 || solver_id == 4 || solver_id == 7 
       || solver_id == 18 || solver_id == 44)
      if ( hpcg==0 )
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      bHYPRE_op_A = bHYPRE_Operator__cast( bHYPRE_parcsr_A );
      bHYPRE_GMRES = bHYPRE_GMRES_Create( bmpicomm, bHYPRE_op_A );
      bHYPRE_Vector_b = bHYPRE_Vector__cast( bHYPRE_b );
      bHYPRE_Vector_x = bHYPRE_Vector__cast( bHYPRE_x );

      ierr += bHYPRE_GMRES_SetIntParameter( bHYPRE_GMRES, "KDim", k_dim );
      ierr += bHYPRE_GMRES_SetIntParameter( bHYPRE_GMRES, "MaxIter", 1000 );
      ierr += bHYPRE_GMRES_SetDoubleParameter( bHYPRE_GMRES, "Tol", tol );
      ierr += bHYPRE_GMRES_SetIntParameter( bHYPRE_GMRES, "Logging", 1 );

      if (solver_id == 3)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) printf("Solver: AMG-GMRES\n");

         bHYPRE_AMG = bHYPRE_BoomerAMG_Create( bmpicomm, bHYPRE_parcsr_A );
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "Tolerance", pc_tol);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CoarsenType",
                                        (hybrid*coarsen_type));
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MeasureType",
                                           measure_type);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "StrongThreshold",
                                              strong_threshold);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "TruncFactor",
                                              trunc_factor);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "PrintLevel", poutdat );
         bHYPRE_BoomerAMG_SetStringParameter( bHYPRE_AMG, "PrintFileName",
                                              "driver.out.log" );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MaxIter", 1 );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CycleType", cycle_type );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0NumSweeps",
                                           num_grid_sweeps[0] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1NumSweeps",
                                           num_grid_sweeps[1] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2NumSweeps",
                                           num_grid_sweeps[2] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3NumSweeps",
                                           num_grid_sweeps[3] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0RelaxType",
                                           grid_relax_type[0] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1RelaxType",
                                           grid_relax_type[1] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2RelaxType",
                                           grid_relax_type[2] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3RelaxType",
                                           grid_relax_type[3] );
         for ( i=0; i<max_levels; ++i )
         {
            bHYPRE_BoomerAMG_SetLevelRelaxWt( bHYPRE_AMG, relax_weight[i], i );
         }
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothType",
                                           smooth_type );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothNumSweeps",
                                           smooth_num_sweep);

         dimsl[0] = 0;   dimsl[1] = 0;   dimsu[0] = 4;   dimsu[1] = 4;
         bHYPRE_grid_relax_points = sidl_int__array_createCol( 2, dimsl, dimsu );
         for ( i=0; i<4; ++i )
         {
            for ( j=0; j<num_grid_sweeps[i]; ++j )
            {
               sidl_int__array_set2( bHYPRE_grid_relax_points, i, j,
                                     grid_relax_points[i][j] );
            }
         }
         bHYPRE_BoomerAMG_SetIntArray2Parameter( bHYPRE_AMG, "GridRelaxPoints",
                                                 bHYPRE_grid_relax_points );
         sidl_int__array_deleteRef( bHYPRE_grid_relax_points );

         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MaxLevels", max_levels);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "MaxRowSum",
                                              max_row_sum);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "NumFunctions",
                                           num_functions);
         if (num_functions > 1)
         {
            bHYPRE_BoomerAMG_SetIntArray1Parameter( bHYPRE_AMG, "DOFFunc",
                                                    dof_func, num_functions );
         }
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Variant", variant);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Overlap", overlap);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "DomainType", domain_type);

         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_AMG );
         ierr += bHYPRE_GMRES_SetPreconditioner( bHYPRE_GMRES, bHYPRE_SolverPC );
         ierr += bHYPRE_GMRES_Setup( bHYPRE_GMRES, bHYPRE_Vector_b,
                                     bHYPRE_Vector_x );
      }
      else if (solver_id == 4)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) printf("Solver: DS-GMRES\n");

         bHYPRE_ParCSRDiagScale = bHYPRE_ParCSRDiagScale_Create(
            bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParCSRDiagScale_Setup( bHYPRE_ParCSRDiagScale,
                                               bHYPRE_Vector_b, bHYPRE_Vector_x );
         bHYPRE_SolverPC =
            bHYPRE_Solver__cast( bHYPRE_ParCSRDiagScale );
         ierr += bHYPRE_GMRES_SetPreconditioner( bHYPRE_GMRES, bHYPRE_SolverPC );
         ierr += bHYPRE_GMRES_Setup( bHYPRE_GMRES, bHYPRE_Vector_b,
                                     bHYPRE_Vector_x );

      }
#ifdef DO_THIS_LATER
      else if (solver_id == 7)
      {
         /* use PILUT as preconditioner */
         if (myid == 0) printf("Solver: PILUT-GMRES\n");

         ierr = HYPRE_ParCSRPilutCreate( mpi_comm, &pcg_precond ); 
         if (ierr) {
            printf("Error in ParPilutCreate\n");
         }

         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSetup,
                               pcg_precond);

         if (drop_tol >= 0 )
            HYPRE_ParCSRPilutSetDropTolerance( pcg_precond,
                                               drop_tol );

         if (nonzeros_to_keep >= 0 )
            HYPRE_ParCSRPilutSetFactorRowSize( pcg_precond,
                                               nonzeros_to_keep );
      }
#endif  /*DO_THIS_LATER*/
      else if (solver_id == 18)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) printf("Solver: ParaSails-GMRES\n");

         bHYPRE_ParaSails = bHYPRE_ParaSails_Create( bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParaSails_SetDoubleParameter( bHYPRE_ParaSails, "Thresh",
                                                      sai_threshold );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Nlevels",
                                                   max_levels );
         ierr += bHYPRE_ParaSails_SetDoubleParameter( bHYPRE_ParaSails, "Filter",
                                                      sai_filter );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Logging",
                                                   ioutdat );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Sym", 0 );
         hypre_assert( ierr==0 );
         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_ParaSails );
         ierr += bHYPRE_GMRES_SetPreconditioner( bHYPRE_GMRES, bHYPRE_SolverPC );
         ierr += bHYPRE_GMRES_Setup( bHYPRE_GMRES, bHYPRE_Vector_b,
                                     bHYPRE_Vector_x );

      }
#ifdef DO_THIS_LATER
      else if (solver_id == 44)
      {
         /* use Euclid preconditioning */
         if (myid == 0) printf("Solver: Euclid-GMRES\n");

         HYPRE_EuclidCreate(mpi_comm, &pcg_precond);

         /* note: There are three three methods of setting run-time 
            parameters for Euclid: (see HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally 
            parse the command line.
         */   
         HYPRE_EuclidSetParams(pcg_precond, argc, argv);

         HYPRE_GMRESSetPrecond (pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                                pcg_precond);
      }
#endif  /*DO_THIS_LATER*/
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      ierr += bHYPRE_GMRES_Apply( bHYPRE_GMRES, bHYPRE_Vector_b, &bHYPRE_Vector_x );
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      ierr += bHYPRE_GMRES_GetIntValue( bHYPRE_GMRES, "NumIterations",
                                        &num_iterations );
      ierr += bHYPRE_GMRES_GetDoubleValue( bHYPRE_GMRES, "Final Relative Residual Norm",
                                           &final_res_norm );
 
      bHYPRE_GMRES_deleteRef( bHYPRE_GMRES );
 
      if (solver_id == 3)
      {
         bHYPRE_BoomerAMG_deleteRef( bHYPRE_AMG );
      }
      else if ( solver_id == 4 )
      {
         bHYPRE_ParCSRDiagScale_deleteRef( bHYPRE_ParCSRDiagScale );
      }
#ifdef DO_THIS_LATER
      if (solver_id == 7)
      {
         HYPRE_ParCSRPilutDestroy(pcg_precond);
      }
#endif  /*DO_THIS_LATER*/
      else if (solver_id == 18)
      {
	 bHYPRE_ParaSails_deleteRef ( bHYPRE_ParaSails );
      }
#ifdef DO_THIS_LATER
      else if (solver_id == 44)
      {
         /* HYPRE_EuclidPrintParams(pcg_precond); */
         HYPRE_EuclidDestroy(pcg_precond);
      }
#endif  /*DO_THIS_LATER*/

      if (myid == 0)
      {
         printf("\n");
         printf("GMRES Iterations = %d\n", num_iterations);
         printf("Final GMRES Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using GMRES, Babel interface working through the HYPRE interface
    *-----------------------------------------------------------*/

   if (solver_id == 3 || solver_id == 4 || solver_id == 7 
       || solver_id == 18 || solver_id == 44)
      if ( hpcg!=0 )
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      bHYPRE_HGMRES = bHYPRE_HGMRES_Create( bmpicomm );
      bHYPRE_Vector_b = bHYPRE_Vector__cast( bHYPRE_b );
      bHYPRE_Vector_x = bHYPRE_Vector__cast( bHYPRE_x );
      bHYPRE_op_A = bHYPRE_Operator__cast( bHYPRE_parcsr_A );
      bHYPRE_HGMRES_SetOperator( bHYPRE_HGMRES, bHYPRE_op_A );

      ierr += bHYPRE_HGMRES_SetIntParameter( bHYPRE_HGMRES, "KDim", k_dim );
      ierr += bHYPRE_HGMRES_SetIntParameter( bHYPRE_HGMRES, "MaxIter", 1000 );
      ierr += bHYPRE_HGMRES_SetDoubleParameter( bHYPRE_HGMRES, "Tol", tol );
      ierr += bHYPRE_HGMRES_SetIntParameter( bHYPRE_HGMRES, "Logging", 1 );

      if (solver_id == 3)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) printf("Solver: AMG-GMRES\n");

         bHYPRE_AMG = bHYPRE_BoomerAMG_Create( bmpicomm, bHYPRE_parcsr_A );
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "Tolerance", pc_tol);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CoarsenType",
                                        (hybrid*coarsen_type));
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MeasureType",
                                           measure_type);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "StrongThreshold",
                                              strong_threshold);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "TruncFactor",
                                              trunc_factor);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "PrintLevel", poutdat );
         bHYPRE_BoomerAMG_SetStringParameter( bHYPRE_AMG, "PrintFileName",
                                              "driver.out.log" );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MaxIter", 1 );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CycleType", cycle_type );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0NumSweeps",
                                           num_grid_sweeps[0] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1NumSweeps",
                                           num_grid_sweeps[1] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2NumSweeps",
                                           num_grid_sweeps[2] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3NumSweeps",
                                           num_grid_sweeps[3] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0RelaxType",
                                           grid_relax_type[0] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1RelaxType",
                                           grid_relax_type[1] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2RelaxType",
                                           grid_relax_type[2] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3RelaxType",
                                           grid_relax_type[3] );
         for ( i=0; i<max_levels; ++i )
         {
            bHYPRE_BoomerAMG_SetLevelRelaxWt( bHYPRE_AMG, relax_weight[i], i );
         }
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothType",
                                           smooth_type );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothNumSweeps",
                                           smooth_num_sweep);

         dimsl[0] = 0;   dimsl[1] = 0;   dimsu[0] = 4;   dimsu[1] = 4;
         bHYPRE_grid_relax_points = sidl_int__array_createCol( 2, dimsl, dimsu );
         for ( i=0; i<4; ++i )
         {
            for ( j=0; j<num_grid_sweeps[i]; ++j )
            {
               sidl_int__array_set2( bHYPRE_grid_relax_points, i, j,
                                     grid_relax_points[i][j] );
            }
         }
         bHYPRE_BoomerAMG_SetIntArray2Parameter( bHYPRE_AMG, "GridRelaxPoints",
                                                 bHYPRE_grid_relax_points );
         sidl_int__array_deleteRef( bHYPRE_grid_relax_points );

         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MaxLevels", max_levels);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "MaxRowSum",
                                              max_row_sum);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "NumFunctions",
                                           num_functions);
         if (num_functions > 1)
         {
            bHYPRE_BoomerAMG_SetIntArray1Parameter( bHYPRE_AMG, "DOFFunc",
                                                    dof_func, num_functions );
         }
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Variant", variant);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Overlap", overlap);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "DomainType", domain_type);

         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_AMG );
         ierr += bHYPRE_HGMRES_SetPreconditioner( bHYPRE_HGMRES, bHYPRE_SolverPC );
         ierr += bHYPRE_HGMRES_Setup( bHYPRE_HGMRES, bHYPRE_Vector_b,
                                     bHYPRE_Vector_x );
      }
      else if (solver_id == 4)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) printf("Solver: DS-GMRES\n");

         bHYPRE_ParCSRDiagScale = bHYPRE_ParCSRDiagScale_Create(
            bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParCSRDiagScale_Setup( bHYPRE_ParCSRDiagScale,
                                               bHYPRE_Vector_b, bHYPRE_Vector_x );
         bHYPRE_SolverPC =
            bHYPRE_Solver__cast( bHYPRE_ParCSRDiagScale );
         ierr += bHYPRE_HGMRES_SetPreconditioner( bHYPRE_HGMRES, bHYPRE_SolverPC );
         ierr += bHYPRE_HGMRES_Setup( bHYPRE_HGMRES, bHYPRE_Vector_b,
                                     bHYPRE_Vector_x );

      }
#ifdef DO_THIS_LATER
      else if (solver_id == 7)
      {
         /* use PILUT as preconditioner */
         if (myid == 0) printf("Solver: PILUT-GMRES\n");

         ierr = HYPRE_ParCSRPilutCreate( mpi_comm, &pcg_precond ); 
         if (ierr) {
            printf("Error in ParPilutCreate\n");
         }

         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSetup,
                               pcg_precond);

         if (drop_tol >= 0 )
            HYPRE_ParCSRPilutSetDropTolerance( pcg_precond,
                                               drop_tol );

         if (nonzeros_to_keep >= 0 )
            HYPRE_ParCSRPilutSetFactorRowSize( pcg_precond,
                                               nonzeros_to_keep );
      }
#endif  /*DO_THIS_LATER*/
      else if (solver_id == 18)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) printf("Solver: ParaSails-GMRES\n");

         bHYPRE_ParaSails = bHYPRE_ParaSails_Create( bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParaSails_SetDoubleParameter( bHYPRE_ParaSails, "Thresh",
                                                      sai_threshold );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Nlevels",
                                                   max_levels );
         ierr += bHYPRE_ParaSails_SetDoubleParameter( bHYPRE_ParaSails, "Filter",
                                                      sai_filter );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Logging",
                                                   ioutdat );
         ierr += bHYPRE_ParaSails_SetIntParameter( bHYPRE_ParaSails, "Sym", 0 );
         hypre_assert( ierr==0 );
         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_ParaSails );
         ierr += bHYPRE_HGMRES_SetPreconditioner( bHYPRE_HGMRES, bHYPRE_SolverPC );
         ierr += bHYPRE_HGMRES_Setup( bHYPRE_HGMRES, bHYPRE_Vector_b,
                                     bHYPRE_Vector_x );

      }
#ifdef DO_THIS_LATER
      else if (solver_id == 44)
      {
         /* use Euclid preconditioning */
         if (myid == 0) printf("Solver: Euclid-GMRES\n");

         HYPRE_EuclidCreate(mpi_comm, &pcg_precond);

         /* note: There are three three methods of setting run-time 
            parameters for Euclid: (see HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally 
            parse the command line.
         */   
         HYPRE_EuclidSetParams(pcg_precond, argc, argv);

         HYPRE_GMRESSetPrecond (pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                                pcg_precond);
      }
#endif  /*DO_THIS_LATER*/
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      ierr += bHYPRE_HGMRES_Apply( bHYPRE_HGMRES, bHYPRE_Vector_b, &bHYPRE_Vector_x );
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      ierr += bHYPRE_HGMRES_GetIntValue( bHYPRE_HGMRES, "NumIterations",
                                        &num_iterations );
      ierr += bHYPRE_HGMRES_GetDoubleValue( bHYPRE_HGMRES, "Final Relative Residual Norm",
                                           &final_res_norm );
 
      bHYPRE_HGMRES_deleteRef( bHYPRE_HGMRES );
 
      if (solver_id == 3)
      {
         bHYPRE_BoomerAMG_deleteRef( bHYPRE_AMG );
      }
      else if ( solver_id == 2 )
      {
         bHYPRE_ParCSRDiagScale_deleteRef( bHYPRE_ParCSRDiagScale );
      }
#ifdef DO_THIS_LATER
      if (solver_id == 7)
      {
         HYPRE_ParCSRPilutDestroy(pcg_precond);
      }
#endif  /*DO_THIS_LATER*/
      else if (solver_id == 18)
      {
	 bHYPRE_ParaSails_deleteRef ( bHYPRE_ParaSails );
      }
#ifdef DO_THIS_LATER
      else if (solver_id == 44)
      {
         /* HYPRE_EuclidPrintParams(pcg_precond); */
         HYPRE_EuclidDestroy(pcg_precond);
      }
#endif  /*DO_THIS_LATER*/

      if (myid == 0)
      {
         printf("\n");
         printf("GMRES Iterations = %d\n", num_iterations);
         printf("Final GMRES Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using BiCGSTAB 
    *-----------------------------------------------------------*/

   if (solver_id == 9 || solver_id == 10 || solver_id == 11 || solver_id == 45)
   {
      time_index = hypre_InitializeTiming("BiCGSTAB Setup");
      hypre_BeginTiming(time_index);
 
      bHYPRE_op_A = bHYPRE_Operator__cast( bHYPRE_parcsr_A );
      bHYPRE_BiCGSTAB = bHYPRE_BiCGSTAB_Create( bmpicomm, bHYPRE_op_A );
      bHYPRE_Vector_b = bHYPRE_Vector__cast( bHYPRE_b );
      bHYPRE_Vector_x = bHYPRE_Vector__cast( bHYPRE_x );

      bHYPRE_BiCGSTAB_SetIntParameter( bHYPRE_BiCGSTAB, "MaxIterations", 500 );
      bHYPRE_BiCGSTAB_SetDoubleParameter( bHYPRE_BiCGSTAB, "Tolerance", tol );
      bHYPRE_BiCGSTAB_SetIntParameter( bHYPRE_BiCGSTAB, "Logging", 1 );
 
      if (solver_id == 9)
      {
         hypre_assert( "solver 9 not implemented"==0 );
#ifdef DO_THIS_LATER
         /* use BoomerAMG as preconditioner */
         if (myid == 0) printf("Solver: AMG-BiCGSTAB\n");

         HYPRE_BoomerAMGCreate(&pcg_precond); 
         HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid*coarsen_type));
         HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         HYPRE_BoomerAMGSetPrintLevel(pcg_precond, ioutdat);
         HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         HYPRE_BoomerAMGSetNumGridSweeps(pcg_precond, num_grid_sweeps);
         HYPRE_BoomerAMGSetGridRelaxType(pcg_precond, grid_relax_type);
         HYPRE_BoomerAMGSetRelaxWeight(pcg_precond, relax_weight);
         HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweep);
         HYPRE_BoomerAMGSetGridRelaxPoints(pcg_precond, grid_relax_points);
         HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         if (num_functions > 1)
            HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                  (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                  pcg_precond);
#endif  /*DO_THIS_LATER*/
      }
      else if (solver_id == 10)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) printf("Solver: DS-BiCGSTAB\n");

         bHYPRE_ParCSRDiagScale = bHYPRE_ParCSRDiagScale_Create(
            bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParCSRDiagScale_Setup( bHYPRE_ParCSRDiagScale,
                                               bHYPRE_Vector_b, bHYPRE_Vector_x );
         bHYPRE_SolverPC =
            bHYPRE_Solver__cast( bHYPRE_ParCSRDiagScale );
         ierr += bHYPRE_BiCGSTAB_SetPreconditioner(
            bHYPRE_BiCGSTAB, bHYPRE_SolverPC );
         ierr += bHYPRE_BiCGSTAB_Setup(
            bHYPRE_BiCGSTAB, bHYPRE_Vector_b, bHYPRE_Vector_x );

      }
      else if (solver_id == 11)
      {
         hypre_assert( "solver 11 not implemented"==0 );
#ifdef DO_THIS_LATER
         /* use PILUT as preconditioner */
         if (myid == 0) printf("Solver: PILUT-BiCGSTAB\n");

         ierr = HYPRE_ParCSRPilutCreate( mpi_comm, &pcg_precond ); 
         if (ierr) {
            printf("Error in ParPilutCreate\n");
         }

         HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSolve,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSetup,
                                  pcg_precond);

         if (drop_tol >= 0 )
            HYPRE_ParCSRPilutSetDropTolerance( pcg_precond,
                                               drop_tol );

         if (nonzeros_to_keep >= 0 )
            HYPRE_ParCSRPilutSetFactorRowSize( pcg_precond,
                                               nonzeros_to_keep );
#endif  /*DO_THIS_LATER*/
      }
      else if (solver_id == 45)
      {
         hypre_assert( "solver 45 not implemented"==0 );
#ifdef DO_THIS_LATER
         /* use Euclid preconditioning */
         if (myid == 0) printf("Solver: Euclid-BICGSTAB\n");

         HYPRE_EuclidCreate(mpi_comm, &pcg_precond);

         /* note: There are three three methods of setting run-time 
            parameters for Euclid: (see HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally 
            parse the command line.
         */   
         HYPRE_EuclidSetParams(pcg_precond, argc, argv);

         HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                                  (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                                  pcg_precond);
#endif  /*DO_THIS_LATER*/
      }
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("BiCGSTAB Solve");
      hypre_BeginTiming(time_index);
 
      ierr += bHYPRE_BiCGSTAB_Apply(
         bHYPRE_BiCGSTAB, bHYPRE_Vector_b, &bHYPRE_Vector_x );
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      ierr += bHYPRE_BiCGSTAB_GetIntValue( bHYPRE_BiCGSTAB, "NumIterations",
                                           &num_iterations );
      ierr += bHYPRE_BiCGSTAB_GetDoubleValue(
         bHYPRE_BiCGSTAB, "Final Relative Residual Norm", &final_res_norm );

      bHYPRE_BiCGSTAB_deleteRef( bHYPRE_BiCGSTAB );
 
      if (solver_id == 9)
      {
#ifdef DO_THIS_LATER
         HYPRE_BoomerAMGDestroy(pcg_precond);
#endif  /*DO_THIS_LATER*/
      }
      else if (solver_id == 10)
      {
         bHYPRE_ParCSRDiagScale_deleteRef( bHYPRE_ParCSRDiagScale );
      }
      else if (solver_id == 11)
      {
#ifdef DO_THIS_LATER
         HYPRE_ParCSRPilutDestroy(pcg_precond);
#endif  /*DO_THIS_LATER*/
      }
      else if (solver_id == 45)
      {
#ifdef DO_THIS_LATER
         /* HYPRE_EuclidPrintParams(pcg_precond); */
         HYPRE_EuclidDestroy(pcg_precond);
#endif  /*DO_THIS_LATER*/
      }

      if (myid == 0)
      {
         printf("\n");
         printf("BiCGSTAB Iterations = %d\n", num_iterations);
         printf("Final BiCGSTAB Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
   }
   /*-----------------------------------------------------------
    * Solve the system using CGNR 
    *-----------------------------------------------------------*/

   if (solver_id == 5 || solver_id == 6)
   {
      time_index = hypre_InitializeTiming("CGNR Setup");
      hypre_BeginTiming(time_index);

      bHYPRE_op_A = bHYPRE_Operator__cast( bHYPRE_parcsr_A );
      bHYPRE_CGNR = bHYPRE_CGNR_Create( bmpicomm, bHYPRE_op_A );
      bHYPRE_Vector_b = bHYPRE_Vector__cast( bHYPRE_b );
      bHYPRE_Vector_x = bHYPRE_Vector__cast( bHYPRE_x );

      bHYPRE_CGNR_SetIntParameter( bHYPRE_CGNR, "MaxIterations", 1000 );
      bHYPRE_CGNR_SetDoubleParameter( bHYPRE_CGNR, "Tolerance", tol );
      bHYPRE_CGNR_SetLogging( bHYPRE_CGNR, 2 );
      bHYPRE_CGNR_SetIntParameter( bHYPRE_CGNR, "PrintLevel", ioutdat );
 
      if (solver_id == 5)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) printf("Solver: AMG-CGNR\n");

         bHYPRE_AMG = bHYPRE_BoomerAMG_Create( bmpicomm, bHYPRE_parcsr_A );

         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "Tolerance", pc_tol);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CoarsenType",
                                        (hybrid*coarsen_type));
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MeasureType",
                                           measure_type);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "StrongThreshold",
                                              strong_threshold);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "TruncFactor",
                                              trunc_factor);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "PrintLevel", poutdat );
         bHYPRE_BoomerAMG_SetStringParameter( bHYPRE_AMG, "PrintFileName",
                                              "driver.out.log" );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MaxIter", 1 );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "CycleType", cycle_type );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0NumSweeps",
                                           num_grid_sweeps[0] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1NumSweeps",
                                           num_grid_sweeps[1] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2NumSweeps",
                                           num_grid_sweeps[2] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3NumSweeps",
                                           num_grid_sweeps[3] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle0RelaxType",
                                           grid_relax_type[0] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle1RelaxType",
                                           grid_relax_type[1] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle2RelaxType",
                                           grid_relax_type[2] );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Cycle3RelaxType",
                                           grid_relax_type[3] );
         for ( i=0; i<max_levels; ++i )
         {
            bHYPRE_BoomerAMG_SetLevelRelaxWt( bHYPRE_AMG, relax_weight[i], i );
         }
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothType",
                                           smooth_type );
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "SmoothNumSweeps",
                                           smooth_num_sweep);

         dimsl[0] = 0;   dimsl[1] = 0;   dimsu[0] = 4;   dimsu[1] = 4;
         bHYPRE_grid_relax_points = sidl_int__array_createCol( 2, dimsl, dimsu );
         for ( i=0; i<4; ++i )
         {
            for ( j=0; j<num_grid_sweeps[i]; ++j )
            {
               sidl_int__array_set2( bHYPRE_grid_relax_points, i, j,
                                     grid_relax_points[i][j] );
            }
         }
         bHYPRE_BoomerAMG_SetIntArray2Parameter( bHYPRE_AMG, "GridRelaxPoints",
                                                 bHYPRE_grid_relax_points );
         sidl_int__array_deleteRef( bHYPRE_grid_relax_points );

         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "MaxLevels", max_levels);
         bHYPRE_BoomerAMG_SetDoubleParameter( bHYPRE_AMG, "MaxRowSum",
                                              max_row_sum);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "NumFunctions",
                                           num_functions);
         if (num_functions > 1)
         {
            bHYPRE_BoomerAMG_SetIntArray1Parameter( bHYPRE_AMG, "DOFFunc",
                                                    dof_func, num_functions );
         }
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Variant", variant);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "Overlap", overlap);
         bHYPRE_BoomerAMG_SetIntParameter( bHYPRE_AMG, "DomainType", domain_type);


         bHYPRE_SolverPC = bHYPRE_Solver__cast( bHYPRE_AMG );
         ierr += bHYPRE_CGNR_SetPreconditioner( bHYPRE_CGNR, bHYPRE_SolverPC );
         ierr += bHYPRE_CGNR_Setup( bHYPRE_CGNR, bHYPRE_Vector_b, bHYPRE_Vector_x );

      }
      else if (solver_id == 6)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) printf("Solver: DS-CGNR\n");
         bHYPRE_ParCSRDiagScale = bHYPRE_ParCSRDiagScale_Create( bmpicomm, bHYPRE_parcsr_A );
         ierr += bHYPRE_ParCSRDiagScale_Setup( bHYPRE_ParCSRDiagScale,
                                               bHYPRE_Vector_b, bHYPRE_Vector_x );
         bHYPRE_SolverPC =
            bHYPRE_Solver__cast( bHYPRE_ParCSRDiagScale );
         ierr += bHYPRE_CGNR_SetPreconditioner( bHYPRE_CGNR, bHYPRE_SolverPC );
         ierr += bHYPRE_CGNR_Setup( bHYPRE_CGNR, bHYPRE_Vector_b, bHYPRE_Vector_x );

      }
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("CGNR Solve");
      hypre_BeginTiming(time_index);
 
      ierr += bHYPRE_CGNR_Apply( bHYPRE_CGNR, bHYPRE_Vector_b, &bHYPRE_Vector_x );
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", mpi_comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      ierr += bHYPRE_CGNR_GetIntValue( bHYPRE_CGNR, "NumIterations",
                                       &num_iterations );
      ierr += bHYPRE_CGNR_GetDoubleValue( bHYPRE_CGNR, "Final Relative Residual Norm",
                                          &final_res_norm );

      bHYPRE_CGNR_deleteRef( bHYPRE_CGNR );
 
      if (solver_id == 5)
      {
         bHYPRE_BoomerAMG_deleteRef( bHYPRE_AMG );
      }
      else if ( solver_id == 6 )
      {
         bHYPRE_ParCSRDiagScale_deleteRef( bHYPRE_ParCSRDiagScale );
      }
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
   }

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   bHYPRE_IJParCSRVector_Print( bHYPRE_b, "driver.out.b");
   bHYPRE_IJParCSRVector_Print( bHYPRE_x, "driver.out.x");

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   /* Programming note: some or all of these sidl array objects, e.g.
    * bHYPRE_num_grid_sweeps, contain data which have been incorporated
    * into bHYPRE_AMG (the sidl objects themselves were just temporary
    * carriers for the data).  The Babel deleteRef doesn't seem to be able
    * to handle doing it twice, so some are commented-out.
    */

   bHYPRE_IJParCSRMatrix_deleteRef( bHYPRE_parcsr_A );
   bHYPRE_IJParCSRVector_deleteRef( bHYPRE_b );
   bHYPRE_IJParCSRVector_deleteRef( bHYPRE_x );

   /* These can be (and do get) freed by HYPRE programs, but not always.
      All are obsolete, better to not pass them in. */
   if ( num_grid_sweeps )
      hypre_TFree(num_grid_sweeps);
   if ( relax_weight )
      hypre_TFree(relax_weight);
   if ( grid_relax_points ) {
      for ( i=0; i<4; ++i )
      {
         if ( grid_relax_points[i] )
         {
            hypre_TFree( grid_relax_points[i] );
         }
      }
      hypre_TFree( grid_relax_points );
   }
   if ( grid_relax_type )
      hypre_TFree( grid_relax_type );

   bHYPRE_MPICommunicator_deleteRef( bmpicomm );
   MPI_Finalize();

   return (0);
}

/*----------------------------------------------------------------------
 * Build matrix from file. Expects three files on each processor.
 * filename.D.n contains the diagonal part, filename.O.n contains
 * the offdiagonal part and filename.INFO.n contains global row
 * and column numbers, number of columns of offdiagonal matrix
 * and the mapping of offdiagonal column numbers to global column numbers.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
BuildParFromFile( int                  argc,
                  char                *argv[],
                  int                  arg_index,
                  HYPRE_ParCSRMatrix  *A_ptr     )
{
   char               *filename;

   HYPRE_ParCSRMatrix A;

   int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  FromFile: %s\n", filename);
   }

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   HYPRE_ParCSRMatrixRead(MPI_COMM_WORLD, filename,&A);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
bBuildParLaplacian( int                  argc,
                    char                *argv[],
                    int                  arg_index,
                    bHYPRE_MPICommunicator bmpi_comm,
                    bHYPRE_IJParCSRMatrix  *bA_ptr     )
{
   int                 nx, ny, nz;
   int                 P, Q, R;
   double              cx, cy, cz;

   bHYPRE_IJParCSRMatrix  bA;

   int                 num_procs, myid;
   int                 p, q, r;
   double             *values;
   int                 nvalues = 4;
   MPI_Comm mpi_comm = bHYPRE_MPICommunicator__get_data(bmpi_comm)->mpi_comm;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(mpi_comm, &num_procs );
   MPI_Comm_rank(mpi_comm, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Laplacian:\n");
      printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 4);

   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0*cx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz;
   }

   bA = bHYPRE_IJParCSRMatrix_GenerateLaplacian(
      bmpi_comm, nx, ny, nz, P, Q, R, p, q, r,
      values, nvalues, 7 );

   hypre_TFree(values);

   *bA_ptr = bA;

   return (0);
}

/* non-Babel version used only for timings... */
int
BuildParLaplacian( int                  argc,
                   char                *argv[],
                   int                  arg_index,
                   HYPRE_ParCSRMatrix  *A_ptr     )
{
   int                 nx, ny, nz;
   int                 P, Q, R;
   double              cx, cy, cz;

   HYPRE_ParCSRMatrix  A;

   int                 num_procs, myid;
   int                 p, q, r;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Laplacian:\n");
      printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 4);

   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0*cx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz;
   }

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian(MPI_COMM_WORLD, 
		nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 7-point convection-diffusion operator 
 * Parameters given in command line.
 * Operator:
 *
 *  -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f
 *
 *----------------------------------------------------------------------*/

int
BuildParDifConv( int                  argc,
                 char                *argv[],
                 int                  arg_index,
                 HYPRE_ParCSRMatrix  *A_ptr     )
{
   int                 nx, ny, nz;
   int                 P, Q, R;
   double              cx, cy, cz;
   double              ax, ay, az;
   double              hinx,hiny,hinz;

   HYPRE_ParCSRMatrix  A;

   int                 num_procs, myid;
   int                 p, q, r;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   hinx = 1./(nx+1);
   hiny = 1./(ny+1);
   hinz = 1./(nz+1);

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

   ax = 1.;
   ay = 1.;
   az = 1.;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-a") == 0 )
      {
         arg_index++;
         ax = atof(argv[arg_index++]);
         ay = atof(argv[arg_index++]);
         az = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Convection-Diffusion: \n");
      printf("    -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f\n");  
      printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      printf("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
      printf("    (ax, ay, az) = (%f, %f, %f)\n\n", ax, ay, az);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 7);

   values[1] = -cx/(hinx*hinx);
   values[2] = -cy/(hiny*hiny);
   values[3] = -cz/(hinz*hinz);
   values[4] = -cx/(hinx*hinx) + ax/hinx;
   values[5] = -cy/(hiny*hiny) + ay/hiny;
   values[6] = -cz/(hinz*hinz) + az/hinz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0*cx/(hinx*hinx) - 1.*ax/hinx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy/(hiny*hiny) - 1.*ay/hiny;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz/(hinz*hinz) - 1.*az/hinz;
   }

   A = (HYPRE_ParCSRMatrix) GenerateDifConv(MPI_COMM_WORLD,
                                            nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build matrix from one file on Proc. 0. Expects matrix to be in
 * CSR format. Distributes matrix across processors giving each about
 * the same number of rows.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
BuildParFromOneFile( int                  argc,
                     char                *argv[],
                     int                  arg_index,
                     HYPRE_ParCSRMatrix  *A_ptr     )
{
   char               *filename;

   HYPRE_ParCSRMatrix  A;
   HYPRE_CSRMatrix  A_CSR = NULL;

   int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * Generate the matrix 
       *-----------------------------------------------------------*/
 
      A_CSR = HYPRE_CSRMatrixRead(filename);
   }
   HYPRE_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD, A_CSR, NULL, NULL, &A);

   *A_ptr = A;

   if (myid == 0) HYPRE_CSRMatrixDestroy(A_CSR);

   return (0);
}

/*----------------------------------------------------------------------
 * Build Function array from files on different processors
 *----------------------------------------------------------------------*/

int
BuildFuncsFromFiles(    int                  argc,
                        char                *argv[],
                        int                  arg_index,
                        bHYPRE_IJParCSRMatrix   parcsr_A,
                        int                **dof_func_ptr     )
{
/*----------------------------------------------------------------------
 * Build Function array from files on different processors
 *----------------------------------------------------------------------*/

   printf (" Feature is not implemented yet!\n");	
   return(0);

}


int
BuildFuncsFromOneFile(  int                  argc,
                        char                *argv[],
                        int                  arg_index,
                        bHYPRE_IJParCSRMatrix   bHYPRE_parcsr_A,
                        int                **dof_func_ptr     )
{
   char           *filename;

   int             myid, num_procs;
   int            *partitioning;
   int            *dof_func;
   int            *dof_func_local;
   int             i, j;
   int             local_size, global_size;
   MPI_Request	  *requests;
   MPI_Status	  *status, status0;
   MPI_Comm	   comm;

   HYPRE_ParCSRMatrix parcsr_A;
   struct bHYPRE_IJParCSRMatrix__data * temp_data;
   void               *object;

   /*-----------------------------------------------------------
    * extract HYPRE_ParCSRMatrix from bHYPRE_IJParCSRMatrix
    *-----------------------------------------------------------*/
      temp_data = bHYPRE_IJParCSRMatrix__get_data( bHYPRE_parcsr_A );
      HYPRE_IJMatrixGetObject( temp_data->ij_A, &object);
      parcsr_A = (HYPRE_ParCSRMatrix) object;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   comm = MPI_COMM_WORLD;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      FILE *fp;
      printf("  Funcs FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * read in the data
       *-----------------------------------------------------------*/
      fp = fopen(filename, "r");

      fscanf(fp, "%d", &global_size);
      dof_func = hypre_CTAlloc(int, global_size);

      for (j = 0; j < global_size; j++)
      {
         fscanf(fp, "%d", &dof_func[j]);
      }

      fclose(fp);
 
   }
   HYPRE_ParCSRMatrixGetRowPartitioning(parcsr_A, &partitioning);
   local_size = partitioning[myid+1]-partitioning[myid];
   dof_func_local = hypre_CTAlloc(int,local_size);

   if (myid == 0)
   {
      requests = hypre_CTAlloc(MPI_Request,num_procs-1);
      status = hypre_CTAlloc(MPI_Status,num_procs-1);
      j = 0;
      for (i=1; i < num_procs; i++)
      {
         MPI_Isend(&dof_func[partitioning[i]],
                   partitioning[i+1]-partitioning[i],
                   MPI_INT, i, 0, comm, &requests[j++]);
      }
      for (i=0; i < local_size; i++)
      {
         dof_func_local[i] = dof_func[i];
      }
      MPI_Waitall(num_procs-1,requests, status);
      hypre_TFree(requests);
      hypre_TFree(status);
   }
   else
   {
      MPI_Recv(dof_func_local,local_size,MPI_INT,0,0,comm,&status0);
   }

   *dof_func_ptr = dof_func_local;

   if (myid == 0) hypre_TFree(dof_func);

   return (0);
}

/*----------------------------------------------------------------------
 * Build Rhs from one file on Proc. 0. Distributes vector across processors 
 * giving each about using the distribution of the matrix A.
 *----------------------------------------------------------------------*/

int
BuildRhsParFromOneFile_( int                  argc,
                         char                *argv[],
                         int                  arg_index,
                         int                 *partitioning,
                         HYPRE_ParVector     *b_ptr     )
{
   char           *filename;

   HYPRE_ParVector b;
   HYPRE_Vector    b_CSR;

   int             myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Rhs FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * Generate the matrix 
       *-----------------------------------------------------------*/
 
      b_CSR = HYPRE_VectorRead(filename);
   }
   HYPRE_VectorToParVector(MPI_COMM_WORLD, b_CSR, partitioning,&b); 

   *b_ptr = b;

   HYPRE_VectorDestroy(b_CSR);

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 9-point laplacian in 2D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
BuildParLaplacian9pt( int                  argc,
                      char                *argv[],
                      int                  arg_index,
                      HYPRE_ParCSRMatrix  *A_ptr     )
{
   int                 nx, ny;
   int                 P, Q;

   HYPRE_ParCSRMatrix  A;

   int                 num_procs, myid;
   int                 p, q;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;

   P  = 1;
   Q  = num_procs;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Laplacian 9pt:\n");
      printf("    (nx, ny) = (%d, %d)\n", nx, ny);
      printf("    (Px, Py) = (%d, %d)\n\n", P,  Q);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q from P,Q and myid */
   p = myid % P;
   q = ( myid - p)/P;

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 2);

   values[1] = -1.;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0;
   }
   if (ny > 1)
   {
      values[0] += 2.0;
   }
   if (nx > 1 && ny > 1)
   {
      values[0] += 4.0;
   }

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian9pt(MPI_COMM_WORLD,
                                                 nx, ny, P, Q, p, q, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}
/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
BuildParLaplacian27pt( int                  argc,
                       char                *argv[],
                       int                  arg_index,
                       HYPRE_ParCSRMatrix  *A_ptr     )
{
   int                 nx, ny, nz;
   int                 P, Q, R;

   HYPRE_ParCSRMatrix  A;

   int                 num_procs, myid;
   int                 p, q, r;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Laplacian_27pt:\n");
      printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      printf("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 2);

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
      values[0] = 8.0;
   if (nx*ny == 1 || nx*nz == 1 || ny*nz == 1)
      values[0] = 2.0;
   values[1] = -1.;

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}
