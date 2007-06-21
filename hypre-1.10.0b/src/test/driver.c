#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "parcsr_mv.h"
#include "krylov.h"


/** Describes the linear system Ax = b using a parallel CSR format.
    The exact solution (if available) is given in "exact" */
typedef struct
{
   hypre_ParCSRMatrix * A;
   hypre_ParVector * x;
   hypre_ParVector * b;
   hypre_ParVector * exact;
} AFEM_ParCSR_Problem;

/** Generates A, x and b, given the description of the mesh, the continuous
    problem, the discretization parameters and the number of refinement
    levels (in serial and in parallel) */
AFEM_ParCSR_Problem *
AFEM_GenerateParCSRProblem (char *meshfile, char *pdefile, char *discrfile,
			    int sref, MPI_Comm MyComm, int pref, int spm);

/** Deallocates the memory */
int AFEM_DestroyParCSRProblem (AFEM_ParCSR_Problem * prob);

/** Visualization of the solution */
int AFEM_VisualizeSolution (char *meshfile, char *pdefile, char *discrfile,
			    int sref, MPI_Comm MyComm, int pref,
                            void *x, void * part, int spm);



int main (int argc, char *argv[])
{
   int num_procs, myid;
   int time_index;

   int solver_id;
   int maxit, kdim;
   double tol, theta;
   HYPRE_Solver solver, precond;

   AFEM_ParCSR_Problem *prob;
   int mesh_index;
   int pde_index;
   int discr_index;
   int sref, spm, pref, vis;
   int print_system, no_solve;
   char filename[100];

   HYPRE_ParCSRMatrix A;
   HYPRE_ParVector x, b;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /* Set defaults */
   solver_id = 0;
   maxit = 100;
   tol = 1e-6;
   kdim = 20;
   theta = 0.25;
   sref = 0;
   spm = 2;
   pref = 0;
   vis = 0;
   mesh_index = pde_index =discr_index = 0;
   print_system = 0;
   no_solve = 0;

   /* Parse command line */
   {
      int arg_index = 0;
      int print_usage = 0;

      while (arg_index < argc)
      {
         if ( strcmp(argv[arg_index], "-solver") == 0 )
         {
            arg_index++;
            solver_id = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-maxit") == 0 )
         {
            arg_index++;
            maxit = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-tol") == 0 )
         {
            arg_index++;
            tol = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-kdim") == 0 )
         {
            arg_index++;
            kdim = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-theta") == 0 )
         {
            arg_index++;
            theta = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-mesh") == 0 )
         {
            arg_index++;
            mesh_index = arg_index;
         }
         else if ( strcmp(argv[arg_index], "-pde") == 0 )
         {
            arg_index++;
            pde_index = arg_index;
         }
         else if ( strcmp(argv[arg_index], "-discr") == 0 )
         {
            arg_index++;
            discr_index = arg_index;
         }
         else if ( strcmp(argv[arg_index], "-sref") == 0 )
         {
            arg_index++;
            sref = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-spm") == 0 )
         {
            arg_index++;
            spm = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-pref") == 0 )
         {
            arg_index++;
            pref = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-vis") == 0 )
         {
            arg_index++;
            vis = 1;
         }
         else if ( strcmp(argv[arg_index], "-print") == 0 )
         {
	   arg_index++;
	   print_system = 1;
	   strcpy(filename, argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-nosolve") == 0 )
         {
	   arg_index++;
	   solver_id = -1;
         }
         else if ( strcmp(argv[arg_index], "-help") == 0 )
         {
            print_usage = 1;
            break;
         }
         else
         {
            arg_index++;
         }
      }

      if ((print_usage) && (myid == 0))
      {
         printf("\n");
         printf("Usage: mpirun -np <np> %s [<options>]\n", argv[0]);
         printf("\n");
         printf("  aFEM Problem generation options:                             \n");
         printf("    -mesh <meshfile>     : the finite element mesh             \n");
         printf("    -sref <num>          : number of serial mesh refinements   \n");
         printf("    -spm <num>           : serial partition method [0-3]       \n");
         printf("    -pref <num>          : number of parallel mesh refinements \n");
         printf("    -pde <pdefile>       : the pde describing A                \n");
         printf("    -discr <discrfile>   : finite element discretization       \n");
         printf("    -vis                 : visualize the solution (GLVis)      \n");
         printf("    -print <filename>    : write the ParCSR matrix to file     \n");
         printf("    -nosolve             : do not solve the linear system      \n");
         printf("  Note: The refinement factor for tetrahedral meshes is 2!     \n");
         printf("\n");
         printf("  Hypre solvers options:                                       \n");
         printf("    -solver <ID>         : solver ID                           \n");
         printf("                           0  - AMG (default)                  \n");
         printf("                           1  - AMG-PCG                        \n");
         printf("                           2  - DS-PCG                         \n");
         printf("                           3  - AMG-GMRES                      \n");
         printf("                           4  - DS-GMRES                       \n");
         printf("                           8  - ParaSails-PCG                  \n");
         printf("                           12 - Schwarz-PCG                    \n");
         printf("                           18 - ParaSails-GMRES                \n");
         printf("    -maxit <num>         : maximum number of iterations (100)  \n");
         printf("    -tol <num>           : convergence tolerance (1e-6)        \n");
         printf("    -kdim <num>          : GMRES restart parameter (20)        \n");
         printf("    -theta <num>         : BoomerAMG threshold (0.25)          \n");
         printf("\n");
      }

      if (print_usage)
      {
         MPI_Finalize();
         return (0);
      }
   }

   if (mesh_index*pde_index*discr_index == 0)
   {
      if (myid == 0)
         printf ("Please specify mesh, pde and discr files\n");
      MPI_Finalize();
      return 0;
   }

   if (spm == 3)
   {
      int num_procs2 = 1<<(int)floor(log(num_procs)/0.69314718055994530942+0.5);
      if (num_procs != num_procs2)
      {
         if (myid == 0)
            printf ("Serial partition method 3 cannot be used for %d processors (try %d)\n",
                    num_procs, num_procs2);
         MPI_Finalize();
         return 0;
      }
   }

   /* Start timing */
   time_index = hypre_InitializeTiming("aFEM problem generation");
   hypre_BeginTiming(time_index);

   /* FEM problem generation */
   prob = AFEM_GenerateParCSRProblem (argv[mesh_index],
                                      argv[pde_index],
                                      argv[discr_index],
                                      sref, MPI_COMM_WORLD, pref, spm);

   /* Finalize aFEM timing */
   hypre_EndTiming(time_index);
   hypre_PrintTiming("aFEM problem generation times", MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   A = (HYPRE_ParCSRMatrix) prob->A;
   x = (HYPRE_ParVector) prob->x;
   b = (HYPRE_ParVector) prob->b;

   if(print_system)
     hypre_ParCSRMatrixPrint((hypre_ParCSRMatrix *)A, filename);

   /* Choose a solver and solve the system */

   /* AMG */
   if (solver_id == 0)
   {
      int num_iterations;
      double final_res_norm;

      /* Start timing */
      time_index = hypre_InitializeTiming("BoomerAMG Setup");
      hypre_BeginTiming(time_index);

      /* Create solver */
      HYPRE_BoomerAMGCreate(&solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_BoomerAMGSetPrintLevel(solver, 3);  /* print solve info + parameters */
      HYPRE_BoomerAMGSetCoarsenType(solver, 6); /* Falgout coarsening */
      HYPRE_BoomerAMGSetRelaxType(solver, 3);   /* G-S/Jacobi hybrid relaxation */
      HYPRE_BoomerAMGSetNumSweeps(solver, 1);   /* Sweeeps on each level */
      HYPRE_BoomerAMGSetMaxLevels(solver, 20);  /* maximum number of levels */
      HYPRE_BoomerAMGSetTol(solver, tol);       /* conv. tolerance */
      HYPRE_BoomerAMGSetMaxIter(solver, maxit); /* maximum number of iterations */
      HYPRE_BoomerAMGSetStrongThreshold(solver, theta);

      HYPRE_BoomerAMGSetup(solver, A, b, x);

      /* Finalize setup timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Start timing again */
      time_index = hypre_InitializeTiming("BoomerAMG Solve");
      hypre_BeginTiming(time_index);

      /* Solve */
      HYPRE_BoomerAMGSolve(solver, A, b, x);

      /* Finalize solve timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Run info - needed logging turned on */
      HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver */
      HYPRE_BoomerAMGDestroy(solver);
   }

   /* PCG solvers */
   else if (solver_id == 1 || solver_id == 2 || solver_id == 8 || solver_id == 12)
   {
      int num_iterations;
      double final_res_norm;

      /* Start timing */
      if (solver_id == 1)
         time_index = hypre_InitializeTiming("BoomerAMG-PCG Setup");
      else if (solver_id == 2)
         time_index = hypre_InitializeTiming("DS-PCG Setup");
      else if (solver_id == 8)
         time_index = hypre_InitializeTiming("ParaSails-PCG Setup");
      else if (solver_id == 12)
         time_index = hypre_InitializeTiming("Schwarz-PCG Setup");
      hypre_BeginTiming(time_index);

      /* Create solver */
      HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_PCGSetMaxIter(solver, maxit); /* max iterations */
      HYPRE_PCGSetTol(solver, tol); /* conv. tolerance */
      HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
      HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
      HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

      /* PCG with AMG preconditioner */
      if (solver_id == 1)
      {
         /* Now set up the AMG preconditioner and specify any parameters */
         HYPRE_BoomerAMGCreate(&precond);
         HYPRE_BoomerAMGSetPrintLevel(precond, 1);  /* print amg solution info */
         HYPRE_BoomerAMGSetCoarsenType(precond, 6); /* Falgout coarsening */
         HYPRE_BoomerAMGSetRelaxType(precond, 6);   /* Sym G.S./Jacobi hybrid */
         HYPRE_BoomerAMGSetNumSweeps(precond, 1);   /* Sweeeps on each level */
         HYPRE_BoomerAMGSetMaxLevels(precond, 20);  /* maximum number of levels */
         HYPRE_BoomerAMGSetTol(precond, 1e-3);      /* conv. tolerance (if needed) */
         HYPRE_BoomerAMGSetMaxIter(precond, 1);     /* do only one iteration! */
         HYPRE_BoomerAMGSetStrongThreshold(precond, theta);

         /* Set the PCG preconditioner */
         HYPRE_PCGSetPrecond(solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                             precond);
      }
      /* PCG with diagonal scaling preconditioner */
      else if (solver_id == 2)
      {
         /* Set the PCG preconditioner */
         HYPRE_PCGSetPrecond(solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                             NULL);
      }
      /* PCG with ParaSails preconditioner */
      else if (solver_id == 8)
      {
         double sai_threshold = 0.1;
         int sai_max_levels   = 1;
         double sai_filter    = 0.1;
         int sai_sym          = 1;
         int sai_logging      = 1;

         /* Now set up the ParaSails preconditioner and specify any parameters */
	 HYPRE_ParaSailsCreate(MPI_COMM_WORLD, &precond);
         HYPRE_ParaSailsSetParams(precond, sai_threshold, sai_max_levels);
         HYPRE_ParaSailsSetFilter(precond, sai_filter);
         HYPRE_ParaSailsSetSym (precond, sai_sym);
         HYPRE_ParaSailsSetLogging(precond, sai_logging);

         /* Set the PCG preconditioner */
         HYPRE_PCGSetPrecond(solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSetup,
                             precond);
      }
      /* PCG with Schwarz preconditioner */
      else if (solver_id == 12)
      {
         int variant               = 0; /* multiplicative */
         int overlap               = 1; /* 1 layer overlap */
         int domain_type           = 2; /* through agglomeration */
         double schwarz_rlx_weight = 1.;

         /* Now set up the Schwarz preconditioner and specify any parameters */
	 HYPRE_SchwarzCreate(&precond);
	 HYPRE_SchwarzSetVariant(precond, variant);
	 HYPRE_SchwarzSetOverlap(precond, overlap);
	 HYPRE_SchwarzSetDomainType(precond, domain_type);
         HYPRE_SchwarzSetRelaxWeight(precond, schwarz_rlx_weight);

         /* Set the PCG preconditioner */
         HYPRE_PCGSetPrecond(solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_SchwarzSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_SchwarzSetup,
                             precond);
      }

      /* Setup */
      HYPRE_ParCSRPCGSetup(solver, A, b, x);

      /* Finalize setup timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Start timing again */
      if (solver_id == 1)
         time_index = hypre_InitializeTiming("BoomerAMG-PCG Solve");
      else if (solver_id == 2)
         time_index = hypre_InitializeTiming("DS-PCG Solve");
      else if (solver_id == 8)
         time_index = hypre_InitializeTiming("ParaSails-PCG Solve");
      else if (solver_id == 12)
         time_index = hypre_InitializeTiming("Schwarz-PCG Solve");
      hypre_BeginTiming(time_index);

      /* Solve */
      HYPRE_ParCSRPCGSolve(solver, A, b, x);

      /* Finalize solve timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Run info - needed logging turned on */
      HYPRE_PCGGetNumIterations(solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver and preconditioner */
      HYPRE_ParCSRPCGDestroy(solver);
      if (solver_id == 1)
         HYPRE_BoomerAMGDestroy(precond);
      else if (solver_id == 8)
         HYPRE_ParaSailsDestroy(precond);
      else if (solver_id == 12)
         HYPRE_SchwarzDestroy(precond);
   }

   /* GMRES solvers */
   else if (solver_id == 3 || solver_id == 4 || solver_id == 18)
   {
      int num_iterations;
      double final_res_norm;

      /* Start timing */
      if (solver_id == 3)
         time_index = hypre_InitializeTiming("BoomerAMG-GMRES Setup");
      else if (solver_id == 4)
         time_index = hypre_InitializeTiming("DS-GMRES Setup");
      else if (solver_id == 18)
         time_index = hypre_InitializeTiming("ParaSails-GMRES Setup");
      hypre_BeginTiming(time_index);

      /* Create solver */
      HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_GMRESSetKDim(solver, kdim); /* restart parameter */
      HYPRE_GMRESSetMaxIter(solver, maxit); /* max iterations */
      HYPRE_GMRESSetTol(solver, tol); /* conv. tolerance */
      HYPRE_GMRESSetLogging(solver, 2); /* print solve info */
      HYPRE_GMRESSetPrintLevel(solver, 2); /* needed to get run info later */

      /* GMRES with AMG preconditioner */
      if (solver_id == 3)
      {
         /* Now set up the AMG preconditioner and specify any parameters */
         HYPRE_BoomerAMGCreate(&precond);
         HYPRE_BoomerAMGSetPrintLevel(precond, 1); /* print amg solution info*/
         HYPRE_BoomerAMGSetCoarsenType(precond, 6);
         HYPRE_BoomerAMGSetRelaxType(precond, 3);
         HYPRE_BoomerAMGSetNumSweeps(precond, 1);
         HYPRE_BoomerAMGSetTol(precond, 1e-3);
         HYPRE_BoomerAMGSetMaxIter(precond, 1);    /* do only one iteration! */
         HYPRE_BoomerAMGSetStrongThreshold(precond, theta);

         /* Set the GMRES preconditioner */
         HYPRE_GMRESSetPrecond(solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                               precond);
      }
      /* GMRES with diagonal scaling preconditioner */
      else if (solver_id == 4)
      {
         /* Set the GMRES preconditioner */
         HYPRE_GMRESSetPrecond(solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                               NULL);
      }
      /* GMRES with ParaSails preconditioner */
      else if (solver_id == 18)
      {
         double sai_threshold = 0.1;
         int sai_max_levels   = 1;
         double sai_filter    = 0.1;
         int sai_sym          = 0;
         int sai_logging      = 1;

         /* Now set up the ParaSails preconditioner and specify any parameters */
	 HYPRE_ParaSailsCreate(MPI_COMM_WORLD, &precond);
         HYPRE_ParaSailsSetParams(precond, sai_threshold, sai_max_levels);
         HYPRE_ParaSailsSetFilter(precond, sai_filter);
         HYPRE_ParaSailsSetSym (precond, sai_sym);
         HYPRE_ParaSailsSetLogging(precond, sai_logging);

         /* Set the PCG preconditioner */
         HYPRE_GMRESSetPrecond(solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSetup,
                               precond);
      }

      /* Setup */
      HYPRE_GMRESSetup(solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);

      /* Finalize setup timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Start timing again */
      if (solver_id == 3)
         time_index = hypre_InitializeTiming("BoomerAMG-GMRES Solve");
      else if (solver_id == 4)
         time_index = hypre_InitializeTiming("DS-GMRES Solve");
      else if (solver_id == 18)
         time_index = hypre_InitializeTiming("ParaSails-GMRES Solve");
      hypre_BeginTiming(time_index);

      /* Solve */
      HYPRE_GMRESSolve(solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);

      /* Finalize solve timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Run info - needed logging turned on */
      HYPRE_GMRESGetNumIterations(solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver and preconditioner */
      HYPRE_ParCSRGMRESDestroy(solver);
      if (solver_id == 3)
         HYPRE_BoomerAMGDestroy(precond);
      else if (solver_id == 18)
         HYPRE_ParaSailsDestroy(precond);
   }

   /* Visualization */
   if (vis)
      AFEM_VisualizeSolution (argv[mesh_index],
                              argv[pde_index],
                              argv[discr_index],
                              sref, MPI_COMM_WORLD, pref,
                              (hypre_ParVector*) x, NULL, spm);

   /* Clean-up */
   AFEM_DestroyParCSRProblem(prob);

   MPI_Finalize();

   return (0);
}
