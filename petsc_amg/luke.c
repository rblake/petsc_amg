/* 
 ------
 log:
  lukeo:  Tue Aug 21 10:31:15 CDT 2007
      - touched
 ------
 
 use
 mpirun -np 8 ./luke args
 or
 $PETSC_DIR/bin/petscmpirun -np 8 ./luke args
 
 */
#include "petscksp.h"

static char help[] = "Easy MG\n\
                      Input Parameters:\n\
                        -kx : integer for sin wave exact soln in x-dir\n\
                        -ky : integer for sin wave exact soln in y-dir\n\
                        -nx : mesh points in x-dir\n\
                        -ny : mesh points in y-dir\n\";

/* TODO: move SCAMG stuff to scamg.h */

/* context for MG preconditioner (0 level) */
typedef struct {
  Vec diag;
} SCAMG;

/* declarations for SCAMG routines */
extern PetscErrorCode SCAMGCreate(SCAMG**);
extern PetscErrorCode SCAMGSetUp(SCAMG*,Mat,Vec);
extern PetscErrorCode SCAMGApply(void*,Vec x,Vec y);
extern PetscErrorCode SCAMGDestroy(SCAMG*);

#undef __FUNCT__  
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            x,b,xex;   /* approx solution, RHS, exact solution */
  Mat            A;         /* linear system matrix */
  KSP            ksp;       /* linear solver context */
  PC             pc;        /* preconditioner context */
  PetscReal      norm;      /* norm of solution error */
  SCAMG          *SCAMGsh;  /* MG context */
  PetscScalar    v,one = 1.0,neg_one = -1.0;
  PetscInt       i,j,Ii,J,Istart,Iend,nx = 10,ny = 8,its,kx=1,ky=1;
  PetscErrorCode ierr;
  PetscTruth     use_SCAMG;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-nx",&nx,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ny",&ny,PETSC_NULL);CHKERRQ(ierr);

  /* Generate Ax=b for a FD 2D Laplacian with Dirichlet BC on [0,1]^2*/
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,nx*ny,nx*ny);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

  for (Ii=Istart; Ii<Iend; Ii++) { 
    v = -1.0; 
    i = Ii/ny; j = Ii - i*ny;  
    if (i>0){  /* west */
      J = Ii - ny; 
      ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (i<nx-1){ /* east */
      J = Ii + ny; 
      ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (j>0){ /* south */
      J = Ii - 1; 
      ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (j<ny-1){ /* north */
      J = Ii + 1;
      ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    v = 4.0; /* center */
    ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&xex);CHKERRQ(ierr);
  ierr = VecSetSizes(xex,PETSC_DECIDE,nx*ny);CHKERRQ(ierr);
  ierr = VecSetFromOptions(xex);CHKERRQ(ierr);
  ierr = VecDuplicate(xex,&b);CHKERRQ(ierr); 
  ierr = VecDuplicate(xex,&x);CHKERRQ(ierr);

  /* xex is sin(kx*pi*x)*sin(ky*pi*y) */
  ierr = VecGetOwnershipRange(xex,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart; i<Iend; i++) {
     x = h*(i % (nx+1)); y = h*(i/(nx+1)); 
     val = sin(kx*3.14159*x)*sin(ky*3.14159*y);
     /* TODO: better PI */
     ierr = VecSetValues(xex,1,&i,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(xex);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(xex);CHKERRQ(ierr);
  ierr = VecSet(x,one);CHKERRQ(ierr);
  ierr = MatMult(A,xex,b);CHKERRQ(ierr);

  /*TODO: got to here */






















  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create linear solver context
  */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

  /* 
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  /* 
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the KSP context,
       we can then directly call any KSP and PC routines
       to set various options.
  */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,
         PETSC_DEFAULT);CHKERRQ(ierr);

  /*
     Set a user-defined "shell" preconditioner if desired
  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-use_SCAMG",&use_SCAMG);CHKERRQ(ierr);
  if (use_SCAMG) {
    /* (Required) Indicate to PETSc that we're using a "shell" preconditioner */
    ierr = PCSetType(pc,PCSHELL);CHKERRQ(ierr);

    /* (Optional) Create a context for the user-defined preconditioner; this
       context can be used to contain any application-specific data. */
    ierr = SCAMGCreate(&SCAMGsh);CHKERRQ(ierr);

    /* (Required) Set the user-defined routine for applying the preconditioner */
    ierr = PCShellSetApply(pc,SCAMGApply);CHKERRQ(ierr);
    ierr = PCShellSetContext(pc,SCAMGsh);CHKERRQ(ierr);

    /* (Optional) Set a name for the preconditioner, used for PCView() */
    ierr = PCShellSetName(pc,"MyPreconditioner");CHKERRQ(ierr);

    /* (Optional) Do any setup required for the preconditioner */
    ierr = SCAMGSetUp(SCAMGsh,A,x);CHKERRQ(ierr);

  } else {
    ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
  }

  /* 
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Check the error
  */
  ierr = VecAXPY(x,neg_one,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A iterations %D\n",norm,its);CHKERRQ(ierr);

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = KSPDestroy(ksp);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);  ierr = MatDestroy(A);CHKERRQ(ierr);

  if (use_SCAMG) {
    ierr = SCAMGDestroy(SCAMGsh);CHKERRQ(ierr);
  }

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;

}

/***********************************************************************/
/*          Routines for a user-defined SCAMG preconditioner           */
/***********************************************************************/

#undef __FUNCT__  
#define __FUNCT__ "SCAMGCreate"
/*
   SCAMGCreate - This routine creates a user-defined
   preconditioner context.

   Output Parameter:
.  SCAMGsh - user-defined preconditioner context
*/
PetscErrorCode SCAMGCreate(SCAMG **SCAMGsh)
{
  SCAMG  *newctx;
  PetscErrorCode ierr;

  ierr         = PetscNew(SCAMG,&newctx);CHKERRQ(ierr);
  newctx->diag = 0;
  *SCAMGsh       = newctx;
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SCAMGSetUp"
/*
   SCAMGSetUp - This routine sets up a user-defined
   preconditioner context.  

   Input Parameters:
.  SCAMGsh - user-defined preconditioner context
.  pmat  - preconditioner matrix
.  x     - vector

   Output Parameter:
.  SCAMGsh - fully set up user-defined preconditioner context

   Notes:
   In this example, we define the shell preconditioner to be Jacobi's
   method.  Thus, here we create a work vector for storing the reciprocal
   of the diagonal of the preconditioner matrix; this vector is then
   used within the routine SCAMGApply().
*/
PetscErrorCode SCAMGSetUp(SCAMG *shell,Mat pmat,Vec x)
{
  Vec            diag;
  PetscErrorCode ierr;

  ierr = VecDuplicate(x,&diag);CHKERRQ(ierr);
  ierr = MatGetDiagonal(pmat,diag);CHKERRQ(ierr);
  ierr = VecReciprocal(diag);CHKERRQ(ierr);
  SCAMGsh>diag = diag;

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SCAMGApply"
/*
   SCAMGApply - This routine demonstrates the use of a
   user-provided preconditioner.

   Input Parameters:
.  ctx - optional user-defined context, as set by PCShellSetContext()
.  x - input vector

   Output Parameter:
.  y - preconditioned vector

   Notes:
   Note that the PCSHELL preconditioner passes a void pointer as the
   first input argument.  This can be cast to be the whatever the user
   has set (via PCSetShellApply()) the application-defined context to be.

   This code implements the Jacobi preconditioner, merely as an
   example of working with a PCSHELL.  Note that the Jacobi method
   is already provided within PETSc.
*/
PetscErrorCode SCAMGApply(void *ctx,Vec x,Vec y)
{
  SCAMG   *SCAMGsh = (SCAMG*)ctx;
  PetscErrorCode  ierr;

  ierr = VecPointwiseMult(y,x,SCAMGsh>diag);CHKERRQ(ierr);

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SCAMGDestroy"
/*
   SCAMGDestroy - This routine destroys a user-defined
   preconditioner context.

   Input Parameter:
.  SCAMGsh - user-defined preconditioner context
*/
PetscErrorCode SCAMGDestroy(SCAMG *SCAMGsh)
{
  PetscErrorCode ierr;

  ierr = VecDestroy(SCAMGsh>diag);CHKERRQ(ierr);
  ierr = PetscFree(SCAMGsh);CHKERRQ(ierr);

  return 0;
}


