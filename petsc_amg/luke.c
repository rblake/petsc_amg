/* 
 ------
 log (use :r!date in vim):
  lukeo: Wed Aug 22 09:15:07 CDT 2007
      - started working on the shell context for AMG
  lukeo: Tue Aug 21 10:31:15 CDT 2007
      - touched
      - copied some structure from tutorial/ex*.c
 ------
 
 use
 mpirun -np 8 ./luke args
 or
 $PETSC_DIR/bin/petscmpirun -np 8 ./luke args
 
 */
#include "petscda.h"
#include "petscksp.h"

static char help[] = "Easy MG\n\
                      Input Parameters:\n\
                        -kx : integer for sin wave exact soln in x-dir\n\
                        -ky : integer for sin wave exact soln in y-dir\n\
                        -nx : mesh points in x-dir\n\
                        -ny : mesh points in y-dir\n";

/* TODO: move ScAMG stuff to scamg.h */

/* context for MG preconditioner (0 level) */
typedef struct {
  Vec diaginv;
} ScAMG;

/* declarations for ScAMG routines */
extern PetscErrorCode ScAMGCreate(ScAMG**);
extern PetscErrorCode ScAMGSetUp(ScAMG*,Mat,Vec);
extern PetscErrorCode ScAMGApply(void*,Vec x,Vec y);
extern PetscErrorCode ScAMGDestroy(ScAMG*);

#undef __FUNCT__  
#define __FUNCT__ "main"
int 
main(int argc,char **args) {
  /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     variables
  */
  Vec            u,b,uex,uerr;   /* approx solution, RHS, exact solution, error */
  Mat            A;         /* linear system matrix */
  KSP            ksp;       /* linear solver context */
  PC             pc;        /* preconditioner context */
  PetscReal      norm;      /* norm of solution error */
  ScAMG          *ScAMGsh;  /* MG context */
  PetscScalar    v,one = 1.0,neg_one = -1.0;
  PetscInt       i,j,Ii,J,Istart,Iend,nx = 10,ny = 8,its,kx=1,ky=1;
  PetscErrorCode ierr;
  PetscTruth     use_ScAMG;
  PetscReal      ksp_tol=1.e-7;
  PetscReal      hx,hy;
  PetscReal      x,y,val;
  PetscLogDouble v1,v2,elapsed_time;

  /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     initialization
  */
  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-nx",&nx,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ny",&ny,PETSC_NULL);CHKERRQ(ierr);

  /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     problem setup
  */
  /* Generate Au=b for a FD 2D Laplacian with Dirichlet BC on [0,1]^2*/
  hx = 1.0/(nx-1);
  hy = 1.0/(ny-1);
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

  ierr = VecCreate(PETSC_COMM_WORLD,&uex);CHKERRQ(ierr);
  ierr = VecSetSizes(uex,PETSC_DECIDE,nx*ny);CHKERRQ(ierr);
  ierr = VecSetFromOptions(uex);CHKERRQ(ierr);
  ierr = VecDuplicate(uex,&b);CHKERRQ(ierr); 
  ierr = VecDuplicate(uex,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(uex,&uerr);CHKERRQ(ierr);

  /* uex is sin(kx*pi*x)*sin(ky*pi*y) */
  ierr = VecGetOwnershipRange(uex,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart; i<Iend; i++) {
    x = hx*(i % (nx+1)); 
    y = hy*(i/(nx+1)); 
     val = sin(kx*PETSC_PI*x)*sin(ky*PETSC_PI*y);
     ierr = VecSetValues(uex,1,&i,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(uex);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(uex);CHKERRQ(ierr);
  ierr = VecSet(u,one);CHKERRQ(ierr);
  ierr = VecSet(uerr,0.0);CHKERRQ(ierr);
  /* ierr = VecSet(uex,one);CHKERRQ(ierr); */
  ierr = MatMult(A,uex,b);CHKERRQ(ierr);

  /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    solver setup
  */
  /* Create linear solver context */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,ksp_tol,PETSC_DEFAULT,PETSC_DEFAULT, PETSC_DEFAULT);CHKERRQ(ierr);

  /* use ScAMG or not */
  ierr = PetscOptionsHasName(PETSC_NULL,"-use_ScAMG",&use_ScAMG);CHKERRQ(ierr);
  if (use_ScAMG) {
    ierr = PCSetType(pc,PCSHELL);CHKERRQ(ierr);
    ierr = ScAMGCreate(&ScAMGsh);CHKERRQ(ierr);
    ierr = PCShellSetApply(pc,ScAMGApply);CHKERRQ(ierr);
    ierr = PCShellSetContext(pc,ScAMGsh);CHKERRQ(ierr);
    ierr = PCShellSetName(pc,"ScAMG");CHKERRQ(ierr);
    ierr = ScAMGSetUp(ScAMGsh,A,u);CHKERRQ(ierr);

  } else {
    ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
  }

  /* override with runtime args */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    solve
  */
  ierr = PetscGetTime(&v1);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,u);CHKERRQ(ierr);
  ierr = PetscGetTime(&v2);CHKERRQ(ierr);
  elapsed_time = v2 - v1;

  /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    post-process
  */
  /* Check the error uerr=uex-u */
  ierr = VecWAXPY(uerr,neg_one,u,uex);CHKERRQ(ierr);
  ierr = VecNorm(uerr,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"time: %A  ||e|| = %G after %D iterations\n",elapsed_time,norm,its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%A\n",elapsed_time);CHKERRQ(ierr);

  /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    clean up
  */
  ierr = KSPDestroy(ksp);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);  
  ierr = VecDestroy(uex);CHKERRQ(ierr);
  ierr = VecDestroy(uerr);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);  
  ierr = MatDestroy(A);CHKERRQ(ierr);

  if (use_ScAMG) {
    ierr = ScAMGDestroy(ScAMGsh);CHKERRQ(ierr);
  }

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

/*>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 Routines for ScAMG
*/

/*
   ScAMGCreate - create ScAMG context

   Output Parameter:
.  ScAMGsh - ScAMG context
*/
#undef __FUNCT__  
#define __FUNCT__ "ScAMGCreate"
PetscErrorCode 
ScAMGCreate(ScAMG **ScAMGsh) {
  ScAMG  *context;
  PetscErrorCode ierr;

  ierr         = PetscNew(ScAMG,&context);CHKERRQ(ierr);
  context->diaginv = 0;
  *ScAMGsh       = context;
  return 0;
}

/*
   ScAMGSetUp - set up ScAMG context  

   Input Parameters:
.  ScAMGsh - ScAMG context
.  M  - preconditioner matrix (usually A; set in KSP setup)
.  x     - vector

   Output Parameter:
.  ScAMGsh - initialized ScAMG context
*/
#undef __FUNCT__  
#define __FUNCT__ "ScAMGSetUp"
PetscErrorCode 
ScAMGSetUp(ScAMG *ScAMGsh,Mat M,Vec x) {
  Vec            diaginv;
  PetscErrorCode ierr;

  ierr = VecDuplicate(x,&diaginv);CHKERRQ(ierr);
  ierr = MatGetDiagonal(M,diaginv);CHKERRQ(ierr);
  ierr = VecReciprocal(diaginv);CHKERRQ(ierr);
  ScAMGsh->diaginv = diaginv;

  return 0;
}

/*
   ScAMGApply - Run ScAMG

   Input Parameters:
.  context - generic context, as set by PCShellSetContext()
.  x - input vector

   Output Parameter:
.  y - preconditioned vector
*/
#undef __FUNCT__  
#define __FUNCT__ "ScAMGApply"
PetscErrorCode
ScAMGApply(void *context,Vec x,Vec y) {
  ScAMG   *ScAMGsh = (ScAMG*)context;
  PetscErrorCode  ierr;

  ierr = VecPointwiseMult(y,x,ScAMGsh->diaginv);CHKERRQ(ierr);

  return 0;
}

/*
   ScAMGDestroy - destroy the ScAMG context

   Input Parameter:
.  ScAMGsh - ScAMG context
*/
#undef __FUNCT__  
#define __FUNCT__ "ScAMGDestroy"
PetscErrorCode
ScAMGDestroy(ScAMG *ScAMGsh) {
  PetscErrorCode ierr;

  ierr = VecDestroy(ScAMGsh->diaginv);CHKERRQ(ierr);
  ierr = PetscFree(ScAMGsh);CHKERRQ(ierr);

  return 0;
}
