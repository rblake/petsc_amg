
#include "petscvec.h"
#include "petscda.h"
#include "petscmat.h"

//extern "C" {
#include "mglib.h"
//}




const char     *bcTypes[2] = {"dirichlet","neumann"};
typedef enum {DIRICHLET, NEUMANN} BCType;
typedef struct {
    PetscScalar   rho;
    PetscScalar   nu;
    BCType        bcType;
    DA da;
} UserContext;


void
find_influences_with_tag
//////////////////////////////////////
(
 Mat A,
 /// Matrix
 IS interest_set,
 /// set of points we're interested in.  Local rows only?
 IS tag,
 /// tag we'd like to select from.  Local rows only.
 IS *pInfluences
 /// Set of influences we need.  Nonlocal by def.
 ) {
    Vec input_vector;
}

void
construct_amg_prolongation
(
 /// index map of local to higher points
 IS coarse,
 IS fine,
 IS depend_strong,
 IS depend_weak,
 IS depend_coarse,
 Mat A,
 Mat* pmat
 ) {

    // Start constructing the local submatricies needed.
    enum submatrix {Fp_Dpc, Fp_Dps, Dps_Dpc, Fp_Dpw, num_submatrix};
    Mat* local_submatrix;
    IS irow[num_submatrix];
    IS icol[num_submatrix];
    
    irow[Fp_Dpc] = fine;
    icol[Fp_Dpc] = depend_coarse;

    irow[Fp_Dps] = fine;
    icol[Fp_Dps] = depend_strong;

    irow[Dps_Dpc] = depend_strong;
    icol[Dps_Dpc] = depend_coarse;

    irow[Fp_Dpw] = fine;
    icol[Fp_Dpw] = depend_weak;

    MatGetSubMatrices(A, num_submatrix, irow, icol, MAT_INITIAL_MATRIX, &local_submatrix);
    
    // Note, due to RS C1, we know that the Fp_Dpc matrix has 
    // the same non-zero pattern as the Fp_Dps x Dps_Dpc matrix
    
    Vec row_sum;
    MatGetVecs(local_submatrix[Dps_Dpc], PETSC_NULL, &row_sum);
    MatGetRowSum(local_submatrix[Dps_Dpc], row_sum);
    VecReciprocal(row_sum);
    MatDiagonalScale(local_submatrix[Dps_Dpc], row_sum, PETSC_NULL);
    VecDestroy(row_sum);

    Mat result;
    MatMatMult(local_submatrix[Fp_Dps], local_submatrix[Dps_Dpc], 
	       MAT_INITIAL_MATRIX, 4, &result);
    
    //Because of the special form of result, we should be able to optimize this
    //addition.  Both matricies should have the same non-zero structure.
    
    //TODO: add a check here to verify that assertion.

    MatAYPX(result, 1, local_submatrix[Fp_Dpc], SAME_NONZERO_PATTERN);
    
    MatGetVecs(local_submatrix[Fp_Dpw], PETSC_NULL, &row_sum);
    // Add entries from the diagonal
    VecReciprocal(row_sum);
    VecScale(row_sum, -1);
    MatDiagonalScale(result, row_sum, PETSC_NULL);
    VecDestroy(row_sum);

    MatView(result, PETSC_VIEWER_DRAW_WORLD);
    
    return;

    /*
    Vec interp_scale;

    
    PetscInt from_size;
    ISGetSize(to_indicies, &from_size);

    

    PetscInt local_from_size;
    ISGetLocalSize(to_indicies, &local_from_size);
    MatCreateMPIAIJ(PETSC_COMM_WORLD,
		    PETSC_DECIDE, //number of local rows
		    local_from_size, //number of local cols
		    to_size, //global rows
		    from_size, //global cols
		    // TODO: figure out what values should go here.
		    3,         // upper bound of diagonal nnz per row
		    PETSC_NULL, // array of diagonal nnz per row
		    // TODO: figure out what values should go here
		    2,         // upper bound of off-processor nnz per row
		    PETSC_NULL, // array of off-processor nnz per row
		    pmat); // matrix
    
    

    PetscInt start;
    PetscInt end;
    MatGetOwnershipRange(*pmat, &start, &end);
    int ii;

    void

    //for all local rows
    for (ii=start; ii<end; ii++) {
	//if the point is a coarse point, 
	if (0) {
	    //introduce an identity row
	    
	} else {
	    //otherwise, we have a fine point

	    //add weights for other connected coarse points
	    //add weights for other strongly connected fine points also

	    //add weights for weak connections to other fine points

	}
    }
    MatAssemblyBegin(*pmat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*pmat, MAT_FINAL_ASSEMBLY);
    */
}



//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////


PetscErrorCode ComputeRHS(UserContext *user, Vec *pb)
{
  DA             da = (DA)user->da;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    Hx,Hy;
  PetscScalar    **array;

  Vec g;
  CHKERR(DAGetGlobalVector(da, &g));
  CHKERR(VecDuplicate(g, pb));
  CHKERR(DARestoreGlobalVector(da, &g));

  ierr = DAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx   = 1.0 / (PetscReal)(mx-1);
  Hy   = 1.0 / (PetscReal)(my-1);
  ierr = DAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAVecGetArray(da, *pb, &array);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++){
    for(i=xs; i<xs+xm; i++){
      array[j][i] = PetscExpScalar(-((PetscReal)i*Hx)*((PetscReal)i*Hx)/user->nu)*PetscExpScalar(-((PetscReal)j*Hy)*((PetscReal)j*Hy)/user->nu)*Hx*Hy;
    }
  }
  ierr = DAVecRestoreArray(da, *pb, &array);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(*pb);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*pb);CHKERRQ(ierr);

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) {
      //MatNullSpace nullspace;

      //ierr = KSPGetNullSpace(dmmg->ksp,&nullspace);CHKERRQ(ierr);
      //ierr = MatNullSpaceRemove(nullspace,b,PETSC_NULL);CHKERRQ(ierr);
  }
  return ierr;
}

    
#undef __FUNCT__
#define __FUNCT__ "ComputeRho"
PetscErrorCode ComputeRho(PetscInt i, PetscInt j, PetscInt mx, PetscInt my, PetscScalar centerRho, PetscScalar *rho)
{
  PetscFunctionBegin;
  *rho = 1.0;
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeJacobian(UserContext *user, Mat *pA)
{
  DA             da = user->da;
  PetscScalar    centerRho = user->rho;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys,num;
  PetscScalar    v[9],Hx,Hy,H,rho;
  MatStencil     row, col[9];

  DAGetMatrix(da, MATMPIAIJ, pA);

  ierr = DAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);  
  Hx    = 1.0 / (PetscReal)(mx-1);
  Hy    = 1.0 / (PetscReal)(my-1);
  H = Hx*Hy;
  ierr = DAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++){
    for(i=xs; i<xs+xm; i++){
      row.i = i; row.j = j;
      ierr = ComputeRho(i, j, mx, my, centerRho, &rho);CHKERRQ(ierr);
      if (i==0 || j==0 || i==mx-1 || j==my-1) {
        if (user->bcType == DIRICHLET) {
           v[0] = 8.0*rho*H;
          ierr = MatSetValuesStencil(*pA,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
        } else if (user->bcType == NEUMANN) {
          num = 0;
          if (j!=0) {
            v[num] = -rho*H;              col[num].i = i;   col[num].j = j-1;
            num++;
          }
          if (i!=0) {
            v[num] = -rho*H;              col[num].i = i-1; col[num].j = j;
            num++;
          }
          if (i!=mx-1) {
            v[num] = -rho*H;              col[num].i = i+1; col[num].j = j;
            num++;
          }
          if (j!=my-1) {
            v[num] = -rho*H;              col[num].i = i;   col[num].j = j+1;
            num++;
          }
          if (j!=0 && i!=0) {
            v[num] = -rho*H;              col[num].i = i-1; col[num].j = j-1;
            num++;
          }
          if (j!=my-1 && i!=0) {
            v[num] = -rho*H;              col[num].i = i-1; col[num].j = j+1;
            num++;
          }
          if (j!=0 && i!=mx-1) {
            v[num] = -rho*H;              col[num].i = i+1; col[num].j = j-1;
            num++;
          }
          if (j!=my-1 && i!=mx-1) {
            v[num] = -rho*H;              col[num].i = i+1; col[num].j = j+1;
            num++;
          }
          v[num]   = (num)*rho*H; col[num].i = i;   col[num].j = j;
          num++;
          ierr = MatSetValuesStencil(*pA,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else {
        v[0] = -rho*H;              col[0].i = i;   col[0].j = j-1;
        v[1] = -rho*H;              col[1].i = i-1; col[1].j = j;
        v[2] = 8.0*rho*H;           col[2].i = i;   col[2].j = j;
        v[3] = -rho*H;              col[3].i = i+1; col[3].j = j;
        v[4] = -rho*H;              col[4].i = i;   col[4].j = j+1;
        v[5] = -rho*H;              col[5].i = i-1; col[5].j = j-1;
        v[6] = -rho*H;              col[6].i = i-1; col[6].j = j+1;
        v[7] = -rho*H;              col[7].i = i+1; col[7].j = j-1;
        v[8] = -rho*H;              col[8].i = i+1; col[8].j = j+1;
        ierr = MatSetValuesStencil(*pA,1,&row,9,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(*pA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*pA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return ierr;
}


int 
main(int argc, char** argv) {
  PetscErrorCode ierr;
  UserContext user;
  user.bcType = NEUMANN;

  PetscInitialize(&argc,&argv,(char *)0,PETSC_NULL);

  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,-3,-3,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&user.da);CHKERRQ(ierr);
  DASetFieldName(user.da, 0, "z");
  DAView(user.da, PETSC_VIEWER_STDOUT_WORLD);

  Vec b;
  ComputeRHS(&user, &b);
  Vec x;
  VecDuplicate(b, &x);
  VecZeroEntries(x);
  
  Mat A;
  ComputeJacobian(&user, &A);
  MatView(A, PETSC_VIEWER_DRAW_WORLD);

  Mat requests_from;
  build_strength_matrix(A, 0.25, &requests_from);
  MatView(requests_from, PETSC_VIEWER_DRAW_WORLD);

  Mat provides_to;
  MatTranspose(requests_from, &provides_to);
  //MatView(provides_to, PETSC_VIEWER_DRAW_WORLD);
  
  IS coarse;
  cljp_coarsening(requests_from, &coarse);
  
  IS fine;
  get_compliment(A, coarse, &fine);
  

  MatDestroy(provides_to);
  MatDestroy(requests_from);

  ierr = VecDestroy(b);
  ierr = MatDestroy(A);
  ierr = DADestroy(user.da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}


