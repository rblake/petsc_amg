
#include <petscksp.h>

typedef struct {
    Mat A;
    Vec x;
    Vec b;
    PetscInt npoints;
    PetscInt num_seg; 
    Mat* interpolation;
} Elliptic1D;

Elliptic1D problem;

void
construct_operator(Elliptic1D* p, int num_seg) {
    // Start out with a simple tridiagonal matrix.
 
    // problem solves laplacian u = 1 on -1 to 1.  Code
    // implements a finite difference to approximate this
    // equation system.

    p->npoints = num_seg + 1;
    p->num_seg = num_seg;

    MatCreateMPIAIJ(PETSC_COMM_WORLD,
		    PETSC_DECIDE, //number of local rows
		    PETSC_DECIDE, //number of local cols
		    p->npoints, //global rows
		    p->npoints, //global cols
		    3,         // upper bound of diagonal nnz per row
		    PETSC_NULL, // array of diagonal nnz per row
		    1,         // upper bound of off-processor nnz per row
		    PETSC_NULL, // array of off-processor nnz per row
		    &p->A); // matrix

    MatSetFromOptions(p->A);

    PetscInt start;
    PetscInt end;
    MatGetOwnershipRange(p->A, &start, &end);
    int ii;
    for (ii=start; ii<end; ii++) {
	PetscInt col_index[3] = {ii-1, ii, ii+1};
	PetscInt row_index = ii;
	PetscScalar stencil[3] = {-2./p->num_seg, 4./p->num_seg, -2/p->num_seg};

	// handle corner cases at beginning and end of matrix.
	if (ii+1 == p->npoints) {
	    col_index[2] = -1;
	} else if (ii == 0) {
	    col_index[0] = -1;
	}

	MatSetValues(p->A, 1, &row_index, 3, col_index, stencil, INSERT_VALUES);
    }
    MatAssemblyBegin(p->A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(p->A, MAT_FINAL_ASSEMBLY);

    //Create the corresponding vectors
    VecCreateMPI(PETSC_COMM_WORLD,
		 PETSC_DECIDE,
		 p->npoints,
		 &p->x
		 );
    VecSetFromOptions(p->x);
    VecDuplicate(p->x, &p->b);
    VecSet(p->b, 1);
    VecZeroEntries(p->x);
}

#define CHKERR(code) { PetscErrorCode ierr##__LINE__ = (code); CHKERRQ(ierr##__LINE__); }

int
main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
    construct_operator(&problem, 4);
    
    KSP ksp;
    PC pc;

    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, problem.A, problem.A, SAME_PRECONDITIONER);
    KSPSetFromOptions(ksp);
    KSPSetType(ksp, KSPCG);

    CHKERR(KSPSolve(ksp, problem.b, problem.x));

    PetscFinalize();
    return 0;
}
