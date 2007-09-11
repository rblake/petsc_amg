
#include <petscksp.h>
#include <petscmg.h>
#include <stdlib.h>


/** This macro outputs a variables name in the code followed by its
 * current value.
 */
#define SHOWVAR(x,t) {				\
    int rank;					\
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);	\
    printf("[%d] " #x " = %" #t "\n", rank, x);	\
    fflush(NULL);				\
}


/// This macro repalces "Got Here" messages
#define LTRACE() {int rank; MPI_Comm_rank(PETSC_COMM_WORLD, &rank); printf(__FILE__ ":%d: [%d] ltrace:\n", __LINE__, rank); }

/// Same as LTRACE(), but can output additional text.
#define LTRACEF(x) printf(__FILE__ ":%d: ltrace: %s\n", __LINE__, x)


typedef struct {
    Mat A;
    Vec x;
    Vec b;
    PetscInt npoints;
    PetscInt num_seg;
    PetscInt levels;
    Mat* prolongation;
} Elliptic1D;

Elliptic1D problem;

float
frand() {
    return (float) random() / (float) 0x7fffffff;
}

void
construct_operator(Elliptic1D* p, int levels) {
    // Start out with a simple tridiagonal matrix.
 
    // problem solves laplacian u = -1 on -1 to 1 with homogeneous
    // dirichlet boundary conditions. Code
    // implements a finite difference to approximate this
    // equation system.

    p->levels = levels;
    p->npoints = (2<<levels)-1;
    p->num_seg = 2<<levels;
    PetscScalar h = 2./p->num_seg;

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
	PetscScalar stencil[3] = {-1./(h*h), 2./(h*h), -1./(h*h)};

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
    
    //Fill in a random initial guess
    PetscInt high;
    PetscInt low;
    VecGetOwnershipRange(p->x, &low, &high);
    for (ii=low; ii<high; ii++) {
	VecSetValue(p->x, ii, frand(), INSERT_VALUES);
    }
    VecAssemblyBegin(p->x);
    VecAssemblyEnd(p->x);
}

void
construct_prolongation_operator(PetscInt level, Mat* pmat) {

    int from_points = (2<<(level-1))-1;
    int to_points = (2<<level)-1;

    MatCreateMPIAIJ(PETSC_COMM_WORLD,
		    PETSC_DECIDE, //number of local rows
		    PETSC_DECIDE, //number of local cols
		    to_points, //global rows
		    from_points, //global cols
		    3,         // upper bound of diagonal nnz per row
		    PETSC_NULL, // array of diagonal nnz per row
		    2,         // upper bound of off-processor nnz per row
		    PETSC_NULL, // array of off-processor nnz per row
		    pmat); // matrix

    MatSetFromOptions(*pmat);

    PetscInt start;
    PetscInt end;
    MatGetOwnershipRange(*pmat, &start, &end);
    int ii;
    for (ii=start; ii<end; ii++) {
	PetscInt row_index = ii;
	if (ii%2 == 0) {
	    PetscInt col_index[2] = {ii/2-1, ii/2};
	    if (ii/2 >= from_points) {
		col_index[1] = -1;
	    }
	    PetscScalar stencil[2] = {.5, .5};
	    MatSetValues(*pmat, 1, &row_index, 2, col_index, stencil, INSERT_VALUES);
	} else {
	    MatSetValue(*pmat, ii, (ii-1)/2, 1, INSERT_VALUES);
	}
    }
    MatAssemblyBegin(*pmat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*pmat, MAT_FINAL_ASSEMBLY);
}

PetscReal
compute_residual_norm(Elliptic1D* problem) {
    Vec r;
    VecDuplicate(problem->b, &r);
    MatMult(problem->A, problem->x, r);
    VecAXPY(r, -1, problem->b);
    VecView(r, PETSC_VIEWER_STDOUT_WORLD);
    PetscReal norm;
    VecNorm(r, NORM_2, &norm);
    return norm;
}

#define CHKERR(code) { PetscErrorCode ierr##__LINE__ = (code); CHKERRQ(ierr##__LINE__); }

int
main(int argc, char** argv) {
    int levels = 8;
    int mg_levels = 3;
    PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
    construct_operator(&problem, levels);

    KSP ksp;
    PC pc;

    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, problem.A, problem.A, SAME_PRECONDITIONER);

    KSPSetType(ksp, KSPRICHARDSON);
    
    if (1) {
	KSPGetPC(ksp, &pc);
	PCSetType(pc, PCMG);
	PCMGSetLevels(pc, mg_levels, NULL);
	PCMGSetGalerkin(pc);
	PCMGSetType(pc, PC_MG_MULTIPLICATIVE);
	PCMGSetCycleType(pc, PC_MG_CYCLE_V);
	int ii;
	for (ii=0; ii<mg_levels; ii++) {
	    if (ii == 0) {
		KSP smooth_ksp;
		PCMGGetSmoother(pc, ii, &smooth_ksp);
		KSPSetType(smooth_ksp, KSPPREONLY);
		PC smooth_pc;
		KSPGetPC(smooth_ksp, &smooth_pc);
		PCSetType(smooth_pc, PCLU);
	    } else {
		// set up the smoother.
		KSP smooth_ksp;
		PC smooth_pc;
		PCMGGetSmoother(pc, ii, &smooth_ksp);
		KSPSetType(smooth_ksp, KSPRICHARDSON);
		KSPRichardsonSetScale(smooth_ksp, 2./3.);
		KSPGetPC(smooth_ksp, &smooth_pc);
		PCSetType(smooth_pc, PCJACOBI);
		KSPSetTolerances(smooth_ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 2);
	
		//set up the interpolation operator
		Mat prolongation;
		construct_prolongation_operator(ii+1+levels-mg_levels, &prolongation);
		PCMGSetInterpolation(pc, ii, prolongation);
		MatScale(prolongation, 1./2.);
		Mat restriction;
		MatTranspose(prolongation, &restriction);
		PCMGSetRestriction(pc, ii, prolongation);
		MatDestroy(prolongation);
		MatDestroy(restriction);
	    }
	}
    } else {
	KSPGetPC(ksp, &pc);
	PCSetType(pc, PCJACOBI);
    }
	//*/
    /*
    if (0) {
	KSPSetType(ksp, KSPRICHARDSON);
	KSPRichardsonSetScale(ksp, 2./3.);
	KSPGetPC(ksp, &pc);
	PCSetType(pc, PCJACOBI);
    } else {
	PetscOptionsInsertString("-ksp_type richardson");
	PetscOptionsInsertString("-ksp_richardson_scale 0.666666666666666666");
	PetscOptionsInsertString("-pc_type jacobi");
    }
    //*/

    KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
    KSPSetFromOptions(ksp);
    KSPSetUp(ksp);

    //VecView(problem.x, PETSC_VIEWER_STDOUT_WORLD);
    {
	//CHKERR(PCApply(pc, problem.b, problem.x));
	CHKERR(KSPSolve(ksp, problem.b, problem.x));

	KSPConvergedReason reason;
	CHKERR(KSPGetConvergedReason(ksp, &reason));
	printf("KSPConvergedReason: %d\n", reason);
	
	PetscInt its;
	CHKERR(KSPGetIterationNumber(ksp, &its));
	printf("Num iterations: %d\n", its);

    }
    //compute_residual_norm(&problem);

    //VecView(problem.x, PETSC_VIEWER_STDOUT_WORLD);

    PetscFinalize();
    return 0;
}
