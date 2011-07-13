
#include "mglib.h"


Elliptic1D problem;

int
main(int argc, char** argv) {
    int levels = 4;
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

    VecView(problem.x, PETSC_VIEWER_STDOUT_WORLD);

    PetscFinalize();
    return 0;
}
