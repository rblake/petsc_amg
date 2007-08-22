#CFLAGS	         = -I${PETSC_DIR}/src/dm/mesh/sieve ${BOOST_INCLUDE} ${TRIANGLE_INCLUDE} ${TETGEN_INCLUDE}
FFLAGS	         = 
CPPFLAGS         = 
FPPFLAGS         =
LOCDIR           = /home/lukeo/repos_working_copies/pamg/trunk/petsc_amg
MANSEC           = KSP
#CLEANFILES       = rhs.vtk solution.vtk
CLEANFILES       =
NP               = 1
DISPLAY          = :0.0

include ${PETSC_DIR}/bmake/common/base

luke: luke.o  chkopts
	-${CLINKER} -o luke luke.o  ${PETSC_KSP_LIB}
	#${RM} luke.o

runluke:
	echo "1 proc"
	@${MPIRUN} -np 1 ./luke -use_ScAMG -ksp_gmres_cgs_refinement_type refine_always -nx 4 -ny 4 > luke.out 2>&1;
	echo "1 proc"
	@${MPIRUN} -np 1 ./luke -use_ScAMG -ksp_gmres_cgs_refinement_type refine_always -nx 8 -ny 8 >> luke.out 2>&1;
	echo "1 proc"
	@${MPIRUN} -np 1 ./luke -use_ScAMG -ksp_gmres_cgs_refinement_type refine_always -nx 16 -ny 16 >> luke.out 2>&1;
	echo "1 proc"
	@${MPIRUN} -np 1 ./luke -use_ScAMG -ksp_gmres_cgs_refinement_type refine_always -nx 32 -ny 32 >> luke.out 2>&1;
	echo "1 proc"
	@${MPIRUN} -np 1 ./luke -use_ScAMG -ksp_gmres_cgs_refinement_type refine_always -nx 64 -ny 64 >> luke.out 2>&1;
#	echo "1 proc"
#	@${MPIRUN} -np 1 ./luke -use_ScAMG -ksp_gmres_cgs_refinement_type refine_always -nx 8 -ny 7 > luke.out 2>&1;
#	echo "2 proc"
#	@${MPIRUN} -np 2 ./luke -use_ScAMG -ksp_gmres_cgs_refinement_type refine_always -nx 8 -ny 7 >> luke.out 2>&1;
#	echo "4 proc"
#	@${MPIRUN} -np 4 ./luke -use_ScAMG -ksp_gmres_cgs_refinement_type refine_always -nx 8 -ny 7 >> luke.out 2>&1;
#	echo "8 proc"
#	@${MPIRUN} -np 8 ./luke -use_ScAMG -ksp_gmres_cgs_refinement_type refine_always -nx 8 -ny 7 >> luke.out 2>&1;
#	echo "16 proc"
#	@${MPIRUN} -np 16 ./luke -use_ScAMG -ksp_gmres_cgs_refinement_type refine_always -nx 8 -ny 7 >> luke.out 2>&1;
#	echo "32 proc"
#	@${MPIRUN} -np 32 ./luke -use_ScAMG -ksp_gmres_cgs_refinement_type refine_always -nx 8 -ny 7 >> luke.out 2>&1;
#	echo "64 proc"
#	@${MPIRUN} -np 64 ./luke -use_ScAMG -ksp_gmres_cgs_refinement_type refine_always -nx 8 -ny 7 >> luke.out 2>&1;
