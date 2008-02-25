
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

#define CHKERR(code) { PetscErrorCode ierr##__LINE__ = (code); CHKERRQ(ierr##__LINE__); }

typedef struct {
    Mat A;
    Vec x;
    Vec b;
    PetscInt npoints;
    PetscInt num_seg;
    PetscInt levels;
    Mat* prolongation;
} Elliptic1D;


float frand();

void
construct_operator(Elliptic1D* p, int levels);

void
construct_prolongation_operator(PetscInt level, Mat* pmat);


int
rank();

int
size();

void
cljp_coarsening(Mat depends_on, IS *pCoarse);

void
build_strength_matrix(Mat A, PetscReal theta, Mat* strength);

void get_compliment(Mat A, IS coarse, IS *pFine);
