
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


void
find_influences
/////////////////////////////////////////////////////////////
/** 
 */
(
 Mat graph,
 //graph of non-zero structure
 IS wanted,
 //nodes we are interested in.  Contains different values on each processor.
 IS *is_influences
 //all the influences of the nodes we are interested in.
 );

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
 );


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
 );
