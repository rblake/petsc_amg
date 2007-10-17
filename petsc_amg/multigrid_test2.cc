
#include "petscvec.h"
#include "petscda.h"
#include "petscmat.h"

extern "C" {
#include "mglib.h"
}

#include <vector>
#include <set>
#include <map>

void
construct_amg_prolongation
(
 /// index map of local to higher points
 IS coarse,
 IS fine,
 IS depend_coarse,
 IS depend_strong,
 IS depend_weak,
 Mat A,
 Mat* pmat
 ) {

    // Start constructing the local submatricies needed.
    enum submatrix {Fp_Dpc, Fp_Dps, Dps_Dpc, num_submatrix};
    Mat* local_submatrix;
    IS irow[num_submatrix];
    IS icol[num_submatrix];
    
    irow[Fp_Dpc] = fine;
    icol[Fp_Dpc] = depend_coarse;

    irow[Fp_Dps] = fine;
    icol[Fp_Dps] = depend_strong;

    irow[Dps_Dpc] = depend_strong;
    icol[Dps_Dpc] = depend_coarse;

    MatGetSubMatrices(A, num_submatrix, irow, icol, MAT_INITIAL_MATRIX, &local_submatrix);
    
    // Note, due to RS C1, we know that the Fp_Dpc matrix has 
    // the same non-zero pattern as the Fp_Dps x Dps_Dpc matrix
    
    Vec row_sum;
    MatGetRowSum(local_submatrix[Dps_Dpc], row_sum);
    VecReciprocal(row_sum);
    MatDiagonalScale(local_submatrix[Dps_Dpc], row_sum, PETSC_NULL);
    
    Mat result;
    MatMatMult(local_submatrix[Fp_Dps], local_submatrix[Dps_Dpc], 
	       MAT_INITIAL_MATRIX, 4, &result);
    
    //Because of the special form of result, we should be able to optimize this
    //addition.  Both matricies should have the same non-zero structure.
    
    //TODO: add a check here to verify that assertion.

    MatAYPX(result, 1, local_submatrix[Fp_Dpc], SAME_NONZERO_PATTERN);
    
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


#define FORALL_ROWS(A) {					\
    PetscInt start;						\
    PetscInt end;						\
    MatGetOwnershipRange((A), &start, &end);			\
    for (int row=start; row<end; row++) {			\
	PetscInt ncols;						\
	const PetscInt *col_indx;				\
	const PetscScalar *col_value;				\
	MatGetRow((A), row, &ncols, &col_indx, &col_value);


#define END_FORALL_ROWS(A)					\
        MatRestoreRow((A), row, &ncols, &col_indx, &col_value);	\
    }								\
}

void
build_strength_matrix(Mat A, PetscReal theta, Mat* strength) {
    
    //Variables for the new matrix structure
    std::vector<PetscInt> rows, cols;
    int cursor = 0;

    FORALL_ROWS(A) {
	rows.push_back(cursor);
	// First, find the threshhold for this row
	PetscScalar strong_threshhold = -col_value[0];
	for (int ii=0; ii<ncols; ii++) {
	    if (-col_value[ii] > strong_threshhold) {
		strong_threshhold = -col_value[ii];
	    }
	}
	strong_threshhold *= theta;

	//if the threshold is negative, assume that this row only has a diagonal entry and skip the row
	if (strong_threshhold > 0) {
	    for (int ii=0; ii<ncols; ii++) {
		if (-col_value[ii] >= strong_threshhold) {
		    cols.push_back(col_indx[ii]);
		    cursor++;
		}
	    }
	}
    } END_FORALL_ROWS(A);
    rows.push_back(cursor);

    PetscInt start;
    PetscInt end;
    MatGetOwnershipRange((A), &start, &end);

    std::vector<PetscScalar> data(cols.size());
    //TODO: control for cases where cols and rows are split differently
    //TODO: replace this PETSC_COMM_WORLD so the strength matrix is using the same communicator as the original matrix.
    MatCreate(PETSC_COMM_WORLD, strength);
    MatSetSizes(*strength, end-start, end-start, PETSC_DETERMINE, PETSC_DETERMINE);
    MatSetType(*strength,MATMPIAIJ);
    MatMPIAIJSetPreallocationCSR(*strength,&rows[0],&cols[0],&data[0]);

    //TODO: the above code is a work around for a bug in the following function call:
    //MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, end-start, end-start, PETSC_DETERMINE, PETSC_DETERMINE, &rows[0], &cols[0], &data[0], strength);
}

void
describe_partition(Mat A, IS *part) {
    PetscInt start;
    PetscInt end;
    MatGetOwnershipRange(A, &start, &end);
    ISCreateStride(PETSC_COMM_WORLD, end-start, start, 1, part);
}

void
describe_partition(Vec v, IS *part) {
    PetscInt start;
    PetscInt end;
    VecGetOwnershipRange(v, &start, &end);
    ISCreateStride(PETSC_COMM_WORLD, end-start, start, 1, part);
}


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
 ) {
    
    //First, get all the matrix rows we are concerned with.
    Mat *subgraph;
    PetscInt ncols;
    MatGetSize(graph, PETSC_NULL, &ncols);
    IS all_columns;
    ISCreateStride(PETSC_COMM_WORLD, ncols, 0, 1, &all_columns);
    MatGetSubMatrices(graph, 1, &wanted, &all_columns, MAT_INITIAL_MATRIX, &subgraph);
    ISDestroy(all_columns);

    PetscInt n;
    PetscInt *ia;
    PetscInt *ja;
    PetscTruth success = PETSC_FALSE;
    MatGetRowIJ(*subgraph, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &success);
    assert(success == PETSC_TRUE);
    std::set<PetscInt> influences;
    for(int ii=ia[0]; ii<ia[n]; ii++) {
	influences.insert(ja[ii]);
    }
    success = PETSC_TRUE;
    MatRestoreRowIJ(*subgraph, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &success);
    MatDestroy(*subgraph);

    std::vector<PetscInt> all_influences;
    std::copy(influences.begin(), influences.end(), std::back_inserter(all_influences));
    
    ISCreateGeneral(PETSC_COMM_WORLD, all_influences.size(), &all_influences[0], is_influences);
}


/** Gets the graph structure in CSR format for the local rows of a global matrix
    It automatically allocates all the information it needs in construction and deallocates all
    the memory during deconstruction.  Yay RAII.
*/
struct RawGraph {
    PetscInt *ia;
    PetscInt *ja;
    PetscInt local_nrows;
    Mat global_mat;
    Mat local_mat;

    RawGraph(Mat new_global_mat) {
	global_mat = new_global_mat;
	//For this code to work, we're going to need the matrix structure for the sequential portion of this matrix.
	//Therefore, extract a sequential AIJ matrix.
	MatGetLocalMat(global_mat, MAT_INITIAL_MATRIX, &local_mat);
	//MatView(local_mat, PETSC_VIEWER_DRAW_SELF);

	PetscTruth done = PETSC_FALSE;
	MatGetRowIJ(local_mat, 0, PETSC_FALSE, PETSC_FALSE, &local_nrows, &ia, &ja, &done);
	assert(done == PETSC_TRUE || "Unexpected error: can't get csr structure from matrix");
    }
    
    ~RawGraph() {
	//clean up the local array structures I allocated.
	PetscTruth done = PETSC_FALSE;
	MatRestoreRowIJ(local_mat, 0, PETSC_FALSE, PETSC_FALSE, &local_nrows, &ia, &ja, &done);
	assert(done == PETSC_TRUE || "Unexpected error: can't return csr structure to matrix");
	MatDestroy(local_mat);
    }
};

#define FOREACH(iter, coll) for(typeof(unknown.begin()) iter=(coll).begin(); iter!=(coll).end(); ++iter)

void
cljp_coarsening(Mat influences, Mat depends_on, IS *pCoarse) {
        
    //create a vector of the weights.
    Vec w;
    MatGetVecs(influences, PETSC_NULL, &w);
    VecZeroEntries(w);

    RawGraph influences_raw(influences);
 
    //Get my local matrix size
    PetscInt start;
    PetscInt end;
    MatGetOwnershipRange(influences, &start, &end);
    assert(influences_raw.local_nrows == end-start);

    //Initialize the weight vector with \norm{S^T_i} + \sigma(i)
    PetscScalar *local_weights;
    VecGetArray(w, &local_weights);
    for (int local_row=0; local_row < influences_raw.local_nrows; local_row++) {
	local_weights[local_row] = 
	    influences_raw.ia[local_row+1]-influences_raw.ia[local_row] + frand();
    }
    VecRestoreArray(w, &local_weights);

    VecView(w, PETSC_VIEWER_DRAW_WORLD);

    //Next, we need to create a matrix that represents all the neighbors of
    //a given node.
    Mat neighbors;
    //TODO: I need to get rid of the values on all these matricies.
    //This should be MAT_DO_NOT_COPY_VALUES
    MatDuplicate(influences, MAT_COPY_VALUES, &neighbors);
    MatAXPY(neighbors, 1, depends_on, DIFFERENT_NONZERO_PATTERN);
    //MatView(neighbors, PETSC_VIEWER_DRAW_WORLD);

    RawGraph neighbors_raw(neighbors);
    
    //get ready to find all the coarse and fine points
    std::vector<PetscInt> coarse;
    std::vector<PetscInt> fine;
    typedef std::set<PetscInt> IntSet;
    IntSet unknown;
    //initialize the unknown set with all points that are local to this processor.
    for (int ii=start; ii<end; ii++) { 
	unknown.insert(ii); 
    }

    //Prepare the scatters needed for the independent set algorithm.
    std::map<PetscInt, PetscInt> needed_map;
    VecScatter needed_scatter;
    Vec w_needed;
    {
	IS needed_nodes;
	describe_partition(neighbors, &needed_nodes);
	find_influences(neighbors, needed_nodes, &needed_nodes);

	PetscInt local_size;
	ISGetLocalSize(needed_nodes, &local_size);
	VecCreateMPI(PETSC_COMM_WORLD, local_size, PETSC_DECIDE, &w_needed);
	IS onto_index_set;
	describe_partition(w_needed, &onto_index_set);
	PetscInt begin;
	PetscInt end;
	VecGetOwnershipRange(w_needed, &begin, &end);
	PetscInt *indicies;
	ISGetIndices(needed_nodes, &indicies);
	assert(local_size == end-begin);
	for (int ii=0; ii<local_size; ii++) {
	    needed_map[indicies[ii]] = ii;
	}
	ISRestoreIndices(needed_nodes, &indicies);
	VecScatterCreate(w, needed_nodes, w_needed, onto_index_set, &needed_scatter);
	ISDestroy(needed_nodes);
	ISDestroy(onto_index_set);
    }

    //we use MPI_INT here because we need to allreduce it with MPI_LAND 
    int all_points_partitioned=0;
    //while C U F != all points
    while(!all_points_partitioned) {
	//select an independent set of points D
	IntSet independent;
	/**
	   The original paper says that j is in the independent set
	   iff w(j) > w(k), where k is the set of nodes that either
	   depend on or influence j.  Therefore, use the neighbors
	   matrix.
	*/
	//get weights from neighbors
	VecScatterBegin(needed_scatter, w, w_needed, INSERT_VALUES, SCATTER_FORWARD);
	VecScatterEnd(needed_scatter, w, w_needed, INSERT_VALUES, SCATTER_FORWARD);
	
	PetscScalar *local_w_needed;
	VecGetArray(w, &local_weights);
	VecGetArray(w_needed, &local_w_needed);
	FOREACH(iter, unknown) {
	    //compare this node's weight to all of it's neighbors
	    PetscScalar w_j = local_weights[*iter-start];
	    bool bigger=true;
	    for (PetscInt col_index=neighbors_raw.ia[*iter-start];
		 col_index<neighbors_raw.ia[*iter-start+1];
		 col_index++
		 ) {

		PetscInt column = neighbors_raw.ja[col_index];
		PetscScalar w_k = local_w_needed[needed_map[column]];
		if (w_j <= w_k) {
		    bigger=false;
		    break;
		}
	    }
	    //if this node is bigger than all it's neighbors, add it to the independent set.
	    if (bigger) {
		independent.insert(*iter);
	    }
	}

	FOREACH(iter, independent) {
	    SHOWVAR(*iter, d);
	}

#if 0
	////////////////////////////////////////////////////////

	//for each item in the independent set
	FOREACH(iter, independent) {
	    //add the point to the list of coarse points
	    unknown.erase(*iter);
	    coarse.push_back(*iter);
	    //update the weights
	    /**
	       Notice that I'm moving the check for fine points outside
	       of this loop.  This encurs more overhead, but is easier
	       to implement in parallel.

	       Unfortunately, this also can create havok with rows
	       with no strong connections.  Obviously, if a node
	       influences nothing or is influenced by nothing, then it
	       needs to be a coarse point.  The algorithm listed in
	       the paper catches this case because it only updates the
	       points in the neighborhood of the new coarse points.
	       However, in my case getting the list of weight updates
	       across processor boundaries is difficult and tedious.
	    */
	    // for each point P that influences c
	    for (int col_index = influences_raw.ia[*iter-start];
		 col_index < influences_raw.ia[*iter-start+1];
		 col_index++
		 ) {
		PetscInt P = influences_raw.ja[col_index];
		//decrement the measure.
		if ((start <= j) && (j < end)) {
		    //if the point is local, update the array copy
		    local_weights[P-start]--;
		} else {
		    //if the point is non-local, use VecSetValue.
		    VecSetValue(w, P, -1, ADD_VALUES);
		}
	    }
	    // for each Q that depends on c
	    for (int col_index = depends_on_raw.ia[*iter-start];
		 col_index < depends_on_raw.ia[*iter-start+1];
		 col_index++
		 ) {
		PetscInt Q = influences_raw.ja[col_index];
		//decrement the measure
		if ((start <= j) && (j < end)) {
		    //if the point is local, update the array copy
		    local_weights[Q-start]--;
		} else {
		    //if the point is non-local, use VecSetValue.
		    VecSetValue(w, Q, -1, ADD_VALUES);
		}
		
	    }    

	    for (int col_index = neighbors_raw.ia[*iter-start];
		 col_index < neighbors_raw.ia[*iter-start+1];
		 col_index++
		 ) {
		PetscInt column = neighbors_raw.ja[col_index];
	    }
	}
	VecRestoreArray(w_nonlocal, &nonlocal_weights);
	VecRestoreArray(w, &local_weights);
	
	//make sure any neighbor updates have propagated.
	VecAssemblyBegin(w);
	VecAssemblyEnd(w);

	//check for fine points.
	IntSet new_fine_points;
	VecGetArray(w, &local_weights);
	//for all the remaining points
	FOREACH(iter, unknown) {
	    //if weight is too low, add it to the list of fine points
	    if (local_weights[*iter-start] < 1) {
		fine.push_back(*iter);
		new_fine_points.insert(*iter);
	    }
	}
	VecRestoreArray(w, &local_weights);
	//cannot remove elements from unknown while we are iterating through it.
	FOREACH(iter, new_fine_points) {
	    unknown.erase(*iter);
	}
#endif
	//finally, determine if we should run the loop again.
	//if one processor has a non-empty unknown set, then
	//we need to rerun the loop.
	int my_points_partitioned = unknown.empty();
	MPI_Allreduce(&my_points_partitioned, &all_points_partitioned, 1, MPI_INT, MPI_LAND, PETSC_COMM_WORLD);
	SHOWVAR(all_points_partitioned, d);
	all_points_partitioned = true;
    }

    VecDestroy(w_needed);
    VecScatterDestroy(needed_scatter);
    VecDestroy(w);

    //clean up the neighbors matrix
    MatDestroy(neighbors);


}

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

const char     *bcTypes[2] = {"dirichlet","neumann"};
typedef enum {DIRICHLET, NEUMANN} BCType;
typedef struct {
    PetscScalar   rho;
    PetscScalar   nu;
    BCType        bcType;
    DA da;
} UserContext;


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
  PetscScalar    v[5],Hx,Hy,HydHx,HxdHy,rho;
  MatStencil     row, col[5];

  DAGetMatrix(da, MATMPIAIJ, pA);

  ierr = DAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);  
  Hx    = 1.0 / (PetscReal)(mx-1);
  Hy    = 1.0 / (PetscReal)(my-1);
  HxdHy = Hx/Hy;
  HydHx = Hy/Hx;
  ierr = DAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++){
    for(i=xs; i<xs+xm; i++){
      row.i = i; row.j = j;
      ierr = ComputeRho(i, j, mx, my, centerRho, &rho);CHKERRQ(ierr);
      if (i==0 || j==0 || i==mx-1 || j==my-1) {
        if (user->bcType == DIRICHLET) {
           v[0] = 2.0*rho*(HxdHy + HydHx);
          ierr = MatSetValuesStencil(*pA,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
        } else if (user->bcType == NEUMANN) {
          num = 0;
          if (j!=0) {
            v[num] = -rho*HxdHy;              col[num].i = i;   col[num].j = j-1;
            num++;
          }
          if (i!=0) {
            v[num] = -rho*HydHx;              col[num].i = i-1; col[num].j = j;
            num++;
          }
          if (i!=mx-1) {
            v[num] = -rho*HydHx;              col[num].i = i+1; col[num].j = j;
            num++;
          }
          if (j!=my-1) {
            v[num] = -rho*HxdHy;              col[num].i = i;   col[num].j = j+1;
            num++;
          }
          v[num]   = (num/2.0)*rho*(HxdHy + HydHx); col[num].i = i;   col[num].j = j;
          num++;
          ierr = MatSetValuesStencil(*pA,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else {
        v[0] = -rho*HxdHy;              col[0].i = i;   col[0].j = j-1;
        v[1] = -rho*HydHx;              col[1].i = i-1; col[1].j = j;
        v[2] = 2.0*rho*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
        v[3] = -rho*HydHx;              col[3].i = i+1; col[3].j = j;
        v[4] = -rho*HxdHy;              col[4].i = i;   col[4].j = j+1;
        ierr = MatSetValuesStencil(*pA,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
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

  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,-3,-3,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&user.da);CHKERRQ(ierr);  

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
  cljp_coarsening(provides_to, requests_from, &coarse);
  
  MatDestroy(provides_to);
  MatDestroy(requests_from);

  ierr = VecDestroy(b);
  ierr = MatDestroy(A);
  ierr = DADestroy(user.da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}


