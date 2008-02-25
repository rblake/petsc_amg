
#include "petscvec.h"
#include "petscda.h"
#include "petscmat.h"

extern "C" {
#include "mglib.h"
}

#include <vector>
#include <set>
#include <map>



const char     *bcTypes[2] = {"dirichlet","neumann"};
typedef enum {DIRICHLET, NEUMANN} BCType;
typedef struct {
    PetscScalar   rho;
    PetscScalar   nu;
    BCType        bcType;
    DA da;
} UserContext;


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
get_matrix_rows(Mat A, IS rows, Mat *pA_rows) {
    //First, get all the matrix rows we are concerned with.
    PetscInt ncols;
    MatGetSize(A, PETSC_NULL, &ncols);
    IS all_columns;
    ISCreateStride(PETSC_COMM_WORLD, ncols, 0, 1, &all_columns);
    MatGetSubMatrix(A, rows, all_columns, PETSC_DECIDE, MAT_INITIAL_MATRIX, pA_rows);
    ISDestroy(all_columns);
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
    
    Mat subgraph_global;
    get_matrix_rows(graph, wanted, &subgraph_global);
    
    Mat subgraph;
    MatGetLocalMat(subgraph_global, MAT_INITIAL_MATRIX, &subgraph);
    MatDestroy(subgraph_global);

    PetscInt n;
    PetscInt *ia;
    PetscInt *ja;
    PetscTruth success = PETSC_FALSE;
    MatGetRowIJ(subgraph, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &success);
    assert(success == PETSC_TRUE);
    std::set<PetscInt> influences;
    for(int ii=ia[0]; ii<ia[n]; ii++) {
	influences.insert(ja[ii]);
    }
    success = PETSC_TRUE;
    MatRestoreRowIJ(subgraph, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &success);
    MatDestroy(subgraph);

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
    PetscScalar *data;
    PetscInt local_nrows;
    PetscInt row_begin;
    PetscInt row_end;
    Mat global_mat;
    Mat local_mat;

    RawGraph(Mat new_global_mat) {
	global_mat = new_global_mat;
	//For this code to work, we're going to need the matrix structure for the sequential portion of this matrix.
	//Therefore, extract a sequential AIJ matrix.
	MatType type;
	MatGetType(global_mat, &type);
	if (!strcmp(type,MATSEQAIJ)) {
	    MatDuplicate(global_mat, MAT_DO_NOT_COPY_VALUES, &local_mat);
	} else {
	    MatGetLocalMat(global_mat, MAT_INITIAL_MATRIX, &local_mat);
	}
	MatZeroEntries(local_mat);
	MatGetOwnershipRange(global_mat, &row_begin, &row_end);
	//MatView(local_mat, PETSC_VIEWER_DRAW_SELF);

	PetscTruth done = PETSC_FALSE;
	MatGetRowIJ(local_mat, 0, PETSC_FALSE, PETSC_FALSE, &local_nrows, &ia, &ja, &done);
	assert(done == PETSC_TRUE || "Unexpected error: can't get csr structure from matrix");
	MatGetArray(local_mat, &data);
    }
    
    ~RawGraph() {
	//clean up the local array structures I allocated.
	MatRestoreArray(local_mat, &data);
	PetscTruth done = PETSC_FALSE;
	MatRestoreRowIJ(local_mat, 0, PETSC_FALSE, PETSC_FALSE, &local_nrows, &ia, &ja, &done);
	assert(done == PETSC_TRUE || "Unexpected error: can't return csr structure to matrix");
	MatDestroy(local_mat);
    }

    PetscInt
    nnz_in_row(PetscInt row) {
	return ia[row-row_begin+1]-ia[row-row_begin];
    }

    PetscInt*
    row_pointer(PetscInt row) {
	return &(ja[ia[row-row_begin]]);
    }

    PetscInt
    col(PetscInt row, PetscInt row_index) {
	return row_pointer(row)[row_index];
    }

    void
    mark(PetscInt row, PetscInt row_index) {
	data[ ia[row-row_begin]+row_index ] = 1;
    }
    
    bool
    is_marked(PetscInt row, PetscInt row_index) {
	return (data[ ia[row-row_begin]+row_index ] == 1);
    }
	
};

struct RawVector {
    PetscInt begin;
    PetscInt end;
    PetscScalar* data;
    Vec vec;
    
    RawVector(Vec new_vec) {
	vec = new_vec;
	VecGetOwnershipRange(vec, &begin, &end);
	VecGetArray(vec, &data);
    }

    ~RawVector() {
	VecRestoreArray(vec, &data);
    }

    PetscScalar& at(PetscInt index) { return data[index-begin]; } 

};

template <typename T>
bool
is_member(T& element, std::set<T>& Set) {
    return (Set.find(element) != Set.end());
}

#define FOREACH(iter, coll) for(typeof((coll).begin()) iter=(coll).begin(); iter!=(coll).end(); ++iter)

void
cljp_coarsening(Mat depends_on, IS *pCoarse, UserContext* user) {
        
    //create a vector of the weights.
    Vec w;
    MatGetVecs(depends_on, PETSC_NULL, &w);
    VecZeroEntries(w);
 
    //Get my local matrix size
    PetscInt start;
    PetscInt end;
    MatGetOwnershipRange(depends_on, &start, &end);

    //TODO: replace with something that doesn't require re-creating the matrix structure.
    //Initialize all the weights
    {
	Mat influences;
	MatTranspose(depends_on, &influences);
	{
	    RawGraph influences_raw(influences);
	    assert(influences_raw.local_nrows == end-start);
	    //Initialize the weight vector with \norm{S^T_i} + \sigma(i)
	    PetscScalar *local_weights;
	    VecGetArray(w, &local_weights);
	    for (int local_row=0; local_row < influences_raw.local_nrows; local_row++) {
		local_weights[local_row] = 
		    influences_raw.ia[local_row+1]-influences_raw.ia[local_row] + frand();
	    }
	    VecRestoreArray(w, &local_weights);
	}
	MatDestroy(influences);
    }
    //VecView(w, PETSC_VIEWER_STDOUT_WORLD);

    //--------------------------------------------------------------

    //Prepare the scatters needed for the independent set algorithm.
    std::map<PetscInt, PetscInt> nonlocal_map;
    VecScatter nonlocal_scatter;
    Vec w_nonlocal;
    Mat extended_depend_mat;
    {
	IS nonlocal_nodes;
	describe_partition(depends_on, &nonlocal_nodes);
	find_influences(depends_on, nonlocal_nodes, &nonlocal_nodes);

	PetscInt local_size;
	ISGetLocalSize(nonlocal_nodes, &local_size);
	VecCreateMPI(PETSC_COMM_WORLD, local_size, PETSC_DECIDE, &w_nonlocal);
	IS onto_index_set;
	describe_partition(w_nonlocal, &onto_index_set);
	PetscInt begin;
	PetscInt end;
	VecGetOwnershipRange(w_nonlocal, &begin, &end);
	PetscInt *indicies;
	ISGetIndices(nonlocal_nodes, &indicies);
	assert(local_size == end-begin);
	for (int ii=0; ii<local_size; ii++) {
	    nonlocal_map[indicies[ii]] = ii+begin;
	}
	ISRestoreIndices(nonlocal_nodes, &indicies);
	VecScatterCreate(w, nonlocal_nodes, w_nonlocal, onto_index_set, &nonlocal_scatter);


	//while we are here, get the matrix + graph nodes that we need.
	get_matrix_rows(depends_on, nonlocal_nodes, &extended_depend_mat);

	ISDestroy(nonlocal_nodes);
	ISDestroy(onto_index_set);
    }

    // Vec used only for display purposes
    enum NodeType {UNKNOWN=-1, FINE, COARSE};
    Vec node_type;
    VecDuplicate(w, &node_type);
    VecSet(node_type, UNKNOWN);
    Vec node_type_nonlocal;
    VecDuplicate(w_nonlocal, &node_type_nonlocal);
    VecSet(node_type_nonlocal, UNKNOWN);

    Vec is_not_independent;
    VecDuplicate(w, &is_not_independent);    
    Vec is_not_independent_nonlocal;
    VecDuplicate(w_nonlocal, &is_not_independent_nonlocal);
    VecScatterBegin(nonlocal_scatter, w, w_nonlocal, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(nonlocal_scatter, w, w_nonlocal, INSERT_VALUES, SCATTER_FORWARD);

    Vec w_update_nonlocal;
    VecDuplicate(w_nonlocal, &w_update_nonlocal);

    //get ready to find all the coarse and fine points
    typedef std::set<PetscInt> IntSet;
    IntSet unknown;
    //initialize the unknown set with all points that are local to this processor.
    for (int ii=start; ii<end; ii++) { 
	unknown.insert(ii); 
    }    

    //we use MPI_INT here because we need to allreduce it with MPI_LAND 
    int all_points_partitioned=0;
    int inc = 0;
    
    {
	RawGraph dep_nonlocal_raw(extended_depend_mat);

	//while not done
	while(!all_points_partitioned) {
	    //Start: non-local weights, non-local coarse points

	    LTRACE();
	    {
		char fname[] = "weightsXXX";
		char selection_graph[] = "selectionXXX";
		sprintf(fname, "weights%03d", inc);
		sprintf(selection_graph, "selection%03d", inc);
		inc++;
	    
		PetscViewer view;
		PetscViewerBinaryMatlabOpen(PETSC_COMM_WORLD, fname, &view);
		PetscViewerBinaryMatlabOutputVecDA(view, "z", w, user->da);
		PetscViewerBinaryMatlabDestroy(view);
	    
		PetscViewerBinaryMatlabOpen(PETSC_COMM_WORLD, selection_graph, &view);
		PetscViewerBinaryMatlabOutputVecDA(view, "z", node_type, user->da);
		PetscViewerBinaryMatlabDestroy(view);
	    }


	    //Pre: non-local weights, non-local coarse points
	    //find the independent set.

	    //By using ADD_VALUES in a scattter, we can perform 
	    //a boolean OR across procesors.

	    //is_not_independent[*] = false
	    VecSet(is_not_independent_nonlocal, 0);
	    //for all unknown points P
	    {
		RawVector node_type_nonlocal_raw(node_type_nonlocal);
		RawVector w_nonlocal_raw(w_nonlocal);
		RawVector is_not_independent_nonlocal_raw(is_not_independent_nonlocal);
		FOREACH(P, unknown) {
		    //get weight(P)
		    PetscScalar weight_P = w_nonlocal_raw.at(nonlocal_map[*P]);

		    //for all dependencies K of P (K st P->K)
		    for (PetscInt ii=0; ii<dep_nonlocal_raw.nnz_in_row(nonlocal_map[*P]); ii++) {
			PetscInt K = dep_nonlocal_raw.col(nonlocal_map[*P], ii);
			//skip if K is fine/coarse
			/*
			  Notice that we don't have to consider the
			  independent set we've been generating here.  By
			  construction, if K is in the independent set, then P
			  cannot be in the independent set.
			*/
			if (node_type_nonlocal_raw.at(nonlocal_map[K]) != UNKNOWN) {
			    continue;
			}

			//skip if P->K is marked
			if (dep_nonlocal_raw.is_marked(nonlocal_map[*P], ii)) {
			    continue;
			}
		    
			//get weight(K)
			PetscScalar weight_K = w_nonlocal_raw.at(nonlocal_map[K]);

			if (weight_K <= weight_P) {
			    //is_not_independent(K) = true
			    is_not_independent_nonlocal_raw.at(nonlocal_map[K]) = 1;
			} else { // (weight(P) < weight_K)
			    is_not_independent_nonlocal_raw.at(nonlocal_map[*P]) = 1;
			}
		    }
		}
	    }

	    LTRACE();
	    //VecView(is_not_independent_nonlocal, PETSC_VIEWER_STDOUT_WORLD);

	    //reconstruct is_not_independent vector with a ADD_VALUES, which
	    //performs boolean OR
	    VecSet(is_not_independent, 0);
	    VecScatterBegin(nonlocal_scatter, is_not_independent_nonlocal, is_not_independent, ADD_VALUES, SCATTER_REVERSE);
	    VecScatterEnd(nonlocal_scatter, is_not_independent_nonlocal, is_not_independent, ADD_VALUES, SCATTER_REVERSE);
	    IntSet new_coarse_points;
	    {
		RawVector is_not_independent_raw(is_not_independent);
		//for all unknown points P
		FOREACH(P, unknown) {
		    //if (!is_not_independent(P))
		    if (is_not_independent_raw.at(*P) == 0) {
			new_coarse_points.insert(*P);
			SHOWVAR(*P, d);
		    }
		}
	    }
	    //Post: new coarse points (independent set)

	    LTRACE();


	    //Pre: independent set
	    {
		RawVector node_type_raw(node_type);
		// for each independent point
		FOREACH(I, new_coarse_points) {
		    //mark that point as coarse
		    node_type_raw.at(*I) = COARSE;
		    unknown.erase(*I);
		}
	    }
	    //Post: updated coarse local

	    LTRACE();

	    //Pre: updated coarse local
	    //scatter changes to other processors
	    VecScatterBegin(nonlocal_scatter, node_type, node_type_nonlocal, INSERT_VALUES, SCATTER_FORWARD);
	    VecScatterEnd(nonlocal_scatter, node_type, node_type_nonlocal, INSERT_VALUES, SCATTER_FORWARD);
	    //Post: updated coarse non-local

	    LTRACE();

	    //Pre: updated coarse non-local, new local coarse points
	    VecSet(w_update_nonlocal, 0);
	    {
		RawVector node_type_nonlocal_raw(node_type_nonlocal);
		RawVector w_update_nonlocal_raw(w_update_nonlocal);
		//for all new coarse points C
		FOREACH(C, new_coarse_points) {
		    //for all K st C->K
		    for(PetscInt ii=0; ii<dep_nonlocal_raw.nnz_in_row(nonlocal_map[*C]); ii++) {
			//mark (C->K)
			dep_nonlocal_raw.mark(nonlocal_map[*C], ii);
			PetscInt K = dep_nonlocal_raw.col(nonlocal_map[*C], ii);
			//if K is unknown
			if (node_type_nonlocal_raw.at(nonlocal_map[K]) == UNKNOWN) {
			    //measure(K)--
			    w_update_nonlocal_raw.at(nonlocal_map[K]) -= 1;
			}
		    }
		}

		//for all unknown points I
		FOREACH(I, unknown) {
		    IntSet common_coarse;
		    //for all (J->K)
		    for (PetscInt kk=0; kk<dep_nonlocal_raw.nnz_in_row(nonlocal_map[*I]); kk++) { 
			if (!dep_nonlocal_raw.is_marked(nonlocal_map[*I], kk)) {
			    //if K is coarse
			    PetscInt K = dep_nonlocal_raw.col(nonlocal_map[*I], kk);
			    if (node_type_nonlocal_raw.at(nonlocal_map[K]) == COARSE) {
				//mark K as common coarse
				common_coarse.insert(K);
				//mark (J->K) if unmarked
				dep_nonlocal_raw.mark(nonlocal_map[*I], kk);
			    }
			}
		    }

		    //for all unmarked (I->J)
		    for (PetscInt jj=0; jj<dep_nonlocal_raw.nnz_in_row(nonlocal_map[*I]); jj++) {
			if (!dep_nonlocal_raw.is_marked(nonlocal_map[*I], jj)) {
			    //for all (J->K), marked or no
			    PetscInt J = dep_nonlocal_raw.col(nonlocal_map[*I], jj);
			    for(PetscInt kk=0; kk<dep_nonlocal_raw.nnz_in_row(nonlocal_map[J]); kk++) {
				//if K is in layer or ghost layer and common-coarse
				PetscInt K = dep_nonlocal_raw.col(nonlocal_map[J], kk);
				if (is_member(K, common_coarse)) {
				    //mark (I->J)
				    dep_nonlocal_raw.mark(nonlocal_map[*I], jj);
				    //measure(J)--
				    w_update_nonlocal_raw.at(nonlocal_map[J]) -= 1;
				}
			    }
			}
		    }
		}
	    }
	    //Post: nonlocal update to local weights

	    LTRACE();
	    
	    //Pre: local weights, update to local weights
	    VecScatterBegin(nonlocal_scatter, w_update_nonlocal, w, ADD_VALUES, SCATTER_REVERSE);
	    VecScatterEnd(nonlocal_scatter, w_update_nonlocal, w, ADD_VALUES, SCATTER_REVERSE);
	    //Post: local weights updated

	    //VecView(w, PETSC_VIEWER_STDOUT_WORLD);
	    
	    //Pre: local weights, local node type
	    {
		RawVector w_raw(w);
		RawVector node_type_raw(node_type);
		IntSet new_fine_points;
		FOREACH(P, unknown) {
		    if (w_raw.at(*P) < 1) {
			w_raw.at(*P) = 0;
			new_fine_points.insert(*P);
			node_type_raw.at(*P) = FINE;
		    }
		}
		FOREACH(F, new_fine_points) {
		    unknown.erase(*F);
		}
		FOREACH(C, new_coarse_points) {
		    w_raw.at(*C) = 0;
		}
	    }
	    //Post: updated node type (with fine points), updated local weights

	    LTRACE();

	    VecScatterBegin(nonlocal_scatter, node_type, node_type_nonlocal, INSERT_VALUES, SCATTER_FORWARD);
	    VecScatterEnd(nonlocal_scatter, node_type, node_type_nonlocal, INSERT_VALUES, SCATTER_FORWARD);
	    VecScatterBegin(nonlocal_scatter, w, w_nonlocal, INSERT_VALUES, SCATTER_FORWARD);
	    VecScatterEnd(nonlocal_scatter, w, w_nonlocal, INSERT_VALUES, SCATTER_FORWARD);
	    //Post: non-local weights

	    //finally, determine if we should run the loop again.
	    //if one processor has a non-empty unknown set, then
	    //we need to rerun the loop.
	    int my_points_partitioned = unknown.empty();
	    MPI_Allreduce(&my_points_partitioned, &all_points_partitioned, 1, MPI_INT, MPI_LAND, PETSC_COMM_WORLD);
	    SHOWVAR(all_points_partitioned, d);
	    //all_points_partitioned = true;
	}
    }

    MatDestroy(extended_depend_mat);
    VecDestroy(node_type);
    VecDestroy(node_type_nonlocal);
    VecDestroy(is_not_independent);    
    VecDestroy(is_not_independent_nonlocal);
    VecDestroy(w_update_nonlocal);
    VecDestroy(w_nonlocal);
    VecScatterDestroy(nonlocal_scatter);
    VecDestroy(w);
}

/*
    while(0) {
	char fname[] = "weightsXXX";
	char selection_graph[] = "selectionXXX";
	sprintf(fname, "weights%03d", inc);
	sprintf(selection_graph, "selection%03d", inc);
	inc++;

	PetscViewer view;
	PetscViewerBinaryMatlabOpen(PETSC_COMM_WORLD, fname, &view);
	PetscViewerBinaryMatlabOutputVecDA(view, "z", w, user->da);
	PetscViewerBinaryMatlabDestroy(view);

	PetscViewerBinaryMatlabOpen(PETSC_COMM_WORLD, selection_graph, &view);
	PetscViewerBinaryMatlabOutputVecDA(view, "z", coarse_vs_fine, user->da);
	PetscViewerBinaryMatlabDestroy(view);

	//select an independent set of points D
	IntSet independent;
	/**
	   The original paper says that j is in the independent set
	   iff w(j) > w(k), where k is the set of nodes that either
	   depend on or influence j.  Therefore, use the neighbors
	   matrix.
	/
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
	    //add the point to the list of coarse points
	    unknown.erase(*iter);
	    coarse.push_back(*iter);
	    local_weights[*iter-start] = 0;
	    VecSetValue(coarse_vs_fine, *iter, COARSE, INSERT_VALUES);
	}

	VecAssemblyBegin(coarse_vs_fine);
	VecAssemblyEnd(coarse_vs_fine);

	////////////////////////////////////////////////////////

	//for each item in the independent set
	FOREACH(iter, independent) {
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
	    /
	    // for each point P that influences c
	    for (int col_index = influences_raw.ia[*iter-start];
		 col_index < influences_raw.ia[*iter-start+1];
		 col_index++
		 ) {
		PetscInt P = influences_raw.ja[col_index];
		//decrement the measure.
		if ((start <= P) && (P < end)) {
		    //if the point is local, update the array copy
		    local_weights[P-start]--;
		} else {
		    //if the point is non-local, use VecSetValue.
		    VecSetValue(w, P, -1, ADD_VALUES);
		}
	    }
	    // for each Q that depends on c
	    for (int Q_index = depends_raw.ia[*iter-start];
		 Q_index < depends_raw.ia[*iter-start+1];
		 Q_index++
		 ) {
		PetscInt Q = depends_raw.ja[Q_index];
		//for each R that depends on Q
		for (int R_index = extended_rowbound[extended_depend_map[Q]];
		     R_index < extended_rowbound[extended_depend_map[Q]+1];
		     R_index++
		     ) {
		    PetscInt R = extended_col_index[R_index];
		    //if R depends on c, decrement the measure at Q.
		    bool R_depends_on_c = false;
		    for (int influence_index = influences_raw.ia[*iter-start];
			 influence_index < influences_raw.ia[*iter-start+1];
			 influence_index++
			 ) {
			PetscInt c_influence = influences_raw.ja[influence_index];
			if (c_influence == R) {
			    R_depends_on_c = true;
			    break;
			}
		    }
		    if (R_depends_on_c) {
			if ((start <= Q) && (Q < end)) {
			    //if the point is local, update the array copy
			    local_weights[Q-start]--;
			} else {
			    //if the point is non-local, use VecSetValue.
			    VecSetValue(w, Q, -1, ADD_VALUES);
			}
		    }
		}
	    }
	}
	VecRestoreArray(w_needed, &local_w_needed);
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
	    SHOWVAR((*iter), d);
	}

	//finally, determine if we should run the loop again.
	//if one processor has a non-empty unknown set, then
	//we need to rerun the loop.
	int my_points_partitioned = unknown.empty();
	MPI_Allreduce(&my_points_partitioned, &all_points_partitioned, 1, MPI_INT, MPI_LAND, PETSC_COMM_WORLD);
	//SHOWVAR(all_points_partitioned, d);
	//all_points_partitioned = true;
    }
//*/

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
  cljp_coarsening(requests_from, &coarse, &user);
  
  MatDestroy(provides_to);
  MatDestroy(requests_from);

  ierr = VecDestroy(b);
  ierr = MatDestroy(A);
  ierr = DADestroy(user.da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}


