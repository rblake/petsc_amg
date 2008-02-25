
#include "mglib.h"

#include <vector>
#include <set>
#include <map>


float
frand() {
    //Taken From
    //http://web.mit.edu/answers/c/c_random_numbers.html
    return (float) random() / (float) 0x7fffffff;
}

int
rank() {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    return rank;
}

int
size() {
    int size;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    return size;
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


void
build_strength_matrix(Mat A, PetscReal theta, Mat* strength) {
    
    //Variables for the new matrix structure
    std::vector<PetscInt> rows, cols;
    int cursor = 0;

    PetscInt start;						
    PetscInt end;						
    MatGetOwnershipRange((A), &start, &end);
    for (int row=start; row<end; row++) {
	PetscInt ncols;
	const PetscInt *col_indx;
	const PetscScalar *col_value;
	MatGetRow((A), row, &ncols, &col_indx, &col_value);
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
	MatRestoreRow((A), row, &ncols, &col_indx, &col_value);
    }         

    rows.push_back(cursor);

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

struct NonlocalCollection {
    NonlocalCollection(Mat depends_on, IS interest_set) {
	find_influences(depends_on, interest_set, &nodes);

	PetscInt local_size;
	ISGetLocalSize(nodes, &local_size);
	VecCreateMPI(PETSC_COMM_WORLD, local_size, PETSC_DECIDE, &vec);
	IS onto_index_set;
	describe_partition(vec, &onto_index_set);
	PetscInt begin;
	PetscInt end;
	VecGetOwnershipRange(vec, &begin, &end);
	PetscInt *indicies;
	ISGetIndices(nodes, &indicies);
	assert(local_size == end-begin);
	for (int ii=0; ii<local_size; ii++) {
	    map[indicies[ii]] = ii+begin;
	}
	ISRestoreIndices(nodes, &indicies);
	Vec w;
	MatGetVecs(depends_on, PETSC_NULL, &w);
	VecScatterCreate(w, nodes, vec, onto_index_set, &scatter);
	VecDestroy(w);

	ISDestroy(onto_index_set);
    }

    ~NonlocalCollection() {
	ISDestroy(nodes);
	VecDestroy(vec);
	VecScatterDestroy(scatter);
    }
    
    IS nodes;
    Vec vec;
    VecScatter scatter;
    std::map<PetscInt, PetscInt> map;
};

#define FOREACH(iter, coll) for(typeof((coll).begin()) iter=(coll).begin(); iter!=(coll).end(); ++iter)



void
cljp_coarsening(Mat depends_on, IS *pCoarse) {

    const int debug = 0;
        
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
    IS all_local_nodes;
    describe_partition(depends_on, &all_local_nodes);
    NonlocalCollection nonlocal(depends_on, all_local_nodes);
    ISDestroy(all_local_nodes);
    //while we are here, get the matrix + graph nodes that we need.
    Mat extended_depend_mat;
    get_matrix_rows(depends_on, nonlocal.nodes, &extended_depend_mat);

    // Vec used only for display purposes
    enum NodeType {UNKNOWN=-1, FINE, COARSE};
    Vec node_type;
    VecDuplicate(w, &node_type);
    VecSet(node_type, UNKNOWN);
    Vec w_nonlocal;
    VecDuplicate(nonlocal.vec, &w_nonlocal);
    Vec node_type_nonlocal;
    VecDuplicate(w_nonlocal, &node_type_nonlocal);
    VecSet(node_type_nonlocal, UNKNOWN);

    Vec is_not_independent;
    VecDuplicate(w, &is_not_independent);    
    Vec is_not_independent_nonlocal;
    VecDuplicate(w_nonlocal, &is_not_independent_nonlocal);
    VecScatterBegin(nonlocal.scatter, w, w_nonlocal, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(nonlocal.scatter, w, w_nonlocal, INSERT_VALUES, SCATTER_FORWARD);

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

	    if (debug) {
		LTRACE();
		char fname[] = "weightsXXX";
		char selection_graph[] = "selectionXXX";
		sprintf(fname, "weights%03d", inc);
		sprintf(selection_graph, "selection%03d", inc);
		inc++;
		
		/*
		PetscViewer view;
		PetscViewerBinaryMatlabOpen(PETSC_COMM_WORLD, fname, &view);
		PetscViewerBinaryMatlabOutputVecDA(view, "z", w, user->da);
		PetscViewerBinaryMatlabDestroy(view);
	    
		PetscViewerBinaryMatlabOpen(PETSC_COMM_WORLD, selection_graph, &view);
		PetscViewerBinaryMatlabOutputVecDA(view, "z", node_type, user->da);
		PetscViewerBinaryMatlabDestroy(view);
		//*/
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
		    PetscScalar weight_P = w_nonlocal_raw.at(nonlocal.map[*P]);

		    //for all dependencies K of P (K st P->K)
		    for (PetscInt ii=0; ii<dep_nonlocal_raw.nnz_in_row(nonlocal.map[*P]); ii++) {
			PetscInt K = dep_nonlocal_raw.col(nonlocal.map[*P], ii);
			//skip if K is fine/coarse
			/*
			  Notice that we don't have to consider the
			  independent set we've been generating here.  By
			  construction, if K is in the independent set, then P
			  cannot be in the independent set.
			*/
			if (node_type_nonlocal_raw.at(nonlocal.map[K]) != UNKNOWN) {
			    continue;
			}

			//skip if P->K is marked
			if (dep_nonlocal_raw.is_marked(nonlocal.map[*P], ii)) {
			    continue;
			}
		    
			//get weight(K)
			PetscScalar weight_K = w_nonlocal_raw.at(nonlocal.map[K]);

			if (weight_K <= weight_P) {
			    //is_not_independent(K) = true
			    is_not_independent_nonlocal_raw.at(nonlocal.map[K]) = 1;
			} else { // (weight(P) < weight_K)
			    is_not_independent_nonlocal_raw.at(nonlocal.map[*P]) = 1;
			}
		    }
		}
	    }

	    if (debug) {LTRACE();}
	    //VecView(is_not_independent_nonlocal, PETSC_VIEWER_STDOUT_WORLD);
	    
	    //reconstruct is_not_independent vector with a ADD_VALUES, which
	    //performs boolean OR
	    VecSet(is_not_independent, 0);
	    VecScatterBegin(nonlocal.scatter, is_not_independent_nonlocal, is_not_independent, ADD_VALUES, SCATTER_REVERSE);
	    VecScatterEnd(nonlocal.scatter, is_not_independent_nonlocal, is_not_independent, ADD_VALUES, SCATTER_REVERSE);
	    IntSet new_coarse_points;
	    {
		RawVector is_not_independent_raw(is_not_independent);
		//for all unknown points P
		FOREACH(P, unknown) {
		    //if (!is_not_independent(P))
		    if (is_not_independent_raw.at(*P) == 0) {
			new_coarse_points.insert(*P);
			if (debug) {SHOWVAR(*P, d);}
		    }
		}
	    }
	    //Post: new coarse points (independent set)
	    
	    if (debug) {LTRACE();}
	    
	    
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
	    
	    if (debug) {LTRACE();}
	    
	    //Pre: updated coarse local
	    //scatter changes to other processors
	    VecScatterBegin(nonlocal.scatter, node_type, node_type_nonlocal, INSERT_VALUES, SCATTER_FORWARD);
	    VecScatterEnd(nonlocal.scatter, node_type, node_type_nonlocal, INSERT_VALUES, SCATTER_FORWARD);
	    //Post: updated coarse non-local
	    
	    if (debug) {LTRACE();}
	    
	    //Pre: updated coarse non-local, new local coarse points
	    VecSet(w_update_nonlocal, 0);
	    {
		RawVector node_type_nonlocal_raw(node_type_nonlocal);
		RawVector w_update_nonlocal_raw(w_update_nonlocal);
		//for all new coarse points C
		FOREACH(C, new_coarse_points) {
		    //for all K st C->K
		    for(PetscInt ii=0; ii<dep_nonlocal_raw.nnz_in_row(nonlocal.map[*C]); ii++) {
			//mark (C->K)
			dep_nonlocal_raw.mark(nonlocal.map[*C], ii);
			PetscInt K = dep_nonlocal_raw.col(nonlocal.map[*C], ii);
			//if K is unknown
			if (node_type_nonlocal_raw.at(nonlocal.map[K]) == UNKNOWN) {
			    //measure(K)--
			    w_update_nonlocal_raw.at(nonlocal.map[K]) -= 1;
			}
		    }
		}
		
		//for all unknown points I
		FOREACH(I, unknown) {
		    IntSet common_coarse;
		    //for all (J->K)
		    for (PetscInt kk=0; kk<dep_nonlocal_raw.nnz_in_row(nonlocal.map[*I]); kk++) { 
			if (!dep_nonlocal_raw.is_marked(nonlocal.map[*I], kk)) {
			    //if K is coarse
			    PetscInt K = dep_nonlocal_raw.col(nonlocal.map[*I], kk);
			    if (node_type_nonlocal_raw.at(nonlocal.map[K]) == COARSE) {
				//mark K as common coarse
				common_coarse.insert(K);
				//mark (J->K) if unmarked
				dep_nonlocal_raw.mark(nonlocal.map[*I], kk);
			    }
			}
		    }

		    //for all unmarked (I->J)
		    for (PetscInt jj=0; jj<dep_nonlocal_raw.nnz_in_row(nonlocal.map[*I]); jj++) {
			if (!dep_nonlocal_raw.is_marked(nonlocal.map[*I], jj)) {
			    //for all (J->K), marked or no
			    PetscInt J = dep_nonlocal_raw.col(nonlocal.map[*I], jj);
			    for(PetscInt kk=0; kk<dep_nonlocal_raw.nnz_in_row(nonlocal.map[J]); kk++) {
				//if K is in layer or ghost layer and common-coarse
				PetscInt K = dep_nonlocal_raw.col(nonlocal.map[J], kk);
				if (is_member(K, common_coarse)) {
				    //mark (I->J)
				    dep_nonlocal_raw.mark(nonlocal.map[*I], jj);
				    //measure(J)--
				    w_update_nonlocal_raw.at(nonlocal.map[J]) -= 1;
				}
			    }
			}
		    }
		}
	    }
	    //Post: nonlocal update to local weights

	    if (debug) {LTRACE();}
	    
	    //Pre: local weights, update to local weights
	    VecScatterBegin(nonlocal.scatter, w_update_nonlocal, w, ADD_VALUES, SCATTER_REVERSE);
	    VecScatterEnd(nonlocal.scatter, w_update_nonlocal, w, ADD_VALUES, SCATTER_REVERSE);
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

	    if (debug) {LTRACE();}

	    VecScatterBegin(nonlocal.scatter, node_type, node_type_nonlocal, INSERT_VALUES, SCATTER_FORWARD);
	    VecScatterEnd(nonlocal.scatter, node_type, node_type_nonlocal, INSERT_VALUES, SCATTER_FORWARD);
	    VecScatterBegin(nonlocal.scatter, w, w_nonlocal, INSERT_VALUES, SCATTER_FORWARD);
	    VecScatterEnd(nonlocal.scatter, w, w_nonlocal, INSERT_VALUES, SCATTER_FORWARD);
	    //Post: non-local weights

	    //finally, determine if we should run the loop again.
	    //if one processor has a non-empty unknown set, then
	    //we need to rerun the loop.
	    int my_points_partitioned = unknown.empty();
	    MPI_Allreduce(&my_points_partitioned, &all_points_partitioned, 1, MPI_INT, MPI_LAND, PETSC_COMM_WORLD);
	    if (debug) {SHOWVAR(all_points_partitioned, d);}
	    //all_points_partitioned = true;
	}
    }

    //now, create an index set with all the coarse points.
    {
	RawVector node_type_raw(node_type);
	std::vector<PetscInt> coarse_points;
	for(int ii=node_type_raw.begin; ii<node_type_raw.end; ii++) {
	    if (node_type_raw.at(ii) == COARSE) {
		if (debug) {SHOWVAR(ii, d);}
		coarse_points.push_back(ii);
	    }
	}
    
	//if (*pCoarse != NULL) {
	//ISDestroy(*pCoarse);
	//}
	ISCreateGeneral(PETSC_COMM_WORLD, coarse_points.size(), &coarse_points[0], pCoarse);
    }

    MatDestroy(extended_depend_mat);
    VecDestroy(node_type);
    VecDestroy(node_type_nonlocal);
    VecDestroy(is_not_independent);    
    VecDestroy(is_not_independent_nonlocal);
    VecDestroy(w_update_nonlocal);
    VecDestroy(w_nonlocal);
    VecDestroy(w);
}

//////////////////////////////////////////////////////////////////////////////////

void get_compliment(Mat A, IS coarse, IS *pFine) {
    PetscInt begin, end;
    MatGetOwnershipRange(A, &begin, &end);

    PetscInt num_coarse;
    ISGetLocalSize(coarse, &num_coarse);
    PetscInt* coarse_points;
    ISGetIndices(coarse, &coarse_points);
    PetscInt cursor = 0;
    std::vector<PetscInt> fine_points;
    for (PetscInt ii=begin; ii<end; ii++) {
	if (cursor==num_coarse || ii < coarse_points[cursor]) {
	    fine_points.push_back(ii);
	    //SHOWVAR(ii, d);
	} else if ( ii == coarse_points[cursor] ) {
	    cursor++; 
	} else {
	    assert(0 && "code shouldn't get here!");
	}
    }
    assert(cursor == num_coarse);
    ISRestoreIndices(coarse, &coarse_points);

    ISCreateGeneral(PETSC_COMM_WORLD, fine_points.size(), &fine_points[0], pFine);
}


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
    NonlocalCollection nonlocal(A, interest_set);
    Vec coarse_marker;
    MatGetVecs(A, PETSC_NULL, &coarse_marker);
    VecSet(coarse_marker, 0);
    {
	RawVector marker_raw(coarse_marker);

	PetscInt tag_size;
	ISGetLocalSize(tag, &tag_size);
	PetscInt *tag_index;
	ISGetIndices(tag, &tag_index);
	for (int ii=0; ii<tag_size; ii++) {
	    marker_raw.at(tag_index[ii]) = 1;
	}
	ISRestoreIndices(tag, &tag_index);
    }
    VecScatterBegin(nonlocal.scatter, coarse_marker, nonlocal.vec, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(nonlocal.scatter, coarse_marker, nonlocal.vec, INSERT_VALUES, SCATTER_FORWARD);
    VecDestroy(coarse_marker);
    //Finally!  Collect all the terms in the nonlocal vector that have been marked.
    std::vector<PetscInt> depend_tag;
    {
	RawVector nonlocal_raw(nonlocal.vec);
	FOREACH(iter, nonlocal.map) {
	    //SHOWVAR(iter->first, d);
	    if (nonlocal_raw.at(iter->first) == 1) {
		//SHOWVAR(iter->second, d);
		depend_tag.push_back(iter->second); 
	    }
	}
    }
    //Now that we have the points, create the index set.
    ISCreateGeneral(PETSC_COMM_WORLD, depend_tag.size(), &depend_tag[0], pInfluences);

    //everything else destroyed by destructors.
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
