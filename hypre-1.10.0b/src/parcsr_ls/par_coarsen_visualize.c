#include "headers.h"
#include "par_amg.h"
#include "HYPRE.h"

/** @file par_coarsen_visualize.c
 * This file contains routines used to write information about the coarse
 * grid selection to disk. The files can then be used by external programs
 * to visualize the results.
 *
 * @author David Alber
 * @date January 2005
 */

#define F_POINT -1
#define C_POINT 1

#define NO_WRITE_FILES 0

void WriteCRRates(hypre_ParCSRMatrix * A, double * candidate_measures, int level)
{
  int i, j, my_id, size, row_start;
  int * diag_i, * diag_j, * offd_i, * offd_j, * offd_map;
  FILE * out;
  hypre_CSRMatrix * diag, * offd;
  char filename[100];

  if(NO_WRITE_FILES)
    return;

  MPI_Comm comm = hypre_ParCSRMatrixComm(A); 

  MPI_Comm_rank(comm,&my_id);

  size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

  sprintf(filename, "cr-rates.out.%i.%i", my_id, level);
  out = fopen(filename, "w");

  size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
  row_start = hypre_ParCSRMatrixRowStarts(A)[my_id];

  for(i = 0; i < size; i++) {
    // for each row owned by this process
    
    // Print this row's global number.
    fprintf(out, "%i %lf ", row_start+i, candidate_measures[i]);

    fprintf(out, "\n");
  }
  fclose(out);
}

void writeCRRatesE(void *amg_vdata, int num_levels)
{
  hypre_ParAMGData * amg_data = amg_vdata;

  hypre_ParCSRMatrix ** A_array = hypre_ParAMGDataAArray(amg_data);
  int ** CF_marker_array = hypre_ParAMGDataCFMarkerArray(amg_data);

  if(NO_WRITE_FILES)
    return;

  MPI_Comm comm = hypre_ParCSRMatrixComm(A_array[0]);
  int my_id;
  MPI_Comm_rank(comm,&my_id);

  int * grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);
  int ** grid_relax_points = hypre_ParAMGDataGridRelaxPoints(amg_data);
  int relax_order = hypre_ParAMGDataRelaxOrder(amg_data);
  double * relax_weight = hypre_ParAMGDataRelaxWeight(amg_data); 
  double * omega = hypre_ParAMGDataOmega(amg_data); 
  int cycle_param = 0;
  int relax_type = grid_relax_type[cycle_param];

  HYPRE_IJVector ij_x, ij_b, ij_x_iter_4;
  void * object;
  hypre_ParVector *x, *b, *x_iter_4;
  hypre_ParVector * Vtemp = hypre_ParAMGDataVtemp(amg_data);
  hypre_Vector * local_x, * local_x_iter_4;
  double * local_x_data, * local_x_iter_4_data;

  int i, level, ierr, Solve_err_flag;
  int first_local_row, last_local_row, local_num_rows;
  int first_local_col, last_local_col, local_num_cols;
  int global_num_rows;
  double * values, * measures;
  double e_f_prev, e_f_curr, rho_cr;

  for(level = 0; level < num_levels-1; level++) {
    ierr = HYPRE_ParCSRMatrixGetLocalRange((HYPRE_ParCSRMatrix)(A_array[level]),
					   &first_local_row,
					   &last_local_row, &first_local_col,
					   &last_local_col);
    local_num_rows = last_local_row - first_local_row + 1;
    local_num_cols = last_local_col - first_local_col + 1;
    global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);

    // Relaxing Ax = b.

    ///////////////////////////////////////////////////////////////////////
    // Set right hand side to all zeros.                                 //
    //                                                                   //
    HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_b);
    HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_b);
   
    values = hypre_CTAlloc(double, local_num_cols);
    for (i = 0; i < local_num_cols; i++)
      values[i] = 0;

    HYPRE_IJVectorSetValues(ij_b, local_num_cols, NULL, values);
    hypre_TFree(values);

    ierr = HYPRE_IJVectorGetObject(ij_b, &object);
    b = (hypre_ParVector *)object;
    //                                                                   //
    ///////////////////////////////////////////////////////////////////////

    HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x_iter_4);
    HYPRE_IJVectorSetObjectType(ij_x_iter_4, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_x_iter_4);
    
    ierr = HYPRE_IJVectorGetObject(ij_x_iter_4, &object);
    x_iter_4 = (hypre_ParVector *)object;

    ///////////////////////////////////////////////////////////////////////
    // Set initial guess for e to 1 + rand[0, 1/2] for all entries in e. //
    //                                                                   //
    hypre_SeedRand(34902+my_id);

    HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x);
    HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_x);
   
    values = hypre_CTAlloc(double, local_num_cols);
    for (i = 0; i < local_num_cols; i++)
      values[i] = 1 + hypre_Rand()/4;

    HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
    hypre_TFree(values);

    ierr = HYPRE_IJVectorGetObject(ij_x, &object);
    x = (hypre_ParVector *)object;
    //                                                                   //
    ///////////////////////////////////////////////////////////////////////

    // Set left hand side to zero for all of the C-points.
    local_x = hypre_ParVectorLocalVector(x);
    local_x_data = hypre_VectorData(local_x);
    for(i = 0; i < local_num_rows; i++) {
      if(CF_marker_array[level][i] == C_POINT)
	local_x_data[i] = 0;
    }

    /////////////////////////////////////////////////////////////////////
    // Perform 5 CR iterations (relaxation on the F-points only with   //
    // C-point values set to zero). Compute rho_cr.                    //
    //                                                                 //
    for(i = 0; i < 5; i++) {
      Solve_err_flag = hypre_BoomerAMGRelax(A_array[level], b, CF_marker_array[level],
					    relax_type, F_POINT,
					    relax_weight[level], omega[level], x,
					    Vtemp);
      if(i == 3) {
	// Compute norm of e_f for fourth iteration. e_f is the norm of
	// only the fine level nodes, but since the C-points have a value
	// of zero, a normal inner product will give the same results.
	// However, this should be written to utilize this information
	// in the future.
	hypre_ParVectorCopy(x, x_iter_4);
	e_f_prev = sqrt(hypre_ParVectorInnerProd(x, x));
      }
      e_f_curr = sqrt(hypre_ParVectorInnerProd(x, x));

      rho_cr = e_f_curr / e_f_prev;
    }
    //                                                                 //
    /////////////////////////////////////////////////////////////////////

    // Write candidate measures for visualization.
    local_x = hypre_ParVectorLocalVector(x);
    local_x_data = hypre_VectorData(local_x);
    local_x_iter_4 =  hypre_ParVectorLocalVector(x_iter_4);
    local_x_iter_4_data = hypre_VectorData(local_x_iter_4);
    measures = hypre_CTAlloc(double, local_num_cols);
    for(i = 0; i < local_num_cols; i++) {
      if(local_x_iter_4_data[i] == 0)
	measures[i] = 0;
      else
	measures[i] = fabs(local_x_data[i] / local_x_iter_4_data[i]);
    }

    WriteCRRates(A_array[level], measures, level);

    hypre_TFree(measures);

    if(my_id == 0)
      printf("CR Rate (level %d): %f\n", level, rho_cr);
  }
}

/** WriteCoarseningColorInformation takes the node color information generated by the coarsening routine and writes it to file. The file can later be used by a graphviz based visualization program which will show the graph with the nodes colored. This is useful to make sure that the coloring algorithm is working well.
 * @param A a hypre_ParCSRMatrix object containing the linear system.
 * @param color_array an array of shorts containing color of each node.
 */
void WriteCoarseningColorInformation(hypre_ParCSRMatrix * A, short * color_array, int level)
{
  int i, j, my_id, size, row_start;
  int * diag_i, * diag_j, * offd_i, * offd_j, * offd_map;
  FILE * out;
  hypre_CSRMatrix * diag, * offd;
  char filename[100];

  if(NO_WRITE_FILES)
    return;

  MPI_Comm comm = hypre_ParCSRMatrixComm(A); 

  MPI_Comm_rank(comm,&my_id);

  size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

  sprintf(filename, "color.out.%i.%i", my_id, level);
  out = fopen(filename, "w");

  size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
  row_start = hypre_ParCSRMatrixRowStarts(A)[my_id];

  for(i = 0; i < size; i++) {
    // for each row owned by this process
    
    // Print this row's global number.
    fprintf(out, "%i %i ", row_start+i, color_array[i]);

    fprintf(out, "\n");
  }
  //fprintf(out, "-1\n");
  fclose(out);
}

void WriteCoarseGridEffectivenessInformation(void * amg_vdata, int num_levels, int num_procs)
{
  hypre_ParAMGData * amg_data = amg_vdata;

  hypre_ParCSRMatrix ** A_array = hypre_ParAMGDataAArray(amg_data);
  hypre_ParCSRMatrix ** P_array = hypre_ParAMGDataPArray(amg_data);
  hypre_ParCSRMatrix ** R_array = hypre_ParAMGDataRArray(amg_data);
  int ** CF_marker_array = hypre_ParAMGDataCFMarkerArray(amg_data);

  if(NO_WRITE_FILES)
    return;

  MPI_Comm comm = hypre_ParCSRMatrixComm(A_array[0]);
  int my_id;
  MPI_Comm_rank(comm,&my_id);

  int * grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);
  int ** grid_relax_points = hypre_ParAMGDataGridRelaxPoints(amg_data);
  int relax_order = hypre_ParAMGDataRelaxOrder(amg_data);
  double * relax_weight = hypre_ParAMGDataRelaxWeight(amg_data); 
  double * omega = hypre_ParAMGDataOmega(amg_data); 
  int cycle_param = 0;
  int relax_type = grid_relax_type[cycle_param];
  FILE * out;
  char filename[100];

  HYPRE_IJVector ij_x_init, ij_b, ij_x_coarse;
  void * object;
  hypre_ParVector *x_init, *b, *x_coarse;
  hypre_ParVector * Vtemp = hypre_ParAMGDataVtemp(amg_data);
  hypre_Vector * local_x;
  double * local_x_data;

  int i, ierr, Solve_err_flag, level, relax_points;
  int first_local_row, last_local_row, local_num_rows;
  int first_local_col, last_local_col, local_num_cols;
  int coarse_first_local_row, coarse_last_local_row;
  int coarse_first_local_col, coarse_last_local_col;
  int global_num_rows;
  double * values;
  double e_f_prev, e_f_curr, local_max_e_f, max_e_f;
  double alpha, beta;

  int num_candidates;
  int * U;
  double * candidate_measures;

  for(level = 0; level < num_levels-1; level++) {
    sprintf(filename, "coarse-grid-effectiveness.out.%i.%i", my_id, level);
    out = fopen(filename, "w");

    ierr = HYPRE_ParCSRMatrixGetLocalRange((HYPRE_ParCSRMatrix)(A_array[level]), &first_local_row,
					   &last_local_row, &first_local_col,
					   &last_local_col);
    local_num_rows = last_local_row - first_local_row + 1;
    local_num_cols = last_local_col - first_local_col + 1;
    global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);
    
    // Relaxing Ax = b.

    ///////////////////////////////////////////////////////////////////////
    // Set right hand side to all zeros.                                 //
    //                                                                   //
    HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_b);
    HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_b);
    
    values = hypre_CTAlloc(double, local_num_cols);
    for (i = 0; i < local_num_cols; i++)
      values[i] = 0;

    HYPRE_IJVectorSetValues(ij_b, local_num_cols, NULL, values);
    hypre_TFree(values);

    ierr = HYPRE_IJVectorGetObject(ij_b, &object);
    b = (hypre_ParVector *)object;
    //                                                                   //
    ///////////////////////////////////////////////////////////////////////
    
    ///////////////////////////////////////////////////////////////////////
    // Set initial guess for e to 1 + rand[0, 1/2] for all entries in e. //
    //                                                                   //
    hypre_SeedRand(34902+my_id);

    HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x_init);
    HYPRE_IJVectorSetObjectType(ij_x_init, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_x_init);
   
    values = hypre_CTAlloc(double, local_num_cols);
    for (i = 0; i < local_num_cols; i++)
      values[i] = 1 + hypre_Rand()/4;

    HYPRE_IJVectorSetValues(ij_x_init, local_num_cols, NULL, values);
    hypre_TFree(values);

    ierr = HYPRE_IJVectorGetObject(ij_x_init, &object);
    x_init = (hypre_ParVector *)object;
    //                                                                   //
    ///////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////
    // Create vector to store restriction result.                        //
    //                                                                   //
    ierr = HYPRE_ParCSRMatrixGetLocalRange((HYPRE_ParCSRMatrix)(A_array[level+1]),
					   &coarse_first_local_row,
					   &coarse_last_local_row, &coarse_first_local_col,
					   &coarse_last_local_col);
    HYPRE_IJVectorCreate(comm, coarse_first_local_col, coarse_last_local_col, &ij_x_coarse);
    HYPRE_IJVectorSetObjectType(ij_x_coarse, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_x_coarse);
    
    ierr = HYPRE_IJVectorGetObject(ij_x_coarse, &object);
    x_coarse = (hypre_ParVector *)object;
    //                                                                   //
    ///////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////
    // Perform five smoother sweeps.                                   //
    //                                                                 //
    for(i = 0; i < 20; i++) {
      relax_points = grid_relax_points[cycle_param][i];
      Solve_err_flag = hypre_BoomerAMGRelax(A_array[level], b, CF_marker_array[level], relax_type,
					    1,relax_weight[level], omega[level], x_init,
					    Vtemp);
      Solve_err_flag = hypre_BoomerAMGRelax(A_array[level], b, CF_marker_array[level], relax_type,
					    -1,relax_weight[level], omega[level], x_init,
					    Vtemp);
    }
    // Compute x - P*R*x for each d.o.f.
    hypre_ParVectorCopy(x_init, Vtemp);
    alpha = -1.0;
    beta = 1.0;
    hypre_ParCSRMatrixMatvec(alpha, A_array[level], x_init, beta, Vtemp);

    alpha = 1.0;
    beta = 0.0;
    hypre_ParCSRMatrixMatvecT(alpha, R_array[level], Vtemp, beta, x_coarse);

    alpha = -1.0;
    beta = 1.0;
    hypre_ParCSRMatrixMatvec(alpha, P_array[level], x_coarse, beta, Vtemp);

    hypre_Vector * local_x, * local_v;
    double * local_x_data, * local_v_data;
    local_x = hypre_ParVectorLocalVector(x_init);
    local_x_data = hypre_VectorData(local_x);
    local_v = hypre_ParVectorLocalVector(Vtemp);
    local_v_data = hypre_VectorData(local_v);
    for(i = 0; i < local_num_rows; i++){
      printf("%e %e %e\n", local_x_data[i], local_v_data[i], local_x_data[i] - local_v_data[i]);
    }

    fclose(out);
  }
}

/** This function is called by each process of BoomerAMG after it finishes the
 * coarsening phase.
 *
 * It writes the C/F point selections and node connectivity from each level to
 * disk.
 *
 * The format of the information in the file is:\n
 *   global_node_#   C/F   neighbor_list
 *
 * @param A_array an array of hypre_ParCSRMatrix objects containing the linear system of each level.
 * @param CF_marker_array an array of int arrays defining each node on each level as a C or F point.
 * @param num_levels an integer containing the number of levels in the multigrid cycle.
 */
void WriteCoarseGridInformation(hypre_ParCSRMatrix ** A_array, hypre_ParCSRMatrix ** P_array, int ** CF_marker_array, int num_levels, int num_procs)
{
  int i, j, my_id, size, level, row_start;
  int * diag_i, * diag_j, * offd_i, * offd_j, * offd_map;
  FILE * out, * nodes_out;
  hypre_CSRMatrix * diag, * offd;
  char filename[100];
  short has_neighbors;
  int * C_list = NULL;  // list of C-points on current level
  int * CF_list = NULL; // list of all nodes on current level

  if(NO_WRITE_FILES)
    return;
  //return; // for now

  MPI_Comm comm = hypre_ParCSRMatrixComm(A_array[0]); 

  MPI_Comm_rank(comm,&my_id);

  // Write general information about the AMG run.
  out = fopen("run-info", "w");
  fprintf(out, "DOFs\tPROCS\tLEVELS\n");
  fprintf(out, "%d\t%d\t%d\n", hypre_ParCSRMatrixGlobalNumRows(A_array[0]), num_procs, num_levels);
  fclose(out);

  // initialize C_list to list of all nodes on this processor
  size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[0]));
  C_list = hypre_CTAlloc(int, size);
  for(i = 0; i < size; i++)
    C_list[i] = i+hypre_ParCSRMatrixRowStarts(A_array[0])[my_id];

  for(level = 0; level < num_levels-1; level++) {
    // for each level except the most coarse
    sprintf(filename, "coarsen.out.%i.%i", my_id, level);
    out = fopen(filename, "w");

    /*sprintf(filename, "nodeids.out.%i.%i", my_id, level+1);
      nodes_out = fopen(filename, "w");*/

    size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level]));
    row_start = hypre_ParCSRMatrixRowStarts(A_array[level])[my_id];

    // Build an array for each level that contains a node's number on the
    // fine level.

    CF_list = C_list;
    C_list = hypre_CTAlloc(int, hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(P_array[level])));
    j = 0;
    for(i = 0; i < size; i++) {
      if(CF_marker_array[level][i] == 1) {
	C_list[j] = CF_list[i];
	j++;
      }
    }
    /*for(i = 0; i < hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(P_array[level])); i++)
      fprintf(nodes_out, "%d %d\n", i+row_start, C_list[i]);*/

    //printf("%d %d (%d)\n", size, hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(P_array[level])), hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(P_array[level])));

    for(i = 0; i < size; i++) {
      // for each row owned by this process

      // Print this row's global number.
      fprintf(out, "%i %i ", row_start+i, CF_marker_array[level][i]);

      // Get the node numbers of all neighbors of this row.
      diag = hypre_ParCSRMatrixDiag(A_array[level]);
      diag_i = hypre_CSRMatrixI(diag);
      diag_j = hypre_CSRMatrixJ(diag);

      // offd is the off-process connections
      offd = hypre_ParCSRMatrixOffd(A_array[level]);
      offd_map = hypre_ParCSRMatrixColMapOffd(A_array[level]);
      offd_i = hypre_CSRMatrixI(offd);
      offd_j = hypre_CSRMatrixJ(offd);

      //fprintf(out, "%i ", diag_i[i+1]-diag_i[i]);
      for(j = diag_i[i]; j < diag_i[i+1]; j++) {
	if(diag_j[j] != i)
	  fprintf(out, "%i ", row_start+diag_j[j]);
      }

      for(j = offd_i[i]; j < offd_i[i+1]; j++) {
	fprintf(out, "%i ", offd_map[offd_j[j]]);
      }
      fprintf(out, "\n");
    }
    fclose(out);
    //fclose(nodes_out);
    hypre_TFree(CF_list);
  }
  hypre_TFree(C_list);
}
