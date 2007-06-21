#include "headers.h"
#include "par_amg.h"
#include "HYPRE.h"

#define F_POINT -1
#define C_POINT 1
#define SF_POINT -3  /* special fine points */

typedef short bool;
#define FALSE 0
#define TRUE 1
#define RHO_LIMIT 0.7

struct column_el {
  struct node * me;
  struct column_el * next;
};
typedef struct column_el Column;

struct node {
  int local_id;
  short major_weight;
  double gamma_weight;
  bool unselected_candidate; // has not been added to F or C set yet
  short update_iteration; // used when weights are being updated to determine if this
                          // node has already been updated
  struct column_el * head;
};
typedef struct node Node;

///////////////////////////////////////////////////////////////////////////////////


Node * extractSubgraph(hypre_ParCSRMatrix * A, int * U, double * candidate_measures,
		       int num_rows, int num_candidates)
{
  int i, j, num_nz_cols, candidate_index, node_id;
  int * node_id_map = hypre_CTAlloc(int, num_rows);
  hypre_CSRMatrix * A_local = hypre_ParCSRMatrixDiag(A);
  int * i_vector = hypre_CSRMatrixI(A_local);
  int * j_vector = hypre_CSRMatrixJ(A_local);

  Node * A_ff = hypre_CTAlloc(Node, num_candidates);
  Column * new_col, * tail;

  // Entries that go into A_ff are marked by 1 in the set U.
  candidate_index = 0;
  for(i = 0; i < num_rows; i++) {
    if(U[i] == 1) {
      // Then this node is a candidate node.
      node_id_map[i] = candidate_index;
      A_ff[candidate_index].local_id = i;
      A_ff[candidate_index].major_weight = 0;
      A_ff[candidate_index].gamma_weight = candidate_measures[i];
      A_ff[candidate_index].unselected_candidate = TRUE;
      A_ff[candidate_index].update_iteration = -1;
      A_ff[candidate_index].head = NULL;
      candidate_index++;
    }
    else
      node_id_map[i] = -1;
  }

  // Now create the neighbor lists for each candidate node.
  for(i = 0; i < num_candidates; i++) {
    node_id = A_ff[i].local_id;
    num_nz_cols = i_vector[node_id+1] - i_vector[node_id];
    tail = A_ff[i].head;
    for(j = 0; j < num_nz_cols; j++) {
      if(node_id_map[j_vector[j + i_vector[node_id]]] > -1 && j_vector[j + i_vector[node_id]] != node_id) {
	// Then this column in A belongs in A_ff.
	A_ff[i].major_weight++;
	new_col = hypre_CTAlloc(Column, 1);
	new_col->me = &A_ff[node_id_map[j_vector[j + i_vector[node_id]]]];
	new_col->next = NULL;

	// Append this new column to the column list of A_ff[i].
	if(tail)
	  tail->next = new_col;
	else
	  A_ff[i].head = new_col;
	tail = new_col;
      }
    }
  }

  hypre_TFree(node_id_map);
  return A_ff;
}

void destroySubgraphInformation(Node * A_ff, int num_candidates)
     // Deallocates subgraph information.
{
  int i;
  Column * curr_col, * prev_col;
  for(i = 0; i < num_candidates; i++) {
    curr_col = A_ff[i].head;
    while(curr_col) {
      prev_col = curr_col;
      curr_col = prev_col->next;
      hypre_TFree(prev_col);
    }
  }
  hypre_TFree(A_ff);
}

void addIndependentSet_Brannick(Node * A_ff, int * CF_marker, int num_rows,
				int num_candidates)
{
  int added = 0;
  int i, max_candidate;
  int assigned_candidates = 0;
  double max_weight;
  Column * curr_col, * curr_d2_col, * curr_d3_col;

if(num_candidates == 0) printf("WHAT IS GOING ON?!!!!\n");
  while(assigned_candidates < num_candidates) {
    // Find candidate with maximal weight. THIS IS A BAD WAY TO DO THIS! THINK OF
    // SOMETHING BETTER.
    max_weight = -1000;
    max_candidate = -1;
    for(i = 0; i < num_candidates; i++) {
      if(A_ff[i].unselected_candidate && 
	 A_ff[i].major_weight + A_ff[i].gamma_weight > max_weight) {
	max_weight = A_ff[i].major_weight + A_ff[i].gamma_weight;
	max_candidate = i;
      }
    }

    // Make candidate with max weight a C-point.
    CF_marker[A_ff[max_candidate].local_id] = C_POINT;
    added++;
    A_ff[max_candidate].unselected_candidate = FALSE;
    assigned_candidates++;

    // Assign nodes distance one from this node to be F-points (more accurate: remove
    // them as candidates to become C-points).
    curr_col = A_ff[max_candidate].head;
    while(curr_col) {
      if(curr_col->me->unselected_candidate) {
	curr_col->me->unselected_candidate = FALSE;
	assigned_candidates++;

	// Add one to the weights of the neighbors of the new F-variables.
	curr_d2_col = curr_col->me->head;
	while(curr_d2_col) {
	  curr_d2_col->me->major_weight++;
	  
	  /*// Subtract one from the weights of the neighbors of the nodes who
	  // had one added to their weights.
	  curr_d3_col = curr_d2_col->me->head;
	  while(curr_d3_col) {
	    curr_d3_col->me->major_weight--;
	    curr_d3_col = curr_d3_col->next;
	    }*/
	  curr_d2_col = curr_d2_col->next;
	}
      }

      curr_col = curr_col->next;
    }
  }
}

/* int */
/* addIndependentSet_CLJP_c( hypre_ParCSRMatrix    *S, */
/*                         hypre_ParCSRMatrix    *A, */
/*                         int                    CF_init, */
/*                         int                    debug_flag, */
/*                         int                  **CF_marker_ptr, */
/* 			int                    global, */
/* 			int                    level) */
/* { */
/*    MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(S); */
/*    hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(A); */
/*    hypre_ParCSRCommHandle   *comm_handle; */

/*    hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S); */
/*    int                *S_diag_i        = hypre_CSRMatrixI(S_diag); */
/*    int                *S_diag_j        = hypre_CSRMatrixJ(S_diag); */

/*    hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S); */
/*    int                *S_offd_i        = hypre_CSRMatrixI(S_offd); */
/*    int                *S_offd_j; */

/*    int 		      *col_map_offd    = hypre_ParCSRMatrixColMapOffd(S); */
/*    int                 num_variables   = hypre_CSRMatrixNumRows(S_diag); */
/*    int		       col_1 = hypre_ParCSRMatrixFirstColDiag(S); */
/*    int		       col_n = col_1 + hypre_CSRMatrixNumCols(S_diag); */
/*    int 		       num_cols_offd = 0; */
                  
/*    hypre_CSRMatrix    *S_ext; */
/*    int                *S_ext_i; */
/*    int                *S_ext_j; */

/*    int		       num_sends = 0; */
/*    int  	      *int_buf_data; */
/*    double	      *buf_data; */

/*    int                *CF_marker; */
/*    int                *CF_marker_offd; */

/*    short              *color_array; */
/*    int                num_colors; */
                      
/*    double             *measure_array; */
/*    int                *graph_array; */
/*    int                *graph_array_offd; */
/*    int                 graph_size; */
/*    int                 graph_offd_size; */
/*    int                 global_graph_size; */
                      
/*    int                 i, j, k, kc, jS, kS, ig; */
/*    int		       index, start, my_id, num_procs, jrow, cnt; */
                      
/*    int                 ierr = 0; */
/*    int                 break_var = 1; */

/*    double	    wall_time; */
/*    int   iter = 0; */

/* #if 0 /\* debugging *\/ */
/*    char  filename[256]; */
/*    FILE *fp; */
/*    int   iter = 0; */
/* #endif */

/*    color_array = hypre_CTAlloc(short, num_variables); */

/*    if(global) */
/*      parColorGraph(A, S, color_array, &num_colors, level); */
/*    else */
/*      seqColorGraphNew(S, color_array, &num_colors, level); */
   
/*    /\*-------------------------------------------------------------- */
/*     * Compute a  ParCSR strength matrix, S. */
/*     * */
/*     * For now, the "strength" of dependence/influence is defined in */
/*     * the following way: i depends on j if */
/*     *     aij > hypre_max (k != i) aik,    aii < 0 */
/*     * or */
/*     *     aij < hypre_min (k != i) aik,    aii >= 0 */
/*     * Then S_ij = 1, else S_ij = 0. */
/*     * */
/*     * NOTE: the entries are negative initially, corresponding */
/*     * to "unaccounted-for" dependence. */
/*     *----------------------------------------------------------------*\/ */

/*    S_ext = NULL; */
/*    if (debug_flag == 3) wall_time = time_getWallclockSeconds(); */
/*    MPI_Comm_size(comm,&num_procs); */
/*    MPI_Comm_rank(comm,&my_id); */

/*    if (!comm_pkg) */
/*    { */
/*         hypre_MatvecCommPkgCreate(A); */
/*         comm_pkg = hypre_ParCSRMatrixCommPkg(A); */
/*    } */

/*    num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg); */

/*    int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg, */
/*                                                 num_sends)); */
/*    buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart(comm_pkg, */
/*                                                 num_sends)); */
 
/*    num_cols_offd = hypre_CSRMatrixNumCols(S_offd); */

/*    S_diag_j = hypre_CSRMatrixJ(S_diag); */

/*    if (num_cols_offd) */
/*    { */
/*       S_offd_j = hypre_CSRMatrixJ(S_offd); */
/*    } */
/*    /\*---------------------------------------------------------- */
/*     * Compute the measures */
/*     * */
/*     * The measures are currently given by the column sums of S. */
/*     * Hence, measure_array[i] is the number of influences */
/*     * of variable i. */
/*     * */
/*     * The measures are augmented by a random number */
/*     * between 0 and 1. */
/*     *----------------------------------------------------------*\/ */

/*    measure_array = hypre_CTAlloc(double, num_variables+num_cols_offd); */

/*    for (i=0; i < S_offd_i[num_variables]; i++) */
/*    { */
/*       measure_array[num_variables + S_offd_j[i]] += 1.0; */
/*    } */
/*    if (num_procs > 1) */
/*    comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg, */
/*                         &measure_array[num_variables], buf_data); */

/*    for (i=0; i < S_diag_i[num_variables]; i++) */
/*    { */
/*       measure_array[S_diag_j[i]] += 1.0; */
/*    } */

/*    if (num_procs > 1) */
/*    hypre_ParCSRCommHandleDestroy(comm_handle); */
      
/*    index = 0; */
/*    for (i=0; i < num_sends; i++) */
/*    { */
/*       start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i); */
/*       for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++) */
/*             measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)] */
/*                         += buf_data[index++]; */
/*    } */

/*    for (i=num_variables; i < num_variables+num_cols_offd; i++) */
/*    /\* This loop zeros out the measures for the off-process nodes since */
/*       this process is not responsible for . *\/ */
/*    { */
/*       measure_array[i] = 0; */
/*    } */

/*    /\* this augments the measures *\/ */
/*    //hypre_BoomerAMGIndepSetInit(S, measure_array); */
/*    hypre_BoomerAMGIndepSetInitb(S, measure_array, color_array, num_colors); */

/*    /\*--------------------------------------------------- */
/*     * Initialize the graph array */
/*     * graph_array contains interior points in elements 0 ... num_variables-1 */
/*     * followed by boundary values */
/*     *---------------------------------------------------*\/ */

/*    graph_array = hypre_CTAlloc(int, num_variables); */
/*    if (num_cols_offd) */
/*       graph_array_offd = hypre_CTAlloc(int, num_cols_offd); */
/*    else */
/*       graph_array_offd = NULL; */

/*    /\* initialize measure array and graph array *\/ */

/*    for (ig = 0; ig < num_cols_offd; ig++) */
/*       graph_array_offd[ig] = ig; */

/*    /\*--------------------------------------------------- */
/*     * Initialize the C/F marker array */
/*     * C/F marker array contains interior points in elements 0 ... */
/*     * num_variables-1  followed by boundary values */
/*     *---------------------------------------------------*\/ */

/*    graph_offd_size = num_cols_offd; */

/*    if (CF_init) */
/*    { */
/*       CF_marker = *CF_marker_ptr; */
/*       cnt = 0; */
/*       for (i=0; i < num_variables; i++) */
/*       { */
/*          if ( (S_offd_i[i+1]-S_offd_i[i]) > 0 */
/*                  || CF_marker[i] == -1) */
/*          { */
/*             CF_marker[i] = 0; */
/*          } */
/*          if ( CF_marker[i] == Z_PT) */
/*          { */
/*             if (measure_array[i] >= 1.0 || */
/*                 (S_diag_i[i+1]-S_diag_i[i]) > 0) */
/*             { */
/*                CF_marker[i] = 0; */
/*                graph_array[cnt++] = i; */
/*             } */
/*             else */
/*             { */
/*                graph_size--; */
/*                CF_marker[i] = F_PT; */
/*             } */
/*          } */
/*          else if (CF_marker[i] == SF_PT) */
/* 	    measure_array[i] = 0; */
/*          else */
/*             graph_array[cnt++] = i; */
/*       } */
/*    } */
/*    else */
/*    { */
/*       CF_marker = hypre_CTAlloc(int, num_variables); */
/*       cnt = 0; */
/*       for (i=0; i < num_variables; i++) */
/*       { */
/* 	 CF_marker[i] = 0; */
/* 	 if ( (S_diag_i[i+1]-S_diag_i[i]) == 0 */
/* 		&& (S_offd_i[i+1]-S_offd_i[i]) == 0) */
/* 	 { */
/* 	    CF_marker[i] = SF_PT; */
/* 	    measure_array[i] = 0; */
/* 	 } */
/* 	 else */
/*             graph_array[cnt++] = i; */
/*       } */
/*    } */
/*    graph_size = cnt; */
/*    if (num_cols_offd) */
/*       CF_marker_offd = hypre_CTAlloc(int, num_cols_offd); */
/*    else */
/*       CF_marker_offd = NULL; */
/*    for (i=0; i < num_cols_offd; i++) */
/* 	CF_marker_offd[i] = 0; */
  
/*    /\*--------------------------------------------------- */
/*     * Loop until all points are either fine or coarse. */
/*     *---------------------------------------------------*\/ */

/*    if (num_procs > 1) */
/*    { */
/*       S_ext      = hypre_ParCSRMatrixExtractBExt(S,A,0); */
/*       S_ext_i    = hypre_CSRMatrixI(S_ext); */
/*       S_ext_j    = hypre_CSRMatrixJ(S_ext); */
/*    } */

/*    /\*  compress S_ext  and convert column numbers*\/ */

/*    index = 0; */
/*    for (i=0; i < num_cols_offd; i++) */
/*    { */
/*       for (j=S_ext_i[i]; j < S_ext_i[i+1]; j++) */
/*       { */
/* 	 k = S_ext_j[j]; */
/* 	 if (k >= col_1 && k < col_n) */
/* 	 { */
/* 	    S_ext_j[index++] = k - col_1; */
/* 	 } */
/* 	 else */
/* 	 { */
/* 	    kc = hypre_BinarySearch(col_map_offd,k,num_cols_offd); */
/* 	    if (kc > -1) S_ext_j[index++] = -kc-1; */
/* 	 } */
/*       } */
/*       S_ext_i[i] = index; */
/*    } */
/*    for (i = num_cols_offd; i > 0; i--) */
/*       S_ext_i[i] = S_ext_i[i-1]; */
/*    if (num_procs > 1) S_ext_i[0] = 0; */

/*    if (debug_flag == 3) */
/*    { */
/*       wall_time = time_getWallclockSeconds() - wall_time; */
/*       printf("Proc = %d    Initialize CLJP phase = %f\n", */
/*                      my_id, wall_time); */
/*    } */

/*    while (1) */
/*    { */
/*       /\*------------------------------------------------ */
/*        * Exchange boundary data, i.i. get measures and S_ext_data */
/*        *------------------------------------------------*\/ */

/*       if (num_procs > 1) */
/*    	 comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg, */
/*                         &measure_array[num_variables], buf_data); */

/*       if (num_procs > 1) */
/*    	 hypre_ParCSRCommHandleDestroy(comm_handle); */
      
/*       index = 0; */
/*       for (i=0; i < num_sends; i++) */
/*       { */
/*          start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i); */
/*          for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++) */
/*             measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)] */
/*                         += buf_data[index++]; */
/*       } */

/*       /\*------------------------------------------------ */
/*        * Set F-pts and update subgraph */
/*        *------------------------------------------------*\/ */
 
/*       if (iter || !CF_init) */
/*       { */
/*          for (ig = 0; ig < graph_size; ig++) */
/*          { */
/*             i = graph_array[ig]; */

/*             if ( (CF_marker[i] != C_PT) && (measure_array[i] < 1) ) */
/*             { */
/*                /\* set to be an F-pt *\/ */
/*                CF_marker[i] = F_PT; */
 
/* 	       /\* make sure all dependencies have been accounted for *\/ */
/*                for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++) */
/*                { */
/*                   if (S_diag_j[jS] > -1) */
/*                   { */
/*                      CF_marker[i] = 0; */
/*                   } */
/*                } */
/*                for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++) */
/*                { */
/*                   if (S_offd_j[jS] > -1) */
/*                   { */
/*                      CF_marker[i] = 0; */
/*                   } */
/*                } */
/*             } */
/*             if (CF_marker[i]) */
/*             { */
/*                measure_array[i] = 0; */
 
/*                /\* take point out of the subgraph *\/ */
/*                graph_size--; */
/*                graph_array[ig] = graph_array[graph_size]; */
/*                graph_array[graph_size] = i; */
/*                ig--; */
/*             } */
/*          } */
/*       } */
 
/*       /\*------------------------------------------------ */
/*        * Exchange boundary data, i.i. get measures */
/*        *------------------------------------------------*\/ */

/*       if (debug_flag == 3) wall_time = time_getWallclockSeconds(); */

/*       index = 0; */
/*       for (i = 0; i < num_sends; i++) */
/*       { */
/*         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i); */
/*         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++) */
/*         { */
/*             jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j); */
/*             buf_data[index++] = measure_array[jrow]; */
/*          } */
/*       } */

/*       if (num_procs > 1) */
/*       { */
/*          comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg, buf_data, */
/*         	&measure_array[num_variables]); */
 
/*          hypre_ParCSRCommHandleDestroy(comm_handle); */
 
/*       } */
/*       /\*------------------------------------------------ */
/*        * Debugging: */
/*        * */
/*        * Uncomment the sections of code labeled */
/*        * "debugging" to generate several files that */
/*        * can be visualized using the `coarsen.m' */
/*        * matlab routine. */
/*        *------------------------------------------------*\/ */

/* #if 0 /\* debugging *\/ */
/*       /\* print out measures *\/ */
/*       char filename[50]; */
/*       FILE * fp; */
/*       sprintf(filename, "coarsen.out.measures.%04d", iter); */
/*       fp = fopen(filename, "w"); */
/*       for (i = 0; i < num_variables; i++) */
/*       { */
/*          fprintf(fp, "%f\n", measure_array[i]); */
/*       } */
/*       fclose(fp); */

/*       /\* print out strength matrix *\/ */
/*       sprintf(filename, "coarsen.out.strength.%04d", iter); */
/*       hypre_CSRMatrixPrint(S, filename); */

/*       /\* print out C/F marker *\/ */
/*       sprintf(filename, "coarsen.out.CF.%04d", iter); */
/*       fp = fopen(filename, "w"); */
/*       for (i = 0; i < num_variables; i++) */
/*       { */
/*          fprintf(fp, "%d\n", CF_marker[i]); */
/*       } */
/*       fclose(fp); */

/*       //iter++; */
/* #endif */

/*       /\*------------------------------------------------ */
/*        * Test for convergence */
/*        *------------------------------------------------*\/ */

/*       MPI_Allreduce(&graph_size,&global_graph_size,1,MPI_INT,MPI_SUM,comm); */

/*       if (global_graph_size == 0) */
/*          break; */

/*       /\*------------------------------------------------ */
/*        * Pick an independent set of points with */
/*        * maximal measure. */
/*        *------------------------------------------------*\/ */
/*       if (iter || !CF_init) */
/*          hypre_BoomerAMGIndepSet(S, measure_array, graph_array, */
/* 				graph_size, */
/* 				graph_array_offd, graph_offd_size, */
/* 				CF_marker, CF_marker_offd); */

/*       iter++; */
/*       /\*------------------------------------------------ */
/*        * Exchange boundary data for CF_marker */
/*        *------------------------------------------------*\/ */

/*       index = 0; */
/*       for (i = 0; i < num_sends; i++) */
/*       { */
/*         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i); */
/*         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++) */
/*                 int_buf_data[index++] */
/*                  = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]; */
/*       } */
 
/*       if (num_procs > 1) */
/*       { */
/*       comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, */
/*         CF_marker_offd); */
 
/*       hypre_ParCSRCommHandleDestroy(comm_handle); */
/*       } */
 
/*       for (ig = 0; ig < graph_offd_size; ig++) */
/*       { */
/*          i = graph_array_offd[ig]; */

/*          if (CF_marker_offd[i] < 0) */
/*          { */
/*             /\* take point out of the subgraph *\/ */
/*             graph_offd_size--; */
/*             graph_array_offd[ig] = graph_array_offd[graph_offd_size]; */
/*             graph_array_offd[graph_offd_size] = i; */
/*             ig--; */
/*          } */
/*       } */
/*       if (debug_flag == 3) */
/*       { */
/*          wall_time = time_getWallclockSeconds() - wall_time; */
/*          printf("Proc = %d  iter %d  comm. and subgraph update = %f\n", */
/*                      my_id, iter, wall_time); */
/*       } */
/*       /\*------------------------------------------------ */
/*        * Set C_pts and apply heuristics. */
/*        *------------------------------------------------*\/ */

/*       for (i=num_variables; i < num_variables+num_cols_offd; i++) */
/*       { */
/*          measure_array[i] = 0; */
/*       } */

/*       if (debug_flag == 3) wall_time = time_getWallclockSeconds(); */
/*       for (ig = 0; ig < graph_size; ig++) */
/*       { */
/*          i = graph_array[ig]; */

/*          /\*--------------------------------------------- */
/*           * Heuristic: C-pts don't interpolate from */
/*           * neighbors that influence them. */
/*           *---------------------------------------------*\/ */

/*          if (CF_marker[i] > 0) */
/*          { */
/*             /\* set to be a C-pt *\/ */
/*             CF_marker[i] = C_PT; */

/*             for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++) */
/*             { */
/*                j = S_diag_j[jS]; */
/*                if (j > -1) */
/*                { */
               
/*                   /\* "remove" edge from S *\/ */
/*                   S_diag_j[jS] = -S_diag_j[jS]-1; */
             
/*                   /\* decrement measures of unmarked neighbors *\/ */
/*                   if (!CF_marker[j]) */
/*                   { */
/*                      measure_array[j]--; */
/*                   } */
/*                } */
/*             } */
/*             for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++) */
/*             { */
/*                j = S_offd_j[jS]; */
/*                if (j > -1) */
/*                { */
             
/*                   /\* "remove" edge from S *\/ */
/*                   S_offd_j[jS] = -S_offd_j[jS]-1; */
               
/*                   /\* decrement measures of unmarked neighbors *\/ */
/*                   if (!CF_marker_offd[j]) */
/*                   { */
/*                      measure_array[j+num_variables]--; */
/*                   } */
/*                } */
/*             } */
/*          } */
/* 	 else */
/*     	 { */
/*             /\* marked dependencies *\/ */
/*             for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++) */
/*             { */
/*                j = S_diag_j[jS]; */
/* 	       if (j < 0) j = -j-1; */
   
/*                if (CF_marker[j] > 0) */
/*                { */
/*                   if (S_diag_j[jS] > -1) */
/*                   { */
/*                      /\* "remove" edge from S *\/ */
/*                      S_diag_j[jS] = -S_diag_j[jS]-1; */
/*                   } */
   
/*                   /\* IMPORTANT: consider all dependencies *\/ */
/*                   /\* temporarily modify CF_marker *\/ */
/*                   CF_marker[j] = COMMON_C_PT; */
/*                } */
/*                else if (CF_marker[j] == SF_PT) */
/*                { */
/*                   if (S_diag_j[jS] > -1) */
/*                   { */
/*                      /\* "remove" edge from S *\/ */
/*                      S_diag_j[jS] = -S_diag_j[jS]-1; */
/*                   } */
/*                } */
/*             } */
/*             for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++) */
/*             { */
/*                j = S_offd_j[jS]; */
/* 	       if (j < 0) j = -j-1; */
   
/*                if (CF_marker_offd[j] > 0) */
/*                { */
/*                   if (S_offd_j[jS] > -1) */
/*                   { */
/*                      /\* "remove" edge from S *\/ */
/*                      S_offd_j[jS] = -S_offd_j[jS]-1; */
/*                   } */
   
/*                   /\* IMPORTANT: consider all dependencies *\/ */
/*                   /\* temporarily modify CF_marker *\/ */
/*                   CF_marker_offd[j] = COMMON_C_PT; */
/*                } */
/*                else if (CF_marker_offd[j] == SF_PT) */
/*                { */
/*                   if (S_offd_j[jS] > -1) */
/*                   { */
/*                      /\* "remove" edge from S *\/ */
/*                      S_offd_j[jS] = -S_offd_j[jS]-1; */
/*                   } */
/*                } */
/*             } */
   
/*             /\* unmarked dependencies *\/ */
/*             for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++) */
/*             { */
/*                if (S_diag_j[jS] > -1) */
/*                { */
/*                   j = S_diag_j[jS]; */
/*    	          break_var = 1; */
/*                   /\* check for common C-pt *\/ */
/*                   for (kS = S_diag_i[j]; kS < S_diag_i[j+1]; kS++) */
/*                   { */
/*                      k = S_diag_j[kS]; */
/* 		     if (k < 0) k = -k-1; */
   
/*                      /\* IMPORTANT: consider all dependencies *\/ */
/*                      if (CF_marker[k] == COMMON_C_PT) */
/*                      { */
/*                         /\* "remove" edge from S and update measure*\/ */
/*                         S_diag_j[jS] = -S_diag_j[jS]-1; */
/*                         measure_array[j]--; */
/*                         break_var = 0; */
/*                         break; */
/*                      } */
/*                   } */
/*    		  if (break_var) */
/*                   { */
/*                      for (kS = S_offd_i[j]; kS < S_offd_i[j+1]; kS++) */
/*                      { */
/*                         k = S_offd_j[kS]; */
/* 		        if (k < 0) k = -k-1; */
   
/*                         /\* IMPORTANT: consider all dependencies *\/ */
/*                         if ( CF_marker_offd[k] == COMMON_C_PT) */
/*                         { */
/*                            /\* "remove" edge from S and update measure*\/ */
/*                            S_diag_j[jS] = -S_diag_j[jS]-1; */
/*                            measure_array[j]--; */
/*                            break; */
/*                         } */
/*                      } */
/*                   } */
/*                } */
/*             } */
/*             for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++) */
/*             { */
/*                if (S_offd_j[jS] > -1) */
/*                { */
/*                   j = S_offd_j[jS]; */
   
/*                   /\* check for common C-pt *\/ */
/*                   for (kS = S_ext_i[j]; kS < S_ext_i[j+1]; kS++) */
/*                   { */
/*                      k = S_ext_j[kS]; */
/*    	             if (k >= 0) */
/*    		     { */
/*                         /\* IMPORTANT: consider all dependencies *\/ */
/*                         if (CF_marker[k] == COMMON_C_PT) */
/*                         { */
/*                            /\* "remove" edge from S and update measure*\/ */
/*                            S_offd_j[jS] = -S_offd_j[jS]-1; */
/*                            measure_array[j+num_variables]--; */
/*                            break; */
/*                         } */
/*                      } */
/*    		     else */
/*    		     { */
/*    		        kc = -k-1; */
/*    		        if (kc > -1 && CF_marker_offd[kc] == COMMON_C_PT) */
/*    		        { */
/*                            /\* "remove" edge from S and update measure*\/ */
/*                            S_offd_j[jS] = -S_offd_j[jS]-1; */
/*                            measure_array[j+num_variables]--; */
/*                            break; */
/*    		        } */
/*    		     } */
/*                   } */
/*                } */
/*             } */
/*          } */

/*          /\* reset CF_marker *\/ */
/* 	 for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++) */
/* 	 { */
/*             j = S_diag_j[jS]; */
/* 	    if (j < 0) j = -j-1; */

/*             if (CF_marker[j] == COMMON_C_PT) */
/*             { */
/*                CF_marker[j] = C_PT; */
/*             } */
/*          } */
/*          for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++) */
/*          { */
/*             j = S_offd_j[jS]; */
/* 	    if (j < 0) j = -j-1; */

/*             if (CF_marker_offd[j] == COMMON_C_PT) */
/*             { */
/*                CF_marker_offd[j] = C_PT; */
/*             } */
/*          } */
/*       } */
/*       if (debug_flag == 3) */
/*       { */
/*          wall_time = time_getWallclockSeconds() - wall_time; */
/*          printf("Proc = %d    CLJP phase = %f graph_size = %d nc_offd = %d\n", */
/*                      my_id, wall_time, graph_size, num_cols_offd); */
/*       } */
/*    } */

/*    /\*--------------------------------------------------- */
/*     * Clean up and return */
/*     *---------------------------------------------------*\/ */

/*    /\* Reset S_matrix *\/ */
/*    for (i=0; i < S_diag_i[num_variables]; i++) */
/*    { */
/*       if (S_diag_j[i] < 0) */
/*          S_diag_j[i] = -S_diag_j[i]-1; */
/*    } */
/*    for (i=0; i < S_offd_i[num_variables]; i++) */
/*    { */
/*       if (S_offd_j[i] < 0) */
/*          S_offd_j[i] = -S_offd_j[i]-1; */
/*    } */
/*    /\*for (i=0; i < num_variables; i++) */
/*       if (CF_marker[i] == SF_PT) CF_marker[i] = F_PT;*\/ */

/*    hypre_TFree(color_array); */

/*    hypre_TFree(measure_array); */
/*    hypre_TFree(graph_array); */
/*    if (num_cols_offd) hypre_TFree(graph_array_offd); */
/*    hypre_TFree(buf_data); */
/*    hypre_TFree(int_buf_data); */
/*    hypre_TFree(CF_marker_offd); */
/*    if (num_procs > 1) hypre_CSRMatrixDestroy(S_ext); */

/*    *CF_marker_ptr   = CF_marker; */

/*    return (ierr); */
/* } */

int precolorGraph(hypre_ParCSRMatrix * S, short ** color_array, int num_variables, int level)
{
  int num_colors;

  *color_array = hypre_CTAlloc(short, num_variables);
  seqColorGraphNew(S, *color_array, &num_colors, level);

  return num_colors;
}

/* Initializes the measures used by CLJP-c. It is done here once, rather than doing it several
   times during the selection of independent sets in the CR algorithm. */
void initializeMeasures(hypre_ParCSRMatrix * S, double ** measure_array_ptr, int num_variables,
			short * color_array, int num_colors, int num_procs)
{
  int i, j, num_cols_offd, num_sends, start, index;
  double * measure_array;
  double * buf_data;

  hypre_ParCSRCommPkg * comm_pkg = hypre_ParCSRMatrixCommPkg(S);
  hypre_ParCSRCommHandle * comm_handle;

  hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
  int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
  int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);

  hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
  int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
  int                *S_offd_j;

  num_cols_offd = hypre_CSRMatrixNumCols(S_offd);

  if (!comm_pkg) {
    hypre_MatvecCommPkgCreate(S);
    comm_pkg = hypre_ParCSRMatrixCommPkg(S);
  }

  num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
  if (num_cols_offd) {
    S_offd_j = hypre_CSRMatrixJ(S_offd);
  }
  /*----------------------------------------------------------
   * Compute the measures
   *
   * The measures are currently given by the column sums of S.
   * Hence, measure_array[i] is the number of influences
   * of variable i.
   *
   * The measures are augmented by a random number
   * between 0 and 1.
   *----------------------------------------------------------*/

  measure_array = hypre_CTAlloc(double, num_variables+num_cols_offd);

  for (i=0; i < S_offd_i[num_variables]; i++) {
    measure_array[num_variables + S_offd_j[i]] += 1.0;
  }
  if (num_procs > 1)
    comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg,
					       &measure_array[num_variables], buf_data);

  for (i=0; i < S_diag_i[num_variables]; i++) {
    measure_array[S_diag_j[i]] += 1.0;
  }

  if (num_procs > 1)
    hypre_ParCSRCommHandleDestroy(comm_handle);
      
  index = 0;
  for (i=0; i < num_sends; i++) {
    start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
    for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
      measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
	+= buf_data[index++];
  }

  for (i=num_variables; i < num_variables+num_cols_offd; i++) {
    /* This loop zeros out the measures for the off-process nodes since
       this process is not responsible for . */
    measure_array[i] = 0;
  }

  /* this augments the measures */
  //hypre_BoomerAMGIndepSetInit(S, measure_array);
  hypre_BoomerAMGIndepSetInitb(S, measure_array, color_array, num_colors);

  *measure_array_ptr = measure_array;
}

int hypre_BoomerAMGCRCoarsen(void *amg_vdata,
			     hypre_ParCSRMatrix * S,
			     hypre_ParCSRMatrix * A,
			     int **CF_marker_ptr,
			     int   level,
			     int   coarsen_type,
			     int   debug_flag)
{
   hypre_ParAMGData * amg_data = amg_vdata;
   //hypre_ParCSRMatrix ** A_array = hypre_ParAMGDataAArray(amg_data);
   //hypre_ParCSRMatrix * A = A_array[level];

   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   int num_procs, my_id;
   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   int * CF_marker, * temp_CF_marker;
   int * grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);
   int ** grid_relax_points = hypre_ParAMGDataGridRelaxPoints(amg_data);
   int relax_order = hypre_ParAMGDataRelaxOrder(amg_data);
   double * relax_weight = hypre_ParAMGDataRelaxWeight(amg_data); 
   double * omega = hypre_ParAMGDataOmega(amg_data); 
   int cycle_param = 0;
   int relax_type = grid_relax_type[cycle_param];

   HYPRE_IJVector ij_x_init, ij_x, ij_b, ij_xt, ij_xt2;
   void * object;
   hypre_ParVector *x_init, *x, *b, *xt, *xt2;
   hypre_ParVector * Vtemp = hypre_ParAMGDataVtemp(amg_data);
   hypre_Vector * local_x;
   double * local_x_data;

   int i, ierr, Solve_err_flag;
   int first_local_row, last_local_row, local_num_rows;
   int first_local_col, last_local_col, local_num_cols;
   int global_num_rows;
   double * values;
   double e_f_prev, e_f_curr, rho_cr, rho_cr_m1, local_max_e_f, max_e_f;

   int num_candidates;
   int * U;
   double * candidate_measures;
   Node * A_ff;

   int num_colors;
   short * color_array = NULL;
   double * measure_array;

   ierr = HYPRE_ParCSRMatrixGetLocalRange((HYPRE_ParCSRMatrix)(A), &first_local_row,
					  &last_local_row, &first_local_col,
					  &last_local_col);
   local_num_rows = last_local_row - first_local_row + 1;
   local_num_cols = last_local_col - first_local_col + 1;
   global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);

   // Initialize the CF_marker_array.
   //CF_marker = hypre_CTAlloc(int, local_num_rows);
   CF_marker = hypre_CTAlloc(int, local_num_rows);
   temp_CF_marker = hypre_CTAlloc(int, local_num_rows);
   for(i = 0; i < local_num_rows; i++) {
     temp_CF_marker[i] = SF_POINT;
     CF_marker[i] = F_POINT;
   }
   U = hypre_CTAlloc(int, local_num_rows);
   candidate_measures = hypre_CTAlloc(double, local_num_rows);

   if(coarsen_type == 30 || coarsen_type == 31) {
     // Color the graph of the matrix -- used by the independent set.
     num_colors = precolorGraph(S, &color_array, local_num_rows, level);
     initializeMeasures(S, &measure_array, local_num_rows, color_array, num_colors, num_procs);
   }

   // Relaxing Ax = b.

   ///////////////////////////////////////////////////////////////////////
   // Set right hand side to all zeros.                                 //
   //                                                                   //
   /*b = hypre_ParVectorCreate(comm, global_num_rows, hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(b);
   hypre_ParVectorSetPartitioningOwner(b, 0);*/

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

   HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_x);
   HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_x);
   
   ierr = HYPRE_IJVectorGetObject(ij_x, &object);
   x = (hypre_ParVector *)object;

   HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_xt);
   HYPRE_IJVectorSetObjectType(ij_xt, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_xt);
   
   ierr = HYPRE_IJVectorGetObject(ij_xt, &object);
   xt = (hypre_ParVector *)object;

   HYPRE_IJVectorCreate(comm, first_local_col, last_local_col, &ij_xt2);
   HYPRE_IJVectorSetObjectType(ij_xt2, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_xt2);
   
   ierr = HYPRE_IJVectorGetObject(ij_xt, &object);
   xt2 = (hypre_ParVector *)object;

   ///////////////////////////////////////////////////////////////////////
   // Set initial guess for e to 1 + rand[0, 1/2] for all entries in e. //
   //                                                                   //
   /*x = hypre_ParVectorCreate(comm, global_num_rows, hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(x);
   hypre_ParVectorSetPartitioningOwner(x, 0);*/

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

   rho_cr = 1;

   double alpha, beta;

   while(rho_cr >= RHO_LIMIT) {
     /////////////////////////////////////////////////////////////////////
     // Perform 5 CR iterations (relaxation on the F-points only with   //
     // C-point values set to zero). Compute rho_cr.                    //
     //                                                                 //
     hypre_ParVectorCopy(x_init, x);
     for(i = 0; i < 5; i++) {
       Solve_err_flag = hypre_BoomerAMGRelax(A, b, CF_marker, relax_type, F_POINT,
					     relax_weight[level], omega[level], x,
					     Vtemp);
       if(i == 3) {
	 // Compute norm of e_f for fourth iteration. e_f is the norm of
	 // only the fine level nodes, but since the C-points have a value
	 // of zero, a normal inner product will give the same results.
	 // However, this should be written to utilize this information
	 // in the future.
	 //e_f_prev = sqrt(hypre_ParVectorInnerProd(x, x));
	 hypre_ParVectorCopy(x, xt);
	 hypre_ParVectorCopy(b, xt2);
	 alpha = 1.0;
	 beta = 0;
	 hypre_ParCSRMatrixMatvec(alpha, A, x, beta, xt);
	 e_f_prev = sqrt(hypre_ParVectorInnerProd(xt, x));
       }
     }
     // Compute norm of e_f for last iteration.
     //e_f_curr = sqrt(hypre_ParVectorInnerProd(x, x));
     hypre_ParVectorCopy(x, xt);
     hypre_ParVectorCopy(b, xt2);
     alpha = 1.0;
     beta = 0;
     hypre_ParCSRMatrixMatvec(alpha, A, x, beta, xt);
     e_f_curr = sqrt(hypre_ParVectorInnerProd(xt, x));

     rho_cr = e_f_curr / e_f_prev;
     //                                                                 //
     /////////////////////////////////////////////////////////////////////

     //printf("%f (%d): %f\n", rho_cr, my_id, e_f_curr);
     if(rho_cr >= RHO_LIMIT) {
       ///////////////////////////////////////////////////////////////////
       // Then some of the F-points need to be made into C-points.      //
       ///////////////////////////////////////////////////////////////////

       ///////////////////////////////////////////////////////////////////
       // First find the largest magnitude value in e_f_curr. [NOTE:    //
       // this can be done more efficiently by finding it when the norm //
       // is computed above because all of the values are examined at   //
       // that time.]                                                   //
       //                                                               //
       local_max_e_f = 0;
       local_x = hypre_ParVectorLocalVector(x);
       local_x_data = hypre_VectorData(local_x);
       for(i = 0; i < local_num_rows; i++) {
	 if(fabs(local_x_data[i]) > local_max_e_f)
	   local_max_e_f = fabs(local_x_data[i]);
       }

       // Now do global reduction of local_max_e_f and store it in
       // max_e_f.
       MPI_Allreduce(&local_max_e_f, &max_e_f, 1, MPI_DOUBLE, MPI_MAX, comm);
       //                                                               //
       ///////////////////////////////////////////////////////////////////

       // Form a set of coarse grid candidates such that:
       //    local_max_e_f / max_e_f >= 1 - rho_cr.
       rho_cr_m1 = 1 - rho_cr;
       num_candidates = 0;
       for(i = 0; i < local_num_rows; i++) {
	 candidate_measures[i] = fabs(local_x_data[i]) / max_e_f;
	 if(candidate_measures[i] >= rho_cr_m1) {
	   U[i] = 1;
	   temp_CF_marker[i] = F_POINT;
	   num_candidates++;
	 }
	 else
	   U[i] = 0;
       }

       // Store all of the subgraph information in a new data structure.
       //A_ff = extractSubgraph(A, U, candidate_measures, local_num_rows, num_candidates);

       // Add independent set of U to coarse variable set.
       //addIndependentSet_Brannick(A_ff, temp_CF_marker, local_num_rows, num_candidates);
       if(coarsen_type == 30)
	 //hypre_BoomerAMGCoarsenCLJP_c(S, A, 1, debug_flag, &temp_CF_marker, 0, level, NULL);
	 hypre_BoomerAMGCoarsenCLJP_c(S, A, 1, debug_flag, &temp_CF_marker, 0, level, measure_array);
       else if(coarsen_type == 31) {
printf("in\n");
	 hypre_BoomerAMGCoarsenPMIS_c(S, A, 1, debug_flag, &temp_CF_marker, 0, level, 1,
				      measure_array);
printf("out\n");
       }
       else if(coarsen_type == 32)
	 hypre_BoomerAMGCoarsen(S, A, 1, debug_flag, &temp_CF_marker);
       else if(coarsen_type == 33)
	 hypre_BoomerAMGCoarsenPMIS(S, A, 1, debug_flag, &temp_CF_marker);

       // Deallocate subgraph memory.
       //destroySubgraphInformation(A_ff, num_candidates);

       // Set left hand side to zero for all of the C-points.
       local_x = hypre_ParVectorLocalVector(x_init);
       local_x_data = hypre_VectorData(local_x);
       for(i = 0; i < local_num_rows; i++) {
	 if(temp_CF_marker[i] == C_POINT) {
	   local_x_data[i] = 0;
	   CF_marker[i] = C_POINT;
	 }
	 temp_CF_marker[i] = SF_POINT;
       }
     }
   }

   // Write candidate measures for visualization.
   //local_x = hypre_ParVectorLocalVector(x);
   //local_x_data = hypre_VectorData(local_x);
   //WriteCRRates(A, local_x_data, level);
 
   hypre_TFree(U);
   hypre_TFree(candidate_measures);
   hypre_TFree(temp_CF_marker);
   if(color_array)
     hypre_TFree(color_array);
   //hypre_TFree(measure_array); // already being done by CLJP_c function

   int total_C_count;
   int C_count = 0;
   for(i = 0; i < local_num_rows; i++) {
     //printf("%d: %d\n", i, CF_marker_array[level][i]);
     if(CF_marker[i] == C_POINT)
       C_count++;
   }

   MPI_Allreduce(&C_count, &total_C_count, 1, MPI_INT, MPI_SUM, comm);
   if(my_id == 0)
     printf("\nLevel %d, number of C-points: %d", level, total_C_count);

   *CF_marker_ptr = CF_marker;
   return total_C_count;
}
