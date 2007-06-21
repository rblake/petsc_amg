#include "headers.h"
#include "limits.h"

#define C_PT  1
#define F_PT -1
#define SF_PT -3
#define COMMON_C_PT  2
#define Z_PT -2

// This struct is used in the parallel graph coloring routine (parColorGraph).
// Its purpose is to store information about all of the off-processor nodes.
// It stores the color of a node and also a list of that node's on-processor
// neighbors.
struct offd_node_color_and_neighbors {
  double rho; // random number used in the coloring algorithm to determine
           // precedence
  int color;
  int proc;
  hypre_Queue * ond_neighbor_list;
};
typedef struct offd_node_color_and_neighbors offdNodeAndNeighbors;

double hash(unsigned int key)
{
  key += ~(key << 15);
  key ^=  (key >> 10);
  key +=  (key << 3);
  key ^=  (key >> 6);
  key += ~(key << 11);
  key ^=  (key >> 16);
  return ((double)key)/UINT_MAX;
}

void setRhoValues(double * rho_array, int * V_i_S, int V_i_S_size, 
		  int V_ghost_S_size, hypre_ParCSRMatrix * S, int my_id)
{
  int i;
  int * V_ghost_S = hypre_ParCSRMatrixColMapOffd(S);
  int * row_starts = hypre_ParCSRMatrixRowStarts(S);

  // First, initialize the rho values for the on-processor nodes.
  for(i = 0; i < V_i_S_size; i++)
    rho_array[i] = hash(V_i_S[i]+row_starts[my_id]);

  // Second, initialize the rho values for the off-processor nodes.
  for(i = 0; i < V_ghost_S_size; i++)
    rho_array[i+V_i_S_size] = hash(V_ghost_S[i]);
}

void setRhoValuesOnd(double * ond_rho, int ond_to_offd_count, int * V_i_S,
		     int row_starts)
{
  int i, rand_index;

  rand_index = 0;
  hypre_SeedRand(2747);
  for (i = 0; i < ond_to_offd_count; i++) {
    while(V_i_S[i]+row_starts > rand_index) {
      hypre_Rand();
      rand_index++;
    }
    ond_rho[i] = hypre_Rand();
    rand_index++;
  }
}

void setRhoValuesOffd(offdNodeAndNeighbors * offd_rho, int offd_count, int * S_offd_map)
{
  int i, rand_index;

  rand_index = 0;
  hypre_SeedRand(2747);
  for (i = 0; i < offd_count; i++) {
    while(S_offd_map[i] > rand_index) {
      hypre_Rand();
      rand_index++;
    }
    offd_rho[i].rho = hypre_Rand();
    rand_index++;
  }
}

void colorQueue(hypre_ParCSRMatrix * S, short * color_array,
		offdNodeAndNeighbors * offd_rho, hypre_Queue * color_queue,
      	        int * num_colors, hypre_Queue ** send_queue, int * proc_list,
	        int ** proc_send_list, int * proc_send_list_ind,
		int proc_count, int * packed, int * V_i_S)
{
  hypre_CSRMatrix *S_diag, *S_offd;
  int *S_diag_i, *S_diag_j, *S_offd_i, *S_offd_j, *data, *used_colors;
  int i, my_id, row_start, n_colored, curr_node, V_i_S_ind;
  int used_colors_ind, curr_choice, proc_ind;
  MPI_Comm comm = hypre_ParCSRMatrixComm(S);

  MPI_Comm_rank(comm,&my_id);

  S_diag = hypre_ParCSRMatrixDiag(S);
  S_diag_i = hypre_CSRMatrixI(S_diag);
  S_diag_j = hypre_CSRMatrixJ(S_diag);

  S_offd = hypre_ParCSRMatrixOffd(S);
  S_offd_i = hypre_CSRMatrixI(S_offd);
  S_offd_j = hypre_CSRMatrixJ(S_offd);

  row_start = hypre_ParCSRMatrixRowStarts(S)[my_id];

  for(i = 0; i < proc_count; i++)
    packed[i] = 0;

  n_colored = 0;
  // Color each of the nodes in the color_queue.
  while(color_queue->head) {
    data = dequeue(color_queue);
    V_i_S_ind = *data;
    curr_node = V_i_S[V_i_S_ind];
    hypre_TFree(data);

    used_colors = hypre_CTAlloc(int, S_diag_i[curr_node+1]-S_diag_i[curr_node] + S_offd_i[curr_node+1]-S_offd_i[curr_node]);
    used_colors_ind = 0;
    // Check the on-processor neighbors.
    for(i = S_diag_i[curr_node]; i < S_diag_i[curr_node+1]; i++) {
      if(color_array[S_diag_j[i]] > 0) {
	// Then this node has been colored.
	used_colors[used_colors_ind] = color_array[S_diag_j[i]];
	used_colors_ind++;
      }
    }

    // Check the off-processor neighbors.
    for(i = S_offd_i[curr_node]; i < S_offd_i[curr_node+1]; i++) {
      if(offd_rho[S_offd_j[i]].color > 0) {
	// Then this node has been colored.
	used_colors[used_colors_ind] = offd_rho[S_offd_j[i]].color;
	used_colors_ind++;
      }
    }

    qsort0(used_colors, 0, used_colors_ind-1);

    // Now search the used_colors array to find the lowest number > 0 that
    // does not appear. Make that number the color of curr_node.
    curr_choice = 1;
    for(i = 0; i < used_colors_ind; i++) {
      if(used_colors[i] == curr_choice)
	// Then the current choice of color is in use. Pick the next one up.
	curr_choice++;
      else if(used_colors[i] > curr_choice)
	// The current choice of color is available. Exit the loop and color
	// curr_node this color.
	i = used_colors_ind; // to break loop
    }
    color_array[curr_node] = curr_choice;
    if(curr_choice > *num_colors)
      *num_colors = curr_choice;
    hypre_TFree(used_colors);

    // Now pack up the sends for this node.
    while(send_queue[V_i_S_ind]->head) {
      data = dequeue(send_queue[V_i_S_ind]);
      proc_ind = hypre_BinarySearch(proc_list, data[1], proc_count);
      proc_send_list[proc_ind][proc_send_list_ind[proc_ind]] = data[0];
      proc_send_list[proc_ind][proc_send_list_ind[proc_ind]+1] = curr_node+row_start;
      proc_send_list[proc_ind][proc_send_list_ind[proc_ind]+2] = curr_choice;
      proc_send_list_ind[proc_ind] += 3;
      packed[proc_ind]++;

      hypre_TFree(data);
    }
  }
}

int procLookup(hypre_ParCSRMatrix * S, int node, int prev_proc)
{
  int ind;
  int proc = 0;
  int * row_starts = hypre_ParCSRMatrixRowStarts(S);

  if(row_starts[prev_proc] <= node && row_starts[prev_proc+1] > node)
    return prev_proc;

  ind = 1;
  while(row_starts[ind] <= node) {
    proc++;
    ind++;
  }
  return proc;
}

int waitforLoopData(hypre_ParCSRCommHandle * comm_handle)
{
  MPI_Status *status0;
  int ierr = 0;

  if ( comm_handle==NULL ) return ierr;
  if (hypre_ParCSRCommHandleNumRequests(comm_handle)) {
    status0 = hypre_CTAlloc(MPI_Status,
			    hypre_ParCSRCommHandleNumRequests(comm_handle));
    MPI_Waitall(hypre_ParCSRCommHandleNumRequests(comm_handle),
		hypre_ParCSRCommHandleRequests(comm_handle), status0);
    hypre_TFree(status0);
  }

  hypre_TFree(hypre_ParCSRCommHandleRequests(comm_handle));
  hypre_TFree(comm_handle);
}

int sendLoopData(hypre_ParCSRCommPkg * comm_pkg,
		 int * color_send_buf,
		 int finished,
		 int local_num_colors,
		 int * ghost_color_array,
		 int * neighborhood_num_colors,
		 int * num_finished_neighbors,
		 int * finished_neighbors_array)
{
  int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
  MPI_Comm comm = hypre_ParCSRCommPkgComm(comm_pkg);

  hypre_ParCSRCommHandle *comm_handle;
  int num_requests;
  MPI_Request *requests;
  int * neighbor_num_colors = hypre_CTAlloc(int, num_recvs);

  int i, j;
  int ip, vec_start, vec_len, ierr;

  num_requests = (4*(num_sends-num_finished_neighbors[0])) +
    (4*(num_recvs-num_finished_neighbors[1]));
  requests = hypre_CTAlloc(MPI_Request, num_requests);
  j = 0;

  for (i = 0; i < num_recvs; i++) {
    if(!finished_neighbors_array[i+num_sends]) {
      ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i); 
      vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i);
      vec_len = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
      MPI_Irecv(&ghost_color_array[vec_start], vec_len, MPI_INT,
		ip, 0, comm, &requests[j++]);
      MPI_Irecv(&finished_neighbors_array[i+num_sends], 1, MPI_INT, ip, 1, comm,
		&requests[j++]);
      MPI_Irecv(&neighbor_num_colors[i], 1, MPI_INT, ip, 2, comm, &requests[j++]);
      MPI_Isend(&finished, 1, MPI_INT, ip, 3, comm, &requests[j++]);
    }
  }
  for (i = 0; i < num_sends; i++) {
    if(!finished_neighbors_array[i]) {
      vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      vec_len = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1)-vec_start;
      ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i); 
      MPI_Isend(&color_send_buf[vec_start], vec_len, MPI_INT,
		ip, 0, comm, &requests[j++]);
      MPI_Isend(&finished, 1, MPI_INT, ip, 1, comm, &requests[j++]);
      MPI_Isend(&local_num_colors, 1, MPI_INT, ip, 2, comm, &requests[j++]);
      MPI_Irecv(&finished_neighbors_array[i], 1, MPI_INT, ip, 3, comm,
		&requests[j++]);
    }
  }

  comm_handle = hypre_CTAlloc(hypre_ParCSRCommHandle, 1);
  hypre_ParCSRCommHandleCommPkg(comm_handle) = comm_pkg;
  hypre_ParCSRCommHandleNumRequests(comm_handle) = num_requests;
  hypre_ParCSRCommHandleRequests(comm_handle) = requests;

  ierr = waitforLoopData(comm_handle);

  // Process the received information.
  num_finished_neighbors[0] = 0;
  num_finished_neighbors[1] = 0;
  for(i = 0; i < num_recvs; i++) {

    // Check if neighbors are finished.
    if(finished_neighbors_array[i+num_sends])
      num_finished_neighbors[1]++;

    // Check if neighbor's num_colors is greatest known num_colors.
    if(*neighborhood_num_colors < neighbor_num_colors[i])
      *neighborhood_num_colors = neighbor_num_colors[i];
  }
  for(i = 0; i < num_sends; i++) {
    // Check if neighbors are finished.
    if(finished_neighbors_array[i])
      num_finished_neighbors[0]++;
  }

  hypre_TFree(neighbor_num_colors);

  return ierr;
}

void seqColorGraphNewAgain(hypre_ParCSRMatrix * S, short * color_array, int * num_colors, int * V_i_S, int V_i_S_len, int level);  /////////DELETE ME!!!!!!!!!!!!!!!

void seqColorGraphTheFinalWord(hypre_ParCSRMatrix * S, short * color_array,
			       int * num_colors, int num_colored, int level)
{
#define QUEUED        -1
#define UNCOLORED     0
#define IDO           0
  // Get pointers to parts of the strength matrix
  hypre_CSRMatrix *S_diag, *S_offd;
  int *S_diag_i, *S_diag_j, *S_offd_i, *S_offd_j, *S_offd_map;
  S_diag = hypre_ParCSRMatrixDiag(S);
  S_diag_i = hypre_CSRMatrixI(S_diag);
  S_diag_j = hypre_CSRMatrixJ(S_diag);
  S_offd = hypre_ParCSRMatrixOffd(S);
  S_offd_i = hypre_CSRMatrixI(S_offd);
  S_offd_j = hypre_CSRMatrixJ(S_offd);
  S_offd_map = hypre_ParCSRMatrixColMapOffd(S);

  int num_variables = hypre_CSRMatrixNumRows(S_diag);

  int i, j, neighborID;
  int degree, max_degree, max_degree_node;
  hypre_Queue * buckets, colored_bucket;
  hypre_QueueElement * bucket_elements;
  int * bucket_element_data, *colored_neighbors_degree, *now_coloring;
  int max_colored_degree = 0, *used_colors, color_assigned;

  if(num_colored < num_variables) {
  // 1. Find maximum degree in graph of S. Keep track of the max
  //    degree node.
  max_degree = 0;
  max_degree_node = 0;
  for(i = 0; i < num_variables; i++) {
    if(color_array[i] == UNCOLORED) {
      degree = S_diag_i[i+1] - S_diag_i[i] - 1;
      if(degree > max_degree) {
	max_degree = degree;
	max_degree_node = i;
      }
    }
  }

  // 2. Declare and initialize max_degree+1 buckets.
  buckets = hypre_CTAlloc(hypre_Queue, max_degree+1);
  for(i = 0; i < max_degree+1; i++)
    initializeQueue(&buckets[i]);
  initializeQueue(&colored_bucket);

  // 3. Declare and initialize num_variable number of
  //    hypre_QueueElements. The data for each element is its variable
  //    number.
  //
  //    Toss each of the queue elements corresponding to an uncolored
  //    node into bucket zero. Just to keep track of the colored
  //    nodes, toss them into bucket one. They will all be removed in
  //    Step 4.
  bucket_elements = hypre_CTAlloc(hypre_QueueElement, num_variables);
  bucket_element_data = hypre_CTAlloc(int, num_variables);
  colored_neighbors_degree = hypre_CTAlloc(int, num_variables);
  // colored_neighbors_degree is used to keep track of which bucket an
  // element is in

  num_colored = 0;
  for(i = 0; i < num_variables; i++) {
    bucket_element_data[i] = i;
    bucket_elements[i].data = &bucket_element_data[i];
    if(color_array[i] == 0) {
      enqueueElement(&bucket_elements[i], &buckets[0]);
      colored_neighbors_degree[i] = 0;
    }
    else {
      enqueueElement(&bucket_elements[i], &colored_bucket);      
      colored_neighbors_degree[i] = -1;
      num_colored++;
    }
  }

  // 4. Process colored nodes. The colored nodes have all been
  //    temporarily put into bucket one.
  if(IDO) {
  for(i = 0; i < num_colored; i++) {
  //while(buckets[1].head) {
    now_coloring = dequeue(&colored_bucket);
    // Node referred to by now_coloring is already colored, but its
    // neighbors need to be processed.
    //for(j = S_diag_i[*now_coloring]+1; j < S_diag_i[(*now_coloring)+1]; j++) {
    for(j = S_diag_i[*now_coloring]; j < S_diag_i[(*now_coloring)+1]; j++) {
      // Find neighbors and move them into the next bucket up.
      neighborID = S_diag_j[j];
      if(colored_neighbors_degree[neighborID] > -1) {
	removeElement(&bucket_elements[neighborID],
		      &buckets[colored_neighbors_degree[neighborID]]);
	colored_neighbors_degree[neighborID]++;
	pushElement(&bucket_elements[neighborID],
		       &buckets[colored_neighbors_degree[neighborID]]);
	if(colored_neighbors_degree[neighborID] > max_colored_degree)
	  // Keep track of the largest index bucket with elements in it.
	  max_colored_degree = colored_neighbors_degree[neighborID];
      }
    }
  }
  }

  // 5. Begin IDO algorithm:
  //    If no nodes have been colored, start with max_degree_node by
  //    moving it to front of the bucket zero queue.
 if(max_colored_degree == 0)
   moveToHead(&bucket_elements[max_degree_node], &buckets[0]);

  while(num_variables > num_colored) {
    // Grab node from non-empty bucket with largest index.
    now_coloring = dequeue(&buckets[max_colored_degree]);

    if(*num_colors == 0) {
      // Trivial case. Nothing has been colored yet, so this
      // automatically becomes "color" 1.
      color_array[*now_coloring] = 1;
      *num_colors = 1;
    }
    else {
      // Color this node based on its neighbors.
      used_colors = hypre_CTAlloc(int, *num_colors);
      for(j = S_diag_i[*now_coloring]; j < S_diag_i[*now_coloring+1]; j++) {
	// check on-processor neighbors
	if(color_array[S_diag_j[j]] != UNCOLORED) {
	  used_colors[color_array[S_diag_j[j]]-1] = 1;
	}
      }

      // Now all used colors have been found. Identify the
      // smallest available color and assign it.
      color_assigned = 0;
      for(j = 0; j < *num_colors; j++) {
	if(used_colors[j] == 0) {
	  // Assign this color.
	  color_array[*now_coloring] = j+1;
	  color_assigned = 1;
	  j = *num_colors; // <-- to break the loop
	}
      }
      if(!color_assigned) { // then a new largest color is needed
	color_array[*now_coloring] = *num_colors + 1;
	*num_colors = *num_colors + 1;
      }
      hypre_TFree(used_colors);
    }
    colored_neighbors_degree[*now_coloring] = -1;
    num_colored++;

    // Update neighbors of node referred to by now_coloring.
    if(IDO) {
    //for(j = S_diag_i[*now_coloring]+1; j < S_diag_i[(*now_coloring)+1]; j++) {
    for(j = S_diag_i[*now_coloring]; j < S_diag_i[(*now_coloring)+1]; j++) {
      // Find neighbors and move them into the next bucket up.
      neighborID = S_diag_j[j];
      if(colored_neighbors_degree[neighborID] > -1) {
	removeElement(&bucket_elements[neighborID],
		      &buckets[colored_neighbors_degree[neighborID]]);
	colored_neighbors_degree[neighborID]++;
	pushElement(&bucket_elements[neighborID],
		       &buckets[colored_neighbors_degree[neighborID]]);
	if(colored_neighbors_degree[neighborID] > max_colored_degree)
	  // Keep track of the largest index bucket with elements in it.
	  max_colored_degree = colored_neighbors_degree[neighborID];
      }
    }

    // Make sure this bucket still has elements; if not, find next
    // non-empty bucket.
    if(!buckets[max_colored_degree].head) {
      // Search for non-empty bucket (will have lower index).
      for(i = max_colored_degree-1; i > -1; i--) {
	if(buckets[i].head) {
	  max_colored_degree = i;
	  i = -1; // <-- to break loop
	}
      }
    }
    }
  }

  hypre_TFree(colored_neighbors_degree);
  hypre_TFree(bucket_element_data);
  hypre_TFree(bucket_elements);
  hypre_TFree(buckets);
  }

  WriteCoarseningColorInformation(S, color_array, level);

/* for(i = 0; i < num_variables; i++) { */
/*   printf("%d:", i); */
/*   for(j = S_diag_i[i]; j < S_diag_i[i+1]; j++) */
/*     printf(" %d", S_diag_j[j]); */
/*   printf("\n"); */
/* } */
/* for(i = 0; i < num_variables; i++) { */
/*   for(j = S_diag_i[i]+1; j < S_diag_i[i+1]; j++) { */
/*     if(color_array[i] == color_array[S_diag_j[j]]) { */
/*       printf("TROUBLE %d %d %d %d!!\n", i, S_diag_j[j], color_array[i], color_array[S_diag_j[j]]); */
/*       exit(0); */
/*     } */
/*   } */
/* } */
//printf("leaving\n");
}

void parColorGraphNewBoundariesOnly(hypre_ParCSRMatrix * S, hypre_ParCSRMatrix * A,
				    short * color_array, int * num_colors)
{
#define QUEUED        -1
#define UNCOLORED     0

  hypre_ParCSRCommHandle * comm_handle;
  hypre_ParCSRCommPkg * comm_pkg = hypre_ParCSRMatrixCommPkg(S);
  MPI_Comm comm = hypre_ParCSRMatrixComm(S);

  // Get pointers to parts of the strength matrix
  hypre_CSRMatrix *S_diag, *S_offd;
  int *S_diag_i, *S_diag_j, *S_offd_i, *S_offd_j, *S_offd_map;
  S_diag = hypre_ParCSRMatrixDiag(S);
  S_diag_i = hypre_CSRMatrixI(S_diag);
  S_diag_j = hypre_CSRMatrixJ(S_diag);
  S_offd = hypre_ParCSRMatrixOffd(S);
  S_offd_i = hypre_CSRMatrixI(S_offd);
  S_offd_j = hypre_CSRMatrixJ(S_offd);
  S_offd_map = hypre_ParCSRMatrixColMapOffd(S);
  int * V_i_S;
  int V_i_S_size, V_ghost_S_size, color_send_buf_size;
  int num_finished_neighbors[2], *finished_neighbors_array;

  int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
  finished_neighbors_array = hypre_CTAlloc(int, num_sends+num_recvs);
  num_finished_neighbors[0] = 0; // holds number of neighbors this
				 // processor sends to who are
				 // finished
  num_finished_neighbors[1] = 0; // holds number of neighbors this
				 // processor receives from who are
				 // finished
  int my_id;

  int i, j, k, row_index, local_num_colors, num_colored, *used_colors, nodeID;
  short ready_to_color, color_assigned;

  double * rho_array;
  int *ghost_color_array, *color_send_buf, *color_send_map;

  MPI_Comm_rank(comm, &my_id);
  int finished;
  local_num_colors = 0;
  *num_colors = 0;

int num_variables = hypre_CSRMatrixNumRows(S_offd);

  // Initialize the pseudo-random numbers on V_i^S.
  //    First, determine nodes and number of nodes in V_i^S.
  //V_i_S_size = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) -
  //  hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);

  V_i_S_size = 0;
  for(i = 0; i < num_variables; i++) {
    if(S_offd_i[i] != S_offd_i[i+1])
      V_i_S_size++;
  }
  V_i_S = hypre_CTAlloc(int, V_i_S_size);
  V_i_S_size = 0;
  for(i = 0; i < num_variables; i++) {
    if(S_offd_i[i] != S_offd_i[i+1]) {
      V_i_S[V_i_S_size] = i;
      V_i_S_size++;
    }
  }




/*   V_i_S_size = hypre_CSRMatrixNumRownnz(S_offd); // Number of empty */
/* 						 // rows in S_offd. */
/*   hypre_CSRMatrixSetRownnz(S_offd); */
/*   V_i_S = hypre_CSRMatrixRownnz(S_offd); */

  V_ghost_S_size = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs) -
    hypre_ParCSRCommPkgRecvVecStart(comm_pkg, 0);
  // Is the above the same as hypre_CSRMatrixNumCols(S_offd)?

  //    Second, declare array to hold pseudo-random numbers, including those
  //    from the ghost points.
  rho_array = hypre_CTAlloc(double, V_i_S_size+V_ghost_S_size);

  //    Third, initialize the pseudo-random numbers corresponding to nodes on
  //    this processor and the nodes on neighboring processors.
  //
  //    Since each processor can compute the "random" number for all
  //    of the off-processor nodes, no communication is needed to
  //    exchange rho values.
  setRhoValues(rho_array, V_i_S, V_i_S_size, V_ghost_S_size, S, my_id);

  // Initialize the ghost_color_array. hypre_CTAlloc automatically
  // initializes each entry to be UNCOLORED. This stores the colors of
  // the ghost points.
  ghost_color_array = hypre_CTAlloc(int, V_ghost_S_size);

  // Initialize the color_send_buf. This is where color values to be
  // sent to other processors are gathered.
  color_send_buf_size = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) -
    hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
  color_send_buf = hypre_CTAlloc(int, color_send_buf_size);

  // Continue looping until all local nodes in V_i^S are colored.
  num_colored = 0;
  finished = 0;
  while(num_colored < V_i_S_size) {
    //MPI_Allreduce(&finished, &global_finished, 1, MPI_INT, MPI_MIN, comm);
    for(i = 0; i < V_i_S_size; i++) {
      nodeID = V_i_S[i];
      if(color_array[nodeID] == UNCOLORED) {
	// Check if this node is now eligible to be colored.
	// This depends solely on the weights of the neighboring ghost
	// points.
	ready_to_color = 1;
	for(j = S_offd_i[nodeID]; j < S_offd_i[nodeID+1]; j++) {
	  if(ghost_color_array[S_offd_j[j]] == UNCOLORED &&
	     rho_array[S_offd_j[j]+V_i_S_size] > rho_array[i]) {
	    // Then i is not ready to be colored yet.
	    ready_to_color = 0;
	    j = S_offd_i[nodeID+1]; // <-- to break the loop
	  }
	}

	if(ready_to_color) {
	  // Color this node.
	  if(*num_colors == 0) {
	    // Trivial case. Nothing has been colored yet, so this
	    // automatically becomes "color" 1.
	    color_array[nodeID] = 1;
	    local_num_colors = 1;
	    *num_colors = 1;
	  }
	  else {
	    // Figure out the colors of i's neighbors (on- and
	    // off-processor).
	    used_colors = hypre_CTAlloc(int, *num_colors);
	    for(j = S_diag_i[nodeID]; j < S_diag_i[nodeID+1]; j++) {
	      // check on-processor neighbors
	      if(color_array[S_diag_j[j]] != UNCOLORED) {
		used_colors[color_array[S_diag_j[j]]-1] = 1;
	      }
	    }

	    for(j = S_offd_i[nodeID]; j < S_offd_i[nodeID+1]; j++) {
	      // check off-processor neighbors
	      if(ghost_color_array[S_offd_j[j]] != UNCOLORED) {
		used_colors[ghost_color_array[S_offd_j[j]]-1] = 1;
	      }
	    }

	    // Now all used colors have been found. Identify the
	    // smallest available color and assign it.
	    color_assigned = 0;
	    for(j = 0; j < *num_colors; j++) {
	      if(used_colors[j] == 0) {
		// Assign this color.
		color_array[nodeID] = j+1;
		if(j+1 > local_num_colors)
		  local_num_colors = j+1;
		color_assigned = 1;
		j = *num_colors; // <-- to break the loop
	      }
	    }
	    if(!color_assigned) { // then a new largest color is
				  // needed
	      color_array[nodeID] = *num_colors + 1;
	      *num_colors = *num_colors + 1;
	      local_num_colors = *num_colors;
	    }

	    hypre_TFree(used_colors);
	  }
	  num_colored++;
	}
      }
    }
    // Exchange color information with neighboring processors. Load
    // information into the color_send_buf first.
    color_send_map = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
    for(i = 0; i < color_send_buf_size; i++) {
      color_send_buf[i] = color_array[color_send_map[i]];
    }

    if(num_colored >= V_i_S_size)
      finished = 1;
    sendLoopData(comm_pkg, color_send_buf, finished, local_num_colors,
		 ghost_color_array, num_colors, num_finished_neighbors,
		 finished_neighbors_array);


    //comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, color_send_buf,
    //ghost_color_array);
    //hypre_ParCSRCommHandleDestroy(comm_handle);
    //MPI_Allreduce(&local_num_colors, num_colors, 1, MPI_INT, MPI_MAX, comm);
  }
  finished = 1;
  //MPI_Allreduce(&finished, &global_finished, 1, MPI_INT, MPI_MIN, comm);

  //while(!neighborhood_finished) { // some processors still need the
			    // information sent
/*     sendLoopData(comm_pkg, color_send_buf, finished, local_num_colors, */
/* 		 ghost_color_array, &neighborhood_finished, num_colors, */
/* 		 &num_finished_neighbors, finished_neighbors_array); */

    //comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, color_send_buf,
    //ghost_color_array);
    //hypre_ParCSRCommHandleDestroy(comm_handle);
    //MPI_Allreduce(&local_num_colors, num_colors, 1, MPI_INT, MPI_MAX, comm);
    //MPI_Allreduce(&finished, &global_finished, 1, MPI_INT, MPI_MIN, comm);
    //}
  //*num_colors = local_num_colors;
  MPI_Allreduce(num_colors, &local_num_colors, 1, MPI_INT, MPI_MAX, comm);

  hypre_TFree(rho_array);
  hypre_TFree(color_send_buf);
  hypre_TFree(finished_neighbors_array);
  hypre_TFree(ghost_color_array);
}

void parColorGraphNew(hypre_ParCSRMatrix * S, hypre_ParCSRMatrix * A,
		      short * color_array, int * num_colors, int level)
{
#define QUEUED        -1
#define UNCOLORED     0

  hypre_ParCSRCommHandle * comm_handle;
  hypre_ParCSRCommPkg * comm_pkg = hypre_ParCSRMatrixCommPkg(S);
  MPI_Comm comm = hypre_ParCSRMatrixComm(S);

  // Get pointers to parts of the strength matrix
  hypre_CSRMatrix *S_diag, *S_offd;
  int *S_diag_i, *S_diag_j, *S_offd_i, *S_offd_j, *S_offd_map;
  S_diag = hypre_ParCSRMatrixDiag(S);
  S_diag_i = hypre_CSRMatrixI(S_diag);
  S_diag_j = hypre_CSRMatrixJ(S_diag);
  S_offd = hypre_ParCSRMatrixOffd(S);
  S_offd_i = hypre_CSRMatrixI(S_offd);
  S_offd_j = hypre_CSRMatrixJ(S_offd);
  S_offd_map = hypre_ParCSRMatrixColMapOffd(S);
  int * V_i_S;
  int V_i_S_size, V_ghost_S_size, color_send_buf_size;
  int num_finished_neighbors[2], *finished_neighbors_array;

  int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
  finished_neighbors_array = hypre_CTAlloc(int, num_sends+num_recvs);
  num_finished_neighbors[0] = 0; // holds number of neighbors this
				 // processor sends to who are
				 // finished
  num_finished_neighbors[1] = 0; // holds number of neighbors this
				 // processor receives from who are
				 // finished
  int my_id;

  int i, j, k, row_index, local_num_colors, num_colored, *used_colors, nodeID;
  short ready_to_color, color_assigned;

  double * rho_array;
  int *ghost_color_array, *color_send_buf, *color_send_map;

  MPI_Comm_rank(comm, &my_id);
  int finished;
  local_num_colors = 0;
  *num_colors = 0;

int num_variables = hypre_CSRMatrixNumRows(S_offd);

  // Initialize the pseudo-random numbers on V_i^S.
  //    First, determine nodes and number of nodes in V_i^S.
  //V_i_S_size = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) -
  //  hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);

  V_i_S_size = 0;
  for(i = 0; i < num_variables; i++) {
    if(S_offd_i[i] != S_offd_i[i+1])
      V_i_S_size++;
  }
  V_i_S = hypre_CTAlloc(int, V_i_S_size);
  V_i_S_size = 0;
  for(i = 0; i < num_variables; i++) {
    if(S_offd_i[i] != S_offd_i[i+1]) {
      V_i_S[V_i_S_size] = i;
      V_i_S_size++;
    }
  }




/*   V_i_S_size = hypre_CSRMatrixNumRownnz(S_offd); // Number of empty */
/* 						 // rows in S_offd. */
/*   hypre_CSRMatrixSetRownnz(S_offd); */
/*   V_i_S = hypre_CSRMatrixRownnz(S_offd); */

  V_ghost_S_size = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs) -
    hypre_ParCSRCommPkgRecvVecStart(comm_pkg, 0);
  // Is the above the same as hypre_CSRMatrixNumCols(S_offd)?

  //    Second, declare array to hold pseudo-random numbers, including those
  //    from the ghost points.
  rho_array = hypre_CTAlloc(double, V_i_S_size+V_ghost_S_size);

  //    Third, initialize the pseudo-random numbers corresponding to nodes on
  //    this processor and the nodes on neighboring processors.
  //
  //    Since each processor can compute the "random" number for all
  //    of the off-processor nodes, no communication is needed to
  //    exchange rho values.
  setRhoValues(rho_array, V_i_S, V_i_S_size, V_ghost_S_size, S, my_id);

  // Initialize the ghost_color_array. hypre_CTAlloc automatically
  // initializes each entry to be UNCOLORED. This stores the colors of
  // the ghost points.
  ghost_color_array = hypre_CTAlloc(int, V_ghost_S_size);

  // Initialize the color_send_buf. This is where color values to be
  // sent to other processors are gathered.
  color_send_buf_size = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) -
    hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
  color_send_buf = hypre_CTAlloc(int, color_send_buf_size);

  // Continue looping until all local nodes in V_i^S are colored.
  num_colored = 0;
  finished = 0;
  while(num_colored < V_i_S_size) {
    //MPI_Allreduce(&finished, &global_finished, 1, MPI_INT, MPI_MIN, comm);
    for(i = 0; i < V_i_S_size; i++) {
      nodeID = V_i_S[i];
      if(color_array[nodeID] == UNCOLORED) {
	// Check if this node is now eligible to be colored.
	// This depends solely on the weights of the neighboring ghost
	// points.
	ready_to_color = 1;
	for(j = S_offd_i[nodeID]; j < S_offd_i[nodeID+1]; j++) {
	  if(ghost_color_array[S_offd_j[j]] == UNCOLORED &&
	     rho_array[S_offd_j[j]+V_i_S_size] > rho_array[i]) {
	    // Then i is not ready to be colored yet.
	    ready_to_color = 0;
	    j = S_offd_i[nodeID+1]; // <-- to break the loop
	  }
	}

	if(ready_to_color) {
	  // Color this node.
	  if(*num_colors == 0) {
	    // Trivial case. Nothing has been colored yet, so this
	    // automatically becomes "color" 1.
	    color_array[nodeID] = 1;
	    local_num_colors = 1;
	    *num_colors = 1;
	  }
	  else {
	    // Figure out the colors of i's neighbors (on- and
	    // off-processor).
	    used_colors = hypre_CTAlloc(int, *num_colors);
	    for(j = S_diag_i[nodeID]; j < S_diag_i[nodeID+1]; j++) {
	      // check on-processor neighbors
	      if(color_array[S_diag_j[j]] != UNCOLORED) {
		used_colors[color_array[S_diag_j[j]]-1] = 1;
	      }
	    }

	    for(j = S_offd_i[nodeID]; j < S_offd_i[nodeID+1]; j++) {
	      // check off-processor neighbors
	      if(ghost_color_array[S_offd_j[j]] != UNCOLORED) {
		used_colors[ghost_color_array[S_offd_j[j]]-1] = 1;
	      }
	    }

	    // Now all used colors have been found. Identify the
	    // smallest available color and assign it.
	    color_assigned = 0;
	    for(j = 0; j < *num_colors; j++) {
	      if(used_colors[j] == 0) {
		// Assign this color.
		color_array[nodeID] = j+1;
		if(j+1 > local_num_colors)
		  local_num_colors = j+1;
		color_assigned = 1;
		j = *num_colors; // <-- to break the loop
	      }
	    }
	    if(!color_assigned) { // then a new largest color is
				  // needed
	      color_array[nodeID] = *num_colors + 1;
	      *num_colors = *num_colors + 1;
	      local_num_colors = *num_colors;
	    }

	    hypre_TFree(used_colors);
	  }
	  num_colored++;
	}
      }
    }
    // Exchange color information with neighboring processors. Load
    // information into the color_send_buf first.
    color_send_map = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
    for(i = 0; i < color_send_buf_size; i++) {
      color_send_buf[i] = color_array[color_send_map[i]];
    }

    if(num_colored >= V_i_S_size)
      finished = 1;
    sendLoopData(comm_pkg, color_send_buf, finished, local_num_colors,
		 ghost_color_array, num_colors, num_finished_neighbors,
		 finished_neighbors_array);


    //comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, color_send_buf,
    //ghost_color_array);
    //hypre_ParCSRCommHandleDestroy(comm_handle);
    //MPI_Allreduce(&local_num_colors, num_colors, 1, MPI_INT, MPI_MAX, comm);
  }
  finished = 1;
  //MPI_Allreduce(&finished, &global_finished, 1, MPI_INT, MPI_MIN, comm);

  //while(!neighborhood_finished) { // some processors still need the
			    // information sent
/*     sendLoopData(comm_pkg, color_send_buf, finished, local_num_colors, */
/* 		 ghost_color_array, &neighborhood_finished, num_colors, */
/* 		 &num_finished_neighbors, finished_neighbors_array); */

    //comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, color_send_buf,
    //ghost_color_array);
    //hypre_ParCSRCommHandleDestroy(comm_handle);
    //MPI_Allreduce(&local_num_colors, num_colors, 1, MPI_INT, MPI_MAX, comm);
    //MPI_Allreduce(&finished, &global_finished, 1, MPI_INT, MPI_MIN, comm);
    //}
  //*num_colors = local_num_colors;
  MPI_Allreduce(num_colors, &local_num_colors, 1, MPI_INT, MPI_MAX, comm);

  hypre_TFree(rho_array);
  hypre_TFree(ghost_color_array);
  hypre_TFree(color_send_buf);
  hypre_TFree(finished_neighbors_array);

  //WriteCoarseningColorInformation(S, color_array, level);
  //seqColorGraphNewAgain(A, color_array, num_colors, V_i_S, V_i_S_size, level); //DELETE ME AND THE DECLARTION ABOVE THIS FUNCTION!!!!!!!!!!!!!!!!!!!!
  seqColorGraphTheFinalWord(S, color_array, num_colors, num_colored, level);
}

/** This routine implements the parallel graph coloring algorithm given in Figure 6 of the Jones and Plassmann paper "A Parallel Graph Coloring Heuristic".
 *
 * This code will use as much of the same notation as the pseudocode in the paper as possible.
 *
 * @param S a hypre_ParCSRMatrix object containing the strength matrix of the linear system A.
 * @param color_array [OUT] an array of shorts containing color of each node.
 * @param num_colors a pointer to an integer storing the total number of colors in the graph.
 */
void parColorGraph(hypre_ParCSRMatrix * S, hypre_ParCSRMatrix * A, short * color_array, int * num_colors, int level)
{
  hypre_CSRMatrix *S_diag, *S_offd;
  int *S_diag_i, *S_diag_j, *S_offd_i, *S_offd_j, *S_offd_map, *global_ind, *local_ind;
  int *temp_local_ind, temp_local_ind_count;
  int *temp_proc_list, proc_count, *proc_list;
  int *proc_send_count_list = NULL;
  int *proc_recv_count_list = NULL;
  int **proc_send_list, *proc_send_list_ind, *packed;
  int **proc_recv_list, *proc_recv_list_ind;
  int *data, *V_i_S, *n_wait;
  int i, j, num_rows, offd_count, local_i, ond_to_offd_count, row_starts;
  int my_id, n_colored, last_proc, msg_size;
  double *ond_rho;
  offdNodeAndNeighbors * offd_rho;
  hypre_Queue *color_queue, **send_queue;
  hypre_QueueElement * curr_el;
  MPI_Request request;
  MPI_Status status;
  MPI_Comm comm = hypre_ParCSRMatrixComm(S);

  MPI_Comm_rank(comm,&my_id);

  S_diag = hypre_ParCSRMatrixDiag(S);
  S_diag_i = hypre_CSRMatrixI(S_diag);
  S_diag_j = hypre_CSRMatrixJ(S_diag);

  S_offd = hypre_ParCSRMatrixOffd(S);
  S_offd_i = hypre_CSRMatrixI(S_offd);
  S_offd_j = hypre_CSRMatrixJ(S_offd);
  S_offd_map = hypre_ParCSRMatrixColMapOffd(S);

  num_rows = hypre_CSRMatrixNumRows(S_diag);
  offd_count = hypre_CSRMatrixNumCols(S_offd);
  row_starts = hypre_ParCSRMatrixRowStarts(S)[my_id];

  // Begin setting up the data structures to keep track of the off-processor nodes that
  // will be important for the parallel phase of this algorithm.
  //
  // First, initialize the data structure used to store information about the
  // off-processor neighbors.
  offd_rho = hypre_CTAlloc(offdNodeAndNeighbors, offd_count);
  global_ind = hypre_CTAlloc(int, offd_count);
  local_ind = hypre_CTAlloc(int, offd_count);
  temp_proc_list = hypre_CTAlloc(int, offd_count);
  last_proc = 0;
  for(i = 0; i < offd_count; i++) {
    offd_rho[i].rho = 0;
    offd_rho[i].color = 0;
    offd_rho[i].proc = procLookup(S, S_offd_map[i], last_proc);
    last_proc = offd_rho[i].proc;
    temp_proc_list[i] = last_proc;
    offd_rho[i].ond_neighbor_list = newQueue();

    global_ind[i] = S_offd_map[i];
    local_ind[i] = i;
  }

  // After the sorting command below, it will be possible to find the local index
  // using the global node number of an off-processor node by doing a binary search
  // of global_ind. This local index is useful for looking at S_offd columns.
  hypre_qsort2i(global_ind, local_ind, 0, offd_count-1);

  // Gather information on all of the processors that are home to adjacent
  // off-processor nodes.
  qsort0(temp_proc_list, 0, offd_count-1);
  // count the number of unique processors
  proc_count = 0;
  last_proc = -1;
  for(i = 0; i < offd_count; i++) {
    if(last_proc < temp_proc_list[i]) {
      last_proc = temp_proc_list[i];
      proc_count++;
    }
  }
  proc_list = hypre_CTAlloc(int, proc_count);
  proc_count = 0;
  last_proc = -1;
  for(i = 0; i < offd_count; i++) {
    if(last_proc < temp_proc_list[i]) {
      last_proc = temp_proc_list[i];
      proc_list[proc_count] = last_proc;
      proc_count++;
    }
  }
  hypre_TFree(temp_proc_list);
  

  // DELETE THIS WHEN THE WHOLE THING IS DONE AND WORKING PROPERLY.
  for(i = 0; i < offd_count; i++) {
    if(global_ind[i] != S_offd_map[local_ind[i]])
      printf("hey!!!!!\n");
  }
  /////////////////////////////////////////////////////////////

  // Now go through the off-processor connections in the matrix and record to which
  // local node each non-local node connects.
  temp_local_ind = hypre_CTAlloc(int, num_rows); // used to get all of the unique local
                                            // nodes that have off-processor connections
  temp_local_ind_count = 0;
  for(i = 0; i < num_rows; i++) {
    if(S_offd_i[i] < S_offd_i[i+1]) {
      temp_local_ind[i] = 1;
      temp_local_ind_count++;
    }
    for(j = S_offd_i[i]; j < S_offd_i[i+1]; j++) {
      // S_offd_map[S_offd_j[j]] is the off-processor node's global number.
      local_i = local_ind[hypre_BinarySearch(global_ind, S_offd_map[S_offd_j[j]], offd_count)];
      data = hypre_CTAlloc(int, 1);
      *data = i;
      enqueueData(data, offd_rho[local_i].ond_neighbor_list);
    }
  }

  // Determine V_i^S (the vertices on this processor who share at least one
  // edge with a vertex on another processor).
  //
  // Basically all of the nodes in the set V_i^s are the nodes that have a
  // row of data in the S_offd matrix. This information is already in
  // temp_local_ind and just needs to be extracted.
  ond_to_offd_count = temp_local_ind_count;
  V_i_S = hypre_CTAlloc(int, ond_to_offd_count);
  n_wait = hypre_CTAlloc(int, ond_to_offd_count);
  send_queue = hypre_CTAlloc(hypre_Queue*, ond_to_offd_count);
  temp_local_ind_count = 0;
  for(i = 0; i < num_rows; i++) {
    if(temp_local_ind[i]) {
      V_i_S[temp_local_ind_count] = i;
      temp_local_ind_count++;
    }
  }
  hypre_TFree(temp_local_ind);


  // Determine the values of rho for the ond and offd nodes.
  ond_rho = hypre_CTAlloc(double, ond_to_offd_count);
  setRhoValuesOnd(ond_rho, ond_to_offd_count, V_i_S, row_starts);
  setRhoValuesOffd(offd_rho, offd_count, S_offd_map);

  // color-queue = {empty set}
  color_queue = newQueue();

  proc_send_count_list = hypre_CTAlloc(int, proc_count);
  proc_recv_count_list = hypre_CTAlloc(int, proc_count);
  // For each v in V_i^S do
  for(i = 0; i < ond_to_offd_count; i++) {
    // n-wait(v) = 0
    n_wait[i] = 0;

    // send-queue(v) = {empty set}
    send_queue[i] = newQueue();

    // For each edge (v,w) in E^S do
    for(j = S_offd_i[V_i_S[i]]; j < S_offd_i[V_i_S[i]+1]; j++) {
      if(offd_rho[S_offd_j[j]].rho > ond_rho[i]) {
	n_wait[i]++;
	proc_recv_count_list[hypre_BinarySearch(proc_list, offd_rho[S_offd_j[j]].proc, proc_count)]++;
      }
      else {
	data = hypre_CTAlloc(int, 2);
	data[0] = S_offd_map[S_offd_j[j]];
	data[1] = offd_rho[S_offd_j[j]].proc;
	proc_send_count_list[hypre_BinarySearch(proc_list, data[1], proc_count)]++;
	enqueueData(data, send_queue[i]);
      }
    }
    if(n_wait[i] == 0) {
      data = hypre_CTAlloc(int, 1);
      *data = i; // questionable
      enqueueData(data, color_queue);
    }
  }


  proc_send_list = hypre_CTAlloc(int*, proc_count);
  proc_send_list_ind = hypre_CTAlloc(int, proc_count);
  packed = hypre_CTAlloc(int, proc_count);
  proc_recv_list = hypre_CTAlloc(int*, proc_count);
  proc_recv_list_ind = hypre_CTAlloc(int, proc_count);
  for(i = 0; i < proc_count; i++) {
    if(proc_send_count_list[i] > 0)
      proc_send_list[i] = hypre_CTAlloc(int, 3*proc_send_count_list[i]);
    else
      proc_send_list[i] = NULL;
    if(proc_recv_count_list[i] > 0)
      proc_recv_list[i] = hypre_CTAlloc(int, 3*proc_recv_count_list[i]);
    else
      proc_recv_list[i] = NULL;
  }
  n_colored = color_queue->length;
  // Color any vertices in V_i^S not waiting for messages.
  colorQueue(S, color_array, offd_rho, color_queue, num_colors, send_queue, proc_list, proc_send_list, proc_send_list_ind, proc_count, packed, V_i_S);

  // Send the packed data in proc_send_list.
  for(i = 0; i < proc_count; i++) {
    if(packed[i]) {
      MPI_Isend(proc_send_list[i], 3*packed[i], MPI_INT,
		proc_list[i], 0, comm, &request);
    }
  }

  int * buffer, local_offd_ind, V_i_S_ind;
  buffer = hypre_CTAlloc(int, 3000);
  while(n_colored < ond_to_offd_count) {
    // Receive message.
    MPI_Recv(buffer, 3000, MPI_INT, MPI_ANY_SOURCE, 0, comm, &status);
    MPI_Get_count(&status, MPI_INT, &msg_size); // get the message size

    // Dig through the message.
    for(i = 0; i < msg_size; i += 3) {
      local_offd_ind = hypre_BinarySearch(global_ind, buffer[i+1], offd_count);
      V_i_S_ind = hypre_BinarySearch(V_i_S, buffer[i]-row_starts, ond_to_offd_count);
      offd_rho[local_ind[local_offd_ind]].color = buffer[i+2];
      if(V_i_S_ind > -1)
	n_wait[V_i_S_ind]--;
      if(V_i_S_ind > -1 && n_wait[V_i_S_ind] == 0) {
	data = hypre_CTAlloc(int, 1);
	*data = V_i_S_ind;
	enqueueData(data, color_queue);
      }
    }
    n_colored += color_queue->length;
    colorQueue(S, color_array, offd_rho, color_queue, num_colors, send_queue, proc_list, proc_send_list, proc_send_list_ind, proc_count, packed, V_i_S);
    // Send the packed data in proc_send_list.
    for(i = 0; i < proc_count; i++) {
      if(packed[i]) {
	MPI_Isend(&proc_send_list[i][proc_send_list_ind[i]-3*packed[i]],
		  3*packed[i], MPI_INT, proc_list[i], 0, comm, &request);
      }
    }
  }
  hypre_TFree(buffer);

/*   int buffer[1000]; */
/*   int wait, ip, size; */
/*   MPI_Status status; */
/*   if(my_id == 0) { */
/*     wait = 4; */
/*     ip = 1; */
/*     size = 12; */
/*   } */
/*   else { */
/*     wait = 2; */
/*     ip = 0; */
/*     size = 6; */
/*   } */
/*     size = MPI_Recv(buffer, 1000, MPI_INT, ip, 0, comm, &status); */
/*     MPI_Get_count(&status, MPI_INT, &size); */
/*     printf("%i %i\n", my_id, size); */

  seqColorGraphNewAgain(A, color_array, num_colors, V_i_S, ond_to_offd_count, level); //DELETE ME AND THE DECLARTION ABOVE THIS FUNCTION!!!!!!!!!!!!!!!!!!!!

  // Memory deallocation.
  for(i = 0; i < offd_count; i++) {
    curr_el = offd_rho[i].ond_neighbor_list->head;
    while(curr_el) {
      hypre_TFree(curr_el->data);
      curr_el = curr_el->next_elt;
    }
    destroyQueue(offd_rho[i].ond_neighbor_list);
  }
  hypre_TFree(offd_rho);
  hypre_TFree(global_ind);
  hypre_TFree(local_ind);
  hypre_TFree(V_i_S);
  hypre_TFree(n_wait);
  hypre_TFree(ond_rho);
  for(i = 0; i < ond_to_offd_count; i++) {
    curr_el = send_queue[i]->head;
    while(curr_el) {
      hypre_TFree(curr_el->data);
      curr_el = curr_el->next_elt;
    }
    destroyQueue(send_queue[i]);
  }
  hypre_TFree(send_queue);

  curr_el = color_queue->head;
  while(curr_el) {
    hypre_TFree(curr_el->data);
    curr_el = curr_el->next_elt;
  }
  destroyQueue(color_queue);
  hypre_TFree(proc_list);
  hypre_TFree(proc_send_count_list);
  hypre_TFree(proc_recv_count_list);
  hypre_TFree(proc_send_list);
  hypre_TFree(proc_send_list_ind);
  hypre_TFree(proc_recv_list);
  hypre_TFree(proc_recv_list_ind);
  hypre_TFree(packed);
}

void incrementNeighborCount(hypre_ParCSRMatrix * S, hypre_Queue * color_queue,
			    hypre_QueueElement * color_queue_array,
			    hypre_QueueElement ** degree_tail_array,
			    int * degree_null_beyond, int i, short * color_array)
{
  hypre_QueueElement * prev_elt;
  hypre_CSRMatrix * diag;
  int *neighbor_node_color_elt, *diag_i, *diag_j;
  int j, curr_node_degree_index;

  diag = hypre_ParCSRMatrixDiag(S);
  diag_i = hypre_CSRMatrixI(diag);
  diag_j = hypre_CSRMatrixJ(diag);

  neighbor_node_color_elt = color_queue_array[diag_j[i]].data;
  neighbor_node_color_elt[1]++;
  curr_node_degree_index = neighbor_node_color_elt[1]-1;

/* hypre_QueueElement *alk; */
/*  int here = 0; */
/* alk = color_queue->head;  */
/* while(alk) { */
/*   if(alk->data[0] == 5) { */
/*      printf("it's here!\n"); */
/*      here = 1; */
/*   } */
/*   alk = alk->next_elt; */
/*  } */
/*   if(!here) */
/*      printf("it's not here!\n"); */

  // The node adjacent in the queue will be needed later.
  prev_elt = color_queue_array[diag_j[i]].prev_elt;

  // Determine the destination location in the queue.
  if(degree_tail_array[curr_node_degree_index]) {
    // Then we are in luck and already know where to put this node.
    // Make sure the new location is not actually where the node is
    // now.
    if(degree_tail_array[curr_node_degree_index] != color_queue_array[diag_j[i]].prev_elt) {
      // Then a move is in order; else the node is in the right place.
      moveAfter(&color_queue_array[diag_j[i]],
		degree_tail_array[curr_node_degree_index], color_queue);
    }
  } 
  else {
    // The value of the pointer to the elements with the same number
    // of neighbors that have been colored as this node is NULL. That
    // means that this is the only node with this number of neighbors
    // that have been colored.
    //
    // The situation at this point can be one of a two.
    // 1) This node could be the node with the largest number of
    //    neighbors that have been colored.
    // 2) This node is not the node with the largest number of
    //    neighbors that have been colored.
    //
    // The difference between the two is visible by looking at
    // degree_null_beyond. If this node has more neighbors that have
    // been colored than the value of degree_null_beyond, then this
    // is the first case.
    if(curr_node_degree_index > *degree_null_beyond ||
       *degree_null_beyond == 0) {
      // Then this is the first case. Now this is the node with the
      // largest number of neighbors that have been colored.
      //
      // Additionally, this node is now at the head of the queue,
      // if it is not already there (i.e., it is not the last element).
      *degree_null_beyond = curr_node_degree_index;
      if(color_queue->head != &color_queue_array[diag_j[i]])
	moveToHead(&color_queue_array[diag_j[i]], color_queue);
    }
    else {
      // Then this is the second case. This is not the node with
      // the largest number of neighbors that have already been
      // colored, but it does have a unique number of neighbors that
      // have already been colored.
      //
      // It needs to be determined where this element is to be moved
      // in the queue. This is done by looking at
      // degree_tail_array[curr_node_degree_index+1,...,max_rownnz]
      // until a non-null element is found in degree_tail_array. The
      // element that is found will be the one to directly precede
      // this element in the queue.
      j = curr_node_degree_index+1;
      while(!degree_tail_array[j])
	j++;
      
      // Now we have the index of the element to precede this node.
      // Move the node there.
      moveAfter(&color_queue_array[diag_j[i]], degree_tail_array[j],
		color_queue);	      
    }
  }
  
  // Update the degree_tail_array for this element.
  degree_tail_array[curr_node_degree_index] = &color_queue_array[diag_j[i]];
  if(curr_node_degree_index > 0 &&
     degree_tail_array[curr_node_degree_index-1] == &color_queue_array[diag_j[i]]) {
    // Then the degree_tail_array points to this element already
    // from before. Remove that pointer.
    if(prev_elt && prev_elt->data[1] == curr_node_degree_index)
      degree_tail_array[curr_node_degree_index-1] = prev_elt;
    else
      degree_tail_array[curr_node_degree_index-1] = NULL;
  }
}

/** The function assigns a color to each node for this process.
 *
 * This works by using a greedy method to assign a color to a node.
 *
 * Basically it does a breadth first search and colors a node with the lowest possible color number such that no two adjacent nodes are the same color. The order in which the nodes are visited is determined using incidence degree ordering (IDO). The node with the largest number of adjacent nodes that have already been colored is colored next.
 *
 * NOTE:
 * Right now this is just a greedy graph coloring algorithm. If this produces good coarsening results the Jones-Plassmann parallel graph coloring should be implemented in its stead.
 *
 * @param S a hypre_ParCSRMatrix object containing the strength matrix of the linear system A.
 * @param color_array [OUT] an array of shorts containing color of each node.
 * @param num_colors a pointer to an integer storing the total number of colors in the graph.
 */
void seqColorGraph(hypre_ParCSRMatrix * S, short * color_array, int * num_colors, int level)
{
#define QUEUED        -1
#define UNCOLORED     0

  hypre_Queue        *color_queue;
  hypre_QueueElement *color_queue_array, *prev_elt;
  hypre_CSRMatrix    *diag;
  int                *node_color_elt, *neighbor_node_color_elt, *diag_i;
  int                *diag_j, *used_colors;
  int                i, j, curr_node, neighbor_color, curr_choice;
  int                num_variables, max_rownnz, curr_node_degree_index;
  int degree_null_beyond;
  hypre_QueueElement **degree_tail_array;

  color_queue = newQueue();
  *num_colors = 0;

  diag = hypre_ParCSRMatrixDiag(S);
  diag_i = hypre_CSRMatrixI(diag);
  diag_j = hypre_CSRMatrixJ(diag);

  num_variables = hypre_CSRMatrixNumRows(diag);

  // Determine the maximum degree of the graph.
  max_rownnz = 0;
  for(i = 0; i < num_variables; i++) {
    if(diag_i[i+1]-diag_i[i] > max_rownnz)
      max_rownnz = diag_i[i+1]-diag_i[i];
  }

  // The degree_tail_array contains pointers to specific elements in the
  // queue. For example, degree_tail_array[2] points to the last element
  // in the queue with three neighbors that have already been colored. The
  // pointer is to the last element so that when an element in the queue
  // needs to be moved up, it can be added to the end of the appropriate
  // "section" of elements.
  degree_tail_array = hypre_CTAlloc(hypre_QueueElement*, max_rownnz);
  // degree_null_beyond contains the largest index into degree_tail_array
  // that is not NULL. This way a node can quickly tell if it needs to go
  // to the head of the queue.
  degree_null_beyond = 0;

  if(num_variables > 0) {
    // Create hypre_QueueElements for each node.
    color_queue_array = hypre_CTAlloc(hypre_QueueElement, num_variables);

    // Set the data in each element and then enqueue the node.
    for(i = 0; i < num_variables; i++) {
      node_color_elt = hypre_CTAlloc(int, 2);
      node_color_elt[0] = i;
      node_color_elt[1] = 0; // zero neighbors that have been colored
      color_queue_array[i].data = node_color_elt;
    }
      enqueueElement(&color_queue_array[0], color_queue);
      color_array[0] = QUEUED;
  }

  while(color_queue->head) {
/*     // Temporary verification code. Make sure the queue is properly ordered. */
/*     hypre_QueueElement * cnode = color_queue->head; */
/*     int val = 1000000; */
/*     int prev_val; */
/*     while(cnode) { */
/*       prev_val = val; */
/*       val = cnode->data[1]; */
/*       if(val > prev_val) { */
/* 	printf("EEP!!!!!!!!!!!!!!!!! %i %i\n", prev_val, val); */
/* 	break; */
/*       } */
/*       //printf("%i(%i) ", val, cnode->data[0]); */
/*       cnode = cnode->next_elt; */
/*     } */
/*     //printf("\n\n"); */

    // while the queue still contains elements
    // color this node and enqueue all of its uncolored neighbors
    node_color_elt = dequeue(color_queue);
    curr_node = node_color_elt[0];

    // Check to see if this element was in the degree_tail_array. If it was,
    // then that entry in the degree_tail_array is to be set to NULL. Also,
    // search to find the next entry and update degree_null_beyond.
    if(node_color_elt[1] > 0 &&
       degree_tail_array[node_color_elt[1]-1] == &color_queue_array[curr_node]) {
      // Then this was the entry for its number of neighbors already colored
      // in degree_tail_array.
      degree_tail_array[node_color_elt[1]-1] = NULL;
      if(node_color_elt[1] > 0) {
	j = node_color_elt[1]-1;
	while(j > 0 && !degree_tail_array[j])
	  j--;
	
	// Now we have the index of next element in the degree_tail_array, or
	// the array is all NULL.
	degree_null_beyond = j;
      } else
	degree_null_beyond = 0;
    }

    used_colors = hypre_CTAlloc(int, diag_i[curr_node+1] - diag_i[curr_node]);
    // Get the neighbors of this node and also determine their colors.
    for(i = diag_i[curr_node]; i < diag_i[curr_node+1]; i++) {
      if(curr_node != diag_j[i]) {
	neighbor_color = color_array[diag_j[i]];
	
	if(neighbor_color == UNCOLORED || neighbor_color == QUEUED) {
	  if(neighbor_color == UNCOLORED) {
	    enqueueElement(&color_queue_array[diag_j[i]], color_queue);
	    color_array[diag_j[i]] = QUEUED;
	  }

	  // Since this neighbor is uncolored, increment the number
	  // of neighbors it has that are colored and move its position
	  // in the queue, if necessary.
	  neighbor_node_color_elt = color_queue_array[diag_j[i]].data;
	  neighbor_node_color_elt[1]++;
	  curr_node_degree_index = neighbor_node_color_elt[1]-1;

	  // The node adjacent in the queue will be needed later.
	  prev_elt = color_queue_array[diag_j[i]].prev_elt;

	  // Determine the destination location in the queue.
	  if(degree_tail_array[curr_node_degree_index]) {
	    // Then we are in luck and already know where to put this node.
	    // Make sure the new location is not actually where the node is
	    // now.
	    if(degree_tail_array[curr_node_degree_index] != color_queue_array[diag_j[i]].prev_elt) {
	      // Then a move is in order; else the node is in the right place.
	      moveAfter(&color_queue_array[diag_j[i]],
			degree_tail_array[curr_node_degree_index], color_queue);
	    }
	  } 
	  else {
	    // The value of the pointer to the elements with the same number
	    // of neighbors that have been colored as this node is NULL. That
	    // means that this is the only node with this number of neighbors
	    // that have been colored.
	    //
	    // The situation at this point can be one of a two.
	    // 1) This node could be the node with the largest number of
	    //    neighbors that have been colored.
	    // 2) This node is not the node with the largest number of
	    //    neighbors that have been colored.
	    //
	    // The difference between the two is visible by looking at
	    // degree_null_beyond. If this node has more neighbors that have
	    // been colored than the value of degree_null_beyond, then this
	    // is the first case.
	    if(curr_node_degree_index > degree_null_beyond ||
	       degree_null_beyond == 0) {
	      // Then this is the first case. Now this is the node with the
	      // largest number of neighbors that have been colored.
	      //
	      // Additionally, this node is now at the head of the queue,
	      // if it is not already there (i.e., it is not the last element).
	      degree_null_beyond = curr_node_degree_index;
	      if(color_queue->head != &color_queue_array[diag_j[i]])
		moveToHead(&color_queue_array[diag_j[i]], color_queue);
	    }
	    else {
	      // Then this is the second case. This is not the node with
	      // the largest number of neighbors that have already been
	      // colored, but it does have a unique number of neighbors that
	      // have already been colored.
	      //
	      // It needs to be determined where this element is to be moved
	      // in the queue. This is done by looking at
	      // degree_tail_array[curr_node_degree_index+1,...,max_rownnz]
	      // until a non-null element is found in degree_tail_array. The
	      // element that is found will be the one to directly precede
	      // this element in the queue.
	      j = curr_node_degree_index+1;
	      while(!degree_tail_array[j])
		j++;

	      // Now we have the index of the element to precede this node.
	      // Move the node there.
	      moveAfter(&color_queue_array[diag_j[i]], degree_tail_array[j],
			color_queue);	      
	    }
	  }

	  // Update the degree_tail_array for this element.
	  degree_tail_array[curr_node_degree_index] = &color_queue_array[diag_j[i]];
	  if(curr_node_degree_index > 0 &&
	     degree_tail_array[curr_node_degree_index-1] == &color_queue_array[diag_j[i]]) {
	    // Then the degree_tail_array points to this element already
	    // from before. Update that pointer.
	    if(prev_elt && prev_elt->data[1] == curr_node_degree_index)
	      degree_tail_array[curr_node_degree_index-1] = prev_elt;
	    else
	      degree_tail_array[curr_node_degree_index-1] = NULL;
	  }

	}
	else {
	  // Take note of this neighbor's color.
	  used_colors[i - diag_i[curr_node]] = neighbor_color;
	}
      }
    }

    // Color this node based on the information gathered.
    // Sort the used_colors array.
    qsort0(used_colors, 0, diag_i[curr_node+1] - diag_i[curr_node] - 1);

    // Now search the used_colors array to find the lowest number > 0 that
    // does not appear. Make that number the color of curr_node.
    curr_choice = 1;
    for(i = 0; i < diag_i[curr_node+1] - diag_i[curr_node]; i++) {
      if(used_colors[i] == curr_choice)
	// Then the current choice of color is in use. Pick the next one up.
	curr_choice++;
      else if(used_colors[i] > curr_choice) {
	// The the current choice of color is available. Exit the loop and
	// color curr_node this color.
	i = diag_i[curr_node+1] - diag_i[curr_node]; // to break loop
      }
    }
    color_array[curr_node] = curr_choice;
    if(curr_choice > *num_colors)
      *num_colors = curr_choice;
    
    hypre_TFree(used_colors);
  }
  WriteCoarseningColorInformation(S, color_array, level);

  hypre_TFree(degree_tail_array);
  for(i = 0; i < num_variables; i++)
    hypre_TFree(color_queue_array[i].data); // free all of the node_color_elt
  if(num_variables > 0)
    hypre_TFree(color_queue_array);
  destroyQueue(color_queue);
}

void seqColorGraphFromQueue(hypre_ParCSRMatrix * S, short * color_array, int * num_colors, int * V_i_S, int V_i_S_len, int level)
{
#define QUEUED        -1
#define UNCOLORED     0

  hypre_Queue        *color_queue;
  hypre_QueueElement *color_queue_array, *prev_elt;
  hypre_CSRMatrix    *diag;
  int                *node_color_elt, *neighbor_node_color_elt, *diag_i;
  int                *diag_j, *used_colors;
  int                i, j, curr_node, neighbor_color, curr_choice;
  int                num_variables, max_rownnz, curr_node_degree_index;
  int degree_null_beyond;
  hypre_QueueElement **degree_tail_array;

  color_queue = newQueue();
  //*num_colors = 0;

  diag = hypre_ParCSRMatrixDiag(S);
  diag_i = hypre_CSRMatrixI(diag);
  diag_j = hypre_CSRMatrixJ(diag);

  num_variables = hypre_CSRMatrixNumRows(diag);

  // Determine the maximum degree of the graph.
  max_rownnz = 0;
  for(i = 0; i < num_variables; i++) {
    if(diag_i[i+1]-diag_i[i] > max_rownnz)
      max_rownnz = diag_i[i+1]-diag_i[i];
  }

  // The degree_tail_array contains pointers to specific elements in the
  // queue. For example, degree_tail_array[2] points to the last element
  // in the queue with three neighbors that have already been colored. The
  // pointer is to the last element so that when an element in the queue
  // needs to be moved up, it can be added to the end of the appropriate
  // "section" of elements.
  degree_tail_array = hypre_CTAlloc(hypre_QueueElement*, max_rownnz);
  // degree_null_beyond contains the largest index into degree_tail_array
  // that is not NULL. This way a node can quickly tell if it needs to go
  // to the head of the queue.
  degree_null_beyond = 0;

  if(num_variables > 0) {
    // Create hypre_QueueElements for each node.
    color_queue_array = hypre_CTAlloc(hypre_QueueElement, num_variables);

    // Set the data in each element and then enqueue the node.
    for(i = 0; i < num_variables; i++) {
      node_color_elt = hypre_CTAlloc(int, 2);
      node_color_elt[0] = i;
      node_color_elt[1] = 0; // zero neighbors that have been colored
      color_queue_array[i].data = node_color_elt;
      if(color_array[i] <= 0) {
	enqueueElement(&color_queue_array[i], color_queue);
	color_array[i] = QUEUED;
      }
    }
      
    for(i = 0; i < V_i_S_len; i++) {
      // Update the number of neighbors that have been colored for all of
      // the nodes adjacent to the V_i_S nodes.
      for(j = diag_i[V_i_S[i]]; j < diag_i[V_i_S[i]+1]; j++) {
	if(color_array[diag_j[j]] == QUEUED) {
	  incrementNeighborCount(S, color_queue, color_queue_array,
				 degree_tail_array, &degree_null_beyond, j, color_array);
	}
      }
    }
  }

  while(color_queue->head) {
    // while the queue still contains elements
    // color this node and update all of its neighbors
    node_color_elt = dequeue(color_queue);
    curr_node = node_color_elt[0];

    // Check to see if this element was in the degree_tail_array. If it was,
    // then that entry in the degree_tail_array is to be set to NULL. Also,
    // search to find the next entry and update degree_null_beyond.
    if(node_color_elt[1] > 0 &&
       degree_tail_array[node_color_elt[1]-1] == &color_queue_array[curr_node]) {
      // Then this was the entry for its number of neighbors already colored
      // in degree_tail_array.
      degree_tail_array[node_color_elt[1]-1] = NULL;
      if(node_color_elt[1] > 0) {
	j = node_color_elt[1]-1;
	while(j > 0 && !degree_tail_array[j])
	  j--;
	
	// Now we have the index of next element in the degree_tail_array, or
	// the array is all NULL.
	degree_null_beyond = j;
      } else
	degree_null_beyond = 0;
    }

    used_colors = hypre_CTAlloc(int, diag_i[curr_node+1] - diag_i[curr_node]);
    // Get the neighbors of this node and also determine their colors.
    for(i = diag_i[curr_node]; i < diag_i[curr_node+1]; i++) {
      if(curr_node != diag_j[i]) {
	neighbor_color = color_array[diag_j[i]];
	
	if(neighbor_color == QUEUED) {
	  // Since this neighbor is uncolored, increment the number
	  // of neighbors it has that are colored and move its position
	  // in the queue, if necessary.
	  incrementNeighborCount(S, color_queue, color_queue_array,
				 degree_tail_array, &degree_null_beyond, i, color_array);
	}
	else {
	  // Take note of this neighbor's color.
	  used_colors[i - diag_i[curr_node]] = neighbor_color;
	}
      }
    }

    // Color this node based on the information gathered.
    // Sort the used_colors array.
    qsort0(used_colors, 0, diag_i[curr_node+1] - diag_i[curr_node] - 1);

    // Now search the used_colors array to find the lowest number > 0 that
    // does not appear. Make that number the color of curr_node.
    curr_choice = 1;
    for(i = 0; i < diag_i[curr_node+1] - diag_i[curr_node]; i++) {
      if(used_colors[i] == curr_choice)
	// Then the current choice of color is in use. Pick the next one up.
	curr_choice++;
      else if(used_colors[i] > curr_choice) {
	// The the current choice of color is available. Exit the loop and
	// color curr_node this color.
	i = diag_i[curr_node+1] - diag_i[curr_node]; // to break loop
      }
    }
    color_array[curr_node] = curr_choice;
    if(curr_choice > *num_colors)
      *num_colors = curr_choice;
    
    hypre_TFree(used_colors);
  }
  WriteCoarseningColorInformation(S, color_array, level);

  hypre_TFree(degree_tail_array);
  for(i = 0; i < num_variables; i++)
    hypre_TFree(color_queue_array[i].data); // free all of the node_color_elt
  if(num_variables > 0)
    hypre_TFree(color_queue_array);
  destroyQueue(color_queue);
}

void seqColorGraphNewAgain(hypre_ParCSRMatrix * S, short * color_array, int * num_colors, int * V_i_S, int V_i_S_len, int level)
{
#define QUEUED        -1
#define UNCOLORED     0

  hypre_Queue        *color_queue;
  hypre_QueueElement *color_queue_array, *prev_elt;
  hypre_CSRMatrix    *diag;
  int                *node_color_elt, *neighbor_node_color_elt, *diag_i;
  int                *diag_j, *used_colors;
  int                i, j, curr_node, neighbor_color, curr_choice;
  int                num_variables, max_rownnz, curr_node_degree_index;
  int degree_null_beyond;
  hypre_QueueElement **degree_tail_array;

  color_queue = newQueue();
  //*num_colors = 0;

  diag = hypre_ParCSRMatrixDiag(S);
  diag_i = hypre_CSRMatrixI(diag);
  diag_j = hypre_CSRMatrixJ(diag);

  num_variables = hypre_CSRMatrixNumRows(diag);

  // Determine the maximum degree of the graph.
  max_rownnz = 0;
  for(i = 0; i < num_variables; i++) {
    if(diag_i[i+1]-diag_i[i] > max_rownnz)
      max_rownnz = diag_i[i+1]-diag_i[i];
  }

  // The degree_tail_array contains pointers to specific elements in the
  // queue. For example, degree_tail_array[2] points to the last element
  // in the queue with three neighbors that have already been colored. The
  // pointer is to the last element so that when an element in the queue
  // needs to be moved up, it can be added to the end of the appropriate
  // "section" of elements.
  degree_tail_array = hypre_CTAlloc(hypre_QueueElement*, max_rownnz);
  // degree_null_beyond contains the largest index into degree_tail_array
  // that is not NULL. This way a node can quickly tell if it needs to go
  // to the head of the queue.
  degree_null_beyond = 0;

  if(num_variables > 0) {
    // Create hypre_QueueElements for each node.
    color_queue_array = hypre_CTAlloc(hypre_QueueElement, num_variables);

    // Set the data in each element and then enqueue the node.
    for(i = 0; i < num_variables; i++) {
      node_color_elt = hypre_CTAlloc(int, 2);
      node_color_elt[0] = i;
      node_color_elt[1] = 0; // zero neighbors that have been colored
      color_queue_array[i].data = node_color_elt;
      if(color_array[i] <= 0) {
	enqueueElement(&color_queue_array[i], color_queue);
	color_array[i] = QUEUED;
      }
    }
      
    for(i = 0; i < V_i_S_len; i++) {
      // Update the number of neighbors that have been colored for all of
      // the nodes adjacent to the V_i_S nodes.
      for(j = diag_i[V_i_S[i]]; j < diag_i[V_i_S[i]+1]; j++) {
	if(color_array[diag_j[j]] == QUEUED) {
	  incrementNeighborCount(S, color_queue, color_queue_array,
				 degree_tail_array, &degree_null_beyond, j, color_array);
	}
      }
    }
  }

  while(color_queue->head) {
    // while the queue still contains elements
    // color this node and update all of its neighbors
    node_color_elt = dequeue(color_queue);
    curr_node = node_color_elt[0];

    // Check to see if this element was in the degree_tail_array. If it was,
    // then that entry in the degree_tail_array is to be set to NULL. Also,
    // search to find the next entry and update degree_null_beyond.
    if(node_color_elt[1] > 0 &&
       degree_tail_array[node_color_elt[1]-1] == &color_queue_array[curr_node]) {
      // Then this was the entry for its number of neighbors already colored
      // in degree_tail_array.
      degree_tail_array[node_color_elt[1]-1] = NULL;
      if(node_color_elt[1] > 0) {
	j = node_color_elt[1]-1;
	while(j > 0 && !degree_tail_array[j])
	  j--;
	
	// Now we have the index of next element in the degree_tail_array, or
	// the array is all NULL.
	degree_null_beyond = j;
      } else
	degree_null_beyond = 0;
    }

    used_colors = hypre_CTAlloc(int, diag_i[curr_node+1] - diag_i[curr_node]);
    // Get the neighbors of this node and also determine their colors.
    for(i = diag_i[curr_node]; i < diag_i[curr_node+1]; i++) {
      if(curr_node != diag_j[i]) {
	neighbor_color = color_array[diag_j[i]];
	
	if(neighbor_color == QUEUED) {
	  // Since this neighbor is uncolored, increment the number
	  // of neighbors it has that are colored and move its position
	  // in the queue, if necessary.
	  incrementNeighborCount(S, color_queue, color_queue_array,
				 degree_tail_array, &degree_null_beyond, i, color_array);
	}
	else {
	  // Take note of this neighbor's color.
	  used_colors[i - diag_i[curr_node]] = neighbor_color;
	}
      }
    }

    // Color this node based on the information gathered.
    // Sort the used_colors array.
    qsort0(used_colors, 0, diag_i[curr_node+1] - diag_i[curr_node] - 1);

    // Now search the used_colors array to find the lowest number > 0 that
    // does not appear. Make that number the color of curr_node.
    curr_choice = 1;
    for(i = 0; i < diag_i[curr_node+1] - diag_i[curr_node]; i++) {
      if(used_colors[i] == curr_choice)
	// Then the current choice of color is in use. Pick the next one up.
	curr_choice++;
      else if(used_colors[i] > curr_choice) {
	// The the current choice of color is available. Exit the loop and
	// color curr_node this color.
	i = diag_i[curr_node+1] - diag_i[curr_node]; // to break loop
      }
    }
    color_array[curr_node] = curr_choice;
    if(curr_choice > *num_colors)
      *num_colors = curr_choice;
    
    hypre_TFree(used_colors);
  }
  WriteCoarseningColorInformation(S, color_array, level);

  hypre_TFree(degree_tail_array);
  for(i = 0; i < num_variables; i++)
    hypre_TFree(color_queue_array[i].data); // free all of the node_color_elt
  if(num_variables > 0)
    hypre_TFree(color_queue_array);
  destroyQueue(color_queue);
}

/* This is the same as the seqColorGraphNew with the exception that the uncommented version
   implements the color search (selecting the candidate color) more efficiently.

void seqColorGraphNew(hypre_ParCSRMatrix * S, short * color_array, int * num_colors, int level)
{
#define QUEUED        -1
#define UNCOLORED     0

  hypre_Queue        *color_queue;
  hypre_QueueElement *color_queue_array, *prev_elt;
  hypre_CSRMatrix    *diag;
  int                *node_color_elt, *neighbor_node_color_elt, *diag_i;
  int                *diag_j, *used_colors;
  int                i, j, curr_node, neighbor_color, curr_choice;
  int                num_variables, max_rownnz, curr_node_degree_index;
  int                degree_null_beyond, max_degree_node;
  hypre_QueueElement **degree_tail_array;

  color_queue = newQueue();
  *num_colors = 0;

  diag = hypre_ParCSRMatrixDiag(S);
  diag_i = hypre_CSRMatrixI(diag);
  diag_j = hypre_CSRMatrixJ(diag);

  num_variables = hypre_CSRMatrixNumRows(diag);

  // Determine the maximum degree of the graph and the first node that is of
  // maximum degree.
  max_rownnz = 0;
  for(i = 0; i < num_variables; i++) {
    if(diag_i[i+1]-diag_i[i] > max_rownnz) {
      max_rownnz = diag_i[i+1]-diag_i[i];
      max_degree_node = i;
    }
  }

  // The degree_tail_array contains pointers to specific elements in the
  // queue. For example, degree_tail_array[2] points to the last element
  // in the queue with three neighbors that have already been colored. The
  // pointer is to the last element so that when an element in the queue
  // needs to be moved up, it can be added to the end of the appropriate
  // "section" of elements.
  degree_tail_array = hypre_CTAlloc(hypre_QueueElement*, max_rownnz);
  // degree_null_beyond contains the largest index into degree_tail_array
  // that is not NULL. This way a node can quickly tell if it needs to go
  // to the head of the queue.
  degree_null_beyond = 0;

  if(num_variables > 0) {
    // Create hypre_QueueElements for each node.
    color_queue_array = hypre_CTAlloc(hypre_QueueElement, num_variables);

    // Set the data in each element and then enqueue the node.
    for(i = 0; i < num_variables; i++) {
      node_color_elt = hypre_CTAlloc(int, 2);
      node_color_elt[0] = i;
      node_color_elt[1] = 0; // zero neighbors that have been colored
      color_queue_array[i].data = node_color_elt;
      enqueueElement(&color_queue_array[i], color_queue);
      color_array[i] = QUEUED;
      if(i == max_degree_node)
	moveToHead(&color_queue_array[i], color_queue);
    }
  }

  while(color_queue->head) {
    // while the queue still contains elements
    node_color_elt = dequeue(color_queue);
    curr_node = node_color_elt[0];

    // Check to see if this element was in the degree_tail_array. If it was,
    // then that entry in the degree_tail_array is to be set to NULL. Also,
    // search to find the next entry and update degree_null_beyond.
    if(node_color_elt[1] > 0 &&
       degree_tail_array[node_color_elt[1]-1] == &color_queue_array[curr_node]) {
      // Then this was the entry for its number of neighbors already colored
      // in degree_tail_array.
      degree_tail_array[node_color_elt[1]-1] = NULL;
      if(node_color_elt[1] > 0) {
	j = node_color_elt[1]-1;
	while(j > 0 && !degree_tail_array[j])
	  j--;
	
	// Now we have the index of next element in the degree_tail_array, or
	// the array is all NULL.
	degree_null_beyond = j;
      } else
	degree_null_beyond = 0;
    }

    used_colors = hypre_CTAlloc(int, diag_i[curr_node+1] - diag_i[curr_node]);
    // Get the neighbors of this node and also determine their colors.
    for(i = diag_i[curr_node]; i < diag_i[curr_node+1]; i++) {
      if(curr_node != diag_j[i]) {
	neighbor_color = color_array[diag_j[i]];
	
	if(neighbor_color == QUEUED) {
	  // Since this neighbor is uncolored, increment the number
	  // of neighbors it has that are colored and move its position
	  // in the queue, if necessary.
	  neighbor_node_color_elt = color_queue_array[diag_j[i]].data;
	  neighbor_node_color_elt[1]++;
	  curr_node_degree_index = neighbor_node_color_elt[1]-1;

	  // The node adjacent in the queue will be needed later.
	  prev_elt = color_queue_array[diag_j[i]].prev_elt;

	  // Determine the destination location in the queue.
	  if(degree_tail_array[curr_node_degree_index]) {
	    // Then we are in luck and already know where to put this node.
	    // Make sure the new location is not actually where the node is
	    // now.
	    if(degree_tail_array[curr_node_degree_index] != color_queue_array[diag_j[i]].prev_elt) {
	      // Then a move is in order; else the node is in the right place.
	      moveAfter(&color_queue_array[diag_j[i]],
			degree_tail_array[curr_node_degree_index], color_queue);
	    }
	  } 
	  else {
	    // The value of the pointer to the elements with the same number
	    // of neighbors that have been colored as this node is NULL. That
	    // means that this is the only node with this number of neighbors
	    // that have been colored.
	    //
	    // The situation at this point can be one of two.
	    // 1) This node could be the node with the largest number of
	    //    neighbors that have been colored.
	    // 2) This node is not the node with the largest number of
	    //    neighbors that have been colored.
	    //
	    // The difference between the two is visible by looking at
	    // degree_null_beyond. If this node has more neighbors that have
	    // been colored than the value of degree_null_beyond, then this
	    // is the first case.
	    if(curr_node_degree_index > degree_null_beyond ||
	       degree_null_beyond == 0) {
	      // Then this is the first case. Now this is the node with the
	      // largest number of neighbors that have been colored.
	      //
	      // Additionally, this node is now at the head of the queue,
	      // if it is not already there (i.e., it is not the last element).
	      degree_null_beyond = curr_node_degree_index;
	      if(color_queue->head != &color_queue_array[diag_j[i]])
		moveToHead(&color_queue_array[diag_j[i]], color_queue);
	    }
	    else {
	      // Then this is the second case. This is not the node with
	      // the largest number of neighbors that have already been
	      // colored, but it does have a unique number of neighbors that
	      // have already been colored.
	      //
	      // It needs to be determined where this element is to be moved
	      // in the queue. This is done by looking at
	      // degree_tail_array[curr_node_degree_index+1,...,max_rownnz]
	      // until a non-null element is found in degree_tail_array. The
	      // element that is found will be the one to directly precede
	      // this element in the queue.
	      j = curr_node_degree_index+1;
	      while(!degree_tail_array[j])
		j++;

	      // Now we have the index of the element to precede this node.
	      // Move the node there.
	      moveAfter(&color_queue_array[diag_j[i]], degree_tail_array[j],
			color_queue);
	    }
	  }

	  // Update the degree_tail_array for this element.
	  degree_tail_array[curr_node_degree_index] = &color_queue_array[diag_j[i]];
	  if(curr_node_degree_index > 0 &&
	     degree_tail_array[curr_node_degree_index-1] == &color_queue_array[diag_j[i]]) {
	    // Then the degree_tail_array points to this element already
	    // from before. Remove that pointer.
	    if(prev_elt && prev_elt->data[1] == curr_node_degree_index)
	      degree_tail_array[curr_node_degree_index-1] = prev_elt;
	    else
	      degree_tail_array[curr_node_degree_index-1] = NULL;
	  }

	}
	else {
	  // Take note of this neighbor's color.
	  used_colors[i - diag_i[curr_node]] = neighbor_color;
	}
      }
    }

    // Color this node based on the information gathered.
    // Sort the used_colors array.
    qsort0(used_colors, 0, diag_i[curr_node+1] - diag_i[curr_node] - 1);

    // Now search the used_colors array to find the lowest number > 0 that
    // does not appear. Make that number the color of curr_node.
    curr_choice = 1;

    for(i = 0; i < diag_i[curr_node+1] - diag_i[curr_node]; i++) {
      if(used_colors[i] == curr_choice)
	// Then the current choice of color is in use. Pick the next one up.
	curr_choice++;
      else if(used_colors[i] > curr_choice) {
	// The the current choice of color is available. Exit the loop and
	// color curr_node this color.
	i = diag_i[curr_node+1] - diag_i[curr_node]; // to break loop
      }
    }
    color_array[curr_node] = curr_choice;
    if(curr_choice > *num_colors)
      *num_colors = curr_choice;
    
    hypre_TFree(used_colors);
  }
  WriteCoarseningColorInformation(S, color_array, level);

  hypre_TFree(degree_tail_array);
  for(i = 0; i < num_variables; i++)
    hypre_TFree(color_queue_array[i].data); // free all of the node_color_elt
  if(num_variables > 0)
    hypre_TFree(color_queue_array);
  destroyQueue(color_queue);
}*/

void seqColorGraphNew(hypre_ParCSRMatrix * S, short * color_array, int * num_colors, int level)
{
#define QUEUED        -1
#define UNCOLORED     0

  hypre_Queue        *color_queue;
  hypre_QueueElement *color_queue_array, *prev_elt;
  hypre_CSRMatrix    *diag;
  int                *node_color_elt, *neighbor_node_color_elt, *diag_i;
  int                *diag_j, *used_colors;
  int                i, j, curr_node, neighbor_color, curr_choice;
  int                num_variables, max_rownnz, curr_node_degree_index;
  int                degree_null_beyond, max_degree_node;
  hypre_QueueElement **degree_tail_array;

  color_queue = newQueue();
  *num_colors = 0;

  diag = hypre_ParCSRMatrixDiag(S);
  diag_i = hypre_CSRMatrixI(diag);
  diag_j = hypre_CSRMatrixJ(diag);

  num_variables = hypre_CSRMatrixNumRows(diag);

  // Determine the maximum degree of the graph and the first node that is of
  // maximum degree.
  max_rownnz = 0;
  for(i = 0; i < num_variables; i++) {
    if(diag_i[i+1]-diag_i[i] > max_rownnz) {
      max_rownnz = diag_i[i+1]-diag_i[i];
      max_degree_node = i;
    }
  }

  // The degree_tail_array contains pointers to specific elements in the
  // queue. For example, degree_tail_array[2] points to the last element
  // in the queue with three neighbors that have already been colored. The
  // pointer is to the last element so that when an element in the queue
  // needs to be moved up, it can be added to the end of the appropriate
  // "section" of elements.
  degree_tail_array = hypre_CTAlloc(hypre_QueueElement*, max_rownnz);
  // degree_null_beyond contains the largest index into degree_tail_array
  // that is not NULL. This way a node can quickly tell if it needs to go
  // to the head of the queue.
  degree_null_beyond = 0;

  if(num_variables > 0) {
    // Create hypre_QueueElements for each node.
    color_queue_array = hypre_CTAlloc(hypre_QueueElement, num_variables);

    // Set the data in each element and then enqueue the node.
    for(i = 0; i < num_variables; i++) {
      node_color_elt = hypre_CTAlloc(int, 2);
      node_color_elt[0] = i;
      node_color_elt[1] = 0; // zero neighbors that have been colored
      color_queue_array[i].data = node_color_elt;
      enqueueElement(&color_queue_array[i], color_queue);
      color_array[i] = QUEUED;
      if(i == max_degree_node)
	moveToHead(&color_queue_array[i], color_queue);
    }
  }

  while(color_queue->head) {
    // while the queue still contains elements
    node_color_elt = dequeue(color_queue);
    curr_node = node_color_elt[0];

    // Check to see if this element was in the degree_tail_array. If it was,
    // then that entry in the degree_tail_array is to be set to NULL. Also,
    // search to find the next entry and update degree_null_beyond.
    if(node_color_elt[1] > 0 &&
       degree_tail_array[node_color_elt[1]-1] == &color_queue_array[curr_node]) {
      // Then this was the entry for its number of neighbors already colored
      // in degree_tail_array.
      degree_tail_array[node_color_elt[1]-1] = NULL;
      if(node_color_elt[1] > 0) {
	j = node_color_elt[1]-1;
	while(j > 0 && !degree_tail_array[j])
	  j--;
	
	// Now we have the index of next element in the degree_tail_array, or
	// the array is all NULL.
	degree_null_beyond = j;
      } else
	degree_null_beyond = 0;
    }

    //used_colors = hypre_CTAlloc(int, diag_i[curr_node+1] - diag_i[curr_node]);
    used_colors = hypre_CTAlloc(int, *num_colors);
    // Get the neighbors of this node and also determine their colors.
    for(i = diag_i[curr_node]; i < diag_i[curr_node+1]; i++) {
      if(curr_node != diag_j[i]) {
	neighbor_color = color_array[diag_j[i]];
	
	if(neighbor_color == QUEUED) {
	  // Since this neighbor is uncolored, increment the number
	  // of neighbors it has that are colored and move its position
	  // in the queue, if necessary.
	  neighbor_node_color_elt = color_queue_array[diag_j[i]].data;
	  neighbor_node_color_elt[1]++;
	  curr_node_degree_index = neighbor_node_color_elt[1]-1;

	  // The node adjacent in the queue will be needed later.
	  prev_elt = color_queue_array[diag_j[i]].prev_elt;

	  // Determine the destination location in the queue.
	  if(degree_tail_array[curr_node_degree_index]) {
	    // Then we are in luck and already know where to put this node.
	    // Make sure the new location is not actually where the node is
	    // now.
	    if(degree_tail_array[curr_node_degree_index] != color_queue_array[diag_j[i]].prev_elt) {
	      // Then a move is in order; else the node is in the right place.
	      moveAfter(&color_queue_array[diag_j[i]],
			degree_tail_array[curr_node_degree_index], color_queue);
	    }
	  } 
	  else {
	    // The value of the pointer to the elements with the same number
	    // of neighbors that have been colored as this node is NULL. That
	    // means that this is the only node with this number of neighbors
	    // that have been colored.
	    //
	    // The situation at this point can be one of two.
	    // 1) This node could be the node with the largest number of
	    //    neighbors that have been colored.
	    // 2) This node is not the node with the largest number of
	    //    neighbors that have been colored.
	    //
	    // The difference between the two is visible by looking at
	    // degree_null_beyond. If this node has more neighbors that have
	    // been colored than the value of degree_null_beyond, then this
	    // is the first case.
	    if(curr_node_degree_index > degree_null_beyond ||
	       degree_null_beyond == 0) {
	      // Then this is the first case. Now this is the node with the
	      // largest number of neighbors that have been colored.
	      //
	      // Additionally, this node is now at the head of the queue,
	      // if it is not already there (i.e., it is not the last element).
	      degree_null_beyond = curr_node_degree_index;
	      if(color_queue->head != &color_queue_array[diag_j[i]])
		moveToHead(&color_queue_array[diag_j[i]], color_queue);
	    }
	    else {
	      // Then this is the second case. This is not the node with
	      // the largest number of neighbors that have already been
	      // colored, but it does have a unique number of neighbors that
	      // have already been colored.
	      //
	      // It needs to be determined where this element is to be moved
	      // in the queue. This is done by looking at
	      // degree_tail_array[curr_node_degree_index+1,...,max_rownnz]
	      // until a non-null element is found in degree_tail_array. The
	      // element that is found will be the one to directly precede
	      // this element in the queue.
	      j = curr_node_degree_index+1;
	      while(!degree_tail_array[j])
		j++;

	      // Now we have the index of the element to precede this node.
	      // Move the node there.
	      moveAfter(&color_queue_array[diag_j[i]], degree_tail_array[j],
			color_queue);
	    }
	  }

	  // Update the degree_tail_array for this element.
	  degree_tail_array[curr_node_degree_index] = &color_queue_array[diag_j[i]];
	  if(curr_node_degree_index > 0 &&
	     degree_tail_array[curr_node_degree_index-1] == &color_queue_array[diag_j[i]]) {
	    // Then the degree_tail_array points to this element already
	    // from before. Remove that pointer.
	    if(prev_elt && prev_elt->data[1] == curr_node_degree_index)
	      degree_tail_array[curr_node_degree_index-1] = prev_elt;
	    else
	      degree_tail_array[curr_node_degree_index-1] = NULL;
	  }

	}
	else {
	  // Take note of this neighbor's color.
	  //used_colors[i - diag_i[curr_node]] = neighbor_color;
	  used_colors[neighbor_color-1] = 1;
	}
      }
    }

    // Color this node based on the information gathered.
    // Search the used_colors array to find the lowest color number that is available.
    for(i = 0; i < *num_colors; i++) {
      if(!used_colors[i]) {
	// Then color i+1 is available. Store that information and exit loop.
	color_array[curr_node] = i+1;
	i = *num_colors;   // This breaks the loop.
      }
    }
    if(color_array[curr_node] == QUEUED) {
      // Then a new color is needed.
      color_array[curr_node] = *num_colors + 1;

      // Since the graph has more colors now, update num_colors.
      (*num_colors)++;
    }

    hypre_TFree(used_colors);
  }
  WriteCoarseningColorInformation(S, color_array, level);

  hypre_TFree(degree_tail_array);
  for(i = 0; i < num_variables; i++)
    hypre_TFree(color_queue_array[i].data); // free all of the node_color_elt
  if(num_variables > 0)
    hypre_TFree(color_queue_array);
  destroyQueue(color_queue);
}
void seqColorGraphD2(hypre_ParCSRMatrix * S, short * color_array, int * num_colors, int level)
     // Distance-two coloring algorithm.
{
#define QUEUED        -1
#define UNCOLORED     0

  hypre_Queue        *color_queue;
  hypre_QueueElement *color_queue_array, *prev_elt;
  hypre_CSRMatrix    *diag;
  int                *node_color_elt, *neighbor_node_color_elt, *diag_i;
  int                *diag_j, *used_colors;
  int                i, j, curr_node, neighbor_color, curr_choice;
  int                num_variables, max_rownnz, curr_node_degree_index;
  int                degree_null_beyond, max_degree_node;
  hypre_QueueElement **degree_tail_array;

  color_queue = newQueue();
  *num_colors = 0;

  diag = hypre_ParCSRMatrixDiag(S);
  diag_i = hypre_CSRMatrixI(diag);
  diag_j = hypre_CSRMatrixJ(diag);

  num_variables = hypre_CSRMatrixNumRows(diag);

  // Determine the maximum degree of the graph and the first node that is of
  // maximum degree.
  max_rownnz = 0;
  for(i = 0; i < num_variables; i++) {
    if(diag_i[i+1]-diag_i[i] > max_rownnz) {
      max_rownnz = diag_i[i+1]-diag_i[i];
      max_degree_node = i;
    }
  }

  // The degree_tail_array contains pointers to specific elements in the
  // queue. For example, degree_tail_array[2] points to the last element
  // in the queue with three neighbors that have already been colored. The
  // pointer is to the last element so that when an element in the queue
  // needs to be moved up, it can be added to the end of the appropriate
  // "section" of elements.
  degree_tail_array = hypre_CTAlloc(hypre_QueueElement*, max_rownnz);
  // degree_null_beyond contains the largest index into degree_tail_array
  // that is not NULL. This way a node can quickly tell if it needs to go
  // to the head of the queue.
  degree_null_beyond = 0;

  if(num_variables > 0) {
    // Create hypre_QueueElements for each node.
    color_queue_array = hypre_CTAlloc(hypre_QueueElement, num_variables);

    // Set the data in each element and then enqueue the node.
    for(i = 0; i < num_variables; i++) {
      node_color_elt = hypre_CTAlloc(int, 2);
      node_color_elt[0] = i;
      node_color_elt[1] = 0; // zero neighbors that have been colored
      color_queue_array[i].data = node_color_elt;
      enqueueElement(&color_queue_array[i], color_queue);
      color_array[i] = QUEUED;
      if(i == max_degree_node)
	moveToHead(&color_queue_array[i], color_queue);
    }
  }

  while(color_queue->head) {
    // while the queue still contains elements
    node_color_elt = dequeue(color_queue);
    curr_node = node_color_elt[0];

    // Check to see if this element was in the degree_tail_array. If it was,
    // then that entry in the degree_tail_array is to be set to NULL. Also,
    // search to find the next entry and update degree_null_beyond.
    if(node_color_elt[1] > 0 &&
       degree_tail_array[node_color_elt[1]-1] == &color_queue_array[curr_node]) {
      // Then this was the entry for its number of neighbors already colored
      // in degree_tail_array.
      degree_tail_array[node_color_elt[1]-1] = NULL;
      if(node_color_elt[1] > 0) {
	j = node_color_elt[1]-1;
	while(j > 0 && !degree_tail_array[j])
	  j--;
	
	// Now we have the index of next element in the degree_tail_array, or
	// the array is all NULL.
	degree_null_beyond = j;
      } else
	degree_null_beyond = 0;
    }

    //used_colors = hypre_CTAlloc(int, diag_i[curr_node+1] - diag_i[curr_node]);
    used_colors = hypre_CTAlloc(int, *num_colors);
    // Get the neighbors of this node and also determine their colors.
    for(i = diag_i[curr_node]; i < diag_i[curr_node+1]; i++) {
      if(curr_node != diag_j[i]) {
	neighbor_color = color_array[diag_j[i]];

	// Determine the color of each of the neighbor node's neighbors and mark those
	// as used colors. This is the step the enforces the distance-two coloring.
	int neighbor_id, neighbors_neighbor_color, k;
	neighbor_id = diag_j[i];
	for(k = diag_i[neighbor_id]; k < diag_i[neighbor_id+1]; k++) {
	  neighbors_neighbor_color = color_array[diag_j[k]];
	  if(neighbors_neighbor_color > 0)
	    used_colors[neighbors_neighbor_color-1] = 1;
	}
	
	if(neighbor_color == QUEUED) {
	  // Since this neighbor is uncolored, increment the number
	  // of neighbors it has that are colored and move its position
	  // in the queue, if necessary.
	  neighbor_node_color_elt = color_queue_array[diag_j[i]].data;
	  neighbor_node_color_elt[1]++;
	  curr_node_degree_index = neighbor_node_color_elt[1]-1;

	  // The node adjacent in the queue will be needed later.
	  prev_elt = color_queue_array[diag_j[i]].prev_elt;

	  // Determine the destination location in the queue.
	  if(degree_tail_array[curr_node_degree_index]) {
	    // Then we are in luck and already know where to put this node.
	    // Make sure the new location is not actually where the node is
	    // now.
	    if(degree_tail_array[curr_node_degree_index] != color_queue_array[diag_j[i]].prev_elt) {
	      // Then a move is in order; else the node is in the right place.
	      moveAfter(&color_queue_array[diag_j[i]],
			degree_tail_array[curr_node_degree_index], color_queue);
	    }
	  } 
	  else {
	    // The value of the pointer to the elements with the same number
	    // of neighbors that have been colored as this node is NULL. That
	    // means that this is the only node with this number of neighbors
	    // that have been colored.
	    //
	    // The situation at this point can be one of two.
	    // 1) This node could be the node with the largest number of
	    //    neighbors that have been colored.
	    // 2) This node is not the node with the largest number of
	    //    neighbors that have been colored.
	    //
	    // The difference between the two is visible by looking at
	    // degree_null_beyond. If this node has more neighbors that have
	    // been colored than the value of degree_null_beyond, then this
	    // is the first case.
	    if(curr_node_degree_index > degree_null_beyond ||
	       degree_null_beyond == 0) {
	      // Then this is the first case. Now this is the node with the
	      // largest number of neighbors that have been colored.
	      //
	      // Additionally, this node is now at the head of the queue,
	      // if it is not already there (i.e., it is not the last element).
	      degree_null_beyond = curr_node_degree_index;
	      if(color_queue->head != &color_queue_array[diag_j[i]])
		moveToHead(&color_queue_array[diag_j[i]], color_queue);
	    }
	    else {
	      // Then this is the second case. This is not the node with
	      // the largest number of neighbors that have already been
	      // colored, but it does have a unique number of neighbors that
	      // have already been colored.
	      //
	      // It needs to be determined where this element is to be moved
	      // in the queue. This is done by looking at
	      // degree_tail_array[curr_node_degree_index+1,...,max_rownnz]
	      // until a non-null element is found in degree_tail_array. The
	      // element that is found will be the one to directly precede
	      // this element in the queue.
	      j = curr_node_degree_index+1;
	      while(!degree_tail_array[j])
		j++;

	      // Now we have the index of the element to precede this node.
	      // Move the node there.
	      moveAfter(&color_queue_array[diag_j[i]], degree_tail_array[j],
			color_queue);
	    }
	  }

	  // Update the degree_tail_array for this element.
	  degree_tail_array[curr_node_degree_index] = &color_queue_array[diag_j[i]];
	  if(curr_node_degree_index > 0 &&
	     degree_tail_array[curr_node_degree_index-1] == &color_queue_array[diag_j[i]]) {
	    // Then the degree_tail_array points to this element already
	    // from before. Remove that pointer.
	    if(prev_elt && prev_elt->data[1] == curr_node_degree_index)
	      degree_tail_array[curr_node_degree_index-1] = prev_elt;
	    else
	      degree_tail_array[curr_node_degree_index-1] = NULL;
	  }

	}
	else {
	  // Take note of this neighbor's color.
	  //used_colors[i - diag_i[curr_node]] = neighbor_color;
	  used_colors[neighbor_color-1] = 1;
	}
      }
    }

    // Color this node based on the information gathered.
    // Search the used_colors array to find the lowest color number that is available.
    for(i = 0; i < *num_colors; i++) {
      if(!used_colors[i]) {
	// Then color i+1 is available. Store that information and exit loop.
	color_array[curr_node] = i+1;
	i = *num_colors;   // This breaks the loop.
      }
    }
    if(color_array[curr_node] == QUEUED) {
      // Then a new color is needed.
      color_array[curr_node] = *num_colors + 1;

      // Since the graph has more colors now, update num_colors.
      (*num_colors)++;
    }

    hypre_TFree(used_colors);
  }
  WriteCoarseningColorInformation(S, color_array, level);

  hypre_TFree(degree_tail_array);
  for(i = 0; i < num_variables; i++)
    hypre_TFree(color_queue_array[i].data); // free all of the node_color_elt
  if(num_variables > 0)
    hypre_TFree(color_queue_array);
  destroyQueue(color_queue);
}

void seqColorGraphOrig(hypre_ParCSRMatrix * S, short * color_array, int * num_colors, int level)
{
#define QUEUED        -1
#define UNCOLORED     0

  hypre_Queue        *color_queue;
  hypre_QueueElement *color_queue_array, *prev_elt;
  hypre_CSRMatrix    *diag;
  int                *node_color_elt, *diag_i, *diag_j, *used_colors;
  int                i, curr_node, neighbor_color, curr_choice, num_variables;
  int max_rownnz;

  color_queue = newQueue();
  *num_colors = 0;

  diag = hypre_ParCSRMatrixDiag(S);
  diag_i = hypre_CSRMatrixI(diag);
  diag_j = hypre_CSRMatrixJ(diag);

  num_variables = hypre_CSRMatrixNumRows(diag);

  max_rownnz = 0;
  for(i = 0; i < num_variables; i++) {
    if(diag_i[i+1]-diag_i[i] > max_rownnz)
      max_rownnz = diag_i[i+1]-diag_i[i];
  }

  if(num_variables > 0) {

    // Create hypre_QueueElements for each node.
    color_queue_array = hypre_CTAlloc(hypre_QueueElement, num_variables);

    // Set the data in each element and then enqueue the node.
    for(i = 0; i < num_variables; i++) {
      node_color_elt = hypre_CTAlloc(int, 2);
      node_color_elt[0] = i;
      node_color_elt[1] = 0; // zero neighbors that have been colored
      color_queue_array[i].data = node_color_elt;
    }
    enqueueElement(&color_queue_array[0], color_queue);
    color_array[0] = QUEUED;
  }

  while(color_queue->head) {
    // while the queue still contains elements
    // color this node and enqueue all of its uncolored neighbors
    node_color_elt = dequeue(color_queue);

    curr_node = node_color_elt[0];
    used_colors = hypre_CTAlloc(int, diag_i[curr_node+1] - diag_i[curr_node]);
    // Get the neighbors of this node and also determine their colors.
    for(i = diag_i[curr_node]; i < diag_i[curr_node+1]; i++) {
      if(curr_node != diag_j[i]) {
	neighbor_color = color_array[diag_j[i]];
	
	if(neighbor_color == UNCOLORED || neighbor_color == QUEUED) {
	  if(neighbor_color == UNCOLORED) {
	    enqueueElement(&color_queue_array[diag_j[i]], color_queue);
	    color_array[diag_j[i]] = QUEUED;
	  }

	  // Since this neighbor is uncolored, increment the number
	  // of neighbors it has that are colored and move its position
	  // in the queue, if necessary.
	  color_queue_array[diag_j[i]].data[1]++;
	  prev_elt = color_queue_array[diag_j[i]].prev_elt;

	  while(prev_elt && prev_elt->data[1] < color_queue_array[diag_j[i]].data[1]) {
	    prev_elt = prev_elt->prev_elt;
	  }
	  if(prev_elt != color_queue_array[diag_j[i]].prev_elt)
	    moveAfter(&color_queue_array[diag_j[i]], prev_elt, color_queue);
	  // else there is no moving to be done
	}
	else {
	  // Take note of this neighbor's color.
	  used_colors[i - diag_i[curr_node]] = neighbor_color;
	}
      }
    }

    // Color this node based on the information gathered.
    // Sort the used_colors array.
    qsort0(used_colors, 0, diag_i[curr_node+1] - diag_i[curr_node] - 1);

    // Now search the used_colors array to find the lowest number > 0 that
    // does not appear. Make that number the color of curr_node.
    curr_choice = 1;
    for(i = 0; i < diag_i[curr_node+1] - diag_i[curr_node]; i++) {
      if(used_colors[i] == curr_choice)
	// Then the current choice of color is in use. Pick the next one up.
	curr_choice++;
      else if(used_colors[i] > curr_choice) {
	// The the current choice of color is available. Exit the loop and
	// color curr_node this color.
	i = diag_i[curr_node+1] - diag_i[curr_node]; // to break loop
      }
    }
    color_array[curr_node] = curr_choice;
    if(curr_choice > *num_colors)
      *num_colors = curr_choice;
    
    hypre_TFree(used_colors);
  }
  WriteCoarseningColorInformation(S, color_array, level);

  destroyQueue(color_queue);
  hypre_TFree(color_queue_array);
  hypre_TFree(node_color_elt);
}

int
hypre_BoomerAMGCoarsenCLJP_c( hypre_ParCSRMatrix    *S,
                        hypre_ParCSRMatrix    *A,
                        int                    CF_init,
                        int                    debug_flag,
                        int                  **CF_marker_ptr,
			int                    global,
			int                    level,
			double                *measure_array)
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle   *comm_handle;

   hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
   int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
   int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
   int                *S_offd_j;

   int 		      *col_map_offd    = hypre_ParCSRMatrixColMapOffd(S);
   int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
   int		       col_1 = hypre_ParCSRMatrixFirstColDiag(S);
   int		       col_n = col_1 + hypre_CSRMatrixNumCols(S_diag);
   int 		       num_cols_offd = 0;
                  
   hypre_CSRMatrix    *S_ext;
   int                *S_ext_i;
   int                *S_ext_j;

   int		       num_sends = 0;
   int  	      *int_buf_data;
   double	      *buf_data;

   int                *CF_marker;
   int                *CF_marker_offd;

   short              *color_array = NULL;
   int                num_colors;
                      
   //double             *measure_array;
   int                *graph_array;
   int                *graph_array_offd;
   int                 graph_size;
   int                 graph_offd_size;
   int                 global_graph_size;
                      
   int                 i, j, k, kc, jS, kS, ig;
   int		       index, start, my_id, num_procs, jrow, cnt;
                      
   int                 ierr = 0;
   int                 break_var = 1;

   double	    wall_time;
#ifdef FINE_GRAIN_TIMINGS
   double           my_setup_wall_time, my_update_wall_time, my_search_wall_time,
                    setup_time=0, update_time=0, search_time=0;
#endif

   int   iter = 0;

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   int   iter = 0;
#endif

#ifdef FINE_GRAIN_TIMINGS
my_setup_wall_time = time_getWallclockSeconds();
#endif
   if(!measure_array) {
     // Then the coloring needs to be done. The only time this is not needed is if
     // the calling function already has the measure_array computed. The CR function
     // does this because it calls CLJP_c several times for each level. By computing
     // the measure_array ahead of time, extra computation can be saved.
     color_array = hypre_CTAlloc(short, num_variables);

     num_colors = 0;
/*      if(global) */
/*        //parColorGraph(A, S, color_array, &num_colors, level); */
/*        parColorGraphNew(A, S, color_array, &num_colors, level); */
/*      else */
/*        //seqColorGraphNew(S, color_array, &num_colors, level); */
/*        seqColorGraphTheFinalWord(A, color_array, &num_colors, 0, level); */
     //seqColorGraphNew(S, color_array, &num_colors, level);
     seqColorGraphTheFinalWord(A, color_array, &num_colors, 0, level);
   }
   
   /*--------------------------------------------------------------
    * Compute a  ParCSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   S_ext = NULL;
   if (debug_flag == 3) wall_time = time_getWallclockSeconds();
   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);
   if (!comm_pkg)
   {
        hypre_MatvecCommPkgCreate(A);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
   buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
 
   num_cols_offd = hypre_CSRMatrixNumCols(S_offd);

   S_diag_j = hypre_CSRMatrixJ(S_diag);

   if (num_cols_offd)
   {
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

   if(!measure_array) {
     // Then the measure array needs to be computed. The CR function
     // computes the measure_array ahead of time to save extra
     // computation.
   measure_array = hypre_CTAlloc(double, num_variables+num_cols_offd);

   for (i=0; i < S_offd_i[num_variables]; i++)
   {
      measure_array[num_variables + S_offd_j[i]] += 1.0;
   }
   if (num_procs > 1)
   comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg,
                        &measure_array[num_variables], buf_data);

   for (i=0; i < S_diag_i[num_variables]; i++)
   {
      measure_array[S_diag_j[i]] += 1.0;
   }

   if (num_procs > 1)
   hypre_ParCSRCommHandleDestroy(comm_handle);
      
   index = 0;
   for (i=0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
                        += buf_data[index++];
   }

   for (i=num_variables; i < num_variables+num_cols_offd; i++)
   /* This loop zeros out the measures for the off-process nodes since
      this process is not responsible for . */
   {
      measure_array[i] = 0;
   }

   /* this augments the measures */
   //hypre_BoomerAMGIndepSetInit(S, measure_array);
   hypre_BoomerAMGIndepSetInitb(S, measure_array, color_array, num_colors);
   }

   /*---------------------------------------------------
    * Initialize the graph array
    * graph_array contains interior points in elements 0 ... num_variables-1
    * followed by boundary values
    *---------------------------------------------------*/

   graph_array = hypre_CTAlloc(int, num_variables);
   if (num_cols_offd)
      graph_array_offd = hypre_CTAlloc(int, num_cols_offd);
   else
      graph_array_offd = NULL;

   /* initialize measure array and graph array */

   for (ig = 0; ig < num_cols_offd; ig++)
      graph_array_offd[ig] = ig;

   /*---------------------------------------------------
    * Initialize the C/F marker array
    * C/F marker array contains interior points in elements 0 ...
    * num_variables-1  followed by boundary values
    *---------------------------------------------------*/

   graph_offd_size = num_cols_offd;

   if (CF_init)
   {
      CF_marker = *CF_marker_ptr;
      cnt = 0;
      for (i=0; i < num_variables; i++)
      {
         if ( (S_offd_i[i+1]-S_offd_i[i]) > 0
                 || CF_marker[i] == -1)
         {
	   if(CF_marker[i] != SF_PT)
            CF_marker[i] = 0;
         }
         if ( CF_marker[i] == Z_PT)
         {
            if (measure_array[i] >= 1.0 ||
                (S_diag_i[i+1]-S_diag_i[i]) > 0)
            {
               CF_marker[i] = 0;
               graph_array[cnt++] = i;
            }
            else
            {
               graph_size--;
               CF_marker[i] = F_PT;
            }
         }
         else if (CF_marker[i] == SF_PT)
	    measure_array[i] = 0;
         else
            graph_array[cnt++] = i;
      }
   }
   else
   {
      CF_marker = hypre_CTAlloc(int, num_variables);
      cnt = 0;
      for (i=0; i < num_variables; i++)
      {
	 CF_marker[i] = 0;
	 if ( (S_diag_i[i+1]-S_diag_i[i]) == 0
		&& (S_offd_i[i+1]-S_offd_i[i]) == 0)
	 {
	    CF_marker[i] = SF_PT;
	    measure_array[i] = 0;
	 }
	 else
            graph_array[cnt++] = i;
      }
   }
   graph_size = cnt;
   if (num_cols_offd)
      CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
   else
      CF_marker_offd = NULL;
   for (i=0; i < num_cols_offd; i++)
	CF_marker_offd[i] = 0;
  
   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   if (num_procs > 1)
   {
      S_ext      = hypre_ParCSRMatrixExtractBExt(S,A,0);
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixJ(S_ext);
   }

   /*  compress S_ext  and convert column numbers*/

   index = 0;
   for (i=0; i < num_cols_offd; i++)
   {
      for (j=S_ext_i[i]; j < S_ext_i[i+1]; j++)
      {
	 k = S_ext_j[j];
	 if (k >= col_1 && k < col_n)
	 {
	    S_ext_j[index++] = k - col_1;
	 }
	 else
	 {
	    kc = hypre_BinarySearch(col_map_offd,k,num_cols_offd);
	    if (kc > -1) S_ext_j[index++] = -kc-1;
	 }
      }
      S_ext_i[i] = index;
   }
   for (i = num_cols_offd; i > 0; i--)
      S_ext_i[i] = S_ext_i[i-1];
   if (num_procs > 1) S_ext_i[0] = 0;

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Initialize CLJP phase = %f\n",
                     my_id, wall_time);
   }
#ifdef FINE_GRAIN_TIMINGS
setup_time += time_getWallclockSeconds() - my_setup_wall_time;
#endif

   while (1)
   {
      /*------------------------------------------------
       * Exchange boundary data, i.i. get measures and S_ext_data
       *------------------------------------------------*/

      if (num_procs > 1)
   	 comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg,
                        &measure_array[num_variables], buf_data);

      if (num_procs > 1)
   	 hypre_ParCSRCommHandleDestroy(comm_handle);
      
      index = 0;
      for (i=0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
                        += buf_data[index++];
      }

      /*------------------------------------------------
       * Set F-pts and update subgraph
       *------------------------------------------------*/
 
#ifdef FINE_GRAIN_TIMINGS
my_update_wall_time = time_getWallclockSeconds();
#endif
      if (iter || !CF_init)
      {
         for (ig = 0; ig < graph_size; ig++)
         {
            i = graph_array[ig];

            if ( (CF_marker[i] != C_PT) && (measure_array[i] < 1) )
            {
               /* set to be an F-pt */
               CF_marker[i] = F_PT;
 
	       /* make sure all dependencies have been accounted for */
               for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     CF_marker[i] = 0;
                  }
               }
               for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     CF_marker[i] = 0;
                  }
               }
            }
            if (CF_marker[i])
            {
               measure_array[i] = 0;
 
               /* take point out of the subgraph */
               graph_size--;
               graph_array[ig] = graph_array[graph_size];
               graph_array[graph_size] = i;
               ig--;
            }
         }
      }
#ifdef FINE_GRAIN_TIMINGS
update_time += time_getWallclockSeconds() - my_update_wall_time;
#endif
 
      /*------------------------------------------------
       * Exchange boundary data, i.i. get measures
       *------------------------------------------------*/

      if (debug_flag == 3) wall_time = time_getWallclockSeconds();

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
        {
            jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            buf_data[index++] = measure_array[jrow];
         }
      }

      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg, buf_data,
        	&measure_array[num_variables]);
 
         hypre_ParCSRCommHandleDestroy(comm_handle);
 
      }
      /*------------------------------------------------
       * Debugging:
       *
       * Uncomment the sections of code labeled
       * "debugging" to generate several files that
       * can be visualized using the `coarsen.m'
       * matlab routine.
       *------------------------------------------------*/

#if 0 /* debugging */
      /* print out measures */
      char filename[50];
      FILE * fp;
      sprintf(filename, "coarsen.out.measures.%04d", iter);
      fp = fopen(filename, "w");
      for (i = 0; i < num_variables; i++)
      {
         fprintf(fp, "%f\n", measure_array[i]);
      }
      fclose(fp);

      /* print out strength matrix */
      sprintf(filename, "coarsen.out.strength.%04d", iter);
      hypre_CSRMatrixPrint(S, filename);

      /* print out C/F marker */
      sprintf(filename, "coarsen.out.CF.%04d", iter);
      fp = fopen(filename, "w");
      for (i = 0; i < num_variables; i++)
      {
         fprintf(fp, "%d\n", CF_marker[i]);
      }
      fclose(fp);

      //iter++;
#endif

      /*------------------------------------------------
       * Test for convergence
       *------------------------------------------------*/

      MPI_Allreduce(&graph_size,&global_graph_size,1,MPI_INT,MPI_SUM,comm);

#ifdef FINE_GRAIN_TIMINGS
my_update_wall_time = time_getWallclockSeconds();
#endif
      if (global_graph_size == 0)
         break;
#ifdef FINE_GRAIN_TIMINGS
update_time += time_getWallclockSeconds() - my_update_wall_time;
#endif

      /*------------------------------------------------
       * Pick an independent set of points with
       * maximal measure.
       *------------------------------------------------*/
#ifdef FINE_GRAIN_TIMINGS
my_search_wall_time = time_getWallclockSeconds();
#endif
      if (iter || !CF_init)
         hypre_BoomerAMGIndepSet(S, measure_array, graph_array,
				graph_size,
				graph_array_offd, graph_offd_size,
				CF_marker, CF_marker_offd);

      iter++;
#ifdef FINE_GRAIN_TIMINGS
search_time += time_getWallclockSeconds() - my_search_wall_time;
#endif
      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/

#ifdef FINE_GRAIN_TIMINGS
my_update_wall_time = time_getWallclockSeconds();
#endif
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                int_buf_data[index++]
                 = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
 
      if (num_procs > 1)
      {
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
        CF_marker_offd);
 
      hypre_ParCSRCommHandleDestroy(comm_handle);
      }
 
      for (ig = 0; ig < graph_offd_size; ig++)
      {
         i = graph_array_offd[ig];

         if (CF_marker_offd[i] < 0)
         {
            /* take point out of the subgraph */
            graph_offd_size--;
            graph_array_offd[ig] = graph_array_offd[graph_offd_size];
            graph_array_offd[graph_offd_size] = i;
            ig--;
         }
      }
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d  iter %d  comm. and subgraph update = %f\n",
                     my_id, iter, wall_time);
      }
      /*------------------------------------------------
       * Set C_pts and apply heuristics.
       *------------------------------------------------*/

      for (i=num_variables; i < num_variables+num_cols_offd; i++)
      {
         measure_array[i] = 0;
      }

      if (debug_flag == 3) wall_time = time_getWallclockSeconds();
      for (ig = 0; ig < graph_size; ig++)
      {
         i = graph_array[ig];

         /*---------------------------------------------
          * Heuristic: C-pts don't interpolate from
          * neighbors that influence them.
          *---------------------------------------------*/

         if (CF_marker[i] > 0)
         {
            /* set to be a C-pt */
            CF_marker[i] = C_PT;

            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
            {
               j = S_diag_j[jS];
               if (j > -1)
               {
               
                  /* "remove" edge from S */
                  S_diag_j[jS] = -S_diag_j[jS]-1;
             
                  /* decrement measures of unmarked neighbors */
                  if (!CF_marker[j])
                  {
                     measure_array[j]--;
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               j = S_offd_j[jS];
               if (j > -1)
               {
             
                  /* "remove" edge from S */
                  S_offd_j[jS] = -S_offd_j[jS]-1;
               
                  /* decrement measures of unmarked neighbors */
                  if (!CF_marker_offd[j])
                  {
                     measure_array[j+num_variables]--;
                  }
               }
            }
         }
	 else
    	 {
            /* marked dependencies */
            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
            {
               j = S_diag_j[jS];
	       if (j < 0) j = -j-1;
   
               if (CF_marker[j] > 0)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_diag_j[jS] = -S_diag_j[jS]-1;
                  }
   
                  /* IMPORTANT: consider all dependencies */
                  /* temporarily modify CF_marker */
                  CF_marker[j] = COMMON_C_PT;
               }
               else if (CF_marker[j] == SF_PT)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_diag_j[jS] = -S_diag_j[jS]-1;
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               j = S_offd_j[jS];
	       if (j < 0) j = -j-1;
   
               if (CF_marker_offd[j] > 0)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_offd_j[jS] = -S_offd_j[jS]-1;
                  }
   
                  /* IMPORTANT: consider all dependencies */
                  /* temporarily modify CF_marker */
                  CF_marker_offd[j] = COMMON_C_PT;
               }
               else if (CF_marker_offd[j] == SF_PT)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_offd_j[jS] = -S_offd_j[jS]-1;
                  }
               }
            }
   
            /* unmarked dependencies */
            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
            {
               if (S_diag_j[jS] > -1)
               {
                  j = S_diag_j[jS];
   	          break_var = 1;
                  /* check for common C-pt */
                  for (kS = S_diag_i[j]; kS < S_diag_i[j+1]; kS++)
                  {
                     k = S_diag_j[kS];
		     if (k < 0) k = -k-1;
   
                     /* IMPORTANT: consider all dependencies */
                     if (CF_marker[k] == COMMON_C_PT)
                     {
                        /* "remove" edge from S and update measure*/
                        S_diag_j[jS] = -S_diag_j[jS]-1;
                        measure_array[j]--;
                        break_var = 0;
                        break;
                     }
                  }
   		  if (break_var)
                  {
                     for (kS = S_offd_i[j]; kS < S_offd_i[j+1]; kS++)
                     {
                        k = S_offd_j[kS];
		        if (k < 0) k = -k-1;
   
                        /* IMPORTANT: consider all dependencies */
                        if ( CF_marker_offd[k] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_diag_j[jS] = -S_diag_j[jS]-1;
                           measure_array[j]--;
                           break;
                        }
                     }
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               if (S_offd_j[jS] > -1)
               {
                  j = S_offd_j[jS];
   
                  /* check for common C-pt */
                  for (kS = S_ext_i[j]; kS < S_ext_i[j+1]; kS++)
                  {
                     k = S_ext_j[kS];
   	             if (k >= 0)
   		     {
                        /* IMPORTANT: consider all dependencies */
                        if (CF_marker[k] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_offd_j[jS] = -S_offd_j[jS]-1;
                           measure_array[j+num_variables]--;
                           break;
                        }
                     }
   		     else
   		     {
   		        kc = -k-1;
   		        if (kc > -1 && CF_marker_offd[kc] == COMMON_C_PT)
   		        {
                           /* "remove" edge from S and update measure*/
                           S_offd_j[jS] = -S_offd_j[jS]-1;
                           measure_array[j+num_variables]--;
                           break;
   		        }
   		     }
                  }
               }
            }
         }

         /* reset CF_marker */
	 for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
	 {
            j = S_diag_j[jS];
	    if (j < 0) j = -j-1;

            if (CF_marker[j] == COMMON_C_PT)
            {
               CF_marker[j] = C_PT;
            }
         }
         for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
         {
            j = S_offd_j[jS];
	    if (j < 0) j = -j-1;

            if (CF_marker_offd[j] == COMMON_C_PT)
            {
               CF_marker_offd[j] = C_PT;
            }
         }
      }
#ifdef FINE_GRAIN_TIMINGS
update_time += time_getWallclockSeconds() - my_update_wall_time;
#endif
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d    CLJP phase = %f graph_size = %d nc_offd = %d\n",
                     my_id, wall_time, graph_size, num_cols_offd);
      }
   }

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

#ifdef FINE_GRAIN_TIMINGS
my_setup_wall_time = time_getWallclockSeconds();
#endif
   /* Reset S_matrix */
   for (i=0; i < S_diag_i[num_variables]; i++)
   {
      if (S_diag_j[i] < 0)
         S_diag_j[i] = -S_diag_j[i]-1;
   }
   for (i=0; i < S_offd_i[num_variables]; i++)
   {
      if (S_offd_j[i] < 0)
         S_offd_j[i] = -S_offd_j[i]-1;
   }
/*    for (i=0; i < num_variables; i++) */
/*       if (CF_marker[i] == SF_PT) CF_marker[i] = F_PT; */

   if(color_array)
     hypre_TFree(color_array);

   hypre_TFree(measure_array);
   hypre_TFree(graph_array);
   if (num_cols_offd) hypre_TFree(graph_array_offd);
   hypre_TFree(buf_data);
   hypre_TFree(int_buf_data);
   hypre_TFree(CF_marker_offd);
   if (num_procs > 1) hypre_CSRMatrixDestroy(S_ext);

   *CF_marker_ptr   = CF_marker;
#ifdef FINE_GRAIN_TIMINGS
setup_time += time_getWallclockSeconds() - my_setup_wall_time;
#endif

#ifdef FINE_GRAIN_TIMINGS
printf(" Setup time: %f\n", setup_time);
printf(" Update time: %f\n", update_time);
printf(" Search time: %f\n", search_time);
#endif

   return (ierr);
}

int
hypre_BoomerAMGCoarsenCLJP_c_improved( hypre_ParCSRMatrix    *S,
                        hypre_ParCSRMatrix    *A,
                        int                    CF_init,
                        int                    debug_flag,
                        int                  **CF_marker_ptr,
			int                    global,
			int                    level,
			int                   *measure_array)
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle   *comm_handle;

   hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
   int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
   int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
   int                *S_offd_j;

   int 		      *col_map_offd    = hypre_ParCSRMatrixColMapOffd(S);
   int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
   int		       col_1 = hypre_ParCSRMatrixFirstColDiag(S);
   int		       col_n = col_1 + hypre_CSRMatrixNumCols(S_diag);
   int 		       num_cols_offd = 0;
                  
   hypre_CSRMatrix    *S_ext;
   int                *S_ext_i;
   int                *S_ext_j;

   int		       num_sends = 0;
   int  	      *int_buf_data;
   double	      *buf_data;

   int                *CF_marker;
   int                *CF_marker_offd;

   short              *color_array = NULL;
   short              *boundary_color_array = NULL;
   int                num_colors;
                      
   //double             *measure_array;
   int                *graph_array;
   int                *graph_array_offd;
   int                 graph_size;
   int                 graph_offd_size;
   int                 global_graph_size;
                      
   int                 i, j, k, kc, jS, kS, ig;
   int		       index, start, my_id, num_procs, jrow, cnt;
                      
   int                 ierr = 0;
   int                 break_var = 1;

   double	    wall_time;
   double           my_setup_wall_time, my_update_wall_time, my_search_wall_time,
                    setup_time=0, update_time=0, search_time=0;

   int   iter = 0;

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   int   iter = 0;
#endif

my_setup_wall_time = time_getWallclockSeconds();
   MPI_Comm_size(comm,&num_procs);
   num_cols_offd = hypre_CSRMatrixNumCols(S_offd);

   if(!measure_array) {
     // Then the coloring needs to be done. The only time this is not needed is if
     // the calling function already has the measure_array computed. The CR function
     // does this because it calls CLJP_c several times for each level. By computing
     // the measure_array ahead of time, extra computation can be saved.
     color_array = hypre_CTAlloc(short, num_variables+num_cols_offd);

     if(num_procs > 1) {
       num_colors = 0;
       boundary_color_array = hypre_CTAlloc(short, num_variables+num_cols_offd);
       parColorGraphNewBoundariesOnly(A, S, boundary_color_array,
				      &num_colors);
     }
     num_colors = 0;
     seqColorGraphTheFinalWord(A, color_array, &num_colors, 0, level);
   }
   
   /*--------------------------------------------------------------
    * Compute a  ParCSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   S_ext = NULL;
   if (debug_flag == 3) wall_time = time_getWallclockSeconds();
   MPI_Comm_rank(comm,&my_id);
   if (!comm_pkg)
   {
        hypre_MatvecCommPkgCreate(A);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
   buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
 
   S_diag_j = hypre_CSRMatrixJ(S_diag);

   if (num_cols_offd)
   {
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

   if(!measure_array) {
     // Then the measure array needs to be computed. The CR function
     // computes the measure_array ahead of time to save extra
     // computation.
   measure_array = hypre_CTAlloc(int, num_variables+num_cols_offd);

   for (i=0; i < S_offd_i[num_variables]; i++)
   {
      measure_array[num_variables + S_offd_j[i]] ++;
   }
   if (num_procs > 1)
     comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg,
                        &measure_array[num_variables], int_buf_data);

   for (i=0; i < S_diag_i[num_variables]; i++)
   {
      measure_array[S_diag_j[i]] ++;
   }

   if (num_procs > 1)
     hypre_ParCSRCommHandleDestroy(comm_handle);
      
   index = 0;
   for (i=0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
                        += int_buf_data[index++];
   }


   // Get colors of off-processor neighbors.
   int * int_recv_data = hypre_CTAlloc(int, num_cols_offd);
   index = 0;
   for (i = 0; i < num_sends; i++) {
     start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
     for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++) {
       jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
       int_buf_data[index++] = color_array[jrow];
     }
   }
   if(num_procs > 1) {
     comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
        	int_recv_data);
 
     hypre_ParCSRCommHandleDestroy(comm_handle);

     for(i = 0; i < num_cols_offd; i++)
       color_array[num_variables+i] = int_recv_data[i];
   }

   // Get boundary colors of off-processor neighbors.
   index = 0;
   for (i = 0; i < num_sends; i++) {
     start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
     for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++) {
       jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
       int_buf_data[index++] = boundary_color_array[jrow];
     }
   }
   if(num_procs > 1) {
     comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
        	int_recv_data);
 
     hypre_ParCSRCommHandleDestroy(comm_handle);

     for(i = 0; i < num_cols_offd; i++)
       boundary_color_array[num_variables+i] = int_recv_data[i];
   }
   hypre_TFree(int_recv_data);

   for (i=num_variables; i < num_variables+num_cols_offd; i++)
   /* This loop zeros out the measures for the off-process nodes since
      this process is not responsible for . */
   {
      measure_array[i] = 0;
   }
   }

   /*---------------------------------------------------
    * Initialize the graph array
    * graph_array contains interior points in elements 0 ... num_variables-1
    * followed by boundary values
    *---------------------------------------------------*/

   graph_array = hypre_CTAlloc(int, num_variables);
   if (num_cols_offd)
      graph_array_offd = hypre_CTAlloc(int, num_cols_offd);
   else
      graph_array_offd = NULL;

   /* initialize measure array and graph array */

   for (ig = 0; ig < num_cols_offd; ig++)
      graph_array_offd[ig] = ig;

   /*---------------------------------------------------
    * Initialize the C/F marker array
    * C/F marker array contains interior points in elements 0 ...
    * num_variables-1  followed by boundary values
    *---------------------------------------------------*/

   graph_offd_size = num_cols_offd;

   if (CF_init)
   {
      CF_marker = *CF_marker_ptr;
      cnt = 0;
      for (i=0; i < num_variables; i++)
      {
         if ( (S_offd_i[i+1]-S_offd_i[i]) > 0
                 || CF_marker[i] == -1)
         {
	   if(CF_marker[i] != SF_PT)
            CF_marker[i] = 0;
         }
         if ( CF_marker[i] == Z_PT)
         {
            if (measure_array[i] >= 1 ||
                (S_diag_i[i+1]-S_diag_i[i]) > 0)
            {
               CF_marker[i] = 0;
               graph_array[cnt++] = i;
            }
            else
            {
               graph_size--;
               CF_marker[i] = F_PT;
            }
         }
         else if (CF_marker[i] == SF_PT)
	    measure_array[i] = 0;
         else
            graph_array[cnt++] = i;
      }
   }
   else
   {
      CF_marker = hypre_CTAlloc(int, num_variables);
      cnt = 0;
      for (i=0; i < num_variables; i++)
      {
	 CF_marker[i] = 0;
	 if ( (S_diag_i[i+1]-S_diag_i[i]) == 0
		&& (S_offd_i[i+1]-S_offd_i[i]) == 0)
	 {
	    CF_marker[i] = SF_PT;
	    measure_array[i] = 0;
	 }
	 else
            graph_array[cnt++] = i;
      }
   }
   graph_size = cnt;
   if (num_cols_offd)
      CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
   else
      CF_marker_offd = NULL;
   for (i=0; i < num_cols_offd; i++)
	CF_marker_offd[i] = 0;
  
   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   if (num_procs > 1)
   {
      S_ext      = hypre_ParCSRMatrixExtractBExt(S,A,0);
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixJ(S_ext);
   }

   /*  compress S_ext  and convert column numbers*/

   index = 0;
   for (i=0; i < num_cols_offd; i++)
   {
      for (j=S_ext_i[i]; j < S_ext_i[i+1]; j++)
      {
	 k = S_ext_j[j];
	 if (k >= col_1 && k < col_n)
	 {
	    S_ext_j[index++] = k - col_1;
	 }
	 else
	 {
	    kc = hypre_BinarySearch(col_map_offd,k,num_cols_offd);
	    if (kc > -1) S_ext_j[index++] = -kc-1;
	 }
      }
      S_ext_i[i] = index;
   }
   for (i = num_cols_offd; i > 0; i--)
      S_ext_i[i] = S_ext_i[i-1];
   if (num_procs > 1) S_ext_i[0] = 0;

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Initialize CLJP phase = %f\n",
                     my_id, wall_time);
   }
setup_time += time_getWallclockSeconds() - my_setup_wall_time;

   while (1)
   {
      /*------------------------------------------------
       * Exchange boundary data, i.i. get measures and S_ext_data
       *------------------------------------------------*/

      if (num_procs > 1)
   	 comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg,
                        &measure_array[num_variables], int_buf_data);

      if (num_procs > 1)
   	 hypre_ParCSRCommHandleDestroy(comm_handle);
      
      index = 0;
      for (i=0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
                        += int_buf_data[index++];
      }

      /*------------------------------------------------
       * Set F-pts and update subgraph
       *------------------------------------------------*/
 
my_update_wall_time = time_getWallclockSeconds();
      if (iter || !CF_init)
      {
         for (ig = 0; ig < graph_size; ig++)
         {
            i = graph_array[ig];

            if ( (CF_marker[i] != C_PT) && (measure_array[i] == 0) )
            {
               /* set to be an F-pt */
               CF_marker[i] = F_PT;
 
	       /* make sure all dependencies have been accounted for */
               for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     CF_marker[i] = 0;
                  }
               }
               for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     CF_marker[i] = 0;
                  }
               }
            }
            if (CF_marker[i])
            {
               measure_array[i] = 0;
 
               /* take point out of the subgraph */
               graph_size--;
               graph_array[ig] = graph_array[graph_size];
               graph_array[graph_size] = i;
               ig--;
            }
         }
      }
update_time += time_getWallclockSeconds() - my_update_wall_time;
 
      /*------------------------------------------------
       * Exchange boundary data, i.i. get measures
       *------------------------------------------------*/

      if (debug_flag == 3) wall_time = time_getWallclockSeconds();

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
        {
            jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            int_buf_data[index++] = measure_array[jrow];
         }
      }

      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
        	&measure_array[num_variables]);
 
         hypre_ParCSRCommHandleDestroy(comm_handle);
      }
      /*------------------------------------------------
       * Debugging:
       *
       * Uncomment the sections of code labeled
       * "debugging" to generate several files that
       * can be visualized using the `coarsen.m'
       * matlab routine.
       *------------------------------------------------*/

#if 0 /* debugging */
      /* print out measures */
      char filename[50];
      FILE * fp;
      sprintf(filename, "coarsen.out.measures.%04d", iter);
      fp = fopen(filename, "w");
      for (i = 0; i < num_variables; i++)
      {
         fprintf(fp, "%d\n", measure_array[i]);
      }
      fclose(fp);

      /* print out strength matrix */
      sprintf(filename, "coarsen.out.strength.%04d", iter);
      hypre_CSRMatrixPrint(S, filename);

      /* print out C/F marker */
      sprintf(filename, "coarsen.out.CF.%04d", iter);
      fp = fopen(filename, "w");
      for (i = 0; i < num_variables; i++)
      {
         fprintf(fp, "%d\n", CF_marker[i]);
      }
      fclose(fp);

      //iter++;
#endif

      /*------------------------------------------------
       * Test for convergence
       *------------------------------------------------*/

      MPI_Allreduce(&graph_size,&global_graph_size,1,MPI_INT,MPI_SUM,comm);

my_update_wall_time = time_getWallclockSeconds();
      if (global_graph_size == 0)
         break;
update_time += time_getWallclockSeconds() - my_update_wall_time;

      /*------------------------------------------------
       * Pick an independent set of points with
       * maximal measure.
       *------------------------------------------------*/
my_search_wall_time = time_getWallclockSeconds();
      if (iter || !CF_init)
         hypre_BoomerAMGIndepSetCLJP_c(S, measure_array, color_array,
				       boundary_color_array, graph_array,
				       graph_size,
				       graph_array_offd, graph_offd_size,
				       CF_marker, CF_marker_offd);

      iter++;
search_time += time_getWallclockSeconds() - my_search_wall_time;
      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/

my_update_wall_time = time_getWallclockSeconds();
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                int_buf_data[index++]
                 = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
 
      if (num_procs > 1)
      {
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
        CF_marker_offd);
 
      hypre_ParCSRCommHandleDestroy(comm_handle);
      }
 
      for (ig = 0; ig < graph_offd_size; ig++)
      {
         i = graph_array_offd[ig];

         if (CF_marker_offd[i] < 0)
         {
            /* take point out of the subgraph */
            graph_offd_size--;
            graph_array_offd[ig] = graph_array_offd[graph_offd_size];
            graph_array_offd[graph_offd_size] = i;
            ig--;
         }
      }
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d  iter %d  comm. and subgraph update = %f\n",
                     my_id, iter, wall_time);
      }
      /*------------------------------------------------
       * Set C_pts and apply heuristics.
       *------------------------------------------------*/

      for (i=num_variables; i < num_variables+num_cols_offd; i++)
      {
         measure_array[i] = 0;
      }

      if (debug_flag == 3) wall_time = time_getWallclockSeconds();
      for (ig = 0; ig < graph_size; ig++)
      {
         i = graph_array[ig];

         /*---------------------------------------------
          * Heuristic: C-pts don't interpolate from
          * neighbors that influence them.
          *---------------------------------------------*/

         if (CF_marker[i] > 0)
         {
            /* set to be a C-pt */
            CF_marker[i] = C_PT;

            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
            {
               j = S_diag_j[jS];
               if (j > -1)
               {
               
                  /* "remove" edge from S */
                  S_diag_j[jS] = -S_diag_j[jS]-1;
             
                  /* decrement measures of unmarked neighbors */
                  if (!CF_marker[j])
                  {
                     measure_array[j]--;
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               j = S_offd_j[jS];
               if (j > -1)
               {
             
                  /* "remove" edge from S */
                  S_offd_j[jS] = -S_offd_j[jS]-1;
               
                  /* decrement measures of unmarked neighbors */
                  if (!CF_marker_offd[j])
                  {
                     measure_array[j+num_variables]--;
                  }
               }
            }
         }
	 else
    	 {
            /* marked dependencies */
            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
            {
               j = S_diag_j[jS];
	       if (j < 0) j = -j-1;
   
               if (CF_marker[j] > 0)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_diag_j[jS] = -S_diag_j[jS]-1;
                  }
   
                  /* IMPORTANT: consider all dependencies */
                  /* temporarily modify CF_marker */
                  CF_marker[j] = COMMON_C_PT;
               }
               else if (CF_marker[j] == SF_PT)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_diag_j[jS] = -S_diag_j[jS]-1;
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               j = S_offd_j[jS];
	       if (j < 0) j = -j-1;
   
               if (CF_marker_offd[j] > 0)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_offd_j[jS] = -S_offd_j[jS]-1;
                  }
   
                  /* IMPORTANT: consider all dependencies */
                  /* temporarily modify CF_marker */
                  CF_marker_offd[j] = COMMON_C_PT;
               }
               else if (CF_marker_offd[j] == SF_PT)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_offd_j[jS] = -S_offd_j[jS]-1;
                  }
               }
            }
   
            /* unmarked dependencies */
            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
            {
               if (S_diag_j[jS] > -1)
               {
                  j = S_diag_j[jS];
   	          break_var = 1;
                  /* check for common C-pt */
                  for (kS = S_diag_i[j]; kS < S_diag_i[j+1]; kS++)
                  {
                     k = S_diag_j[kS];
		     if (k < 0) k = -k-1;
   
                     /* IMPORTANT: consider all dependencies */
                     if (CF_marker[k] == COMMON_C_PT)
                     {
                        /* "remove" edge from S and update measure*/
                        S_diag_j[jS] = -S_diag_j[jS]-1;
                        measure_array[j]--;
                        break_var = 0;
                        break;
                     }
                  }
   		  if (break_var)
                  {
                     for (kS = S_offd_i[j]; kS < S_offd_i[j+1]; kS++)
                     {
                        k = S_offd_j[kS];
		        if (k < 0) k = -k-1;
   
                        /* IMPORTANT: consider all dependencies */
                        if ( CF_marker_offd[k] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_diag_j[jS] = -S_diag_j[jS]-1;
                           measure_array[j]--;
                           break;
                        }
                     }
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               if (S_offd_j[jS] > -1)
               {
                  j = S_offd_j[jS];
   
                  /* check for common C-pt */
                  for (kS = S_ext_i[j]; kS < S_ext_i[j+1]; kS++)
                  {
                     k = S_ext_j[kS];
   	             if (k >= 0)
   		     {
                        /* IMPORTANT: consider all dependencies */
                        if (CF_marker[k] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_offd_j[jS] = -S_offd_j[jS]-1;
                           measure_array[j+num_variables]--;
                           break;
                        }
                     }
   		     else
   		     {
   		        kc = -k-1;
   		        if (kc > -1 && CF_marker_offd[kc] == COMMON_C_PT)
   		        {
                           /* "remove" edge from S and update measure*/
                           S_offd_j[jS] = -S_offd_j[jS]-1;
                           measure_array[j+num_variables]--;
                           break;
   		        }
   		     }
                  }
               }
            }
         }

         /* reset CF_marker */
	 for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
	 {
            j = S_diag_j[jS];
	    if (j < 0) j = -j-1;

            if (CF_marker[j] == COMMON_C_PT)
            {
               CF_marker[j] = C_PT;
            }
         }
         for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
         {
            j = S_offd_j[jS];
	    if (j < 0) j = -j-1;

            if (CF_marker_offd[j] == COMMON_C_PT)
            {
               CF_marker_offd[j] = C_PT;
            }
         }
      }
update_time += time_getWallclockSeconds() - my_update_wall_time;
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d    CLJP phase = %f graph_size = %d nc_offd = %d\n",
                     my_id, wall_time, graph_size, num_cols_offd);
      }
   }

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

my_setup_wall_time = time_getWallclockSeconds();
   /* Reset S_matrix */
   for (i=0; i < S_diag_i[num_variables]; i++)
   {
      if (S_diag_j[i] < 0)
         S_diag_j[i] = -S_diag_j[i]-1;
   }
   for (i=0; i < S_offd_i[num_variables]; i++)
   {
      if (S_offd_j[i] < 0)
         S_offd_j[i] = -S_offd_j[i]-1;
   }
/*    for (i=0; i < num_variables; i++) */
/*       if (CF_marker[i] == SF_PT) CF_marker[i] = F_PT; */

   if(color_array)
     hypre_TFree(color_array);
   if(num_procs > 1)
     hypre_TFree(boundary_color_array);

   //hypre_TFree(measure_array);
   hypre_TFree(graph_array);
   if (num_cols_offd) hypre_TFree(graph_array_offd);
   hypre_TFree(buf_data);
   hypre_TFree(int_buf_data);
   hypre_TFree(CF_marker_offd);
   if (num_procs > 1) hypre_CSRMatrixDestroy(S_ext);

   *CF_marker_ptr   = CF_marker;
setup_time += time_getWallclockSeconds() - my_setup_wall_time;

printf(" Setup time: %f\n", setup_time);
printf(" Update time: %f\n", update_time);
printf(" Search time: %f\n", search_time);

   return (ierr);
}

int
hypre_BoomerAMGCoarsenBSISBoundaries( hypre_ParCSRMatrix    *S,
				      hypre_ParCSRMatrix    *A,
				      int                    debug_flag,
				      int                   *CF_marker,
				      int                   *measure_array,
				      hypre_CSRMatrix       *S_ext,
				      int                   *S_ext_i,
				      int                   *S_ext_j,
				      short                 *color_array)
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle   *comm_handle;

   hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
   int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
   int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
   int                *S_offd_j;

   int 		      *col_map_offd    = hypre_ParCSRMatrixColMapOffd(S);
   int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
   int		       col_1 = hypre_ParCSRMatrixFirstColDiag(S);
   int		       col_n = col_1 + hypre_CSRMatrixNumCols(S_diag);
   int 		       num_cols_offd = 0;
                  
   int		       num_sends = 0;
   int  	      *int_buf_data;
   double	      *buf_data;

   int                *CF_marker_offd;

   short              *boundary_color_array = NULL;
   int                num_colors;
                      
   int                *graph_array;
   int                *graph_array_offd;
   int                 graph_size;
   int                 graph_offd_size;
   int                 global_graph_size;
                      
   int                 i, j, k, kc, jS, kS, ig;
   int		       index, start, my_id, num_procs, jrow, cnt;
                      
   int                 ierr = 0;
   int                 break_var = 1;

   double	    wall_time;
#ifdef FINE_GRAIN_TIMINGS
   double           my_setup_wall_time, my_update_wall_time, my_search_wall_time,
                    setup_time=0, update_time=0, search_time=0;
#endif

   int   iter = 0;

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   int   iter = 0;
#endif

#ifdef FINE_GRAIN_TIMINGS
my_setup_wall_time = time_getWallclockSeconds();
#endif
   MPI_Comm_size(comm,&num_procs);
   num_cols_offd = hypre_CSRMatrixNumCols(S_offd);

   num_colors = 0;
   boundary_color_array = hypre_CTAlloc(short, num_variables+num_cols_offd);
   parColorGraphNewBoundariesOnly(A, S, boundary_color_array, &num_colors);
   
   /*--------------------------------------------------------------
    * Compute a  ParCSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();
   MPI_Comm_rank(comm,&my_id);
   if (!comm_pkg)
   {
        hypre_MatvecCommPkgCreate(A);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
   buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
 
   if (num_cols_offd)
   {
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
   
   // Get colors of off-processor neighbors.
   int * int_recv_data = hypre_CTAlloc(int, num_cols_offd);
   index = 0;
   for (i = 0; i < num_sends; i++) {
     start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
     for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++) {
       jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
       int_buf_data[index++] = color_array[jrow];
     }
   }
   if(num_procs > 1) {
     comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
        	int_recv_data);
 
     hypre_ParCSRCommHandleDestroy(comm_handle);

     for(i = 0; i < num_cols_offd; i++)
       color_array[num_variables+i] = int_recv_data[i];
   }

   // Get boundary colors of off-processor neighbors.
   index = 0;
   for (i = 0; i < num_sends; i++) {
     start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
     for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++) {
       jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
       int_buf_data[index++] = boundary_color_array[jrow];
     }
   }
   if(num_procs > 1) {

     comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
        	int_recv_data);
 
     hypre_ParCSRCommHandleDestroy(comm_handle);

     for(i = 0; i < num_cols_offd; i++)
       boundary_color_array[num_variables+i] = int_recv_data[i];
   }
   hypre_TFree(int_recv_data);

   /*---------------------------------------------------
    * Initialize the graph array
    * graph_array contains interior points in elements 0 ... num_variables-1
    * followed by boundary values
    *---------------------------------------------------*/

   graph_array = hypre_CTAlloc(int, num_variables);
   if (num_cols_offd)
      graph_array_offd = hypre_CTAlloc(int, num_cols_offd);
   else
      graph_array_offd = NULL;

   /* initialize measure array and graph array */

   for (ig = 0; ig < num_cols_offd; ig++)
      graph_array_offd[ig] = ig;

   /*---------------------------------------------------
    * Initialize the C/F marker array
    * C/F marker array contains interior points in elements 0 ...
    * num_variables-1  followed by boundary values
    *---------------------------------------------------*/

   graph_offd_size = num_cols_offd;
   
   cnt = 0;
   for (i=0; i < num_variables; i++) {
/*      if(CF_marker[i] == F_PT) */
/*        CF_marker[i] = 0; */
     if(CF_marker[i] == 0)
       graph_array[cnt++] = i;
   }
   graph_size = cnt;
   if (num_cols_offd)
      CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
   else
      CF_marker_offd = NULL;
   for (i=0; i < num_cols_offd; i++)
	CF_marker_offd[i] = 0;
  
   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Initialize CLJP phase = %f\n",
                     my_id, wall_time);
   }
#ifdef FINE_GRAIN_TIMINGS
setup_time += time_getWallclockSeconds() - my_setup_wall_time;
#endif

   while (1)
   {
      /*------------------------------------------------
       * Exchange boundary data, i.i. get measures and S_ext_data
       *------------------------------------------------*/

      if (num_procs > 1)
   	 comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg,
                        &measure_array[num_variables], int_buf_data);

      if (num_procs > 1)
   	 hypre_ParCSRCommHandleDestroy(comm_handle);
      
      index = 0;
      for (i=0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
                        += int_buf_data[index++];
      }

      /*------------------------------------------------
       * Set F-pts and update subgraph
       *------------------------------------------------*/
 
#ifdef FINE_GRAIN_TIMINGS
my_update_wall_time = time_getWallclockSeconds();
#endif
      for (ig = 0; ig < graph_size; ig++) {
	i = graph_array[ig];
	
	if ( (CF_marker[i] != C_PT) && (measure_array[i] == 0) ) {
	  /* set to be an F-pt */
	  CF_marker[i] = F_PT;
 
	  /* make sure all dependencies have been accounted for */
	  for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++) {
	    if (S_diag_j[jS] > -1) {
	      CF_marker[i] = 0;
	    }
	  }
	  for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++) {
	    if (S_offd_j[jS] > -1) {
	      CF_marker[i] = 0;
	    }
	  }
	}
	if (CF_marker[i]) {
	  measure_array[i] = 0;
 
	  /* take point out of the subgraph */
	  graph_size--;
	  graph_array[ig] = graph_array[graph_size];
	  graph_array[graph_size] = i;
	  ig--;
	}
      }
#ifdef FINE_GRAIN_TIMINGS
update_time += time_getWallclockSeconds() - my_update_wall_time;
#endif
 
      /*------------------------------------------------
       * Exchange boundary data, i.i. get measures
       *------------------------------------------------*/

      if (debug_flag == 3) wall_time = time_getWallclockSeconds();

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
        {
            jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            int_buf_data[index++] = measure_array[jrow];
         }
      }

      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
        	&measure_array[num_variables]);
 
         hypre_ParCSRCommHandleDestroy(comm_handle);
      }
      /*------------------------------------------------
       * Test for convergence
       *------------------------------------------------*/

      MPI_Allreduce(&graph_size,&global_graph_size,1,MPI_INT,MPI_SUM,comm);

#ifdef FINE_GRAIN_TIMINGS
my_update_wall_time = time_getWallclockSeconds();
#endif
      if (global_graph_size == 0)
         break;
#ifdef FINE_GRAIN_TIMINGS
update_time += time_getWallclockSeconds() - my_update_wall_time;
#endif

      /*------------------------------------------------
       * Pick an independent set of points with
       * maximal measure.
       *------------------------------------------------*/
#ifdef FINE_GRAIN_TIMINGS
my_search_wall_time = time_getWallclockSeconds();
#endif
      if (iter)
         hypre_BoomerAMGIndepSetCLJP_c(S, measure_array, color_array,
				       boundary_color_array, graph_array,
				       graph_size,
				       graph_array_offd, graph_offd_size,
				       CF_marker, CF_marker_offd);

      iter++;
#ifdef FINE_GRAIN_TIMINGS
search_time += time_getWallclockSeconds() - my_search_wall_time;
#endif
      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/

#ifdef FINE_GRAIN_TIMINGS
my_update_wall_time = time_getWallclockSeconds();
#endif
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                int_buf_data[index++]
                 = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
 
      if (num_procs > 1)
      {
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
        CF_marker_offd);
 
      hypre_ParCSRCommHandleDestroy(comm_handle);
      }
 
      for (ig = 0; ig < graph_offd_size; ig++)
      {
         i = graph_array_offd[ig];

         if (CF_marker_offd[i] < 0)
         {
            /* take point out of the subgraph */
            graph_offd_size--;
            graph_array_offd[ig] = graph_array_offd[graph_offd_size];
            graph_array_offd[graph_offd_size] = i;
            ig--;
         }
      }
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d  iter %d  comm. and subgraph update = %f\n",
                     my_id, iter, wall_time);
      }
      /*------------------------------------------------
       * Set C_pts and apply heuristics.
       *------------------------------------------------*/

      for (i=num_variables; i < num_variables+num_cols_offd; i++)
      {
         measure_array[i] = 0;
      }

      if (debug_flag == 3) wall_time = time_getWallclockSeconds();
      for (ig = 0; ig < graph_size; ig++)
      {
         i = graph_array[ig];

         /*---------------------------------------------
          * Heuristic: C-pts don't interpolate from
          * neighbors that influence them.
          *---------------------------------------------*/

         if (CF_marker[i] > 0)
         {
            /* set to be a C-pt */
            CF_marker[i] = C_PT;

            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
            {
               j = S_diag_j[jS];
               if (j > -1)
               {
               
                  /* "remove" edge from S */
                  S_diag_j[jS] = -S_diag_j[jS]-1;
             
                  /* decrement measures of unmarked neighbors */
                  if (!CF_marker[j])
                  {
                     measure_array[j]--;
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               j = S_offd_j[jS];
               if (j > -1)
               {
             
                  /* "remove" edge from S */
                  S_offd_j[jS] = -S_offd_j[jS]-1;
               
                  /* decrement measures of unmarked neighbors */
                  if (!CF_marker_offd[j])
                  {
                     measure_array[j+num_variables]--;
                  }
               }
            }
         }
	 else
    	 {
            /* marked dependencies */
            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
            {
               j = S_diag_j[jS];
	       if (j < 0) j = -j-1;
   
               if (CF_marker[j] > 0)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_diag_j[jS] = -S_diag_j[jS]-1;
                  }
   
                  /* IMPORTANT: consider all dependencies */
                  /* temporarily modify CF_marker */
                  CF_marker[j] = COMMON_C_PT;
               }
               else if (CF_marker[j] == SF_PT)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_diag_j[jS] = -S_diag_j[jS]-1;
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               j = S_offd_j[jS];
	       if (j < 0) j = -j-1;
   
               if (CF_marker_offd[j] > 0)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_offd_j[jS] = -S_offd_j[jS]-1;
                  }
   
                  /* IMPORTANT: consider all dependencies */
                  /* temporarily modify CF_marker */
                  CF_marker_offd[j] = COMMON_C_PT;
               }
               else if (CF_marker_offd[j] == SF_PT)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_offd_j[jS] = -S_offd_j[jS]-1;
                  }
               }
            }
   
            /* unmarked dependencies */
            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
            {
               if (S_diag_j[jS] > -1)
               {
                  j = S_diag_j[jS];
   	          break_var = 1;
                  /* check for common C-pt */
                  for (kS = S_diag_i[j]; kS < S_diag_i[j+1]; kS++)
                  {
                     k = S_diag_j[kS];
		     if (k < 0) k = -k-1;
   
                     /* IMPORTANT: consider all dependencies */
                     if (CF_marker[k] == COMMON_C_PT)
                     {
                        /* "remove" edge from S and update measure*/
                        S_diag_j[jS] = -S_diag_j[jS]-1;
                        measure_array[j]--;
                        break_var = 0;
                        break;
                     }
                  }
   		  if (break_var)
                  {
                     for (kS = S_offd_i[j]; kS < S_offd_i[j+1]; kS++)
                     {
                        k = S_offd_j[kS];
		        if (k < 0) k = -k-1;
   
                        /* IMPORTANT: consider all dependencies */
                        if ( CF_marker_offd[k] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_diag_j[jS] = -S_diag_j[jS]-1;
                           measure_array[j]--;
                           break;
                        }
                     }
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               if (S_offd_j[jS] > -1)
               {
                  j = S_offd_j[jS];
   
                  /* check for common C-pt */
                  for (kS = S_ext_i[j]; kS < S_ext_i[j+1]; kS++)
                  {
                     k = S_ext_j[kS];
   	             if (k >= 0)
   		     {
                        /* IMPORTANT: consider all dependencies */
                        if (CF_marker[k] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_offd_j[jS] = -S_offd_j[jS]-1;
                           measure_array[j+num_variables]--;
                           break;
                        }
                     }
   		     else
   		     {
   		        kc = -k-1;
   		        if (kc > -1 && CF_marker_offd[kc] == COMMON_C_PT)
   		        {
                           /* "remove" edge from S and update measure*/
                           S_offd_j[jS] = -S_offd_j[jS]-1;
                           measure_array[j+num_variables]--;
                           break;
   		        }
   		     }
                  }
               }
            }
         }

         /* reset CF_marker */
	 for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
	 {
            j = S_diag_j[jS];
	    if (j < 0) j = -j-1;

            if (CF_marker[j] == COMMON_C_PT)
            {
               CF_marker[j] = C_PT;
            }
         }
         for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
         {
            j = S_offd_j[jS];
	    if (j < 0) j = -j-1;

            if (CF_marker_offd[j] == COMMON_C_PT)
            {
               CF_marker_offd[j] = C_PT;
            }
         }
      }
#ifdef FINE_GRAIN_TIMINGS
update_time += time_getWallclockSeconds() - my_update_wall_time;
#endif
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d    CLJP phase = %f graph_size = %d nc_offd = %d\n",
                     my_id, wall_time, graph_size, num_cols_offd);
      }
   }

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

#ifdef FINE_GRAIN_TIMINGS
my_setup_wall_time = time_getWallclockSeconds();
#endif

   if(num_procs > 1)
     hypre_TFree(boundary_color_array);

   hypre_TFree(graph_array);
   if (num_cols_offd) hypre_TFree(graph_array_offd);
   hypre_TFree(buf_data);
   hypre_TFree(int_buf_data);
   hypre_TFree(CF_marker_offd);

#ifdef FINE_GRAIN_TIMINGS
setup_time += time_getWallclockSeconds() - my_setup_wall_time;
#endif

#ifdef FINE_GRAIN_TIMINGS
printf(" Setup time: %f\n", setup_time);
printf(" Update time: %f\n", update_time);
printf(" Search time: %f\n", search_time);
#endif

   return (ierr);
}

int
hypre_BoomerAMGCoarsenBSISCLJP_cBoundaries( hypre_ParCSRMatrix    *S,
					    hypre_ParCSRMatrix    *A,
					    int                    debug_flag,
					    int                   *CF_marker,
					    int                    level,
					    int                   *int_measure_array,
					    hypre_CSRMatrix       *S_ext,
					    int                   *S_ext_i,
					    int                   *S_ext_j,
					    short                 *color_array,
					    int                    num_colors)
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle   *comm_handle;

   hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
   int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
   int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
   int                *S_offd_j;

   int 		      *col_map_offd    = hypre_ParCSRMatrixColMapOffd(S);
   int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
   int		       col_1 = hypre_ParCSRMatrixFirstColDiag(S);
   int		       col_n = col_1 + hypre_CSRMatrixNumCols(S_diag);
   int 		       num_cols_offd = 0;
                  
   int		       num_sends = 0;
   int  	      *int_buf_data;
   double	      *buf_data;

   int                *CF_marker_offd;

   double             *measure_array;

   int                *graph_array;
   int                *graph_array_offd;
   int                 graph_size;
   int                 graph_offd_size;
   int                 global_graph_size;
                      
   int                 i, j, k, kc, jS, kS, ig;
   int		       index, start, my_id, num_procs, jrow, cnt;
                      
   int                 ierr = 0;
   int                 break_var = 1;

   double	    wall_time;
#ifdef FINE_GRAIN_TIMINGS
   double           my_setup_wall_time, my_update_wall_time, my_search_wall_time,
                    setup_time=0, update_time=0, search_time=0;
#endif

   int   iter = 0;

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   int   iter = 0;
#endif

#ifdef FINE_GRAIN_TIMINGS
my_setup_wall_time = time_getWallclockSeconds();
#endif
   if(!measure_array) {
     // Then the coloring needs to be done. The only time this is not needed is if
     // the calling function already has the measure_array computed. The CR function
     // does this because it calls CLJP_c several times for each level. By computing
     // the measure_array ahead of time, extra computation can be saved.
     color_array = hypre_CTAlloc(short, num_variables);

     num_colors = 0;
/*      if(global) */
/*        //parColorGraph(A, S, color_array, &num_colors, level); */
/*        parColorGraphNew(A, S, color_array, &num_colors, level); */
/*      else */
/*        //seqColorGraphNew(S, color_array, &num_colors, level); */
/*        seqColorGraphTheFinalWord(A, color_array, &num_colors, 0, level); */
     //seqColorGraphNew(S, color_array, &num_colors, level);
     //seqColorGraphTheFinalWord(A, color_array, &num_colors, 0, level);
   }
   
   /*--------------------------------------------------------------
    * Compute a  ParCSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();
   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);
   if (!comm_pkg)
   {
        hypre_MatvecCommPkgCreate(A);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
   buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
 
   num_cols_offd = hypre_CSRMatrixNumCols(S_offd);

   S_diag_j = hypre_CSRMatrixJ(S_diag);

   if (num_cols_offd)
   {
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

   if (num_procs > 1)
     comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg,
                        &measure_array[num_variables], buf_data);

   if (num_procs > 1)
   hypre_ParCSRCommHandleDestroy(comm_handle);
      
   index = 0;
   for (i=0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
                        += buf_data[index++];
   }

   for (i=num_variables; i < num_variables+num_cols_offd; i++)
   /* This loop zeros out the measures for the off-process nodes since
      this process is not responsible for . */
   {
      measure_array[i] = 0;
   }

   /* this augments the measures */
   //hypre_BoomerAMGIndepSetInit(S, measure_array);
   hypre_BoomerAMGIndepSetInitb2(S, measure_array, int_measure_array, color_array, num_colors,
				 CF_marker);

   /*---------------------------------------------------
    * Initialize the graph array
    * graph_array contains interior points in elements 0 ... num_variables-1
    * followed by boundary values
    *---------------------------------------------------*/

   graph_array = hypre_CTAlloc(int, num_variables);
   if (num_cols_offd)
      graph_array_offd = hypre_CTAlloc(int, num_cols_offd);
   else
      graph_array_offd = NULL;

   /* initialize measure array and graph array */

   for (ig = 0; ig < num_cols_offd; ig++)
      graph_array_offd[ig] = ig;

   /*---------------------------------------------------
    * Initialize the C/F marker array
    * C/F marker array contains interior points in elements 0 ...
    * num_variables-1  followed by boundary values
    *---------------------------------------------------*/

   graph_offd_size = num_cols_offd;

   cnt = 0;
   for (i=0; i < num_variables; i++) {
     if(CF_marker[i] == 0)
       graph_array[cnt++] = i;
   }
   graph_size = cnt;
   if (num_cols_offd)
      CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
   else
      CF_marker_offd = NULL;
   for (i=0; i < num_cols_offd; i++)
	CF_marker_offd[i] = 0;
  
   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Initialize CLJP phase = %f\n",
                     my_id, wall_time);
   }
#ifdef FINE_GRAIN_TIMINGS
setup_time += time_getWallclockSeconds() - my_setup_wall_time;
#endif

   while (1)
   {
      /*------------------------------------------------
       * Exchange boundary data, i.i. get measures and S_ext_data
       *------------------------------------------------*/

      if (num_procs > 1)
   	 comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg,
                        &measure_array[num_variables], buf_data);

      if (num_procs > 1)
   	 hypre_ParCSRCommHandleDestroy(comm_handle);
      
      index = 0;
      for (i=0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
                        += buf_data[index++];
      }

      /*------------------------------------------------
       * Set F-pts and update subgraph
       *------------------------------------------------*/
 
#ifdef FINE_GRAIN_TIMINGS
my_update_wall_time = time_getWallclockSeconds();
#endif
      if (iter)
      {
         for (ig = 0; ig < graph_size; ig++)
         {
            i = graph_array[ig];

            if ( (CF_marker[i] != C_PT) && (measure_array[i] < 1) )
            {
               /* set to be an F-pt */
               CF_marker[i] = F_PT;
 
	       /* make sure all dependencies have been accounted for */
               for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     CF_marker[i] = 0;
                  }
               }
               for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     CF_marker[i] = 0;
                  }
               }
            }
            if (CF_marker[i])
            {
               measure_array[i] = 0;
 
               /* take point out of the subgraph */
               graph_size--;
               graph_array[ig] = graph_array[graph_size];
               graph_array[graph_size] = i;
               ig--;
            }
         }
      }
#ifdef FINE_GRAIN_TIMINGS
update_time += time_getWallclockSeconds() - my_update_wall_time;
#endif
 
      /*------------------------------------------------
       * Exchange boundary data, i.i. get measures
       *------------------------------------------------*/

      if (debug_flag == 3) wall_time = time_getWallclockSeconds();

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
        {
            jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            buf_data[index++] = measure_array[jrow];
         }
      }

      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg, buf_data,
        	&measure_array[num_variables]);
 
         hypre_ParCSRCommHandleDestroy(comm_handle);
 
      }
      /*------------------------------------------------
       * Debugging:
       *
       * Uncomment the sections of code labeled
       * "debugging" to generate several files that
       * can be visualized using the `coarsen.m'
       * matlab routine.
       *------------------------------------------------*/

#if 0 /* debugging */
      /* print out measures */
      char filename[50];
      FILE * fp;
      sprintf(filename, "coarsen.out.measures.%04d", iter);
      fp = fopen(filename, "w");
      for (i = 0; i < num_variables; i++)
      {
         fprintf(fp, "%f\n", measure_array[i]);
      }
      fclose(fp);

      /* print out strength matrix */
      sprintf(filename, "coarsen.out.strength.%04d", iter);
      hypre_CSRMatrixPrint(S, filename);

      /* print out C/F marker */
      sprintf(filename, "coarsen.out.CF.%04d", iter);
      fp = fopen(filename, "w");
      for (i = 0; i < num_variables; i++)
      {
         fprintf(fp, "%d\n", CF_marker[i]);
      }
      fclose(fp);

      //iter++;
#endif

      /*------------------------------------------------
       * Test for convergence
       *------------------------------------------------*/

      MPI_Allreduce(&graph_size,&global_graph_size,1,MPI_INT,MPI_SUM,comm);

#ifdef FINE_GRAIN_TIMINGS
my_update_wall_time = time_getWallclockSeconds();
#endif
      if (global_graph_size == 0)
         break;
#ifdef FINE_GRAIN_TIMINGS
update_time += time_getWallclockSeconds() - my_update_wall_time;
#endif

      /*------------------------------------------------
       * Pick an independent set of points with
       * maximal measure.
       *------------------------------------------------*/
#ifdef FINE_GRAIN_TIMINGS
my_search_wall_time = time_getWallclockSeconds();
#endif
      if (iter)
         hypre_BoomerAMGIndepSet(S, measure_array, graph_array,
				graph_size,
				graph_array_offd, graph_offd_size,
				CF_marker, CF_marker_offd);

      iter++;
#ifdef FINE_GRAIN_TIMINGS
search_time += time_getWallclockSeconds() - my_search_wall_time;
#endif
      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/

#ifdef FINE_GRAIN_TIMINGS
my_update_wall_time = time_getWallclockSeconds();
#endif
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                int_buf_data[index++]
                 = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
 
      if (num_procs > 1)
      {
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
        CF_marker_offd);
 
      hypre_ParCSRCommHandleDestroy(comm_handle);
      }
 
      for (ig = 0; ig < graph_offd_size; ig++)
      {
         i = graph_array_offd[ig];

         if (CF_marker_offd[i] < 0)
         {
            /* take point out of the subgraph */
            graph_offd_size--;
            graph_array_offd[ig] = graph_array_offd[graph_offd_size];
            graph_array_offd[graph_offd_size] = i;
            ig--;
         }
      }
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d  iter %d  comm. and subgraph update = %f\n",
                     my_id, iter, wall_time);
      }
      /*------------------------------------------------
       * Set C_pts and apply heuristics.
       *------------------------------------------------*/

      for (i=num_variables; i < num_variables+num_cols_offd; i++)
      {
         measure_array[i] = 0;
      }

      if (debug_flag == 3) wall_time = time_getWallclockSeconds();
      for (ig = 0; ig < graph_size; ig++)
      {
         i = graph_array[ig];

         /*---------------------------------------------
          * Heuristic: C-pts don't interpolate from
          * neighbors that influence them.
          *---------------------------------------------*/

         if (CF_marker[i] > 0)
         {
            /* set to be a C-pt */
            CF_marker[i] = C_PT;

            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
            {
               j = S_diag_j[jS];
               if (j > -1)
               {
               
                  /* "remove" edge from S */
                  S_diag_j[jS] = -S_diag_j[jS]-1;
             
                  /* decrement measures of unmarked neighbors */
                  if (!CF_marker[j])
                  {
                     measure_array[j]--;
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               j = S_offd_j[jS];
               if (j > -1)
               {
             
                  /* "remove" edge from S */
                  S_offd_j[jS] = -S_offd_j[jS]-1;
               
                  /* decrement measures of unmarked neighbors */
                  if (!CF_marker_offd[j])
                  {
                     measure_array[j+num_variables]--;
                  }
               }
            }
         }
	 else
    	 {
            /* marked dependencies */
            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
            {
               j = S_diag_j[jS];
	       if (j < 0) j = -j-1;
   
               if (CF_marker[j] > 0)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_diag_j[jS] = -S_diag_j[jS]-1;
                  }
   
                  /* IMPORTANT: consider all dependencies */
                  /* temporarily modify CF_marker */
                  CF_marker[j] = COMMON_C_PT;
               }
               else if (CF_marker[j] == SF_PT)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_diag_j[jS] = -S_diag_j[jS]-1;
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               j = S_offd_j[jS];
	       if (j < 0) j = -j-1;
   
               if (CF_marker_offd[j] > 0)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_offd_j[jS] = -S_offd_j[jS]-1;
                  }
   
                  /* IMPORTANT: consider all dependencies */
                  /* temporarily modify CF_marker */
                  CF_marker_offd[j] = COMMON_C_PT;
               }
               else if (CF_marker_offd[j] == SF_PT)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_offd_j[jS] = -S_offd_j[jS]-1;
                  }
               }
            }
   
            /* unmarked dependencies */
            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
            {
               if (S_diag_j[jS] > -1)
               {
                  j = S_diag_j[jS];
   	          break_var = 1;
                  /* check for common C-pt */
                  for (kS = S_diag_i[j]; kS < S_diag_i[j+1]; kS++)
                  {
                     k = S_diag_j[kS];
		     if (k < 0) k = -k-1;
   
                     /* IMPORTANT: consider all dependencies */
                     if (CF_marker[k] == COMMON_C_PT)
                     {
                        /* "remove" edge from S and update measure*/
                        S_diag_j[jS] = -S_diag_j[jS]-1;
                        measure_array[j]--;
                        break_var = 0;
                        break;
                     }
                  }
   		  if (break_var)
                  {
                     for (kS = S_offd_i[j]; kS < S_offd_i[j+1]; kS++)
                     {
                        k = S_offd_j[kS];
		        if (k < 0) k = -k-1;
   
                        /* IMPORTANT: consider all dependencies */
                        if ( CF_marker_offd[k] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_diag_j[jS] = -S_diag_j[jS]-1;
                           measure_array[j]--;
                           break;
                        }
                     }
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               if (S_offd_j[jS] > -1)
               {
                  j = S_offd_j[jS];
   
                  /* check for common C-pt */
                  for (kS = S_ext_i[j]; kS < S_ext_i[j+1]; kS++)
                  {
                     k = S_ext_j[kS];
   	             if (k >= 0)
   		     {
                        /* IMPORTANT: consider all dependencies */
                        if (CF_marker[k] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_offd_j[jS] = -S_offd_j[jS]-1;
                           measure_array[j+num_variables]--;
                           break;
                        }
                     }
   		     else
   		     {
   		        kc = -k-1;
   		        if (kc > -1 && CF_marker_offd[kc] == COMMON_C_PT)
   		        {
                           /* "remove" edge from S and update measure*/
                           S_offd_j[jS] = -S_offd_j[jS]-1;
                           measure_array[j+num_variables]--;
                           break;
   		        }
   		     }
                  }
               }
            }
         }

         /* reset CF_marker */
	 for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
	 {
            j = S_diag_j[jS];
	    if (j < 0) j = -j-1;

            if (CF_marker[j] == COMMON_C_PT)
            {
               CF_marker[j] = C_PT;
            }
         }
         for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
         {
            j = S_offd_j[jS];
	    if (j < 0) j = -j-1;

            if (CF_marker_offd[j] == COMMON_C_PT)
            {
               CF_marker_offd[j] = C_PT;
            }
         }
      }
#ifdef FINE_GRAIN_TIMINGS
update_time += time_getWallclockSeconds() - my_update_wall_time;
#endif
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d    CLJP phase = %f graph_size = %d nc_offd = %d\n",
                     my_id, wall_time, graph_size, num_cols_offd);
      }
   }

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

#ifdef FINE_GRAIN_TIMINGS
my_setup_wall_time = time_getWallclockSeconds();
#endif
   /* Reset S_matrix */
   for (i=0; i < S_diag_i[num_variables]; i++)
   {
      if (S_diag_j[i] < 0)
         S_diag_j[i] = -S_diag_j[i]-1;
   }
   for (i=0; i < S_offd_i[num_variables]; i++)
   {
      if (S_offd_j[i] < 0)
         S_offd_j[i] = -S_offd_j[i]-1;
   }
/*    for (i=0; i < num_variables; i++) */
/*       if (CF_marker[i] == SF_PT) CF_marker[i] = F_PT; */

   hypre_TFree(measure_array);
   hypre_TFree(graph_array);
   if (num_cols_offd) hypre_TFree(graph_array_offd);
   hypre_TFree(buf_data);
   hypre_TFree(int_buf_data);
   hypre_TFree(CF_marker_offd);

#ifdef FINE_GRAIN_TIMINGS
setup_time += time_getWallclockSeconds() - my_setup_wall_time;
#endif

#ifdef FINE_GRAIN_TIMINGS
printf(" Setup time: %f\n", setup_time);
printf(" Update time: %f\n", update_time);
printf(" Search time: %f\n", search_time);
#endif

   return (ierr);
}

#define bucketIndex(measure, color, num_colors) ((measure-1)*num_colors + (num_colors - color))
#define measureWeight(bucket_index, num_colors) (bucket_index/num_colors + 1)
#define colorWeight(bucket_index, num_colors) (num_colors - (bucket_index%num_colors))

int sendMaxWeight(hypre_ParCSRCommPkg * comm_pkg, int * max_weight_measure,
		  int * max_weight_color, int finished,
		  int * num_finished_neighbors, int * finished_neighbors_array)
{
  int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
  MPI_Comm comm = hypre_ParCSRCommPkgComm(comm_pkg);

  hypre_ParCSRCommHandle *comm_handle;
  int num_requests;
  MPI_Request *requests;
  int weight[2], *neighbor_weights;

  int i, j;
  int ip, ierr;

  weight[0] = *max_weight_measure;
  weight[1] = *max_weight_color;
  neighbor_weights = hypre_CTAlloc(int, 2*num_recvs);

  num_requests = 3*(num_sends-num_finished_neighbors[0]) +
    3*(num_recvs-num_finished_neighbors[1]);
  requests = hypre_CTAlloc(MPI_Request, num_requests);
  j = 0;

  for (i = 0; i < num_recvs; i++) {
    if(!finished_neighbors_array[i+num_sends]) {
      ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i); 
      MPI_Irecv(&finished_neighbors_array[i+num_sends], 1, MPI_INT, ip, 0, comm,
		&requests[j++]);
      MPI_Irecv(&neighbor_weights[2*i], 2, MPI_INT, ip, 1, comm, &requests[j++]);
      MPI_Isend(&finished, 1, MPI_INT, ip, 2, comm, &requests[j++]);
    }
  }
  for (i = 0; i < num_sends; i++) {
    if(!finished_neighbors_array[i]) {
      ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i); 
      MPI_Isend(&finished, 1, MPI_INT, ip, 0, comm, &requests[j++]);
      MPI_Isend(weight, 2, MPI_INT, ip, 1, comm, &requests[j++]);
      MPI_Irecv(&finished_neighbors_array[i], 1, MPI_INT, ip, 2, comm,
		&requests[j++]);
    }
  }

  comm_handle = hypre_CTAlloc(hypre_ParCSRCommHandle, 1);
  hypre_ParCSRCommHandleCommPkg(comm_handle) = comm_pkg;
  hypre_ParCSRCommHandleNumRequests(comm_handle) = num_requests;
  hypre_ParCSRCommHandleRequests(comm_handle) = requests;

  ierr = waitforLoopData(comm_handle);

  // Process the received information.
  num_finished_neighbors[0] = 0;
  num_finished_neighbors[1] = 0;
  for(i = 0; i < num_recvs; i++) {
    // Check if neighbors are finished.
    if(finished_neighbors_array[i+num_sends])
      num_finished_neighbors[1]++;

    // Check if neighbor's num_colors is greatest known num_colors.
    //if(*neighborhood_num_colors < neighbor_num_colors[i])
    //*neighborhood_num_colors = neighbor_num_colors[i];
    if(neighbor_weights[2*i] >= *max_weight_measure) {
      if(neighbor_weights[2*i] > *max_weight_measure) {
	*max_weight_measure = neighbor_weights[2*i];
	*max_weight_color = neighbor_weights[(2*i)+1];
      }
      else if(neighbor_weights[(2*i)+1] > *max_weight_color)
	*max_weight_color = neighbor_weights[(2*i)+1];	
    }
  }
  for(i = 0; i < num_sends; i++) {
    // Check if neighbors are finished.
    if(finished_neighbors_array[i])
      num_finished_neighbors[0]++;
  }

  hypre_TFree(neighbor_weights);

  return ierr;
}

int exchangeIntBuf(hypre_ParCSRCommPkg * comm_pkg, int * send_buf_data,
		   int * recv_buf_data, int * num_finished_neighbors,
		   int * finished_neighbors_array)
{
  int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
  MPI_Comm comm = hypre_ParCSRCommPkgComm(comm_pkg);

  hypre_ParCSRCommHandle *comm_handle;
  int num_requests;
  MPI_Request *requests;

  int i, j, vec_start, vec_len;
  int ip, ierr;

  num_requests = (num_sends-num_finished_neighbors[0]) +
    (num_recvs-num_finished_neighbors[1]);
  requests = hypre_CTAlloc(MPI_Request, num_requests);
  j = 0;

  for (i = 0; i < num_recvs; i++) {
    if(!finished_neighbors_array[i+num_sends]) {
      ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i); 
      vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i);
      vec_len = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
      MPI_Irecv(&recv_buf_data[vec_start], vec_len, MPI_INT, ip, 0, comm, &requests[j++]);
    }
  }
  for (i = 0; i < num_sends; i++) {
    if(!finished_neighbors_array[i]) {
      ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i); 
      vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      vec_len = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1)-vec_start;
      MPI_Isend(&send_buf_data[vec_start], vec_len, MPI_INT, ip, 0, comm, &requests[j++]);
    }
  }

  comm_handle = hypre_CTAlloc(hypre_ParCSRCommHandle, 1);
  hypre_ParCSRCommHandleCommPkg(comm_handle) = comm_pkg;
  hypre_ParCSRCommHandleNumRequests(comm_handle) = num_requests;
  hypre_ParCSRCommHandleRequests(comm_handle) = requests;

  ierr = waitforLoopData(comm_handle);

  return ierr;
}

int updateWeightsOld(int newCPT,
		     int * measure_array,
		     short * color_array,
		     int num_colors,
		     int * CF_marker,
		     int * CF_marker_offd,
		     hypre_ParCSRMatrix * S,
		     hypre_ParCSRMatrix * A,
		     int * S_ext_i,
		     int * S_ext_j,
		     hypre_QueueElement * bucket_elements,
		     hypre_Queue * buckets)
// This function updates weights after a new C-point has been
// selected. This is done in the CLJP manner. Nodes are marked as
// F-points in this function if their measures reach zero.
{
  MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(S);
  hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
  hypre_ParCSRCommHandle   *comm_handle;
  
  hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
  int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
  int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);
  
  hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
  int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
  int                *S_offd_j        = hypre_CSRMatrixJ(S_offd);
  
  int 		      *col_map_offd    = hypre_ParCSRMatrixColMapOffd(S);
  int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
  int                 j, jS, k, kS, kc, break_var, neighborNode,
    *origBucketIndexes, newBucketIndex, numAssigned;

  hypre_Queue         *updated;

  // Remove edges from the new C-point to its neighbors. Update
  // weights and buckets accordingly.
  origBucketIndexes = hypre_CTAlloc(int, S_diag_i[newCPT+1]-S_diag_i[newCPT]);
  for(j = S_offd_i[newCPT]; j < S_offd_i[newCPT+1]; j++) {
    if(S_offd_j[j] > -1 && CF_marker_offd[S_offd_j[j]] == 0) {
      // Then this edge is still in the graph, so update its weight
      // and mark it as a neighbor to the new C-point.
      //measure_array[num_variables+j]--;
      CF_marker_offd[S_offd_j[j]] = COMMON_C_PT;
    }
  }
  for(j = S_diag_i[newCPT]; j < S_diag_i[newCPT+1]; j++) {
    neighborNode = S_diag_j[j];
    if(neighborNode < 0) neighborNode = -neighborNode-1;
    if(CF_marker[neighborNode] != C_PT) {
      // Then this edge is still in the graph, so update its weight
      // and mark it as a neighbor to the new C-point.

      if(CF_marker[neighborNode] == F_PT)
	origBucketIndexes[j-S_diag_i[newCPT]] = -1;
      else
	origBucketIndexes[j-S_diag_i[newCPT]] = bucketIndex(measure_array[neighborNode],
							    color_array[neighborNode],
							    num_colors);
      if(S_diag_j[j] > -1)
	measure_array[neighborNode]--;

      CF_marker[neighborNode] = COMMON_C_PT;
    }
  }
    
  for(j = S_diag_i[newCPT]; j < S_diag_i[newCPT+1]; j++) {
    // Loop for on-processor nodes that strongly influence the new
    // C-point.
    neighborNode = S_diag_j[j];
    if(neighborNode < 0) neighborNode = -neighborNode-1;
    if(CF_marker[neighborNode] == COMMON_C_PT) {
      // Check for nodes connected to this node and the new
      // C-point.
      for(k = S_offd_i[neighborNode]; k < S_offd_i[neighborNode+1]; k++) {
	if((S_offd_j[k] > -1) && (CF_marker_offd[S_offd_j[k]] == COMMON_C_PT)) {
	  S_offd_j[k] = -S_offd_j[k] - 1;
	  //measure_array[num_variables+S_offd_j[k]]--;
	  // No effect to the neighborNode's weight here. It will be
	  //affected if there is a connection from S_offd_j[k] to
	  //neighborNode.
	}
      }
      for(k = S_diag_i[neighborNode]; k < S_diag_i[neighborNode+1]; k++) {
	if((S_diag_j[k] > -1) && (CF_marker[S_diag_j[k]] == COMMON_C_PT)) {
	  measure_array[S_diag_j[k]]--;
	  S_diag_j[k] = -S_diag_j[k] - 1;
	}
	else if(S_diag_j[k] == newCPT)
	  // Remove the edge from this node to the new C-point.
	  S_diag_j[k] = -S_diag_j[k] - 1;
      }
    }
  }
  for(j = S_offd_i[newCPT]; j < S_offd_i[newCPT+1]; j++) {
    // Loop for off-processor nodes that strongly influence the new
    // C-point.
    neighborNode = S_offd_j[j];
    if(neighborNode < 0) neighborNode = -neighborNode-1;
    if(CF_marker_offd[neighborNode] == COMMON_C_PT) {
      // Check for nodes connected to this node and the new
      // C-point.
      for(k = S_ext_i[neighborNode]; k < S_ext_i[neighborNode+1]; k++) {
	if(S_ext_j[k] > -1 && CF_marker[S_ext_j[k]] == COMMON_C_PT) {
	  measure_array[S_ext_j[k]]--;
	  S_ext_j[k] = INT_MIN;
	}
	else if(S_ext_j[k] == newCPT)
	  S_ext_j[k] = INT_MIN;
      }
    }
  }

  // Restore the CF_marker values for all COMMON_C_PTs to an
  // unassigned value.
  for(j = S_offd_i[newCPT]; j < S_offd_i[newCPT+1]; j++) {
    neighborNode = S_offd_j[j];
    if(neighborNode < 0) neighborNode = -neighborNode-1;
    if(CF_marker_offd[neighborNode] == COMMON_C_PT) {
      CF_marker_offd[S_offd_j[j]] = 0;
      // "Remove" the edge from the new C-point to this node.
      if(S_offd_j[j] > -1)
	S_offd_j[j] = -S_offd_j[j] - 1;
    }
  }
/*   for(j = S_diag_i[newCPT]; j < S_diag_i[newCPT+1]; j++) { */
/*     if(S_diag_j[j] > -1) { */
/*       if(CF_marker[S_diag_j[j]] == COMMON_C_PT) */
/* 	CF_marker[S_diag_j[j]] = 0; */
/*     } */
/*   } */

  // Update each node neighboring the new C-point on this
  // processor. If the node's measure_array value is now zero, mark
  // node as F-point. Otherwise, move the node to its new bucket.
  numAssigned = 0;
  for(j = S_diag_i[newCPT]; j < S_diag_i[newCPT+1]; j++) {
    neighborNode = S_diag_j[j];
    if(neighborNode < 0) neighborNode = -neighborNode-1;
    if(CF_marker[neighborNode] == COMMON_C_PT) {
      // "Remove" the edge from the new C-point to this node.
      if(S_diag_j[j] > -1)
	S_diag_j[j] = -S_diag_j[j] - 1;

      // In either case, the node needs to be removed from its current
      // bucket.
      if(origBucketIndexes[j-S_diag_i[newCPT]] > -1)
	removeElement(&bucket_elements[neighborNode],
		      &buckets[origBucketIndexes[j-S_diag_i[newCPT]]]);
      else
	numAssigned--; // this counteracts the increment below -- want
                       // to do that becaues this node was already
	               // marked F_PT and counted earlier.
      if(measure_array[neighborNode] == 0) {
	// This node is now an F-point. Mark it as such.
	CF_marker[neighborNode] = F_PT;
	numAssigned++;
      }
      else {
	// This node is not an F-point. Add it to a new bucket.
	CF_marker[neighborNode] = 0;
	newBucketIndex = bucketIndex(measure_array[neighborNode],
				     color_array[neighborNode], num_colors);
	enqueueElement(&bucket_elements[neighborNode], &buckets[newBucketIndex]);
      }
    }
  }

  hypre_TFree(origBucketIndexes);
  return numAssigned;
}

void removeIfDisconnected(int nodeID, int * S_diag_i, int * S_diag_j, int * S_offd_i,
			  int * S_offd_j, int * painted_dependers)
{
  int i, nodeIsDisconnected;

  nodeIsDisconnected = 1;
  for(i = S_diag_i[nodeID]; i < S_diag_i[nodeID+1]; i++) {
    if(S_diag_j[i] > -1) {
      // Then this nodes still has edges from other nodes
      // influencing it. Keep this node in the connected_nodes
      // graph until that edge is gone.
      nodeIsDisconnected = 0;
      i = S_diag_i[nodeID+1]; // to break loop
    }
  }
  if(nodeIsDisconnected) {
    for(i = S_offd_i[nodeID]; i < S_offd_i[nodeID+1]; i++) {
      if(S_offd_j[i] > -1) {
	// Then this nodes still has edges from other nodes
	// influencing it. Keep this node in the connected_nodes
	// graph until that edge is gone.
	nodeIsDisconnected = 0;
	i = S_offd_i[nodeID+1]; // to break loop
      }
    }
  }
  if(nodeIsDisconnected)
    painted_dependers[nodeID] = -1;
}

int assignNewCPT(int newCPT,
		 int * measure_array,
		 short * color_array,
		 int num_colors,
		 int * CF_marker,
		 int * CF_marker_offd,
		 hypre_ParCSRMatrix * S,
		 hypre_ParCSRMatrix * A,
		 int * S_ext_i,
		 int * S_ext_j,
		 hypre_QueueElement * bucket_elements,
		 hypre_Queue * buckets,
		 hypre_QueueElement * nodes,
		 hypre_Queue * connected_nodes,
		 int * painted_dependers)
// This function updates weights corresponding to inbound edges to a
// new C-point. All inbound edges are removed and the measure_array
// entry for the neighbor node is decremented.
{
  hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
  hypre_ParCSRCommHandle   *comm_handle;
  
  hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
  int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
  int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);
  
  hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
  int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
  int                *S_offd_j        = hypre_CSRMatrixJ(S_offd);
  
  int 		      *col_map_offd    = hypre_ParCSRMatrixColMapOffd(S);
  int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
  int                 j, jS, k, kS, kc, break_var, neighborNode, currBucketIndex,
                      numAssigned;

  hypre_Queue         *updated;
  double              my_search_wall_time;

  // Mark the new C-point and remove it from the connected_nodes
  // list.
  numAssigned = 1;
  CF_marker[newCPT] = C_PT;
  removeElement(&nodes[newCPT], connected_nodes);
  painted_dependers[newCPT] = -1;

  // Remove edges from the new C-point to its neighbors. Update
  // weights and buckets accordingly.
  for(j = S_offd_i[newCPT]; j < S_offd_i[newCPT+1]; j++) {
    if(S_offd_j[j] > -1 && CF_marker_offd[S_offd_j[j]] == 0) {
      // Then this edge is still in the graph, so update its weight
      // and mark it as a neighbor to the new C-point.
      //measure_array[num_variables+j]--;
      CF_marker_offd[S_offd_j[j]] = COMMON_C_PT;
    }
  }
  for(j = S_diag_i[newCPT]; j < S_diag_i[newCPT+1]; j++) {
    neighborNode = S_diag_j[j];
    if(neighborNode > -1) {
      // Then this edge is still in the graph, so update its weight.

      // "Remove" the edge from the graph.
      S_diag_j[j] = -S_diag_j[j] - 1;

      // Now move the neighborNode to its new bucket.
      currBucketIndex = bucketIndex(measure_array[neighborNode],
				    color_array[neighborNode], num_colors);
      removeElement(&bucket_elements[neighborNode],
		    &buckets[currBucketIndex]);
      measure_array[neighborNode]--;
      if(measure_array[neighborNode] == 0) {
	// This node is now an F-point. Mark it as such.
	CF_marker[neighborNode] = F_PT;
	numAssigned++;

	// If this F-point has no remaining nodes strongly influencing
	// it, remove it from the connected_nodes list.
	removeIfDisconnected(neighborNode, S_diag_i, S_diag_j, S_offd_i, S_offd_j,
			     painted_dependers);
      }
      else {
	// This node is not an F-point. Add it to a new bucket.
	enqueueElement(&bucket_elements[neighborNode],
		       &buckets[currBucketIndex-num_colors]);
      }
    }
  }

  return numAssigned;
}

int assignNewCPT_aggregate(int newCPT,
		 int * measure_array,
		 int * curr_bucket_measure_array,
		 short * color_array,
		 int num_colors,
		 int * CF_marker,
		 hypre_ParCSRMatrix * S,
		 hypre_ParCSRMatrix * A,
		 hypre_QueueElement * bucket_elements,
		 hypre_Queue * buckets,
		 hypre_QueueElement * nodes,
		 hypre_Queue * connected_nodes,
		 int * painted_dependers)
// This function updates weights corresponding to inbound edges to a
// new C-point. All inbound edges are removed and the measure_array
// entry for the neighbor node is decremented.
{
  hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
  hypre_ParCSRCommHandle   *comm_handle;
  
  hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
  int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
  int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);
  
  hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
  int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
  int                *S_offd_j        = hypre_CSRMatrixJ(S_offd);
  
  int 		      *col_map_offd    = hypre_ParCSRMatrixColMapOffd(S);
  int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
  int                 j, jS, k, kS, kc, break_var, neighborNode, currBucketIndex,
                      numAssigned;

  hypre_Queue         *updated;
  double              my_search_wall_time;

  // Mark the new C-point and remove it from the connected_nodes
  // list.
  numAssigned = 1;
  CF_marker[newCPT] = C_PT;
  removeElement(&nodes[newCPT], connected_nodes);
  painted_dependers[newCPT] = -1;

  for(j = S_diag_i[newCPT]; j < S_diag_i[newCPT+1]; j++) {
    neighborNode = S_diag_j[j];
    if(neighborNode > -1) {
      // "Remove" the edge from the graph.
      S_diag_j[j] = -S_diag_j[j] - 1;

      // Then this edge is still in the graph, so update its weight.
      measure_array[neighborNode]--;

      // Do not move the node to its new bucket. Wait to do that until
      // it is necessary (i.e. it is in the selected bucket or it
      // becomes an F-point).
      if(measure_array[neighborNode] == 0) {

	// Remove it from whichever bucket it currently resides in.
	currBucketIndex = bucketIndex(curr_bucket_measure_array[neighborNode],
				      color_array[neighborNode], num_colors);
	removeElement(&bucket_elements[neighborNode], &buckets[currBucketIndex]);

	if(painted_dependers[neighborNode] > -1) {
	// This node is now an F-point. Mark it as such.
	CF_marker[neighborNode] = F_PT;
	numAssigned++;

	// If this F-point has no remaining nodes strongly influencing
	// it, remove it from the connected_nodes list.
	removeIfDisconnected(neighborNode, S_diag_i, S_diag_j, S_offd_i, S_offd_j,
			     painted_dependers);
	}
      }
    }
  }

  return numAssigned;
}

int updatePaintedNode(int nodeID,
		      int * measure_array,
		      short * color_array,
		      int num_colors,
		      int * CF_marker,
		      int * CF_marker_offd,
		      hypre_ParCSRMatrix * S,
		      hypre_ParCSRMatrix * A,
		      int * S_ext_i,
		      int * S_ext_j,
		      hypre_QueueElement * bucket_elements,
		      hypre_Queue * buckets,
		      int * painted_dependers,
		      int iteration)
// This function updates weights after a new C-point has been
// selected. This is done in the CLJP manner. Nodes are marked as
// F-points in this function if their measures reach zero.
{
  MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(S);
  hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
  hypre_ParCSRCommHandle   *comm_handle;
  
  hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
  int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
  int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);
  
  hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
  int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
  int                *S_offd_j        = hypre_CSRMatrixJ(S_offd);
  
  int 		      *col_map_offd    = hypre_ParCSRMatrixColMapOffd(S);
  int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
  int                 j, k , currBucketIndex, numAssigned, neighborNode, fullyDisconnected;

  double my_search_wall_time;

  numAssigned = 0;
  fullyDisconnected = 1;

  // Loop through nodes strongly influencing nodeID and see if they
  // are also painted. If a painted node is found, remove the edge and
  // decrement that node's weight (not nodeID's weight).
  for(j = S_diag_i[nodeID]; j < S_diag_i[nodeID+1]; j++) {
    neighborNode = S_diag_j[j];
    if(neighborNode > -1) {
      // Then the edge between nodeID and neighborNode still
      // exists. Check if neighborNode is painted and do appropriate
      // updates if it is.
      if(painted_dependers[neighborNode] == iteration) {
	// Remove the edge.
	S_diag_j[j] = -neighborNode-1;
	    
	// Update the weight and bucket.
	currBucketIndex = bucketIndex(measure_array[neighborNode],
				      color_array[neighborNode], num_colors);
	removeElement(&bucket_elements[neighborNode], &buckets[currBucketIndex]);
	measure_array[neighborNode]--;
	    
	// Finally, see if neighborNode becomes an F-point as a
	// result.
	if(measure_array[neighborNode] == 0) {
	  CF_marker[neighborNode] = F_PT;
	  numAssigned++;
	  removeIfDisconnected(neighborNode, S_diag_i, S_diag_j, S_offd_i, S_offd_j,
			       painted_dependers);
	}
	else {
	  enqueueElement(&bucket_elements[neighborNode],
			 &buckets[currBucketIndex-num_colors]);
	}
      }
      else if(CF_marker[neighborNode] == C_PT) {
	// Remove the edge.
	S_diag_j[j] = -neighborNode-1;
      }
      else
	fullyDisconnected = 0;
    }
  }

  if(fullyDisconnected && CF_marker[nodeID] == F_PT) {
    // Then this node (nodeID) is has no inbound edges. If it is an
    // F-point, then it is completely disconnected and should be
    // marked as such.
    painted_dependers[nodeID] = -1;
  }

  return numAssigned;
}

int updatePaintedNode_aggregate(int nodeID,
		      int * measure_array,
		      int * curr_bucket_measure_array,
		      short * color_array,
		      int num_colors,
		      int * CF_marker,
		      hypre_ParCSRMatrix * S,
		      hypre_ParCSRMatrix * A,
		      hypre_QueueElement * bucket_elements,
		      hypre_Queue * buckets,
		      int * painted_dependers,
		      int iteration)
// This function updates weights after a new C-point has been
// selected. This is done in the CLJP manner. Nodes are marked as
// F-points in this function if their measures reach zero.
{
  MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(S);
  hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
  hypre_ParCSRCommHandle   *comm_handle;
  
  hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
  int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
  int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);
  
  hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
  int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
  int                *S_offd_j        = hypre_CSRMatrixJ(S_offd);
  
  int 		      *col_map_offd    = hypre_ParCSRMatrixColMapOffd(S);
  int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
  int                 j, k , currBucketIndex, numAssigned, neighborNode, fullyDisconnected;

  double my_search_wall_time;

  numAssigned = 0;
  fullyDisconnected = 1;

  // Loop through nodes strongly influencing nodeID and see if they
  // are also painted. If a painted node is found, remove the edge and
  // decrement that node's weight (not nodeID's weight).
  for(j = S_diag_i[nodeID]; j < S_diag_i[nodeID+1]; j++) {
    neighborNode = S_diag_j[j];
    if(neighborNode > -1) {
      // Then the edge between nodeID and neighborNode still
      // exists. Check if neighborNode is painted and do appropriate
      // updates if it is.
      if(painted_dependers[neighborNode] == iteration ||
	 painted_dependers[neighborNode] == -3) {
	// Remove the edge.
	S_diag_j[j] = -neighborNode-1;
	    
	// Update the weight.
	measure_array[neighborNode]--;

	// Do not move the node to its new bucket. Wait to do that until
	// it is necessary (i.e. it is in the selected bucket or it
	// becomes an F-point).
	    
	// Finally, see if neighborNode becomes an F-point as a
	// result.
	if(measure_array[neighborNode] == 0) {

	  // Remove it from whichever bucket it currently resides in.
	  currBucketIndex = bucketIndex(curr_bucket_measure_array[neighborNode],
					color_array[neighborNode], num_colors);
	  removeElement(&bucket_elements[neighborNode], &buckets[currBucketIndex]);
//if(S_offd_i[neighborNode] != S_offd_i[neighborNode+1]) printf("that should not have happened %d\n", painted_dependers[neighborNode]);

	  if(painted_dependers[neighborNode] > -3) {
	  CF_marker[neighborNode] = F_PT;
	  numAssigned++;

	  removeIfDisconnected(neighborNode, S_diag_i, S_diag_j, S_offd_i, S_offd_j,
			       painted_dependers);
	  }
	}
      }
      else if(CF_marker[neighborNode] == C_PT) {
	// Remove the edge.
	S_diag_j[j] = -neighborNode-1;
      }
      else
	fullyDisconnected = 0;
    }
  }

  if(fullyDisconnected && CF_marker[nodeID] == F_PT) {
    // Then this node (nodeID) is has no inbound edges. If it is an
    // F-point, then it is completely disconnected and should be
    // marked as such.
    painted_dependers[nodeID] = -1;
  }

  return numAssigned;
}

int updateOffProcWeights(int newCPT,
		 int * measure_array,
			 short * color_array,
			 int num_colors,
			 int * CF_marker,
			 int * CF_marker_offd,
			 int * curr_offd_marker,
			 hypre_ParCSRMatrix * S,
			 hypre_ParCSRMatrix * A,
			 int * S_ext_i,
			 int * S_ext_j,
			 hypre_QueueElement * bucket_elements,
			 hypre_Queue * buckets)
{
  MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(S);
  hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
  hypre_ParCSRCommHandle   *comm_handle;
  
  hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
  int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
  int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);
  
  hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
  int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
  int                *S_offd_j        = hypre_CSRMatrixJ(S_offd);
  
  int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
  int                 j, jS, k, kS, kc, break_var, neighborNode, offProcNode,
    *origBucketIndexes, newBucketIndex, numAssigned;

  int                 num_cols_offd = hypre_CSRMatrixNumCols(S_offd);
  hypre_Queue         *updated;

  numAssigned = 0;
  origBucketIndexes = hypre_CTAlloc(int, S_ext_i[newCPT+1]-S_ext_i[newCPT]);

  // Determine which off-diagonal nodes became C-points in the
  // last iteration and update on-processor node's weights
  // accordingly.
  if((CF_marker_offd[newCPT]) != C_PT && (curr_offd_marker[newCPT] == C_PT)) {
    // This i points to a new C-point. Mark all nodes that
    // influence it as COMMON_C_PTs.
    for(j = S_ext_i[newCPT]; j < S_ext_i[newCPT+1]; j++) {
      if(S_ext_j[j] > -1 && CF_marker[S_ext_j[j]] == 0) {
	origBucketIndexes[j-S_ext_i[newCPT]] = bucketIndex(measure_array[S_ext_j[j]],
					    color_array[S_ext_j[j]], num_colors);
	measure_array[S_ext_j[j]]--;
	CF_marker[S_ext_j[j]] = COMMON_C_PT;
      }
      else if(S_ext_j[j] < 0 && S_ext_j[j] > INT_MIN)
	if(CF_marker_offd[-S_ext_j[j]-1] == 0)
	  CF_marker_offd[-S_ext_j[j]-1] = COMMON_C_PT;
    }

    // Now process all of the COMMON_C_PTs.
    for(j = S_ext_i[newCPT]; j < S_ext_i[newCPT+1]; j++) {
//printf("hi %d %d %d\n", j, S_ext_j, S_ext_j[j]);
      if(S_ext_j[j] > -1 && CF_marker[S_ext_j[j]] == COMMON_C_PT) {
	neighborNode = S_ext_j[j];
	// Check all nodes that influence this node. If they
	// are marked as COMMON_C_PTs, then remove the edge
	// and update the weight of the influencing node.
	for(k = S_diag_i[neighborNode]; k < S_diag_i[neighborNode+1]; k++) {
	  if(S_diag_j[k] > -1 && CF_marker[S_diag_j[k]] == COMMON_C_PT) {
	    measure_array[S_diag_j[k]]--;
	    S_diag_j[k] = -S_diag_j[k] - 1;
	  }
	}
	for(k = S_offd_i[neighborNode]; k < S_offd_i[neighborNode+1]; k++) {
	  if(S_offd_j[k] > -1 && CF_marker_offd[S_offd_j[k]] == COMMON_C_PT) {
	    S_offd_j[k] = -S_offd_j[k] - 1;
	    // The measure of S_offd_j[k] is updated on its
	    // processor. The code where that happens in the
	    // next loop.
	  }
	  else if(S_offd_j[k] == newCPT)
	    // Remove edge from neighborNode to new C-point.
	    S_offd_j[k] = -S_offd_j[k] - 1;
	}
      }
      else if(S_ext_j[j] < 0 && S_ext_j[j] > INT_MIN
	      && CF_marker_offd[-S_ext_j[j]-1] == COMMON_C_PT) {
	// Check if this off-processor nodes marked
	// COMMON_C_PT are strongly influeced by
	// neighborNode. This operation affects the measure
	// value at neighborNode.
	offProcNode = -S_ext_j[j] - 1;
	for(k = S_ext_i[offProcNode]; k < S_ext_i[offProcNode+1]; k++) {
	  if(S_ext_j[k] > -1 && CF_marker[S_ext_j[k]] == COMMON_C_PT) {
	    measure_array[S_ext_j[k]]--;
	    S_ext_j[k] = INT_MIN; //NEW
	  }
	  else if(-S_ext_j[k]-1 == newCPT)
	    // Remove edge from offProcNode to newCPT.
	    S_ext_j[k] = INT_MIN;
	}
      }
    }
    // Reset values of nodes that were made
    // COMMON_C_PTs. Also, check to see if on-processor nodes
    // have become F-points. Move nodes to their new buckets.
      
    for(j = S_ext_i[newCPT]; j < S_ext_i[newCPT+1]; j++) {
      if(S_ext_j[j] > -1 && CF_marker[S_ext_j[j]] == COMMON_C_PT) {
	// On-processor nodes.
	neighborNode = S_ext_j[j];
	// "Remove" edge from newCPT to neighbor node.
	S_ext_j[j] = INT_MIN;
	// Whether node is now an F-point or not, it needs to
	// be removed from its current bucket.
	removeElement(&bucket_elements[neighborNode],
		      &buckets[origBucketIndexes[j-S_ext_i[newCPT]]]);

	if(measure_array[neighborNode] == 0) {
	  // This node is now an F-point.
	  CF_marker[neighborNode] = F_PT;
	  numAssigned++;
	}
	else {
	  CF_marker[neighborNode] = 0;
	  // This node is not an F-point. Add it to a new bucket.
	  newBucketIndex = bucketIndex(measure_array[neighborNode],
				       color_array[neighborNode], num_colors);
	  enqueueElement(&bucket_elements[neighborNode],
			 &buckets[newBucketIndex]);
	}
      }
      else if(S_ext_j[j] < 0 && S_ext_j[j] > INT_MIN) {
	// Off-processor nodes.
	CF_marker_offd[-S_ext_j[j]-1] = 0;
	// "Remove" edge from newCPT to this node.
	S_ext_j[j] = INT_MIN;
      }
    }
    // Update the value in CF_marker_offd to the current value.
    CF_marker_offd[newCPT] = curr_offd_marker[newCPT];
  }

  hypre_TFree(origBucketIndexes);
  return numAssigned;
}

int hypre_BoomerAMGCoarsenBSIS( hypre_ParCSRMatrix    *S,
				hypre_ParCSRMatrix    *A,
				int                    CF_init,
				int                    debug_flag,
				int                  **CF_marker_ptr,
				int                    global,
				int                    level,
				int                   *measure_array)
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle   *comm_handle;

   hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
   int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
   int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
   int                *S_offd_j;

   int 		      *col_map_offd    = hypre_ParCSRMatrixColMapOffd(S);
   int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
   int		       col_1 = hypre_ParCSRMatrixFirstColDiag(S);
   int		       col_n = col_1 + hypre_CSRMatrixNumCols(S_diag);
   int 		       num_cols_offd = 0;
                  
   hypre_CSRMatrix    *S_ext;
   int                *S_ext_i;
   int                *S_ext_j;

   int		       num_sends = 0;
   int  	      *int_buf_data;

   int                *CF_marker;
   int                *CF_marker_offd, *curr_offd_marker;

   short              *color_array = NULL;
   int                num_colors;
                      
   //double             *measure_array;
   int                *graph_array;
   int                *graph_array_offd;
   int                 graph_size;
   int                 graph_offd_size;
   int                 global_graph_size;
                      
   int                 i, j, k, kc, jS, kS, ig;
   int		       index, start, my_id, num_procs, jrow, cnt;

   int                 max_strong_connections;

   hypre_Queue         *buckets, connected_nodes, *distTwoNodes;
   hypre_QueueElement  *bucket_elements, *nodes, *temp_element;
   int                 *bucket_element_data, *temp_node;
   int                 num_buckets, bucket_index, max_weight_measure, max_weight_color,
     local_max_weight_measure, local_max_weight_color, max_bucket_weight;

   int                 num_finished_neighbors[2], *finished_neighbors_array, finished,
                       num_assigned, c_iteration, nodeID, dependerID;
                      
   int                 ierr = 0;
   int                 break_var = 1;

   double	    wall_time;
#ifdef FINE_GRAIN_TIMINGS
   double           my_setup_wall_time, my_update_wall_time, my_search_wall_time,
                    setup_time=0, update_time=0, search_time=0, temp_time;
#endif
   int   iter = 0;

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   int   iter = 0;
#endif

#ifdef FINE_GRAIN_TIMINGS
my_setup_wall_time = time_getWallclockSeconds();
#endif
   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);
   color_array = hypre_CTAlloc(short, num_variables);
   num_colors = 0;

   finished_neighbors_array = hypre_CTAlloc(int, hypre_ParCSRCommPkgNumSends(comm_pkg)+
					    hypre_ParCSRCommPkgNumRecvs(comm_pkg));
   num_finished_neighbors[0] = 0; // holds number of neighbors this
                                  // processor sends to who are
				  // finished
   num_finished_neighbors[1] = 0; // holds number of neighbors this
				  // processor receives from who are
				  // finished

   if(num_procs > 1)
     parColorGraphNew(A, S, color_array, &num_colors, level);
   else
     seqColorGraphTheFinalWord(A, color_array, &num_colors, 0, level);
   /*--------------------------------------------------------------
    * Compute a  ParCSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   S_ext = NULL;
   if (debug_flag == 3) wall_time = time_getWallclockSeconds();
   if (!comm_pkg)
   {
        hypre_MatvecCommPkgCreate(A);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
 
   num_cols_offd = hypre_CSRMatrixNumCols(S_offd);

   S_diag_j = hypre_CSRMatrixJ(S_diag);

   if (num_cols_offd)
   {
      S_offd_j = hypre_CSRMatrixJ(S_offd);
   }
   /*----------------------------------------------------------
    * Compute the measures
    *
    * The measures are currently given by the column sums of S.
    * Hence, measure_array[i] is the number of influences
    * of variable i.
    *----------------------------------------------------------*/

   if(!measure_array) {
     // Then the measure array needs to be computed. The CR function
     // computes the measure_array ahead of time to save extra
     // computation.
     measure_array = hypre_CTAlloc(int, num_variables+num_cols_offd);

     for (i=0; i < S_offd_i[num_variables]; i++) {
       measure_array[num_variables + S_offd_j[i]] ++;
     }
     if (num_procs > 1)
       comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg,
						  &measure_array[num_variables],
						  int_buf_data);
     // THE ABOVE COMMUNICATION IS UNNECESSARY IF S_ext is used instead.

     // Count strong connections and store number of maximum
     // connections.
     for(i = 0; i < num_variables; i++) {
       for(j = S_diag_i[i]; j < S_diag_i[i+1]; j++) {
	 measure_array[S_diag_j[j]]++;
       }
     }

     if (num_procs > 1)
       hypre_ParCSRCommHandleDestroy(comm_handle);
      
     index = 0;
     for (i=0; i < num_sends; i++) {
       start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
       for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
	 measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
	   += int_buf_data[index++];
     }

     max_strong_connections = 0;
     for(i = 0; i < num_variables; i++) {
       if(measure_array[i] > max_strong_connections)
	 max_strong_connections = measure_array[i];
     }

     for (i=num_variables; i < num_variables+num_cols_offd; i++) {
       /* This loop zeros out the measures for the off-process nodes since
	  they currently contain the number of influences on those
	  nodes. Futher down the measures of those nodes will be
	  received from their processors. */
       measure_array[i] = 0;
     }
   }

   if (CF_init)
      CF_marker = *CF_marker_ptr;
   else
      CF_marker = hypre_CTAlloc(int, num_variables);

   if (num_cols_offd) {
      CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
      curr_offd_marker = hypre_CTAlloc(int, num_cols_offd);
   }
   else {
      CF_marker_offd = NULL;
      curr_offd_marker = NULL;
   }


   if (num_procs > 1)
   {
      S_ext      = hypre_ParCSRMatrixExtractBExt(S,A,0);
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixJ(S_ext);
   }

   /*  compress S_ext  and convert column numbers*/
   index = 0;
   for (i=0; i < num_cols_offd; i++)
   {
      for (j=S_ext_i[i]; j < S_ext_i[i+1]; j++)
      {
	 k = S_ext_j[j];
	 if (k >= col_1 && k < col_n)
	 {
	    S_ext_j[index++] = k - col_1;
	 }
	 else
	 {
	    kc = hypre_BinarySearch(col_map_offd,k,num_cols_offd);
	    if (kc > -1) S_ext_j[index++] = -kc-1;
	 }
      }
      S_ext_i[i] = index;
   }
   for (i = num_cols_offd; i > 0; i--)
      S_ext_i[i] = S_ext_i[i-1];
   if (num_procs > 1) S_ext_i[0] = 0;

   /*---------------------------------------------------
    * Initialize the bucket data structure.
    *---------------------------------------------------*/
   //distTwoNodes = buildDistTwoUpdateList(S, A, S_ext_i, S_ext_j);
   //destroyDistTwoUpdateList(num_variables, distTwoNodes);
   hypre_CSRMatrix * S_diagT;
   int * painted_dependers = hypre_CTAlloc(int, num_variables);
   hypre_Queue painted_dependers_list;
   initializeQueue(&painted_dependers_list);
   hypre_CSRMatrixTranspose(S_diag, &S_diagT, 0);
   int * S_diagT_i        = hypre_CSRMatrixI(S_diagT);
   int * S_diagT_j        = hypre_CSRMatrixJ(S_diagT);

   num_buckets = max_strong_connections*num_colors;
   buckets = hypre_CTAlloc(hypre_Queue, num_buckets);

   for(i = 0; i < num_buckets; i++)
     initializeQueue(&buckets[i]);
   initializeQueue(&connected_nodes);

   // Initialize the elements that go into the buckets.
   bucket_elements = hypre_CTAlloc(hypre_QueueElement, num_variables);
   bucket_element_data = hypre_CTAlloc(int, num_variables);
   nodes = hypre_CTAlloc(hypre_QueueElement, num_variables);

   // Add each node to the appropriate bucket.
   max_bucket_weight = 0;
   num_assigned = 0;
   for(i = 0; i < num_variables; i++) {
     bucket_element_data[i] = i;
     bucket_elements[i].data = &bucket_element_data[i];
     nodes[i].data = &bucket_element_data[i];
     enqueueElement(&nodes[i], &connected_nodes);
     if(measure_array[i] > 0) {
       bucket_index = bucketIndex(measure_array[i], color_array[i], num_colors);
       enqueueElement(&bucket_elements[i],
		      &buckets[bucket_index]);

       // Keep track of maximum weight bucket that is not empty.
       if(bucket_index > max_bucket_weight)
	 max_bucket_weight = bucket_index;
     }
     else {
       // This node influences no other node. Make it an F-point.
       CF_marker[i] = F_PT;
       num_assigned++;
       removeIfDisconnected(i, S_diag_i, S_diag_j, S_offd_i, S_offd_j, painted_dependers);
     }
   }

   local_max_weight_measure = measureWeight(max_bucket_weight, num_colors);
   local_max_weight_color = colorWeight(max_bucket_weight, num_colors);

#ifdef FINE_GRAIN_TIMINGS
setup_time += time_getWallclockSeconds() - my_setup_wall_time;
#endif

   finished = (num_variables == 0);
   c_iteration = 0;
   while(num_assigned < num_variables) {
     /*---------------------------------------------------
      * Send this node's maximum weight value on the processor
      * boundaries to neighboring processors.
      *---------------------------------------------------*/
#ifdef FINE_GRAIN_TIMINGS
my_search_wall_time = time_getWallclockSeconds();
#endif
     max_weight_measure = local_max_weight_measure;
     max_weight_color = local_max_weight_color;
     if (num_procs > 1)
       sendMaxWeight(comm_pkg, &max_weight_measure, &max_weight_color, finished,
		     num_finished_neighbors, finished_neighbors_array);
#ifdef FINE_GRAIN_TIMINGS
search_time += time_getWallclockSeconds() - my_search_wall_time;
#endif

     // Now the max weight on this processor and the boundaries of all
     // neighboring processors is known. We can now select all nodes
     // in the bucket with this max weight and add them to the C-point
     // set. Then the weights of their neighbors are updated.
#ifdef FINE_GRAIN_TIMINGS
my_update_wall_time = time_getWallclockSeconds();
#endif
     if(max_weight_measure == local_max_weight_measure
	&& max_weight_color == local_max_weight_color) {
#ifdef FINE_GRAIN_TIMINGS
my_search_wall_time = time_getWallclockSeconds();
#endif
       bucket_index = bucketIndex(max_weight_measure, max_weight_color, num_colors);
       temp_element = buckets[bucket_index].head;
#ifdef FINE_GRAIN_TIMINGS
temp_time = time_getWallclockSeconds() - my_search_wall_time;
search_time += temp_time;
update_time -= temp_time;
#endif
       while(temp_element) {
	 c_iteration++;
	 nodeID = *(temp_element->data);
	 // "Paint" all nodes that initially depended on the new
	 // C-point (unless they have no outgoing or incident
	 // edges). Add all painted nodes to the
	 // painted_dependers_list.
	 for(j = S_diagT_i[nodeID]; j < S_diagT_i[nodeID+1]; j++) {
	   dependerID = S_diagT_j[j];
	   if(painted_dependers[dependerID] > -1 &&
	      painted_dependers[dependerID] < c_iteration) {
	     painted_dependers[dependerID] = c_iteration;
	     enqueueData(&bucket_element_data[dependerID], &painted_dependers_list);
	   }
	 }

	 // Remove all incoming edges (edges from nodes the new
	 // C-point depends on). This updates decrements the weight of
	 // the neighbor node.
	 num_assigned += assignNewCPT(nodeID, measure_array, color_array,
				      num_colors, CF_marker, CF_marker_offd, S, A,
				      S_ext_i, S_ext_j, bucket_elements, buckets, nodes,
				      &connected_nodes, painted_dependers);

	 // Loop through all nodes in connected_nodes and check for
	 // edges shared between two nodes that are both strongly
	 // influenced by a new C-point. Also, remove strong influences
	 // by nodes in connected_nodes that connect to new C-points.
	 temp_node = dequeue(&painted_dependers_list);
	 while(temp_node) {
	   if(painted_dependers[*temp_node] > -1) {
	     num_assigned += updatePaintedNode(*temp_node, measure_array, color_array,
					       num_colors, CF_marker, CF_marker_offd, S, A,
					       S_ext_i, S_ext_j, bucket_elements, buckets,
					       painted_dependers, c_iteration);
	   }
	   temp_node = dequeue(&painted_dependers_list);
	 }

#ifdef FINE_GRAIN_TIMINGS
my_search_wall_time = time_getWallclockSeconds();
#endif
	 temp_element = temp_element->next_elt;
#ifdef FINE_GRAIN_TIMINGS
temp_time = time_getWallclockSeconds() - my_search_wall_time;
search_time += temp_time;
update_time -= temp_time;
#endif
       }

       // Now mark all of the new C-points as having a measure weight
       // of zero. This is important because it is used to
       // differentiate between new C-points and previously selected
       // C-points in updateConnectedNodes.
       temp_node = dequeue(&buckets[bucket_index]);
       while(temp_node) {
	 measure_array[*temp_node] = 0;
	 temp_node = dequeue(&buckets[bucket_index]);
       }
     }

     // The previous loop updated all of the weight information for
     // the on-processor connections to new C-points. Now
     // communicate with neighbors to send them the new
     // information. Each neighbor must then update the weights of
     // its nodes that neighbor new C-points on this processor.
     //
     // Send off-diagonal portion of measure array. This contains
     // the number of connections each off diagonal node has to new
     // C-points on this processor.
     //
     // Also send this processors list of C-points on the processor
     // boundary. The neighboring processors can use this
     // information to determine which nodes have been marked as
     // C-points in this iteration.
     if (num_procs > 1) {
       index = 0;
       for (i = 0; i < num_sends; i++) {
	 start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	 for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
	   int_buf_data[index++]
	     = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
       }
       
       exchangeIntBuf(comm_pkg, int_buf_data, curr_offd_marker,
		      num_finished_neighbors, finished_neighbors_array);

/*        comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, */
/* 						  curr_offd_marker); */
/*        hypre_ParCSRCommHandleDestroy(comm_handle); */
       
       for(i = 0; i < num_cols_offd; i++) {
	 num_assigned += updateOffProcWeights(i, measure_array, color_array,
					      num_colors, CF_marker, CF_marker_offd,
					      curr_offd_marker, S, A, S_ext_i,
					      S_ext_j, bucket_elements, buckets);
       }
     }
#ifdef FINE_GRAIN_TIMINGS
update_time += time_getWallclockSeconds() - my_update_wall_time;
#endif

       // Determine the index of the next largest non-empty bucket.
#ifdef FINE_GRAIN_TIMINGS
my_search_wall_time = time_getWallclockSeconds();
#endif
     bucket_index = bucketIndex(local_max_weight_measure, local_max_weight_color,
				num_colors);
     for(i = bucket_index; i > -1; i--) {
       //for(i = num_buckets-1; i > -1; i--) {
       if(buckets[i].head) {
	 // Then the bucket is non-empty. Set max_weight_measure and
	 // max_weight_color.
	 local_max_weight_measure = measureWeight(i, num_colors);
	 local_max_weight_color = colorWeight(i, num_colors);
	 i = -1; // to break loop
       }
     }
#ifdef FINE_GRAIN_TIMINGS
search_time += time_getWallclockSeconds() - my_search_wall_time;
#endif
   }

#ifdef FINE_GRAIN_TIMINGS
my_setup_wall_time = time_getWallclockSeconds();
#endif
   finished = 1;
   max_weight_measure = 0;
   /*---------------------------------------------------
    * Send this node's maximum weight value on the processor
    * boundaries to neighboring processors.
    *
    * This is the last time this will be done for this proceesor.
    *---------------------------------------------------*/
   sendMaxWeight(comm_pkg, &max_weight_measure, &max_weight_color, finished,
		 num_finished_neighbors, finished_neighbors_array);

//if(my_id == 1) {for(i = 0; i < num_variables; i++) printf("!!! %d\n", measure_array[i]); hypre_TFree(measure_array); printf("done\n");}

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   /* Reset S_matrix */
   for (i=0; i < S_diag_i[num_variables]; i++)
   {
      if (S_diag_j[i] < 0)
         S_diag_j[i] = -S_diag_j[i]-1;
   }
   for (i=0; i < S_offd_i[num_variables]; i++)
   {
      if (S_offd_j[i] < 0)
         S_offd_j[i] = -S_offd_j[i]-1;
   }
   /*for (i=0; i < num_variables; i++)
      if (CF_marker[i] == SF_PT) CF_marker[i] = F_PT;*/

   if(color_array)
     hypre_TFree(color_array);

   hypre_TFree(painted_dependers);
   hypre_TFree(S_diagT);

   hypre_TFree(measure_array);
   hypre_TFree(int_buf_data);
   hypre_TFree(bucket_element_data);
   hypre_TFree(bucket_elements);
   hypre_TFree(nodes);
   hypre_TFree(buckets);
   hypre_TFree(CF_marker_offd);
   hypre_TFree(curr_offd_marker);
   if (num_procs > 1) hypre_CSRMatrixDestroy(S_ext);

   *CF_marker_ptr   = CF_marker;

/* MPI_Barrier(comm); */
/* printf("Color results %d %d\n", my_id, num_variables); */
/* for(i = 0; i < num_variables; i++) { */
/*   printf("%d: %d\n", i, (*CF_marker_ptr)[i]); */
/* } */
/* MPI_Barrier(comm); */
#ifdef FINE_GRAIN_TIMINGS
setup_time += time_getWallclockSeconds() - my_setup_wall_time;
#endif

#ifdef FINE_GRAIN_TIMINGS
printf(" Setup time: %f\n", setup_time);
printf(" Update time: %f\n", update_time);
printf(" Search time: %f\n", search_time);
#endif

   return (ierr);
}

int hypre_BoomerAMGCoarsenBSIS_aggregateUpdate( hypre_ParCSRMatrix    *S,
				hypre_ParCSRMatrix    *A,
				int                    CF_init,
				int                    debug_flag,
				int                  **CF_marker_ptr,
				int                    global,
				int                    level,
				int                   *measure_array)
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle   *comm_handle;

   hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
   int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
   int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
   int                *S_offd_j;

   int 		      *col_map_offd    = hypre_ParCSRMatrixColMapOffd(S);
   int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
   int		       col_1 = hypre_ParCSRMatrixFirstColDiag(S);
   int		       col_n = col_1 + hypre_CSRMatrixNumCols(S_diag);
   int 		       num_cols_offd = 0;
                  
   hypre_CSRMatrix    *S_ext;
   int                *S_ext_i;
   int                *S_ext_j;

   int		       num_sends = 0;
   int  	      *int_buf_data;

   int                *CF_marker;

   short              *color_array = NULL;
   int                num_colors;
                      
   //double             *measure_array;
   int                *curr_bucket_measure_array;
   int                *graph_array;
   int                *graph_array_offd;
   int                 graph_size;
   int                 graph_offd_size;
   int                 global_graph_size;
                      
   int                 i, j, k, kc, jS, kS, ig;
   int		       index, start, my_id, num_procs, jrow, cnt;

   int                 max_strong_connections;

   hypre_Queue         *buckets, connected_nodes, *distTwoNodes;
   hypre_QueueElement  *bucket_elements, *nodes, *temp_element;
   int                 *bucket_element_data, *temp_node;
   int                 num_buckets, bucket_index, max_weight_measure, max_weight_color,
     max_bucket_weight,
     new_bucket_index, prev_bucket_index;

   int                 num_finished_neighbors[2], *finished_neighbors_array, finished,
                       num_assigned, c_iteration, nodeID, dependerID;
                      
   int                 ierr = 0;
   int                 break_var = 1;

   double	    wall_time;
#ifdef FINE_GRAIN_TIMINGS
   double           my_setup_wall_time, my_update_wall_time, my_search_wall_time,
                    setup_time=0, update_time=0, search_time=0, temp_time;
#endif
   int   iter = 0;
   int * painted_dependers = hypre_CTAlloc(int, num_variables);

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   int   iter = 0;
#endif

#ifdef FINE_GRAIN_TIMINGS
my_setup_wall_time = time_getWallclockSeconds();
#endif
   num_cols_offd = hypre_CSRMatrixNumCols(S_offd);

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);
   color_array = hypre_CTAlloc(short, num_variables+num_cols_offd);
   num_colors = 0;

   finished_neighbors_array = hypre_CTAlloc(int, hypre_ParCSRCommPkgNumSends(comm_pkg)+
					    hypre_ParCSRCommPkgNumRecvs(comm_pkg));
   num_finished_neighbors[0] = 0; // holds number of neighbors this
                                  // processor sends to who are
				  // finished
   num_finished_neighbors[1] = 0; // holds number of neighbors this
				  // processor receives from who are
				  // finished

   seqColorGraphTheFinalWord(A, color_array, &num_colors, 0, level);
   /*--------------------------------------------------------------
    * Compute a  ParCSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   S_ext = NULL;
   if (debug_flag == 3) wall_time = time_getWallclockSeconds();
   if (!comm_pkg)
   {
        hypre_MatvecCommPkgCreate(A);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
 
   S_diag_j = hypre_CSRMatrixJ(S_diag);

   if (num_cols_offd)
   {
      S_offd_j = hypre_CSRMatrixJ(S_offd);
   }
   /*----------------------------------------------------------
    * Compute the measures
    *
    * The measures are currently given by the column sums of S.
    * Hence, measure_array[i] is the number of influences
    * of variable i.
    *----------------------------------------------------------*/

   if(!measure_array) {
     // Then the measure array needs to be computed. The CR function
     // computes the measure_array ahead of time to save extra
     // computation.
     measure_array = hypre_CTAlloc(int, num_variables+num_cols_offd);
     curr_bucket_measure_array = hypre_CTAlloc(int, num_variables+num_cols_offd);

     for (i=0; i < S_offd_i[num_variables]; i++) {
       measure_array[num_variables + S_offd_j[i]] ++;
     }
     if (num_procs > 1)
       comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg,
						  &measure_array[num_variables],
						  int_buf_data);
     // THE ABOVE COMMUNICATION IS UNNECESSARY IF S_ext is used instead.

     // Count strong connections and store number of maximum
     // connections.
     for(i = 0; i < num_variables; i++) {
       for(j = S_diag_i[i]; j < S_diag_i[i+1]; j++) {
	 measure_array[S_diag_j[j]]++;
       }
     }

     if (num_procs > 1)
       hypre_ParCSRCommHandleDestroy(comm_handle);
      
     index = 0;
     for (i=0; i < num_sends; i++) {
       start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
       for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++) {
	 measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
	   += int_buf_data[index++];
	 painted_dependers[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)] = -2;
       }
     }

     max_strong_connections = 0;
     for(i = 0; i < num_variables; i++) {
       if(measure_array[i] > max_strong_connections)
	 max_strong_connections = measure_array[i];
     }

     for (i=num_variables; i < num_variables+num_cols_offd; i++) {
       /* This loop zeros out the measures for the off-process nodes since
	  they currently contain the number of influences on those
	  nodes. Futher down the measures of those nodes will be
	  received from their processors. */
       measure_array[i] = 0;
     }
   }

   if (CF_init)
      CF_marker = *CF_marker_ptr;
   else
      CF_marker = hypre_CTAlloc(int, num_variables);

   if (num_procs > 1)
   {
      S_ext      = hypre_ParCSRMatrixExtractBExt(S,A,0);
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixJ(S_ext);
   }

   /*  compress S_ext  and convert column numbers*/
   index = 0;
   for (i=0; i < num_cols_offd; i++)
   {
      for (j=S_ext_i[i]; j < S_ext_i[i+1]; j++)
      {
	 k = S_ext_j[j];
	 if (k >= col_1 && k < col_n)
	 {
	    S_ext_j[index++] = k - col_1;
	 }
	 else
	 {
	    kc = hypre_BinarySearch(col_map_offd,k,num_cols_offd);
	    if (kc > -1) S_ext_j[index++] = -kc-1;
	 }
      }
      S_ext_i[i] = index;
   }
   for (i = num_cols_offd; i > 0; i--)
      S_ext_i[i] = S_ext_i[i-1];
   if (num_procs > 1) S_ext_i[0] = 0;

   /*---------------------------------------------------
    * Initialize the bucket data structure.
    *---------------------------------------------------*/
   //distTwoNodes = buildDistTwoUpdateList(S, A, S_ext_i, S_ext_j);
   //destroyDistTwoUpdateList(num_variables, distTwoNodes);
   hypre_CSRMatrix * S_diagT;
   hypre_Queue painted_dependers_list;
   initializeQueue(&painted_dependers_list);
   hypre_Queue boundary_painted_dependers_list;
   initializeQueue(&boundary_painted_dependers_list);
   hypre_CSRMatrixTranspose(S_diag, &S_diagT, 0);
   int * S_diagT_i        = hypre_CSRMatrixI(S_diagT);
   int * S_diagT_j        = hypre_CSRMatrixJ(S_diagT);

   num_buckets = max_strong_connections*num_colors;
   buckets = hypre_CTAlloc(hypre_Queue, num_buckets);

   for(i = 0; i < num_buckets; i++)
     initializeQueue(&buckets[i]);
   initializeQueue(&connected_nodes);

   // Initialize the elements that go into the buckets.
   bucket_elements = hypre_CTAlloc(hypre_QueueElement, num_variables);
   bucket_element_data = hypre_CTAlloc(int, num_variables);
   nodes = hypre_CTAlloc(hypre_QueueElement, num_variables);

   // Add each node to the appropriate bucket.
   max_bucket_weight = 0;
   num_assigned = 0;
   for(i = 0; i < num_variables; i++) {
     bucket_element_data[i] = i;
     bucket_elements[i].data = &bucket_element_data[i];
     nodes[i].data = &bucket_element_data[i];
     enqueueElement(&nodes[i], &connected_nodes);
     //if(measure_array[i] > 0 && (S_offd_i[i+1] == S_offd_i[i])) {
     if(measure_array[i] > 0) {
       bucket_index = bucketIndex(measure_array[i], color_array[i], num_colors);
       enqueueElement(&bucket_elements[i],
		      &buckets[bucket_index]);
       curr_bucket_measure_array[i] = measure_array[i]; // this
	     // variable is used for the aggregate update

       // Keep track of maximum weight bucket that is not empty.
       if(bucket_index > max_bucket_weight)
	 max_bucket_weight = bucket_index;
     }
     else if(measure_array[i] == 0) {
       // This node influences no other node. Make it an F-point.
       CF_marker[i] = F_PT;
       num_assigned++;
       removeIfDisconnected(i, S_diag_i, S_diag_j, S_offd_i, S_offd_j, painted_dependers);
     }
/*      else { */
/*        //the node influences nodes off-processor -- do not add to buckets */
/*        painted_dependers[i] = -1; */
/*      } */
   }

   max_weight_measure = measureWeight(max_bucket_weight, num_colors);
   max_weight_color = colorWeight(max_bucket_weight, num_colors);

#ifdef FINE_GRAIN_TIMINGS
setup_time += time_getWallclockSeconds() - my_setup_wall_time;
#endif

   finished = (num_variables == 0);
   c_iteration = 0;
   while(num_assigned < num_variables) {
#ifdef FINE_GRAIN_TIMINGS
my_search_wall_time = time_getWallclockSeconds();
#endif
#ifdef FINE_GRAIN_TIMINGS
search_time += time_getWallclockSeconds() - my_search_wall_time;
#endif

     // Now the max weight on this processor and the boundaries of all
     // neighboring processors is known. We can now select all nodes
     // in the bucket with this max weight and add them to the C-point
     // set. Then the weights of their neighbors are updated.
#ifdef FINE_GRAIN_TIMINGS
my_update_wall_time = time_getWallclockSeconds();
#endif
#ifdef FINE_GRAIN_TIMINGS
my_search_wall_time = time_getWallclockSeconds();
#endif
     bucket_index = bucketIndex(max_weight_measure, max_weight_color, num_colors);
     if(prev_bucket_index == bucket_index)
       break;
     else
       prev_bucket_index = bucket_index;
     temp_element = buckets[bucket_index].head;
#ifdef FINE_GRAIN_TIMINGS
temp_time = time_getWallclockSeconds() - my_search_wall_time;
search_time += temp_time;
update_time -= temp_time;
#endif
     while(temp_element) {
       c_iteration++;
       nodeID = *(temp_element->data);

#ifdef FINE_GRAIN_TIMINGS
my_search_wall_time = time_getWallclockSeconds();
#endif
       temp_element = temp_element->next_elt;
#ifdef FINE_GRAIN_TIMINGS
temp_time = time_getWallclockSeconds() - my_search_wall_time;
search_time += temp_time;
update_time -= temp_time;
#endif
       if(measure_array[nodeID] == curr_bucket_measure_array[nodeID]) {
	 // Then this node has no updates to be done. If this is not
	 // true, then this node is in the wrong bucket and is to be
	 // moved.

	 // "Paint" all nodes that initially depended on the new
	 // C-point (unless they have no outgoing or incident
	 // edges). Add all painted nodes to the
	 // painted_dependers_list.
	 if(painted_dependers[nodeID] > -2) {
	   for(j = S_diagT_i[nodeID]; j < S_diagT_i[nodeID+1]; j++) {
	     dependerID = S_diagT_j[j];
	     if(painted_dependers[dependerID] != -1 &&
		painted_dependers[dependerID] < c_iteration) {
	       if(painted_dependers[dependerID] > -1) {
		 painted_dependers[dependerID] = c_iteration;
		 enqueueData(&bucket_element_data[dependerID], &painted_dependers_list);
	       }
	       else if(painted_dependers[dependerID] == -2) { // processor boundary node
		 painted_dependers[dependerID] = -3;
		 enqueueData(&bucket_element_data[dependerID], &painted_dependers_list);
		 enqueueData(&bucket_element_data[dependerID],
			     &boundary_painted_dependers_list);
	       }
	     }
	   }

	   // Remove all incoming edges (edges from nodes the new
	   // C-point depends on). This updates decrements the weight of
	   // the neighbor node.
	   num_assigned += assignNewCPT_aggregate(nodeID, measure_array,
				      curr_bucket_measure_array, color_array,
				      num_colors, CF_marker, S, A,
				      bucket_elements, buckets, nodes,
				      &connected_nodes, painted_dependers);

	   // Loop through all nodes in connected_nodes and check for
	   // edges shared between two nodes that are both strongly
	   // influenced by a new C-point. Also, remove strong influences
	   // by nodes in connected_nodes that connect to new C-points.
	   temp_node = dequeue(&painted_dependers_list);
	   while(temp_node) {
	     if(painted_dependers[*temp_node] != -1) {
	       num_assigned += updatePaintedNode_aggregate(*temp_node, measure_array,
					       curr_bucket_measure_array, color_array,
					       num_colors, CF_marker, S, A,
					       bucket_elements, buckets,
					       painted_dependers, c_iteration);
	     }
	     temp_node = dequeue(&painted_dependers_list);
	   }
	   temp_node = dequeue(&boundary_painted_dependers_list);
	   while(temp_node) { // reset boundary node's paint
	     painted_dependers[*temp_node] = -2;
	     temp_node = dequeue(&boundary_painted_dependers_list);
	   }
	 }
	 else { // this is a processor boundary node -- remove it and mark
	   //its influencers
	   removeElement(&bucket_elements[nodeID], &buckets[bucket_index]);
	   for(j = S_diagT_i[nodeID]; j < S_diagT_i[nodeID+1]; j++) {
	     dependerID = S_diagT_j[j];
	     if(painted_dependers[dependerID] > -1)
	       painted_dependers[dependerID] = -2;
	   }
	   for(j = S_diag_i[nodeID]; j < S_diag_i[nodeID+1]; j++) {
	     dependerID = S_diag_j[j];
	     if(dependerID < 0)
	       dependerID = -dependerID-1;
	     if(painted_dependers[dependerID] > -1)
	       painted_dependers[dependerID] = -2;
	   }
	 }
       }
       else {
	 // Remove the node from whichever bucket it currently
	 // resides in.
	 removeElement(&bucket_elements[nodeID], &buckets[bucket_index]);

	 // Move it to its new bucket.
	 new_bucket_index = bucketIndex(measure_array[nodeID], color_array[nodeID],
					num_colors);
	 enqueueElement(&bucket_elements[nodeID], &buckets[new_bucket_index]);

	 curr_bucket_measure_array[nodeID] = measure_array[nodeID];
       }
     }

       // Now mark all of the new C-points as having a measure weight
       // of zero. This is important because it is used to
       // differentiate between new C-points and previously selected
       // C-points in updateConnectedNodes.
       temp_node = dequeue(&buckets[bucket_index]);
       while(temp_node) {
	 measure_array[*temp_node] = 0;
	 temp_node = dequeue(&buckets[bucket_index]);
       }

#ifdef FINE_GRAIN_TIMINGS
update_time += time_getWallclockSeconds() - my_update_wall_time;
#endif

       // Determine the index of the next largest non-empty bucket.
#ifdef FINE_GRAIN_TIMINGS
my_search_wall_time = time_getWallclockSeconds();
#endif
     bucket_index = bucketIndex(max_weight_measure, max_weight_color,
				num_colors);
     for(i = bucket_index; i > -1; i--) {
       //for(i = num_buckets-1; i > -1; i--) {
       if(buckets[i].head) {
	 // Then the bucket is non-empty. Set max_weight_measure and
	 // max_weight_color.
	 max_weight_measure = measureWeight(i, num_colors);
	 max_weight_color = colorWeight(i, num_colors);
	 i = -1; // to break loop
       }
     }
#ifdef FINE_GRAIN_TIMINGS
search_time += time_getWallclockSeconds() - my_search_wall_time;
#endif
   }

#ifdef FINE_GRAIN_TIMINGS
my_setup_wall_time = time_getWallclockSeconds();
#endif
   finished = 1;
   max_weight_measure = 0;

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   if(num_procs > 1) {
//printf("entering!\n");
/*     ierr += hypre_BoomerAMGCoarsen (S, A, 1, debug_flag, &CF_marker); */
/* /\*      hypre_BoomerAMGCoarsenCLJP_c_improved(S, A, 1, debug_flag, &CF_marker, global, *\/ */
/* /\* 					   level, NULL); *\/ */

/*      hypre_BoomerAMGCoarsenBSISBoundaries(S, A, debug_flag, CF_marker, */
/* 					  measure_array, S_ext, S_ext_i, S_ext_j, */
/* 					  color_array); */

     hypre_BoomerAMGCoarsenBSISCLJP_cBoundaries(S, A, debug_flag, CF_marker,
						level, measure_array, S_ext, S_ext_i, S_ext_j,
						color_array, num_colors);

/* for(i = 0; i < num_variables; i++) { if(CF_marker[i] == 0) CF_marker[i] = F_PT;}  */
//printf("exiting!\n");
   }

   /* Reset S_matrix */
   for (i=0; i < S_diag_i[num_variables]; i++)
   {
      if (S_diag_j[i] < 0)
         S_diag_j[i] = -S_diag_j[i]-1;
   }
   for (i=0; i < S_offd_i[num_variables]; i++)
   {
      if (S_offd_j[i] < 0)
         S_offd_j[i] = -S_offd_j[i]-1;
   }
   /*for (i=0; i < num_variables; i++)
      if (CF_marker[i] == SF_PT) CF_marker[i] = F_PT;*/

   if(color_array)
     hypre_TFree(color_array);

   hypre_TFree(painted_dependers);
   hypre_TFree(S_diagT);

   hypre_TFree(measure_array);
   hypre_TFree(curr_bucket_measure_array);
   hypre_TFree(int_buf_data);
   hypre_TFree(bucket_element_data);
   hypre_TFree(bucket_elements);
   hypre_TFree(nodes);
   hypre_TFree(buckets);
   if (num_procs > 1) hypre_CSRMatrixDestroy(S_ext);

   *CF_marker_ptr   = CF_marker;

/* MPI_Barrier(comm); */
/* printf("Color results %d %d\n", my_id, num_variables); */
/* for(i = 0; i < num_variables; i++) { */
/*   printf("%d: %d\n", i, (*CF_marker_ptr)[i]); */
/* } */
/* MPI_Barrier(comm); */
#ifdef FINE_GRAIN_TIMINGS
setup_time += time_getWallclockSeconds() - my_setup_wall_time;
#endif

#ifdef FINE_GRAIN_TIMINGS
printf(" Setup time: %f\n", setup_time);
printf(" Update time: %f\n", update_time);
printf(" Search time: %f\n", search_time);
#endif

   return (ierr);
}

/**************************************************************
 *
 *      Modified Independent Set Coarsening routine
 *          (don't worry about strong F-F connections
 *           without a common C point)
 *
 *      Pre-color the graph using same technique as in
 *          CLJP-c.
 *
 **************************************************************/
int
hypre_BoomerAMGCoarsenPMIS_c( hypre_ParCSRMatrix    *S,
			      hypre_ParCSRMatrix    *A,
			      int                    CF_init,
			      int                    debug_flag,
			      int                  **CF_marker_ptr,
			      int                    global,
                              int                    level,
			      int                    distance_one_color,
			      double                *measure_array)
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommHandle   *comm_handle;

   hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
   int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
   int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
   int                *S_offd_j;

   int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
   int 		       num_cols_offd = 0;
                  
   /* hypre_CSRMatrix    *S_ext;
   int                *S_ext_i;
   int                *S_ext_j; */

   int		       num_sends = 0;
   int  	      *int_buf_data;
   double	      *buf_data;

   int                *CF_marker;
   int                *CF_marker_offd;
                      
   short              *color_array;
   int                num_colors;
                      
   //double             *measure_array;
   int                *graph_array;
   int                *graph_array_offd;
   int                 graph_size;
   int                 graph_offd_size;
   int                 global_graph_size;
                      
   int                 i, j, jS, ig;
   int		       index, start, my_id, num_procs, jrow, cnt, elmt;
                      
   int                 ierr = 0;
   int                 use_commpkg_A = 0;

   double	    wall_time;
   int   iter = 0;



#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   int   iter = 0;
#endif

   /*******************************************************************************
    BEFORE THE INDEPENDENT SET COARSENING LOOP:
      measure_array: calculate the measures, and communicate them
        (this array contains measures for both local and external nodes)
      CF_marker, CF_marker_offd: initialize CF_marker
        (separate arrays for local and external; 0=unassigned, negative=F point, positive=C point)
   ******************************************************************************/      

   if(!measure_array) {
     // Then the coloring needs to be done. The only time this is not needed is if
     // the calling function already has the measure_array computed. The CR function
     // does this because it calls CLJP_c several times for each level. By computing
     // the measure_array ahead of time, extra computation can be saved.
     color_array = hypre_CTAlloc(short, num_variables);

     if(global)
       parColorGraph(A, S, color_array, &num_colors, level);
     else if (distance_one_color)
       seqColorGraphNew(S, color_array, &num_colors, level);
     else
       seqColorGraphD2(S, color_array, &num_colors, level);
   }
   
   /*--------------------------------------------------------------
    * Use the ParCSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: S_data is not used; in stead, only strong columns are retained
    *       in S_j, which can then be used like S_data
    *----------------------------------------------------------------*/

   /*S_ext = NULL; */
   if (debug_flag == 3) wall_time = time_getWallclockSeconds();
   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   if (!comm_pkg)
   {
        use_commpkg_A = 1;
        comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   if (!comm_pkg)
   {
        hypre_MatvecCommPkgCreate(A);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
   buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
 
   num_cols_offd = hypre_CSRMatrixNumCols(S_offd);

   S_diag_j = hypre_CSRMatrixJ(S_diag);

   if (num_cols_offd)
   {
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

   if(!measure_array) {
     // Then the measure array needs to be computed. The CR function
     // computes the measure_array ahead of time to save extra
     // computation.
   measure_array = hypre_CTAlloc(double, num_variables+num_cols_offd);

   /* first calculate the local part of the sums for the external nodes */
   for (i=0; i < S_offd_i[num_variables]; i++)
   { 
      measure_array[num_variables + S_offd_j[i]] += 1.0;
   }

   /* now send those locally calculated values for the external nodes to the neighboring processors */
   if (num_procs > 1)
   comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg, 
                        &measure_array[num_variables], buf_data);

   /* calculate the local part for the local nodes */
   for (i=0; i < S_diag_i[num_variables]; i++)
   { 
      measure_array[S_diag_j[i]] += 1.0;
   }

   /* finish the communication */
   if (num_procs > 1)
   hypre_ParCSRCommHandleDestroy(comm_handle);
      
   /* now add the externally calculated part of the local nodes to the local nodes */
   index = 0;
   for (i=0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
                        += buf_data[index++];
   }

   /* set the measures of the external nodes to zero */
   for (i=num_variables; i < num_variables+num_cols_offd; i++)
   { 
      measure_array[i] = 0;
   }

   /* this augments the measures with a random number between 0 and 1 */
   /* (only for the local part) */
   //hypre_BoomerAMGIndepSetInit(S, measure_array, CF_init);
   hypre_BoomerAMGIndepSetInitb(S, measure_array, color_array, num_colors);
   }

   /*---------------------------------------------------
    * Initialize the graph arrays, and CF_marker arrays
    *---------------------------------------------------*/

   /* first the off-diagonal part of the graph array */
   if (num_cols_offd) 
      graph_array_offd = hypre_CTAlloc(int, num_cols_offd);
   else
      graph_array_offd = NULL;

   for (ig = 0; ig < num_cols_offd; ig++)
      graph_array_offd[ig] = ig;

   graph_offd_size = num_cols_offd;

   /* now the local part of the graph array, and the local CF_marker array */
   graph_array = hypre_CTAlloc(int, num_variables);

   if (CF_init==1)
   { 
      CF_marker = *CF_marker_ptr;
      cnt = 0;
      for (i=0; i < num_variables; i++)
      {
         if ( (S_offd_i[i+1]-S_offd_i[i]) > 0 || CF_marker[i] == -1)
	 {
	   if(CF_marker[i] != SF_PT) CF_marker[i] = 0;
	   //CF_marker[i] = 0;
	 }
         if ( CF_marker[i] == Z_PT)
         {
            if (measure_array[i] >= 1.0 ||
                (S_diag_i[i+1]-S_diag_i[i]) > 0)
            {
               CF_marker[i] = 0;
               graph_array[cnt++] = i;
            }
            else
            {
               CF_marker[i] = F_PT;
            }
         }
         else if (CF_marker[i] == SF_PT)
            measure_array[i] = 0;
         else
            graph_array[cnt++] = i;
      }
   }
   else
   {
      CF_marker = hypre_CTAlloc(int, num_variables);
      cnt = 0;
      for (i=0; i < num_variables; i++)
      {
         CF_marker[i] = 0;
         if ( (S_diag_i[i+1]-S_diag_i[i]) == 0
                && (S_offd_i[i+1]-S_offd_i[i]) == 0)
         {
            CF_marker[i] = SF_PT;
            measure_array[i] = 0;
         }
         else
            graph_array[cnt++] = i;
      }
   }
   graph_size = cnt;

   /* now the off-diagonal part of CF_marker */
   if (num_cols_offd)
     CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
   else
     CF_marker_offd = NULL;

   for (i=0; i < num_cols_offd; i++)
	CF_marker_offd[i] = 0;
  
   /*------------------------------------------------
    * Communicate the local measures, which are complete,
      to the external nodes
    *------------------------------------------------*/
   index = 0;
   for (i = 0; i < num_sends; i++)
     {
       start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
       for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
	 {
	   jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
	   buf_data[index++] = measure_array[jrow];
         }
     }
   
   if (num_procs > 1)
     { 
       comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg, buf_data, 
						  &measure_array[num_variables]);
       
       hypre_ParCSRCommHandleDestroy(comm_handle);   
       
     } 
      
   /* we need S_ext: the columns of the S matrix for the local nodes */
   /* we need this because the independent set routine can only decide
      which local nodes are in it when it knows both the rows and columns
      of S */

   /* if (num_procs > 1)
   {
      S_ext      = hypre_ParCSRMatrixExtractBExt(S,A,0);
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixJ(S_ext);
   } */

   /*  compress S_ext and convert column numbers*/

   /* index = 0;
   for (i=0; i < num_cols_offd; i++)
   {
      for (j=S_ext_i[i]; j < S_ext_i[i+1]; j++)
      {
	 k = S_ext_j[j];
	 if (k >= col_1 && k < col_n)
	 {
	    S_ext_j[index++] = k - col_1;
	 }
	 else
	 {
	    kc = hypre_BinarySearch(col_map_offd,k,num_cols_offd);
	    if (kc > -1) S_ext_j[index++] = -kc-1;
	 }
      }
      S_ext_i[i] = index;
   }
   for (i = num_cols_offd; i > 0; i--)
      S_ext_i[i] = S_ext_i[i-1];
   if (num_procs > 1) S_ext_i[0] = 0; */
 
   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Initialize CLJP phase = %f\n",
                     my_id, wall_time); 
   }

   /*******************************************************************************
    THE INDEPENDENT SET COARSENING LOOP:
   ******************************************************************************/      

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   while (1)
   {

     /* stop the coarsening if nothing left to be coarsened */
     MPI_Allreduce(&graph_size,&global_graph_size,1,MPI_INT,MPI_SUM,comm);

     if (global_graph_size == 0)
       break;

     /*     printf("\n");
     printf("*** MIS iteration %d\n",iter);
     printf("graph_size remaining %d\n",graph_size);*/

     /*------------------------------------------------
      * Pick an independent set of points with
      * maximal measure.
        At the end, CF_marker is complete, but still needs to be
        communicated to CF_marker_offd
      *------------------------------------------------*/
      if (!CF_init || iter)
      {
          hypre_BoomerAMGIndepSet(S, measure_array, graph_array, 
				graph_size, 
				graph_array_offd, graph_offd_size, 
				CF_marker, CF_marker_offd);

      /*------------------------------------------------
       * Exchange boundary data for CF_marker: send internal
         points to external points
       *------------------------------------------------*/

      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg, 
		CF_marker_offd, int_buf_data);
 
         hypre_ParCSRCommHandleDestroy(comm_handle);   
      }

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         {
            elmt = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            if (!int_buf_data[index] && CF_marker[elmt] > 0)
            {
               CF_marker[elmt] = 0; 
               index++;
            }
            else
            {
               int_buf_data[index++] = CF_marker[elmt];
            }
         }
      }
 
      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, 
        	CF_marker_offd);
 
         hypre_ParCSRCommHandleDestroy(comm_handle);   
      }
      }

      iter++;
     /*------------------------------------------------
      * Set C-pts and F-pts.
      *------------------------------------------------*/

     for (ig = 0; ig < graph_size; ig++) {
       i = graph_array[ig];

       /*---------------------------------------------
	* If the measure of i is smaller than 1, then
        * make i and F point (because it does not influence
        * any other point), and remove all edges of
	* equation i.
	*---------------------------------------------*/

       if(measure_array[i]<1.){
	 /* make point i an F point*/
	 CF_marker[i]= F_PT;

         /* remove the edges in equation i */
	 /* first the local part */
	 for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++) {
	   j = S_diag_j[jS];
	   if (j > -1){ /* column number is still positive; not accounted for yet */
	     S_diag_j[jS]  = -S_diag_j[jS]-1;
	   }
	 }
	 /* now the external part */
	 for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++) {
	   j = S_offd_j[jS];
	   if (j > -1){ /* column number is still positive; not accounted for yet */
	     S_offd_j[jS]  = -S_offd_j[jS]-1;
	   }
	 }
       }

       /*---------------------------------------------
	* First treat the case where point i is in the
	* independent set: make i a C point, 
        * take out all the graph edges for
        * equation i.
	*---------------------------------------------*/
       
       if (CF_marker[i] > 0) {
	 /* set to be a C-pt */
	 CF_marker[i] = C_PT;

         /* remove the edges in equation i */
	 /* first the local part */
	 for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++) {
	   j = S_diag_j[jS];
	   if (j > -1){ /* column number is still positive; not accounted for yet */
	     S_diag_j[jS]  = -S_diag_j[jS]-1;
	   }
	 }
	 /* now the external part */
	 for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++) {
	   j = S_offd_j[jS];
	   if (j > -1){ /* column number is still positive; not accounted for yet */
	     S_offd_j[jS]  = -S_offd_j[jS]-1;
	   }
	 }
       }  

       /*---------------------------------------------
	* Now treat the case where point i is not in the
	* independent set: loop over
	* all the points j that influence equation i; if
	* j is a C point, then make i an F point.
	* If i is a new F point, then remove all the edges
        * from the graph for equation i.
	*---------------------------------------------*/

       else {

	 /* first the local part */
	 for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++) {
	   /* j is the column number, or the local number of the point influencing i */
	   j = S_diag_j[jS];
           if(j<0) j=-j-1;

	   if (CF_marker[j] > 0){ /* j is a C-point */
	     CF_marker[i] = F_PT;
	   }
	 }
	 /* now the external part */
	 for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++) {
	   j = S_offd_j[jS];
           if(j<0) j=-j-1;
	   if (CF_marker_offd[j] > 0){ /* j is a C-point */
	     CF_marker[i] = F_PT;
	   }
	 }

         /* remove all the edges for equation i if i is a new F point */
	 if (CF_marker[i] == F_PT){
	   /* first the local part */
	   for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++) {
	     j = S_diag_j[jS];
	     if (j > -1){ /* column number is still positive; not accounted for yet */
	       S_diag_j[jS]  = -S_diag_j[jS]-1;
	     }
	   }
	   /* now the external part */
	   for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++) {
	     j = S_offd_j[jS];
	     if (j > -1){ /* column number is still positive; not accounted for yet */
	       S_offd_j[jS]  = -S_offd_j[jS]-1;
	     }
	   }
	 }   
       } /* end else */
     } /* end first loop over graph */

     /* now communicate CF_marker to CF_marker_offd, to make
        sure that new external F points are known on this processor */

      /*------------------------------------------------
       * Exchange boundary data for CF_marker: send internal
         points to external points
       *------------------------------------------------*/

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                int_buf_data[index++] 
                 = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
 
      if (num_procs > 1)
      {
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, 
        CF_marker_offd);
 
      hypre_ParCSRCommHandleDestroy(comm_handle);   
      }

     /*---------------------------------------------
      * Now loop over the points i in the unassigned
      * graph again. For all points i that are no new C or
      * F points, remove the edges in equation i that
      * connect to C or F points.
      * (We have removed the rows for the new C and F
      * points above; now remove the columns.)
      *---------------------------------------------*/

     for (ig = 0; ig < graph_size; ig++) {
       i = graph_array[ig];

       if(CF_marker[i]==0) {

	 /* first the local part */
	 for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++) {
	   j = S_diag_j[jS];
           if(j<0) j=-j-1;

	   if (!CF_marker[j]==0 && S_diag_j[jS] > -1){ /* connection to C or F point, and
                                                 column number is still positive; not accounted for yet */
	     S_diag_j[jS]  = -S_diag_j[jS]-1;
	   }
	 }
	 /* now the external part */
	 for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++) {
	   j = S_offd_j[jS];
           if(j<0) j=-j-1;

	   if (!CF_marker_offd[j]==0 && S_offd_j[jS] > -1){ /* connection to C or F point, and
                                                 column number is still positive; not accounted for yet */
	     S_offd_j[jS]  = -S_offd_j[jS]-1;
	   }
	 }
       }
     } /* end second loop over graph */

     /*------------------------------------------------
      * Update subgraph
      *------------------------------------------------*/

     for (ig = 0; ig < graph_size; ig++) {
       i = graph_array[ig];
       
       if (!CF_marker[i]==0) /* C or F point */
	 {
	   /* the independent set subroutine needs measure 0 for
              removed nodes */
	   measure_array[i] = 0;
	   /* take point out of the subgraph */
	   graph_size--;
	   graph_array[ig] = graph_array[graph_size];
	   graph_array[graph_size] = i;
	   ig--;
	 }
     }
     for (ig = 0; ig < graph_offd_size; ig++) {
       i = graph_array_offd[ig];
       
       if (!CF_marker_offd[i]==0) /* C or F point */
	 {
	   /* the independent set subroutine needs measure 0 for
              removed nodes */
	   measure_array[i+num_variables] = 0;
	   /* take point out of the subgraph */
	   graph_offd_size--;
	   graph_array_offd[ig] = graph_array_offd[graph_offd_size];
	   graph_array_offd[graph_offd_size] = i;
	   ig--;
	 }
     }
     
   } /* end while */

   /*   printf("*** MIS iteration %d\n",iter);
   printf("graph_size remaining %d\n",graph_size);

   printf("num_cols_offd %d\n",num_cols_offd);
   for (i=0;i<num_variables;i++)
     {
              if(CF_marker[i]==1)
       printf("node %d CF %d\n",i,CF_marker[i]);
       }*/


   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   /* Reset S_matrix */
   for (i=0; i < S_diag_i[num_variables]; i++)
   {
      if (S_diag_j[i] < 0)
         S_diag_j[i] = -S_diag_j[i]-1;
   }
   for (i=0; i < S_offd_i[num_variables]; i++)
   {
      if (S_offd_j[i] < 0)
         S_offd_j[i] = -S_offd_j[i]-1;
   }
   /*for (i=0; i < num_variables; i++)
      if (CF_marker[i] == SF_PT) CF_marker[i] = F_PT;*/

   hypre_TFree(measure_array);

   hypre_TFree(measure_array);
   hypre_TFree(graph_array);
   if (num_cols_offd) hypre_TFree(graph_array_offd);
   hypre_TFree(buf_data);
   hypre_TFree(int_buf_data);
   hypre_TFree(CF_marker_offd);
   /*if (num_procs > 1) hypre_CSRMatrixDestroy(S_ext);*/

   *CF_marker_ptr   = CF_marker;

   return (ierr);
}
