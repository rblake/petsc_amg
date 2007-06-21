
#include <HYPRE_config.h>

#include "HYPRE_parcsr_mv.h"

#ifndef hypre_PARCSR_MV_HEADER
#define hypre_PARCSR_MV_HEADER

#include "utilities.h"
#include "seq_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HYPRE_PAR_CSR_COMMUNICATION_HEADER
#define HYPRE_PAR_CSR_COMMUNICATION_HEADER

/*--------------------------------------------------------------------------
 * hypre_ParCSRCommPkg:
 *   Structure containing information for doing communications
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm               comm;

   int                    num_sends;
   int                   *send_procs;
   int			 *send_map_starts;
   int			 *send_map_elmts;

   int                    num_recvs;
   int                   *recv_procs;
   int                   *recv_vec_starts;

   /* remote communication information */
   MPI_Datatype          *send_mpi_types;
   MPI_Datatype          *recv_mpi_types;

} hypre_ParCSRCommPkg;

/*--------------------------------------------------------------------------
 * hypre_ParCSRCommHandle:
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_ParCSRCommPkg  *comm_pkg;
   void 	  *send_data;
   void 	  *recv_data;

   int             num_requests;
   MPI_Request    *requests;

} hypre_ParCSRCommHandle;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ParCSRCommPkg
 *--------------------------------------------------------------------------*/
 
#define hypre_ParCSRCommPkgComm(comm_pkg)          (comm_pkg -> comm)
                                               
#define hypre_ParCSRCommPkgNumSends(comm_pkg)      (comm_pkg -> num_sends)
#define hypre_ParCSRCommPkgSendProcs(comm_pkg)     (comm_pkg -> send_procs)
#define hypre_ParCSRCommPkgSendProc(comm_pkg, i)   (comm_pkg -> send_procs[i])
#define hypre_ParCSRCommPkgSendMapStarts(comm_pkg) (comm_pkg -> send_map_starts)
#define hypre_ParCSRCommPkgSendMapStart(comm_pkg,i)(comm_pkg -> send_map_starts[i])
#define hypre_ParCSRCommPkgSendMapElmts(comm_pkg)  (comm_pkg -> send_map_elmts)
#define hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i) (comm_pkg -> send_map_elmts[i])

#define hypre_ParCSRCommPkgNumRecvs(comm_pkg)      (comm_pkg -> num_recvs)
#define hypre_ParCSRCommPkgRecvProcs(comm_pkg)     (comm_pkg -> recv_procs)
#define hypre_ParCSRCommPkgRecvProc(comm_pkg, i)   (comm_pkg -> recv_procs[i])
#define hypre_ParCSRCommPkgRecvVecStarts(comm_pkg) (comm_pkg -> recv_vec_starts)
#define hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i)(comm_pkg -> recv_vec_starts[i])

#define hypre_ParCSRCommPkgSendMPITypes(comm_pkg)  (comm_pkg -> send_mpi_types)
#define hypre_ParCSRCommPkgSendMPIType(comm_pkg,i) (comm_pkg -> send_mpi_types[i])

#define hypre_ParCSRCommPkgRecvMPITypes(comm_pkg)  (comm_pkg -> recv_mpi_types)
#define hypre_ParCSRCommPkgRecvMPIType(comm_pkg,i) (comm_pkg -> recv_mpi_types[i])

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ParCSRCommHandle
 *--------------------------------------------------------------------------*/
 
#define hypre_ParCSRCommHandleCommPkg(comm_handle)     (comm_handle -> comm_pkg)
#define hypre_ParCSRCommHandleSendData(comm_handle)    (comm_handle -> send_data)
#define hypre_ParCSRCommHandleRecvData(comm_handle)    (comm_handle -> recv_data)
#define hypre_ParCSRCommHandleNumRequests(comm_handle) (comm_handle -> num_requests)
#define hypre_ParCSRCommHandleRequests(comm_handle)    (comm_handle -> requests)
#define hypre_ParCSRCommHandleRequest(comm_handle, i)  (comm_handle -> requests[i])

#endif /* HYPRE_PAR_CSR_COMMUNICATION_HEADER */
/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.34 $
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for Parallel Vector data structure
 *
 *****************************************************************************/

#ifndef hypre_PAR_VECTOR_HEADER
#define hypre_PAR_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * hypre_ParVector
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm	 comm;

   int      	 global_size;
   int      	 first_index;
   int           last_index;
   int      	*partitioning;
   hypre_Vector	*local_vector; 

   /* Does the Vector create/destroy `data'? */
   int      	 owns_data;
   int      	 owns_partitioning;

} hypre_ParVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_ParVectorComm(vector)  	        ((vector) -> comm)
#define hypre_ParVectorGlobalSize(vector)       ((vector) -> global_size)
#define hypre_ParVectorFirstIndex(vector)       ((vector) -> first_index)
#define hypre_ParVectorLastIndex(vector)        ((vector) -> last_index)
#define hypre_ParVectorPartitioning(vector)     ((vector) -> partitioning)
#define hypre_ParVectorLocalVector(vector)      ((vector) -> local_vector)
#define hypre_ParVectorOwnsData(vector)         ((vector) -> owns_data)
#define hypre_ParVectorOwnsPartitioning(vector) ((vector) -> owns_partitioning)
#define hypre_ParVectorNumVectors(vector)\
 (hypre_VectorNumVectors( hypre_ParVectorLocalVector(vector) ))




#endif
/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.34 $
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for Parallel CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_PAR_CSR_MATRIX_HEADER
#define hypre_PAR_CSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Parallel CSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm		comm;

   int     		global_num_rows;
   int     		global_num_cols;
   int			first_row_index;
   int			first_col_diag;
   /* need to know entire local range in case row_starts and col_starts 
      are null  (i.e., bgl) AHB 6/05*/
   int                  last_row_index;
   int                  last_col_diag;

   hypre_CSRMatrix	*diag;
   hypre_CSRMatrix	*offd;
   int			*col_map_offd; 
	/* maps columns of offd to global columns */
   int 			*row_starts; 
	/* array of length num_procs+1, row_starts[i] contains the 
	   global number of the first row on proc i,  
	   first_row_index = row_starts[my_id],
	   row_starts[num_procs] = global_num_rows */
   int 			*col_starts;
	/* array of length num_procs+1, col_starts[i] contains the 
	   global number of the first column of diag on proc i,  
	   first_col_diag = col_starts[my_id],
	   col_starts[num_procs] = global_num_cols */

   hypre_ParCSRCommPkg	*comm_pkg;
   hypre_ParCSRCommPkg	*comm_pkgT;
   
   /* Does the ParCSRMatrix create/destroy `diag', `offd', `col_map_offd'? */
   int      owns_data;
   /* Does the ParCSRMatrix create/destroy `row_starts', `col_starts'? */
   int      owns_row_starts;
   int      owns_col_starts;

   int      num_nonzeros;
   double   d_num_nonzeros;

   /* Buffers used by GetRow to hold row currently being accessed. AJC, 4/99 */
   int     *rowindices;
   double  *rowvalues;
   int      getrowactive;

} hypre_ParCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRMatrixComm(matrix)		  ((matrix) -> comm)
#define hypre_ParCSRMatrixGlobalNumRows(matrix)   ((matrix) -> global_num_rows)
#define hypre_ParCSRMatrixGlobalNumCols(matrix)   ((matrix) -> global_num_cols)
#define hypre_ParCSRMatrixFirstRowIndex(matrix)   ((matrix) -> first_row_index)
#define hypre_ParCSRMatrixFirstColDiag(matrix)    ((matrix) -> first_col_diag)
#define hypre_ParCSRMatrixLastRowIndex(matrix)    ((matrix) -> last_row_index)
#define hypre_ParCSRMatrixLastColDiag(matrix)     ((matrix) -> last_col_diag)
#define hypre_ParCSRMatrixDiag(matrix)  	  ((matrix) -> diag)
#define hypre_ParCSRMatrixOffd(matrix)  	  ((matrix) -> offd)
#define hypre_ParCSRMatrixColMapOffd(matrix)  	  ((matrix) -> col_map_offd)
#define hypre_ParCSRMatrixRowStarts(matrix)       ((matrix) -> row_starts)
#define hypre_ParCSRMatrixColStarts(matrix)       ((matrix) -> col_starts)
#define hypre_ParCSRMatrixCommPkg(matrix)	  ((matrix) -> comm_pkg)
#define hypre_ParCSRMatrixCommPkgT(matrix)	  ((matrix) -> comm_pkgT)
#define hypre_ParCSRMatrixOwnsData(matrix)        ((matrix) -> owns_data)
#define hypre_ParCSRMatrixOwnsRowStarts(matrix)   ((matrix) -> owns_row_starts)
#define hypre_ParCSRMatrixOwnsColStarts(matrix)   ((matrix) -> owns_col_starts)
#define hypre_ParCSRMatrixNumRows(matrix) \
hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(matrix))
#define hypre_ParCSRMatrixNumCols(matrix) \
hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(matrix))
#define hypre_ParCSRMatrixNumNonzeros(matrix)     ((matrix) -> num_nonzeros)
#define hypre_ParCSRMatrixDNumNonzeros(matrix)    ((matrix) -> d_num_nonzeros)
#define hypre_ParCSRMatrixRowindices(matrix)      ((matrix) -> rowindices)
#define hypre_ParCSRMatrixRowvalues(matrix)       ((matrix) -> rowvalues)
#define hypre_ParCSRMatrixGetrowactive(matrix)    ((matrix) -> getrowactive)



/*--------------------------------------------------------------------------
 * Parallel CSR Boolean Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;
   int                   global_num_rows;
   int                   global_num_cols;
   int                   first_row_index;
   int                   first_col_diag;
   int                   last_row_index;
   int                   last_col_diag;
   hypre_CSRBooleanMatrix *diag;
   hypre_CSRBooleanMatrix *offd;
   int	                *col_map_offd; 
   int 	                *row_starts; 
   int 	                *col_starts;
   hypre_ParCSRCommPkg  *comm_pkg;
   hypre_ParCSRCommPkg  *comm_pkgT;
   int                   owns_data;
   int                   owns_row_starts;
   int                   owns_col_starts;
   int                   num_nonzeros;
   int                  *rowindices;
   int                   getrowactive;

} hypre_ParCSRBooleanMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Boolean Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRBooleanMatrix_Get_Comm(matrix)          ((matrix)->comm)
#define hypre_ParCSRBooleanMatrix_Get_GlobalNRows(matrix)   ((matrix)->global_num_rows)
#define hypre_ParCSRBooleanMatrix_Get_GlobalNCols(matrix)   ((matrix)->global_num_cols)
#define hypre_ParCSRBooleanMatrix_Get_StartRow(matrix)      ((matrix)->first_row_index)
#define hypre_ParCSRBooleanMatrix_Get_FirstRowIndex(matrix) ((matrix)->first_row_index)
#define hypre_ParCSRBooleanMatrix_Get_FirstColDiag(matrix)  ((matrix)->first_col_diag)
#define hypre_ParCSRBooleanMatrix_Get_LastRowIndex(matrix)  ((matrix)->last_row_index)
#define hypre_ParCSRBooleanMatrix_Get_LastColDiag(matrix)   ((matrix)->last_col_diag)
#define hypre_ParCSRBooleanMatrix_Get_Diag(matrix)          ((matrix)->diag)
#define hypre_ParCSRBooleanMatrix_Get_Offd(matrix)          ((matrix)->offd)
#define hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix)    ((matrix)->col_map_offd)
#define hypre_ParCSRBooleanMatrix_Get_RowStarts(matrix)     ((matrix)->row_starts)
#define hypre_ParCSRBooleanMatrix_Get_ColStarts(matrix)     ((matrix)->col_starts)
#define hypre_ParCSRBooleanMatrix_Get_CommPkg(matrix)       ((matrix)->comm_pkg)
#define hypre_ParCSRBooleanMatrix_Get_CommPkgT(matrix)      ((matrix)->comm_pkgT)
#define hypre_ParCSRBooleanMatrix_Get_OwnsData(matrix)      ((matrix)->owns_data)
#define hypre_ParCSRBooleanMatrix_Get_OwnsRowStarts(matrix) ((matrix)->owns_row_starts)
#define hypre_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) ((matrix)->owns_col_starts)
#define hypre_ParCSRBooleanMatrix_Get_NRows(matrix)         ((matrix->diag->num_rows))
#define hypre_ParCSRBooleanMatrix_Get_NCols(matrix)         ((matrix->diag->num_cols))
#define hypre_ParCSRBooleanMatrix_Get_NNZ(matrix)           ((matrix)->num_nonzeros)
#define hypre_ParCSRBooleanMatrix_Get_Rowindices(matrix)    ((matrix)->rowindices)
#define hypre_ParCSRBooleanMatrix_Get_Getrowactive(matrix)  ((matrix)->getrowactive)

#endif
/*BHEADER**********************************************************************
 * (c) 2002   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.34 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Tree structure for keeping track of numbers (e.g. column numbers) -
 * when you get them one at a time, in no particular order, possibly very
 * sparse.  In a scalable manner you want to be able to store them and find
 * out whether a number has been stored.
 * All decimal numbers will fit in a tree with 10 branches (digits)
 * off each node.  We also have a terminal "digit" to indicate that the entire
 * number has been seen.  E.g., 1234 would be entered in a tree as:
 * (numbering the digits off a node as 0 1 2 3 4 5 6 7 8 9 TERM )
 *                          root
 *                           |
 *                   - - - - 4 - - - - - -
 *                           |
 *                     - - - 3 - - - - - - -
 *                           |
 *                       - - 2 - - - - - - - -
 *                           |
 *                         - 1 - - - - - - - - -
 *                           |
 *       - - - - - - - - - - T
 *
 *
 * This tree represents a number through its decimal expansion, but if needed
 * this code can be changed to a different base, e.g. binary.  The appropriate
 * base depends on how the numbers encountered are distributed.  Totally
 * random (independent, equally likely in a large range) calls for binary.
 * The more clustered, the larger the base should be in my judgement.
 *
 *****************************************************************************/

#ifndef hypre_NUMBERS_HEADER
#define hypre_NUMBERS_HEADER

typedef struct {
   void * digit[11];
/* ... should be   hypre_NumbersNode * digit[11]; */
} hypre_NumbersNode;


hypre_NumbersNode * hypre_NumbersNewNode(void);
void hypre_NumbersDeleteNode( hypre_NumbersNode * node );
int hypre_NumbersEnter( hypre_NumbersNode * node, const int n );
int hypre_NumbersNEntered( hypre_NumbersNode * node );
int hypre_NumbersQuery( hypre_NumbersNode * node, const int n );
int * hypre_NumbersArray( hypre_NumbersNode * node );


#endif

#ifndef hypre_NEW_COMMPKG
#define hypre_NEW_COMMPKG

typedef struct
{
   int                   length;
   int                   row_start;
   int                   row_end;
   int                   storage_length;
   int                   *proc_list;
   int		         *row_start_list;
   int                   *row_end_list;  
  int                    *sort_index;
} hypre_IJAssumedPart;

typedef struct
{
  int                   length;
  int                   storage_length; 
  int                   *id;
  int                   *vec_starts;
  int                   element_storage_length; 
  int                   *elements;
}  hypre_ProcListElements;   


int hypre_NewCommPkgCreate( hypre_ParCSRMatrix* );
int hypre_NewCommPkgDestroy( hypre_ParCSRMatrix* );


#endif /* hypre_NEW_COMMPKG */

/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.34 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header info for Parallel Chord Matrix data structures
 *
 *
 *****************************************************************************/
#include <HYPRE_config.h>



#ifndef hypre_PAR_CHORD_MATRIX_HEADER
#define hypre_PAR_CHORD_MATRIX_HEADER

#include "utilities.h"
#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * Parallel Chord Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm comm;

  /*  A structure: -------------------------------------------------------- */
  int num_inprocessors;
  int *inprocessor;

  /* receiving in idof from different (in)processors; ---------------------- */
  int *num_idofs_inprocessor; 
  int **idof_inprocessor; 


  /* symmetric information: ----------------------------------------------- */
  /* this can be replaces by CSR format: ---------------------------------- */
  int *num_inchords;
  int **inchord_idof;
  int **inchord_rdof;
  double **inchord_data;

  int num_idofs;
  int num_rdofs;

  int *firstindex_idof; /* not owned by my_id; ----------------------------- */
  int *firstindex_rdof; /* not owned by my_id; ----------------------------- */

  /* --------------------------- mirror information: ---------------------- */
  /* participation of rdof in different processors; ------------------------ */

  int num_toprocessors;
  int *toprocessor;

  /* rdofs to be sentto toprocessors; --------------------------------------
     ----------------------------------------------------------------------- */
  int *num_rdofs_toprocessor;
  int **rdof_toprocessor;


} hypre_ParChordMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParChordMatrixComm(matrix)		  ((matrix) -> comm)

/*  matrix structure: ----------------------------------------------------- */

#define hypre_ParChordMatrixNumInprocessors(matrix)  ((matrix) -> num_inprocessors)
#define hypre_ParChordMatrixInprocessor(matrix) ((matrix) -> inprocessor)
#define hypre_ParChordMatrixNumIdofsInprocessor(matrix) ((matrix) -> num_idofs_inprocessor)
#define hypre_ParChordMatrixIdofInprocessor(matrix) ((matrix) -> idof_inprocessor)


#define hypre_ParChordMatrixNumInchords(matrix) ((matrix) -> num_inchords)

#define hypre_ParChordMatrixInchordIdof(matrix) ((matrix) -> inchord_idof)
#define hypre_ParChordMatrixInchordRdof(matrix) ((matrix) -> inchord_rdof)
#define hypre_ParChordMatrixInchordData(matrix) ((matrix) -> inchord_data)
#define hypre_ParChordMatrixNumIdofs(matrix)    ((matrix) -> num_idofs)
#define hypre_ParChordMatrixNumRdofs(matrix)    ((matrix) -> num_rdofs)

#define hypre_ParChordMatrixFirstindexIdof(matrix) ((matrix) -> firstindex_idof)
#define hypre_ParChordMatrixFirstindexRdof(matrix) ((matrix) -> firstindex_rdof) 

/* participation of rdof in different processors; ---------- */


#define hypre_ParChordMatrixNumToprocessors(matrix) ((matrix) -> num_toprocessors)
#define hypre_ParChordMatrixToprocessor(matrix)  ((matrix) -> toprocessor)
#define hypre_ParChordMatrixNumRdofsToprocessor(matrix) ((matrix) -> num_rdofs_toprocessor)
#define hypre_ParChordMatrixRdofToprocessor(matrix) ((matrix) -> rdof_toprocessor)


#endif

/* communicationT.c */
void RowsWithColumn_original( int *rowmin , int *rowmax , int column , hypre_ParCSRMatrix *A );
void RowsWithColumn( int *rowmin , int *rowmax , int column , int num_rows_diag , int firstColDiag , int *colMapOffd , int *mat_i_diag , int *mat_j_diag , int *mat_i_offd , int *mat_j_offd );
void hypre_MatTCommPkgCreate_core( MPI_Comm comm , int *col_map_offd , int first_col_diag , int *col_starts , int num_rows_diag , int num_cols_diag , int num_cols_offd , int *row_starts , int firstColDiag , int *colMapOffd , int *mat_i_diag , int *mat_j_diag , int *mat_i_offd , int *mat_j_offd , int data , int *p_num_recvs , int **p_recv_procs , int **p_recv_vec_starts , int *p_num_sends , int **p_send_procs , int **p_send_map_starts , int **p_send_map_elmts );
int hypre_MatTCommPkgCreate( hypre_ParCSRMatrix *A );

/* driver_aat2.c */

/* driver_aat.c */

/* driver_ab.c */

/* driver_boolaat.c */

/* driver_boolmatmul.c */

/* driver.c */

/* driver_matmul.c */

/* driver_mat_multivec.c */

/* driver_matvec.c */

/* driver_multivec.c */

/* HYPRE_parcsr_matrix.c */
int HYPRE_ParCSRMatrixCreate( MPI_Comm comm , int global_num_rows , int global_num_cols , int *row_starts , int *col_starts , int num_cols_offd , int num_nonzeros_diag , int num_nonzeros_offd , HYPRE_ParCSRMatrix *matrix );
int HYPRE_ParCSRMatrixDestroy( HYPRE_ParCSRMatrix matrix );
int HYPRE_ParCSRMatrixInitialize( HYPRE_ParCSRMatrix matrix );
int HYPRE_ParCSRMatrixRead( MPI_Comm comm , const char *file_name , HYPRE_ParCSRMatrix *matrix );
int HYPRE_ParCSRMatrixPrint( HYPRE_ParCSRMatrix matrix , const char *file_name );
int HYPRE_ParCSRMatrixGetComm( HYPRE_ParCSRMatrix matrix , MPI_Comm *comm );
int HYPRE_ParCSRMatrixGetDims( HYPRE_ParCSRMatrix matrix , int *M , int *N );
int HYPRE_ParCSRMatrixGetRowPartitioning( HYPRE_ParCSRMatrix matrix , int **row_partitioning_ptr );
int HYPRE_ParCSRMatrixGetColPartitioning( HYPRE_ParCSRMatrix matrix , int **col_partitioning_ptr );
int HYPRE_ParCSRMatrixGetLocalRange( HYPRE_ParCSRMatrix matrix , int *row_start , int *row_end , int *col_start , int *col_end );
int HYPRE_ParCSRMatrixGetRow( HYPRE_ParCSRMatrix matrix , int row , int *size , int **col_ind , double **values );
int HYPRE_ParCSRMatrixRestoreRow( HYPRE_ParCSRMatrix matrix , int row , int *size , int **col_ind , double **values );
int HYPRE_CSRMatrixToParCSRMatrix( MPI_Comm comm , HYPRE_CSRMatrix A_CSR , int *row_partitioning , int *col_partitioning , HYPRE_ParCSRMatrix *matrix );
int HYPRE_ParCSRMatrixMatvec( double alpha , HYPRE_ParCSRMatrix A , HYPRE_ParVector x , double beta , HYPRE_ParVector y );
int HYPRE_ParCSRMatrixMatvecT( double alpha , HYPRE_ParCSRMatrix A , HYPRE_ParVector x , double beta , HYPRE_ParVector y );

/* HYPRE_parcsr_vector.c */
int HYPRE_ParVectorCreate( MPI_Comm comm , int global_size , int *partitioning , HYPRE_ParVector *vector );
int HYPRE_ParMultiVectorCreate( MPI_Comm comm , int global_size , int *partitioning , int number_vectors , HYPRE_ParVector *vector );
int HYPRE_ParVectorDestroy( HYPRE_ParVector vector );
int HYPRE_ParVectorInitialize( HYPRE_ParVector vector );
int HYPRE_ParVectorRead( MPI_Comm comm , const char *file_name , HYPRE_ParVector *vector );
int HYPRE_ParVectorPrint( HYPRE_ParVector vector , const char *file_name );
int HYPRE_ParVectorSetConstantValues( HYPRE_ParVector vector , double value );
int HYPRE_ParVectorSetRandomValues( HYPRE_ParVector vector , int seed );
int HYPRE_ParVectorCopy( HYPRE_ParVector x , HYPRE_ParVector y );
HYPRE_ParVector HYPRE_ParVectorCloneShallow( HYPRE_ParVector x );
int HYPRE_ParVectorScale( double value , HYPRE_ParVector x );
int HYPRE_ParVectorAxpy( double alpha , HYPRE_ParVector x , HYPRE_ParVector y );
int HYPRE_ParVectorInnerProd( HYPRE_ParVector x , HYPRE_ParVector y , double *prod );
int HYPRE_VectorToParVector( MPI_Comm comm , HYPRE_Vector b , int *partitioning , HYPRE_ParVector *vector );

/* new_commpkg.c */
int PrintCommpkg( hypre_ParCSRMatrix *A , const char *file_name );
int hypre_NewCommPkgCreate_core( MPI_Comm comm , int *col_map_off_d , int first_col_diag , int col_start , int col_end , int num_cols_off_d , int global_num_cols , int *p_num_recvs , int **p_recv_procs , int **p_recv_vec_starts , int *p_num_sends , int **p_send_procs , int **p_send_map_starts , int **p_send_map_elements );
int hypre_NewCommPkgCreate( hypre_ParCSRMatrix *parcsr_A );
int hypre_NewCommPkgDestroy( hypre_ParCSRMatrix *parcsr_A );
int hypre_LocateAssummedPartition( int row_start , int row_end , int global_num_rows , hypre_IJAssumedPart *part , int myid );
int hypre_RangeFillResponseIJDetermineRecvProcs( void *p_recv_contact_buf , int contact_size , int contact_proc , void *ro , MPI_Comm comm , void **p_send_response_buf , int *response_message_size );
int hypre_FillResponseIJDetermineSendProcs( void *p_recv_contact_buf , int contact_size , int contact_proc , void *ro , MPI_Comm comm , void **p_send_response_buf , int *response_message_size );
int hypre_GetAssumedPartitionProcFromRow( int row , int global_num_rows , int *proc_id );
int hypre_GetAssumedPartitionRowRange( int proc_id , int global_num_rows , int *row_start , int *row_end );

/* numbers.c */
hypre_NumbersNode *hypre_NumbersNewNode( void );
void hypre_NumbersDeleteNode( hypre_NumbersNode *node );
int hypre_NumbersEnter( hypre_NumbersNode *node , const int n );
int hypre_NumbersNEntered( hypre_NumbersNode *node );
int hypre_NumbersQuery( hypre_NumbersNode *node , const int n );
int *hypre_NumbersArray( hypre_NumbersNode *node );

/* parchord_to_parcsr.c */
void hypre_ParChordMatrix_RowStarts( hypre_ParChordMatrix *Ac , MPI_Comm comm , int **row_starts , int *global_num_cols );
int hypre_ParChordMatrixToParCSRMatrix( hypre_ParChordMatrix *Ac , MPI_Comm comm , hypre_ParCSRMatrix **pAp );
int hypre_ParCSRMatrixToParChordMatrix( hypre_ParCSRMatrix *Ap , MPI_Comm comm , hypre_ParChordMatrix **pAc );

/* par_csr_aat.c */
void hypre_ParAat_RowSizes( int **C_diag_i , int **C_offd_i , int *B_marker , int *A_diag_i , int *A_diag_j , int *A_offd_i , int *A_offd_j , int *A_col_map_offd , int *A_ext_i , int *A_ext_j , int *A_ext_row_map , int *C_diag_size , int *C_offd_size , int num_rows_diag_A , int num_cols_offd_A , int num_rows_A_ext , int first_col_diag_A , int first_row_index_A );
hypre_ParCSRMatrix *hypre_ParCSRAAt( hypre_ParCSRMatrix *A );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractAExt( hypre_ParCSRMatrix *A , int data , int **pA_ext_row_map );

/* par_csr_at.c */
int proc_of_col( int col , int num_cols_diag , int num_procs , int *col_starts );
int pushnew( int item , int *stack , int *stacklen );
hypre_ParCSRMatrix *hypre_ParCSRAt( hypre_ParCSRMatrix *A );

/* par_csr_bool_matop.c */
hypre_ParCSRBooleanMatrix *hypre_ParBooleanMatmul( hypre_ParCSRBooleanMatrix *A , hypre_ParCSRBooleanMatrix *B );
hypre_CSRBooleanMatrix *hypre_ParCSRBooleanMatrixExtractBExt( hypre_ParCSRBooleanMatrix *B , hypre_ParCSRBooleanMatrix *A );
hypre_CSRBooleanMatrix *hypre_ParCSRBooleanMatrixExtractAExt( hypre_ParCSRBooleanMatrix *A , int **pA_ext_row_map );
hypre_ParCSRBooleanMatrix *hypre_ParBooleanAAt( hypre_ParCSRBooleanMatrix *A );
int hypre_BooleanMatTCommPkgCreate( hypre_ParCSRBooleanMatrix *A );
int hypre_BooleanMatvecCommPkgCreate( hypre_ParCSRBooleanMatrix *A );

/* par_csr_bool_matrix.c */
hypre_CSRBooleanMatrix *hypre_CSRBooleanMatrixCreate( int num_rows , int num_cols , int num_nonzeros );
int hypre_CSRBooleanMatrixDestroy( hypre_CSRBooleanMatrix *matrix );
int hypre_CSRBooleanMatrixInitialize( hypre_CSRBooleanMatrix *matrix );
int hypre_CSRBooleanMatrixSetDataOwner( hypre_CSRBooleanMatrix *matrix , int owns_data );
hypre_CSRBooleanMatrix *hypre_CSRBooleanMatrixRead( const char *file_name );
int hypre_CSRBooleanMatrixPrint( hypre_CSRBooleanMatrix *matrix , const char *file_name );
hypre_ParCSRBooleanMatrix *hypre_ParCSRBooleanMatrixCreate( MPI_Comm comm , int global_num_rows , int global_num_cols , int *row_starts , int *col_starts , int num_cols_offd , int num_nonzeros_diag , int num_nonzeros_offd );
int hypre_ParCSRBooleanMatrixDestroy( hypre_ParCSRBooleanMatrix *matrix );
int hypre_ParCSRBooleanMatrixInitialize( hypre_ParCSRBooleanMatrix *matrix );
int hypre_ParCSRBooleanMatrixSetNNZ( hypre_ParCSRBooleanMatrix *matrix );
int hypre_ParCSRBooleanMatrixSetDataOwner( hypre_ParCSRBooleanMatrix *matrix , int owns_data );
int hypre_ParCSRBooleanMatrixSetRowStartsOwner( hypre_ParCSRBooleanMatrix *matrix , int owns_row_starts );
int hypre_ParCSRBooleanMatrixSetColStartsOwner( hypre_ParCSRBooleanMatrix *matrix , int owns_col_starts );
hypre_ParCSRBooleanMatrix *hypre_ParCSRBooleanMatrixRead( MPI_Comm comm , const char *file_name );
int hypre_ParCSRBooleanMatrixPrint( hypre_ParCSRBooleanMatrix *matrix , const char *file_name );
int hypre_ParCSRBooleanMatrixPrintIJ( hypre_ParCSRBooleanMatrix *matrix , const char *filename );
int hypre_ParCSRBooleanMatrixGetLocalRange( hypre_ParCSRBooleanMatrix *matrix , int *row_start , int *row_end , int *col_start , int *col_end );
int hypre_ParCSRBooleanMatrixGetRow( hypre_ParCSRBooleanMatrix *mat , int row , int *size , int **col_ind );
int hypre_ParCSRBooleanMatrixRestoreRow( hypre_ParCSRBooleanMatrix *matrix , int row , int *size , int **col_ind );
int hypre_BuildCSRBooleanMatrixMPIDataType( int num_nonzeros , int num_rows , int *a_i , int *a_j , MPI_Datatype *csr_matrix_datatype );
hypre_ParCSRBooleanMatrix *hypre_CSRBooleanMatrixToParCSRBooleanMatrix( MPI_Comm comm , hypre_CSRBooleanMatrix *A , int *row_starts , int *col_starts );
int BooleanGenerateDiagAndOffd( hypre_CSRBooleanMatrix *A , hypre_ParCSRBooleanMatrix *matrix , int first_col_diag , int last_col_diag );

/* par_csr_communication.c */
hypre_ParCSRCommHandle *hypre_ParCSRCommHandleCreate( int job , hypre_ParCSRCommPkg *comm_pkg , void *send_data , void *recv_data );
int hypre_ParCSRCommHandleDestroy( hypre_ParCSRCommHandle *comm_handle );
void hypre_MatvecCommPkgCreate_core( MPI_Comm comm , int *col_map_offd , int first_col_diag , int *col_starts , int num_cols_diag , int num_cols_offd , int firstColDiag , int *colMapOffd , int data , int *p_num_recvs , int **p_recv_procs , int **p_recv_vec_starts , int *p_num_sends , int **p_send_procs , int **p_send_map_starts , int **p_send_map_elmts );
int hypre_MatvecCommPkgCreate( hypre_ParCSRMatrix *A );
int hypre_MatvecCommPkgDestroy( hypre_ParCSRCommPkg *comm_pkg );
int hypre_BuildCSRMatrixMPIDataType( int num_nonzeros , int num_rows , double *a_data , int *a_i , int *a_j , MPI_Datatype *csr_matrix_datatype );
int hypre_BuildCSRJDataType( int num_nonzeros , double *a_data , int *a_j , MPI_Datatype *csr_jdata_datatype );

/* par_csr_matop.c */
void hypre_ParMatmul_RowSizes( int **C_diag_i , int **C_offd_i , int **B_marker , int *A_diag_i , int *A_diag_j , int *A_offd_i , int *A_offd_j , int *B_diag_i , int *B_diag_j , int *B_offd_i , int *B_offd_j , int *B_ext_diag_i , int *B_ext_diag_j , int *B_ext_offd_i , int *B_ext_offd_j , int *map_B_to_C , int *C_diag_size , int *C_offd_size , int num_rows_diag_A , int num_cols_offd_A , int allsquare , int num_cols_diag_B , int num_cols_offd_B , int num_cols_offd_C );
hypre_ParCSRMatrix *hypre_ParMatmul( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *B );
void hypre_ParCSRMatrixExtractBExt_Arrays( int **pB_ext_i , int **pB_ext_j , double **pB_ext_data , int **pB_ext_row_map , int *num_nonzeros , int data , int find_row_map , MPI_Comm comm , hypre_ParCSRCommPkg *comm_pkg , int num_cols_B , int num_recvs , int num_sends , int first_col_diag , int first_row_index , int *recv_vec_starts , int *send_map_starts , int *send_map_elmts , int *diag_i , int *diag_j , int *offd_i , int *offd_j , int *col_map_offd , double *diag_data , double *offd_data );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractBExt( hypre_ParCSRMatrix *B , hypre_ParCSRMatrix *A , int data );
int hypre_ParCSRMatrixTranspose( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix **AT_ptr , int data );
void hypre_ParCSRMatrixGenSpanningTree( hypre_ParCSRMatrix *G_csr , int **indices , int G_type );
void hypre_ParCSRMatrixExtractSubmatrices( hypre_ParCSRMatrix *A_csr , int *indices2 , hypre_ParCSRMatrix ***submatrices );

/* par_csr_matrix.c */
hypre_ParCSRMatrix *hypre_ParCSRMatrixCreate( MPI_Comm comm , int global_num_rows , int global_num_cols , int *row_starts , int *col_starts , int num_cols_offd , int num_nonzeros_diag , int num_nonzeros_offd );
int hypre_ParCSRMatrixDestroy( hypre_ParCSRMatrix *matrix );
int hypre_ParCSRMatrixInitialize( hypre_ParCSRMatrix *matrix );
int hypre_ParCSRMatrixSetNumNonzeros( hypre_ParCSRMatrix *matrix );
int hypre_ParCSRMatrixSetDNumNonzeros( hypre_ParCSRMatrix *matrix );
int hypre_ParCSRMatrixSetDataOwner( hypre_ParCSRMatrix *matrix , int owns_data );
int hypre_ParCSRMatrixSetRowStartsOwner( hypre_ParCSRMatrix *matrix , int owns_row_starts );
int hypre_ParCSRMatrixSetColStartsOwner( hypre_ParCSRMatrix *matrix , int owns_col_starts );
hypre_ParCSRMatrix *hypre_ParCSRMatrixRead( MPI_Comm comm , const char *file_name );
int hypre_ParCSRMatrixPrint( hypre_ParCSRMatrix *matrix , const char *file_name );
int hypre_ParCSRMatrixPrintIJ( hypre_ParCSRMatrix *matrix , int base_i , int base_j , const char *filename );
int hypre_ParCSRMatrixReadIJ( MPI_Comm comm , const char *filename , int *base_i_ptr , int *base_j_ptr , hypre_ParCSRMatrix **matrix_ptr );
int hypre_ParCSRMatrixGetLocalRange( hypre_ParCSRMatrix *matrix , int *row_start , int *row_end , int *col_start , int *col_end );
int hypre_ParCSRMatrixGetRow( hypre_ParCSRMatrix *mat , int row , int *size , int **col_ind , double **values );
int hypre_ParCSRMatrixRestoreRow( hypre_ParCSRMatrix *matrix , int row , int *size , int **col_ind , double **values );
hypre_ParCSRMatrix *hypre_CSRMatrixToParCSRMatrix( MPI_Comm comm , hypre_CSRMatrix *A , int *row_starts , int *col_starts );
int GenerateDiagAndOffd( hypre_CSRMatrix *A , hypre_ParCSRMatrix *matrix , int first_col_diag , int last_col_diag );
hypre_CSRMatrix *hypre_MergeDiagAndOffd( hypre_ParCSRMatrix *par_matrix );
hypre_CSRMatrix *hypre_ParCSRMatrixToCSRMatrixAll( hypre_ParCSRMatrix *par_matrix );
int hypre_ParCSRMatrixCopy( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *B , int copy_data );
int hypre_FillResponseParToCSRMatrix( void *p_recv_contact_buf , int contact_size , int contact_proc , void *ro , MPI_Comm comm , void **p_send_response_buf , int *response_message_size );

/* par_csr_matvec.c */
int hypre_ParCSRMatrixMatvec( double alpha , hypre_ParCSRMatrix *A , hypre_ParVector *x , double beta , hypre_ParVector *y );
int hypre_ParCSRMatrixMatvecT( double alpha , hypre_ParCSRMatrix *A , hypre_ParVector *x , double beta , hypre_ParVector *y );

/* par_vector.c */
hypre_ParVector *hypre_ParVectorCreate( MPI_Comm comm , int global_size , int *partitioning );
hypre_ParVector *hypre_ParMultiVectorCreate( MPI_Comm comm , int global_size , int *partitioning , int num_vectors );
int hypre_ParVectorDestroy( hypre_ParVector *vector );
int hypre_ParVectorInitialize( hypre_ParVector *vector );
int hypre_ParVectorSetDataOwner( hypre_ParVector *vector , int owns_data );
int hypre_ParVectorSetPartitioningOwner( hypre_ParVector *vector , int owns_partitioning );
int hypre_ParVectorSetNumVectors( hypre_ParVector *vector , int num_vectors );
hypre_ParVector *hypre_ParVectorRead( MPI_Comm comm , const char *file_name );
int hypre_ParVectorPrint( hypre_ParVector *vector , const char *file_name );
int hypre_ParVectorSetConstantValues( hypre_ParVector *v , double value );
int hypre_ParVectorSetRandomValues( hypre_ParVector *v , int seed );
int hypre_ParVectorCopy( hypre_ParVector *x , hypre_ParVector *y );
hypre_ParVector *hypre_ParVectorCloneShallow( hypre_ParVector *x );
int hypre_ParVectorScale( double alpha , hypre_ParVector *y );
int hypre_ParVectorAxpy( double alpha , hypre_ParVector *x , hypre_ParVector *y );
double hypre_ParVectorInnerProd( hypre_ParVector *x , hypre_ParVector *y );
hypre_ParVector *hypre_VectorToParVector( MPI_Comm comm , hypre_Vector *v , int *vec_starts );
hypre_Vector *hypre_ParVectorToVectorAll( hypre_ParVector *par_v );
int hypre_ParVectorPrintIJ( hypre_ParVector *vector , int base_j , const char *filename );
int hypre_ParVectorReadIJ( MPI_Comm comm , const char *filename , int *base_j_ptr , hypre_ParVector **vector_ptr );
int hypre_FillResponseParToVectorAll( void *p_recv_contact_buf , int contact_size , int contact_proc , void *ro , MPI_Comm comm , void **p_send_response_buf , int *response_message_size );


#ifdef __cplusplus
}
#endif

#endif

