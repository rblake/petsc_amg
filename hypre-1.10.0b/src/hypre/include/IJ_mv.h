
#include <HYPRE_config.h>

#ifndef hypre_IJ_HEADER
#define hypre_IJ_HEADER

#include "utilities.h"
#include "seq_mv.h"
#include "parcsr_mv.h"
#include "HYPRE_IJ_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.8 $
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for Auxiliary Parallel CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_AUX_PARCSR_MATRIX_HEADER
#define hypre_AUX_PARCSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary Parallel CSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      local_num_rows;   /* defines number of rows on this processors */
   int      local_num_cols;   /* defines number of cols of diag */

   int      need_aux; /* if need_aux = 1, aux_j, aux_data are used to
			generate the parcsr matrix (default),
			for need_aux = 0, data is put directly into
			parcsr structure (requires the knowledge of
			offd_i and diag_i ) */

   int     *row_length; /* row_length_diag[i] contains number of stored
				elements in i-th row */
   int     *row_space; /* row_space_diag[i] contains space allocated to
				i-th row */
   int    **aux_j;	/* contains collected column indices */
   double **aux_data; /* contains collected data */

   int     *indx_diag; /* indx_diag[i] points to first empty space of portion
			 in diag_j , diag_data assigned to row i */  
   int     *indx_offd; /* indx_offd[i] points to first empty space of portion
			 in offd_j , offd_data assigned to row i */  
   int	    max_off_proc_elmts; /* length of off processor stash set for
					SetValues and AddTOValues */
   int	    current_num_elmts; /* current no. of elements stored in stash */
   int	    off_proc_i_indx; /* pointer to first empty space in 
				set_off_proc_i_set */
   int     *off_proc_i; /* length 2*num_off_procs_elmts, contains info pairs
			(code, no. of elmts) where code contains global
			row no. if  SetValues, and (-global row no. -1)
			if  AddToValues*/
   int     *off_proc_j; /* contains column indices */
   double  *off_proc_data; /* contains corresponding data */
} hypre_AuxParCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_AuxParCSRMatrixLocalNumRows(matrix)  ((matrix) -> local_num_rows)
#define hypre_AuxParCSRMatrixLocalNumCols(matrix)  ((matrix) -> local_num_cols)

#define hypre_AuxParCSRMatrixNeedAux(matrix)   ((matrix) -> need_aux)
#define hypre_AuxParCSRMatrixRowLength(matrix) ((matrix) -> row_length)
#define hypre_AuxParCSRMatrixRowSpace(matrix)  ((matrix) -> row_space)
#define hypre_AuxParCSRMatrixAuxJ(matrix)      ((matrix) -> aux_j)
#define hypre_AuxParCSRMatrixAuxData(matrix)   ((matrix) -> aux_data)

#define hypre_AuxParCSRMatrixIndxDiag(matrix)  ((matrix) -> indx_diag)
#define hypre_AuxParCSRMatrixIndxOffd(matrix)  ((matrix) -> indx_offd)

#define hypre_AuxParCSRMatrixMaxOffProcElmts(matrix)  ((matrix) -> max_off_proc_elmts)
#define hypre_AuxParCSRMatrixCurrentNumElmts(matrix)  ((matrix) -> current_num_elmts)
#define hypre_AuxParCSRMatrixOffProcIIndx(matrix)  ((matrix) -> off_proc_i_indx)
#define hypre_AuxParCSRMatrixOffProcI(matrix)  ((matrix) -> off_proc_i)
#define hypre_AuxParCSRMatrixOffProcJ(matrix)  ((matrix) -> off_proc_j)
#define hypre_AuxParCSRMatrixOffProcData(matrix)  ((matrix) -> off_proc_data)

#endif
/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.8 $
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for Auxiliary Parallel Vector data structures
 *
 * Note: this vector currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_AUX_PAR_VECTOR_HEADER
#define hypre_AUX_PAR_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary Parallel Vector
 *--------------------------------------------------------------------------*/

typedef struct
{
   int	    max_off_proc_elmts; /* length of off processor stash for
					SetValues and AddToValues*/
   int	    current_num_elmts; /* current no. of elements stored in stash */
   int     *off_proc_i; /* contains column indices */
   double  *off_proc_data; /* contains corresponding data */
} hypre_AuxParVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_AuxParVectorMaxOffProcElmts(matrix)  ((matrix) -> max_off_proc_elmts)
#define hypre_AuxParVectorCurrentNumElmts(matrix)  ((matrix) -> current_num_elmts)
#define hypre_AuxParVectorOffProcI(matrix)  ((matrix) -> off_proc_i)
#define hypre_AuxParVectorOffProcData(matrix)  ((matrix) -> off_proc_data)

#endif
/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.8 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header info for the hypre_IJMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_IJ_MATRIX_HEADER
#define hypre_IJ_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * hypre_IJMatrix:
 *--------------------------------------------------------------------------*/

typedef struct hypre_IJMatrix_struct
{
   MPI_Comm    comm;

   int        *row_partitioning;    /* distribution of rows across processors */
   int        *col_partitioning;    /* distribution of columns */

   int         object_type;         /* Indicates the type of "object" */
   void       *object;              /* Structure for storing local portion */
   void       *translator;          /* optional storage_type specfic structure
                                       for holding additional local info */
   int         assemble_flag;       /* indicates whether matrix has been 
				       assembled */

   int         global_first_row;    /* these for data items are necessary */
   int         global_first_col;    /*   to be able to avoind using the global */
   int         global_num_rows;     /*   global partition */ 
   int         global_num_cols;


} hypre_IJMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_IJMatrix
 *--------------------------------------------------------------------------*/

#define hypre_IJMatrixComm(matrix)              ((matrix) -> comm)

#define hypre_IJMatrixRowPartitioning(matrix)   ((matrix) -> row_partitioning)
#define hypre_IJMatrixColPartitioning(matrix)   ((matrix) -> col_partitioning)

#define hypre_IJMatrixObjectType(matrix)        ((matrix) -> object_type)
#define hypre_IJMatrixObject(matrix)            ((matrix) -> object)
#define hypre_IJMatrixTranslator(matrix)        ((matrix) -> translator)

#define hypre_IJMatrixAssembleFlag(matrix)      ((matrix) -> assemble_flag)


#define hypre_IJMatrixGlobalFirstRow(matrix)      ((matrix) -> global_first_row)
#define hypre_IJMatrixGlobalFirstCol(matrix)      ((matrix) -> global_first_col)
#define hypre_IJMatrixGlobalNumRows(matrix)       ((matrix) -> global_num_rows)
#define hypre_IJMatrixGlobalNumCols(matrix)       ((matrix) -> global_num_cols)

/*--------------------------------------------------------------------------
 * prototypes for operations on local objects
 *--------------------------------------------------------------------------*/

#ifdef PETSC_AVAILABLE
/* IJMatrix_petsc.c */
int
hypre_GetIJMatrixParCSRMatrix( HYPRE_IJMatrix IJmatrix, Mat *reference )
#endif
  
#ifdef ISIS_AVAILABLE
/* IJMatrix_isis.c */
int
hypre_GetIJMatrixISISMatrix( HYPRE_IJMatrix IJmatrix, RowMatrix *reference )
#endif

#endif
/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.8 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header info for the hypre_IJMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_IJ_VECTOR_HEADER
#define hypre_IJ_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * hypre_IJVector:
 *--------------------------------------------------------------------------*/

typedef struct hypre_IJVector_struct
{
   MPI_Comm      comm;

   int 		*partitioning;      /* Indicates partitioning over tasks */

   int           object_type;       /* Indicates the type of "local storage" */

   void         *object;            /* Structure for storing local portion */

   void         *translator;        /* Structure for storing off processor
				       information */

   int         global_first_row;    /* these for data items are necessary */
   int         global_num_rows;     /*    to be able to avoind using the global */
                                    /*    global partition */ 
   
   
} hypre_IJVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_IJVector
 *--------------------------------------------------------------------------*/

#define hypre_IJVectorComm(vector)           ((vector) -> comm)

#define hypre_IJVectorPartitioning(vector)   ((vector) -> partitioning)

#define hypre_IJVectorObjectType(vector)     ((vector) -> object_type)

#define hypre_IJVectorObject(vector)         ((vector) -> object)

#define hypre_IJVectorTranslator(vector)     ((vector) -> translator)

#define hypre_IJVectorGlobalFirstRow(vector)  ((vector) -> global_first_row)

#define hypre_IJVectorGlobalNumRows(vector)  ((vector) -> global_num_rows)

/*--------------------------------------------------------------------------
 * prototypes for operations on local objects
 *--------------------------------------------------------------------------*/
/* #include "./internal_protos.h" */

#endif

/* aux_parcsr_matrix.c */
int hypre_AuxParCSRMatrixCreate( hypre_AuxParCSRMatrix **aux_matrix , int local_num_rows , int local_num_cols , int *sizes );
int hypre_AuxParCSRMatrixDestroy( hypre_AuxParCSRMatrix *matrix );
int hypre_AuxParCSRMatrixInitialize( hypre_AuxParCSRMatrix *matrix );
int hypre_AuxParCSRMatrixSetMaxOffPRocElmts( hypre_AuxParCSRMatrix *matrix , int max_off_proc_elmts );

/* aux_par_vector.c */
int hypre_AuxParVectorCreate( hypre_AuxParVector **aux_vector );
int hypre_AuxParVectorDestroy( hypre_AuxParVector *vector );
int hypre_AuxParVectorInitialize( hypre_AuxParVector *vector );
int hypre_AuxParVectorSetMaxOffPRocElmts( hypre_AuxParVector *vector , int max_off_proc_elmts );


/* IJMatrix.c */
int hypre_IJMatrixGetRowPartitioning( HYPRE_IJMatrix matrix , int **row_partitioning );
int hypre_IJMatrixGetColPartitioning( HYPRE_IJMatrix matrix , int **col_partitioning );
int hypre_IJMatrixSetObject( HYPRE_IJMatrix matrix , void *object );

/* IJMatrix_isis.c */
int hypre_IJMatrixSetLocalSizeISIS( hypre_IJMatrix *matrix , int local_m , int local_n );
int hypre_IJMatrixCreateISIS( hypre_IJMatrix *matrix );
int hypre_IJMatrixSetRowSizesISIS( hypre_IJMatrix *matrix , int *sizes );
int hypre_IJMatrixSetDiagRowSizesISIS( hypre_IJMatrix *matrix , int *sizes );
int hypre_IJMatrixSetOffDiagRowSizesISIS( hypre_IJMatrix *matrix , int *sizes );
int hypre_IJMatrixInitializeISIS( hypre_IJMatrix *matrix );
int hypre_IJMatrixInsertBlockISIS( hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs );
int hypre_IJMatrixAddToBlockISIS( hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs );
int hypre_IJMatrixInsertRowISIS( hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs );
int hypre_IJMatrixAddToRowISIS( hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs );
int hypre_IJMatrixAssembleISIS( hypre_IJMatrix *matrix );
int hypre_IJMatrixDistributeISIS( hypre_IJMatrix *matrix , int *row_starts , int *col_starts );
int hypre_IJMatrixApplyISIS( hypre_IJMatrix *matrix , hypre_ParVector *x , hypre_ParVector *b );
int hypre_IJMatrixDestroyISIS( hypre_IJMatrix *matrix );
int hypre_IJMatrixSetTotalSizeISIS( hypre_IJMatrix *matrix , int size );

/* IJMatrix_parcsr.c */
int hypre_IJMatrixCreateParCSR( hypre_IJMatrix *matrix );
int hypre_IJMatrixSetRowSizesParCSR( hypre_IJMatrix *matrix , const int *sizes );
int hypre_IJMatrixSetDiagOffdSizesParCSR( hypre_IJMatrix *matrix , const int *diag_sizes , const int *offdiag_sizes );
int hypre_IJMatrixSetMaxOffProcElmtsParCSR( hypre_IJMatrix *matrix , int max_off_proc_elmts );
int hypre_IJMatrixInitializeParCSR( hypre_IJMatrix *matrix );
int hypre_IJMatrixGetRowCountsParCSR( hypre_IJMatrix *matrix , int nrows , int *rows , int *ncols );
int hypre_IJMatrixGetValuesParCSR( hypre_IJMatrix *matrix , int nrows , int *ncols , int *rows , int *cols , double *values );
int hypre_IJMatrixSetValuesParCSR( hypre_IJMatrix *matrix , int nrows , int *ncols , const int *rows , const int *cols , const double *values );
int hypre_IJMatrixAddToValuesParCSR( hypre_IJMatrix *matrix , int nrows , int *ncols , const int *rows , const int *cols , const double *values );
int hypre_IJMatrixAssembleParCSR( hypre_IJMatrix *matrix );
int hypre_IJMatrixDestroyParCSR( hypre_IJMatrix *matrix );
int hypre_IJMatrixAssembleOffProcValsParCSR( hypre_IJMatrix *matrix , int off_proc_i_indx , int max_off_proc_elmts , int current_num_elmts , int *off_proc_i , int *off_proc_j , double *off_proc_data );
int hypre_FindProc( int *list , int value , int list_length );

/* IJMatrix_petsc.c */
int hypre_IJMatrixSetLocalSizePETSc( hypre_IJMatrix *matrix , int local_m , int local_n );
int hypre_IJMatrixCreatePETSc( hypre_IJMatrix *matrix );
int hypre_IJMatrixSetRowSizesPETSc( hypre_IJMatrix *matrix , int *sizes );
int hypre_IJMatrixSetDiagRowSizesPETSc( hypre_IJMatrix *matrix , int *sizes );
int hypre_IJMatrixSetOffDiagRowSizesPETSc( hypre_IJMatrix *matrix , int *sizes );
int hypre_IJMatrixInitializePETSc( hypre_IJMatrix *matrix );
int hypre_IJMatrixInsertBlockPETSc( hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs );
int hypre_IJMatrixAddToBlockPETSc( hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs );
int hypre_IJMatrixInsertRowPETSc( hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs );
int hypre_IJMatrixAddToRowPETSc( hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs );
int hypre_IJMatrixAssemblePETSc( hypre_IJMatrix *matrix );
int hypre_IJMatrixDistributePETSc( hypre_IJMatrix *matrix , int *row_starts , int *col_starts );
int hypre_IJMatrixApplyPETSc( hypre_IJMatrix *matrix , hypre_ParVector *x , hypre_ParVector *b );
int hypre_IJMatrixDestroyPETSc( hypre_IJMatrix *matrix );
int hypre_IJMatrixSetTotalSizePETSc( hypre_IJMatrix *matrix , int size );

/* IJVector.c */
int hypre_IJVectorDistribute( HYPRE_IJVector vector , const int *vec_starts );
int hypre_IJVectorZeroValues( HYPRE_IJVector vector );

/* IJVector_parcsr.c */
int hypre_IJVectorCreatePar( hypre_IJVector *vector , int *IJpartitioning );
int hypre_IJVectorDestroyPar( hypre_IJVector *vector );
int hypre_IJVectorInitializePar( hypre_IJVector *vector );
int hypre_IJVectorSetMaxOffProcElmtsPar( hypre_IJVector *vector , int max_off_proc_elmts );
int hypre_IJVectorDistributePar( hypre_IJVector *vector , const int *vec_starts );
int hypre_IJVectorZeroValuesPar( hypre_IJVector *vector );
int hypre_IJVectorSetValuesPar( hypre_IJVector *vector , int num_values , const int *indices , const double *values );
int hypre_IJVectorAddToValuesPar( hypre_IJVector *vector , int num_values , const int *indices , const double *values );
int hypre_IJVectorAssemblePar( hypre_IJVector *vector );
int hypre_IJVectorGetValuesPar( hypre_IJVector *vector , int num_values , const int *indices , double *values );
int hypre_IJVectorAssembleOffProcValsPar( hypre_IJVector *vector , int max_off_proc_elmts , int current_num_elmts , int *off_proc_i , double *off_proc_data );


/* HYPRE_IJMatrix.c */
int HYPRE_IJMatrixCreate( MPI_Comm comm , int ilower , int iupper , int jlower , int jupper , HYPRE_IJMatrix *matrix );
int HYPRE_IJMatrixDestroy( HYPRE_IJMatrix matrix );
int HYPRE_IJMatrixInitialize( HYPRE_IJMatrix matrix );
int HYPRE_IJMatrixSetValues( HYPRE_IJMatrix matrix , int nrows , int *ncols , const int *rows , const int *cols , const double *values );
int HYPRE_IJMatrixAddToValues( HYPRE_IJMatrix matrix , int nrows , int *ncols , const int *rows , const int *cols , const double *values );
int HYPRE_IJMatrixAssemble( HYPRE_IJMatrix matrix );
int HYPRE_IJMatrixGetRowCounts( HYPRE_IJMatrix matrix , int nrows , int *rows , int *ncols );
int HYPRE_IJMatrixGetValues( HYPRE_IJMatrix matrix , int nrows , int *ncols , int *rows , int *cols , double *values );
int HYPRE_IJMatrixSetObjectType( HYPRE_IJMatrix matrix , int type );
int HYPRE_IJMatrixGetObjectType( HYPRE_IJMatrix matrix , int *type );
int HYPRE_IJMatrixGetLocalRange( HYPRE_IJMatrix matrix , int *ilower , int *iupper , int *jlower , int *jupper );
int HYPRE_IJMatrixGetObject( HYPRE_IJMatrix matrix , void **object );
int HYPRE_IJMatrixSetRowSizes( HYPRE_IJMatrix matrix , const int *sizes );
int HYPRE_IJMatrixSetDiagOffdSizes( HYPRE_IJMatrix matrix , const int *diag_sizes , const int *offdiag_sizes );
int HYPRE_IJMatrixSetMaxOffProcElmts( HYPRE_IJMatrix matrix , int max_off_proc_elmts );
int HYPRE_IJMatrixRead( const char *filename , MPI_Comm comm , int type , HYPRE_IJMatrix *matrix_ptr );
int HYPRE_IJMatrixPrint( HYPRE_IJMatrix matrix , const char *filename );

/* HYPRE_IJVector.c */
int HYPRE_IJVectorCreate( MPI_Comm comm , int jlower , int jupper , HYPRE_IJVector *vector );
int HYPRE_IJVectorDestroy( HYPRE_IJVector vector );
int HYPRE_IJVectorInitialize( HYPRE_IJVector vector );
int HYPRE_IJVectorSetValues( HYPRE_IJVector vector , int nvalues , const int *indices , const double *values );
int HYPRE_IJVectorAddToValues( HYPRE_IJVector vector , int nvalues , const int *indices , const double *values );
int HYPRE_IJVectorAssemble( HYPRE_IJVector vector );
int HYPRE_IJVectorGetValues( HYPRE_IJVector vector , int nvalues , const int *indices , double *values );
int HYPRE_IJVectorSetMaxOffProcElmts( HYPRE_IJVector vector , int max_off_proc_elmts );
int HYPRE_IJVectorSetObjectType( HYPRE_IJVector vector , int type );
int HYPRE_IJVectorGetObjectType( HYPRE_IJVector vector , int *type );
int HYPRE_IJVectorGetLocalRange( HYPRE_IJVector vector , int *jlower , int *jupper );
int HYPRE_IJVectorGetObject( HYPRE_IJVector vector , void **object );
int HYPRE_IJVectorRead( const char *filename , MPI_Comm comm , int type , HYPRE_IJVector *vector_ptr );
int HYPRE_IJVectorPrint( HYPRE_IJVector vector , const char *filename );


#ifdef __cplusplus
}
#endif

#endif

