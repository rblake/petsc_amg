#ifndef TEMPORARY_MULTIVECTOR_FUNCTION_PROTOTYPES
#define TEMPORARY_MULTIVECTOR_FUNCTION_PROTOTYPES

#include "interpreter.h"

typedef struct
{
  long	 numVectors;
  int*   mask;
  void** vector;
  int	 ownsVectors;
  int    ownsMask;
  
  mv_InterfaceInterpreter* interpreter;
  
} mv_TempMultiVector;

/*typedef struct mv_TempMultiVector* mv_TempMultiVectorPtr;  */
typedef  mv_TempMultiVector* mv_TempMultiVectorPtr;

/*******************************************************************/
/*
The above is a temporary implementation of the hypre_MultiVector
data type, just to get things going with LOBPCG eigensolver.

A more proper implementation would be to define hypre_MultiParVector,
hypre_MultiStructVector and hypre_MultiSStructVector by adding a new 
record

int numVectors;

in hypre_ParVector, hypre_StructVector and hypre_SStructVector,
and increasing the size of data numVectors times. Respective
modifications of most vector operations are straightforward
(it is strongly suggested that BLAS routines are used wherever
possible), efficient implementation of matrix-by-multivector 
multiplication may be more difficult.

With the above implementation of hypre vectors, the definition
of hypre_MultiVector becomes simply (cf. multivector.h)

typedef struct
{
  void*	multiVector;
  HYPRE_InterfaceInterpreter* interpreter;  
} hypre_MultiVector;

with pointers to abstract multivector functions added to the structure
HYPRE_InterfaceInterpreter (cf. HYPRE_interpreter.h; particular values
are assigned to these pointers by functions 
HYPRE_ParCSRSetupInterpreter, HYPRE_StructSetupInterpreter and
int HYPRE_SStructSetupInterpreter),
and the abstract multivector functions become simply interfaces
to the actual multivector functions of the form (cf. multivector.c):

void 
hypre_MultiVectorCopy( hypre_MultiVectorPtr src_, hypre_MultiVectorPtr dest_ ) {

  hypre_MultiVector* src = (hypre_MultiVector*)src_;
  hypre_MultiVector* dest = (hypre_MultiVector*)dest_;
  assert( src != NULL && dest != NULL );
  (src->interpreter->CopyMultiVector)( src->data, dest->data );
}


*/
/*********************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

void*
mv_TempMultiVectorCreateFromSampleVector( void*, int n, void* sample );

void*
mv_TempMultiVectorCreateCopy( void*, int copyValues );

void 
mv_TempMultiVectorDestroy( void* );

int
mv_TempMultiVectorWidth( void* v );

int
mv_TempMultiVectorHeight( void* v );

void
mv_TempMultiVectorSetMask( void* v, int* mask );

void 
mv_TempMultiVectorClear( void* );

void 
mv_TempMultiVectorSetRandom( void* v, int seed );

void 
mv_TempMultiVectorCopy( void* src, void* dest );

void 
mv_TempMultiVectorAxpy( double, void*, void* ); 

void 
mv_TempMultiVectorByMultiVector( void*, void*,
				    int gh, int h, int w, double* v );

void 
mv_TempMultiVectorByMultiVectorDiag( void* x, void* y,
					int* mask, int n, double* diag );

void 
mv_TempMultiVectorByMatrix( void*, 
			       int gh, int h, int w, double* v,
			       void* );

void 
mv_TempMultiVectorXapy( void* x, 
			   int gh, int h, int w, double* v,
			   void* y );

void mv_TempMultiVectorByDiagonal( void* x, 
				      int* mask, int n, double* diag,
				      void* y );

void 
mv_TempMultiVectorEval( void (*f)( void*, void*, void* ), void* par,
			   void* x, void* y );

#ifdef __cplusplus
}
#endif

#endif /* MULTIVECTOR_FUNCTION_PROTOTYPES */

