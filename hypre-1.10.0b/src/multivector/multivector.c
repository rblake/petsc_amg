#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "multivector.h"

/* abstract multivector */
struct mv_MultiVector
{
  void*	data;      /* the pointer to the actual multivector */
  int	ownsData;

  mv_InterfaceInterpreter* interpreter; /* a structure that defines
					      multivector operations */
} ;

void *
mv_MultiVectorGetData (mv_MultiVectorPtr x)
{
  hypre_assert (x!=NULL);
  return x->data;
}

mv_MultiVectorPtr
mv_MultiVectorWrap( mv_InterfaceInterpreter* ii, void * data, int ownsData )
{
  mv_MultiVectorPtr x;
  
  x = (mv_MultiVectorPtr) malloc(sizeof(struct mv_MultiVector));
  hypre_assert( x != NULL );
  
  x->interpreter = ii;
  x->data = data;
  x->ownsData = ownsData;
  
  return x;
}

mv_MultiVectorPtr 
mv_MultiVectorCreateFromSampleVector( void* ii_, int n, void* sample ) { 

  mv_MultiVectorPtr x;
  mv_InterfaceInterpreter* ii = (mv_InterfaceInterpreter*)ii_;

  x = (mv_MultiVectorPtr) malloc(sizeof(struct mv_MultiVector));
  hypre_assert( x != NULL );
  
  x->interpreter = ii;
  x->data = (ii->CreateMultiVector)( ii, n, sample );
  x->ownsData = 1;

  return x;
}

mv_MultiVectorPtr 
mv_MultiVectorCreateCopy( mv_MultiVectorPtr x, int copyValues ) {

  mv_MultiVectorPtr y;
  void* data;
  mv_InterfaceInterpreter* ii;

  hypre_assert( x != NULL );
  ii = x->interpreter;

  y = (mv_MultiVectorPtr) malloc(sizeof(struct mv_MultiVector));
  hypre_assert( y != NULL );
  
  data = (ii->CopyCreateMultiVector)( x->data, copyValues );

  y->interpreter = ii;
  y->data = data;
  y->ownsData = 1;

  return y;
}

void 
mv_MultiVectorDestroy( mv_MultiVectorPtr v) {

  if ( v == NULL )
    return;

  if ( v->ownsData )
    (v->interpreter->DestroyMultiVector)( v->data );
  free( v );
}

void
mv_MultiVectorSetMask( mv_MultiVectorPtr v, int* mask ) {

  hypre_assert( v != NULL );
  (v->interpreter->SetMask)( v->data, mask );
}

int
mv_MultiVectorWidth( mv_MultiVectorPtr v ) {

  if ( v == NULL )
    return 0;

  return (v->interpreter->Width)( v->data );
}

int
mv_MultiVectorHeight( mv_MultiVectorPtr v ) {

  if ( v == NULL )
    return 0;
    	  
  return (v->interpreter->Height)(v->data);
}

void
mv_MultiVectorClear( mv_MultiVectorPtr v ) {

  hypre_assert( v != NULL );
  (v->interpreter->ClearMultiVector)( v->data );
}

void
mv_MultiVectorSetRandom( mv_MultiVectorPtr v, int seed ) {

  hypre_assert( v != NULL );
  (v->interpreter->SetRandomVectors)( v->data, seed );
}

void 
mv_MultiVectorCopy( mv_MultiVectorPtr src, mv_MultiVectorPtr dest ) {

  hypre_assert( src != NULL && dest != NULL );
  (src->interpreter->CopyMultiVector)( src->data, dest->data );
}

void 
mv_MultiVectorAxpy( double a, mv_MultiVectorPtr x, mv_MultiVectorPtr y ) { 
	
  hypre_assert( x != NULL && y != NULL );
  (x->interpreter->MultiAxpy)( a, x->data, y->data );
}

void 
mv_MultiVectorByMultiVector( mv_MultiVectorPtr x, mv_MultiVectorPtr y,
				     int xyGHeight, int xyHeight, 
				     int xyWidth, double* xy ) { 
/* xy = x'*y */	

  hypre_assert( x != NULL && y != NULL );
  (x->interpreter->MultiInnerProd)
    ( x->data, y->data, xyGHeight, xyHeight, xyWidth, xy );
}

void 
mv_MultiVectorByMultiVectorDiag( mv_MultiVectorPtr x, mv_MultiVectorPtr y,
					 int* mask, int n, double* d ) {
/* d = diag(x'*y) */	

  hypre_assert( x != NULL && y != NULL );
  (x->interpreter->MultiInnerProdDiag)( x->data, y->data, mask, n, d );
}

void 
mv_MultiVectorByMatrix( mv_MultiVectorPtr x, 
			   int rGHeight, int rHeight, 
			   int rWidth, double* rVal,
			   mv_MultiVectorPtr y ) {

  /* y = x*r */

  hypre_assert( x != NULL && y != NULL );
  (x->interpreter->MultiVecMat)
    ( x->data, rGHeight, rHeight, rWidth, rVal, y->data );
}

void 
mv_MultiVectorXapy( mv_MultiVectorPtr x, 
		       int rGHeight, int rHeight, 
		       int rWidth, double* rVal,
		       mv_MultiVectorPtr y ) {

  /* y = y + x*a */

  hypre_assert( x != NULL && y != NULL );
  (x->interpreter->MultiXapy)
    ( x->data, rGHeight, rHeight, rWidth, rVal, y->data );
}

void 
mv_MultiVectorByDiagonal( mv_MultiVectorPtr x, 
			     int* mask, int n, double* d,
			     mv_MultiVectorPtr y ) {

  /* y = x*d */

  hypre_assert( x != NULL && y != NULL );
  (x->interpreter->MultiVecMatDiag)( x->data, mask, n, d, y->data );
}

void 
mv_MultiVectorEval( void (*f)( void*, void*, void* ), void* par,
		       mv_MultiVectorPtr x, mv_MultiVectorPtr y ) {

  /* y = f(x) computed vector-wise */

  hypre_assert( x != NULL && y != NULL );
  (x->interpreter->Eval)( f, par, x->data, y->data );
}
							

