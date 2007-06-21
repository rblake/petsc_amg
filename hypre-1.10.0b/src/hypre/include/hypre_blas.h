/* hypre_blas.h  --  Contains BLAS prototypes needed by Hypre */

#ifndef HYPRE_BLAS_H
#define HYPRE_BLAS_H
#include "f2c.h"

/* --------------------------------------------------------------------------
 *   Change all names to hypre_ to avoid link conflicts
 * --------------------------------------------------------------------------*/

#define dasum   hypre_dasum
#define daxpy   hypre_daxpy
#define dcopy   hypre_dcopy
#define ddot    hypre_ddot
#define dgemm   hypre_dgemm
#define dgemv   hypre_dgemv
#define dger    hypre_dger
#define dnrm2   hypre_dnrm2
#define drot    hypre_drot
#define dscal   hypre_dscal
#define dswap   hypre_dswap
#define dsymm   hypre_dsymm
#define dsymv   hypre_dsymv
#define dsyr2   hypre_dsyr2
#define dsyr2k  hypre_dsyr2k
#define dsyrk   hypre_dsyrk
#define dtrmm   hypre_dtrmm
#define dtrmv   hypre_dtrmv
#define dtrsm   hypre_dtrsm
#define dtrsv   hypre_dtrsv
#define idamax  hypre_idamax

/* blas_utils.c */
logical hypre_lsame_ ( char *ca , char *cb );
int hypre_xerbla_ ( char *srname , integer *info );
integer s_cmp ( char *a0 , char *b0 , ftnlen la , ftnlen lb );
VOID s_copy ( char *a , char *b , ftnlen la , ftnlen lb );

/* dasum.c */
doublereal hypre_dasum_ ( integer *n , doublereal *dx , integer *incx );

/* daxpy.c */
int hypre_daxpy_ ( integer *n , doublereal *da , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* dcopy.c */
int hypre_dcopy_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* ddot.c */
doublereal hypre_ddot_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* dgemm.c */
int hypre_dgemm_ ( char *transa , char *transb , integer *m , integer *n , integer *k , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *beta , doublereal *c , integer *ldc );

/* dgemv.c */
int hypre_dgemv_ ( char *trans , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *x , integer *incx , doublereal *beta , doublereal *y , integer *incy );

/* dger.c */
int hypre_dger_ ( integer *m , integer *n , doublereal *alpha , doublereal *x , integer *incx , doublereal *y , integer *incy , doublereal *a , integer *lda );

/* dnrm2.c */
doublereal hypre_dnrm2_ ( integer *n , doublereal *dx , integer *incx );

/* drot.c */
int hypre_drot_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy , doublereal *c , doublereal *s );

/* dscal.c */
int hypre_dscal_ ( integer *n , doublereal *da , doublereal *dx , integer *incx );

/* dswap.c */
int hypre_dswap_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* dsymm.c */
int hypre_dsymm_ ( char *side , char *uplo , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *beta , doublereal *c__ , integer *ldc );

/* dsymv.c */
int hypre_dsymv_ ( char *uplo , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *x , integer *incx , doublereal *beta , doublereal *y , integer *incy );

/* dsyr2.c */
int hypre_dsyr2_ ( char *uplo , integer *n , doublereal *alpha , doublereal *x , integer *incx , doublereal *y , integer *incy , doublereal *a , integer *lda );

/* dsyr2k.c */
int hypre_dsyr2k_ ( char *uplo , char *trans , integer *n , integer *k , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *beta , doublereal *c__ , integer *ldc );

/* dsyrk.c */
int hypre_dsyrk_ ( char *uplo , char *trans , integer *n , integer *k , doublereal *alpha , doublereal *a , integer *lda , doublereal *beta , doublereal *c , integer *ldc );

/* dtrmm.c */
int hypre_dtrmm_ ( char *side , char *uplo , char *transa , char *diag , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb );

/* dtrmv.c */
int hypre_dtrmv_ ( char *uplo , char *trans , char *diag , integer *n , doublereal *a , integer *lda , doublereal *x , integer *incx );

/* dtrsm.c */
int hypre_dtrsm_ ( char *side , char *uplo , char *transa , char *diag , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb );

/* dtrsv.c */
int hypre_dtrsv_ ( char *uplo , char *trans , char *diag , integer *n , doublereal *a , integer *lda , doublereal *x , integer *incx );

/* idamax.c */
integer hypre_idamax_ ( integer *n , doublereal *dx , integer *incx );

#endif
