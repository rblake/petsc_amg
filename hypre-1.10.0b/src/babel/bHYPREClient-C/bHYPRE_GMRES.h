/*
 * File:          bHYPRE_GMRES.h
 * Symbol:        bHYPRE.GMRES-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.GMRES
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_GMRES_h
#define included_bHYPRE_GMRES_h

/**
 * Symbol "bHYPRE.GMRES" (version 1.0.0)
 * 
 * Objects of this type can be cast to PreconditionedSolver objects
 * using the {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 * The regular GMRES solver calls Babel-interface matrix and vector functions.
 * The HGMRES solver calls HYPRE interface functions.
 * The regular solver will work with any consistent matrix, vector, and
 * preconditioner classes.  The HGMRES solver will work with the more common
 * combinations.
 * 
 * The HGMRES solver checks whether the matrix, vectors, and preconditioner
 * are of known types, and will not work with any other types.
 * Presently, the recognized data types are:
 * matrix, vector: IJParCSRMatrix, IJParCSRVector
 * preconditioner: BoomerAMG, ParCSRDiagScale
 * 
 * 
 */
struct bHYPRE_GMRES__object;
struct bHYPRE_GMRES__array;
typedef struct bHYPRE_GMRES__object* bHYPRE_GMRES;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_bHYPRE_PreconditionedSolver_h
#include "bHYPRE_PreconditionedSolver.h"
#endif
#ifndef included_bHYPRE_Solver_h
#include "bHYPRE_Solver.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif

#ifndef included_sidl_io_Serializer_h
#include "sidl_io_Serializer.h"
#endif
#ifndef included_sidl_io_Deserializer_h
#include "sidl_io_Deserializer.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
struct bHYPRE_GMRES__object*
bHYPRE_GMRES__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_GMRES
bHYPRE_GMRES__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_GMRES
bHYPRE_GMRES__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_GMRES_addRef(
  /* in */ bHYPRE_GMRES self);

void
bHYPRE_GMRES_deleteRef(
  /* in */ bHYPRE_GMRES self);

sidl_bool
bHYPRE_GMRES_isSame(
  /* in */ bHYPRE_GMRES self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_GMRES_queryInt(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name);

sidl_bool
bHYPRE_GMRES_isType(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_GMRES_getClassInfo(
  /* in */ bHYPRE_GMRES self);

/**
 * Method:  Create[]
 */
bHYPRE_GMRES
bHYPRE_GMRES_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A);

/**
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */
int32_t
bHYPRE_GMRES_SetCommunicator(
  /* in */ bHYPRE_GMRES self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_GMRES_SetIntParameter(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name,
  /* in */ int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_GMRES_SetDoubleParameter(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name,
  /* in */ double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_GMRES_SetStringParameter(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name,
  /* in */ const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_GMRES_SetIntArray1Parameter(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_GMRES_SetIntArray2Parameter(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_GMRES_SetDoubleArray1Parameter(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_GMRES_SetDoubleArray2Parameter(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_GMRES_GetIntValue(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name,
  /* out */ int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_GMRES_GetDoubleValue(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name,
  /* out */ double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_GMRES_Setup(
  /* in */ bHYPRE_GMRES self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_GMRES_Apply(
  /* in */ bHYPRE_GMRES self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_GMRES_ApplyAdjoint(
  /* in */ bHYPRE_GMRES self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 * 
 */
int32_t
bHYPRE_GMRES_SetOperator(
  /* in */ bHYPRE_GMRES self,
  /* in */ bHYPRE_Operator A);

/**
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */
int32_t
bHYPRE_GMRES_SetTolerance(
  /* in */ bHYPRE_GMRES self,
  /* in */ double tolerance);

/**
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */
int32_t
bHYPRE_GMRES_SetMaxIterations(
  /* in */ bHYPRE_GMRES self,
  /* in */ int32_t max_iterations);

/**
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 * 
 */
int32_t
bHYPRE_GMRES_SetLogging(
  /* in */ bHYPRE_GMRES self,
  /* in */ int32_t level);

/**
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 * 
 */
int32_t
bHYPRE_GMRES_SetPrintLevel(
  /* in */ bHYPRE_GMRES self,
  /* in */ int32_t level);

/**
 * (Optional) Return the number of iterations taken.
 * 
 */
int32_t
bHYPRE_GMRES_GetNumIterations(
  /* in */ bHYPRE_GMRES self,
  /* out */ int32_t* num_iterations);

/**
 * (Optional) Return the norm of the relative residual.
 * 
 */
int32_t
bHYPRE_GMRES_GetRelResidualNorm(
  /* in */ bHYPRE_GMRES self,
  /* out */ double* norm);

/**
 * Set the preconditioner.
 * 
 */
int32_t
bHYPRE_GMRES_SetPreconditioner(
  /* in */ bHYPRE_GMRES self,
  /* in */ bHYPRE_Solver s);

/**
 * Method:  GetPreconditioner[]
 */
int32_t
bHYPRE_GMRES_GetPreconditioner(
  /* in */ bHYPRE_GMRES self,
  /* out */ bHYPRE_Solver* s);

/**
 * Method:  Clone[]
 */
int32_t
bHYPRE_GMRES_Clone(
  /* in */ bHYPRE_GMRES self,
  /* out */ bHYPRE_PreconditionedSolver* x);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_GMRES__object*
bHYPRE_GMRES__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_GMRES__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_GMRES__exec(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_GMRES__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_GMRES__getURL(
  /* in */ bHYPRE_GMRES self);
struct bHYPRE_GMRES__array*
bHYPRE_GMRES__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_GMRES__array*
bHYPRE_GMRES__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_GMRES__array*
bHYPRE_GMRES__array_create1d(int32_t len);

struct bHYPRE_GMRES__array*
bHYPRE_GMRES__array_create1dInit(
  int32_t len, 
  bHYPRE_GMRES* data);

struct bHYPRE_GMRES__array*
bHYPRE_GMRES__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_GMRES__array*
bHYPRE_GMRES__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_GMRES__array*
bHYPRE_GMRES__array_borrow(
  bHYPRE_GMRES* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_GMRES__array*
bHYPRE_GMRES__array_smartCopy(
  struct bHYPRE_GMRES__array *array);

void
bHYPRE_GMRES__array_addRef(
  struct bHYPRE_GMRES__array* array);

void
bHYPRE_GMRES__array_deleteRef(
  struct bHYPRE_GMRES__array* array);

bHYPRE_GMRES
bHYPRE_GMRES__array_get1(
  const struct bHYPRE_GMRES__array* array,
  const int32_t i1);

bHYPRE_GMRES
bHYPRE_GMRES__array_get2(
  const struct bHYPRE_GMRES__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_GMRES
bHYPRE_GMRES__array_get3(
  const struct bHYPRE_GMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_GMRES
bHYPRE_GMRES__array_get4(
  const struct bHYPRE_GMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_GMRES
bHYPRE_GMRES__array_get5(
  const struct bHYPRE_GMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_GMRES
bHYPRE_GMRES__array_get6(
  const struct bHYPRE_GMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_GMRES
bHYPRE_GMRES__array_get7(
  const struct bHYPRE_GMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_GMRES
bHYPRE_GMRES__array_get(
  const struct bHYPRE_GMRES__array* array,
  const int32_t indices[]);

void
bHYPRE_GMRES__array_set1(
  struct bHYPRE_GMRES__array* array,
  const int32_t i1,
  bHYPRE_GMRES const value);

void
bHYPRE_GMRES__array_set2(
  struct bHYPRE_GMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_GMRES const value);

void
bHYPRE_GMRES__array_set3(
  struct bHYPRE_GMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_GMRES const value);

void
bHYPRE_GMRES__array_set4(
  struct bHYPRE_GMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_GMRES const value);

void
bHYPRE_GMRES__array_set5(
  struct bHYPRE_GMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_GMRES const value);

void
bHYPRE_GMRES__array_set6(
  struct bHYPRE_GMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_GMRES const value);

void
bHYPRE_GMRES__array_set7(
  struct bHYPRE_GMRES__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_GMRES const value);

void
bHYPRE_GMRES__array_set(
  struct bHYPRE_GMRES__array* array,
  const int32_t indices[],
  bHYPRE_GMRES const value);

int32_t
bHYPRE_GMRES__array_dimen(
  const struct bHYPRE_GMRES__array* array);

int32_t
bHYPRE_GMRES__array_lower(
  const struct bHYPRE_GMRES__array* array,
  const int32_t ind);

int32_t
bHYPRE_GMRES__array_upper(
  const struct bHYPRE_GMRES__array* array,
  const int32_t ind);

int32_t
bHYPRE_GMRES__array_length(
  const struct bHYPRE_GMRES__array* array,
  const int32_t ind);

int32_t
bHYPRE_GMRES__array_stride(
  const struct bHYPRE_GMRES__array* array,
  const int32_t ind);

int
bHYPRE_GMRES__array_isColumnOrder(
  const struct bHYPRE_GMRES__array* array);

int
bHYPRE_GMRES__array_isRowOrder(
  const struct bHYPRE_GMRES__array* array);

struct bHYPRE_GMRES__array*
bHYPRE_GMRES__array_slice(
  struct bHYPRE_GMRES__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_GMRES__array_copy(
  const struct bHYPRE_GMRES__array* src,
  struct bHYPRE_GMRES__array* dest);

struct bHYPRE_GMRES__array*
bHYPRE_GMRES__array_ensure(
  struct bHYPRE_GMRES__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
