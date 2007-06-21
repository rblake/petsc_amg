/*
 * File:          bHYPRE_HGMRES_Impl.h
 * Symbol:        bHYPRE.HGMRES-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side implementation for bHYPRE.HGMRES
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_HGMRES_Impl_h
#define included_bHYPRE_HGMRES_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_Solver_h
#include "bHYPRE_Solver.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif
#ifndef included_bHYPRE_HGMRES_h
#include "bHYPRE_HGMRES.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_bHYPRE_PreconditionedSolver_h
#include "bHYPRE_PreconditionedSolver.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES._includes) */
/* Insert-Code-Here {bHYPRE.HGMRES._includes} (include files) */
#include "HYPRE.h"
#include "utilities.h"
#include "krylov.h"
#include "HYPRE_parcsr_ls.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES._includes) */

/*
 * Private data for class bHYPRE.HGMRES
 */

struct bHYPRE_HGMRES__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES._data) */
  /* Insert-Code-Here {bHYPRE.HGMRES._data} (private data members) */

   MPI_Comm comm;
   HYPRE_Solver solver;
   bHYPRE_Operator matrix;
   char * vector_type;

   /* parameter cache, to save in Set*Parameter functions and copy in Apply: */
   double tol;
   int k_dim;
   int min_iter;
   int max_iter;
   int rel_change;
   int stop_crit;
   int log_level;
   int printlevel;

   /* preconditioner cache, to save in SetPreconditioner and apply in Apply:*/
   char * precond_name;
   HYPRE_Solver * solverprecond;
   HYPRE_PtrToSolverFcn precond; /* function */
   HYPRE_PtrToSolverFcn precond_setup; /* function */

   bHYPRE_Solver bprecond;  /* just used in GetPreconditioner */

  /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_HGMRES__data*
bHYPRE_HGMRES__get_data(
  bHYPRE_HGMRES);

extern void
bHYPRE_HGMRES__set_data(
  bHYPRE_HGMRES,
  struct bHYPRE_HGMRES__data*);

extern
void
impl_bHYPRE_HGMRES__load(
  void);

extern
void
impl_bHYPRE_HGMRES__ctor(
  /* in */ bHYPRE_HGMRES self);

extern
void
impl_bHYPRE_HGMRES__dtor(
  /* in */ bHYPRE_HGMRES self);

/*
 * User-defined object methods
 */

extern
bHYPRE_HGMRES
impl_bHYPRE_HGMRES_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_Solver(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_HGMRES_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_Vector(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_HGMRES__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_HGMRES(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_HGMRES(struct 
  bHYPRE_HGMRES__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_HGMRES_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_HGMRES_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_PreconditionedSolver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_PreconditionedSolver(struct 
  bHYPRE_PreconditionedSolver__object* obj);
extern
int32_t
impl_bHYPRE_HGMRES_SetCommunicator(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_HGMRES_SetIntParameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_HGMRES_SetDoubleParameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_HGMRES_SetStringParameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_HGMRES_SetIntArray1Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_HGMRES_SetIntArray2Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_HGMRES_SetDoubleArray1Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_HGMRES_SetDoubleArray2Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_HGMRES_GetIntValue(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_HGMRES_GetDoubleValue(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_HGMRES_Setup(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_HGMRES_Apply(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_HGMRES_ApplyAdjoint(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_HGMRES_SetOperator(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Operator A);

extern
int32_t
impl_bHYPRE_HGMRES_SetTolerance(
  /* in */ bHYPRE_HGMRES self,
  /* in */ double tolerance);

extern
int32_t
impl_bHYPRE_HGMRES_SetMaxIterations(
  /* in */ bHYPRE_HGMRES self,
  /* in */ int32_t max_iterations);

extern
int32_t
impl_bHYPRE_HGMRES_SetLogging(
  /* in */ bHYPRE_HGMRES self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_HGMRES_SetPrintLevel(
  /* in */ bHYPRE_HGMRES self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_HGMRES_GetNumIterations(
  /* in */ bHYPRE_HGMRES self,
  /* out */ int32_t* num_iterations);

extern
int32_t
impl_bHYPRE_HGMRES_GetRelResidualNorm(
  /* in */ bHYPRE_HGMRES self,
  /* out */ double* norm);

extern
int32_t
impl_bHYPRE_HGMRES_SetPreconditioner(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Solver s);

extern
int32_t
impl_bHYPRE_HGMRES_GetPreconditioner(
  /* in */ bHYPRE_HGMRES self,
  /* out */ bHYPRE_Solver* s);

extern
int32_t
impl_bHYPRE_HGMRES_Clone(
  /* in */ bHYPRE_HGMRES self,
  /* out */ bHYPRE_PreconditionedSolver* x);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_Solver(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_HGMRES_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_Vector(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_HGMRES__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_HGMRES(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_HGMRES(struct 
  bHYPRE_HGMRES__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_HGMRES_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_HGMRES_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_PreconditionedSolver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_PreconditionedSolver(struct 
  bHYPRE_PreconditionedSolver__object* obj);
#ifdef __cplusplus
}
#endif
#endif
