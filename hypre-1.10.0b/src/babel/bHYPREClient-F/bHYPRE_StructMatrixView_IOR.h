/*
 * File:          bHYPRE_StructMatrixView_IOR.h
 * Symbol:        bHYPRE.StructMatrixView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.4
 * Description:   Intermediate Object Representation for bHYPRE.StructMatrixView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_StructMatrixView_IOR_h
#define included_bHYPRE_StructMatrixView_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.StructMatrixView" (version 1.0.0)
 */

struct bHYPRE_StructMatrixView__array;
struct bHYPRE_StructMatrixView__object;

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_MPICommunicator__array;
struct bHYPRE_MPICommunicator__object;
struct bHYPRE_StructGrid__array;
struct bHYPRE_StructGrid__object;
struct bHYPRE_StructStencil__array;
struct bHYPRE_StructStencil__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_io_Deserializer__array;
struct sidl_io_Deserializer__object;
struct sidl_io_Serializer__array;
struct sidl_io_Serializer__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_StructMatrixView__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ void* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ void* self);
  void (*f__exec)(
    /* in */ void* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ void* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ void* self);
  void (*f_deleteRef)(
    /* in */ void* self);
  sidl_bool (*f_isSame)(
    /* in */ void* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ void* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ void* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ void* self);
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ void* self,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm);
  int32_t (*f_Initialize)(
    /* in */ void* self);
  int32_t (*f_Assemble)(
    /* in */ void* self);
  /* Methods introduced in bHYPRE.MatrixVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.StructMatrixView-v1.0.0 */
  int32_t (*f_SetGrid)(
    /* in */ void* self,
    /* in */ struct bHYPRE_StructGrid__object* grid);
  int32_t (*f_SetStencil)(
    /* in */ void* self,
    /* in */ struct bHYPRE_StructStencil__object* stencil);
  int32_t (*f_SetValues)(
    /* in */ void* self,
    /* in */ struct sidl_int__array* index,
    /* in */ struct sidl_int__array* stencil_indices,
    /* in */ struct sidl_double__array* values);
  int32_t (*f_SetBoxValues)(
    /* in */ void* self,
    /* in */ struct sidl_int__array* ilower,
    /* in */ struct sidl_int__array* iupper,
    /* in */ struct sidl_int__array* stencil_indices,
    /* in */ struct sidl_double__array* values);
  int32_t (*f_SetNumGhost)(
    /* in */ void* self,
    /* in */ struct sidl_int__array* num_ghost);
  int32_t (*f_SetSymmetric)(
    /* in */ void* self,
    /* in */ int32_t symmetric);
  int32_t (*f_SetConstantEntries)(
    /* in */ void* self,
    /* in */ struct sidl_int__array* stencil_constant_points);
  int32_t (*f_SetConstantValues)(
    /* in */ void* self,
    /* in */ struct sidl_int__array* stencil_indices,
    /* in */ struct sidl_double__array* values);
};

/*
 * Define the interface object structure.
 */

struct bHYPRE_StructMatrixView__object {
  struct bHYPRE_StructMatrixView__epv* d_epv;
  void*                                d_object;
};

/**
 * 
 * 
 * Anonymous class definition
 * 
 * 
 */
#ifndef included_bHYPRE_MatrixVectorView_IOR_h
#include "bHYPRE_MatrixVectorView_IOR.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_IOR_h
#include "bHYPRE_ProblemDefinition_IOR.h"
#endif
#ifndef included_bHYPRE_StructMatrixView_IOR_h
#include "bHYPRE_StructMatrixView_IOR.h"
#endif
#ifndef included_sidl_BaseInterface_IOR_h
#include "sidl_BaseInterface_IOR.h"
#endif

/*
 * Symbol "bHYPRE._StructMatrixView" (version 1.0)
 */

struct bHYPRE__StructMatrixView__array;
struct bHYPRE__StructMatrixView__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE__StructMatrixView__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct bHYPRE__StructMatrixView__object* self);
  void (*f__exec)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct bHYPRE__StructMatrixView__object* self);
  void (*f__ctor)(
    /* in */ struct bHYPRE__StructMatrixView__object* self);
  void (*f__dtor)(
    /* in */ struct bHYPRE__StructMatrixView__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct bHYPRE__StructMatrixView__object* self);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE__StructMatrixView__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE__StructMatrixView__object* self);
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm);
  int32_t (*f_Initialize)(
    /* in */ struct bHYPRE__StructMatrixView__object* self);
  int32_t (*f_Assemble)(
    /* in */ struct bHYPRE__StructMatrixView__object* self);
  /* Methods introduced in bHYPRE.MatrixVectorView-v1.0.0 */
  /* Methods introduced in bHYPRE.StructMatrixView-v1.0.0 */
  int32_t (*f_SetGrid)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ struct bHYPRE_StructGrid__object* grid);
  int32_t (*f_SetStencil)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ struct bHYPRE_StructStencil__object* stencil);
  int32_t (*f_SetValues)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ struct sidl_int__array* index,
    /* in */ struct sidl_int__array* stencil_indices,
    /* in */ struct sidl_double__array* values);
  int32_t (*f_SetBoxValues)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ struct sidl_int__array* ilower,
    /* in */ struct sidl_int__array* iupper,
    /* in */ struct sidl_int__array* stencil_indices,
    /* in */ struct sidl_double__array* values);
  int32_t (*f_SetNumGhost)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ struct sidl_int__array* num_ghost);
  int32_t (*f_SetSymmetric)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ int32_t symmetric);
  int32_t (*f_SetConstantEntries)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ struct sidl_int__array* stencil_constant_points);
  int32_t (*f_SetConstantValues)(
    /* in */ struct bHYPRE__StructMatrixView__object* self,
    /* in */ struct sidl_int__array* stencil_indices,
    /* in */ struct sidl_double__array* values);
  /* Methods introduced in bHYPRE._StructMatrixView-v1.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE__StructMatrixView__object {
  struct bHYPRE_MatrixVectorView__object  d_bhypre_matrixvectorview;
  struct bHYPRE_ProblemDefinition__object d_bhypre_problemdefinition;
  struct bHYPRE_StructMatrixView__object  d_bhypre_structmatrixview;
  struct sidl_BaseInterface__object       d_sidl_baseinterface;
  struct bHYPRE__StructMatrixView__epv*   d_epv;
  void*                                   d_data;
};


#ifdef __cplusplus
}
#endif
#endif
