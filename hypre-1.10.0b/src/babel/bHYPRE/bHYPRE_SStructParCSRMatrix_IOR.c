/*
 * File:          bHYPRE_SStructParCSRMatrix_IOR.c
 * Symbol:        bHYPRE.SStructParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Intermediate Object Representation for bHYPRE.SStructParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#include "sidl_rmi_InstanceHandle.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "bHYPRE_SStructParCSRMatrix_IOR.h"
#ifndef included_sidl_BaseClass_Impl_h
#include "sidl_BaseClass_Impl.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_ClassInfoI_h
#include "sidl_ClassInfoI.h"
#endif

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t bHYPRE_SStructParCSRMatrix__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &bHYPRE_SStructParCSRMatrix__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &bHYPRE_SStructParCSRMatrix__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &bHYPRE_SStructParCSRMatrix__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

/*
 * Static variables to hold version of IOR
 */

static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 9;

/*
 * Static variable to hold shared ClassInfo interface.
 */

static sidl_ClassInfo s_classInfo = NULL;
static int s_classInfo_init = 1;

/*
 * Static variable to make sure _load called no more than once
 */

static int s_load_called = 0;
/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_static_initialized = 0;

static struct bHYPRE_SStructParCSRMatrix__epv  
  s_new_epv__bhypre_sstructparcsrmatrix;
static struct bHYPRE_SStructParCSRMatrix__sepv 
  s_stc_epv__bhypre_sstructparcsrmatrix;

static struct bHYPRE_MatrixVectorView__epv s_new_epv__bhypre_matrixvectorview;

static struct bHYPRE_Operator__epv s_new_epv__bhypre_operator;

static struct bHYPRE_ProblemDefinition__epv s_new_epv__bhypre_problemdefinition;

static struct bHYPRE_SStructMatrixView__epv s_new_epv__bhypre_sstructmatrixview;

static struct bHYPRE_SStruct_MatrixVectorView__epv 
  s_new_epv__bhypre_sstruct_matrixvectorview;

static struct sidl_BaseClass__epv  s_new_epv__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_new_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv__sidl_baseinterface;

/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void bHYPRE_SStructParCSRMatrix__set_epv(
  struct bHYPRE_SStructParCSRMatrix__epv* epv);
extern void bHYPRE_SStructParCSRMatrix__set_sepv(
  struct bHYPRE_SStructParCSRMatrix__sepv* sepv);
extern void bHYPRE_SStructParCSRMatrix__call_load(void);
#ifdef __cplusplus
}
#endif

static void
bHYPRE_SStructParCSRMatrix_addRef__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_addRef)(
    self);

  /* pack return value */
  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_deleteRef__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_deleteRef)(
    self);

  /* pack return value */
  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_isSame__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_BaseInterface__object* iobj;
  sidl_bool _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_isSame)(
    self,
    iobj);

  /* pack return value */
  sidl_io_Serializer_packBool( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_queryInt__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  struct sidl_BaseInterface__object* _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_queryInt)(
    self,
    name);

  /* pack return value */
  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_isType__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  sidl_bool _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_isType)(
    self,
    name);

  /* pack return value */
  sidl_io_Serializer_packBool( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_getClassInfo__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_ClassInfo__object* _retval;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getClassInfo)(
    self);

  /* pack return value */
  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_SetCommunicator__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* mpi_comm_str= NULL;
  struct bHYPRE_MPICommunicator__object* mpi_comm= NULL;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "mpi_comm", &mpi_comm_str, _ex2);
  mpi_comm = 
    skel_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_MPICommunicator(
    mpi_comm_str, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetCommunicator)(
    self,
    mpi_comm);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_Initialize__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_Initialize)(
    self);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_Assemble__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_Assemble)(
    self);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_GetObject__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_BaseInterface__object* A_tmp;
  struct sidl_BaseInterface__object** A= &A_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_GetObject)(
    self,
    A);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_SetGraph__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* graph_str= NULL;
  struct bHYPRE_SStructGraph__object* graph= NULL;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "graph", &graph_str, _ex2);
  graph = 
    skel_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructGraph(graph_str,
    _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetGraph)(
    self,
    graph);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_SetValues__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t part;
  struct sidl_int__array* index;
  int32_t var;
  struct sidl_int__array* entries;
  struct sidl_double__array* values;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackInt( inArgs, "part", &part, _ex2);
  sidl_io_Deserializer_unpackInt( inArgs, "var", &var, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetValues)(
    self,
    part,
    index,
    var,
    entries,
    values);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_SetBoxValues__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t part;
  struct sidl_int__array* ilower;
  struct sidl_int__array* iupper;
  int32_t var;
  struct sidl_int__array* entries;
  struct sidl_double__array* values;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackInt( inArgs, "part", &part, _ex2);
  sidl_io_Deserializer_unpackInt( inArgs, "var", &var, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetBoxValues)(
    self,
    part,
    ilower,
    iupper,
    var,
    entries,
    values);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_AddToValues__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t part;
  struct sidl_int__array* index;
  int32_t var;
  struct sidl_int__array* entries;
  struct sidl_double__array* values;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackInt( inArgs, "part", &part, _ex2);
  sidl_io_Deserializer_unpackInt( inArgs, "var", &var, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_AddToValues)(
    self,
    part,
    index,
    var,
    entries,
    values);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_AddToBoxValues__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t part;
  struct sidl_int__array* ilower;
  struct sidl_int__array* iupper;
  int32_t var;
  struct sidl_int__array* entries;
  struct sidl_double__array* values;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackInt( inArgs, "part", &part, _ex2);
  sidl_io_Deserializer_unpackInt( inArgs, "var", &var, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_AddToBoxValues)(
    self,
    part,
    ilower,
    iupper,
    var,
    entries,
    values);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_SetSymmetric__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t part;
  int32_t var;
  int32_t to_var;
  int32_t symmetric;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackInt( inArgs, "part", &part, _ex2);
  sidl_io_Deserializer_unpackInt( inArgs, "var", &var, _ex2);
  sidl_io_Deserializer_unpackInt( inArgs, "to_var", &to_var, _ex2);
  sidl_io_Deserializer_unpackInt( inArgs, "symmetric", &symmetric, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetSymmetric)(
    self,
    part,
    var,
    to_var,
    symmetric);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_SetNSSymmetric__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t symmetric;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackInt( inArgs, "symmetric", &symmetric, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetNSSymmetric)(
    self,
    symmetric);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_SetComplex__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_SetComplex)(
    self);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_Print__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* filename= NULL;
  int32_t all;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "filename", &filename, _ex2);
  sidl_io_Deserializer_unpackInt( inArgs, "all", &all, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_Print)(
    self,
    filename,
    all);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_SetIntParameter__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  int32_t value;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);
  sidl_io_Deserializer_unpackInt( inArgs, "value", &value, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetIntParameter)(
    self,
    name,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_SetDoubleParameter__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  double value;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);
  sidl_io_Deserializer_unpackDouble( inArgs, "value", &value, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetDoubleParameter)(
    self,
    name,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_SetStringParameter__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  char* value= NULL;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);
  sidl_io_Deserializer_unpackString( inArgs, "value", &value, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetStringParameter)(
    self,
    name,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  struct sidl_int__array* value;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetIntArray1Parameter)(
    self,
    name,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  struct sidl_int__array* value;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetIntArray2Parameter)(
    self,
    name,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  struct sidl_double__array* value;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetDoubleArray1Parameter)(
    self,
    name,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  struct sidl_double__array* value;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetDoubleArray2Parameter)(
    self,
    name,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_GetIntValue__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  int32_t value_tmp;
  int32_t* value= &value_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_GetIntValue)(
    self,
    name,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */
  sidl_io_Serializer_packInt( outArgs, "value", *value, _ex2);

}

static void
bHYPRE_SStructParCSRMatrix_GetDoubleValue__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  double value_tmp;
  double* value= &value_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_GetDoubleValue)(
    self,
    name,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */
  sidl_io_Serializer_packDouble( outArgs, "value", *value, _ex2);

}

static void
bHYPRE_SStructParCSRMatrix_Setup__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct bHYPRE_Vector__object* b;
  struct bHYPRE_Vector__object* x;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_Setup)(
    self,
    b,
    x);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_Apply__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct bHYPRE_Vector__object* b;
  struct bHYPRE_Vector__object* x_tmp;
  struct bHYPRE_Vector__object** x= &x_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_Apply)(
    self,
    b,
    x);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_SStructParCSRMatrix_ApplyAdjoint__exec(
        struct bHYPRE_SStructParCSRMatrix__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct bHYPRE_Vector__object* b;
  struct bHYPRE_Vector__object* x_tmp;
  struct bHYPRE_Vector__object** x= &x_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_ApplyAdjoint)(
    self,
    b,
    x);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void ior_bHYPRE_SStructParCSRMatrix__ensure_load_called(void) {
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  if (! s_load_called ) {
    bHYPRE_SStructParCSRMatrix__call_load();
    s_load_called=1;
  }
}
/*
 * CAST: dynamic type casting support.
 */

static void* ior_bHYPRE_SStructParCSRMatrix__cast(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  const char* name)
{
  void* cast = NULL;

  struct bHYPRE_SStructParCSRMatrix__object* s0 = self;
  struct sidl_BaseClass__object*             s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "bHYPRE.SStructParCSRMatrix")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "bHYPRE.MatrixVectorView")) {
    cast = (void*) &s0->d_bhypre_matrixvectorview;
  } else if (!strcmp(name, "bHYPRE.Operator")) {
    cast = (void*) &s0->d_bhypre_operator;
  } else if (!strcmp(name, "bHYPRE.ProblemDefinition")) {
    cast = (void*) &s0->d_bhypre_problemdefinition;
  } else if (!strcmp(name, "bHYPRE.SStructMatrixView")) {
    cast = (void*) &s0->d_bhypre_sstructmatrixview;
  } else if (!strcmp(name, "bHYPRE.SStruct_MatrixVectorView")) {
    cast = (void*) &s0->d_bhypre_sstruct_matrixvectorview;
  } else if (!strcmp(name, "sidl.BaseClass")) {
    cast = (void*) s1;
  } else if (!strcmp(name, "sidl.BaseInterface")) {
    cast = (void*) &s1->d_sidl_baseinterface;
  }

  return cast;
}

/*
 * DELETE: call destructor and free object memory.
 */

static void ior_bHYPRE_SStructParCSRMatrix__delete(
  struct bHYPRE_SStructParCSRMatrix__object* self)
{
  bHYPRE_SStructParCSRMatrix__fini(self);
  memset((void*)self, 0, sizeof(struct bHYPRE_SStructParCSRMatrix__object));
  free((void*) self);
}

static char*
ior_bHYPRE_SStructParCSRMatrix__getURL(
    struct bHYPRE_SStructParCSRMatrix__object* self) {
  sidl_rmi_InstanceHandle conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_BaseInterface _ex = NULL;
  /* TODO: Make this work for local object! */
  return NULL;
}
struct bHYPRE_SStructParCSRMatrix__method {
  const char *d_name;
  void (*d_func)(struct bHYPRE_SStructParCSRMatrix__object*,
    struct sidl_io_Deserializer__object *,
    struct sidl_io_Serializer__object *);
};

static void
ior_bHYPRE_SStructParCSRMatrix__exec(
    struct bHYPRE_SStructParCSRMatrix__object* self,
    const char* methodName,
    struct sidl_io_Deserializer__object* inArgs,
    struct sidl_io_Serializer__object* outArgs ) { 
  static const struct bHYPRE_SStructParCSRMatrix__method  s_methods[] = {
    { "AddToBoxValues", bHYPRE_SStructParCSRMatrix_AddToBoxValues__exec },
    { "AddToValues", bHYPRE_SStructParCSRMatrix_AddToValues__exec },
    { "Apply", bHYPRE_SStructParCSRMatrix_Apply__exec },
    { "ApplyAdjoint", bHYPRE_SStructParCSRMatrix_ApplyAdjoint__exec },
    { "Assemble", bHYPRE_SStructParCSRMatrix_Assemble__exec },
    { "GetDoubleValue", bHYPRE_SStructParCSRMatrix_GetDoubleValue__exec },
    { "GetIntValue", bHYPRE_SStructParCSRMatrix_GetIntValue__exec },
    { "GetObject", bHYPRE_SStructParCSRMatrix_GetObject__exec },
    { "Initialize", bHYPRE_SStructParCSRMatrix_Initialize__exec },
    { "Print", bHYPRE_SStructParCSRMatrix_Print__exec },
    { "SetBoxValues", bHYPRE_SStructParCSRMatrix_SetBoxValues__exec },
    { "SetCommunicator", bHYPRE_SStructParCSRMatrix_SetCommunicator__exec },
    { "SetComplex", bHYPRE_SStructParCSRMatrix_SetComplex__exec },
    { "SetDoubleArray1Parameter",
      bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter__exec },
    { "SetDoubleArray2Parameter",
      bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter__exec },
    { "SetDoubleParameter",
      bHYPRE_SStructParCSRMatrix_SetDoubleParameter__exec },
    { "SetGraph", bHYPRE_SStructParCSRMatrix_SetGraph__exec },
    { "SetIntArray1Parameter",
      bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter__exec },
    { "SetIntArray2Parameter",
      bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter__exec },
    { "SetIntParameter", bHYPRE_SStructParCSRMatrix_SetIntParameter__exec },
    { "SetNSSymmetric", bHYPRE_SStructParCSRMatrix_SetNSSymmetric__exec },
    { "SetStringParameter",
      bHYPRE_SStructParCSRMatrix_SetStringParameter__exec },
    { "SetSymmetric", bHYPRE_SStructParCSRMatrix_SetSymmetric__exec },
    { "SetValues", bHYPRE_SStructParCSRMatrix_SetValues__exec },
    { "Setup", bHYPRE_SStructParCSRMatrix_Setup__exec },
    { "addRef", bHYPRE_SStructParCSRMatrix_addRef__exec },
    { "deleteRef", bHYPRE_SStructParCSRMatrix_deleteRef__exec },
    { "getClassInfo", bHYPRE_SStructParCSRMatrix_getClassInfo__exec },
    { "isSame", bHYPRE_SStructParCSRMatrix_isSame__exec },
    { "isType", bHYPRE_SStructParCSRMatrix_isType__exec },
    { "queryInt", bHYPRE_SStructParCSRMatrix_queryInt__exec }
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct bHYPRE_SStructParCSRMatrix__method);
  if (methodName) {
    /* Use binary search to locate method */
    while (l < u) {
      i = (l + u) >> 1;
      if (!(cmp=strcmp(methodName, s_methods[i].d_name))) {
        (s_methods[i].d_func)(self, inArgs, outArgs);
        return;
      }
      else if (cmp < 0) u = i;
      else l = i + 1;
    }
  }
  /* TODO: add code for method not found */
}
/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void bHYPRE_SStructParCSRMatrix__init_epv(
  struct bHYPRE_SStructParCSRMatrix__object* self)
{
/*
 * assert( HAVE_LOCKED_STATIC_GLOBALS );
 */

  struct bHYPRE_SStructParCSRMatrix__object* s0 = self;
  struct sidl_BaseClass__object*             s1 = &s0->d_sidl_baseclass;

  struct bHYPRE_SStructParCSRMatrix__epv*      epv  = 
    &s_new_epv__bhypre_sstructparcsrmatrix;
  struct bHYPRE_MatrixVectorView__epv*         e0   = 
    &s_new_epv__bhypre_matrixvectorview;
  struct bHYPRE_Operator__epv*                 e1   = 
    &s_new_epv__bhypre_operator;
  struct bHYPRE_ProblemDefinition__epv*        e2   = 
    &s_new_epv__bhypre_problemdefinition;
  struct bHYPRE_SStructMatrixView__epv*        e3   = 
    &s_new_epv__bhypre_sstructmatrixview;
  struct bHYPRE_SStruct_MatrixVectorView__epv* e4   = 
    &s_new_epv__bhypre_sstruct_matrixvectorview;
  struct sidl_BaseClass__epv*                  e5   = 
    &s_new_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*              e6   = 
    &s_new_epv__sidl_baseinterface;

  s_old_epv__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old_epv__sidl_baseclass     = s1->d_epv;

  epv->f__cast                         = ior_bHYPRE_SStructParCSRMatrix__cast;
  epv->f__delete                       = ior_bHYPRE_SStructParCSRMatrix__delete;
  epv->f__exec                         = ior_bHYPRE_SStructParCSRMatrix__exec;
  epv->f__getURL                       = ior_bHYPRE_SStructParCSRMatrix__getURL;
  epv->f__ctor                         = NULL;
  epv->f__dtor                         = NULL;
  epv->f_addRef                        = (void (*)(struct 
    bHYPRE_SStructParCSRMatrix__object*)) s1->d_epv->f_addRef;
  epv->f_deleteRef                     = (void (*)(struct 
    bHYPRE_SStructParCSRMatrix__object*)) s1->d_epv->f_deleteRef;
  epv->f_isSame                        = (sidl_bool (*)(struct 
    bHYPRE_SStructParCSRMatrix__object*,
    struct sidl_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt                      = (struct sidl_BaseInterface__object* 
    (*)(struct bHYPRE_SStructParCSRMatrix__object*,
    const char*)) s1->d_epv->f_queryInt;
  epv->f_isType                        = (sidl_bool (*)(struct 
    bHYPRE_SStructParCSRMatrix__object*,const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo                  = (struct sidl_ClassInfo__object* 
    (*)(struct bHYPRE_SStructParCSRMatrix__object*)) s1->d_epv->f_getClassInfo;
  epv->f_SetCommunicator               = NULL;
  epv->f_Initialize                    = NULL;
  epv->f_Assemble                      = NULL;
  epv->f_GetObject                     = NULL;
  epv->f_SetGraph                      = NULL;
  epv->f_SetValues                     = NULL;
  epv->f_SetBoxValues                  = NULL;
  epv->f_AddToValues                   = NULL;
  epv->f_AddToBoxValues                = NULL;
  epv->f_SetSymmetric                  = NULL;
  epv->f_SetNSSymmetric                = NULL;
  epv->f_SetComplex                    = NULL;
  epv->f_Print                         = NULL;
  epv->f_SetIntParameter               = NULL;
  epv->f_SetDoubleParameter            = NULL;
  epv->f_SetStringParameter            = NULL;
  epv->f_SetIntArray1Parameter         = NULL;
  epv->f_SetIntArray2Parameter         = NULL;
  epv->f_SetDoubleArray1Parameter      = NULL;
  epv->f_SetDoubleArray2Parameter      = NULL;
  epv->f_GetIntValue                   = NULL;
  epv->f_GetDoubleValue                = NULL;
  epv->f_Setup                         = NULL;
  epv->f_Apply                         = NULL;
  epv->f_ApplyAdjoint                  = NULL;

  bHYPRE_SStructParCSRMatrix__set_epv(epv);

  e0->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete             = (void (*)(void*)) epv->f__delete;
  e0->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e0->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e0->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e0->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e0->f_SetCommunicator     = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*)) epv->f_SetCommunicator;
  e0->f_Initialize          = (int32_t (*)(void*)) epv->f_Initialize;
  e0->f_Assemble            = (int32_t (*)(void*)) epv->f_Assemble;

  e1->f__cast                    = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete                  = (void (*)(void*)) epv->f__delete;
  e1->f__exec                    = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e1->f_addRef                   = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef                = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame                   = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(void*,const char*)) epv->f_queryInt;
  e1->f_isType                   = (sidl_bool (*)(void*,
    const char*)) epv->f_isType;
  e1->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e1->f_SetCommunicator          = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*)) epv->f_SetCommunicator;
  e1->f_SetIntParameter          = (int32_t (*)(void*,const char*,
    int32_t)) epv->f_SetIntParameter;
  e1->f_SetDoubleParameter       = (int32_t (*)(void*,const char*,
    double)) epv->f_SetDoubleParameter;
  e1->f_SetStringParameter       = (int32_t (*)(void*,const char*,
    const char*)) epv->f_SetStringParameter;
  e1->f_SetIntArray1Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray1Parameter;
  e1->f_SetIntArray2Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray2Parameter;
  e1->f_SetDoubleArray1Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray1Parameter;
  e1->f_SetDoubleArray2Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray2Parameter;
  e1->f_GetIntValue              = (int32_t (*)(void*,const char*,
    int32_t*)) epv->f_GetIntValue;
  e1->f_GetDoubleValue           = (int32_t (*)(void*,const char*,
    double*)) epv->f_GetDoubleValue;
  e1->f_Setup                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object*)) epv->f_Setup;
  e1->f_Apply                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object**)) epv->f_Apply;
  e1->f_ApplyAdjoint             = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,
    struct bHYPRE_Vector__object**)) epv->f_ApplyAdjoint;

  e2->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete             = (void (*)(void*)) epv->f__delete;
  e2->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e2->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e2->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e2->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e2->f_SetCommunicator     = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*)) epv->f_SetCommunicator;
  e2->f_Initialize          = (int32_t (*)(void*)) epv->f_Initialize;
  e2->f_Assemble            = (int32_t (*)(void*)) epv->f_Assemble;

  e3->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete             = (void (*)(void*)) epv->f__delete;
  e3->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e3->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e3->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e3->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e3->f_SetCommunicator     = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*)) epv->f_SetCommunicator;
  e3->f_Initialize          = (int32_t (*)(void*)) epv->f_Initialize;
  e3->f_Assemble            = (int32_t (*)(void*)) epv->f_Assemble;
  e3->f_GetObject           = (int32_t (*)(void*,
    struct sidl_BaseInterface__object**)) epv->f_GetObject;
  e3->f_SetGraph            = (int32_t (*)(void*,
    struct bHYPRE_SStructGraph__object*)) epv->f_SetGraph;
  e3->f_SetValues           = (int32_t (*)(void*,int32_t,
    struct sidl_int__array*,int32_t,struct sidl_int__array*,
    struct sidl_double__array*)) epv->f_SetValues;
  e3->f_SetBoxValues        = (int32_t (*)(void*,int32_t,
    struct sidl_int__array*,struct sidl_int__array*,int32_t,
    struct sidl_int__array*,struct sidl_double__array*)) epv->f_SetBoxValues;
  e3->f_AddToValues         = (int32_t (*)(void*,int32_t,
    struct sidl_int__array*,int32_t,struct sidl_int__array*,
    struct sidl_double__array*)) epv->f_AddToValues;
  e3->f_AddToBoxValues      = (int32_t (*)(void*,int32_t,
    struct sidl_int__array*,struct sidl_int__array*,int32_t,
    struct sidl_int__array*,struct sidl_double__array*)) epv->f_AddToBoxValues;
  e3->f_SetSymmetric        = (int32_t (*)(void*,int32_t,int32_t,int32_t,
    int32_t)) epv->f_SetSymmetric;
  e3->f_SetNSSymmetric      = (int32_t (*)(void*,
    int32_t)) epv->f_SetNSSymmetric;
  e3->f_SetComplex          = (int32_t (*)(void*)) epv->f_SetComplex;
  e3->f_Print               = (int32_t (*)(void*,const char*,
    int32_t)) epv->f_Print;

  e4->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e4->f__delete             = (void (*)(void*)) epv->f__delete;
  e4->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e4->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e4->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e4->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e4->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e4->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e4->f_SetCommunicator     = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*)) epv->f_SetCommunicator;
  e4->f_Initialize          = (int32_t (*)(void*)) epv->f_Initialize;
  e4->f_Assemble            = (int32_t (*)(void*)) epv->f_Assemble;
  e4->f_GetObject           = (int32_t (*)(void*,
    struct sidl_BaseInterface__object**)) epv->f_GetObject;

  e5->f__cast               = (void* (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f__cast;
  e5->f__delete             = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f__delete;
  e5->f__exec               = (void (*)(struct sidl_BaseClass__object*,
    const char*,struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e5->f_addRef              = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_addRef;
  e5->f_deleteRef           = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_deleteRef;
  e5->f_isSame              = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e5->f_queryInt            = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_BaseClass__object*,const char*)) epv->f_queryInt;
  e5->f_isType              = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f_isType;
  e5->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*)) epv->f_getClassInfo;

  e6->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e6->f__delete             = (void (*)(void*)) epv->f__delete;
  e6->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e6->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e6->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e6->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e6->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e6->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e6->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  s_method_initialized = 1;
  ior_bHYPRE_SStructParCSRMatrix__ensure_load_called();
}

/*
 * SEPV: create the static entry point vector (SEPV).
 */

static void bHYPRE_SStructParCSRMatrix__init_sepv(void)
{
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  struct bHYPRE_SStructParCSRMatrix__sepv*  s = 
    &s_stc_epv__bhypre_sstructparcsrmatrix;

  s->f_Create         = NULL;

  bHYPRE_SStructParCSRMatrix__set_sepv(s);

  s_static_initialized = 1;
  ior_bHYPRE_SStructParCSRMatrix__ensure_load_called();
}

/*
 * STATIC: return pointer to static EPV structure.
 */

struct bHYPRE_SStructParCSRMatrix__sepv*
bHYPRE_SStructParCSRMatrix__statics(void)
{
  LOCK_STATIC_GLOBALS;
  if (!s_static_initialized) {
    bHYPRE_SStructParCSRMatrix__init_sepv();
  }
  UNLOCK_STATIC_GLOBALS;
  return &s_stc_epv__bhypre_sstructparcsrmatrix;
}

/*
 * SUPER: return's parent's non-overrided EPV
 */

static struct sidl_BaseClass__epv* bHYPRE_SStructParCSRMatrix__super(void) {
  return s_old_epv__sidl_baseclass;
}

/*
 * initClassInfo: create a ClassInfo interface if necessary.
 */

static void
initClassInfo(sidl_ClassInfo *info)
{
  LOCK_STATIC_GLOBALS;
  if (s_classInfo_init) {
    sidl_ClassInfoI impl;
    s_classInfo_init = 0;
    impl = sidl_ClassInfoI__create();
    s_classInfo = sidl_ClassInfo__cast(impl);
    if (impl) {
      sidl_ClassInfoI_setName(impl, "bHYPRE.SStructParCSRMatrix");
      sidl_ClassInfoI_setIORVersion(impl, s_IOR_MAJOR_VERSION,
        s_IOR_MINOR_VERSION);
    }
  }
  if (s_classInfo) {
    if (*info) {
      sidl_ClassInfo_deleteRef(*info);
    }
    *info = s_classInfo;
    sidl_ClassInfo_addRef(*info);
  }
UNLOCK_STATIC_GLOBALS;
}

/*
 * initMetadata: store IOR version & class in sidl.BaseClass's data
 */

static void
initMetadata(struct bHYPRE_SStructParCSRMatrix__object* self)
{
  if (self) {
    struct sidl_BaseClass__data *data = 
      sidl_BaseClass__get_data(sidl_BaseClass__cast(self));
    if (data) {
      data->d_IOR_major_version = s_IOR_MAJOR_VERSION;
      data->d_IOR_minor_version = s_IOR_MINOR_VERSION;
      initClassInfo(&(data->d_classinfo));
    }
  }
}

/*
 * NEW: allocate object and initialize it.
 */

struct bHYPRE_SStructParCSRMatrix__object*
bHYPRE_SStructParCSRMatrix__new(void)
{
  struct bHYPRE_SStructParCSRMatrix__object* self =
    (struct bHYPRE_SStructParCSRMatrix__object*) malloc(
      sizeof(struct bHYPRE_SStructParCSRMatrix__object));
  bHYPRE_SStructParCSRMatrix__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void bHYPRE_SStructParCSRMatrix__init(
  struct bHYPRE_SStructParCSRMatrix__object* self)
{
  struct bHYPRE_SStructParCSRMatrix__object* s0 = self;
  struct sidl_BaseClass__object*             s1 = &s0->d_sidl_baseclass;

  sidl_BaseClass__init(s1);

  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    bHYPRE_SStructParCSRMatrix__init_epv(s0);
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv = &s_new_epv__sidl_baseinterface;
  s1->d_epv                      = &s_new_epv__sidl_baseclass;

  s0->d_bhypre_matrixvectorview.d_epv         = 
    &s_new_epv__bhypre_matrixvectorview;
  s0->d_bhypre_operator.d_epv                 = &s_new_epv__bhypre_operator;
  s0->d_bhypre_problemdefinition.d_epv        = 
    &s_new_epv__bhypre_problemdefinition;
  s0->d_bhypre_sstructmatrixview.d_epv        = 
    &s_new_epv__bhypre_sstructmatrixview;
  s0->d_bhypre_sstruct_matrixvectorview.d_epv = 
    &s_new_epv__bhypre_sstruct_matrixvectorview;
  s0->d_epv                                   = 
    &s_new_epv__bhypre_sstructparcsrmatrix;

  s0->d_bhypre_matrixvectorview.d_object = self;

  s0->d_bhypre_operator.d_object = self;

  s0->d_bhypre_problemdefinition.d_object = self;

  s0->d_bhypre_sstructmatrixview.d_object = self;

  s0->d_bhypre_sstruct_matrixvectorview.d_object = self;

  s0->d_data = NULL;


  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void bHYPRE_SStructParCSRMatrix__fini(
  struct bHYPRE_SStructParCSRMatrix__object* self)
{
  struct bHYPRE_SStructParCSRMatrix__object* s0 = self;
  struct sidl_BaseClass__object*             s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old_epv__sidl_baseinterface;
  s1->d_epv                      = s_old_epv__sidl_baseclass;

  sidl_BaseClass__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
bHYPRE_SStructParCSRMatrix__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}

static const struct bHYPRE_SStructParCSRMatrix__external
s_externalEntryPoints = {
  bHYPRE_SStructParCSRMatrix__new,
  bHYPRE_SStructParCSRMatrix__statics,
  bHYPRE_SStructParCSRMatrix__super
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_SStructParCSRMatrix__external*
bHYPRE_SStructParCSRMatrix__externals(void)
{
  return &s_externalEntryPoints;
}

