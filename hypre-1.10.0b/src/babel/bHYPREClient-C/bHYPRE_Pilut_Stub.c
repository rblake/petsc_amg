/*
 * File:          bHYPRE_Pilut_Stub.c
 * Symbol:        bHYPRE.Pilut-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.Pilut
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#include "bHYPRE_Pilut.h"
#include "bHYPRE_Pilut_IOR.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#include <stddef.h>
#include <string.h>
#include "sidl_BaseInterface_IOR.h"
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include <stdlib.h>
#include "sidl_Loader.h"
#endif

/*
 * connect_loaded is a boolean value showing if the IHConnect for this object has been loaded into the connectRegistry
 */

static int connect_loaded = 0;

/*
 * Hold pointer to IOR functions.
 */

static const struct bHYPRE_Pilut__external *_externals = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct bHYPRE_Pilut__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
#ifdef SIDL_STATIC_LIBRARY
  _externals = bHYPRE_Pilut__externals();
#else
  _externals = (struct 
    bHYPRE_Pilut__external*)sidl_dynamicLoadIOR("bHYPRE.Pilut",
    "bHYPRE_Pilut__externals") ;
#endif
  return _externals;
}

#define _getExternals() (_externals ? _externals : _loadIOR())

/*
 * Hold pointer to static entry point vector
 */

static const struct bHYPRE_Pilut__sepv *_sepv = NULL;
/*
 * Return pointer to static functions.
 */

#define _getSEPV() (_sepv ? _sepv : (_sepv = (*(_getExternals()->getStaticEPV))()))
/*
 * Reset point to static functions.
 */

#define _resetSEPV() (_sepv = (*(_getExternals()->getStaticEPV))())

/*
 * Constructor function for the class.
 */

bHYPRE_Pilut
bHYPRE_Pilut__create()
{
  return (*(_getExternals()->createObject))();
}

static bHYPRE_Pilut bHYPRE_Pilut__remote(const char* url,
  sidl_BaseInterface *_ex);
/*
 * RMI constructor function for the class.
 */

bHYPRE_Pilut
bHYPRE_Pilut__createRemote(const char* url, sidl_BaseInterface *_ex)
{
  return bHYPRE_Pilut__remote(url, _ex);
}

static struct bHYPRE_Pilut__object* bHYPRE_Pilut__remoteConnect(const char* url,
  sidl_BaseInterface *_ex);
static struct bHYPRE_Pilut__object* 
  bHYPRE_Pilut__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex);
/*
 * RMI connector function for the class.
 */

bHYPRE_Pilut
bHYPRE_Pilut__connect(const char* url, sidl_BaseInterface *_ex)
{
  return bHYPRE_Pilut__remoteConnect(url, _ex);
}

/*
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>sidl</code> have an intrinsic reference count.
 * Objects continue to exist as long as the reference count is
 * positive. Clients should call this method whenever they
 * create another ongoing reference to an object or interface.
 * </p>
 * <p>
 * This does not have a return value because there is no language
 * independent type that can refer to an interface or a
 * class.
 * </p>
 */

void
bHYPRE_Pilut_addRef(
  /* in */ bHYPRE_Pilut self)
{
  (*self->d_epv->f_addRef)(
    self);
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
bHYPRE_Pilut_deleteRef(
  /* in */ bHYPRE_Pilut self)
{
  (*self->d_epv->f_deleteRef)(
    self);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

sidl_bool
bHYPRE_Pilut_isSame(
  /* in */ bHYPRE_Pilut self,
  /* in */ sidl_BaseInterface iobj)
{
  return (*self->d_epv->f_isSame)(
    self,
    iobj);
}

/*
 * Check whether the object can support the specified interface or
 * class.  If the <code>sidl</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */

sidl_BaseInterface
bHYPRE_Pilut_queryInt(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name)
{
  return (*self->d_epv->f_queryInt)(
    self,
    name);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

sidl_bool
bHYPRE_Pilut_isType(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name)
{
  return (*self->d_epv->f_isType)(
    self,
    name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

sidl_ClassInfo
bHYPRE_Pilut_getClassInfo(
  /* in */ bHYPRE_Pilut self)
{
  return (*self->d_epv->f_getClassInfo)(
    self);
}

/*
 * Method:  Create[]
 */

bHYPRE_Pilut
bHYPRE_Pilut_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A)
{
  return (_getSEPV()->f_Create)(
    mpi_comm,
    A);
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */

int32_t
bHYPRE_Pilut_SetCommunicator(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_MPICommunicator mpi_comm)
{
  return (*self->d_epv->f_SetCommunicator)(
    self,
    mpi_comm);
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_Pilut_SetIntParameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in */ int32_t value)
{
  return (*self->d_epv->f_SetIntParameter)(
    self,
    name,
    value);
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_Pilut_SetDoubleParameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in */ double value)
{
  return (*self->d_epv->f_SetDoubleParameter)(
    self,
    name,
    value);
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_Pilut_SetStringParameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in */ const char* value)
{
  return (*self->d_epv->f_SetStringParameter)(
    self,
    name,
    value);
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_Pilut_SetIntArray1Parameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues)
{
  int32_t value_lower[1], value_upper[1], value_stride[1]; 
  struct sidl_int__array value_real;
  struct sidl_int__array*value_tmp = &value_real;
  value_upper[0] = nvalues-1;
  sidl_int__array_init(value, value_tmp, 1, value_lower, value_upper,
    value_stride);
  return (*self->d_epv->f_SetIntArray1Parameter)(
    self,
    name,
    value_tmp);
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_Pilut_SetIntArray2Parameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value)
{
  return (*self->d_epv->f_SetIntArray2Parameter)(
    self,
    name,
    value);
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_Pilut_SetDoubleArray1Parameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues)
{
  int32_t value_lower[1], value_upper[1], value_stride[1]; 
  struct sidl_double__array value_real;
  struct sidl_double__array*value_tmp = &value_real;
  value_upper[0] = nvalues-1;
  sidl_double__array_init(value, value_tmp, 1, value_lower, value_upper,
    value_stride);
  return (*self->d_epv->f_SetDoubleArray1Parameter)(
    self,
    name,
    value_tmp);
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_Pilut_SetDoubleArray2Parameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value)
{
  return (*self->d_epv->f_SetDoubleArray2Parameter)(
    self,
    name,
    value);
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_Pilut_GetIntValue(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* out */ int32_t* value)
{
  return (*self->d_epv->f_GetIntValue)(
    self,
    name,
    value);
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_Pilut_GetDoubleValue(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* out */ double* value)
{
  return (*self->d_epv->f_GetDoubleValue)(
    self,
    name,
    value);
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

int32_t
bHYPRE_Pilut_Setup(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x)
{
  return (*self->d_epv->f_Setup)(
    self,
    b,
    x);
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

int32_t
bHYPRE_Pilut_Apply(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x)
{
  return (*self->d_epv->f_Apply)(
    self,
    b,
    x);
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */

int32_t
bHYPRE_Pilut_ApplyAdjoint(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x)
{
  return (*self->d_epv->f_ApplyAdjoint)(
    self,
    b,
    x);
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 * 
 */

int32_t
bHYPRE_Pilut_SetOperator(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_Operator A)
{
  return (*self->d_epv->f_SetOperator)(
    self,
    A);
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */

int32_t
bHYPRE_Pilut_SetTolerance(
  /* in */ bHYPRE_Pilut self,
  /* in */ double tolerance)
{
  return (*self->d_epv->f_SetTolerance)(
    self,
    tolerance);
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */

int32_t
bHYPRE_Pilut_SetMaxIterations(
  /* in */ bHYPRE_Pilut self,
  /* in */ int32_t max_iterations)
{
  return (*self->d_epv->f_SetMaxIterations)(
    self,
    max_iterations);
}

/*
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 * 
 */

int32_t
bHYPRE_Pilut_SetLogging(
  /* in */ bHYPRE_Pilut self,
  /* in */ int32_t level)
{
  return (*self->d_epv->f_SetLogging)(
    self,
    level);
}

/*
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 * 
 */

int32_t
bHYPRE_Pilut_SetPrintLevel(
  /* in */ bHYPRE_Pilut self,
  /* in */ int32_t level)
{
  return (*self->d_epv->f_SetPrintLevel)(
    self,
    level);
}

/*
 * (Optional) Return the number of iterations taken.
 * 
 */

int32_t
bHYPRE_Pilut_GetNumIterations(
  /* in */ bHYPRE_Pilut self,
  /* out */ int32_t* num_iterations)
{
  return (*self->d_epv->f_GetNumIterations)(
    self,
    num_iterations);
}

/*
 * (Optional) Return the norm of the relative residual.
 * 
 */

int32_t
bHYPRE_Pilut_GetRelResidualNorm(
  /* in */ bHYPRE_Pilut self,
  /* out */ double* norm)
{
  return (*self->d_epv->f_GetRelResidualNorm)(
    self,
    norm);
}

void
bHYPRE_Pilut_Create__sexec(
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  bHYPRE_MPICommunicator mpi_comm;
  bHYPRE_Operator A;
  bHYPRE_Pilut _retval;

  /* unpack in and inout argments */

  /* make the call */
  _retval = (_getSEPV()->f_Create)(
    mpi_comm,
    A);

  /* pack return value */
  /* pack out and inout argments */

}

/*
 * Cast method for interface and class type conversions.
 */

bHYPRE_Pilut
bHYPRE_Pilut__cast(
  void* obj)
{
  bHYPRE_Pilut cast = NULL;

  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.Pilut",
      (void*)bHYPRE_Pilut__IHConnect);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (bHYPRE_Pilut) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.Pilut");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
bHYPRE_Pilut__cast2(
  void* obj,
  const char* type)
{
  void* cast = NULL;

  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (*base->d_epv->f__cast)(base->d_object, type);
  }

  return cast;
}
/*
 * Select and execute a method by name
 */

void
bHYPRE_Pilut__exec(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs)
{
  (*self->d_epv->f__exec)(
  self,
  methodName,
  inArgs,
  outArgs);
}

struct bHYPRE_Pilut__smethod {
  const char *d_name;
  void (*d_func)(struct sidl_io_Deserializer__object *,
    struct sidl_io_Serializer__object *);
};

void
bHYPRE_Pilut__sexec(
        const char* methodName,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs ) { 
  static const struct bHYPRE_Pilut__smethod s_methods[] = {
    { "Create", bHYPRE_Pilut_Create__sexec }
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct bHYPRE_Pilut__smethod);
  if (methodName) {
    /* Use binary search to locate method */
    while (l < u) {
      i = (l + u) >> 1;
      if (!(cmp=strcmp(methodName, s_methods[i].d_name))) {
        (s_methods[i].d_func)(inArgs, outArgs);
        return;
      }
      else if (cmp < 0) u = i;
      else l = i + 1;
    }
  }
  /* TODO: add code for method not found */
}
/*
 * Get the URL of the Implementation of this object (for RMI)
 */

char*
bHYPRE_Pilut__getURL(
  /* in */ bHYPRE_Pilut self)
{
  return (*self->d_epv->f__getURL)(
  self);
}

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct bHYPRE_Pilut__array*)sidl_interface__array_createCol(dimen,
    lower, upper);
}

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct bHYPRE_Pilut__array*)sidl_interface__array_createRow(dimen,
    lower, upper);
}

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_create1d(int32_t len)
{
  return (struct bHYPRE_Pilut__array*)sidl_interface__array_create1d(len);
}

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_create1dInit(
  int32_t len, 
  bHYPRE_Pilut* data)
{
  return (struct bHYPRE_Pilut__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_create2dCol(int32_t m, int32_t n)
{
  return (struct bHYPRE_Pilut__array*)sidl_interface__array_create2dCol(m, n);
}

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_create2dRow(int32_t m, int32_t n)
{
  return (struct bHYPRE_Pilut__array*)sidl_interface__array_create2dRow(m, n);
}

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_borrow(
  bHYPRE_Pilut* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct bHYPRE_Pilut__array*)sidl_interface__array_borrow(
    (struct sidl_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_smartCopy(
  struct bHYPRE_Pilut__array *array)
{
  return (struct bHYPRE_Pilut__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

void
bHYPRE_Pilut__array_addRef(
  struct bHYPRE_Pilut__array* array)
{
  sidl_interface__array_addRef((struct sidl_interface__array *)array);
}

void
bHYPRE_Pilut__array_deleteRef(
  struct bHYPRE_Pilut__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

bHYPRE_Pilut
bHYPRE_Pilut__array_get1(
  const struct bHYPRE_Pilut__array* array,
  const int32_t i1)
{
  return (bHYPRE_Pilut)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

bHYPRE_Pilut
bHYPRE_Pilut__array_get2(
  const struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (bHYPRE_Pilut)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

bHYPRE_Pilut
bHYPRE_Pilut__array_get3(
  const struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (bHYPRE_Pilut)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

bHYPRE_Pilut
bHYPRE_Pilut__array_get4(
  const struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (bHYPRE_Pilut)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

bHYPRE_Pilut
bHYPRE_Pilut__array_get5(
  const struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (bHYPRE_Pilut)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

bHYPRE_Pilut
bHYPRE_Pilut__array_get6(
  const struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (bHYPRE_Pilut)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

bHYPRE_Pilut
bHYPRE_Pilut__array_get7(
  const struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (bHYPRE_Pilut)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

bHYPRE_Pilut
bHYPRE_Pilut__array_get(
  const struct bHYPRE_Pilut__array* array,
  const int32_t indices[])
{
  return (bHYPRE_Pilut)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

void
bHYPRE_Pilut__array_set1(
  struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  bHYPRE_Pilut const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_Pilut__array_set2(
  struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_Pilut const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_Pilut__array_set3(
  struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_Pilut const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_Pilut__array_set4(
  struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_Pilut const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_Pilut__array_set5(
  struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_Pilut const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_Pilut__array_set6(
  struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_Pilut const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_Pilut__array_set7(
  struct bHYPRE_Pilut__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_Pilut const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_Pilut__array_set(
  struct bHYPRE_Pilut__array* array,
  const int32_t indices[],
  bHYPRE_Pilut const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

int32_t
bHYPRE_Pilut__array_dimen(
  const struct bHYPRE_Pilut__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

int32_t
bHYPRE_Pilut__array_lower(
  const struct bHYPRE_Pilut__array* array,
  const int32_t ind)
{
  return sidl_interface__array_lower((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_Pilut__array_upper(
  const struct bHYPRE_Pilut__array* array,
  const int32_t ind)
{
  return sidl_interface__array_upper((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_Pilut__array_length(
  const struct bHYPRE_Pilut__array* array,
  const int32_t ind)
{
  return sidl_interface__array_length((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_Pilut__array_stride(
  const struct bHYPRE_Pilut__array* array,
  const int32_t ind)
{
  return sidl_interface__array_stride((struct sidl_interface__array *)array,
    ind);
}

int
bHYPRE_Pilut__array_isColumnOrder(
  const struct bHYPRE_Pilut__array* array)
{
  return sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)array);
}

int
bHYPRE_Pilut__array_isRowOrder(
  const struct bHYPRE_Pilut__array* array)
{
  return sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)array);
}

void
bHYPRE_Pilut__array_copy(
  const struct bHYPRE_Pilut__array* src,
  struct bHYPRE_Pilut__array* dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array *)src,
                             (struct sidl_interface__array *)dest);
}

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_slice(
  struct bHYPRE_Pilut__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct bHYPRE_Pilut__array*)
    sidl_interface__array_slice((struct sidl_interface__array *)src,
                                dimen, numElem, srcStart, srcStride, newStart);
}

struct bHYPRE_Pilut__array*
bHYPRE_Pilut__array_ensure(
  struct bHYPRE_Pilut__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct bHYPRE_Pilut__array*)
    sidl_interface__array_ensure((struct sidl_interface__array *)src, dimen,
      ordering);
}

#include <stdlib.h>
#include <string.h>
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#include "sidl_rmi_ProtocolFactory.h"
#include "sidl_rmi_Invocation.h"
#include "sidl_rmi_Response.h"

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t bHYPRE_Pilut__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &bHYPRE_Pilut__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &bHYPRE_Pilut__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &bHYPRE_Pilut__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

/* Static variables to hold version of IOR */
static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 9;

/* Static variables for managing EPV initialization. */
static int s_remote_initialized = 0;

static struct bHYPRE_Pilut__epv s_rem_epv__bhypre_pilut;

static struct bHYPRE_Operator__epv s_rem_epv__bhypre_operator;

static struct bHYPRE_Solver__epv s_rem_epv__bhypre_solver;

static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;

/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_bHYPRE_Pilut__cast(
struct bHYPRE_Pilut__object* self,
const char* name)
{
  void* cast = NULL;

  struct bHYPRE_Pilut__object* s0;
  struct sidl_BaseClass__object* s1;
   s0 =                         self;
   s1 =                         &s0->d_sidl_baseclass;

  if (!strcmp(name, "bHYPRE.Pilut")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "bHYPRE.Operator")) {
    cast = (void*) &s0->d_bhypre_operator;
  } else if (!strcmp(name, "bHYPRE.Solver")) {
    cast = (void*) &s0->d_bhypre_solver;
  } else if (!strcmp(name, "sidl.BaseClass")) {
    cast = (void*) s1;
  } else if (!strcmp(name, "sidl.BaseInterface")) {
    cast = (void*) &s1->d_sidl_baseinterface;
  }
  else if(bHYPRE_Pilut_isType(self, name)) {
    void* (*func)(sidl_rmi_InstanceHandle) = 
      (void* (*)(sidl_rmi_InstanceHandle)) 
      sidl_rmi_ConnectRegistry_getConnect(name);
    cast =  (*func)((sidl_rmi_InstanceHandle)self->d_data);
  }

  return cast;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_bHYPRE_Pilut__delete(
  struct bHYPRE_Pilut__object* self)
{
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_bHYPRE_Pilut__getURL(
  struct bHYPRE_Pilut__object* self)
{
  sidl_rmi_InstanceHandle conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_BaseInterface _ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getURL(conn, &_ex);
  }
  return NULL;
}

/* REMOTE EXEC: call the exec function for the object. */
static void remote_bHYPRE_Pilut__exec(
  struct bHYPRE_Pilut__object* self,
  const char* methodName,
  sidl_io_Deserializer inArgs,
  sidl_io_Serializer outArgs)
{
}

/* REMOTE METHOD STUB:addRef */
static void
remote_bHYPRE_Pilut_addRef(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */)
{
  /* FIXME  need to think through all of these special cases */
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_bHYPRE_Pilut_deleteRef(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "deleteRef", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:isSame */
static sidl_bool
remote_bHYPRE_Pilut_isSame(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ struct sidl_BaseInterface__object* iobj)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:queryInt */
static struct sidl_BaseInterface__object*
remote_bHYPRE_Pilut_queryInt(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ const char* name)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:isType */
static sidl_bool
remote_bHYPRE_Pilut_isType(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ const char* name)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "isType", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  sidl_bool _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:getClassInfo */
static struct sidl_ClassInfo__object*
remote_bHYPRE_Pilut_getClassInfo(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:SetCommunicator */
static int32_t
remote_bHYPRE_Pilut_SetCommunicator(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetCommunicator", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "mpi_comm",
    bHYPRE_MPICommunicator__getURL(mpi_comm), _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:SetIntParameter */
static int32_t
remote_bHYPRE_Pilut_SetIntParameter(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ const char* name,
  /* in */ int32_t value)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetIntParameter", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);
  sidl_rmi_Invocation_packInt( _inv, "value", value, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:SetDoubleParameter */
static int32_t
remote_bHYPRE_Pilut_SetDoubleParameter(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ const char* name,
  /* in */ double value)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetDoubleParameter", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);
  sidl_rmi_Invocation_packDouble( _inv, "value", value, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:SetStringParameter */
static int32_t
remote_bHYPRE_Pilut_SetStringParameter(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ const char* name,
  /* in */ const char* value)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetStringParameter", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);
  sidl_rmi_Invocation_packString( _inv, "value", value, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:SetIntArray1Parameter */
static int32_t
remote_bHYPRE_Pilut_SetIntArray1Parameter(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetIntArray1Parameter", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:SetIntArray2Parameter */
static int32_t
remote_bHYPRE_Pilut_SetIntArray2Parameter(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetIntArray2Parameter", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:SetDoubleArray1Parameter */
static int32_t
remote_bHYPRE_Pilut_SetDoubleArray1Parameter(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetDoubleArray1Parameter", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:SetDoubleArray2Parameter */
static int32_t
remote_bHYPRE_Pilut_SetDoubleArray2Parameter(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetDoubleArray2Parameter", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:GetIntValue */
static int32_t
remote_bHYPRE_Pilut_GetIntValue(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ const char* name,
  /* out */ int32_t* value)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "GetIntValue", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackInt( _rsvp, "value", value, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:GetDoubleValue */
static int32_t
remote_bHYPRE_Pilut_GetDoubleValue(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ const char* name,
  /* out */ double* value)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "GetDoubleValue", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackDouble( _rsvp, "value", value, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:Setup */
static int32_t
remote_bHYPRE_Pilut_Setup(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ struct bHYPRE_Vector__object* b,
  /* in */ struct bHYPRE_Vector__object* x)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "Setup", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:Apply */
static int32_t
remote_bHYPRE_Pilut_Apply(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ struct bHYPRE_Vector__object* b,
  /* inout */ struct bHYPRE_Vector__object** x)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "Apply", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackString( _rsvp, "x", x, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:ApplyAdjoint */
static int32_t
remote_bHYPRE_Pilut_ApplyAdjoint(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ struct bHYPRE_Vector__object* b,
  /* inout */ struct bHYPRE_Vector__object** x)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "ApplyAdjoint", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackString( _rsvp, "x", x, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:SetOperator */
static int32_t
remote_bHYPRE_Pilut_SetOperator(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ struct bHYPRE_Operator__object* A)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetOperator", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:SetTolerance */
static int32_t
remote_bHYPRE_Pilut_SetTolerance(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ double tolerance)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetTolerance", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packDouble( _inv, "tolerance", tolerance, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:SetMaxIterations */
static int32_t
remote_bHYPRE_Pilut_SetMaxIterations(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ int32_t max_iterations)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetMaxIterations", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "max_iterations", max_iterations, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:SetLogging */
static int32_t
remote_bHYPRE_Pilut_SetLogging(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ int32_t level)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetLogging", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "level", level, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:SetPrintLevel */
static int32_t
remote_bHYPRE_Pilut_SetPrintLevel(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* in */ int32_t level)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetPrintLevel", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "level", level, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:GetNumIterations */
static int32_t
remote_bHYPRE_Pilut_GetNumIterations(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* out */ int32_t* num_iterations)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "GetNumIterations", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackInt( _rsvp, "num_iterations", num_iterations, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:GetRelResidualNorm */
static int32_t
remote_bHYPRE_Pilut_GetRelResidualNorm(
  /* in */ struct bHYPRE_Pilut__object* self /* TLD */,
  /* out */ double* norm)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "GetRelResidualNorm", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackDouble( _rsvp, "norm", norm, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void bHYPRE_Pilut__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct bHYPRE_Pilut__epv*       epv = &s_rem_epv__bhypre_pilut;
  struct bHYPRE_Operator__epv*    e0  = &s_rem_epv__bhypre_operator;
  struct bHYPRE_Solver__epv*      e1  = &s_rem_epv__bhypre_solver;
  struct sidl_BaseClass__epv*     e2  = &s_rem_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv* e3  = &s_rem_epv__sidl_baseinterface;

  epv->f__cast                         = remote_bHYPRE_Pilut__cast;
  epv->f__delete                       = remote_bHYPRE_Pilut__delete;
  epv->f__exec                         = remote_bHYPRE_Pilut__exec;
  epv->f__getURL                       = remote_bHYPRE_Pilut__getURL;
  epv->f__ctor                         = NULL;
  epv->f__dtor                         = NULL;
  epv->f_addRef                        = remote_bHYPRE_Pilut_addRef;
  epv->f_deleteRef                     = remote_bHYPRE_Pilut_deleteRef;
  epv->f_isSame                        = remote_bHYPRE_Pilut_isSame;
  epv->f_queryInt                      = remote_bHYPRE_Pilut_queryInt;
  epv->f_isType                        = remote_bHYPRE_Pilut_isType;
  epv->f_getClassInfo                  = remote_bHYPRE_Pilut_getClassInfo;
  epv->f_SetCommunicator               = remote_bHYPRE_Pilut_SetCommunicator;
  epv->f_SetIntParameter               = remote_bHYPRE_Pilut_SetIntParameter;
  epv->f_SetDoubleParameter            = remote_bHYPRE_Pilut_SetDoubleParameter;
  epv->f_SetStringParameter            = remote_bHYPRE_Pilut_SetStringParameter;
  epv->f_SetIntArray1Parameter         = 
    remote_bHYPRE_Pilut_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter         = 
    remote_bHYPRE_Pilut_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter      = 
    remote_bHYPRE_Pilut_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter      = 
    remote_bHYPRE_Pilut_SetDoubleArray2Parameter;
  epv->f_GetIntValue                   = remote_bHYPRE_Pilut_GetIntValue;
  epv->f_GetDoubleValue                = remote_bHYPRE_Pilut_GetDoubleValue;
  epv->f_Setup                         = remote_bHYPRE_Pilut_Setup;
  epv->f_Apply                         = remote_bHYPRE_Pilut_Apply;
  epv->f_ApplyAdjoint                  = remote_bHYPRE_Pilut_ApplyAdjoint;
  epv->f_SetOperator                   = remote_bHYPRE_Pilut_SetOperator;
  epv->f_SetTolerance                  = remote_bHYPRE_Pilut_SetTolerance;
  epv->f_SetMaxIterations              = remote_bHYPRE_Pilut_SetMaxIterations;
  epv->f_SetLogging                    = remote_bHYPRE_Pilut_SetLogging;
  epv->f_SetPrintLevel                 = remote_bHYPRE_Pilut_SetPrintLevel;
  epv->f_GetNumIterations              = remote_bHYPRE_Pilut_GetNumIterations;
  epv->f_GetRelResidualNorm            = remote_bHYPRE_Pilut_GetRelResidualNorm;

  e0->f__cast                    = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete                  = (void (*)(void*)) epv->f__delete;
  e0->f__exec                    = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e0->f_addRef                   = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef                = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame                   = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(void*,const char*)) epv->f_queryInt;
  e0->f_isType                   = (sidl_bool (*)(void*,
    const char*)) epv->f_isType;
  e0->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e0->f_SetCommunicator          = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*)) epv->f_SetCommunicator;
  e0->f_SetIntParameter          = (int32_t (*)(void*,const char*,
    int32_t)) epv->f_SetIntParameter;
  e0->f_SetDoubleParameter       = (int32_t (*)(void*,const char*,
    double)) epv->f_SetDoubleParameter;
  e0->f_SetStringParameter       = (int32_t (*)(void*,const char*,
    const char*)) epv->f_SetStringParameter;
  e0->f_SetIntArray1Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray1Parameter;
  e0->f_SetIntArray2Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray2Parameter;
  e0->f_SetDoubleArray1Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray1Parameter;
  e0->f_SetDoubleArray2Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray2Parameter;
  e0->f_GetIntValue              = (int32_t (*)(void*,const char*,
    int32_t*)) epv->f_GetIntValue;
  e0->f_GetDoubleValue           = (int32_t (*)(void*,const char*,
    double*)) epv->f_GetDoubleValue;
  e0->f_Setup                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object*)) epv->f_Setup;
  e0->f_Apply                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object**)) epv->f_Apply;
  e0->f_ApplyAdjoint             = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,
    struct bHYPRE_Vector__object**)) epv->f_ApplyAdjoint;

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
  e1->f_SetOperator              = (int32_t (*)(void*,
    struct bHYPRE_Operator__object*)) epv->f_SetOperator;
  e1->f_SetTolerance             = (int32_t (*)(void*,
    double)) epv->f_SetTolerance;
  e1->f_SetMaxIterations         = (int32_t (*)(void*,
    int32_t)) epv->f_SetMaxIterations;
  e1->f_SetLogging               = (int32_t (*)(void*,
    int32_t)) epv->f_SetLogging;
  e1->f_SetPrintLevel            = (int32_t (*)(void*,
    int32_t)) epv->f_SetPrintLevel;
  e1->f_GetNumIterations         = (int32_t (*)(void*,
    int32_t*)) epv->f_GetNumIterations;
  e1->f_GetRelResidualNorm       = (int32_t (*)(void*,
    double*)) epv->f_GetRelResidualNorm;

  e2->f__cast        = (void* (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f__cast;
  e2->f__delete      = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f__delete;
  e2->f__exec        = (void (*)(struct sidl_BaseClass__object*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e2->f_addRef       = (void (*)(struct sidl_BaseClass__object*)) epv->f_addRef;
  e2->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_deleteRef;
  e2->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt     = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_BaseClass__object*,const char*)) epv->f_queryInt;
  e2->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f_isType;
  e2->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*)) epv->f_getClassInfo;

  e3->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete      = (void (*)(void*)) epv->f__delete;
  e3->f__exec        = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e3->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e3->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e3->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct bHYPRE_Pilut__object*
bHYPRE_Pilut__remoteConnect(const char *url, sidl_BaseInterface *_ex)
{
  struct bHYPRE_Pilut__object* self;

  struct bHYPRE_Pilut__object* s0;
  struct sidl_BaseClass__object* s1;

  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_connectInstance(url, _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_Pilut__object*) malloc(
      sizeof(struct bHYPRE_Pilut__object));

   s0 =                         self;
   s1 =                         &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_Pilut__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) instance;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_bhypre_operator.d_epv    = &s_rem_epv__bhypre_operator;
  s0->d_bhypre_operator.d_object = (void*) self;

  s0->d_bhypre_solver.d_epv    = &s_rem_epv__bhypre_solver;
  s0->d_bhypre_solver.d_object = (void*) self;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__bhypre_pilut;

  self->d_data = (void*) instance;
  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_Pilut__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;


  return self;
}
/* Create an instance that uses an already existing InstanceHandel to connect 
  to an existing remote object. */
static struct bHYPRE_Pilut__object*
bHYPRE_Pilut__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE_Pilut__object* self;

  struct bHYPRE_Pilut__object* s0;
  struct sidl_BaseClass__object* s1;

  self =
    (struct bHYPRE_Pilut__object*) malloc(
      sizeof(struct bHYPRE_Pilut__object));

   s0 =                         self;
   s1 =                         &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_Pilut__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) instance;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_bhypre_operator.d_epv    = &s_rem_epv__bhypre_operator;
  s0->d_bhypre_operator.d_object = (void*) self;

  s0->d_bhypre_solver.d_epv    = &s_rem_epv__bhypre_solver;
  s0->d_bhypre_solver.d_object = (void*) self;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__bhypre_pilut;

  self->d_data = (void*) instance;

  sidl_rmi_InstanceHandle_addRef(instance);
  return self;
}
/* REMOTE: generate remote instance given URL string. */
static struct bHYPRE_Pilut__object*
bHYPRE_Pilut__remote(const char *url, sidl_BaseInterface *_ex)
{
  struct bHYPRE_Pilut__object* self;

  struct bHYPRE_Pilut__object* s0;
  struct sidl_BaseClass__object* s1;

  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_createInstance(url, "bHYPRE.Pilut", _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_Pilut__object*) malloc(
      sizeof(struct bHYPRE_Pilut__object));

   s0 =                         self;
   s1 =                         &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_Pilut__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) instance;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_bhypre_operator.d_epv    = &s_rem_epv__bhypre_operator;
  s0->d_bhypre_operator.d_object = (void*) self;

  s0->d_bhypre_solver.d_epv    = &s_rem_epv__bhypre_solver;
  s0->d_bhypre_solver.d_object = (void*) self;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__bhypre_pilut;

  self->d_data = (void*) instance;

  return self;
}
