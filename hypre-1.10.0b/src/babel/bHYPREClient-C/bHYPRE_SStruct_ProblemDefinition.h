/*
 * File:          bHYPRE_SStruct_ProblemDefinition.h
 * Symbol:        bHYPRE.SStruct_ProblemDefinition-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.8
 * Description:   Client-side glue code for bHYPRE.SStruct_ProblemDefinition
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#ifndef included_bHYPRE_SStruct_ProblemDefinition_h
#define included_bHYPRE_SStruct_ProblemDefinition_h

/**
 * Symbol "bHYPRE.SStruct_ProblemDefinition" (version 1.0.0)
 */
struct bHYPRE_SStruct_ProblemDefinition__object;
struct bHYPRE_SStruct_ProblemDefinition__array;
typedef struct bHYPRE_SStruct_ProblemDefinition__object* 
  bHYPRE_SStruct_ProblemDefinition;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
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
 * RMI connector function for the class.
 */
bHYPRE_SStruct_ProblemDefinition
bHYPRE_SStruct_ProblemDefinition__connect(const char *,
  sidl_BaseInterface *_ex);
void
bHYPRE_SStruct_ProblemDefinition_addRef(
  /* in */ bHYPRE_SStruct_ProblemDefinition self);

void
bHYPRE_SStruct_ProblemDefinition_deleteRef(
  /* in */ bHYPRE_SStruct_ProblemDefinition self);

sidl_bool
bHYPRE_SStruct_ProblemDefinition_isSame(
  /* in */ bHYPRE_SStruct_ProblemDefinition self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_SStruct_ProblemDefinition_queryInt(
  /* in */ bHYPRE_SStruct_ProblemDefinition self,
  /* in */ const char* name);

sidl_bool
bHYPRE_SStruct_ProblemDefinition_isType(
  /* in */ bHYPRE_SStruct_ProblemDefinition self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_SStruct_ProblemDefinition_getClassInfo(
  /* in */ bHYPRE_SStruct_ProblemDefinition self);

int32_t
bHYPRE_SStruct_ProblemDefinition_SetCommunicator(
  /* in */ bHYPRE_SStruct_ProblemDefinition self,
  /* in */ void* mpi_comm);

int32_t
bHYPRE_SStruct_ProblemDefinition_Initialize(
  /* in */ bHYPRE_SStruct_ProblemDefinition self);

int32_t
bHYPRE_SStruct_ProblemDefinition_Assemble(
  /* in */ bHYPRE_SStruct_ProblemDefinition self);

/**
 *  A semi-structured matrix or vector contains a Struct or IJ matrix
 *  or vector.  GetObject returns it.
 * The returned type is a sidl.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */
int32_t
bHYPRE_SStruct_ProblemDefinition_GetObject(
  /* in */ bHYPRE_SStruct_ProblemDefinition self,
  /* out */ sidl_BaseInterface* A);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStruct_ProblemDefinition__object*
bHYPRE_SStruct_ProblemDefinition__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStruct_ProblemDefinition__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_SStruct_ProblemDefinition__exec(
  /* in */ bHYPRE_SStruct_ProblemDefinition self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_SStruct_ProblemDefinition__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_SStruct_ProblemDefinition__getURL(
  /* in */ bHYPRE_SStruct_ProblemDefinition self);
struct bHYPRE_SStruct_ProblemDefinition__array*
bHYPRE_SStruct_ProblemDefinition__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStruct_ProblemDefinition__array*
bHYPRE_SStruct_ProblemDefinition__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStruct_ProblemDefinition__array*
bHYPRE_SStruct_ProblemDefinition__array_create1d(int32_t len);

struct bHYPRE_SStruct_ProblemDefinition__array*
bHYPRE_SStruct_ProblemDefinition__array_create1dInit(
  int32_t len, 
  bHYPRE_SStruct_ProblemDefinition* data);

struct bHYPRE_SStruct_ProblemDefinition__array*
bHYPRE_SStruct_ProblemDefinition__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStruct_ProblemDefinition__array*
bHYPRE_SStruct_ProblemDefinition__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStruct_ProblemDefinition__array*
bHYPRE_SStruct_ProblemDefinition__array_borrow(
  bHYPRE_SStruct_ProblemDefinition* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_SStruct_ProblemDefinition__array*
bHYPRE_SStruct_ProblemDefinition__array_smartCopy(
  struct bHYPRE_SStruct_ProblemDefinition__array *array);

void
bHYPRE_SStruct_ProblemDefinition__array_addRef(
  struct bHYPRE_SStruct_ProblemDefinition__array* array);

void
bHYPRE_SStruct_ProblemDefinition__array_deleteRef(
  struct bHYPRE_SStruct_ProblemDefinition__array* array);

bHYPRE_SStruct_ProblemDefinition
bHYPRE_SStruct_ProblemDefinition__array_get1(
  const struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t i1);

bHYPRE_SStruct_ProblemDefinition
bHYPRE_SStruct_ProblemDefinition__array_get2(
  const struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_SStruct_ProblemDefinition
bHYPRE_SStruct_ProblemDefinition__array_get3(
  const struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_SStruct_ProblemDefinition
bHYPRE_SStruct_ProblemDefinition__array_get4(
  const struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_SStruct_ProblemDefinition
bHYPRE_SStruct_ProblemDefinition__array_get5(
  const struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_SStruct_ProblemDefinition
bHYPRE_SStruct_ProblemDefinition__array_get6(
  const struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_SStruct_ProblemDefinition
bHYPRE_SStruct_ProblemDefinition__array_get7(
  const struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_SStruct_ProblemDefinition
bHYPRE_SStruct_ProblemDefinition__array_get(
  const struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t indices[]);

void
bHYPRE_SStruct_ProblemDefinition__array_set1(
  struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t i1,
  bHYPRE_SStruct_ProblemDefinition const value);

void
bHYPRE_SStruct_ProblemDefinition__array_set2(
  struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStruct_ProblemDefinition const value);

void
bHYPRE_SStruct_ProblemDefinition__array_set3(
  struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStruct_ProblemDefinition const value);

void
bHYPRE_SStruct_ProblemDefinition__array_set4(
  struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStruct_ProblemDefinition const value);

void
bHYPRE_SStruct_ProblemDefinition__array_set5(
  struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStruct_ProblemDefinition const value);

void
bHYPRE_SStruct_ProblemDefinition__array_set6(
  struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStruct_ProblemDefinition const value);

void
bHYPRE_SStruct_ProblemDefinition__array_set7(
  struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStruct_ProblemDefinition const value);

void
bHYPRE_SStruct_ProblemDefinition__array_set(
  struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t indices[],
  bHYPRE_SStruct_ProblemDefinition const value);

int32_t
bHYPRE_SStruct_ProblemDefinition__array_dimen(
  const struct bHYPRE_SStruct_ProblemDefinition__array* array);

int32_t
bHYPRE_SStruct_ProblemDefinition__array_lower(
  const struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStruct_ProblemDefinition__array_upper(
  const struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStruct_ProblemDefinition__array_length(
  const struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStruct_ProblemDefinition__array_stride(
  const struct bHYPRE_SStruct_ProblemDefinition__array* array,
  const int32_t ind);

int
bHYPRE_SStruct_ProblemDefinition__array_isColumnOrder(
  const struct bHYPRE_SStruct_ProblemDefinition__array* array);

int
bHYPRE_SStruct_ProblemDefinition__array_isRowOrder(
  const struct bHYPRE_SStruct_ProblemDefinition__array* array);

struct bHYPRE_SStruct_ProblemDefinition__array*
bHYPRE_SStruct_ProblemDefinition__array_slice(
  struct bHYPRE_SStruct_ProblemDefinition__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_SStruct_ProblemDefinition__array_copy(
  const struct bHYPRE_SStruct_ProblemDefinition__array* src,
  struct bHYPRE_SStruct_ProblemDefinition__array* dest);

struct bHYPRE_SStruct_ProblemDefinition__array*
bHYPRE_SStruct_ProblemDefinition__array_ensure(
  struct bHYPRE_SStruct_ProblemDefinition__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
