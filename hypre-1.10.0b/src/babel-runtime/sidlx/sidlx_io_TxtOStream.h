/*
 * File:          sidlx_io_TxtOStream.h
 * Symbol:        sidlx.io.TxtOStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Client-side glue code for sidlx.io.TxtOStream
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_sidlx_io_TxtOStream_h
#define included_sidlx_io_TxtOStream_h

/**
 * Symbol "sidlx.io.TxtOStream" (version 0.1)
 * 
 * Simple text-based output stream appends spaces
 */
struct sidlx_io_TxtOStream__object;
struct sidlx_io_TxtOStream__array;
typedef struct sidlx_io_TxtOStream__object* sidlx_io_TxtOStream;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseException_h
#include "sidl_BaseException.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_SIDLException_h
#include "sidl_SIDLException.h"
#endif
#ifndef included_sidlx_io_IOException_h
#include "sidlx_io_IOException.h"
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
struct sidlx_io_TxtOStream__object*
sidlx_io_TxtOStream__create(void);

/**
 * RMI constructor function for the class.
 */
sidlx_io_TxtOStream
sidlx_io_TxtOStream__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
sidlx_io_TxtOStream
sidlx_io_TxtOStream__connect(const char *, sidl_BaseInterface *_ex);
/**
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
sidlx_io_TxtOStream_addRef(
  /* in */ sidlx_io_TxtOStream self);

/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
sidlx_io_TxtOStream_deleteRef(
  /* in */ sidlx_io_TxtOStream self);

/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
sidl_bool
sidlx_io_TxtOStream_isSame(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ sidl_BaseInterface iobj);

/**
 * Check whether the object can support the specified interface or
 * class.  If the <code>sidl</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */
sidl_BaseInterface
sidlx_io_TxtOStream_queryInt(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ const char* name);

/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
sidl_bool
sidlx_io_TxtOStream_isType(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ const char* name);

/**
 * Return the meta-data about the class implementing this interface.
 */
sidl_ClassInfo
sidlx_io_TxtOStream_getClassInfo(
  /* in */ sidlx_io_TxtOStream self);

/**
 * Method:  setFD[]
 */
void
sidlx_io_TxtOStream_setFD(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ int32_t fd);

/**
 * flushes the buffer, if any 
 */
void
sidlx_io_TxtOStream_flush(
  /* in */ sidlx_io_TxtOStream self);

/**
 * low level write for an array of bytes 
 */
int32_t
sidlx_io_TxtOStream_write(
  /* in */ sidlx_io_TxtOStream self,
  /* in array<char,row-major> */ struct sidl_char__array* data,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Method:  put[Bool]
 */
void
sidlx_io_TxtOStream_putBool(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ sidl_bool item,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Method:  put[Char]
 */
void
sidlx_io_TxtOStream_putChar(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ char item,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Method:  put[Int]
 */
void
sidlx_io_TxtOStream_putInt(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ int32_t item,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Method:  put[Long]
 */
void
sidlx_io_TxtOStream_putLong(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ int64_t item,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Method:  put[Float]
 */
void
sidlx_io_TxtOStream_putFloat(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ float item,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Method:  put[Double]
 */
void
sidlx_io_TxtOStream_putDouble(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ double item,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Method:  put[Fcomplex]
 */
void
sidlx_io_TxtOStream_putFcomplex(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ struct sidl_fcomplex item,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Method:  put[Dcomplex]
 */
void
sidlx_io_TxtOStream_putDcomplex(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ struct sidl_dcomplex item,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Method:  put[String]
 */
void
sidlx_io_TxtOStream_putString(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ const char* item,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Cast method for interface and class type conversions.
 */
struct sidlx_io_TxtOStream__object*
sidlx_io_TxtOStream__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
sidlx_io_TxtOStream__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
sidlx_io_TxtOStream__exec(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
sidlx_io_TxtOStream__getURL(
  /* in */ sidlx_io_TxtOStream self);
/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in column-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_io_TxtOStream__array*
sidlx_io_TxtOStream__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_io_TxtOStream__array*
sidlx_io_TxtOStream__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Create a contiguous one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_io_TxtOStream__array*
sidlx_io_TxtOStream__array_create1d(int32_t len);

/**
 * Create a dense one-dimensional vector with a lower
 * index of 0 and an upper index of len-1. The initial data for this
 * new array is copied from data. This will increment the reference
 * count of each non-NULL object/interface reference in data.
 * 
 * This array owns and manages its data.
 */
struct sidlx_io_TxtOStream__array*
sidlx_io_TxtOStream__array_create1dInit(
  int32_t len, 
  sidlx_io_TxtOStream* data);

/**
 * Create a contiguous two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_io_TxtOStream__array*
sidlx_io_TxtOStream__array_create2dCol(int32_t m, int32_t n);

/**
 * Create a contiguous two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_io_TxtOStream__array*
sidlx_io_TxtOStream__array_create2dRow(int32_t m, int32_t n);

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteRef will be called on the
 * value being replaced if it is not NULL.
 */
struct sidlx_io_TxtOStream__array*
sidlx_io_TxtOStream__array_borrow(
  sidlx_io_TxtOStream* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

/**
 * If array is borrowed, allocate a new self-sufficient
 * array and copy the borrowed array into the new array;
 * otherwise, increment the reference count and return
 * the array passed in. Use this whenever you want to
 * make a copy of a method argument because arrays
 * passed into methods aren't guaranteed to exist after
 * the method call.
 */
struct sidlx_io_TxtOStream__array*
sidlx_io_TxtOStream__array_smartCopy(
  struct sidlx_io_TxtOStream__array *array);

/**
 * Increment the array's internal reference count by one.
 */
void
sidlx_io_TxtOStream__array_addRef(
  struct sidlx_io_TxtOStream__array* array);

/**
 * Decrement the array's internal reference count by one.
 * If the reference count goes to zero, destroy the array.
 * If the array isn't borrowed, this releases all the
 * object references held by the array.
 */
void
sidlx_io_TxtOStream__array_deleteRef(
  struct sidlx_io_TxtOStream__array* array);

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
sidlx_io_TxtOStream
sidlx_io_TxtOStream__array_get1(
  const struct sidlx_io_TxtOStream__array* array,
  const int32_t i1);

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
sidlx_io_TxtOStream
sidlx_io_TxtOStream__array_get2(
  const struct sidlx_io_TxtOStream__array* array,
  const int32_t i1,
  const int32_t i2);

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
sidlx_io_TxtOStream
sidlx_io_TxtOStream__array_get3(
  const struct sidlx_io_TxtOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
sidlx_io_TxtOStream
sidlx_io_TxtOStream__array_get4(
  const struct sidlx_io_TxtOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

/**
 * Retrieve element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array.
 */
sidlx_io_TxtOStream
sidlx_io_TxtOStream__array_get5(
  const struct sidlx_io_TxtOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array.
 */
sidlx_io_TxtOStream
sidlx_io_TxtOStream__array_get6(
  const struct sidlx_io_TxtOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array.
 */
sidlx_io_TxtOStream
sidlx_io_TxtOStream__array_get7(
  const struct sidlx_io_TxtOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
sidlx_io_TxtOStream
sidlx_io_TxtOStream__array_get(
  const struct sidlx_io_TxtOStream__array* array,
  const int32_t indices[]);

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
sidlx_io_TxtOStream__array_set1(
  struct sidlx_io_TxtOStream__array* array,
  const int32_t i1,
  sidlx_io_TxtOStream const value);

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
sidlx_io_TxtOStream__array_set2(
  struct sidlx_io_TxtOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  sidlx_io_TxtOStream const value);

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
sidlx_io_TxtOStream__array_set3(
  struct sidlx_io_TxtOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  sidlx_io_TxtOStream const value);

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
sidlx_io_TxtOStream__array_set4(
  struct sidlx_io_TxtOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  sidlx_io_TxtOStream const value);

/**
 * Set element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array to value.
 */
void
sidlx_io_TxtOStream__array_set5(
  struct sidlx_io_TxtOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  sidlx_io_TxtOStream const value);

/**
 * Set element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array to value.
 */
void
sidlx_io_TxtOStream__array_set6(
  struct sidlx_io_TxtOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  sidlx_io_TxtOStream const value);

/**
 * Set element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array to value.
 */
void
sidlx_io_TxtOStream__array_set7(
  struct sidlx_io_TxtOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  sidlx_io_TxtOStream const value);

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
sidlx_io_TxtOStream__array_set(
  struct sidlx_io_TxtOStream__array* array,
  const int32_t indices[],
  sidlx_io_TxtOStream const value);

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
sidlx_io_TxtOStream__array_dimen(
  const struct sidlx_io_TxtOStream__array* array);

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidlx_io_TxtOStream__array_lower(
  const struct sidlx_io_TxtOStream__array* array,
  const int32_t ind);

/**
 * Return the upper bound of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidlx_io_TxtOStream__array_upper(
  const struct sidlx_io_TxtOStream__array* array,
  const int32_t ind);

/**
 * Return the length of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidlx_io_TxtOStream__array_length(
  const struct sidlx_io_TxtOStream__array* array,
  const int32_t ind);

/**
 * Return the stride of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidlx_io_TxtOStream__array_stride(
  const struct sidlx_io_TxtOStream__array* array,
  const int32_t ind);

/**
 * Return a true value iff the array is a contiguous
 * column-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
sidlx_io_TxtOStream__array_isColumnOrder(
  const struct sidlx_io_TxtOStream__array* array);

/**
 * Return a true value iff the array is a contiguous
 * row-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
sidlx_io_TxtOStream__array_isRowOrder(
  const struct sidlx_io_TxtOStream__array* array);

/**
 * Create a sub-array of another array. This resulting
 * array shares data with the original array. The new
 * array can be of the same dimension or potentially
 * less assuming the original array has dimension
 * greater than 1.  If you are removing dimension,
 * indicate the dimensions to remove by setting
 * numElem[i] to zero for any dimension i wthat should
 * go away in the new array.  The meaning of each
 * argument is covered below.
 * 
 * src       the array to be created will be a subset
 *           of this array. If this argument is NULL,
 *           NULL will be returned. The array returned
 *           borrows data from src, so modifying src or
 *           the returned array will modify both
 *           arrays.
 * 
 * dimen     this argument must be greater than zero
 *           and less than or equal to the dimension of
 *           src. An illegal value will cause a NULL
 *           return value.
 * 
 * numElem   this specifies how many elements from src
 *           should be taken in each dimension. A zero
 *           entry indicates that the dimension should
 *           not appear in the new array.  This
 *           argument should be an array with an entry
 *           for each dimension of src.  Passing NULL
 *           here will cause NULL to be returned.  If
 *           srcStart[i] + numElem[i]*srcStride[i] is
 *           greater than upper[i] for src or if
 *           srcStart[i] + numElem[i]*srcStride[i] is
 *           less than lower[i] for src, NULL will be
 *           returned.
 * 
 * srcStart  this array holds the coordinates of the
 *           first element of the new array. If this
 *           argument is NULL, the first element of src
 *           will be the first element of the new
 *           array. If non-NULL, this argument should
 *           be an array with an entry for each
 *           dimension of src.  If srcStart[i] is less
 *           than lower[i] for the array src, NULL will
 *           be returned.
 * 
 * srcStride this array lets you specify the stride
 *           between elements in each dimension of
 *           src. This stride is relative to the
 *           coordinate system of the src array. If
 *           this argument is NULL, the stride is taken
 *           to be one in each dimension.  If non-NULL,
 *           this argument should be an array with an
 *           entry for each dimension of src.
 * 
 * newLower  this argument is like lower in a create
 *           method. It sets the coordinates for the
 *           first element in the new array.  If this
 *           argument is NULL, the values indicated by
 *           srcStart will be used. If non-NULL, this
 *           should be an array with dimen elements.
 */
struct sidlx_io_TxtOStream__array*
sidlx_io_TxtOStream__array_slice(
  struct sidlx_io_TxtOStream__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

/**
 * Copy the contents of one array (src) to a second array
 * (dest). For the copy to take place, both arrays must
 * exist and be of the same dimension. This method will
 * not modify dest's size, index bounds, or stride; only
 * the array element values of dest may be changed by
 * this function. No part of src is ever changed by copy.
 * 
 * On exit, dest[i][j][k]... = src[i][j][k]... for all
 * indices i,j,k...  that are in both arrays. If dest and
 * src have no indices in common, nothing is copied. For
 * example, if src is a 1-d array with elements 0-5 and
 * dest is a 1-d array with elements 2-3, this function
 * will make the following assignments:
 *   dest[2] = src[2],
 *   dest[3] = src[3].
 * The function copied the elements that both arrays have
 * in common.  If dest had elements 4-10, this function
 * will make the following assignments:
 *   dest[4] = src[4],
 *   dest[5] = src[5].
 */
void
sidlx_io_TxtOStream__array_copy(
  const struct sidlx_io_TxtOStream__array* src,
  struct sidlx_io_TxtOStream__array* dest);

/**
 * If necessary, convert a general matrix into a matrix
 * with the required properties. This checks the
 * dimension and ordering of the matrix.  If both these
 * match, it simply returns a new reference to the
 * existing matrix. If the dimension of the incoming
 * array doesn't match, it returns NULL. If the ordering
 * of the incoming array doesn't match the specification,
 * a new array is created with the desired ordering and
 * the content of the incoming array is copied to the new
 * array.
 * 
 * The ordering parameter should be one of the constants
 * defined in enum sidl_array_ordering
 * (e.g. sidl_general_order, sidl_column_major_order, or
 * sidl_row_major_order). If you specify
 * sidl_general_order, this routine will only check the
 * dimension because any matrix is sidl_general_order.
 * 
 * The caller assumes ownership of the returned reference
 * unless it's NULL.
 */
struct sidlx_io_TxtOStream__array*
sidlx_io_TxtOStream__array_ensure(
  struct sidlx_io_TxtOStream__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
