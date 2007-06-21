/*
 * File:          sidl_io_Deserializer_Stub.c
 * Symbol:        sidl.io.Deserializer-v0.9.3
 * Symbol Type:   interface
 * Babel Version: 0.10.12
 * Release:       $Name: V1-10-0b $
 * Revision:      @(#) $Id: sidl_io_Deserializer_Stub.c,v 1.4 2005/11/14 21:20:25 painter Exp $
 * Description:   Client-side glue code for sidl.io.Deserializer
 * 
 * Copyright (c) 2000-2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
 * All rights reserved.
 * 
 * This file is part of Babel. For more information, see
 * http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
 * for Our Notice and the LICENSE file for the GNU Lesser General Public
 * License.
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License (as published by
 * the Free Software Foundation) version 2.1 dated February 1999.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
 * conditions of the GNU Lesser General Public License for more details.
 * 
 * You should have recieved a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#include "sidl_io_Deserializer.h"
#include "sidl_io_Deserializer_IOR.h"
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

/*
 * connect_loaded is a boolean value showing if the IHConnect for this object has been loaded into the connectRegistry
 */

static int connect_loaded = 0;

static struct sidl_io_Deserializer__object* 
  sidl_io_Deserializer__remoteConnect(const char* url, sidl_BaseInterface *_ex);
static struct sidl_io_Deserializer__object* 
  sidl_io_Deserializer__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex);
/*
 * RMI connector function for the class.
 */

sidl_io_Deserializer
sidl_io_Deserializer__connect(const char* url, sidl_BaseInterface *_ex)
{
  return sidl_io_Deserializer__remoteConnect(url, _ex);
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
sidl_io_Deserializer_addRef(
  /* in */ sidl_io_Deserializer self)
{
  (*self->d_epv->f_addRef)(
    self->d_object);
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
sidl_io_Deserializer_deleteRef(
  /* in */ sidl_io_Deserializer self)
{
  (*self->d_epv->f_deleteRef)(
    self->d_object);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

sidl_bool
sidl_io_Deserializer_isSame(
  /* in */ sidl_io_Deserializer self,
  /* in */ sidl_BaseInterface iobj)
{
  return (*self->d_epv->f_isSame)(
    self->d_object,
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
sidl_io_Deserializer_queryInt(
  /* in */ sidl_io_Deserializer self,
  /* in */ const char* name)
{
  return (*self->d_epv->f_queryInt)(
    self->d_object,
    name);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

sidl_bool
sidl_io_Deserializer_isType(
  /* in */ sidl_io_Deserializer self,
  /* in */ const char* name)
{
  return (*self->d_epv->f_isType)(
    self->d_object,
    name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

sidl_ClassInfo
sidl_io_Deserializer_getClassInfo(
  /* in */ sidl_io_Deserializer self)
{
  return (*self->d_epv->f_getClassInfo)(
    self->d_object);
}

/*
 * Method:  unpackBool[]
 */

void
sidl_io_Deserializer_unpackBool(
  /* in */ sidl_io_Deserializer self,
  /* in */ const char* key,
  /* out */ sidl_bool* value,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_unpackBool)(
    self->d_object,
    key,
    value,
    _ex);
}

/*
 * Method:  unpackChar[]
 */

void
sidl_io_Deserializer_unpackChar(
  /* in */ sidl_io_Deserializer self,
  /* in */ const char* key,
  /* out */ char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_unpackChar)(
    self->d_object,
    key,
    value,
    _ex);
}

/*
 * Method:  unpackInt[]
 */

void
sidl_io_Deserializer_unpackInt(
  /* in */ sidl_io_Deserializer self,
  /* in */ const char* key,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_unpackInt)(
    self->d_object,
    key,
    value,
    _ex);
}

/*
 * Method:  unpackLong[]
 */

void
sidl_io_Deserializer_unpackLong(
  /* in */ sidl_io_Deserializer self,
  /* in */ const char* key,
  /* out */ int64_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_unpackLong)(
    self->d_object,
    key,
    value,
    _ex);
}

/*
 * Method:  unpackFloat[]
 */

void
sidl_io_Deserializer_unpackFloat(
  /* in */ sidl_io_Deserializer self,
  /* in */ const char* key,
  /* out */ float* value,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_unpackFloat)(
    self->d_object,
    key,
    value,
    _ex);
}

/*
 * Method:  unpackDouble[]
 */

void
sidl_io_Deserializer_unpackDouble(
  /* in */ sidl_io_Deserializer self,
  /* in */ const char* key,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_unpackDouble)(
    self->d_object,
    key,
    value,
    _ex);
}

/*
 * Method:  unpackFcomplex[]
 */

void
sidl_io_Deserializer_unpackFcomplex(
  /* in */ sidl_io_Deserializer self,
  /* in */ const char* key,
  /* out */ struct sidl_fcomplex* value,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_unpackFcomplex)(
    self->d_object,
    key,
    value,
    _ex);
}

/*
 * Method:  unpackDcomplex[]
 */

void
sidl_io_Deserializer_unpackDcomplex(
  /* in */ sidl_io_Deserializer self,
  /* in */ const char* key,
  /* out */ struct sidl_dcomplex* value,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_unpackDcomplex)(
    self->d_object,
    key,
    value,
    _ex);
}

/*
 * Method:  unpackString[]
 */

void
sidl_io_Deserializer_unpackString(
  /* in */ sidl_io_Deserializer self,
  /* in */ const char* key,
  /* out */ char** value,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_unpackString)(
    self->d_object,
    key,
    value,
    _ex);
}

/*
 * Cast method for interface and class type conversions.
 */

sidl_io_Deserializer
sidl_io_Deserializer__cast(
  void* obj)
{
  sidl_io_Deserializer cast = NULL;

  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("sidl.io.Deserializer",
      (void*)sidl_io_Deserializer__IHConnect);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (sidl_io_Deserializer) (*base->d_epv->f__cast)(
      base->d_object,
      "sidl.io.Deserializer");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
sidl_io_Deserializer__cast2(
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
sidl_io_Deserializer__exec(
  /* in */ sidl_io_Deserializer self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs)
{
  (*self->d_epv->f__exec)(
  self->d_object,
  methodName,
  inArgs,
  outArgs);
}

/*
 * Get the URL of the Implementation of this object (for RMI)
 */

char*
sidl_io_Deserializer__getURL(
  /* in */ sidl_io_Deserializer self)
{
  return (*self->d_epv->f__getURL)(
  self->d_object);
}

/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in column-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_io_Deserializer__array*
sidl_io_Deserializer__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    sidl_io_Deserializer__array*)sidl_interface__array_createCol(dimen, lower,
    upper);
}

/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_io_Deserializer__array*
sidl_io_Deserializer__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    sidl_io_Deserializer__array*)sidl_interface__array_createRow(dimen, lower,
    upper);
}

/**
 * Create a contiguous one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_io_Deserializer__array*
sidl_io_Deserializer__array_create1d(int32_t len)
{
  return (struct 
    sidl_io_Deserializer__array*)sidl_interface__array_create1d(len);
}

/**
 * Create a dense one-dimensional vector with a lower
 * index of 0 and an upper index of len-1. The initial data for this
 * new array is copied from data. This will increment the reference
 * count of each non-NULL object/interface reference in data.
 * 
 * This array owns and manages its data.
 */
struct sidl_io_Deserializer__array*
sidl_io_Deserializer__array_create1dInit(
  int32_t len, 
  sidl_io_Deserializer* data)
{
  return (struct 
    sidl_io_Deserializer__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

/**
 * Create a contiguous two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_io_Deserializer__array*
sidl_io_Deserializer__array_create2dCol(int32_t m, int32_t n)
{
  return (struct 
    sidl_io_Deserializer__array*)sidl_interface__array_create2dCol(m, n);
}

/**
 * Create a contiguous two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_io_Deserializer__array*
sidl_io_Deserializer__array_create2dRow(int32_t m, int32_t n)
{
  return (struct 
    sidl_io_Deserializer__array*)sidl_interface__array_create2dRow(m, n);
}

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteRef will be called on the
 * value being replaced if it is not NULL.
 */
struct sidl_io_Deserializer__array*
sidl_io_Deserializer__array_borrow(
  sidl_io_Deserializer* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct sidl_io_Deserializer__array*)sidl_interface__array_borrow(
    (struct sidl_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

/**
 * If array is borrowed, allocate a new self-sufficient
 * array and copy the borrowed array into the new array;
 * otherwise, increment the reference count and return
 * the array passed in. Use this whenever you want to
 * make a copy of a method argument because arrays
 * passed into methods aren't guaranteed to exist after
 * the method call.
 */
struct sidl_io_Deserializer__array*
sidl_io_Deserializer__array_smartCopy(
  struct sidl_io_Deserializer__array *array)
{
  return (struct sidl_io_Deserializer__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

/**
 * Increment the array's internal reference count by one.
 */
void
sidl_io_Deserializer__array_addRef(
  struct sidl_io_Deserializer__array* array)
{
  sidl_interface__array_addRef((struct sidl_interface__array *)array);
}

/**
 * Decrement the array's internal reference count by one.
 * If the reference count goes to zero, destroy the array.
 * If the array isn't borrowed, this releases all the
 * object references held by the array.
 */
void
sidl_io_Deserializer__array_deleteRef(
  struct sidl_io_Deserializer__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
sidl_io_Deserializer
sidl_io_Deserializer__array_get1(
  const struct sidl_io_Deserializer__array* array,
  const int32_t i1)
{
  return (sidl_io_Deserializer)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
sidl_io_Deserializer
sidl_io_Deserializer__array_get2(
  const struct sidl_io_Deserializer__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (sidl_io_Deserializer)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
sidl_io_Deserializer
sidl_io_Deserializer__array_get3(
  const struct sidl_io_Deserializer__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (sidl_io_Deserializer)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
sidl_io_Deserializer
sidl_io_Deserializer__array_get4(
  const struct sidl_io_Deserializer__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (sidl_io_Deserializer)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

/**
 * Retrieve element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array.
 */
sidl_io_Deserializer
sidl_io_Deserializer__array_get5(
  const struct sidl_io_Deserializer__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (sidl_io_Deserializer)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array.
 */
sidl_io_Deserializer
sidl_io_Deserializer__array_get6(
  const struct sidl_io_Deserializer__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (sidl_io_Deserializer)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array.
 */
sidl_io_Deserializer
sidl_io_Deserializer__array_get7(
  const struct sidl_io_Deserializer__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (sidl_io_Deserializer)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
sidl_io_Deserializer
sidl_io_Deserializer__array_get(
  const struct sidl_io_Deserializer__array* array,
  const int32_t indices[])
{
  return (sidl_io_Deserializer)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
sidl_io_Deserializer__array_set1(
  struct sidl_io_Deserializer__array* array,
  const int32_t i1,
  sidl_io_Deserializer const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
sidl_io_Deserializer__array_set2(
  struct sidl_io_Deserializer__array* array,
  const int32_t i1,
  const int32_t i2,
  sidl_io_Deserializer const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
sidl_io_Deserializer__array_set3(
  struct sidl_io_Deserializer__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  sidl_io_Deserializer const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
sidl_io_Deserializer__array_set4(
  struct sidl_io_Deserializer__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  sidl_io_Deserializer const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array to value.
 */
void
sidl_io_Deserializer__array_set5(
  struct sidl_io_Deserializer__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  sidl_io_Deserializer const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array to value.
 */
void
sidl_io_Deserializer__array_set6(
  struct sidl_io_Deserializer__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  sidl_io_Deserializer const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array to value.
 */
void
sidl_io_Deserializer__array_set7(
  struct sidl_io_Deserializer__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  sidl_io_Deserializer const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
sidl_io_Deserializer__array_set(
  struct sidl_io_Deserializer__array* array,
  const int32_t indices[],
  sidl_io_Deserializer const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
sidl_io_Deserializer__array_dimen(
  const struct sidl_io_Deserializer__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidl_io_Deserializer__array_lower(
  const struct sidl_io_Deserializer__array* array,
  const int32_t ind)
{
  return sidl_interface__array_lower((struct sidl_interface__array *)array,
    ind);
}

/**
 * Return the upper bound of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidl_io_Deserializer__array_upper(
  const struct sidl_io_Deserializer__array* array,
  const int32_t ind)
{
  return sidl_interface__array_upper((struct sidl_interface__array *)array,
    ind);
}

/**
 * Return the length of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidl_io_Deserializer__array_length(
  const struct sidl_io_Deserializer__array* array,
  const int32_t ind)
{
  return sidl_interface__array_length((struct sidl_interface__array *)array,
    ind);
}

/**
 * Return the stride of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidl_io_Deserializer__array_stride(
  const struct sidl_io_Deserializer__array* array,
  const int32_t ind)
{
  return sidl_interface__array_stride((struct sidl_interface__array *)array,
    ind);
}

/**
 * Return a true value iff the array is a contiguous
 * column-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
sidl_io_Deserializer__array_isColumnOrder(
  const struct sidl_io_Deserializer__array* array)
{
  return sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)array);
}

/**
 * Return a true value iff the array is a contiguous
 * row-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
sidl_io_Deserializer__array_isRowOrder(
  const struct sidl_io_Deserializer__array* array)
{
  return sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)array);
}

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
sidl_io_Deserializer__array_copy(
  const struct sidl_io_Deserializer__array* src,
  struct sidl_io_Deserializer__array* dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array *)src,
                             (struct sidl_interface__array *)dest);
}

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
struct sidl_io_Deserializer__array*
sidl_io_Deserializer__array_slice(
  struct sidl_io_Deserializer__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct sidl_io_Deserializer__array*)
    sidl_interface__array_slice((struct sidl_interface__array *)src,
                                dimen, numElem, srcStart, srcStride, newStart);
}

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
struct sidl_io_Deserializer__array*
sidl_io_Deserializer__array_ensure(
  struct sidl_io_Deserializer__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct sidl_io_Deserializer__array*)
    sidl_interface__array_ensure((struct sidl_interface__array *)src, dimen,
      ordering);
}

#include <stdlib.h>
#include <string.h>
#include "sidl_rmi_ProtocolFactory.h"
#include "sidl_rmi_InstanceHandle.h"
#include "sidl_rmi_Invocation.h"
#include "sidl_rmi_Response.h"

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t sidl_io__Deserializer__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidl_io__Deserializer__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidl_io__Deserializer__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidl_io__Deserializer__mutex )==EDEADLOCK) */
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

static struct sidl_io__Deserializer__epv s_rem_epv__sidl_io__deserializer;

static struct sidl_BaseInterface__epv s_rem_epv__sidl_baseinterface;

static struct sidl_io_Deserializer__epv s_rem_epv__sidl_io_deserializer;

/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_sidl_io__Deserializer__cast(
struct sidl_io__Deserializer__object* self,
const char* name)
{
  void* cast = NULL;

  struct sidl_io__Deserializer__object* s0;
   s0 =                                self;

  if (!strcmp(name, "sidl.io._Deserializer")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "sidl.BaseInterface")) {
    cast = (void*) &s0->d_sidl_baseinterface;
  } else if (!strcmp(name, "sidl.io.Deserializer")) {
    cast = (void*) &s0->d_sidl_io_deserializer;
  }
  else if ((*self->d_epv->f_isType)(self,name)) {
    void* (*func)(sidl_rmi_InstanceHandle) = 
      (void* (*)(sidl_rmi_InstanceHandle)) 
      sidl_rmi_ConnectRegistry_getConnect(name);
    cast =  (*func)((sidl_rmi_InstanceHandle)self->d_data);
  }

  return cast;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_sidl_io__Deserializer__delete(
  struct sidl_io__Deserializer__object* self)
{
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_sidl_io__Deserializer__getURL(
  struct sidl_io__Deserializer__object* self)
{
  sidl_rmi_InstanceHandle conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_BaseInterface _ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getURL(conn, &_ex);
  }
  return NULL;
}

/* REMOTE EXEC: call the exec function for the object. */
static void remote_sidl_io__Deserializer__exec(
  struct sidl_io__Deserializer__object* self,
  const char* methodName,
  sidl_io_Deserializer inArgs,
  sidl_io_Serializer outArgs)
{
}

/* REMOTE METHOD STUB:addRef */
static void
remote_sidl_io__Deserializer_addRef(
  /* in */ struct sidl_io__Deserializer__object* self /* TLD */)
{
  /* FIXME  need to think through all of these special cases */
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_sidl_io__Deserializer_deleteRef(
  /* in */ struct sidl_io__Deserializer__object* self /* TLD */)
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
remote_sidl_io__Deserializer_isSame(
  /* in */ struct sidl_io__Deserializer__object* self /* TLD */,
  /* in */ struct sidl_BaseInterface__object* iobj)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:queryInt */
static struct sidl_BaseInterface__object*
remote_sidl_io__Deserializer_queryInt(
  /* in */ struct sidl_io__Deserializer__object* self /* TLD */,
  /* in */ const char* name)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:isType */
static sidl_bool
remote_sidl_io__Deserializer_isType(
  /* in */ struct sidl_io__Deserializer__object* self /* TLD */,
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
remote_sidl_io__Deserializer_getClassInfo(
  /* in */ struct sidl_io__Deserializer__object* self /* TLD */)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:unpackBool */
static void
remote_sidl_io__Deserializer_unpackBool(
  /* in */ struct sidl_io__Deserializer__object* self /* TLD */,
  /* in */ const char* key,
  /* out */ sidl_bool* value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "unpackBool", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "key", key, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackBool( _rsvp, "value", value, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:unpackChar */
static void
remote_sidl_io__Deserializer_unpackChar(
  /* in */ struct sidl_io__Deserializer__object* self /* TLD */,
  /* in */ const char* key,
  /* out */ char* value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "unpackChar", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "key", key, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackChar( _rsvp, "value", value, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:unpackInt */
static void
remote_sidl_io__Deserializer_unpackInt(
  /* in */ struct sidl_io__Deserializer__object* self /* TLD */,
  /* in */ const char* key,
  /* out */ int32_t* value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "unpackInt", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "key", key, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackInt( _rsvp, "value", value, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:unpackLong */
static void
remote_sidl_io__Deserializer_unpackLong(
  /* in */ struct sidl_io__Deserializer__object* self /* TLD */,
  /* in */ const char* key,
  /* out */ int64_t* value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "unpackLong", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "key", key, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackLong( _rsvp, "value", value, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:unpackFloat */
static void
remote_sidl_io__Deserializer_unpackFloat(
  /* in */ struct sidl_io__Deserializer__object* self /* TLD */,
  /* in */ const char* key,
  /* out */ float* value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "unpackFloat", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "key", key, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackFloat( _rsvp, "value", value, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:unpackDouble */
static void
remote_sidl_io__Deserializer_unpackDouble(
  /* in */ struct sidl_io__Deserializer__object* self /* TLD */,
  /* in */ const char* key,
  /* out */ double* value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "unpackDouble", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "key", key, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackDouble( _rsvp, "value", value, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:unpackFcomplex */
static void
remote_sidl_io__Deserializer_unpackFcomplex(
  /* in */ struct sidl_io__Deserializer__object* self /* TLD */,
  /* in */ const char* key,
  /* out */ struct sidl_fcomplex* value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "unpackFcomplex", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "key", key, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackFcomplex( _rsvp, "value", value, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:unpackDcomplex */
static void
remote_sidl_io__Deserializer_unpackDcomplex(
  /* in */ struct sidl_io__Deserializer__object* self /* TLD */,
  /* in */ const char* key,
  /* out */ struct sidl_dcomplex* value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "unpackDcomplex", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "key", key, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackDcomplex( _rsvp, "value", value, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:unpackString */
static void
remote_sidl_io__Deserializer_unpackString(
  /* in */ struct sidl_io__Deserializer__object* self /* TLD */,
  /* in */ const char* key,
  /* out */ char** value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "unpackString", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "key", key, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackString( _rsvp, "value", value, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void sidl_io__Deserializer__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct sidl_io__Deserializer__epv* epv = &s_rem_epv__sidl_io__deserializer;
  struct sidl_BaseInterface__epv*    e0  = &s_rem_epv__sidl_baseinterface;
  struct sidl_io_Deserializer__epv*  e1  = &s_rem_epv__sidl_io_deserializer;

  epv->f__cast               = remote_sidl_io__Deserializer__cast;
  epv->f__delete             = remote_sidl_io__Deserializer__delete;
  epv->f__exec               = remote_sidl_io__Deserializer__exec;
  epv->f__getURL             = remote_sidl_io__Deserializer__getURL;
  epv->f__ctor               = NULL;
  epv->f__dtor               = NULL;
  epv->f_addRef              = remote_sidl_io__Deserializer_addRef;
  epv->f_deleteRef           = remote_sidl_io__Deserializer_deleteRef;
  epv->f_isSame              = remote_sidl_io__Deserializer_isSame;
  epv->f_queryInt            = remote_sidl_io__Deserializer_queryInt;
  epv->f_isType              = remote_sidl_io__Deserializer_isType;
  epv->f_getClassInfo        = remote_sidl_io__Deserializer_getClassInfo;
  epv->f_unpackBool          = remote_sidl_io__Deserializer_unpackBool;
  epv->f_unpackChar          = remote_sidl_io__Deserializer_unpackChar;
  epv->f_unpackInt           = remote_sidl_io__Deserializer_unpackInt;
  epv->f_unpackLong          = remote_sidl_io__Deserializer_unpackLong;
  epv->f_unpackFloat         = remote_sidl_io__Deserializer_unpackFloat;
  epv->f_unpackDouble        = remote_sidl_io__Deserializer_unpackDouble;
  epv->f_unpackFcomplex      = remote_sidl_io__Deserializer_unpackFcomplex;
  epv->f_unpackDcomplex      = remote_sidl_io__Deserializer_unpackDcomplex;
  epv->f_unpackString        = remote_sidl_io__Deserializer_unpackString;

  e0->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete      = (void (*)(void*)) epv->f__delete;
  e0->f__exec        = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e0->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e0->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  e1->f__cast          = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete        = (void (*)(void*)) epv->f__delete;
  e1->f__exec          = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e1->f_addRef         = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef      = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame         = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt       = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType         = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_getClassInfo   = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e1->f_unpackBool     = (void (*)(void*,const char*,sidl_bool*,
    struct sidl_BaseInterface__object **)) epv->f_unpackBool;
  e1->f_unpackChar     = (void (*)(void*,const char*,char*,
    struct sidl_BaseInterface__object **)) epv->f_unpackChar;
  e1->f_unpackInt      = (void (*)(void*,const char*,int32_t*,
    struct sidl_BaseInterface__object **)) epv->f_unpackInt;
  e1->f_unpackLong     = (void (*)(void*,const char*,int64_t*,
    struct sidl_BaseInterface__object **)) epv->f_unpackLong;
  e1->f_unpackFloat    = (void (*)(void*,const char*,float*,
    struct sidl_BaseInterface__object **)) epv->f_unpackFloat;
  e1->f_unpackDouble   = (void (*)(void*,const char*,double*,
    struct sidl_BaseInterface__object **)) epv->f_unpackDouble;
  e1->f_unpackFcomplex = (void (*)(void*,const char*,struct sidl_fcomplex*,
    struct sidl_BaseInterface__object **)) epv->f_unpackFcomplex;
  e1->f_unpackDcomplex = (void (*)(void*,const char*,struct sidl_dcomplex*,
    struct sidl_BaseInterface__object **)) epv->f_unpackDcomplex;
  e1->f_unpackString   = (void (*)(void*,const char*,char**,
    struct sidl_BaseInterface__object **)) epv->f_unpackString;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct sidl_io_Deserializer__object*
sidl_io_Deserializer__remoteConnect(const char *url, sidl_BaseInterface *_ex)
{
  struct sidl_io__Deserializer__object* self;

  struct sidl_io__Deserializer__object* s0;

  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_connectInstance(url, _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct sidl_io__Deserializer__object*) malloc(
      sizeof(struct sidl_io__Deserializer__object));

   s0 =                                self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_io__Deserializer__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_sidl_io_deserializer.d_epv    = &s_rem_epv__sidl_io_deserializer;
  s0->d_sidl_io_deserializer.d_object = (void*) self;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__sidl_io__deserializer;

  self->d_data = (void*) instance;

  return sidl_io_Deserializer__cast(self);
}
/* Create an instance that uses an already existing InstanceHandel to connect 
  to an existing remote object. */
static struct sidl_io_Deserializer__object*
sidl_io_Deserializer__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct sidl_io__Deserializer__object* self;

  struct sidl_io__Deserializer__object* s0;

  self =
    (struct sidl_io__Deserializer__object*) malloc(
      sizeof(struct sidl_io__Deserializer__object));

   s0 =                                self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_io__Deserializer__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_sidl_io_deserializer.d_epv    = &s_rem_epv__sidl_io_deserializer;
  s0->d_sidl_io_deserializer.d_object = (void*) self;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__sidl_io__deserializer;

  self->d_data = (void*) instance;

  sidl_rmi_InstanceHandle_addRef(instance);
  return sidl_io_Deserializer__cast(self);
}
