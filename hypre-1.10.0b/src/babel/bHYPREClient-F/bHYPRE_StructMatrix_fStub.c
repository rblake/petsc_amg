/*
 * File:          bHYPRE_StructMatrix_fStub.c
 * Symbol:        bHYPRE.StructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.StructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

/*
 * Symbol "bHYPRE.StructMatrix" (version 1.0.0)
 * 
 * A single class that implements both a view interface and an
 * operator interface.
 * A StructMatrix is a matrix on a structured grid.
 * One function unique to a StructMatrix is SetConstantEntries.
 * This declares that matrix entries corresponding to certain stencil points
 * (supplied as stencil element indices) will be constant throughout the grid.
 * 
 */

#include <stddef.h>
#include <stdlib.h>
#include "sidlfortran.h"
#include "sidl_header.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#include <stdio.h>
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include "sidl_Loader.h"
#endif
#include "bHYPRE_StructMatrix_IOR.h"
#include "bHYPRE_StructGrid_IOR.h"
#include "bHYPRE_MPICommunicator_IOR.h"
#include "sidl_ClassInfo_IOR.h"
#include "bHYPRE_Vector_IOR.h"
#include "bHYPRE_StructStencil_IOR.h"
#include "sidl_BaseInterface_IOR.h"

/*
 * Return pointer to internal IOR functions.
 */

static const struct bHYPRE_StructMatrix__external* _getIOR(void)
{
  static const struct bHYPRE_StructMatrix__external *_ior = NULL;
  if (!_ior) {
#ifdef SIDL_STATIC_LIBRARY
    _ior = bHYPRE_StructMatrix__externals();
#else
    _ior = (struct 
      bHYPRE_StructMatrix__external*)sidl_dynamicLoadIOR("bHYPRE.StructMatrix",
      "bHYPRE_StructMatrix__externals") ;
#endif
  }
  return _ior;
}

/*
 * Return pointer to static functions.
 */

static const struct bHYPRE_StructMatrix__sepv* _getSEPV(void)
{
  static const struct bHYPRE_StructMatrix__sepv *_sepv = NULL;
  if (!_sepv) {
    _sepv = (*(_getIOR()->getStaticEPV))();
  }
  return _sepv;
}

/*
 * Constructor for the class.
 */

void
SIDLFortran77Symbol(bhypre_structmatrix__create_f,BHYPRE_STRUCTMATRIX__CREATE_F,bHYPRE_StructMatrix__create_f)
(
  int64_t *self
)
{
  *self = (ptrdiff_t) (*(_getIOR()->createObject))();
}

/*
 * Cast method for interface and type conversions.
 */

void
SIDLFortran77Symbol(bhypre_structmatrix__cast_f,BHYPRE_STRUCTMATRIX__CAST_F,bHYPRE_StructMatrix__cast_f)
(
  int64_t *ref,
  int64_t *retval
)
{
  struct sidl_BaseInterface__object  *_base =
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*ref;
  if (_base) {
    *retval = (ptrdiff_t)(
      *_base->d_epv->f__cast)(
      _base->d_object,
      "bHYPRE.StructMatrix");
  } else {
    *retval = 0;
  }
}

/*
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran77Symbol(bhypre_structmatrix__cast2_f,BHYPRE_STRUCTMATRIX__CAST2_F,bHYPRE_StructMatrix__cast2_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__cast))(
      _proxy_self,
      _proxy_name
    );
  *retval = (ptrdiff_t)_proxy_retval;
  free((void *)_proxy_name);
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
SIDLFortran77Symbol(bhypre_structmatrix_addref_f,BHYPRE_STRUCTMATRIX_ADDREF_F,bHYPRE_StructMatrix_addRef_f)
(
  int64_t *self
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_addRef))(
    _proxy_self
  );
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_deleteref_f,BHYPRE_STRUCTMATRIX_DELETEREF_F,bHYPRE_StructMatrix_deleteRef_f)
(
  int64_t *self
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_deleteRef))(
    _proxy_self
  );
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_issame_f,BHYPRE_STRUCTMATRIX_ISSAME_F,bHYPRE_StructMatrix_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_iobj = NULL;
  sidl_bool _proxy_retval;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_iobj =
    (struct sidl_BaseInterface__object*)
    (ptrdiff_t)(*iobj);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isSame))(
      _proxy_self,
      _proxy_iobj
    );
  *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
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

void
SIDLFortran77Symbol(bhypre_structmatrix_queryint_f,BHYPRE_STRUCTMATRIX_QUERYINT_F,bHYPRE_StructMatrix_queryInt_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_BaseInterface__object* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_queryInt))(
      _proxy_self,
      _proxy_name
    );
  *retval = (ptrdiff_t)_proxy_retval;
  free((void *)_proxy_name);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_istype_f,BHYPRE_STRUCTMATRIX_ISTYPE_F,bHYPRE_StructMatrix_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  sidl_bool _proxy_retval;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isType))(
      _proxy_self,
      _proxy_name
    );
  *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  free((void *)_proxy_name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_getclassinfo_f,BHYPRE_STRUCTMATRIX_GETCLASSINFO_F,bHYPRE_StructMatrix_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  struct sidl_ClassInfo__object* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getClassInfo))(
      _proxy_self
    );
  *retval = (ptrdiff_t)_proxy_retval;
}

/*
 * Method:  Create[]
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_create_f,BHYPRE_STRUCTMATRIX_CREATE_F,bHYPRE_StructMatrix_Create_f)
(
  int64_t *mpi_comm,
  int64_t *grid,
  int64_t *stencil,
  int64_t *retval
)
{
  const struct bHYPRE_StructMatrix__sepv *_epv = _getSEPV();
  struct bHYPRE_MPICommunicator__object* _proxy_mpi_comm = NULL;
  struct bHYPRE_StructGrid__object* _proxy_grid = NULL;
  struct bHYPRE_StructStencil__object* _proxy_stencil = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_retval = NULL;
  _proxy_mpi_comm =
    (struct bHYPRE_MPICommunicator__object*)
    (ptrdiff_t)(*mpi_comm);
  _proxy_grid =
    (struct bHYPRE_StructGrid__object*)
    (ptrdiff_t)(*grid);
  _proxy_stencil =
    (struct bHYPRE_StructStencil__object*)
    (ptrdiff_t)(*stencil);
  _proxy_retval = 
    (*(_epv->f_Create))(
      _proxy_mpi_comm,
      _proxy_grid,
      _proxy_stencil
    );
  *retval = (ptrdiff_t)_proxy_retval;
}

/*
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 * 
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_setcommunicator_f,BHYPRE_STRUCTMATRIX_SETCOMMUNICATOR_F,bHYPRE_StructMatrix_SetCommunicator_f)
(
  int64_t *self,
  int64_t *mpi_comm,
  int32_t *retval
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  struct bHYPRE_MPICommunicator__object* _proxy_mpi_comm = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_mpi_comm =
    (struct bHYPRE_MPICommunicator__object*)
    (ptrdiff_t)(*mpi_comm);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetCommunicator))(
      _proxy_self,
      _proxy_mpi_comm
    );
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_initialize_f,BHYPRE_STRUCTMATRIX_INITIALIZE_F,bHYPRE_StructMatrix_Initialize_f)
(
  int64_t *self,
  int32_t *retval
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Initialize))(
      _proxy_self
    );
}

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_assemble_f,BHYPRE_STRUCTMATRIX_ASSEMBLE_F,bHYPRE_StructMatrix_Assemble_f)
(
  int64_t *self,
  int32_t *retval
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Assemble))(
      _proxy_self
    );
}

/*
 * Method:  SetGrid[]
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_setgrid_f,BHYPRE_STRUCTMATRIX_SETGRID_F,bHYPRE_StructMatrix_SetGrid_f)
(
  int64_t *self,
  int64_t *grid,
  int32_t *retval
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  struct bHYPRE_StructGrid__object* _proxy_grid = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_grid =
    (struct bHYPRE_StructGrid__object*)
    (ptrdiff_t)(*grid);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetGrid))(
      _proxy_self,
      _proxy_grid
    );
}

/*
 * Method:  SetStencil[]
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_setstencil_f,BHYPRE_STRUCTMATRIX_SETSTENCIL_F,bHYPRE_StructMatrix_SetStencil_f)
(
  int64_t *self,
  int64_t *stencil,
  int32_t *retval
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  struct bHYPRE_StructStencil__object* _proxy_stencil = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_stencil =
    (struct bHYPRE_StructStencil__object*)
    (ptrdiff_t)(*stencil);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetStencil))(
      _proxy_self,
      _proxy_stencil
    );
}

/*
 * Method:  SetValues[]
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_setvalues_f,BHYPRE_STRUCTMATRIX_SETVALUES_F,bHYPRE_StructMatrix_SetValues_f)
(
  int64_t *self,
  int32_t *index,
  int32_t *dim,
  int32_t *num_stencil_indices,
  int32_t *stencil_indices,
  double *values,
  int32_t *retval
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  struct sidl_int__array _alt_index;
  struct sidl_int__array* _proxy_index = &_alt_index;
  int32_t index_lower[1], index_upper[1], index_stride[1];
  struct sidl_int__array _alt_stencil_indices;
  struct sidl_int__array* _proxy_stencil_indices = &_alt_stencil_indices;
  int32_t stencil_indices_lower[1], stencil_indices_upper[1],
    stencil_indices_stride[1];
  struct sidl_double__array _alt_values;
  struct sidl_double__array* _proxy_values = &_alt_values;
  int32_t values_lower[1], values_upper[1], values_stride[1];
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  index_upper[0] = (*dim)-1;
  sidl_int__array_init(index, _proxy_index, 1, index_lower, index_upper,
    index_stride);
  stencil_indices_upper[0] = (*num_stencil_indices)-1;
  sidl_int__array_init(stencil_indices, _proxy_stencil_indices, 1,
    stencil_indices_lower, stencil_indices_upper, stencil_indices_stride);
  values_upper[0] = (*num_stencil_indices)-1;
  sidl_double__array_init(values, _proxy_values, 1, values_lower, values_upper,
    values_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetValues))(
      _proxy_self,
      _proxy_index,
      _proxy_stencil_indices,
      _proxy_values
    );
}

/*
 * Method:  SetBoxValues[]
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_setboxvalues_f,BHYPRE_STRUCTMATRIX_SETBOXVALUES_F,bHYPRE_StructMatrix_SetBoxValues_f)
(
  int64_t *self,
  int32_t *ilower,
  int32_t *iupper,
  int32_t *dim,
  int32_t *num_stencil_indices,
  int32_t *stencil_indices,
  double *values,
  int32_t *nvalues,
  int32_t *retval
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  struct sidl_int__array _alt_ilower;
  struct sidl_int__array* _proxy_ilower = &_alt_ilower;
  int32_t ilower_lower[1], ilower_upper[1], ilower_stride[1];
  struct sidl_int__array _alt_iupper;
  struct sidl_int__array* _proxy_iupper = &_alt_iupper;
  int32_t iupper_lower[1], iupper_upper[1], iupper_stride[1];
  struct sidl_int__array _alt_stencil_indices;
  struct sidl_int__array* _proxy_stencil_indices = &_alt_stencil_indices;
  int32_t stencil_indices_lower[1], stencil_indices_upper[1],
    stencil_indices_stride[1];
  struct sidl_double__array _alt_values;
  struct sidl_double__array* _proxy_values = &_alt_values;
  int32_t values_lower[1], values_upper[1], values_stride[1];
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  ilower_upper[0] = (*dim)-1;
  sidl_int__array_init(ilower, _proxy_ilower, 1, ilower_lower, ilower_upper,
    ilower_stride);
  iupper_upper[0] = (*dim)-1;
  sidl_int__array_init(iupper, _proxy_iupper, 1, iupper_lower, iupper_upper,
    iupper_stride);
  stencil_indices_upper[0] = (*num_stencil_indices)-1;
  sidl_int__array_init(stencil_indices, _proxy_stencil_indices, 1,
    stencil_indices_lower, stencil_indices_upper, stencil_indices_stride);
  values_upper[0] = (*nvalues)-1;
  sidl_double__array_init(values, _proxy_values, 1, values_lower, values_upper,
    values_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetBoxValues))(
      _proxy_self,
      _proxy_ilower,
      _proxy_iupper,
      _proxy_stencil_indices,
      _proxy_values
    );
}

/*
 * Method:  SetNumGhost[]
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_setnumghost_f,BHYPRE_STRUCTMATRIX_SETNUMGHOST_F,bHYPRE_StructMatrix_SetNumGhost_f)
(
  int64_t *self,
  int32_t *num_ghost,
  int32_t *dim2,
  int32_t *retval
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  struct sidl_int__array _alt_num_ghost;
  struct sidl_int__array* _proxy_num_ghost = &_alt_num_ghost;
  int32_t num_ghost_lower[1], num_ghost_upper[1], num_ghost_stride[1];
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  num_ghost_upper[0] = (*dim2)-1;
  sidl_int__array_init(num_ghost, _proxy_num_ghost, 1, num_ghost_lower,
    num_ghost_upper, num_ghost_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetNumGhost))(
      _proxy_self,
      _proxy_num_ghost
    );
}

/*
 * Method:  SetSymmetric[]
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_setsymmetric_f,BHYPRE_STRUCTMATRIX_SETSYMMETRIC_F,bHYPRE_StructMatrix_SetSymmetric_f)
(
  int64_t *self,
  int32_t *symmetric,
  int32_t *retval
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetSymmetric))(
      _proxy_self,
      *symmetric
    );
}

/*
 * Method:  SetConstantEntries[]
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_setconstantentries_f,BHYPRE_STRUCTMATRIX_SETCONSTANTENTRIES_F,bHYPRE_StructMatrix_SetConstantEntries_f)
(
  int64_t *self,
  int32_t *num_stencil_constant_points,
  int32_t *stencil_constant_points,
  int32_t *retval
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  struct sidl_int__array _alt_stencil_constant_points;
  struct sidl_int__array* _proxy_stencil_constant_points = 
    &_alt_stencil_constant_points;
  int32_t stencil_constant_points_lower[1], stencil_constant_points_upper[1],
    stencil_constant_points_stride[1];
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  stencil_constant_points_upper[0] = (*num_stencil_constant_points)-1;
  sidl_int__array_init(stencil_constant_points, _proxy_stencil_constant_points,
    1, stencil_constant_points_lower, stencil_constant_points_upper,
    stencil_constant_points_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetConstantEntries))(
      _proxy_self,
      _proxy_stencil_constant_points
    );
}

/*
 * Method:  SetConstantValues[]
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_setconstantvalues_f,BHYPRE_STRUCTMATRIX_SETCONSTANTVALUES_F,bHYPRE_StructMatrix_SetConstantValues_f)
(
  int64_t *self,
  int32_t *num_stencil_indices,
  int32_t *stencil_indices,
  double *values,
  int32_t *retval
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  struct sidl_int__array _alt_stencil_indices;
  struct sidl_int__array* _proxy_stencil_indices = &_alt_stencil_indices;
  int32_t stencil_indices_lower[1], stencil_indices_upper[1],
    stencil_indices_stride[1];
  struct sidl_double__array _alt_values;
  struct sidl_double__array* _proxy_values = &_alt_values;
  int32_t values_lower[1], values_upper[1], values_stride[1];
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  stencil_indices_upper[0] = (*num_stencil_indices)-1;
  sidl_int__array_init(stencil_indices, _proxy_stencil_indices, 1,
    stencil_indices_lower, stencil_indices_upper, stencil_indices_stride);
  values_upper[0] = (*num_stencil_indices)-1;
  sidl_double__array_init(values, _proxy_values, 1, values_lower, values_upper,
    values_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetConstantValues))(
      _proxy_self,
      _proxy_stencil_indices,
      _proxy_values
    );
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_setintparameter_f,BHYPRE_STRUCTMATRIX_SETINTPARAMETER_F,bHYPRE_StructMatrix_SetIntParameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int32_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetIntParameter))(
      _proxy_self,
      _proxy_name,
      *value
    );
  free((void *)_proxy_name);
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_setdoubleparameter_f,BHYPRE_STRUCTMATRIX_SETDOUBLEPARAMETER_F,bHYPRE_StructMatrix_SetDoubleParameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  double *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetDoubleParameter))(
      _proxy_self,
      _proxy_name,
      *value
    );
  free((void *)_proxy_name);
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_setstringparameter_f,BHYPRE_STRUCTMATRIX_SETSTRINGPARAMETER_F,bHYPRE_StructMatrix_SetStringParameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_String value
  SIDL_F77_STR_NEAR_LEN_DECL(value),
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
  SIDL_F77_STR_FAR_LEN_DECL(value)
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  char* _proxy_value = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    sidl_copy_fortran_str(SIDL_F77_STR(value),
      SIDL_F77_STR_LEN(value));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetStringParameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value
    );
  free((void *)_proxy_name);
  free((void *)_proxy_value);
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_setintarray1parameter_f,BHYPRE_STRUCTMATRIX_SETINTARRAY1PARAMETER_F,bHYPRE_StructMatrix_SetIntArray1Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int32_t *value,
  int32_t *nvalues,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_int__array _alt_value;
  struct sidl_int__array* _proxy_value = &_alt_value;
  int32_t value_lower[1], value_upper[1], value_stride[1];
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  value_upper[0] = (*nvalues)-1;
  sidl_int__array_init(value, _proxy_value, 1, value_lower, value_upper,
    value_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetIntArray1Parameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value
    );
  free((void *)_proxy_name);
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_setintarray2parameter_f,BHYPRE_STRUCTMATRIX_SETINTARRAY2PARAMETER_F,bHYPRE_StructMatrix_SetIntArray2Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_int__array* _proxy_value = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    (struct sidl_int__array*)
    (ptrdiff_t)(*value);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetIntArray2Parameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value
    );
  free((void *)_proxy_name);
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_setdoublearray1parameter_f,BHYPRE_STRUCTMATRIX_SETDOUBLEARRAY1PARAMETER_F,bHYPRE_StructMatrix_SetDoubleArray1Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  double *value,
  int32_t *nvalues,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_double__array _alt_value;
  struct sidl_double__array* _proxy_value = &_alt_value;
  int32_t value_lower[1], value_upper[1], value_stride[1];
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  value_upper[0] = (*nvalues)-1;
  sidl_double__array_init(value, _proxy_value, 1, value_lower, value_upper,
    value_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetDoubleArray1Parameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value
    );
  free((void *)_proxy_name);
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_setdoublearray2parameter_f,BHYPRE_STRUCTMATRIX_SETDOUBLEARRAY2PARAMETER_F,bHYPRE_StructMatrix_SetDoubleArray2Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_double__array* _proxy_value = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    (struct sidl_double__array*)
    (ptrdiff_t)(*value);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetDoubleArray2Parameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value
    );
  free((void *)_proxy_name);
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_getintvalue_f,BHYPRE_STRUCTMATRIX_GETINTVALUE_F,bHYPRE_StructMatrix_GetIntValue_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int32_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetIntValue))(
      _proxy_self,
      _proxy_name,
      value
    );
  free((void *)_proxy_name);
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_getdoublevalue_f,BHYPRE_STRUCTMATRIX_GETDOUBLEVALUE_F,bHYPRE_StructMatrix_GetDoubleValue_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  double *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetDoubleValue))(
      _proxy_self,
      _proxy_name,
      value
    );
  free((void *)_proxy_name);
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_setup_f,BHYPRE_STRUCTMATRIX_SETUP_F,bHYPRE_StructMatrix_Setup_f)
(
  int64_t *self,
  int64_t *b,
  int64_t *x,
  int32_t *retval
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  struct bHYPRE_Vector__object* _proxy_b = NULL;
  struct bHYPRE_Vector__object* _proxy_x = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_b =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*b);
  _proxy_x =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*x);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Setup))(
      _proxy_self,
      _proxy_b,
      _proxy_x
    );
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_apply_f,BHYPRE_STRUCTMATRIX_APPLY_F,bHYPRE_StructMatrix_Apply_f)
(
  int64_t *self,
  int64_t *b,
  int64_t *x,
  int32_t *retval
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  struct bHYPRE_Vector__object* _proxy_b = NULL;
  struct bHYPRE_Vector__object* _proxy_x = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_b =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*b);
  _proxy_x =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*x);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Apply))(
      _proxy_self,
      _proxy_b,
      &_proxy_x
    );
  *x = (ptrdiff_t)_proxy_x;
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_structmatrix_applyadjoint_f,BHYPRE_STRUCTMATRIX_APPLYADJOINT_F,bHYPRE_StructMatrix_ApplyAdjoint_f)
(
  int64_t *self,
  int64_t *b,
  int64_t *x,
  int32_t *retval
)
{
  struct bHYPRE_StructMatrix__epv *_epv = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_self = NULL;
  struct bHYPRE_Vector__object* _proxy_b = NULL;
  struct bHYPRE_Vector__object* _proxy_x = NULL;
  _proxy_self =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_b =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*b);
  _proxy_x =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*x);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_ApplyAdjoint))(
      _proxy_self,
      _proxy_b,
      &_proxy_x
    );
  *x = (ptrdiff_t)_proxy_x;
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_createcol_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_CREATECOL_F,
                  bHYPRE_StructMatrix__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_createrow_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_CREATEROW_F,
                  bHYPRE_StructMatrix__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_create1d_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_CREATE1D_F,
                  bHYPRE_StructMatrix__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_create2dcol_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_CREATE2DCOL_F,
                  bHYPRE_StructMatrix__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_create2drow_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_CREATE2DROW_F,
                  bHYPRE_StructMatrix__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_addref_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_ADDREF_F,
                  bHYPRE_StructMatrix__array_addRef_f)
  (int64_t *array)
{
  sidl_interface__array_addRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_deleteref_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_DELETEREF_F,
                  bHYPRE_StructMatrix__array_deleteRef_f)
  (int64_t *array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_get1_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_GET1_F,
                  bHYPRE_StructMatrix__array_get1_f)
  (int64_t *array, 
   int32_t *i1, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get1((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_get2_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_GET2_F,
                  bHYPRE_StructMatrix__array_get2_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get2((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_get3_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_GET3_F,
                  bHYPRE_StructMatrix__array_get3_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get3((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_get4_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_GET4_F,
                  bHYPRE_StructMatrix__array_get4_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get4((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_get5_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_GET5_F,
                  bHYPRE_StructMatrix__array_get5_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get5((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_get6_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_GET6_F,
                  bHYPRE_StructMatrix__array_get6_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int32_t *i6, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get6((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5, *i6);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_get7_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_GET7_F,
                  bHYPRE_StructMatrix__array_get7_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int32_t *i6, 
   int32_t *i7, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get7((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5, *i6, *i7);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_get_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_GET_F,
                  bHYPRE_StructMatrix__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get((const struct sidl_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_set1_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_SET1_F,
                  bHYPRE_StructMatrix__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_set2_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_SET2_F,
                  bHYPRE_StructMatrix__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_set3_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_SET3_F,
                  bHYPRE_StructMatrix__array_set3_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int64_t *value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_set4_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_SET4_F,
                  bHYPRE_StructMatrix__array_set4_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int64_t *value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_set5_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_SET5_F,
                  bHYPRE_StructMatrix__array_set5_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int64_t *value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5,
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_set6_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_SET6_F,
                  bHYPRE_StructMatrix__array_set6_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int32_t *i6,
   int64_t *value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5, *i6,
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_set7_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_SET7_F,
                  bHYPRE_StructMatrix__array_set7_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int32_t *i6,
   int32_t *i7,
   int64_t *value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5, *i6, *i7,
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_set_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_SET_F,
                  bHYPRE_StructMatrix__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_interface__array_set((struct sidl_interface__array *)(ptrdiff_t)*array,
    indices, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_dimen_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_DIMEN_F,
                  bHYPRE_StructMatrix__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_interface__array_dimen((struct sidl_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_lower_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_LOWER_F,
                  bHYPRE_StructMatrix__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_lower((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_upper_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_UPPER_F,
                  bHYPRE_StructMatrix__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_upper((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_length_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_LENGTH_F,
                  bHYPRE_StructMatrix__array_length_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_length((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_stride_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_STRIDE_F,
                  bHYPRE_StructMatrix__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_stride((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_iscolumnorder_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_ISCOLUMNORDER_F,
                  bHYPRE_StructMatrix__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_isroworder_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_ISROWORDER_F,
                  bHYPRE_StructMatrix__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_copy_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_COPY_F,
                  bHYPRE_StructMatrix__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array 
    *)(ptrdiff_t)*src,
                             (struct sidl_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_smartcopy_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_SMARTCOPY_F,
                  bHYPRE_StructMatrix__array_smartCopy_f)
  (int64_t *src)
{
  sidl_interface__array_smartCopy((struct sidl_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_slice_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_SLICE_F,
                  bHYPRE_StructMatrix__array_slice_f)
  (int64_t *src,
   int32_t *dimen,
   int32_t numElem[],
   int32_t srcStart[],
   int32_t srcStride[],
   int32_t newStart[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_slice((struct sidl_interface__array *)(ptrdiff_t)*src,
      *dimen, numElem, srcStart, srcStride, newStart);
}

void
SIDLFortran77Symbol(bhypre_structmatrix__array_ensure_f,
                  BHYPRE_STRUCTMATRIX__ARRAY_ENSURE_F,
                  bHYPRE_StructMatrix__array_ensure_f)
  (int64_t *src,
   int32_t *dimen,
   int     *ordering,
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_ensure((struct sidl_interface__array 
      *)(ptrdiff_t)*src,
    *dimen, *ordering);
}

