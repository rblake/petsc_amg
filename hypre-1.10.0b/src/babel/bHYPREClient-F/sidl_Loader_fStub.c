/*
 * File:          sidl_Loader_fStub.c
 * Symbol:        sidl.Loader-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Release:       $Name: V1-10-0b $
 * Revision:      @(#) $Id: sidl_Loader_fStub.c,v 1.33 2005/12/15 00:36:08 painter Exp $
 * Description:   Client-side glue code for sidl.Loader
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
 * babel-version = 0.10.4
 * xml-url       = /home/painter/babel-0.10.4/bin/.././share/repository/sidl.Loader-v0.9.3.xml
 */

/*
 * Symbol "sidl.Loader" (version 0.9.3)
 * 
 * Class <code>Loader</code> manages dyanamic loading and symbol name
 * resolution for the sidl runtime system.  The <code>Loader</code> class
 * manages a library search path and keeps a record of all libraries
 * loaded through this interface, including the initial "global" symbols
 * in the main program.
 * 
 * Unless explicitly set, the <code>Loader</code> uses the default
 * <code>sidl.Finder</code> implemented in <code>sidl.DFinder</code>.
 * This class searches the filesystem for <code>.scl</code> files when
 * trying to find a class. The initial path is taken from the
 * environment variable SIDL_DLL_PATH, which is a semi-colon
 * separated sequence of URIs as described in class <code>DLL</code>.
 */

#include <stddef.h>
#include <stdlib.h>
#include "sidlfortran.h"
#include "sidl_header.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#include <stdio.h>
#include "sidl_Loader_IOR.h"
#include "sidl_Finder_IOR.h"
#include "sidl_DLL_IOR.h"
#include "sidl_ClassInfo_IOR.h"
#include "sidl_Resolve_IOR.h"
#include "sidl_BaseInterface_IOR.h"
#include "sidl_Scope_IOR.h"

/*
 * Return pointer to internal IOR functions.
 */

static const struct sidl_Loader__external* _getIOR(void)
{
  static const struct sidl_Loader__external *_ior = NULL;
  if (!_ior) {
    _ior = sidl_Loader__externals();
  }
  return _ior;
}

/*
 * Return pointer to static functions.
 */

static const struct sidl_Loader__sepv* _getSEPV(void)
{
  static const struct sidl_Loader__sepv *_sepv = NULL;
  if (!_sepv) {
    _sepv = (*(_getIOR()->getStaticEPV))();
  }
  return _sepv;
}

/*
 * Constructor for the class.
 */

void
SIDLFortran77Symbol(sidl_loader__create_f,SIDL_LOADER__CREATE_F,sidl_Loader__create_f)
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
SIDLFortran77Symbol(sidl_loader__cast_f,SIDL_LOADER__CAST_F,sidl_Loader__cast_f)
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
      "sidl.Loader");
  } else {
    *retval = 0;
  }
}

/*
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran77Symbol(sidl_loader__cast2_f,SIDL_LOADER__CAST2_F,sidl_Loader__cast2_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_Loader__epv *_epv = NULL;
  struct sidl_Loader__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  _proxy_self =
    (struct sidl_Loader__object*)
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
SIDLFortran77Symbol(sidl_loader_addref_f,SIDL_LOADER_ADDREF_F,sidl_Loader_addRef_f)
(
  int64_t *self
)
{
  struct sidl_Loader__epv *_epv = NULL;
  struct sidl_Loader__object* _proxy_self = NULL;
  _proxy_self =
    (struct sidl_Loader__object*)
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
SIDLFortran77Symbol(sidl_loader_deleteref_f,SIDL_LOADER_DELETEREF_F,sidl_Loader_deleteRef_f)
(
  int64_t *self
)
{
  struct sidl_Loader__epv *_epv = NULL;
  struct sidl_Loader__object* _proxy_self = NULL;
  _proxy_self =
    (struct sidl_Loader__object*)
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
SIDLFortran77Symbol(sidl_loader_issame_f,SIDL_LOADER_ISSAME_F,sidl_Loader_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval
)
{
  struct sidl_Loader__epv *_epv = NULL;
  struct sidl_Loader__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_iobj = NULL;
  sidl_bool _proxy_retval;
  _proxy_self =
    (struct sidl_Loader__object*)
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
SIDLFortran77Symbol(sidl_loader_queryint_f,SIDL_LOADER_QUERYINT_F,sidl_Loader_queryInt_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_Loader__epv *_epv = NULL;
  struct sidl_Loader__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_BaseInterface__object* _proxy_retval = NULL;
  _proxy_self =
    (struct sidl_Loader__object*)
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
SIDLFortran77Symbol(sidl_loader_istype_f,SIDL_LOADER_ISTYPE_F,sidl_Loader_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct sidl_Loader__epv *_epv = NULL;
  struct sidl_Loader__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  sidl_bool _proxy_retval;
  _proxy_self =
    (struct sidl_Loader__object*)
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
SIDLFortran77Symbol(sidl_loader_getclassinfo_f,SIDL_LOADER_GETCLASSINFO_F,sidl_Loader_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval
)
{
  struct sidl_Loader__epv *_epv = NULL;
  struct sidl_Loader__object* _proxy_self = NULL;
  struct sidl_ClassInfo__object* _proxy_retval = NULL;
  _proxy_self =
    (struct sidl_Loader__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getClassInfo))(
      _proxy_self
    );
  *retval = (ptrdiff_t)_proxy_retval;
}

/*
 * Load the specified library if it has not already been loaded.
 * The URI format is defined in class <code>DLL</code>.  The search
 * path is not searched to resolve the library name.
 * 
 * @param uri          the URI to load. This can be a .la file
 *                     (a metadata file produced by libtool) or
 *                     a shared library binary (i.e., .so,
 *                     .dll or whatever is appropriate for your
 *                     OS)
 * @param loadGlobally <code>true</code> means that the shared
 *                     library symbols will be loaded into the
 *                     global namespace; <code>false</code> 
 *                     means they will be loaded into a 
 *                     private namespace. Some operating systems
 *                     may not be able to honor the value presented
 *                     here.
 * @param loadLazy     <code>true</code> instructs the loader to
 *                     that symbols can be resolved as needed (lazy)
 *                     instead of requiring everything to be resolved
 *                     now.
 * @return if the load was successful, a non-NULL DLL object is returned.
 */

void
SIDLFortran77Symbol(sidl_loader_loadlibrary_f,SIDL_LOADER_LOADLIBRARY_F,sidl_Loader_loadLibrary_f)
(
  SIDL_F77_String uri
  SIDL_F77_STR_NEAR_LEN_DECL(uri),
  SIDL_F77_Bool *loadGlobally,
  SIDL_F77_Bool *loadLazy,
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(uri)
)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  char* _proxy_uri = NULL;
  sidl_bool _proxy_loadGlobally;
  sidl_bool _proxy_loadLazy;
  struct sidl_DLL__object* _proxy_retval = NULL;
  _proxy_uri =
    sidl_copy_fortran_str(SIDL_F77_STR(uri),
      SIDL_F77_STR_LEN(uri));
  _proxy_loadGlobally = ((*loadGlobally == SIDL_F77_TRUE) ? TRUE : FALSE);
  _proxy_loadLazy = ((*loadLazy == SIDL_F77_TRUE) ? TRUE : FALSE);
  _proxy_retval = 
    (*(_epv->f_loadLibrary))(
      _proxy_uri,
      _proxy_loadGlobally,
      _proxy_loadLazy
    );
  *retval = (ptrdiff_t)_proxy_retval;
  free((void *)_proxy_uri);
}

/*
 * Append the specified DLL to the beginning of the list of already
 * loaded DLLs.
 */

void
SIDLFortran77Symbol(sidl_loader_adddll_f,SIDL_LOADER_ADDDLL_F,sidl_Loader_addDLL_f)
(
  int64_t *dll
)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  struct sidl_DLL__object* _proxy_dll = NULL;
  _proxy_dll =
    (struct sidl_DLL__object*)
    (ptrdiff_t)(*dll);
  (*(_epv->f_addDLL))(
    _proxy_dll
  );
}

/*
 * Unload all dynamic link libraries.  The library may no longer
 * be used to access symbol names.  When the library is actually
 * unloaded from the memory image depends on details of the operating
 * system.
 */

void
SIDLFortran77Symbol(sidl_loader_unloadlibraries_f,SIDL_LOADER_UNLOADLIBRARIES_F,sidl_Loader_unloadLibraries_f)
(
void)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  (*(_epv->f_unloadLibraries))(

  );
}

/*
 * Find a DLL containing the specified information for a sidl
 * class. This method searches SCL files in the search path looking
 * for a shared library that contains the client-side or IOR
 * for a particular sidl class.
 * 
 * This call is implemented by calling the current
 * <code>Finder</code>. The default finder searches the local
 * file system for <code>.scl</code> files to locate the
 * target class/interface.
 * 
 * @param sidl_name  the fully qualified (long) name of the
 *                   class/interface to be found. Package names
 *                   are separated by period characters from each
 *                   other and the class/interface name.
 * @param target     to find a client-side binding, this is
 *                   normally the name of the language.
 *                   To find the implementation of a class
 *                   in order to make one, you should pass
 *                   the string "ior/impl" here.
 * @param lScope     this specifies whether the symbols should
 *                   be loaded into the global scope, a local
 *                   scope, or use the setting in the SCL file.
 * @param lResolve   this specifies whether symbols should be
 *                   resolved as needed (LAZY), completely
 *                   resolved at load time (NOW), or use the
 *                   setting from the SCL file.
 * @return a non-NULL object means the search was successful.
 *         The DLL has already been added.
 */

void
SIDLFortran77Symbol(sidl_loader_findlibrary_f,SIDL_LOADER_FINDLIBRARY_F,sidl_Loader_findLibrary_f)
(
  SIDL_F77_String sidl_name
  SIDL_F77_STR_NEAR_LEN_DECL(sidl_name),
  SIDL_F77_String target
  SIDL_F77_STR_NEAR_LEN_DECL(target),
  int32_t *lScope,
  int32_t *lResolve,
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(sidl_name)
  SIDL_F77_STR_FAR_LEN_DECL(target)
)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  char* _proxy_sidl_name = NULL;
  char* _proxy_target = NULL;
  enum sidl_Scope__enum _proxy_lScope;
  enum sidl_Resolve__enum _proxy_lResolve;
  struct sidl_DLL__object* _proxy_retval = NULL;
  _proxy_sidl_name =
    sidl_copy_fortran_str(SIDL_F77_STR(sidl_name),
      SIDL_F77_STR_LEN(sidl_name));
  _proxy_target =
    sidl_copy_fortran_str(SIDL_F77_STR(target),
      SIDL_F77_STR_LEN(target));
  _proxy_lScope =
    (enum sidl_Scope__enum)*
    lScope;
  _proxy_lResolve =
    (enum sidl_Resolve__enum)*
    lResolve;
  _proxy_retval = 
    (*(_epv->f_findLibrary))(
      _proxy_sidl_name,
      _proxy_target,
      _proxy_lScope,
      _proxy_lResolve
    );
  *retval = (ptrdiff_t)_proxy_retval;
  free((void *)_proxy_sidl_name);
  free((void *)_proxy_target);
}

/*
 * Set the search path, which is a semi-colon separated sequence of
 * URIs as described in class <code>DLL</code>.  This method will
 * invalidate any existing search path.
 * 
 * This updates the search path in the current <code>Finder</code>.
 */

void
SIDLFortran77Symbol(sidl_loader_setsearchpath_f,SIDL_LOADER_SETSEARCHPATH_F,sidl_Loader_setSearchPath_f)
(
  SIDL_F77_String path_name
  SIDL_F77_STR_NEAR_LEN_DECL(path_name)
  SIDL_F77_STR_FAR_LEN_DECL(path_name)
)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  char* _proxy_path_name = NULL;
  _proxy_path_name =
    sidl_copy_fortran_str(SIDL_F77_STR(path_name),
      SIDL_F77_STR_LEN(path_name));
  (*(_epv->f_setSearchPath))(
    _proxy_path_name
  );
  free((void *)_proxy_path_name);
}

/*
 * Return the current search path.  The default
 * <code>Finder</code> initializes the search path
 * from environment variable SIDL_DLL_PATH.
 * 
 */

void
SIDLFortran77Symbol(sidl_loader_getsearchpath_f,SIDL_LOADER_GETSEARCHPATH_F,sidl_Loader_getSearchPath_f)
(
  SIDL_F77_String retval
  SIDL_F77_STR_NEAR_LEN_DECL(retval)
  SIDL_F77_STR_FAR_LEN_DECL(retval)
)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  char* _proxy_retval = NULL;
  _proxy_retval = 
    (*(_epv->f_getSearchPath))(

    );
  sidl_copy_c_str(
    SIDL_F77_STR(retval),
    SIDL_F77_STR_LEN(retval),
    _proxy_retval);
  free((void *)_proxy_retval);
}

/*
 * Append the specified path fragment to the beginning of the
 * current search path.  This method operates on the Loader's
 * current <code>Finder</code>. This will add a path to the
 * current search path. Normally, the search path is initialized
 * from the SIDL_DLL_PATH environment variable.
 */

void
SIDLFortran77Symbol(sidl_loader_addsearchpath_f,SIDL_LOADER_ADDSEARCHPATH_F,sidl_Loader_addSearchPath_f)
(
  SIDL_F77_String path_fragment
  SIDL_F77_STR_NEAR_LEN_DECL(path_fragment)
  SIDL_F77_STR_FAR_LEN_DECL(path_fragment)
)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  char* _proxy_path_fragment = NULL;
  _proxy_path_fragment =
    sidl_copy_fortran_str(SIDL_F77_STR(path_fragment),
      SIDL_F77_STR_LEN(path_fragment));
  (*(_epv->f_addSearchPath))(
    _proxy_path_fragment
  );
  free((void *)_proxy_path_fragment);
}

/*
 * This method sets the <code>Finder</code> that
 * <code>Loader</code> will use to find DLLs.  If no
 * <code>Finder</code> is set or if NULL is passed in, the Default
 * Finder <code>DFinder</code> will be used.
 * 
 * Future calls to <code>findLibrary</code>,
 * <code>addSearchPath</code>, <code>getSearchPath</code>, and
 * <code>setSearchPath</code> are deligated to the
 * <code>Finder</code> set here.
 */

void
SIDLFortran77Symbol(sidl_loader_setfinder_f,SIDL_LOADER_SETFINDER_F,sidl_Loader_setFinder_f)
(
  int64_t *f
)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  struct sidl_Finder__object* _proxy_f = NULL;
  _proxy_f =
    (struct sidl_Finder__object*)
    (ptrdiff_t)(*f);
  (*(_epv->f_setFinder))(
    _proxy_f
  );
}

/*
 * This method gets the <code>Finder</code> that <code>Loader</code>
 * uses to find DLLs.  
 */

void
SIDLFortran77Symbol(sidl_loader_getfinder_f,SIDL_LOADER_GETFINDER_F,sidl_Loader_getFinder_f)
(
  int64_t *retval
)
{
  const struct sidl_Loader__sepv *_epv = _getSEPV();
  struct sidl_Finder__object* _proxy_retval = NULL;
  _proxy_retval = 
    (*(_epv->f_getFinder))(

    );
  *retval = (ptrdiff_t)_proxy_retval;
}

void
SIDLFortran77Symbol(sidl_loader__array_createcol_f,
                  SIDL_LOADER__ARRAY_CREATECOL_F,
                  sidl_Loader__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_loader__array_createrow_f,
                  SIDL_LOADER__ARRAY_CREATEROW_F,
                  sidl_Loader__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(sidl_loader__array_create1d_f,
                  SIDL_LOADER__ARRAY_CREATE1D_F,
                  sidl_Loader__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(sidl_loader__array_create2dcol_f,
                  SIDL_LOADER__ARRAY_CREATE2DCOL_F,
                  sidl_Loader__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(sidl_loader__array_create2drow_f,
                  SIDL_LOADER__ARRAY_CREATE2DROW_F,
                  sidl_Loader__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(sidl_loader__array_addref_f,
                  SIDL_LOADER__ARRAY_ADDREF_F,
                  sidl_Loader__array_addRef_f)
  (int64_t *array)
{
  sidl_interface__array_addRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_loader__array_deleteref_f,
                  SIDL_LOADER__ARRAY_DELETEREF_F,
                  sidl_Loader__array_deleteRef_f)
  (int64_t *array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_loader__array_get1_f,
                  SIDL_LOADER__ARRAY_GET1_F,
                  sidl_Loader__array_get1_f)
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
SIDLFortran77Symbol(sidl_loader__array_get2_f,
                  SIDL_LOADER__ARRAY_GET2_F,
                  sidl_Loader__array_get2_f)
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
SIDLFortran77Symbol(sidl_loader__array_get3_f,
                  SIDL_LOADER__ARRAY_GET3_F,
                  sidl_Loader__array_get3_f)
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
SIDLFortran77Symbol(sidl_loader__array_get4_f,
                  SIDL_LOADER__ARRAY_GET4_F,
                  sidl_Loader__array_get4_f)
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
SIDLFortran77Symbol(sidl_loader__array_get5_f,
                  SIDL_LOADER__ARRAY_GET5_F,
                  sidl_Loader__array_get5_f)
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
SIDLFortran77Symbol(sidl_loader__array_get6_f,
                  SIDL_LOADER__ARRAY_GET6_F,
                  sidl_Loader__array_get6_f)
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
SIDLFortran77Symbol(sidl_loader__array_get7_f,
                  SIDL_LOADER__ARRAY_GET7_F,
                  sidl_Loader__array_get7_f)
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
SIDLFortran77Symbol(sidl_loader__array_get_f,
                  SIDL_LOADER__ARRAY_GET_F,
                  sidl_Loader__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get((const struct sidl_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(sidl_loader__array_set1_f,
                  SIDL_LOADER__ARRAY_SET1_F,
                  sidl_Loader__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_loader__array_set2_f,
                  SIDL_LOADER__ARRAY_SET2_F,
                  sidl_Loader__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_loader__array_set3_f,
                  SIDL_LOADER__ARRAY_SET3_F,
                  sidl_Loader__array_set3_f)
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
SIDLFortran77Symbol(sidl_loader__array_set4_f,
                  SIDL_LOADER__ARRAY_SET4_F,
                  sidl_Loader__array_set4_f)
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
SIDLFortran77Symbol(sidl_loader__array_set5_f,
                  SIDL_LOADER__ARRAY_SET5_F,
                  sidl_Loader__array_set5_f)
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
SIDLFortran77Symbol(sidl_loader__array_set6_f,
                  SIDL_LOADER__ARRAY_SET6_F,
                  sidl_Loader__array_set6_f)
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
SIDLFortran77Symbol(sidl_loader__array_set7_f,
                  SIDL_LOADER__ARRAY_SET7_F,
                  sidl_Loader__array_set7_f)
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
SIDLFortran77Symbol(sidl_loader__array_set_f,
                  SIDL_LOADER__ARRAY_SET_F,
                  sidl_Loader__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_interface__array_set((struct sidl_interface__array *)(ptrdiff_t)*array,
    indices, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(sidl_loader__array_dimen_f,
                  SIDL_LOADER__ARRAY_DIMEN_F,
                  sidl_Loader__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_interface__array_dimen((struct sidl_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_loader__array_lower_f,
                  SIDL_LOADER__ARRAY_LOWER_F,
                  sidl_Loader__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_lower((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_loader__array_upper_f,
                  SIDL_LOADER__ARRAY_UPPER_F,
                  sidl_Loader__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_upper((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_loader__array_length_f,
                  SIDL_LOADER__ARRAY_LENGTH_F,
                  sidl_Loader__array_length_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_length((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_loader__array_stride_f,
                  SIDL_LOADER__ARRAY_STRIDE_F,
                  sidl_Loader__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_stride((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(sidl_loader__array_iscolumnorder_f,
                  SIDL_LOADER__ARRAY_ISCOLUMNORDER_F,
                  sidl_Loader__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_loader__array_isroworder_f,
                  SIDL_LOADER__ARRAY_ISROWORDER_F,
                  sidl_Loader__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(sidl_loader__array_copy_f,
                  SIDL_LOADER__ARRAY_COPY_F,
                  sidl_Loader__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array 
    *)(ptrdiff_t)*src,
                             (struct sidl_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(sidl_loader__array_smartcopy_f,
                  SIDL_LOADER__ARRAY_SMARTCOPY_F,
                  sidl_Loader__array_smartCopy_f)
  (int64_t *src)
{
  sidl_interface__array_smartCopy((struct sidl_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(sidl_loader__array_slice_f,
                  SIDL_LOADER__ARRAY_SLICE_F,
                  sidl_Loader__array_slice_f)
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
SIDLFortran77Symbol(sidl_loader__array_ensure_f,
                  SIDL_LOADER__ARRAY_ENSURE_F,
                  sidl_Loader__array_ensure_f)
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

