/*
 * File:          sidl_DFinder_Impl.h
 * Symbol:        sidl.DFinder-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Release:       $Name: V1-10-0b $
 * Revision:      @(#) $Id: sidl_DFinder_Impl.h,v 1.4 2005/11/14 21:20:24 painter Exp $
 * Description:   Server-side implementation for sidl.DFinder
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
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

#ifndef included_sidl_DFinder_Impl_h
#define included_sidl_DFinder_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_DFinder_h
#include "sidl_DFinder.h"
#endif
#ifndef included_sidl_Finder_h
#include "sidl_Finder.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_DLL_h
#include "sidl_DLL.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

#line 63 "../../../babel/runtime/sidl/sidl_DFinder_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidl.DFinder._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(sidl.DFinder._includes) */
#line 67 "sidl_DFinder_Impl.h"

/*
 * Private data for class sidl.DFinder
 */

struct sidl_DFinder__data {
#line 72 "../../../babel/runtime/sidl/sidl_DFinder_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidl.DFinder._data) */
  /* Put private data members here... */
  char*     d_search_path;
  /* DO-NOT-DELETE splicer.end(sidl.DFinder._data) */
#line 79 "sidl_DFinder_Impl.h"
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidl_DFinder__data*
sidl_DFinder__get_data(
  sidl_DFinder);

extern void
sidl_DFinder__set_data(
  sidl_DFinder,
  struct sidl_DFinder__data*);

extern
void
impl_sidl_DFinder__load(
  void);

extern
void
impl_sidl_DFinder__ctor(
  /* in */ sidl_DFinder self);

extern
void
impl_sidl_DFinder__dtor(
  /* in */ sidl_DFinder self);

/*
 * User-defined object methods
 */

extern struct sidl_DFinder__object* 
  impl_sidl_DFinder_fconnect_sidl_DFinder(char* url, sidl_BaseInterface *_ex);
extern char* impl_sidl_DFinder_fgetURL_sidl_DFinder(struct 
  sidl_DFinder__object* obj);
extern struct sidl_Finder__object* impl_sidl_DFinder_fconnect_sidl_Finder(char* 
  url, sidl_BaseInterface *_ex);
extern char* impl_sidl_DFinder_fgetURL_sidl_Finder(struct sidl_Finder__object* 
  obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_DFinder_fconnect_sidl_ClassInfo(char* url, sidl_BaseInterface *_ex);
extern char* impl_sidl_DFinder_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_DLL__object* impl_sidl_DFinder_fconnect_sidl_DLL(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_DFinder_fgetURL_sidl_DLL(struct sidl_DLL__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_DFinder_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_DFinder_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_DFinder_fconnect_sidl_BaseClass(char* url, sidl_BaseInterface *_ex);
extern char* impl_sidl_DFinder_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
sidl_DLL
impl_sidl_DFinder_findLibrary(
  /* in */ sidl_DFinder self,
  /* in */ const char* sidl_name,
  /* in */ const char* target,
  /* in */ enum sidl_Scope__enum lScope,
  /* in */ enum sidl_Resolve__enum lResolve);

extern
void
impl_sidl_DFinder_setSearchPath(
  /* in */ sidl_DFinder self,
  /* in */ const char* path_name);

extern
char*
impl_sidl_DFinder_getSearchPath(
  /* in */ sidl_DFinder self);

extern
void
impl_sidl_DFinder_addSearchPath(
  /* in */ sidl_DFinder self,
  /* in */ const char* path_fragment);

extern struct sidl_DFinder__object* 
  impl_sidl_DFinder_fconnect_sidl_DFinder(char* url, sidl_BaseInterface *_ex);
extern char* impl_sidl_DFinder_fgetURL_sidl_DFinder(struct 
  sidl_DFinder__object* obj);
extern struct sidl_Finder__object* impl_sidl_DFinder_fconnect_sidl_Finder(char* 
  url, sidl_BaseInterface *_ex);
extern char* impl_sidl_DFinder_fgetURL_sidl_Finder(struct sidl_Finder__object* 
  obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_DFinder_fconnect_sidl_ClassInfo(char* url, sidl_BaseInterface *_ex);
extern char* impl_sidl_DFinder_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_DLL__object* impl_sidl_DFinder_fconnect_sidl_DLL(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_DFinder_fgetURL_sidl_DLL(struct sidl_DLL__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_DFinder_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_DFinder_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_DFinder_fconnect_sidl_BaseClass(char* url, sidl_BaseInterface *_ex);
extern char* impl_sidl_DFinder_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
