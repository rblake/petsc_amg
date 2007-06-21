/*
 * File:          sidl_InvViolation_IOR.h
 * Symbol:        sidl.InvViolation-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Release:       $Name: V1-10-0b $
 * Revision:      @(#) $Id: sidl_InvViolation_IOR.h,v 1.4 2005/11/14 21:20:12 painter Exp $
 * Description:   Intermediate Object Representation for sidl.InvViolation
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

#ifndef included_sidl_InvViolation_IOR_h
#define included_sidl_InvViolation_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_SIDLException_IOR_h
#include "sidl_SIDLException_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidl.InvViolation" (version 0.9.3)
 * 
 * <code>InvViolation</code> provides the basic marker for 
 * a invariant exception.
 */

struct sidl_InvViolation__array;
struct sidl_InvViolation__object;

extern struct sidl_InvViolation__object*
sidl_InvViolation__new(void);

extern void sidl_InvViolation__init(
  struct sidl_InvViolation__object* self);
extern void sidl_InvViolation__fini(
  struct sidl_InvViolation__object* self);
extern void sidl_InvViolation__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

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

struct sidl_InvViolation__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidl_InvViolation__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidl_InvViolation__object* self);
  void (*f__exec)(
    /* in */ struct sidl_InvViolation__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidl_InvViolation__object* self);
  void (*f__ctor)(
    /* in */ struct sidl_InvViolation__object* self);
  void (*f__dtor)(
    /* in */ struct sidl_InvViolation__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidl_InvViolation__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidl_InvViolation__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidl_InvViolation__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidl_InvViolation__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidl_InvViolation__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidl_InvViolation__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in sidl.BaseException-v0.9.3 */
  char* (*f_getNote)(
    /* in */ struct sidl_InvViolation__object* self);
  void (*f_setNote)(
    /* in */ struct sidl_InvViolation__object* self,
    /* in */ const char* message);
  char* (*f_getTrace)(
    /* in */ struct sidl_InvViolation__object* self);
  void (*f_addLine)(
    /* in */ struct sidl_InvViolation__object* self,
    /* in */ const char* traceline);
  void (*f_add)(
    /* in */ struct sidl_InvViolation__object* self,
    /* in */ const char* filename,
    /* in */ int32_t lineno,
    /* in */ const char* methodname);
  /* Methods introduced in sidl.SIDLException-v0.9.3 */
  /* Methods introduced in sidl.InvViolation-v0.9.3 */
};

/*
 * Define the class object structure.
 */

struct sidl_InvViolation__object {
  struct sidl_SIDLException__object d_sidl_sidlexception;
  struct sidl_InvViolation__epv*    d_epv;
  void*                             d_data;
};

struct sidl_InvViolation__external {
  struct sidl_InvViolation__object*
  (*createObject)(void);

  struct sidl_SIDLException__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidl_InvViolation__external*
sidl_InvViolation__externals(void);

struct sidl_SIDLException__object* 
  skel_sidl_InvViolation_fconnect_sidl_SIDLException(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_InvViolation_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj); 

struct sidl_InvViolation__object* 
  skel_sidl_InvViolation_fconnect_sidl_InvViolation(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_InvViolation_fgetURL_sidl_InvViolation(struct 
  sidl_InvViolation__object* obj); 

struct sidl_ClassInfo__object* 
  skel_sidl_InvViolation_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_InvViolation_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct sidl_BaseInterface__object* 
  skel_sidl_InvViolation_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_InvViolation_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseException__object* 
  skel_sidl_InvViolation_fconnect_sidl_BaseException(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_InvViolation_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj); 

struct sidl_BaseClass__object* 
  skel_sidl_InvViolation_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_InvViolation_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
