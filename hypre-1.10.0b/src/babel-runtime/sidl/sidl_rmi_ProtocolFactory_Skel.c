/*
 * File:          sidl_rmi_ProtocolFactory_Skel.c
 * Symbol:        sidl.rmi.ProtocolFactory-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Release:       $Name: V1-10-0b $
 * Revision:      @(#) $Id: sidl_rmi_ProtocolFactory_Skel.c,v 1.4 2005/11/14 21:20:26 painter Exp $
 * Description:   Server-side glue code for sidl.rmi.ProtocolFactory
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

#include "sidl_rmi_ProtocolFactory_IOR.h"
#include "sidl_rmi_ProtocolFactory.h"
#include <stddef.h>

extern
void
impl_sidl_rmi_ProtocolFactory__load(
  void);

extern
void
impl_sidl_rmi_ProtocolFactory__ctor(
  /* in */ sidl_rmi_ProtocolFactory self);

extern
void
impl_sidl_rmi_ProtocolFactory__dtor(
  /* in */ sidl_rmi_ProtocolFactory self);

extern
sidl_bool
impl_sidl_rmi_ProtocolFactory_addProtocol(
  /* in */ const char* prefix,
  /* in */ const char* typeName,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidl_rmi_ProtocolFactory_getProtocol(
  /* in */ const char* prefix,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_sidl_rmi_ProtocolFactory_deleteProtocol(
  /* in */ const char* prefix,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_rmi_InstanceHandle
impl_sidl_rmi_ProtocolFactory_createInstance(
  /* in */ const char* url,
  /* in */ const char* typeName,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_rmi_InstanceHandle
impl_sidl_rmi_ProtocolFactory_connectInstance(
  /* in */ const char* url,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_rmi_InstanceHandle__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_InstanceHandle(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_InstanceHandle(struct 
  sidl_rmi_InstanceHandle__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_rmi_ProtocolFactory__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_ProtocolFactory(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_ProtocolFactory(struct 
  sidl_rmi_ProtocolFactory__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct sidl_rmi_InstanceHandle__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_InstanceHandle(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_InstanceHandle(struct 
  sidl_rmi_InstanceHandle__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_rmi_ProtocolFactory__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_ProtocolFactory(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_ProtocolFactory(struct 
  sidl_rmi_ProtocolFactory__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
sidl_rmi_ProtocolFactory__set_epv(struct sidl_rmi_ProtocolFactory__epv *epv)
{
  epv->f__ctor = impl_sidl_rmi_ProtocolFactory__ctor;
  epv->f__dtor = impl_sidl_rmi_ProtocolFactory__dtor;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
sidl_rmi_ProtocolFactory__set_sepv(struct sidl_rmi_ProtocolFactory__sepv *sepv)
{
  sepv->f_addProtocol = impl_sidl_rmi_ProtocolFactory_addProtocol;
  sepv->f_getProtocol = impl_sidl_rmi_ProtocolFactory_getProtocol;
  sepv->f_deleteProtocol = impl_sidl_rmi_ProtocolFactory_deleteProtocol;
  sepv->f_createInstance = impl_sidl_rmi_ProtocolFactory_createInstance;
  sepv->f_connectInstance = impl_sidl_rmi_ProtocolFactory_connectInstance;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidl_rmi_ProtocolFactory__call_load(void) { 
  impl_sidl_rmi_ProtocolFactory__load();
}
struct sidl_rmi_InstanceHandle__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_InstanceHandle(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_InstanceHandle(url,
    _ex);
}

char* skel_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_InstanceHandle(struct 
  sidl_rmi_InstanceHandle__object* obj) { 
  return impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_InstanceHandle(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidl_rmi_ProtocolFactory_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_rmi_NetworkException__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_NetworkException(url,
    _ex);
}

char* skel_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) { 
  return impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_NetworkException(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidl_rmi_ProtocolFactory_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_rmi_ProtocolFactory__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_ProtocolFactory(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_ProtocolFactory(url,
    _ex);
}

char* skel_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_ProtocolFactory(struct 
  sidl_rmi_ProtocolFactory__object* obj) { 
  return impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_ProtocolFactory(obj);
}

struct sidl_BaseClass__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidl_rmi_ProtocolFactory_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_BaseClass(obj);
}

struct sidl_rmi_ProtocolFactory__data*
sidl_rmi_ProtocolFactory__get_data(sidl_rmi_ProtocolFactory self)
{
  return (struct sidl_rmi_ProtocolFactory__data*)(self ? self->d_data : NULL);
}

void sidl_rmi_ProtocolFactory__set_data(
  sidl_rmi_ProtocolFactory self,
  struct sidl_rmi_ProtocolFactory__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
