/*
 * File:          sidl_rmi_ConnectRegistry_Skel.c
 * Symbol:        sidl.rmi.ConnectRegistry-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Release:       $Name: V1-10-0b $
 * Revision:      @(#) $Id: sidl_rmi_ConnectRegistry_Skel.c,v 1.4 2005/11/14 21:20:26 painter Exp $
 * Description:   Server-side glue code for sidl.rmi.ConnectRegistry
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

#include "sidl_rmi_ConnectRegistry_IOR.h"
#include "sidl_rmi_ConnectRegistry.h"
#include <stddef.h>

extern
void
impl_sidl_rmi_ConnectRegistry__load(
  void);

extern
void
impl_sidl_rmi_ConnectRegistry__ctor(
  /* in */ sidl_rmi_ConnectRegistry self);

extern
void
impl_sidl_rmi_ConnectRegistry__dtor(
  /* in */ sidl_rmi_ConnectRegistry self);

extern
void
impl_sidl_rmi_ConnectRegistry_registerConnect(
  /* in */ const char* key,
  /* in */ void* func);

extern
void*
impl_sidl_rmi_ConnectRegistry_getConnect(
  /* in */ const char* key);

extern
void*
impl_sidl_rmi_ConnectRegistry_removeConnect(
  /* in */ const char* key);

extern struct sidl_rmi_ConnectRegistry__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_rmi_ConnectRegistry(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_ConnectRegistry_fgetURL_sidl_rmi_ConnectRegistry(struct 
  sidl_rmi_ConnectRegistry__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_ConnectRegistry_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_ConnectRegistry_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_ConnectRegistry_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct sidl_rmi_ConnectRegistry__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_rmi_ConnectRegistry(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_ConnectRegistry_fgetURL_sidl_rmi_ConnectRegistry(struct 
  sidl_rmi_ConnectRegistry__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_ConnectRegistry_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_ConnectRegistry_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_ConnectRegistry_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
sidl_rmi_ConnectRegistry__set_epv(struct sidl_rmi_ConnectRegistry__epv *epv)
{
  epv->f__ctor = impl_sidl_rmi_ConnectRegistry__ctor;
  epv->f__dtor = impl_sidl_rmi_ConnectRegistry__dtor;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
sidl_rmi_ConnectRegistry__set_sepv(struct sidl_rmi_ConnectRegistry__sepv *sepv)
{
  sepv->f_registerConnect = impl_sidl_rmi_ConnectRegistry_registerConnect;
  sepv->f_getConnect = impl_sidl_rmi_ConnectRegistry_getConnect;
  sepv->f_removeConnect = impl_sidl_rmi_ConnectRegistry_removeConnect;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidl_rmi_ConnectRegistry__call_load(void) { 
  impl_sidl_rmi_ConnectRegistry__load();
}
struct sidl_rmi_ConnectRegistry__object* 
  skel_sidl_rmi_ConnectRegistry_fconnect_sidl_rmi_ConnectRegistry(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ConnectRegistry_fconnect_sidl_rmi_ConnectRegistry(url,
    _ex);
}

char* skel_sidl_rmi_ConnectRegistry_fgetURL_sidl_rmi_ConnectRegistry(struct 
  sidl_rmi_ConnectRegistry__object* obj) { 
  return impl_sidl_rmi_ConnectRegistry_fgetURL_sidl_rmi_ConnectRegistry(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidl_rmi_ConnectRegistry_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ConnectRegistry_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidl_rmi_ConnectRegistry_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidl_rmi_ConnectRegistry_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidl_rmi_ConnectRegistry_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ConnectRegistry_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidl_rmi_ConnectRegistry_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidl_rmi_ConnectRegistry_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_sidl_rmi_ConnectRegistry_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ConnectRegistry_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidl_rmi_ConnectRegistry_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidl_rmi_ConnectRegistry_fgetURL_sidl_BaseClass(obj);
}

struct sidl_rmi_ConnectRegistry__data*
sidl_rmi_ConnectRegistry__get_data(sidl_rmi_ConnectRegistry self)
{
  return (struct sidl_rmi_ConnectRegistry__data*)(self ? self->d_data : NULL);
}

void sidl_rmi_ConnectRegistry__set_data(
  sidl_rmi_ConnectRegistry self,
  struct sidl_rmi_ConnectRegistry__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
