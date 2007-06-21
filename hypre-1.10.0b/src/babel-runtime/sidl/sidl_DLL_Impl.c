/*
 * File:          sidl_DLL_Impl.c
 * Symbol:        sidl.DLL-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Release:       $Name: V1-10-0b $
 * Revision:      @(#) $Id: sidl_DLL_Impl.c,v 1.6 2005/11/14 21:20:24 painter Exp $
 * Description:   Server-side implementation for sidl.DLL
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

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidl.DLL" (version 0.9.3)
 * 
 * The <code>DLL</code> class encapsulates access to a single
 * dynamically linked library.  DLLs are loaded at run-time using
 * the <code>loadLibrary</code> method and later unloaded using
 * <code>unloadLibrary</code>.  Symbols in a loaded library are
 * resolved to an opaque pointer by method <code>lookupSymbol</code>.
 * Class instances are created by <code>createClass</code>.
 */

#include "sidl_DLL_Impl.h"

#line 56 "../../../babel/runtime/sidl/sidl_DLL_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidl.DLL._includes) */
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include "sidl_String.h"
#ifdef HAVE_LIBWWW
#include "wwwconf.h"
#include "WWWLib.h"
#include "WWWInit.h"
#endif

#ifndef NULL
#define NULL 0
#endif
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

#ifdef HAVE_LIBWWW
static char* url_to_local_file(const char* url)
{
  HTRequest* request = NULL;
#ifdef __CYGWIN__
  char* tmpfile = sidl_String_concat3("cyg", tmpnam(NULL), ".dll");
#else
  char* tmpfile = sidl_String_concat3("lib", tmpnam(NULL), ".so");
#endif

  HTProfile_newPreemptiveClient("sidl_DLL", "1.0");
  HTAlert_setInteractive(FALSE);

  request = HTRequest_new();
  HTLoadToFile(url, request, tmpfile);
  HTRequest_delete(request);
  HTProfile_delete();

  return tmpfile;
}
#endif

/*
 * The libtool dynamic loading library must be initialized before any calls
 * are made to library routines.  Initialize only once.
 */
static void check_lt_initialized(void)
{
  static int initialized = FALSE;
  if (!initialized) {
#if defined(PIC) || !defined(SIDL_PURE_STATIC_RUNTIME)
    (void) lt_dlinit();
#endif /* defined(PIC) || !defined(SIDL_PURE_STATIC_RUNTIME) */
    initialized = TRUE;
  }
}

static int s_sidl_debug_init = 0;
static int s_sidl_debug_dlopen = 0;

static void showLoading(const char *dllname)
{
  if (dllname) {
    fprintf(stderr, "Loading %s: ", dllname);
  }
  else {
    fprintf(stderr, "Loading main: ");
  }
}

static void showLoadResult(void *handle)
{
  if (handle) {
    fprintf(stderr, "ok\n");
  }
  else {
#if defined(PIC) || !defined(SIDL_PURE_STATIC_RUNTIME)
    const char *errmsg = lt_dlerror(); 
    fprintf(stderr,"ERROR\n%s\n", errmsg);
#endif
  }
}

/* DO-NOT-DELETE splicer.end(sidl.DLL._includes) */
#line 142 "sidl_DLL_Impl.c"

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_DLL__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_DLL__load(
  void)
{
#line 156 "../../../babel/runtime/sidl/sidl_DLL_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.DLL._load) */
  /* Insert the implementation of the static class initializer method here... */
  /* DO-NOT-DELETE splicer.end(sidl.DLL._load) */
#line 162 "sidl_DLL_Impl.c"
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_DLL__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_DLL__ctor(
  /* in */ sidl_DLL self)
{
#line 174 "../../../babel/runtime/sidl/sidl_DLL_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.DLL._ctor) */
  struct sidl_DLL__data* data =
    (struct sidl_DLL__data*) malloc(sizeof (struct sidl_DLL__data));

  if (!s_sidl_debug_init) {
    s_sidl_debug_dlopen = (getenv("sidl_DEBUG_DLOPEN") || getenv("SIDL_DEBUG_DLOPEN"));
    s_sidl_debug_init = 1;
  }
  data->d_library_handle = NULL;
  data->d_library_name   = NULL;

  sidl_DLL__set_data(self, data);
  /* DO-NOT-DELETE splicer.end(sidl.DLL._ctor) */
#line 192 "sidl_DLL_Impl.c"
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_DLL__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_DLL__dtor(
  /* in */ sidl_DLL self)
{
#line 203 "../../../babel/runtime/sidl/sidl_DLL_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.DLL._dtor) */
  struct sidl_DLL__data *data = sidl_DLL__get_data(self);

  impl_sidl_DLL_unloadLibrary(self);

  free((void*) data);
  sidl_DLL__set_data(self, NULL);
  /* DO-NOT-DELETE splicer.end(sidl.DLL._dtor) */
#line 218 "sidl_DLL_Impl.c"
}

/*
 * Load a dynamic link library using the specified URI.  The
 * URI may be of the form "main:", "lib:", "file:", "ftp:", or
 * "http:".  A URI that starts with any other protocol string
 * is assumed to be a file name.  The "main:" URI creates a
 * library that allows access to global symbols in the running
 * program's main address space.  The "lib:X" URI converts the
 * library "X" into a platform-specific name (e.g., libX.so) and
 * loads that library.  The "file:" URI opens the DLL from the
 * specified file path.  The "ftp:" and "http:" URIs copy the
 * specified library from the remote site into a local temporary
 * file and open that file.  This method returns true if the
 * DLL was loaded successfully and false otherwise.  Note that
 * the "ftp:" and "http:" protocols are valid only if the W3C
 * WWW library is available.
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
 *                     now (at load time).
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_DLL_loadLibrary"

#ifdef __cplusplus
extern "C"
#endif
sidl_bool
impl_sidl_DLL_loadLibrary(
  /* in */ sidl_DLL self,
  /* in */ const char* uri,
  /* in */ sidl_bool loadGlobally,
  /* in */ sidl_bool loadLazy)
{
#line 260 "../../../babel/runtime/sidl/sidl_DLL_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.DLL.loadLibrary) */
  struct sidl_DLL__data *data = sidl_DLL__get_data(self);

  int         ok      = FALSE;
  lt_dlhandle handle  = NULL;
  char*       dllfile = NULL;
  char*       dllname = NULL;

  if (data->d_library_handle) {
    impl_sidl_DLL_unloadLibrary(self);
  }

  if (sidl_String_equals(uri, "main:")) {
    dllfile = NULL;
    dllname = sidl_String_strdup(uri);
#if !defined(PIC) && defined(SIDL_PURE_STATIC_RUNTIME)
    data->d_library_handle = NULL;
    data->d_library_name = dllname;
    return TRUE;
#endif

  } else if (sidl_String_startsWith(uri, "lib:")) {
    char* dll = sidl_String_substring(uri, 4);
    dllfile = sidl_String_concat3("lib", dll, ".la");
    dllname = sidl_String_strdup(uri);
    sidl_String_free(dll);

  } else if (sidl_String_startsWith(uri, "file:")) {
    dllfile = sidl_String_substring(uri, 5);
    dllname = sidl_String_strdup(uri);

#ifdef HAVE_LIBWWW
  } else if (sidl_String_startsWith(uri, "ftp:")) {
    dllfile = url_to_local_file(uri);
    dllname = sidl_String_strdup(uri);

  } else if (sidl_String_startsWith(uri, "http:")) {
    dllfile = url_to_local_file(uri);
    dllname = sidl_String_strdup(uri);
#endif

  } else {
    dllfile = sidl_String_strdup(uri);
    dllname = sidl_String_concat2("file:", uri);
  }

  if (s_sidl_debug_dlopen) showLoading(dllfile);
  check_lt_initialized();
#if defined(PIC) || !defined(SIDL_PURE_STATIC_RUNTIME)
  handle = lt_dlopen(dllfile, loadGlobally, loadLazy);
#else
  handle = NULL;
#endif
  if (s_sidl_debug_dlopen) showLoadResult((void *)handle);
  sidl_String_free(dllfile);

  if (handle) {
    ok = TRUE;
    data->d_library_handle = handle;
    data->d_library_name   = dllname;
  } else {
    ok = FALSE;
    sidl_String_free(dllname);
  }

  return ok;
  /* DO-NOT-DELETE splicer.end(sidl.DLL.loadLibrary) */
#line 336 "sidl_DLL_Impl.c"
}

/*
 * Get the library name.  This is the name used to load the
 * library in <code>loadLibrary</code> except that all file names
 * contain the "file:" protocol.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_DLL_getName"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidl_DLL_getName(
  /* in */ sidl_DLL self)
{
#line 345 "../../../babel/runtime/sidl/sidl_DLL_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.DLL.getName) */
  struct sidl_DLL__data *data = sidl_DLL__get_data(self);

  char* name = NULL;
  if (data->d_library_name) {
    name = sidl_String_strdup(data->d_library_name);
  }
  return name;
  /* DO-NOT-DELETE splicer.end(sidl.DLL.getName) */
#line 365 "sidl_DLL_Impl.c"
}

/*
 * Unload the dynamic link library.  The library may no longer
 * be used to access symbol names.  When the library is actually
 * unloaded from the memory image depends on details of the operating
 * system.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_DLL_unloadLibrary"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_DLL_unloadLibrary(
  /* in */ sidl_DLL self)
{
#line 373 "../../../babel/runtime/sidl/sidl_DLL_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.DLL.unloadLibrary) */
  struct sidl_DLL__data *data = sidl_DLL__get_data(self);

  if (data->d_library_handle) {
#if defined(PIC) || !defined(SIDL_PURE_STATIC_RUNTIME)
    (void) lt_dlclose(data->d_library_handle);
#endif
    sidl_String_free(data->d_library_name);
    data->d_library_handle = NULL;
    data->d_library_name   = NULL;
  }
  /* DO-NOT-DELETE splicer.end(sidl.DLL.unloadLibrary) */
#line 398 "sidl_DLL_Impl.c"
}

/*
 * Lookup a symbol from the DLL and return the associated pointer.
 * A null value is returned if the name does not exist.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_DLL_lookupSymbol"

#ifdef __cplusplus
extern "C"
#endif
void*
impl_sidl_DLL_lookupSymbol(
  /* in */ sidl_DLL self,
  /* in */ const char* linker_name)
{
#line 403 "../../../babel/runtime/sidl/sidl_DLL_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.DLL.lookupSymbol) */
  struct sidl_DLL__data *data = sidl_DLL__get_data(self);

  void* address = NULL;
  if (data->d_library_handle) {
#if defined(PIC) || !defined(SIDL_PURE_STATIC_RUNTIME)
    address = (void*) lt_dlsym(data->d_library_handle, linker_name);
#endif
  }
  return address;
  /* DO-NOT-DELETE splicer.end(sidl.DLL.lookupSymbol) */
#line 429 "sidl_DLL_Impl.c"
}

/*
 * Create an instance of the sidl class.  If the class constructor
 * is not defined in this DLL, then return null.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_DLL_createClass"

#ifdef __cplusplus
extern "C"
#endif
sidl_BaseClass
impl_sidl_DLL_createClass(
  /* in */ sidl_DLL self,
  /* in */ const char* sidl_name)
{
#line 432 "../../../babel/runtime/sidl/sidl_DLL_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.DLL.createClass) */
  struct sidl_DLL__data *data = sidl_DLL__get_data(self);
  sidl_BaseClass (*ctor)(void) = NULL;

  if (data->d_library_handle) {
    char* linker = sidl_String_concat2(sidl_name, "__new");
    char* ptr    = linker;

    while (*ptr) {
      if (*ptr == '.') {
        *ptr = '_';
      }
      ptr++;
    }

#if defined(PIC) || !defined(SIDL_PURE_STATIC_RUNTIME)
    ctor = (sidl_BaseClass (*)(void)) lt_dlsym(data->d_library_handle, linker);
#else
    ctor = NULL;
#endif
    sidl_String_free(linker);
  }

  return ctor ? (*ctor)() : NULL;
  /* DO-NOT-DELETE splicer.end(sidl.DLL.createClass) */
#line 474 "sidl_DLL_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_DLL__object* impl_sidl_DLL_fconnect_sidl_DLL(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_DLL__connect(url, _ex);
}
char * impl_sidl_DLL_fgetURL_sidl_DLL(struct sidl_DLL__object* obj) {
  return sidl_DLL__getURL(obj);
}
struct sidl_ClassInfo__object* impl_sidl_DLL_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidl_DLL_fgetURL_sidl_ClassInfo(struct sidl_ClassInfo__object* obj) 
  {
  return sidl_ClassInfo__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidl_DLL_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidl_DLL_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* impl_sidl_DLL_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidl_DLL_fgetURL_sidl_BaseClass(struct sidl_BaseClass__object* obj) 
  {
  return sidl_BaseClass__getURL(obj);
}
