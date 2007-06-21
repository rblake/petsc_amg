/*
 * File:          sidlx_rmi_GenNetworkException_IOR.h
 * Symbol:        sidlx.rmi.GenNetworkException-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Intermediate Object Representation for sidlx.rmi.GenNetworkException
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_sidlx_rmi_GenNetworkException_IOR_h
#define included_sidlx_rmi_GenNetworkException_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_rmi_NetworkException_IOR_h
#include "sidl_rmi_NetworkException_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidlx.rmi.GenNetworkException" (version 0.1)
 * 
 * Generic Network Exception
 */

struct sidlx_rmi_GenNetworkException__array;
struct sidlx_rmi_GenNetworkException__object;

extern struct sidlx_rmi_GenNetworkException__object*
sidlx_rmi_GenNetworkException__new(void);

extern void sidlx_rmi_GenNetworkException__init(
  struct sidlx_rmi_GenNetworkException__object* self);
extern void sidlx_rmi_GenNetworkException__fini(
  struct sidlx_rmi_GenNetworkException__object* self);
extern void sidlx_rmi_GenNetworkException__IOR_version(int32_t *major,
  int32_t *minor);

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

struct sidlx_rmi_GenNetworkException__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidlx_rmi_GenNetworkException__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidlx_rmi_GenNetworkException__object* self);
  void (*f__exec)(
    /* in */ struct sidlx_rmi_GenNetworkException__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidlx_rmi_GenNetworkException__object* self);
  void (*f__ctor)(
    /* in */ struct sidlx_rmi_GenNetworkException__object* self);
  void (*f__dtor)(
    /* in */ struct sidlx_rmi_GenNetworkException__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidlx_rmi_GenNetworkException__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidlx_rmi_GenNetworkException__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidlx_rmi_GenNetworkException__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidlx_rmi_GenNetworkException__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidlx_rmi_GenNetworkException__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidlx_rmi_GenNetworkException__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in sidl.BaseException-v0.9.3 */
  char* (*f_getNote)(
    /* in */ struct sidlx_rmi_GenNetworkException__object* self);
  void (*f_setNote)(
    /* in */ struct sidlx_rmi_GenNetworkException__object* self,
    /* in */ const char* message);
  char* (*f_getTrace)(
    /* in */ struct sidlx_rmi_GenNetworkException__object* self);
  void (*f_addLine)(
    /* in */ struct sidlx_rmi_GenNetworkException__object* self,
    /* in */ const char* traceline);
  void (*f_add)(
    /* in */ struct sidlx_rmi_GenNetworkException__object* self,
    /* in */ const char* filename,
    /* in */ int32_t lineno,
    /* in */ const char* methodname);
  /* Methods introduced in sidl.SIDLException-v0.9.3 */
  /* Methods introduced in sidl.io.IOException-v0.9.3 */
  /* Methods introduced in sidl.rmi.NetworkException-v0.9.3 */
  /* Methods introduced in sidlx.rmi.GenNetworkException-v0.1 */
};

/*
 * Define the class object structure.
 */

struct sidlx_rmi_GenNetworkException__object {
  struct sidl_rmi_NetworkException__object   d_sidl_rmi_networkexception;
  struct sidlx_rmi_GenNetworkException__epv* d_epv;
  void*                                      d_data;
};

struct sidlx_rmi_GenNetworkException__external {
  struct sidlx_rmi_GenNetworkException__object*
  (*createObject)(void);

  struct sidl_rmi_NetworkException__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_rmi_GenNetworkException__external*
sidlx_rmi_GenNetworkException__externals(void);

struct sidlx_rmi_GenNetworkException__object* 
  skel_sidlx_rmi_GenNetworkException_fconnect_sidlx_rmi_GenNetworkException(
  char* url, struct sidl_BaseInterface__object **_ex);
char* 
  skel_sidlx_rmi_GenNetworkException_fgetURL_sidlx_rmi_GenNetworkException(
  struct sidlx_rmi_GenNetworkException__object* obj); 

struct sidl_SIDLException__object* 
  skel_sidlx_rmi_GenNetworkException_fconnect_sidl_SIDLException(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_GenNetworkException_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj); 

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_GenNetworkException_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_GenNetworkException_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct sidl_io_IOException__object* 
  skel_sidlx_rmi_GenNetworkException_fconnect_sidl_io_IOException(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_GenNetworkException_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj); 

struct sidl_rmi_NetworkException__object* 
  skel_sidlx_rmi_GenNetworkException_fconnect_sidl_rmi_NetworkException(char* 
  url, struct sidl_BaseInterface__object **_ex);
char* 
  skel_sidlx_rmi_GenNetworkException_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj); 

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseException__object* 
  skel_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseException(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj); 

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
