/*
 * File:          sidlx_rmi_Simsponse_IOR.h
 * Symbol:        sidlx.rmi.Simsponse-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Intermediate Object Representation for sidlx.rmi.Simsponse
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_sidlx_rmi_Simsponse_IOR_h
#define included_sidlx_rmi_Simsponse_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif
#ifndef included_sidl_io_Deserializer_IOR_h
#include "sidl_io_Deserializer_IOR.h"
#endif
#ifndef included_sidl_rmi_Response_IOR_h
#include "sidl_rmi_Response_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidlx.rmi.Simsponse" (version 0.1)
 * 
 * implementation of Response using the Simocol (simple-protocol), 
 * 	contains all the serialization code
 */

struct sidlx_rmi_Simsponse__array;
struct sidlx_rmi_Simsponse__object;

extern struct sidlx_rmi_Simsponse__object*
sidlx_rmi_Simsponse__new(void);

extern void sidlx_rmi_Simsponse__init(
  struct sidlx_rmi_Simsponse__object* self);
extern void sidlx_rmi_Simsponse__fini(
  struct sidlx_rmi_Simsponse__object* self);
extern void sidlx_rmi_Simsponse__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseException__array;
struct sidl_BaseException__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_io_IOException__array;
struct sidl_io_IOException__object;
struct sidl_io_Serializer__array;
struct sidl_io_Serializer__object;
struct sidl_rmi_NetworkException__array;
struct sidl_rmi_NetworkException__object;
struct sidlx_rmi_Socket__array;
struct sidlx_rmi_Socket__object;

/*
 * Declare the method entry point vector.
 */

struct sidlx_rmi_Simsponse__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidlx_rmi_Simsponse__object* self);
  void (*f__exec)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidlx_rmi_Simsponse__object* self);
  void (*f__ctor)(
    /* in */ struct sidlx_rmi_Simsponse__object* self);
  void (*f__dtor)(
    /* in */ struct sidlx_rmi_Simsponse__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidlx_rmi_Simsponse__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidlx_rmi_Simsponse__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidlx_rmi_Simsponse__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in sidl.io.Deserializer-v0.9.3 */
  void (*f_unpackBool)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* in */ const char* key,
    /* out */ sidl_bool* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackChar)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* in */ const char* key,
    /* out */ char* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackInt)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* in */ const char* key,
    /* out */ int32_t* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackLong)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* in */ const char* key,
    /* out */ int64_t* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackFloat)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* in */ const char* key,
    /* out */ float* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackDouble)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* in */ const char* key,
    /* out */ double* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackFcomplex)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* in */ const char* key,
    /* out */ struct sidl_fcomplex* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackDcomplex)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* in */ const char* key,
    /* out */ struct sidl_dcomplex* value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_unpackString)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* in */ const char* key,
    /* out */ char** value,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.rmi.Response-v0.9.3 */
  struct sidl_BaseException__object* (*f_getExceptionThrown)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_done)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidlx.rmi.Simsponse-v0.1 */
  void (*f_init)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* in */ const char* methodName,
    /* in */ const char* className,
    /* in */ const char* objectid,
    /* in */ struct sidlx_rmi_Socket__object* sock,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_getMethodName)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_getClassName)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_getObjectID)(
    /* in */ struct sidlx_rmi_Simsponse__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Define the class object structure.
 */

struct sidlx_rmi_Simsponse__object {
  struct sidl_BaseClass__object       d_sidl_baseclass;
  struct sidl_io_Deserializer__object d_sidl_io_deserializer;
  struct sidl_rmi_Response__object    d_sidl_rmi_response;
  struct sidlx_rmi_Simsponse__epv*    d_epv;
  void*                               d_data;
};

struct sidlx_rmi_Simsponse__external {
  struct sidlx_rmi_Simsponse__object*
  (*createObject)(void);

  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_rmi_Simsponse__external*
sidlx_rmi_Simsponse__externals(void);

struct sidl_rmi_Response__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidl_rmi_Response(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_Simsponse_fgetURL_sidl_rmi_Response(struct 
  sidl_rmi_Response__object* obj); 

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_Simsponse_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct sidl_io_Deserializer__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidl_io_Deserializer(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_Simsponse_fgetURL_sidl_io_Deserializer(struct 
  sidl_io_Deserializer__object* obj); 

struct sidlx_rmi_Simsponse__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Simsponse(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_Simsponse_fgetURL_sidlx_rmi_Simsponse(struct 
  sidlx_rmi_Simsponse__object* obj); 

struct sidl_io_IOException__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidl_io_IOException(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_Simsponse_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj); 

struct sidl_rmi_NetworkException__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidl_rmi_NetworkException(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_Simsponse_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj); 

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Socket(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_Simsponse_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj); 

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_Simsponse_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseException__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidl_BaseException(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_Simsponse_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj); 

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_rmi_Simsponse_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
