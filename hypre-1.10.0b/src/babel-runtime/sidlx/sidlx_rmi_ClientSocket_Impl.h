/*
 * File:          sidlx_rmi_ClientSocket_Impl.h
 * Symbol:        sidlx.rmi.ClientSocket-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for sidlx.rmi.ClientSocket
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

#ifndef included_sidlx_rmi_ClientSocket_Impl_h
#define included_sidlx_rmi_ClientSocket_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidlx_rmi_ClientSocket_h
#include "sidlx_rmi_ClientSocket.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidlx_rmi_Socket_h
#include "sidlx_rmi_Socket.h"
#endif
#ifndef included_sidl_rmi_NetworkException_h
#include "sidl_rmi_NetworkException.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidlx_rmi_IPv4Socket_h
#include "sidlx_rmi_IPv4Socket.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

#line 41 "../../../babel/runtime/sidlx/sidlx_rmi_ClientSocket_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.ClientSocket._includes) */
#include <stdio.h>
#include <stddef.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "sidlx_rmi_GenNetworkException.h"
#include "sidl_String.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.end(sidlx.rmi.ClientSocket._includes) */
#line 57 "sidlx_rmi_ClientSocket_Impl.h"

/*
 * Private data for class sidlx.rmi.ClientSocket
 */

struct sidlx_rmi_ClientSocket__data {
#line 62 "../../../babel/runtime/sidlx/sidlx_rmi_ClientSocket_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ClientSocket._data) */
  /* insert implementation here: sidlx.rmi.ClientSocket._data (private data members) */
  int addrlen;
  struct sockaddr_in d_serv_addr;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ClientSocket._data) */
#line 70 "sidlx_rmi_ClientSocket_Impl.h"
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_ClientSocket__data*
sidlx_rmi_ClientSocket__get_data(
  sidlx_rmi_ClientSocket);

extern void
sidlx_rmi_ClientSocket__set_data(
  sidlx_rmi_ClientSocket,
  struct sidlx_rmi_ClientSocket__data*);

extern
void
impl_sidlx_rmi_ClientSocket__load(
  void);

extern
void
impl_sidlx_rmi_ClientSocket__ctor(
  /* in */ sidlx_rmi_ClientSocket self);

extern
void
impl_sidlx_rmi_ClientSocket__dtor(
  /* in */ sidlx_rmi_ClientSocket self);

/*
 * User-defined object methods
 */

extern struct sidlx_rmi_ClientSocket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_ClientSocket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_ClientSocket(struct 
  sidlx_rmi_ClientSocket__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_ClientSocket_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_IPv4Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_IPv4Socket(struct 
  sidlx_rmi_IPv4Socket__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_sidlx_rmi_ClientSocket_init(
  /* in */ sidlx_rmi_ClientSocket self,
  /* in */ const char* hostname,
  /* in */ int32_t port,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidlx_rmi_ClientSocket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_ClientSocket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_ClientSocket(struct 
  sidlx_rmi_ClientSocket__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_ClientSocket_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_IPv4Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_IPv4Socket(struct 
  sidlx_rmi_IPv4Socket__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
