/*
 * File:          sidl_rmi_InstanceHandle_Module.c
 * Symbol:        sidl.rmi.InstanceHandle-v0.9.3
 * Symbol Type:   interface
 * Babel Version: 0.10.12
 * Release:       $Name: V1-10-0b $
 * Revision:      @(#) $Id: sidl_rmi_InstanceHandle_Module.c,v 1.4 2005/11/14 21:20:20 painter Exp $
 * Description:   implement a C extension type for a sidl extendable
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

/*
 * THIS CODE IS AUTOMATICALLY GENERATED BY THE BABEL
 * COMPILER. DO NOT EDIT THIS!
 * 
 * This file contains the implementation of a Python C
 * extension type (i.e. a Python type implemented in C).
 * This extension type provides Python interface to the
 * sidl type sidl.rmi.InstanceHandle.
 */


/**
 * Symbol "sidl.rmi.InstanceHandle" (version 0.9.3)
 * 
 * This interface holds the state information for handles to remote
 * objects.  Client-side messaging libraries are expected to implement
 * <code>sidl.rmi.InstanceHandle</code>, <code>sidl.rmi.Invocation</code>
 * and <code>sidl.rmi.Response</code>.
 * 
 *  When a connection is created between a stub and a real object:
 *       sidl_rmi_InstanceHandle c = sidl_rmi_ProtocolFactory_createInstance( url, typeName );
 * 
 *  When a method is invoked:
 *       sidl_rmi_Invocation i = sidl_rmi_InstanceHandle_createInvocationHandle( methodname );
 *       sidl_rmi_Invocation_packDouble( i, "input_val" , 2.0 );
 *       sidl_rmi_Invocation_packString( i, "input_str", "Hello" );
 *       ...
 *       sidl_rmi_Response r = sidl_rmi_Invocation_invokeMethod( i );
 *       sidl_rmi_Response_unpackBool( i, "_retval", &succeeded );
 *       sidl_rmi_Response_unpackFloat( i, "output_val", &f );
 * 
 */
#define sidl_rmi_InstanceHandle_INTERNAL 1
#include "sidl_rmi_InstanceHandle_Module.h"
#ifndef included_sidl_rmi_InstanceHandle_IOR_h
#include "sidl_rmi_InstanceHandle_IOR.h"
#endif
#include "sidlObjA.h"
#include "sidlPyArrays.h"
#include "Numeric/arrayobject.h"
#ifndef included_sidl_Loader_h
#include "sidl_Loader.h"
#endif
#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#include "sidl_BaseInterface_Module.h"
#include "sidl_ClassInfo_Module.h"
#include "sidl_rmi_Invocation_Module.h"
#include "sidl_rmi_NetworkException_Module.h"
#include <stdlib.h>
#include <string.h>

staticforward PyTypeObject _sidl_rmi_InstanceHandleType;

static PyObject *
pStub_InstanceHandle_createInvocation(PyObject *_self, PyObject *_args,       \
  PyObject *_kwdict) {
  PyObject *_return_value = NULL;
  struct sidl_rmi_InstanceHandle__object *_self_ior =
    ((struct sidl_rmi_InstanceHandle__object *)
     sidl_Cast(_self, "sidl.rmi.InstanceHandle"));
  if (_self_ior) {
    char* methodName = NULL;
    struct sidl_BaseInterface__object *_exception = NULL;
    static char *_kwlist[] = {
      "methodName",
      NULL
    };
    const int _okay = PyArg_ParseTupleAndKeywords(
      _args, _kwdict, 
      "z", _kwlist,
      &methodName);
    if (_okay) {
      struct sidl_rmi_Invocation__object* _return = NULL;
      _return = (*(_self_ior->d_epv->f_createInvocation))(_self_ior->d_object,\
        methodName, &_exception);
      if (_exception) {
        struct sidl_rmi_NetworkException__object *_ex0;
        if ((_ex0 = (struct sidl_rmi_NetworkException__object *)
          sidl_PyExceptionCast(_exception, "sidl.rmi.NetworkException")))
        {
          PyObject *obj = sidl_rmi_NetworkException__wrap(_ex0);
          PyObject *_args = PyTuple_New(1);
          PyTuple_SetItem(_args, 0, obj);
          obj = PyObject_CallObject(sidl_rmi_NetworkException__type, _args);
          PyErr_SetObject(sidl_rmi_NetworkException__type, obj);
          Py_XDECREF(_args);
        }
      }
      else {
        _return_value = Py_BuildValue(
          "O&",
          (void *)sidl_rmi_Invocation__wrap, _return);
      }
    }
  }
  else {
    PyErr_SetString(PyExc_TypeError, 
      "self pointer is not a sidl.rmi.InstanceHandle");
  }
  return _return_value;
}

static PyObject *
pStub_InstanceHandle_getObjectID(PyObject *_self, PyObject *_args,            \
  PyObject *_kwdict) {
  PyObject *_return_value = NULL;
  struct sidl_rmi_InstanceHandle__object *_self_ior =
    ((struct sidl_rmi_InstanceHandle__object *)
     sidl_Cast(_self, "sidl.rmi.InstanceHandle"));
  if (_self_ior) {
    struct sidl_BaseInterface__object *_exception = NULL;
    static char *_kwlist[] = {
      NULL
    };
    const int _okay = PyArg_ParseTupleAndKeywords(
      _args, _kwdict, 
      "", _kwlist);
    if (_okay) {
      char* _return = NULL;
      _return = (*(_self_ior->d_epv->f_getObjectID))(_self_ior->d_object,     \
        &_exception);
      if (_exception) {
        struct sidl_rmi_NetworkException__object *_ex0;
        if ((_ex0 = (struct sidl_rmi_NetworkException__object *)
          sidl_PyExceptionCast(_exception, "sidl.rmi.NetworkException")))
        {
          PyObject *obj = sidl_rmi_NetworkException__wrap(_ex0);
          PyObject *_args = PyTuple_New(1);
          PyTuple_SetItem(_args, 0, obj);
          obj = PyObject_CallObject(sidl_rmi_NetworkException__type, _args);
          PyErr_SetObject(sidl_rmi_NetworkException__type, obj);
          Py_XDECREF(_args);
        }
      }
      else {
        _return_value = Py_BuildValue(
          "z",
          _return);
      }
      free((void *)_return);
    }
  }
  else {
    PyErr_SetString(PyExc_TypeError, 
      "self pointer is not a sidl.rmi.InstanceHandle");
  }
  return _return_value;
}

static PyObject *
pStub_InstanceHandle_close(PyObject *_self, PyObject *_args,                  \
  PyObject *_kwdict) {
  PyObject *_return_value = NULL;
  struct sidl_rmi_InstanceHandle__object *_self_ior =
    ((struct sidl_rmi_InstanceHandle__object *)
     sidl_Cast(_self, "sidl.rmi.InstanceHandle"));
  if (_self_ior) {
    struct sidl_BaseInterface__object *_exception = NULL;
    static char *_kwlist[] = {
      NULL
    };
    const int _okay = PyArg_ParseTupleAndKeywords(
      _args, _kwdict, 
      "", _kwlist);
    if (_okay) {
      sidl_bool _return = (sidl_bool) 0;
      int _proxy__return;
      _return = (*(_self_ior->d_epv->f_close))(_self_ior->d_object,           \
        &_exception);
      _proxy__return = _return;
      if (_exception) {
        struct sidl_rmi_NetworkException__object *_ex0;
        if ((_ex0 = (struct sidl_rmi_NetworkException__object *)
          sidl_PyExceptionCast(_exception, "sidl.rmi.NetworkException")))
        {
          PyObject *obj = sidl_rmi_NetworkException__wrap(_ex0);
          PyObject *_args = PyTuple_New(1);
          PyTuple_SetItem(_args, 0, obj);
          obj = PyObject_CallObject(sidl_rmi_NetworkException__type, _args);
          PyErr_SetObject(sidl_rmi_NetworkException__type, obj);
          Py_XDECREF(_args);
        }
      }
      else {
        _return_value = Py_BuildValue(
          "i",
          _proxy__return);
      }
    }
  }
  else {
    PyErr_SetString(PyExc_TypeError, 
      "self pointer is not a sidl.rmi.InstanceHandle");
  }
  return _return_value;
}

static PyObject *
pStub_InstanceHandle_getProtocol(PyObject *_self, PyObject *_args,            \
  PyObject *_kwdict) {
  PyObject *_return_value = NULL;
  struct sidl_rmi_InstanceHandle__object *_self_ior =
    ((struct sidl_rmi_InstanceHandle__object *)
     sidl_Cast(_self, "sidl.rmi.InstanceHandle"));
  if (_self_ior) {
    struct sidl_BaseInterface__object *_exception = NULL;
    static char *_kwlist[] = {
      NULL
    };
    const int _okay = PyArg_ParseTupleAndKeywords(
      _args, _kwdict, 
      "", _kwlist);
    if (_okay) {
      char* _return = NULL;
      _return = (*(_self_ior->d_epv->f_getProtocol))(_self_ior->d_object,     \
        &_exception);
      if (_exception) {
        struct sidl_rmi_NetworkException__object *_ex0;
        if ((_ex0 = (struct sidl_rmi_NetworkException__object *)
          sidl_PyExceptionCast(_exception, "sidl.rmi.NetworkException")))
        {
          PyObject *obj = sidl_rmi_NetworkException__wrap(_ex0);
          PyObject *_args = PyTuple_New(1);
          PyTuple_SetItem(_args, 0, obj);
          obj = PyObject_CallObject(sidl_rmi_NetworkException__type, _args);
          PyErr_SetObject(sidl_rmi_NetworkException__type, obj);
          Py_XDECREF(_args);
        }
      }
      else {
        _return_value = Py_BuildValue(
          "z",
          _return);
      }
      free((void *)_return);
    }
  }
  else {
    PyErr_SetString(PyExc_TypeError, 
      "self pointer is not a sidl.rmi.InstanceHandle");
  }
  return _return_value;
}

static PyObject *
pStub_InstanceHandle_initConnect(PyObject *_self, PyObject *_args,            \
  PyObject *_kwdict) {
  PyObject *_return_value = NULL;
  struct sidl_rmi_InstanceHandle__object *_self_ior =
    ((struct sidl_rmi_InstanceHandle__object *)
     sidl_Cast(_self, "sidl.rmi.InstanceHandle"));
  if (_self_ior) {
    char* url = NULL;
    struct sidl_BaseInterface__object *_exception = NULL;
    static char *_kwlist[] = {
      "url",
      NULL
    };
    const int _okay = PyArg_ParseTupleAndKeywords(
      _args, _kwdict, 
      "z", _kwlist,
      &url);
    if (_okay) {
      sidl_bool _return = (sidl_bool) 0;
      int _proxy__return;
      _return = (*(_self_ior->d_epv->f_initConnect))(_self_ior->d_object, url,\
        &_exception);
      _proxy__return = _return;
      if (_exception) {
        struct sidl_rmi_NetworkException__object *_ex0;
        if ((_ex0 = (struct sidl_rmi_NetworkException__object *)
          sidl_PyExceptionCast(_exception, "sidl.rmi.NetworkException")))
        {
          PyObject *obj = sidl_rmi_NetworkException__wrap(_ex0);
          PyObject *_args = PyTuple_New(1);
          PyTuple_SetItem(_args, 0, obj);
          obj = PyObject_CallObject(sidl_rmi_NetworkException__type, _args);
          PyErr_SetObject(sidl_rmi_NetworkException__type, obj);
          Py_XDECREF(_args);
        }
      }
      else {
        _return_value = Py_BuildValue(
          "i",
          _proxy__return);
      }
    }
  }
  else {
    PyErr_SetString(PyExc_TypeError, 
      "self pointer is not a sidl.rmi.InstanceHandle");
  }
  return _return_value;
}

static PyObject *
pStub_InstanceHandle_initCreate(PyObject *_self, PyObject *_args,             \
  PyObject *_kwdict) {
  PyObject *_return_value = NULL;
  struct sidl_rmi_InstanceHandle__object *_self_ior =
    ((struct sidl_rmi_InstanceHandle__object *)
     sidl_Cast(_self, "sidl.rmi.InstanceHandle"));
  if (_self_ior) {
    char* url = NULL;
    char* typeName = NULL;
    struct sidl_BaseInterface__object *_exception = NULL;
    static char *_kwlist[] = {
      "url",
      "typeName",
      NULL
    };
    const int _okay = PyArg_ParseTupleAndKeywords(
      _args, _kwdict, 
      "zz", _kwlist,
      &url,
      &typeName);
    if (_okay) {
      sidl_bool _return = (sidl_bool) 0;
      int _proxy__return;
      _return = (*(_self_ior->d_epv->f_initCreate))(_self_ior->d_object, url, \
        typeName, &_exception);
      _proxy__return = _return;
      if (_exception) {
        struct sidl_rmi_NetworkException__object *_ex0;
        if ((_ex0 = (struct sidl_rmi_NetworkException__object *)
          sidl_PyExceptionCast(_exception, "sidl.rmi.NetworkException")))
        {
          PyObject *obj = sidl_rmi_NetworkException__wrap(_ex0);
          PyObject *_args = PyTuple_New(1);
          PyTuple_SetItem(_args, 0, obj);
          obj = PyObject_CallObject(sidl_rmi_NetworkException__type, _args);
          PyErr_SetObject(sidl_rmi_NetworkException__type, obj);
          Py_XDECREF(_args);
        }
      }
      else {
        _return_value = Py_BuildValue(
          "i",
          _proxy__return);
      }
    }
  }
  else {
    PyErr_SetString(PyExc_TypeError, 
      "self pointer is not a sidl.rmi.InstanceHandle");
  }
  return _return_value;
}

static PyObject *
pStub_InstanceHandle_getURL(PyObject *_self, PyObject *_args,                 \
  PyObject *_kwdict) {
  PyObject *_return_value = NULL;
  struct sidl_rmi_InstanceHandle__object *_self_ior =
    ((struct sidl_rmi_InstanceHandle__object *)
     sidl_Cast(_self, "sidl.rmi.InstanceHandle"));
  if (_self_ior) {
    struct sidl_BaseInterface__object *_exception = NULL;
    static char *_kwlist[] = {
      NULL
    };
    const int _okay = PyArg_ParseTupleAndKeywords(
      _args, _kwdict, 
      "", _kwlist);
    if (_okay) {
      char* _return = NULL;
      _return = (*(_self_ior->d_epv->f_getURL))(_self_ior->d_object,          \
        &_exception);
      if (_exception) {
        struct sidl_rmi_NetworkException__object *_ex0;
        if ((_ex0 = (struct sidl_rmi_NetworkException__object *)
          sidl_PyExceptionCast(_exception, "sidl.rmi.NetworkException")))
        {
          PyObject *obj = sidl_rmi_NetworkException__wrap(_ex0);
          PyObject *_args = PyTuple_New(1);
          PyTuple_SetItem(_args, 0, obj);
          obj = PyObject_CallObject(sidl_rmi_NetworkException__type, _args);
          PyErr_SetObject(sidl_rmi_NetworkException__type, obj);
          Py_XDECREF(_args);
        }
      }
      else {
        _return_value = Py_BuildValue(
          "z",
          _return);
      }
      free((void *)_return);
    }
  }
  else {
    PyErr_SetString(PyExc_TypeError, 
      "self pointer is not a sidl.rmi.InstanceHandle");
  }
  return _return_value;
}

static int
sidl_rmi_InstanceHandle_createCast(PyObject *self, PyObject *args,            \
  PyObject *kwds) {
  struct sidl_rmi_InstanceHandle__object *optarg = NULL;
  static char *_kwlist[] = { "sobj", NULL };
  int _okay = PyArg_ParseTupleAndKeywords(args, kwds, "O&", _kwlist,          \
    (void *)sidl_rmi_InstanceHandle__convert, &optarg);
  if (_okay) {
    return sidl_Object_Init(
      (SPObject *)self,
      (struct sidl_BaseInterface__object *)optarg->d_object,
      sidl_PyStealRef);
  }
  return -1;
}

static PyMethodDef _InstanceHandleModuleMethods[] = {
  { NULL, NULL }
};

static PyMethodDef _InstanceHandleObjectMethods[] = {
  { "close", (PyCFunction)pStub_InstanceHandle_close,
  (METH_VARARGS | METH_KEYWORDS),
"\
close()\n\
RETURNS\n\
   (bool _return)\n\
RAISES\n\
    sidl.rmi.NetworkException\n\
\n\
\
closes the connection (called be destructor, if not done explicitly) \n\
returns true if successful, false otherwise (including subsequent calls)"
   },
  { "createInvocation", (PyCFunction)pStub_InstanceHandle_createInvocation,
  (METH_VARARGS | METH_KEYWORDS),
"\
createInvocation(in string methodName)\n\
RETURNS\n\
   (sidl.rmi.Invocation _return)\n\
RAISES\n\
    sidl.rmi.NetworkException\n\
\n\
\
create a handle to invoke a named method "
   },
  { "getObjectID", (PyCFunction)pStub_InstanceHandle_getObjectID,
  (METH_VARARGS | METH_KEYWORDS),
"\
getObjectID()\n\
RETURNS\n\
   (string _return)\n\
RAISES\n\
    sidl.rmi.NetworkException\n\
\n\
\
return the session ID "
   },
  { "getProtocol", (PyCFunction)pStub_InstanceHandle_getProtocol,
  (METH_VARARGS | METH_KEYWORDS),
"\
getProtocol()\n\
RETURNS\n\
   (string _return)\n\
RAISES\n\
    sidl.rmi.NetworkException\n\
\n\
\
return the name of the protocol "
   },
  { "getURL", (PyCFunction)pStub_InstanceHandle_getURL,
  (METH_VARARGS | METH_KEYWORDS),
"\
getURL()\n\
RETURNS\n\
   (string _return)\n\
RAISES\n\
    sidl.rmi.NetworkException\n\
\n\
\
return the full URL for this object, takes the form: \n\
 protocol://server:port/class/objectID"
   },
  { "initConnect", (PyCFunction)pStub_InstanceHandle_initConnect,
  (METH_VARARGS | METH_KEYWORDS),
"\
initConnect(in string url)\n\
RETURNS\n\
   (bool _return)\n\
RAISES\n\
    sidl.rmi.NetworkException\n\
\n\
\
initialize a connection (intended for use by the ProtocolFactory) "
   },
  { "initCreate", (PyCFunction)pStub_InstanceHandle_initCreate,
  (METH_VARARGS | METH_KEYWORDS),
"\
initCreate(in string url,\n\
           in string typeName)\n\
RETURNS\n\
   (bool _return)\n\
RAISES\n\
    sidl.rmi.NetworkException\n\
\n\
\
initialize a connection (intended for use by the ProtocolFactory) "
   },
  { NULL, NULL }
};

static PyTypeObject _sidl_rmi_InstanceHandleType = {
  PyObject_HEAD_INIT(NULL)
  0,      /* ob_size */
  "sidl.rmi.InstanceHandle.InstanceHandle", /* tp_name */
  0,      /* tp_basicsize */
  0,      /* tp_itemsize */
  0,      /* tp_dealloc */
  0,      /* tp_print */
  0,      /* tp_getattr */
  0,      /* tp_setattr */
  0,      /* tp_compare */
  0,      /* tp_repr */
  0,      /* tp_as_number */
  0,      /* tp_as_sequence */
  0,      /* tp_as_mapping */
  0,      /* tp_hash  */
  0,      /* tp_call */
  0,      /* tp_str */
  0,      /* tp_getattro */
  0,      /* tp_setattro */
  0,      /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT, /* tp_flags */
  "\
\
This interface holds the state information for handles to remote\n\
objects.  Client-side messaging libraries are expected to implement\n\
<code>sidl.rmi.InstanceHandle</code>, <code>sidl.rmi.Invocation</code>\n\
and <code>sidl.rmi.Response</code>.\n\
\n\
 When a connection is created between a stub and a real object:\n\
      sidl_rmi_InstanceHandle c = sidl_rmi_ProtocolFactory_createInstance( url, typeName );\n\
\n\
 When a method is invoked:\n\
      sidl_rmi_Invocation i = sidl_rmi_InstanceHandle_createInvocationHandle( methodname );\n\
      sidl_rmi_Invocation_packDouble( i, \"input_val\" , 2.0 );\n\
      sidl_rmi_Invocation_packString( i, \"input_str\", \"Hello\" );\n\
      ...\n\
      sidl_rmi_Response r = sidl_rmi_Invocation_invokeMethod( i );\n\
      sidl_rmi_Response_unpackBool( i, \"_retval\", &succeeded );\n\
      sidl_rmi_Response_unpackFloat( i, \"output_val\", &f );\n\
", /* tp_doc */
  0,      /* tp_traverse */
  0,       /* tp_clear */
  0,       /* tp_richcompare */
  0,       /* tp_weaklistoffset */
  0,       /* tp_iter */
  0,       /* tp_iternext */
  _InstanceHandleObjectMethods, /* tp_methods */
  0,       /* tp_members */
  0,       /* tp_getset */
  0,       /* tp_base */
  0,       /* tp_dict */
  0,       /* tp_descr_get */
  0,       /* tp_descr_set */
  0,       /* tp_dictoffset */
  sidl_rmi_InstanceHandle_createCast,   /* tp_init */
  0,       /* tp_alloc */
  0,       /* tp_new */
};

sidl_rmi_InstanceHandle__wrap_RETURN
sidl_rmi_InstanceHandle__wrap sidl_rmi_InstanceHandle__wrap_PROTO {
  PyObject *result;
  if (sidlobj) {
    result =                                                                  \
      _sidl_rmi_InstanceHandleType.tp_new(&_sidl_rmi_InstanceHandleType, NULL,\
      NULL);
    if (result) {
      if (sidl_Object_Init(
        (SPObject *)result,
        (struct sidl_BaseInterface__object *)(sidlobj->d_object),
        sidl_PyStealRef))
      {
        Py_DECREF(result);
        result = NULL;
      }
    }
  }
  else {
    result = Py_None;
    Py_INCREF(result);
  }
  return result;
}

sidl_rmi_InstanceHandle__weakRef_RETURN
sidl_rmi_InstanceHandle__weakRef sidl_rmi_InstanceHandle__weakRef_PROTO {
  PyObject *result;
  if (sidlobj) {
    result =                                                                  \
      _sidl_rmi_InstanceHandleType.tp_new(&_sidl_rmi_InstanceHandleType, NULL,\
      NULL);
    if (result) {
      if (sidl_Object_Init(
        (SPObject *)result,
        (struct sidl_BaseInterface__object *)(sidlobj->d_object),
        sidl_PyWeakRef))
      {
        Py_DECREF(result);
        result = NULL;
      }
    }
  }
  else {
    result = Py_None;
    Py_INCREF(result);
  }
  return result;
}

sidl_rmi_InstanceHandle_deref_RETURN
sidl_rmi_InstanceHandle_deref sidl_rmi_InstanceHandle_deref_PROTO {
  if (sidlobj) {
    (*(sidlobj->d_epv->f_deleteRef))(sidlobj->d_object);
  }
}

sidl_rmi_InstanceHandle__newRef_RETURN
sidl_rmi_InstanceHandle__newRef sidl_rmi_InstanceHandle__newRef_PROTO {
  PyObject *result;
  if (sidlobj) {
    result =                                                                  \
      _sidl_rmi_InstanceHandleType.tp_new(&_sidl_rmi_InstanceHandleType, NULL,\
      NULL);
    if (result) {
      if (sidl_Object_Init(
        (SPObject *)result,
        (struct sidl_BaseInterface__object *)(sidlobj->d_object),
        sidl_PyNewRef))
      {
        Py_DECREF(result);
        result = NULL;
      }
    }
  }
  else {
    result = Py_None;
    Py_INCREF(result);
  }
  return result;
}

sidl_rmi_InstanceHandle__addRef_RETURN
sidl_rmi_InstanceHandle__addRef sidl_rmi_InstanceHandle__addRef_PROTO {
  if (sidlobj) {
    (*(sidlobj->d_epv->f_addRef))(sidlobj->d_object);
  }
}

sidl_rmi_InstanceHandle_PyType_RETURN
sidl_rmi_InstanceHandle_PyType sidl_rmi_InstanceHandle_PyType_PROTO {
  Py_INCREF(&_sidl_rmi_InstanceHandleType);
  return &_sidl_rmi_InstanceHandleType;
}

sidl_rmi_InstanceHandle__convert_RETURN
sidl_rmi_InstanceHandle__convert sidl_rmi_InstanceHandle__convert_PROTO {
  *sidlobj = sidl_Cast(obj, "sidl.rmi.InstanceHandle");
  if (*sidlobj) {
    (*((*sidlobj)->d_epv->f_addRef))((*sidlobj)->d_object);
  }
  else if (obj != Py_None) {
    PyErr_SetString(PyExc_TypeError, 
      "argument is not a(n) sidl.rmi.InstanceHandle");
    return 0;
  }
  return 1;
}

static int
_convertPython(void *sidlarray, const int *ind, PyObject *pyobj)
{
  struct sidl_rmi_InstanceHandle__object *sidlobj;
  if (sidl_rmi_InstanceHandle__convert(pyobj, &sidlobj)) {
    sidl_interface__array_set((struct sidl_interface__array *)sidlarray,
    ind, (struct sidl_BaseInterface__object *)sidlobj);
    if (sidlobj) {
      sidl_BaseInterface_deleteRef((struct sidl_BaseInterface__object         \
        *)sidlobj);
    }
    return FALSE;
  }
  return TRUE;
}

sidl_rmi_InstanceHandle__convert_python_array_RETURN
sidl_rmi_InstanceHandle__convert_python_array                                 \
  sidl_rmi_InstanceHandle__convert_python_array_PROTO {
  int result = 0;
  *sidlarray = NULL;
  if (obj == Py_None) {
    result = TRUE;
  }
  else {
    PyObject *pya = PyArray_FromObject(obj, PyArray_OBJECT, 0, 0);
    if (pya) {
      if (PyArray_OBJECT == ((PyArrayObject *)pya)->descr->type_num) {
        int dimen, lower[SIDL_MAX_ARRAY_DIMENSION],
          upper[SIDL_MAX_ARRAY_DIMENSION],
          stride[SIDL_MAX_ARRAY_DIMENSION];
        if (sidl_array__extract_python_info
          (pya, &dimen, lower, upper, stride))
        {
            *sidlarray = (struct                                              \
              sidl_rmi_InstanceHandle__array*)sidl_interface__array_createRow
            (dimen, lower, upper);
          result = sidl_array__convert_python
            (pya, dimen, *sidlarray, _convertPython);
          if (*sidlarray && !result) {
            sidl_interface__array_deleteRef(
              (struct  sidl_interface__array *)*sidlarray);
            *sidlarray = NULL;
          }
        }
      }
      Py_DECREF(pya);
    }
  }
  return result;
}

static int
_convertSIDL(void *sidlarray, const int *ind, PyObject **dest)
{
  struct sidl_rmi_InstanceHandle__object *sidlobj = (struct                   \
    sidl_rmi_InstanceHandle__object*)
  sidl_interface__array_get((struct sidl_interface__array *)
    sidlarray, ind);
  *dest = sidl_rmi_InstanceHandle__wrap(sidlobj);
  return (*dest == NULL);
}

sidl_rmi_InstanceHandle__convert_sidl_array_RETURN
sidl_rmi_InstanceHandle__convert_sidl_array                                   \
  sidl_rmi_InstanceHandle__convert_sidl_array_PROTO {
  PyObject *pya = NULL;
  if (sidlarray) {
    const int dimen = sidlArrayDim(sidlarray);
    int i;
    int *lower = (int *)malloc(sizeof(int) * dimen);
    int *upper = (int *)malloc(sizeof(int) * dimen);
    int *numelem = (int *)malloc(sizeof(int) * dimen);
    for(i = 0; i < dimen; ++i) {
      lower[i] = sidlLower(sidlarray, i);
      upper[i] = sidlUpper(sidlarray, i);
      numelem[i] = 1 + upper[i] - lower[i];
    }
    pya = PyArray_FromDims(dimen, numelem, PyArray_OBJECT);
    if (pya) {
      if (!sidl_array__convert_sidl(pya, dimen, lower, upper,
        numelem, sidlarray, _convertSIDL))
      {
        Py_DECREF(pya);
        pya = NULL;
      }
    }
    free(numelem);
    free(upper);
    free(lower);
  }
  else {
    Py_INCREF(Py_None);
    pya = Py_None;
  }
  return pya;
}

void
initInstanceHandle(void) {
  PyObject *module, *dict, *c_api;
  static void *ExternalAPI[sidl_rmi_InstanceHandle__API_NUM];
  module = Py_InitModule3("InstanceHandle", _InstanceHandleModuleMethods, "\
\
This interface holds the state information for handles to remote\n\
objects.  Client-side messaging libraries are expected to implement\n\
<code>sidl.rmi.InstanceHandle</code>, <code>sidl.rmi.Invocation</code>\n\
and <code>sidl.rmi.Response</code>.\n\
\n\
 When a connection is created between a stub and a real object:\n\
      sidl_rmi_InstanceHandle c = sidl_rmi_ProtocolFactory_createInstance( url, typeName );\n\
\n\
 When a method is invoked:\n\
      sidl_rmi_Invocation i = sidl_rmi_InstanceHandle_createInvocationHandle( methodname );\n\
      sidl_rmi_Invocation_packDouble( i, \"input_val\" , 2.0 );\n\
      sidl_rmi_Invocation_packString( i, \"input_str\", \"Hello\" );\n\
      ...\n\
      sidl_rmi_Response r = sidl_rmi_Invocation_invokeMethod( i );\n\
      sidl_rmi_Response_unpackBool( i, \"_retval\", &succeeded );\n\
      sidl_rmi_Response_unpackFloat( i, \"output_val\", &f );\n\
"
  );
  dict = PyModule_GetDict(module);
  ExternalAPI[sidl_rmi_InstanceHandle__wrap_NUM] =                            \
    (void*)sidl_rmi_InstanceHandle__wrap;
  ExternalAPI[sidl_rmi_InstanceHandle__convert_NUM] =                         \
    (void*)sidl_rmi_InstanceHandle__convert;
  ExternalAPI[sidl_rmi_InstanceHandle__convert_python_array_NUM] =            \
    (void*)sidl_rmi_InstanceHandle__convert_python_array;
  ExternalAPI[sidl_rmi_InstanceHandle__convert_sidl_array_NUM] =              \
    (void*)sidl_rmi_InstanceHandle__convert_sidl_array;
  ExternalAPI[sidl_rmi_InstanceHandle__weakRef_NUM] =                         \
    (void*)sidl_rmi_InstanceHandle__weakRef;
  ExternalAPI[sidl_rmi_InstanceHandle_deref_NUM] =                            \
    (void*)sidl_rmi_InstanceHandle_deref;
  ExternalAPI[sidl_rmi_InstanceHandle__newRef_NUM] =                          \
    (void*)sidl_rmi_InstanceHandle__newRef;
  ExternalAPI[sidl_rmi_InstanceHandle__addRef_NUM] =                          \
    (void*)sidl_rmi_InstanceHandle__addRef;
  ExternalAPI[sidl_rmi_InstanceHandle_PyType_NUM] =                           \
    (void*)sidl_rmi_InstanceHandle_PyType;
  import_SIDLObjA();
  if (PyErr_Occurred()) {
    Py_FatalError("Error importing sidlObjA module.");
  }
  c_api = PyCObject_FromVoidPtr((void *)ExternalAPI, NULL);
  PyDict_SetItemString(dict, "_C_API", c_api);
  Py_XDECREF(c_api);
  import_SIDLPyArrays();
  if (PyErr_Occurred()) {
    Py_FatalError("Error importing sidlPyArrays module.");
  }
  import_array();
  if (PyErr_Occurred()) {
    Py_FatalError("Error importing Numeric Python module.");
  }
  sidl_BaseInterface__import();
  _sidl_rmi_InstanceHandleType.tp_base = sidl_BaseInterface_PyType();
  _sidl_rmi_InstanceHandleType.tp_bases = PyTuple_New(1);
  PyTuple_SetItem(_sidl_rmi_InstanceHandleType.tp_bases,0,                    \
    (PyObject *)sidl_BaseInterface_PyType());
  if (PyType_Ready(&_sidl_rmi_InstanceHandleType) < 0) {
    PyErr_Print();
    fprintf(stderr, "PyType_Ready on sidl.rmi.InstanceHandle failed.\n");
    return;
  }
  Py_INCREF(&_sidl_rmi_InstanceHandleType);
  PyDict_SetItemString(dict, "InstanceHandle",                                \
    (PyObject *)&_sidl_rmi_InstanceHandleType);
  sidl_ClassInfo__import();
  sidl_rmi_NetworkException__import();
  sidl_rmi_Invocation__import();
}
