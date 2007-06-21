/*
 * File:          sidl_BaseClass_Module.c
 * Symbol:        sidl.BaseClass-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Release:       $Name: V1-10-0b $
 * Revision:      @(#) $Id: sidl_BaseClass_Module.c,v 1.4 2005/11/14 21:20:15 painter Exp $
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
 * sidl type sidl.BaseClass.
 */


/**
 * Symbol "sidl.BaseClass" (version 0.9.3)
 * 
 * Every class implicitly inherits from <code>BaseClass</code>.  This
 * class implements the methods in <code>BaseInterface</code>.
 */
#define sidl_BaseClass_INTERNAL 1
#include "sidl_BaseClass_Module.h"
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
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
#include <stdlib.h>
#include <string.h>

staticforward PyTypeObject _sidl_BaseClassType;

static const struct sidl_BaseClass__external *_implEPV = NULL;

static int
sidl_BaseClass_createCast(PyObject *self, PyObject *args, PyObject *kwds) {
  struct sidl_BaseClass__object *optarg = NULL;
  static char *_kwlist[] = { "sobj", NULL };
  int _okay = PyArg_ParseTupleAndKeywords(args, kwds, "|O&", _kwlist,         \
    (void *)sidl_BaseClass__convert, &optarg);
  if (_okay) {
    if (!optarg) {
      optarg = (*(_implEPV->createObject))();
    }
    return sidl_Object_Init(
      (SPObject *)self,
      (struct sidl_BaseInterface__object *)optarg,
      sidl_PyStealRef);
  }
  return -1;
}

static PyMethodDef _BaseClassModuleMethods[] = {
  { NULL, NULL }
};

static PyMethodDef _BaseClassObjectMethods[] = {
  { NULL, NULL }
};

static PyTypeObject _sidl_BaseClassType = {
  PyObject_HEAD_INIT(NULL)
  0,      /* ob_size */
  "sidl.BaseClass.BaseClass", /* tp_name */
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
Every class implicitly inherits from <code>BaseClass</code>.  This\n\
class implements the methods in <code>BaseInterface</code>.", /* tp_doc */
  0,      /* tp_traverse */
  0,       /* tp_clear */
  0,       /* tp_richcompare */
  0,       /* tp_weaklistoffset */
  0,       /* tp_iter */
  0,       /* tp_iternext */
  _BaseClassObjectMethods, /* tp_methods */
  0,       /* tp_members */
  0,       /* tp_getset */
  0,       /* tp_base */
  0,       /* tp_dict */
  0,       /* tp_descr_get */
  0,       /* tp_descr_set */
  0,       /* tp_dictoffset */
  sidl_BaseClass_createCast,   /* tp_init */
  0,       /* tp_alloc */
  0,       /* tp_new */
};

sidl_BaseClass__wrap_RETURN
sidl_BaseClass__wrap sidl_BaseClass__wrap_PROTO {
  PyObject *result;
  if (sidlobj) {
    result = _sidl_BaseClassType.tp_new(&_sidl_BaseClassType, NULL, NULL);
    if (result) {
      if (sidl_Object_Init(
        (SPObject *)result,
        (struct sidl_BaseInterface__object *)(sidlobj),
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

sidl_BaseClass__weakRef_RETURN
sidl_BaseClass__weakRef sidl_BaseClass__weakRef_PROTO {
  PyObject *result;
  if (sidlobj) {
    result = _sidl_BaseClassType.tp_new(&_sidl_BaseClassType, NULL, NULL);
    if (result) {
      if (sidl_Object_Init(
        (SPObject *)result,
        (struct sidl_BaseInterface__object *)(sidlobj),
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

sidl_BaseClass_deref_RETURN
sidl_BaseClass_deref sidl_BaseClass_deref_PROTO {
  if (sidlobj) {
    (*(sidlobj->d_epv->f_deleteRef))(sidlobj);
  }
}

sidl_BaseClass__newRef_RETURN
sidl_BaseClass__newRef sidl_BaseClass__newRef_PROTO {
  PyObject *result;
  if (sidlobj) {
    result = _sidl_BaseClassType.tp_new(&_sidl_BaseClassType, NULL, NULL);
    if (result) {
      if (sidl_Object_Init(
        (SPObject *)result,
        (struct sidl_BaseInterface__object *)(sidlobj),
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

sidl_BaseClass__addRef_RETURN
sidl_BaseClass__addRef sidl_BaseClass__addRef_PROTO {
  if (sidlobj) {
    (*(sidlobj->d_epv->f_addRef))(sidlobj);
  }
}

sidl_BaseClass_PyType_RETURN
sidl_BaseClass_PyType sidl_BaseClass_PyType_PROTO {
  Py_INCREF(&_sidl_BaseClassType);
  return &_sidl_BaseClassType;
}

sidl_BaseClass__convert_RETURN
sidl_BaseClass__convert sidl_BaseClass__convert_PROTO {
  *sidlobj = sidl_Cast(obj, "sidl.BaseClass");
  if (*sidlobj) {
    (*((*sidlobj)->d_epv->f_addRef))(*sidlobj);
  }
  else if (obj != Py_None) {
    PyErr_SetString(PyExc_TypeError, 
      "argument is not a(n) sidl.BaseClass");
    return 0;
  }
  return 1;
}

static int
_convertPython(void *sidlarray, const int *ind, PyObject *pyobj)
{
  struct sidl_BaseClass__object *sidlobj;
  if (sidl_BaseClass__convert(pyobj, &sidlobj)) {
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

sidl_BaseClass__convert_python_array_RETURN
sidl_BaseClass__convert_python_array                                          \
  sidl_BaseClass__convert_python_array_PROTO {
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
              sidl_BaseClass__array*)sidl_interface__array_createRow
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
  struct sidl_BaseClass__object *sidlobj = (struct sidl_BaseClass__object*)
  sidl_interface__array_get((struct sidl_interface__array *)
    sidlarray, ind);
  *dest = sidl_BaseClass__wrap(sidlobj);
  return (*dest == NULL);
}

sidl_BaseClass__convert_sidl_array_RETURN
sidl_BaseClass__convert_sidl_array sidl_BaseClass__convert_sidl_array_PROTO {
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
initBaseClass(void) {
  PyObject *module, *dict, *c_api;
  static void *ExternalAPI[sidl_BaseClass__API_NUM];
  module = Py_InitModule3("BaseClass", _BaseClassModuleMethods, "\
\
Every class implicitly inherits from <code>BaseClass</code>.  This\n\
class implements the methods in <code>BaseInterface</code>."
  );
  dict = PyModule_GetDict(module);
  ExternalAPI[sidl_BaseClass__wrap_NUM] = (void*)sidl_BaseClass__wrap;
  ExternalAPI[sidl_BaseClass__convert_NUM] = (void*)sidl_BaseClass__convert;
  ExternalAPI[sidl_BaseClass__convert_python_array_NUM] =                     \
    (void*)sidl_BaseClass__convert_python_array;
  ExternalAPI[sidl_BaseClass__convert_sidl_array_NUM] =                       \
    (void*)sidl_BaseClass__convert_sidl_array;
  ExternalAPI[sidl_BaseClass__weakRef_NUM] = (void*)sidl_BaseClass__weakRef;
  ExternalAPI[sidl_BaseClass_deref_NUM] = (void*)sidl_BaseClass_deref;
  ExternalAPI[sidl_BaseClass__newRef_NUM] = (void*)sidl_BaseClass__newRef;
  ExternalAPI[sidl_BaseClass__addRef_NUM] = (void*)sidl_BaseClass__addRef;
  ExternalAPI[sidl_BaseClass_PyType_NUM] = (void*)sidl_BaseClass_PyType;
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
  _sidl_BaseClassType.tp_base = sidl_BaseInterface_PyType();
  _sidl_BaseClassType.tp_bases = PyTuple_New(1);
  PyTuple_SetItem(_sidl_BaseClassType.tp_bases,0,                             \
    (PyObject *)sidl_BaseInterface_PyType());
  if (PyType_Ready(&_sidl_BaseClassType) < 0) {
    PyErr_Print();
    fprintf(stderr, "PyType_Ready on sidl.BaseClass failed.\n");
    return;
  }
  Py_INCREF(&_sidl_BaseClassType);
  PyDict_SetItemString(dict, "BaseClass", (PyObject *)&_sidl_BaseClassType);
  sidl_ClassInfo__import();
  _implEPV = sidl_BaseClass__externals();
  if (!_implEPV) {
    Py_FatalError("Cannot load implementation for sidl class sidl.BaseClass");
  }
}
