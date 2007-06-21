/*
 * File:        sidlOps.h
 * Copyright:   (c) 2005 The Regents of the University of California
 * Revision:    @(#) $Revision: 1.5 $
 * Date:        $Date: 2005/11/14 21:20:23 $
 * Description: Special options that are common through out babel.
 *
 */

#ifndef included_sidlOps_h
#define included_sidlOps_h
#include "babel_config.h"
#ifdef __cplusplus
extern "C" {
#endif

#ifdef SIDL_DYNAMIC_LIBRARY
  void* sidl_dynamicLoadIOR(char* objName, char* extName);
#endif

#ifdef __cplusplus
}
#endif
#endif /*  included_sidlOps_h */
