/*
 * File:          sidl_Resolve_IOR.h
 * Symbol:        sidl.Resolve-v0.9.3
 * Symbol Type:   enumeration
 * Babel Version: 0.10.12
 * Release:       $Name: V1-10-0b $
 * Revision:      @(#) $Id: sidl_Resolve_IOR.h,v 1.5 2005/11/14 21:20:25 painter Exp $
 * Description:   Intermediate Object Representation for sidl.Resolve
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

#ifndef included_sidl_Resolve_IOR_h
#define included_sidl_Resolve_IOR_h

#ifndef included_sidlType_h
#include "sidlType.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidl.Resolve" (version 0.9.3)
 * 
 * When loading a dynmaically linked library, there are three
 * settings: LAZY, NOW, SCLRESOLVE
 */


/* Opaque forward declaration of array struct */
struct sidl_Resolve__array;

enum sidl_Resolve__enum {
  /**
   * Resolve symbols on an as needed basis. 
   */
  sidl_Resolve_LAZY       = 0,

  /**
   * Resolve all symbols at load time. 
   */
  sidl_Resolve_NOW        = 1,

  /**
   * Use the resolve setting from the SCL file. 
   */
  sidl_Resolve_SCLRESOLVE = 2

};

#ifdef __cplusplus
}
#endif
#endif
