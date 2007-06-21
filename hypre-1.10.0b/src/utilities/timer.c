/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.7 $
 *********************************************************************EHEADER*/

/*
 * File:	timer.c
 * Copyright:	(c) 1997 The Regents of the University of California
 * Author:	Scott Kohn (skohn@llnl.gov)
 * Description:	somewhat portable timing routines for C++, C, and Fortran
 *
 * If TIMER_USE_MPI is defined, then the MPI timers are used to get
 * wallclock seconds, since we assume that the MPI timers have better
 * resolution than the system timers.
 */

#include <time.h>
#include <unistd.h>
#ifndef WIN32
#include <sys/times.h>
#endif
#ifdef TIMER_USE_MPI
#include "mpi.h"
#endif

double time_getWallclockSeconds(void)
{
#ifdef TIMER_USE_MPI
   return(MPI_Wtime());
#else
#ifdef WIN32
   clock_t cl=clock();
   return(((double) cl)/((double) CLOCKS_PER_SEC));
#else
   struct tms usage;
   long wallclock = times(&usage);
   return(((double) wallclock)/((double) sysconf(_SC_CLK_TCK)));
#endif
#endif
}

double time_getCPUSeconds(void)
{
#ifndef TIMER_NO_SYS
   clock_t cpuclock = clock();
   return(((double) (cpuclock))/((double) CLOCKS_PER_SEC));
#else
   return(0.0);
#endif
}

double time_get_wallclock_seconds_(void)
{
   return(time_getWallclockSeconds());
}

double time_get_cpu_seconds_(void)
{
   return(time_getCPUSeconds());
}
