/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.0 $
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 *
 *****************************************************************************/

int hypre_SStructKrylovCopyVector( void *x, void *y );

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovIdentitySetup
 *--------------------------------------------------------------------------*/

int
hypre_SStructKrylovIdentitySetup( void *vdata,
                           void *A,
                           void *b,
                           void *x )

{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovIdentity
 *--------------------------------------------------------------------------*/

int
hypre_SStructKrylovIdentity( void *vdata,
                      void *A,
                      void *b,
                      void *x )

{
   return( hypre_SStructKrylovCopyVector(b, x) );
}

