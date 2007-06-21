/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.3 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_SStructMatrix interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixcreate, HYPRE_SSTRUCTMATRIXCREATE)
                                                        (int        *comm,
                                                         long int   *graph,
                                                         long int   *matrix_ptr,
                                                         int        *ierr)
{
   *ierr = (int) (HYPRE_SStructMatrixCreate( (MPI_Comm)              *comm,
                                             (HYPRE_SStructGraph)    *graph,
                                             (HYPRE_SStructMatrix *)  matrix_ptr ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixdestroy, HYPRE_SSTRUCTMATRIXDESTROY)
                                                        (long int   *matrix,
                                                         int        *ierr)
{
   *ierr = (int) (HYPRE_SStructMatrixDestroy( (HYPRE_SStructMatrix) *matrix ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixinitialize, HYPRE_SSTRUCTMATRIXINITIALIZE)
                                                        (long int   *matrix,
                                                         int        *ierr)
{
   *ierr = (int) (HYPRE_SStructMatrixInitialize( (HYPRE_SStructMatrix) *matrix ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetvalues, HYPRE_SSTRUCTMATRIXSETVALUES)
                                                        (long int   *matrix,
                                                         int        *part,
                                                         int        *index,
                                                         int        *var,
                                                         int        *nentries,
                                                         int        *entries,
                                                         double     *values,
                                                         int        *ierr)
{
   *ierr = (int) (HYPRE_SStructMatrixSetValues( (HYPRE_SStructMatrix) *matrix,
                                                (int)                 *part,
                                                (int *)                index,
                                                (int)                 *var,
                                                (int)                 *nentries,
                                                (int *)                entries,
                                                (double *)             values ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixSetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetboxvalues, HYPRE_SSTRUCTMATRIXSETBOXVALUES)
                                                        (long int   *matrix,
                                                         int        *part,
                                                         int        *ilower,
                                                         int        *iupper,
                                                         int        *var,
                                                         int        *nentries,
                                                         int        *entries,
                                                         double     *values,
                                                         int        *ierr)
{
   *ierr = (int) (HYPRE_SStructMatrixSetBoxValues( (HYPRE_SStructMatrix)  *matrix,
                                                   (int)                  *part,
                                                   (int *)                 ilower,
                                                   (int *)                 iupper,
                                                   (int)                  *var,
                                                   (int)                  *nentries,
                                                   (int *)                 entries,
                                                   (double *)              values));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixGetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixgetvalues, HYPRE_SSTRUCTMATRIXGETVALUES)
                                                        (long int   *matrix,
                                                         int        *part,
                                                         int        *index,
                                                         int        *var,
                                                         int        *nentries,
                                                         int        *entries,
                                                         double     *values,
                                                         int        *ierr)
{
   *ierr = (int) (HYPRE_SStructMatrixGetValues( (HYPRE_SStructMatrix) *matrix,
                                                (int)                 *part,
                                                (int *)                index,
                                                (int)                 *var,
                                                (int)                 *nentries,
                                                (int *)                entries,
                                                (double *)             values) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixGetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixgetboxvalues, HYPRE_SSTRUCTMATRIXGETBOXVALUES)
                                                        (long int   *matrix,
                                                         int        *part,
                                                         int        *ilower,
                                                         int        *iupper,
                                                         int        *var,
                                                         int        *nentries,
                                                         int        *entries,
                                                         double     *values,
                                                         int        *ierr)
{
   *ierr = (int) (HYPRE_SStructMatrixGetBoxValues( (HYPRE_SStructMatrix)  *matrix,
                                                   (int)                  *part,
                                                   (int *)                 ilower,
                                                   (int *)                 iupper,
                                                   (int)                  *var,
                                                   (int)                  *nentries,
                                                   (int *)                 entries,
                                                   (double *)              values));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixaddtovalues, HYPRE_SSTRUCTMATRIXADDTOVALUES)
                                                        (long int   *matrix,
                                                         int        *part,
                                                         int        *index,
                                                         int        *var,
                                                         int        *nentries,
                                                         int        *entries,
                                                         double     *values,
                                                         int        *ierr)
{
   *ierr = (int) (HYPRE_SStructMatrixAddToValues( (HYPRE_SStructMatrix) *matrix,
                                                  (int)                 *part,
                                                  (int *)                index,
                                                  (int)                 *var,
                                                  (int)                 *nentries,
                                                  (int *)                entries,
                                                  (double *)             values) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixAddToBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixaddtoboxvalue, HYPRE_SSTRUCTMATRIXADDTOBOXVALUE)
                                                        (long int   *matrix,
                                                         int        *part,
                                                         int        *ilower,
                                                         int        *iupper,
                                                         int        *var,
                                                         int        *nentries,
                                                         int        *entries,
                                                         double     *values,
                                                         int        *ierr)
{
   *ierr = (int) (HYPRE_SStructMatrixAddToBoxValues( (HYPRE_SStructMatrix)  *matrix,
                                                     (int)                  *part,
                                                     (int *)                 ilower,
                                                     (int *)                 iupper,
                                                     (int)                  *var,
                                                     (int)                  *nentries,
                                                     (int *)                 entries,
                                                     (double *)              values ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixassemble, HYPRE_SSTRUCTMATRIXASSEMBLE)
                                                        (long int   *matrix,
                                                         int        *ierr)
{
   *ierr = (int) (HYPRE_SStructMatrixAssemble( (HYPRE_SStructMatrix) *matrix ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixSetSymmetric
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetsymmetric, HYPRE_SSTRUCTMATRIXSETSYMMETRIC)
                                                        (long int   *matrix,
                                                         int        *part,
                                                         int        *var,
                                                         int        *to_var,
                                                         int        *symmetric,
                                                         int        *ierr)
{
   *ierr = (int) (HYPRE_SStructMatrixSetSymmetric( (HYPRE_SStructMatrix) *matrix,
                                                   (int)                 *part,
                                                   (int)                 *var,
                                                   (int)                 *to_var,
                                                   (int)                 *symmetric ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixSetNSSymmetric
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetnssymmetr, HYPRE_SSTRUCTMATRIXSETNSSYMMETR)
                                                        (long int   *matrix,
                                                         int        *symmetric,
                                                         int        *ierr)
{
   *ierr = (int) (HYPRE_SStructMatrixSetNSSymmetric( (HYPRE_SStructMatrix) *matrix,
                                                     (int)                 *symmetric ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixSetObjectType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetobjecttyp, HYPRE_SSTRUCTMATRIXSETOBJECTTYP)
                                                        (long int   *matrix,
                                                         int        *type,
                                                         int        *ierr)
{
   *ierr = (int) (HYPRE_SStructMatrixSetObjectType( (HYPRE_SStructMatrix) *matrix,
                                                    (int)                 *type ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixGetObject
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixgetobject, HYPRE_SSTRUCTMATRIXGETOBJECT)
                                                        (long int   *matrix,
                                                         long int   *object,
                                                         int        *ierr)
{
   *ierr = (int) (HYPRE_SStructMatrixGetObject( (HYPRE_SStructMatrix) *matrix,
                                                (void *)              *object )) ;
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixprint, HYPRE_SSTRUCTMATRIXPRINT)
                                                        (const char *filename,
                                                         long int   *matrix,
                                                         int        *all,
                                                         int        *ierr)
{
   *ierr = (int) (HYPRE_SStructMatrixPrint( (const char *)           filename,
                                            (HYPRE_SStructMatrix)   *matrix,
                                            (int)                   *all ) );
}
