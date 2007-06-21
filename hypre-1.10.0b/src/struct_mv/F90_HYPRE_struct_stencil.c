/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.0 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_StructStencil interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructStencilCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structstencilcreate, HYPRE_STRUCTSTENCILCREATE)( int      *dim,
                                            int      *size,
                                            long int *stencil,
                                            int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructStencilCreate( (int)                   *dim,
                                   (int)                   *size,
                                   (HYPRE_StructStencil *)  stencil ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructStencilSetElement
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structstencilsetelement, HYPRE_STRUCTSTENCILSETELEMENT)( long int *stencil,
                                                int      *element_index,
                                                int      *offset,
                                                int      *ierr          )
{
   *ierr = (int)
      ( HYPRE_StructStencilSetElement( (HYPRE_StructStencil) *stencil,
                                       (int)                 *element_index,
                                       (int *)                offset       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructStencilDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structstencildestroy, HYPRE_STRUCTSTENCILDESTROY)( long int *stencil,
                                             int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructStencilDestroy( (HYPRE_StructStencil) *stencil ) );
}
