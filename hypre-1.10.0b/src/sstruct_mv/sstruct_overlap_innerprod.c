/******************************************************************************
 *
 * SStruct inner product routine for overlapped grids.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SStructPOverlapInnerProd
 *--------------------------------------------------------------------------*/

int
hypre_SStructPOverlapInnerProd( hypre_SStructPVector *px,
                                hypre_SStructPVector *py,
                                double               *presult_ptr )
{
   int    ierr = 0;
   int    nvars = hypre_SStructPVectorNVars(px);
   double presult;
   double sresult;
   int    var;

   presult = 0.0;
   for (var = 0; var < nvars; var++)
   {
      sresult = hypre_StructOverlapInnerProd(hypre_SStructPVectorSVector(px, var),
                                             hypre_SStructPVectorSVector(py, var));
      presult += sresult;
   }

   *presult_ptr = presult;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructOverlapInnerProd
 *--------------------------------------------------------------------------*/

int
hypre_SStructOverlapInnerProd( hypre_SStructVector *x,
                               hypre_SStructVector *y,
                               double              *result_ptr )
{
   int    ierr = 0;
   int    nparts = hypre_SStructVectorNParts(x);
   double result;
   double presult;
   int    part;

   result = 0.0;
   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPOverlapInnerProd(hypre_SStructVectorPVector(x, part),
                                     hypre_SStructVectorPVector(y, part), &presult);
      result += presult;
   }

   *result_ptr = result;

   return ierr;
}
