/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.0 $
 *********************************************************************EHEADER*/

#include "headers.h"

#if 0

/*--------------------------------------------------------------------------
 * hypre_SparseMSGFilterSetup
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGFilterSetup( hypre_StructMatrix *A,
                            int                *num_grids,
                            int                 lx,
                            int                 ly,
                            int                 lz,
                            int                 jump,
                            hypre_StructVector *visitx,
                            hypre_StructVector *visity,
                            hypre_StructVector *visitz    )
{
   int                   ierr = 0;

   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;
                        
   hypre_Box             *A_dbox;
   hypre_Box             *v_dbox;
                        
   int                    Ai;
   int                    vi;
                        
   double                *Ap;
   double                *vxp;
   double                *vyp;
   double                *vzp;
   double                 lambdax;
   double                 lambday;
   double                 lambdaz;
   double                 lambda_max;
                        
   hypre_StructStencil   *stencil;
   hypre_Index           *stencil_shape;
   int                    stencil_size;
                        
   int                    Astenc;
                        
   hypre_Index            loop_size;
   hypre_Index            cindex;
   hypre_IndexRef         start;
   hypre_Index            startv;
   hypre_Index            stride;
   hypre_Index            stridev;
                        
   int                    i, si, dir, k, l;
   int                    loopi, loopj, loopk;

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);

   /*-----------------------------------------------------
    * Compute encoding digit and strides
    *-----------------------------------------------------*/

   hypre_SetIndex(stride, 1, 1, 1);

   l = lx + ly + lz;
   if ((l >= 1) && (l <= jump))
   {
      k = 1 >> l;
      hypre_SetIndex(stridev, (1 >> lx), (1 >> ly), (1 >> lz));
   }
   else
   {
      k = 1;
      hypre_SetIndex(stridev, 1, 1, 1);

      hypre_StructVectorSetConstantValues(visitx, 0.0);
      hypre_StructVectorSetConstantValues(visity, 0.0);
      hypre_StructVectorSetConstantValues(visitz, 0.0);
   }

   /*-----------------------------------------------------
    * Compute visit vectors
    *-----------------------------------------------------*/

   hypre_SetIndex(cindex, 0, 0, 0);

   compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   hypre_ForBoxI(i, compute_boxes)
      {
         compute_box = hypre_BoxArrayBox(compute_boxes, i);

         A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
         v_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(visitx), i);

         vxp = hypre_StructVectorBoxData(visitx, i);
         vyp = hypre_StructVectorBoxData(visity, i);
         vzp = hypre_StructVectorBoxData(visitz, i);

         start = hypre_BoxIMin(compute_box);
         hypre_StructMapCoarseToFine(start, cindex, stridev, startv);
         hypre_BoxGetSize(compute_box, loop_size);

         hypre_BoxLoop2Begin(loop_size,
                             A_dbox, start,  stride,  Ai,
                             v_dbox, startv, stridev, vi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,vi,lambdax,lambday,lambdaz,si,Ap,Astenc,lambda_max,dir
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop2For(loopi, loopj, loopk, Ai, vi)
            {
               lambdax = 0.0;
               lambday = 0.0;
               lambdaz = 0.0;

               for (si = 0; si < stencil_size; si++)
               {
                  Ap = hypre_StructMatrixBoxData(A, i, si);

                  /* compute lambdax */
                  Astenc = hypre_IndexD(stencil_shape[si], 0);
                  if (Astenc == 0)
                  {
                     lambdax += Ap[Ai];
                  }
                  else
                  {
                     lambdax -= Ap[Ai];
                  }

                  /* compute lambday */
                  Astenc = hypre_IndexD(stencil_shape[si], 1);
                  if (Astenc == 0)
                  {
                     lambday += Ap[Ai];
                  }
                  else
                  {
                     lambday -= Ap[Ai];
                  }

                  /* compute lambdaz */
                  Astenc = hypre_IndexD(stencil_shape[si], 2);
                  if (Astenc == 0)
                  {
                     lambdaz += Ap[Ai];
                  }
                  else
                  {
                     lambdaz -= Ap[Ai];
                  }
               }

               lambdax *= lambdax;
               lambday *= lambday;
               lambdaz *= lambdaz;

               lambda_max = 0;
               dir = -1;
               if ((lx < num_grids[0] - 1) && (lambdax > lambda_max))
               {
                  lambda_max = lambdax;
                  dir = 0;
               }
               if ((ly < num_grids[1] - 1) && (lambday > lambda_max))
               {
                  lambda_max = lambday;
                  dir = 1;
               }
               if ((lz < num_grids[2] - 1) && (lambdaz > lambda_max))
               {
                  lambda_max = lambdaz;
                  dir = 2;
               }

               if (dir == 0)
               {
                  vxp[vi] = (double) ( ((int) vxp[vi]) | k );
               }
               else if (dir == 1)
               {
                  vyp[vi] = (double) ( ((int) vyp[vi]) | k );
               }
               else if (dir == 2)
               {
                  vzp[vi] = (double) ( ((int) vzp[vi]) | k );
               }
            }
         hypre_BoxLoop2End(Ai, vi);
      }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGFilter
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGFilter( hypre_StructVector *visit,
                       hypre_StructVector *e,
                       int                 lx,
                       int                 ly,
                       int                 lz,
                       int                 jump  )
{
   int                   ierr = 0;

   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;
                        
   hypre_Box             *e_dbox;
   hypre_Box             *v_dbox;
                        
   int                    ei;
   int                    vi;
                        
   double                *ep;
   double                *vp;
                        
   hypre_Index            loop_size;
   hypre_Index            cindex;
   hypre_IndexRef         start;
   hypre_Index            startv;
   hypre_Index            stride;
   hypre_Index            stridev;
                        
   int                    i, k, l;
   int                    loopi, loopj, loopk;

   /*-----------------------------------------------------
    * Compute encoding digit and strides
    *-----------------------------------------------------*/

   hypre_SetIndex(stride, 1, 1, 1);

   l = lx + ly + lz;
   if ((l >= 1) && (l <= jump))
   {
      k = 1 >> l;
      hypre_SetIndex(stridev, (1 >> lx), (1 >> ly), (1 >> lz));
   }
   else
   {
      k = 1;
      hypre_SetIndex(stridev, 1, 1, 1);
   }

   /*-----------------------------------------------------
    * Filter interpolated error
    *-----------------------------------------------------*/

   hypre_SetIndex(cindex, 0, 0, 0);

   compute_boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(e));
   hypre_ForBoxI(i, compute_boxes)
      {
         compute_box = hypre_BoxArrayBox(compute_boxes, i);

         e_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(e), i);
         v_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(visit), i);

         ep = hypre_StructVectorBoxData(e, i);
         vp = hypre_StructVectorBoxData(visit, i);

         start = hypre_BoxIMin(compute_box);
         hypre_StructMapCoarseToFine(start, cindex, stridev, startv);
         hypre_BoxGetSize(compute_box, loop_size);

         hypre_BoxLoop2Begin(loop_size,
                             e_dbox, start,  stride,  ei,
                             v_dbox, startv, stridev, vi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,ei,vi
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop2For(loopi, loopj, loopk, ei, vi)
            {
               if ( !(((int) vp[vi]) & k) )
               {
                  ep[ei] = 0.0;
               }
            }
         hypre_BoxLoop2End(ei, vi);
      }

   return ierr;
}

#else

/*--------------------------------------------------------------------------
 * hypre_SparseMSGFilterSetup
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGFilterSetup( hypre_StructMatrix *A,
                            int                *num_grids,
                            int                 lx,
                            int                 ly,
                            int                 lz,
                            int                 jump,
                            hypre_StructVector *visitx,
                            hypre_StructVector *visity,
                            hypre_StructVector *visitz    )
{
   int                   ierr = 0;

   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;
                        
   hypre_Box             *A_dbox;
   hypre_Box             *v_dbox;
                        
   int                    Ai;
   int                    vi;
                        
   double                *Ap;
   double                *vxp;
   double                *vyp;
   double                *vzp;
   double                 lambdax;
   double                 lambday;
   double                 lambdaz;
                        
   hypre_StructStencil   *stencil;
   hypre_Index           *stencil_shape;
   int                    stencil_size;
                        
   int                    Astenc;
                        
   hypre_Index            loop_size;
   hypre_Index            cindex;
   hypre_IndexRef         start;
   hypre_Index            startv;
   hypre_Index            stride;
   hypre_Index            stridev;
                        
   int                    i, si;
   int                    loopi, loopj, loopk;

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);

   /*-----------------------------------------------------
    * Compute encoding digit and strides
    *-----------------------------------------------------*/

   hypre_SetIndex(stride, 1, 1, 1);
   hypre_SetIndex(stridev, 1, 1, 1);

   /*-----------------------------------------------------
    * Compute visit vectors
    *-----------------------------------------------------*/

   hypre_SetIndex(cindex, 0, 0, 0);

   compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   hypre_ForBoxI(i, compute_boxes)
      {
         compute_box = hypre_BoxArrayBox(compute_boxes, i);

         A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
         v_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(visitx), i);

         vxp = hypre_StructVectorBoxData(visitx, i);
         vyp = hypre_StructVectorBoxData(visity, i);
         vzp = hypre_StructVectorBoxData(visitz, i);

         start = hypre_BoxIMin(compute_box);
         hypre_StructMapCoarseToFine(start, cindex, stridev, startv);
         hypre_BoxGetSize(compute_box, loop_size);

         hypre_BoxLoop2Begin(loop_size,
                             A_dbox, start,  stride,  Ai,
                             v_dbox, startv, stridev, vi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,vi,lambdax,lambday,lambdaz,si,Ap,Astenc
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop2For(loopi, loopj, loopk, Ai, vi)
            {
               lambdax = 0.0;
               lambday = 0.0;
               lambdaz = 0.0;

               for (si = 0; si < stencil_size; si++)
               {
                  Ap = hypre_StructMatrixBoxData(A, i, si);

                  /* compute lambdax */
                  Astenc = hypre_IndexD(stencil_shape[si], 0);
                  if (Astenc == 0)
                  {
                     lambdax += Ap[Ai];
                  }
                  else
                  {
                     lambdax -= Ap[Ai];
                  }

                  /* compute lambday */
                  Astenc = hypre_IndexD(stencil_shape[si], 1);
                  if (Astenc == 0)
                  {
                     lambday += Ap[Ai];
                  }
                  else
                  {
                     lambday -= Ap[Ai];
                  }

                  /* compute lambdaz */
                  Astenc = hypre_IndexD(stencil_shape[si], 2);
                  if (Astenc == 0)
                  {
                     lambdaz += Ap[Ai];
                  }
                  else
                  {
                     lambdaz -= Ap[Ai];
                  }
               }

               lambdax *= lambdax;
               lambday *= lambday;
               lambdaz *= lambdaz;

               vxp[vi] = lambdax / (lambdax + lambday + lambdaz);
               vyp[vi] = lambday / (lambdax + lambday + lambdaz);
               vzp[vi] = lambdaz / (lambdax + lambday + lambdaz);
            }
         hypre_BoxLoop2End(Ai, vi);
      }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGFilter
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGFilter( hypre_StructVector *visit,
                       hypre_StructVector *e,
                       int                 lx,
                       int                 ly,
                       int                 lz,
                       int                 jump  )
{
   int                   ierr = 0;

   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;
                        
   hypre_Box             *e_dbox;
   hypre_Box             *v_dbox;
                        
   int                    ei;
   int                    vi;
                        
   double                *ep;
   double                *vp;
                        
   hypre_Index            loop_size;
   hypre_Index            cindex;
   hypre_IndexRef         start;
   hypre_Index            startv;
   hypre_Index            stride;
   hypre_Index            stridev;
                        
   int                    i;
   int                    loopi, loopj, loopk;

   /*-----------------------------------------------------
    * Compute encoding digit and strides
    *-----------------------------------------------------*/

   hypre_SetIndex(stride, 1, 1, 1);
   hypre_SetIndex(stridev, 1, 1, 1);

   /*-----------------------------------------------------
    * Filter interpolated error
    *-----------------------------------------------------*/

   hypre_SetIndex(cindex, 0, 0, 0);

   compute_boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(e));
   hypre_ForBoxI(i, compute_boxes)
      {
         compute_box = hypre_BoxArrayBox(compute_boxes, i);

         e_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(e), i);
         v_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(visit), i);

         ep = hypre_StructVectorBoxData(e, i);
         vp = hypre_StructVectorBoxData(visit, i);

         start = hypre_BoxIMin(compute_box);
         hypre_StructMapCoarseToFine(start, cindex, stridev, startv);
         hypre_BoxGetSize(compute_box, loop_size);

         hypre_BoxLoop2Begin(loop_size,
                             e_dbox, start,  stride,  ei,
                             v_dbox, startv, stridev, vi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,ei,vi
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop2For(loopi, loopj, loopk, ei, vi)
            {
               ep[ei] *= vp[vi];
            }
         hypre_BoxLoop2End(ei, vi);
      }

   return ierr;
}

#endif
