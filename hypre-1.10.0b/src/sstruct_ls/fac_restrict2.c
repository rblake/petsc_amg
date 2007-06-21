/******************************************************************************
 *  FAC composite level restriction.
 *  Injection away from the refinement patches; constant restriction
 *  inside patch.
 ******************************************************************************/

#include "headers.h"

#define MapCellRank(i, j , k, rank) \
{\
   rank = 4*k + 2*j + i;\
}

#define InverseMapCellRank(rank, stencil) \
{\
   int ij,ii,jj,kk;\
   ij = (rank%4);\
   ii = (ij%2);\
   jj = (ij-ii)/2;\
   kk = (rank-2*jj-ii)/4;\
   hypre_SetIndex(stencil, ii, jj, kk);\
}

/*--------------------------------------------------------------------------
 * hypre_FacSemiRestrictData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   int                   nvars;
   hypre_Index           stride;

   hypre_SStructPVector *fgrid_cvectors;     /* the grid of this vector may not
                                                be on the actual grid */
   hypre_BoxArrayArray **identity_arrayboxes;
   hypre_BoxArrayArray **fullwgt_ownboxes;
   hypre_BoxArrayArray **fullwgt_sendboxes;

   int                ***own_cboxnums;       /* local crs boxnums of ownboxes */

   hypre_CommPkg       **interlevel_comm;
/*   hypre_CommPkg       **intralevel_comm;*/ /* may need to build an intra comm so
                                                 that each processor only fullwts its
                                                 own fine data- may need to add contrib */

} hypre_FacSemiRestrictData2;

/*--------------------------------------------------------------------------
 * hypre_FacSemiRestrictCreate
 *--------------------------------------------------------------------------*/

int
hypre_FacSemiRestrictCreate2( void **fac_restrict_vdata_ptr)
{
   int                         ierr = 0;
   hypre_FacSemiRestrictData2 *fac_restrict_data;

   fac_restrict_data       = hypre_CTAlloc(hypre_FacSemiRestrictData2, 1);
  *fac_restrict_vdata_ptr  = (void *) fac_restrict_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FacSemiRestrictSetup:
 *   Two types of communication are needed- one for the interlevel coarsened
 *   fine boxes, and the other for the ghostlayer of the restricted vector.
 *--------------------------------------------------------------------------*/

int
hypre_FacSemiRestrictSetup2( void                 *fac_restrict_vdata,
                             hypre_SStructVector  *r,
                             int                   part_crse,
                             int                   part_fine,
                             hypre_SStructPVector *rc,
                             hypre_Index           rfactors )
{
   int                      ierr = 0;

   hypre_FacSemiRestrictData2 *fac_restrict_data = fac_restrict_vdata;
   MPI_Comm                    comm= hypre_SStructPVectorComm(rc);
   hypre_CommInfo             *comm_info;
   hypre_CommPkg             **interlevel_comm;

   hypre_SStructPVector       *rf= hypre_SStructVectorPVector(r, part_fine);
   hypre_StructVector         *s_rc, *s_cvector;
   hypre_SStructPGrid         *pgrid;

   hypre_SStructPVector       *fgrid_cvectors;
   hypre_SStructPGrid         *fgrid_coarsen;
   hypre_BoxArrayArray       **identity_arrayboxes;
   hypre_BoxArrayArray       **fullwgt_ownboxes;
   hypre_BoxArrayArray       **fullwgt_sendboxes;
   hypre_BoxArray             *boxarray;
   hypre_BoxArray             *tmp_boxarray, *intersect_boxes;
   int                      ***own_cboxnums;

   hypre_BoxArrayArray       **send_boxes, *send_rboxes;
   int                      ***send_processes;
   int                      ***send_remote_boxnums;

   hypre_BoxArrayArray       **recv_boxes;
   int                      ***recv_processes;

   hypre_BoxMap               *map;
   hypre_BoxMapEntry         **map_entries;
   int                         nmap_entries;

   hypre_Box                   box, scaled_box;

   hypre_Index                 zero_index, index, ilower, iupper;
   int                         ndim= hypre_SStructVectorNDim(r);
   int                         myproc, proc;
   int                         nvars, vars;
   int                         num_values;

   int                         i, j, k, cnt1, cnt2;
   int                         fi, ci;

   MPI_Comm_rank(comm, &myproc);
   hypre_ClearIndex(zero_index);

   nvars= hypre_SStructPVectorNVars(rc);
  (fac_restrict_data -> nvars)=  nvars;
   hypre_CopyIndex(rfactors, (fac_restrict_data -> stride));

   /* work vector for storing the fullweighted fgrid boxes */
   hypre_SStructPGridCreate(hypre_SStructPVectorComm(rf), ndim, &fgrid_coarsen);
   pgrid= hypre_SStructPVectorPGrid(rf);
   for (vars= 0; vars< nvars; vars++)
   {
      boxarray= hypre_StructGridBoxes(hypre_SStructPGridSGrid(pgrid, vars));
      hypre_ForBoxI(fi, boxarray)
      { 
         box= *hypre_BoxArrayBox(boxarray, fi);
         hypre_StructMapFineToCoarse(hypre_BoxIMin(&box), zero_index,
                                     rfactors, hypre_BoxIMin(&box));
         hypre_StructMapFineToCoarse(hypre_BoxIMax(&box), zero_index,
                                     rfactors, hypre_BoxIMax(&box));
         hypre_SStructPGridSetExtents(fgrid_coarsen,
                                      hypre_BoxIMin(&box),
                                      hypre_BoxIMax(&box));
      }
   }
   hypre_SStructPGridSetVariables( fgrid_coarsen, nvars,
                                   hypre_SStructPGridVarTypes(pgrid) );
   hypre_SStructPGridAssemble(fgrid_coarsen);

   hypre_SStructPVectorCreate(hypre_SStructPGridComm(fgrid_coarsen), fgrid_coarsen,
                              &fgrid_cvectors);
   hypre_SStructPVectorInitialize(fgrid_cvectors);
   hypre_SStructPVectorAssemble(fgrid_cvectors);

   /* pgrid fgrid_coarsen no longer needed */
   hypre_SStructPGridDestroy(fgrid_coarsen);

   fac_restrict_data -> fgrid_cvectors= fgrid_cvectors;

   /*--------------------------------------------------------------------------
    * boxes that are not underlying a fine box:
    *
    * algorithm: subtract all coarsened fine grid boxes that intersect with 
    * this processor's coarse boxes. Note that we cannot loop over all the 
    * coarsened fine boxes and subtract them from the coarse grid since we do
    * not know if some of the overlying fine boxes belong on another 
    * processor. For each cbox, we get a boxarray of boxes that are not
    * underlying-> size(identity_arrayboxes[vars])= #cboxes.
    *--------------------------------------------------------------------------*/
   identity_arrayboxes= hypre_CTAlloc(hypre_BoxArrayArray *, nvars);
   pgrid= hypre_SStructPVectorPGrid(rc);
   hypre_SetIndex(index, rfactors[0]-1, rfactors[1]-1, rfactors[2]-1);

   tmp_boxarray = hypre_BoxArrayCreate(0);
   for (vars= 0; vars< nvars; vars++)
   {
      map= hypre_SStructGridMap(hypre_SStructVectorGrid(r),
                                part_fine, vars);
      boxarray= hypre_StructGridBoxes(hypre_SStructPGridSGrid(pgrid, vars));

      identity_arrayboxes[vars]= hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxarray));

      hypre_ForBoxI(ci, boxarray)
      { 
         box= *hypre_BoxArrayBox(boxarray, ci);
         hypre_AppendBox(&box, 
                          hypre_BoxArrayArrayBoxArray(identity_arrayboxes[vars], ci));
       
         hypre_StructMapCoarseToFine(hypre_BoxIMin(&box), zero_index,
                                     rfactors, hypre_BoxIMin(&scaled_box));
         hypre_StructMapCoarseToFine(hypre_BoxIMax(&box), index,
                                     rfactors, hypre_BoxIMax(&scaled_box));

         hypre_BoxMapIntersect(map, hypre_BoxIMin(&scaled_box),
                               hypre_BoxIMax(&scaled_box), &map_entries,
                              &nmap_entries);

         /* all send and coarsened fboxes on this processor are collected */
         intersect_boxes= hypre_BoxArrayCreate(0);
         for (i= 0; i< nmap_entries; i++)
         {
            hypre_BoxMapEntryGetExtents(map_entries[i], ilower, iupper);
            hypre_BoxSetExtents(&box, ilower, iupper);
            hypre_IntersectBoxes(&box, &scaled_box, &box);

            hypre_StructMapFineToCoarse(hypre_BoxIMin(&box), zero_index,
                                        rfactors, hypre_BoxIMin(&box));
            hypre_StructMapFineToCoarse(hypre_BoxIMax(&box), zero_index,
                                        rfactors, hypre_BoxIMax(&box));
            hypre_AppendBox(&box, intersect_boxes);
         }

         hypre_SubtractBoxArrays(hypre_BoxArrayArrayBoxArray(identity_arrayboxes[vars], ci),
                                 intersect_boxes, tmp_boxarray);
         hypre_MinUnionBoxes(hypre_BoxArrayArrayBoxArray(identity_arrayboxes[vars], ci));

         hypre_TFree(map_entries);
         hypre_BoxArrayDestroy(intersect_boxes);
      }
   } 
   hypre_BoxArrayDestroy(tmp_boxarray);
   fac_restrict_data -> identity_arrayboxes= identity_arrayboxes;

   /*--------------------------------------------------------------------------
    * fboxes that are coarsened. Some will be sent. We create the communication
    * pattern. For each fbox, we need a boxarray of sendboxes or ownboxes.
    *
    * Algorithm: Coarsen each fbox and see which cboxes it intersects using
    * BoxMapIntersect. Cboxes that do not belong on the processor will have
    * a chunk sent to it.
    *--------------------------------------------------------------------------*/
   interlevel_comm= hypre_CTAlloc(hypre_CommPkg *, nvars);
   fullwgt_sendboxes= hypre_CTAlloc(hypre_BoxArrayArray *, nvars);
   fullwgt_ownboxes= hypre_CTAlloc(hypre_BoxArrayArray *, nvars);
   own_cboxnums= hypre_CTAlloc(int **, nvars);

   send_boxes= hypre_CTAlloc(hypre_BoxArrayArray *, nvars);
   send_processes= hypre_CTAlloc(int **, nvars);
   send_remote_boxnums= hypre_CTAlloc(int **, nvars);

   pgrid= hypre_SStructPVectorPGrid(rf);
   for (vars= 0; vars< nvars; vars++)
   {
      map= hypre_SStructGridMap(hypre_SStructVectorGrid(r),
                                part_crse, vars);
      boxarray= hypre_StructGridBoxes(hypre_SStructPGridSGrid(pgrid, vars));
      fullwgt_sendboxes[vars]= hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxarray));
      fullwgt_ownboxes[vars] = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxarray));
      own_cboxnums[vars]     = hypre_CTAlloc(int *, hypre_BoxArraySize(boxarray));

      send_boxes[vars]         = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxarray));
      send_processes[vars]     = hypre_CTAlloc(int *, hypre_BoxArraySize(boxarray));
      send_remote_boxnums[vars]= hypre_CTAlloc(int *, hypre_BoxArraySize(boxarray));

      hypre_ForBoxI(fi, boxarray)
      { 
         box= *hypre_BoxArrayBox(boxarray, fi);
         hypre_StructMapFineToCoarse(hypre_BoxIMin(&box), zero_index,
                                     rfactors, hypre_BoxIMin(&scaled_box));
         hypre_StructMapFineToCoarse(hypre_BoxIMax(&box), zero_index,
                                     rfactors, hypre_BoxIMax(&scaled_box));

         hypre_BoxMapIntersect(map, hypre_BoxIMin(&scaled_box), 
                               hypre_BoxIMax(&scaled_box), &map_entries, &nmap_entries);

         cnt1= 0; cnt2= 0;
         for (i= 0; i< nmap_entries; i++)
         {
            hypre_SStructMapEntryGetProcess(map_entries[i], &proc);
            if (proc != myproc)
            {
               cnt1++;
            }
            else
            {
               cnt2++;
            }
         }
         send_processes[vars][fi]     = hypre_CTAlloc(int, cnt1);
         send_remote_boxnums[vars][fi]= hypre_CTAlloc(int, cnt1);
         own_cboxnums[vars][fi]       = hypre_CTAlloc(int, cnt2);

         cnt1= 0; cnt2= 0;
         for (i= 0; i< nmap_entries; i++)
         {
            hypre_BoxMapEntryGetExtents(map_entries[i], ilower, iupper);
            hypre_BoxSetExtents(&box, ilower, iupper);
            hypre_IntersectBoxes(&box, &scaled_box, &box);

            hypre_SStructMapEntryGetProcess(map_entries[i], &proc);
            if (proc != myproc)
            {
               hypre_AppendBox(&box,
                                hypre_BoxArrayArrayBoxArray(fullwgt_sendboxes[vars], fi));
               hypre_AppendBox(&box,
                                hypre_BoxArrayArrayBoxArray(send_boxes[vars], fi));

               send_processes[vars][fi][cnt1]= proc;
               hypre_SStructMapEntryGetBox(map_entries[i],
                                           &send_remote_boxnums[vars][fi][cnt1]);
               cnt1++;
            }
           
            else
            {
               hypre_AppendBox(&box,
                                hypre_BoxArrayArrayBoxArray(fullwgt_ownboxes[vars], fi));
               hypre_SStructMapEntryGetBox(map_entries[i],
                                          &own_cboxnums[vars][fi][cnt2]);
               cnt2++;
            }
         }
         hypre_TFree(map_entries);

      }  /* hypre_ForBoxI(fi, boxarray) */
   }     /* for (vars= 0; vars< nvars; vars++) */

  (fac_restrict_data -> fullwgt_sendboxes)= fullwgt_sendboxes;
  (fac_restrict_data -> fullwgt_ownboxes)= fullwgt_ownboxes;
  (fac_restrict_data -> own_cboxnums)= own_cboxnums;
   
   /*--------------------------------------------------------------------------
    * coarsened fboxes this processor will receive. 
    *
    * Algorithm: For each cbox on this processor, refine it and find which
    * processors the refinement belongs in. The processors owning a chunk
    * are the recv_processors.
    *--------------------------------------------------------------------------*/
   recv_boxes= hypre_CTAlloc(hypre_BoxArrayArray *, nvars);
   recv_processes= hypre_CTAlloc(int **, nvars);

   pgrid= hypre_SStructPVectorPGrid(rc);
   for (vars= 0; vars< nvars; vars++)
   {
      map= hypre_SStructGridMap(hypre_SStructVectorGrid(r),
                                part_fine, vars);
      boxarray= hypre_StructGridBoxes(hypre_SStructPGridSGrid(pgrid, vars));
      
      recv_boxes[vars]    = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxarray));
      recv_processes[vars]= hypre_CTAlloc(int *, hypre_BoxArraySize(boxarray));

      hypre_ForBoxI(ci, boxarray)
      { 
         box= *hypre_BoxArrayBox(boxarray, ci);
         hypre_StructMapCoarseToFine(hypre_BoxIMin(&box), zero_index,
                                     rfactors, hypre_BoxIMin(&scaled_box));
         hypre_StructMapCoarseToFine(hypre_BoxIMax(&box), index,
                                     rfactors, hypre_BoxIMax(&scaled_box));

         hypre_BoxMapIntersect(map, hypre_BoxIMin(&scaled_box), 
                               hypre_BoxIMax(&scaled_box), &map_entries, &nmap_entries);

         cnt1= 0;
         for (i= 0; i< nmap_entries; i++)
         {
            hypre_SStructMapEntryGetProcess(map_entries[i], &proc);
            if (proc != myproc)
            {
               cnt1++;
            }
         }
         recv_processes[vars][ci]= hypre_CTAlloc(int, cnt1);

         cnt1= 0;
         for (i= 0; i< nmap_entries; i++)
         {
            hypre_SStructMapEntryGetProcess(map_entries[i], &proc);
            if (proc != myproc)
            {
               hypre_BoxMapEntryGetExtents(map_entries[i], ilower, iupper);
               hypre_BoxSetExtents(&box, ilower, iupper);
               hypre_IntersectBoxes(&box, &scaled_box, &box);

               /* contract this refined box so that only the coarse nodes on this
                  processor are received . */
               for (j= 0; j< ndim; j++)
               {
                  k= hypre_BoxIMin(&box)[j] % rfactors[j];
                  if (k)
                  {
                     hypre_BoxIMin(&box)[j]+= rfactors[j] - k;
                  }
               }

               hypre_StructMapFineToCoarse(hypre_BoxIMin(&box), zero_index,
                                           rfactors, hypre_BoxIMin(&box));
               hypre_StructMapFineToCoarse(hypre_BoxIMax(&box), zero_index,
                                           rfactors, hypre_BoxIMax(&box));
               hypre_AppendBox(&box,
                                hypre_BoxArrayArrayBoxArray(recv_boxes[vars], ci));

               recv_processes[vars][ci][cnt1]= proc;
               cnt1++;

            }  /* if (proc != myproc) */
         }     /* for (i= 0; i< nmap_entries; i++) */

         hypre_TFree(map_entries);

      }        /* hypre_ForBoxI(ci, boxarray) */
   }           /* for (vars= 0; vars< nvars; vars++) */

   num_values= 1;
   for (vars= 0; vars< nvars; vars++)
   {
      s_rc     = hypre_SStructPVectorSVector(rc, vars);
      s_cvector= hypre_SStructPVectorSVector(fgrid_cvectors, vars);
      send_rboxes= hypre_BoxArrayArrayDuplicate(send_boxes[vars]);

      hypre_CommInfoCreate(send_boxes[vars], recv_boxes[vars], send_processes[vars],
                           recv_processes[vars], send_remote_boxnums[vars],
                           send_rboxes, &comm_info);

      hypre_CommPkgCreate(comm_info,
                          hypre_StructVectorDataSpace(s_cvector),
                          hypre_StructVectorDataSpace(s_rc),
                          num_values,
                          hypre_StructVectorComm(s_rc),
                         &interlevel_comm[vars]);
  }
  hypre_TFree(send_boxes);
  hypre_TFree(recv_boxes);
  hypre_TFree(send_processes);
  hypre_TFree(recv_processes);
  hypre_TFree(send_remote_boxnums);
      
  (fac_restrict_data -> interlevel_comm)= interlevel_comm;

   return ierr;

}

int
hypre_FACRestrict2( void                 *  fac_restrict_vdata,
                    hypre_SStructVector  *  xf,
                    hypre_SStructPVector *  xc)
{
   int ierr = 0;

   hypre_FacSemiRestrictData2 *restrict_data = fac_restrict_vdata;

   hypre_SStructPVector   *fgrid_cvectors     = restrict_data->fgrid_cvectors;
   hypre_BoxArrayArray   **identity_arrayboxes= restrict_data->identity_arrayboxes;
   hypre_BoxArrayArray   **fullwgt_ownboxes   = restrict_data->fullwgt_ownboxes;
   int                  ***own_cboxnums       = restrict_data->own_cboxnums;
   hypre_CommPkg         **interlevel_comm= restrict_data-> interlevel_comm;
   hypre_CommHandle       *comm_handle;

   hypre_BoxArrayArray    *arrayarray_ownboxes;

   hypre_IndexRef          stride;  /* refinement factors */

   hypre_StructGrid       *fgrid;
   hypre_BoxArray         *fgrid_boxes;
   hypre_Box              *fgrid_box;
   hypre_StructGrid       *cgrid;
   hypre_BoxArray         *cgrid_boxes;
   hypre_Box              *cgrid_box;
   hypre_BoxArray         *own_boxes;
   hypre_Box              *own_box;
   int                    *boxnums;

   hypre_Box              *xc_temp_dbox;
   hypre_Box              *xf_dbox;

   hypre_StructVector     *xc_temp;
   hypre_StructVector     *xc_var;
   hypre_StructVector     *xf_var;

   int                     xci;
   int                     xfi;

   double               ***xfp;
   double               ***xcp;
   double               ***xcp_temp;

   hypre_Index             loop_size;
   hypre_Index             start, fbox_size, node_offset;
   hypre_Index             startc;
   hypre_Index             stridec;
   hypre_Index             rfactors;
   hypre_Index             temp_index1, temp_index2;

   int                     fi, ci;
   int                     loopi, loopj, loopk;
   int                     nvars, var, ndim;
   int                     volume_crse_cell;

   int                     i, j, k;
   int                     imax, jmax, kmax;
   int                     icell, jcell, kcell, ijkcell;

   double                 *sum;
   double                  scaling;

   int                     part_crse= 0;
   int                     part_fine= 1;
   int                     num_coarse_cells;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/
   stride        = (restrict_data -> stride);

   hypre_SetIndex(stridec, 1, 1, 1);
   hypre_CopyIndex(stride, rfactors);

   volume_crse_cell= 1;
   for (i= 0; i< 3; i++)
   {
      volume_crse_cell *= rfactors[i];
   }

   /*-----------------------------------------------------------------------
    * We are assuming the refinement and coarsening have same variable
    * types.
    *-----------------------------------------------------------------------*/
   nvars=  hypre_SStructPVectorNVars(xc);
   ndim =  hypre_SStructVectorNDim(xf);

   /*-----------------------------------------------------------------------
    * For each coordinate direction, a fine node can contribute only to the 
    * left or right cell=> only 2 coarse cells per direction.
    *-----------------------------------------------------------------------*/
   num_coarse_cells= 1;
   for (i= 0; i< ndim; i++)
   {
      num_coarse_cells*= 2;
   }
   sum= hypre_CTAlloc(double, num_coarse_cells);

   /*--------------------------------------------------------------------------
    * Scaling for averaging restriction.
    *--------------------------------------------------------------------------*/
   scaling= 1.0;
   for (i= 0; i< ndim-2; i++)
   {
      scaling*= rfactors[0];
   }

   /*-----------------------------------------------------------------------
    * Copy the coarse data: xf[part_crse] -> xc
    *-----------------------------------------------------------------------*/
   hypre_SStructPartialPCopy(hypre_SStructVectorPVector(xf, part_crse), 
                             xc, identity_arrayboxes);

   /*-----------------------------------------------------------------------
    * Piecewise constant restriction over the refinement patch. 
    *
    * Initialize the work vector by setting to zero.
    *-----------------------------------------------------------------------*/
   hypre_SStructPVectorSetConstantValues(fgrid_cvectors, 0.0);

   /*-----------------------------------------------------------------------
    * Allocate memory for the data pointers. Assuming constant restriction.
    * We stride through the refinement patch by the refinement factors, and 
    * so we must have pointers to the intermediate fine nodes=> xfp will
    * be size rfactors[2]*rfactors[1].
    *-----------------------------------------------------------------------*/
   xcp_temp= hypre_TAlloc(double **, (ndim-1));
   xcp     = hypre_TAlloc(double **, (ndim-1));
   xfp     = hypre_TAlloc(double **, rfactors[2]);

   for (k= 0; k< (ndim-1); k++)
   {
      xcp_temp[k]= hypre_TAlloc(double *, 2);
      xcp[k]     = hypre_TAlloc(double *, 2);
   }

   for (k= 0; k< rfactors[2]; k++)
   {
      xfp[k]= hypre_TAlloc(double *, rfactors[1]);
   }

   for (var= 0; var< nvars; var++)
   {
      xc_temp= hypre_SStructPVectorSVector(fgrid_cvectors, var);
      xf_var= hypre_SStructPVectorSVector(hypre_SStructVectorPVector(xf,part_fine),
                                          var);

      fgrid        = hypre_StructVectorGrid(xf_var);
      fgrid_boxes  = hypre_StructGridBoxes(fgrid);
      cgrid        = hypre_StructVectorGrid(xc_temp);
      cgrid_boxes  = hypre_StructGridBoxes(cgrid);

      hypre_ForBoxI(fi, fgrid_boxes)
      {
          fgrid_box= hypre_BoxArrayBox(fgrid_boxes, fi);

         /*--------------------------------------------------------------------
          * Get the ptrs for the fine struct_vectors.
          *--------------------------------------------------------------------*/
          xf_dbox  = hypre_BoxArrayBox(hypre_StructVectorDataSpace(xf_var), fi);
          for (k= 0; k< rfactors[2]; k++)
          {
             for (j=0; j< rfactors[1]; j++)
             {
                hypre_SetIndex(temp_index1, 0, j, k);
                xfp[k][j]= hypre_StructVectorBoxData(xf_var, fi) +
                           hypre_BoxOffsetDistance(xf_dbox, temp_index1);
             }
          }

         /*--------------------------------------------------------------------
          * Get the ptrs for the coarse struct_vectors. Note that the coarse
          * work vector is indexed with respect to the local fine box no.'s.
          *--------------------------------------------------------------------*/
          xc_temp_dbox= hypre_BoxArrayBox(hypre_StructVectorDataSpace(xc_temp), fi);
          for (k= 0; k< (ndim-1); k++)
          {
             for (j=0; j< 2; j++)
             {
                hypre_SetIndex(temp_index1, 0, j, k);
                xcp_temp[k][j]= hypre_StructVectorBoxData(xc_temp, fi) +
                                hypre_BoxOffsetDistance(xc_temp_dbox, temp_index1);
             }
          }

          hypre_CopyIndex(hypre_BoxIMin(fgrid_box), start);
          hypre_CopyIndex(hypre_BoxIMax(fgrid_box), fbox_size);

         /*--------------------------------------------------------------------
          * Adjust "fbox_size" so that this hypre_Index is appropriate for
          * ndim < 3.
          *--------------------------------------------------------------------*/
          for (i= 0; i< 3; i++)
          {
             fbox_size[i]-= (start[i]-1);
          }

         /*--------------------------------------------------------------------
          * The fine intersection box may not be divisible by the refinement
          * factor. We need to know the remainder to determine which
          * coarse node gets the restricted values.
          *--------------------------------------------------------------------*/
          for (i= 0; i< 3; i++)
          {
             node_offset[i]= rfactors[i]-(start[i]%rfactors[i])-1;
          }

          hypre_SetIndex(temp_index2, 0, 0, 0);
          hypre_StructMapFineToCoarse(start, temp_index2, rfactors, startc);

          hypre_BoxGetSize(fgrid_box, temp_index1);
          hypre_StructMapFineToCoarse(temp_index1, temp_index2, rfactors, loop_size);

          hypre_BoxLoop2Begin(loop_size,
                              xf_dbox, start, stride,  xfi,
                              xc_temp_dbox, startc, stridec, xci);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,xfi,xci
#include "hypre_box_smp_forloop.h"
          hypre_BoxLoop2For(loopi, loopj, loopk, xfi, xci)
          {
              /*-----------------------------------------------------------------
               * Arithmetic average the refinement patch values to get 
               * restricted coarse grid values in an agglomerate; i.e.,
               * piecewise constant restriction.
               *-----------------------------------------------------------------*/
               imax= hypre_min( (fbox_size[0]-loopi*stride[0]), rfactors[0] );
               jmax= hypre_min( (fbox_size[1]-loopj*stride[1]), rfactors[1] );
               kmax= hypre_min( (fbox_size[2]-loopk*stride[2]), rfactors[2] );

               for (i= 0; i< num_coarse_cells; i++)
               {
                  sum[i]= 0.0;
               }

               for (k= 0; k< kmax; k++)
               {
                  kcell= 1;
                  if (k <= node_offset[2])
                  {
                     kcell= 0;
                  }

                  for (j= 0; j< jmax; j++)
                  {
                     jcell= 1;
                     if (j <= node_offset[1])
                     {
                         jcell= 0;
                     }

                     for (i= 0; i< imax; i++)
                     {
                        icell= 1;
                        if (i <= node_offset[0])
                        {
                           icell= 0;
                        }

                        MapCellRank(icell, jcell , kcell, ijkcell);
                        sum[ijkcell]+= xfp[k][j][xfi+i];
                     }   
                  }     
               }       
 
              /*-----------------------------------------------------------------
               * Add the compute averages to the correct coarse cell.
               *-----------------------------------------------------------------*/
               for (ijkcell= 0; ijkcell< num_coarse_cells; ijkcell++)
               {
                  if (sum[ijkcell] != 0.0)
                  {
                     sum[ijkcell]/= scaling;
                     InverseMapCellRank(ijkcell, temp_index2);
                     i=  temp_index2[0];
                     j=  temp_index2[1];
                     k=  temp_index2[2];
                     xcp_temp[k][j][xci+i]+= sum[ijkcell];
                  }
               }

          }
          hypre_BoxLoop2End(xfi, xci);

      }   /* hypre_ForBoxI(fi, fgrid_boxes) */
   }      /* for (var= 0; var< nvars; var++)*/

   /*------------------------------------------------------------------
    * Communicate calculated restricted function over the coarsened
    * patch. Only actual communicated values will be put in the
    * coarse vector.
    *------------------------------------------------------------------*/
   for (var= 0; var< nvars; var++)
   {
       xc_temp= hypre_SStructPVectorSVector(fgrid_cvectors, var);
       xc_var= hypre_SStructPVectorSVector(xc, var);
       hypre_InitializeCommunication(interlevel_comm[var], 
                                     hypre_StructVectorData(xc_temp),
                                     hypre_StructVectorData(xc_var), 
                                    &comm_handle);

       hypre_FinalizeCommunication(comm_handle);
   }

   /*------------------------------------------------------------------
    * Need to add the coarsened patches that belong on this processor
    * to the coarse vector.
    *------------------------------------------------------------------*/
   for (var= 0; var< nvars; var++)
   {
      xc_temp= hypre_SStructPVectorSVector(fgrid_cvectors, var);
      xc_var= hypre_SStructPVectorSVector(xc, var);

      cgrid        = hypre_StructVectorGrid(xc_temp);
      cgrid_boxes  = hypre_StructGridBoxes(cgrid);

      arrayarray_ownboxes= fullwgt_ownboxes[var];
      hypre_ForBoxI(ci, cgrid_boxes)
      {
         cgrid_box= hypre_BoxArrayBox(cgrid_boxes, ci);
         xc_temp_dbox= hypre_BoxArrayBox(hypre_StructVectorDataSpace(xc_temp), ci);
         xcp_temp[0][0]= hypre_StructVectorBoxData(xc_temp, ci);

        /*--------------------------------------------------------------
         * Each ci box of cgrid_box has a boxarray of subboxes. Copy
         * each of these subboxes to the coarse vector.
         *--------------------------------------------------------------*/
         own_boxes= hypre_BoxArrayArrayBoxArray(arrayarray_ownboxes, ci);
         boxnums  = own_cboxnums[var][ci];
         hypre_ForBoxI(i, own_boxes)
         {
            own_box= hypre_BoxArrayBox(own_boxes, i);
            xf_dbox= hypre_BoxArrayBox(hypre_StructVectorDataSpace(xc_var), boxnums[i]);
            xcp[0][0]= hypre_StructVectorBoxData(xc_var, boxnums[i]);

            hypre_BoxGetSize(own_box, loop_size);
            hypre_BoxLoop2Begin(loop_size,
                                xc_temp_dbox, hypre_BoxIMin(own_box), stridec, xfi,
                                xf_dbox, hypre_BoxIMin(own_box), stridec, xci);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,xfi,xci
#include "hypre_box_smp_forloop.h"
            hypre_BoxLoop2For(loopi, loopj, loopk, xfi, xci)
            {
                xcp[0][0][xci]= xcp_temp[0][0][xfi];
            }
            hypre_BoxLoop2End(xfi, xci);
         
         }  /* hypre_ForBoxI(i, own_boxes) */
      }     /* hypre_ForBoxI(ci, cgrid_boxes) */
   }        /* for (var= 0; var< nvars; var++) */
      
   hypre_TFree(sum);
   for (k= 0; k< rfactors[2]; k++)
   {
       hypre_TFree(xfp[k]);
   }
   hypre_TFree(xfp);

   for (k= 0; k< (ndim-1); k++)
   {
       hypre_TFree(xcp_temp[k]);
       hypre_TFree(xcp[k]);
   }

   hypre_TFree(xcp_temp);
   hypre_TFree(xcp);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FacSemiRestrictDestroy
 *--------------------------------------------------------------------------*/

int
hypre_FacSemiRestrictDestroy2( void *fac_restrict_vdata )
{
   int                         ierr = 0;
   hypre_FacSemiRestrictData2 *fac_restrict_data = fac_restrict_vdata;
   int                         nvars= (fac_restrict_data-> nvars); 
   int                         i, j;


   if (fac_restrict_data)
   {
      hypre_SStructPVectorDestroy(fac_restrict_data-> fgrid_cvectors);

      for (i= 0; i< nvars; i++)
      {
         hypre_BoxArrayArrayDestroy((fac_restrict_data -> identity_arrayboxes)[i]);
         hypre_BoxArrayArrayDestroy((fac_restrict_data -> fullwgt_sendboxes)[i]);
         for (j= 0; j< hypre_BoxArrayArraySize(fac_restrict_data->fullwgt_ownboxes[i]); j++)
         {
             hypre_TFree((fac_restrict_data -> own_cboxnums)[i][j]);
         }
         hypre_TFree((fac_restrict_data -> own_cboxnums)[i]);

         hypre_BoxArrayArrayDestroy((fac_restrict_data -> fullwgt_ownboxes)[i]);
         hypre_CommPkgDestroy((fac_restrict_data -> interlevel_comm)[i]);
      }

      hypre_TFree(fac_restrict_data -> identity_arrayboxes);
      hypre_TFree(fac_restrict_data -> fullwgt_sendboxes);
      hypre_TFree(fac_restrict_data -> own_cboxnums);
      hypre_TFree(fac_restrict_data -> fullwgt_ownboxes);
      hypre_TFree(fac_restrict_data -> interlevel_comm);

      hypre_TFree(fac_restrict_data);
   }
   return ierr;

}
