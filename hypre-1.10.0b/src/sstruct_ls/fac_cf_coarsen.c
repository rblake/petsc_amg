#include "headers.h" 

#define MapStencilRank(stencil, rank) \
{\
   int ii,jj,kk;\
   ii = hypre_IndexX(stencil);\
   jj = hypre_IndexY(stencil);\
   kk = hypre_IndexZ(stencil);\
   if (ii==-1)\
      ii=2;\
   if (jj==-1)\
      jj=2;\
   if (kk==-1)\
      kk=2;\
   rank = ii + 3*jj + 9*kk;\
}

#define InverseMapStencilRank(rank, stencil) \
{\
   int ij,ii,jj,kk;\
   ij = (rank%9);\
   ii = (ij%3);\
   jj = (ij-ii)/3;\
   kk = (rank-3*jj-ii)/9;\
   if (ii==2)\
      ii= -1;\
   if (jj==2)\
      jj= -1;\
   if (kk==2)\
      kk= -1;\
   hypre_SetIndex(stencil, ii, jj, kk);\
}


#define AbsStencilShape(stencil, abs_shape) \
{\
   int ii,jj,kk;\
   ii = hypre_IndexX(stencil);\
   jj = hypre_IndexY(stencil);\
   kk = hypre_IndexZ(stencil);\
   abs_shape= abs(ii) + abs(jj) + abs(kk); \
}

/*--------------------------------------------------------------------------
 * hypre_AMR_CFCoarsen: Coarsens the CF interface to get the stencils 
 * reaching into a coarsened fbox. Also sets the centre coefficient of CF
 * interface nodes to have "preserved" row sum. 
 * 
 * On entry, fac_A already has all the coefficient values of the cgrid
 * chunks that are not underlying a fbox.  Note that A & fac_A have the
 * same grid & graph. Therefore, we will use A's grid & graph.
 *
 * ASSUMING ONLY LIKE-VARIABLES COUPLE THROUGH CF CONNECTIONS.
 *--------------------------------------------------------------------------*/

int
hypre_AMR_CFCoarsen( hypre_SStructMatrix  *   A,
                     hypre_SStructMatrix  *   fac_A,
                     hypre_Index              refine_factors,
                     int                      level ) 

{
   MPI_Comm                comm       = hypre_SStructMatrixComm(A);
   hypre_SStructGraph     *graph      = hypre_SStructMatrixGraph(A);
   int                     graph_type = hypre_SStructGraphObjectType(graph);
   hypre_SStructGrid      *grid       = hypre_SStructGraphGrid(graph);
   int                     nUventries = hypre_SStructGraphNUVEntries(graph);
   HYPRE_IJMatrix          ij_A       = hypre_SStructMatrixIJMatrix(A);
   int                     matrix_type= hypre_SStructMatrixObjectType(A);
   int                     ndim       = hypre_SStructMatrixNDim(A);

   hypre_SStructPMatrix   *A_pmatrix;
   hypre_StructMatrix     *smatrix_var;
   hypre_StructStencil    *stencils;
   int                     stencil_size;
   hypre_Index             stencil_shape_i;
   hypre_Index             loop_size;
   hypre_Box               refined_box;
   double                **a_ptrs;
   hypre_Box              *A_dbox;

   int                     part_crse= level-1;
   int                     part_fine= level;
 
   hypre_BoxMap           *fmap;
   hypre_BoxMapEntry     **map_entries, *map_entry;
   int                     nmap_entries;
   hypre_Box               map_entry_box;

   hypre_BoxArrayArray  ***fgrid_cinterface_extents;

   hypre_StructGrid       *cgrid;
   hypre_BoxArray         *cgrid_boxes;
   hypre_Box              *cgrid_box;
   hypre_Index             node_extents;
   hypre_Index             stridec, stridef;

   hypre_BoxArrayArray    *cinterface_arrays;
   hypre_BoxArray         *cinterface_array;
   hypre_Box              *fgrid_cinterface;

   int                     centre;

   int                     ci, fi, boxi;
   int                     max_stencil_size= 27;
   int                     false= 0;
   int                     true = 1;
   int                     found;
   int                    *stencil_ranks, *rank_stencils;
   int                     rank, startrank;
   double                 *vals;

   int                     i, j, iA;
   int                     loopi, loopj, loopk;
   int                     nvars, var1; 

   hypre_Index             zero_index;
   hypre_Index             index1, index2;
   hypre_Index             index_temp;

   hypre_SStructUVEntry   *Uventry;
   int                     nUentries, cnt1;
   int                     box_array_size;

   int                    *ncols, *rows, *cols;
   
   int                    *temp1, *temp2;

   int                     myid;

   MPI_Comm_rank(comm, &myid);
   hypre_SetIndex(zero_index, 0, 0, 0);
   
   /*--------------------------------------------------------------------------
    *  Task: Coarsen the CF interface connections of A into fac_A so that 
    *  fac_A will have the stencil coefficients extending into a coarsened
    *  fbox. The centre coefficient is constructed to preserve the row sum.
    *--------------------------------------------------------------------------*/

   if (graph_type == HYPRE_SSTRUCT)
   {
      startrank   = hypre_SStructGridGhstartRank(grid);
   }
   if (graph_type == HYPRE_PARCSR)
   {
      startrank   = hypre_SStructGridStartRank(grid);
   }

   /*--------------------------------------------------------------------------
    * Fine grid strides by the refinement factors.
    *--------------------------------------------------------------------------*/
    hypre_SetIndex(stridec, 1, 1, 1);
    for (i= 0; i< ndim; i++)
    {
        stridef[i]= refine_factors[i];
    }
    for (i= ndim; i< 3; i++)
    {
        stridef[i]= 1;
    }

   /*--------------------------------------------------------------------------
    *  Determine the c/f interface index boxes: fgrid_cinterface_extents.
    *  These are between fpart= level and cpart= (level-1). The 
    *  fgrid_cinterface_extents are indexed by cboxes, but fboxes that
    *  abutt a given cbox must be considered. Moreover, for each fbox,
    *  we can have a c/f interface from a number of different stencil
    *  directions- i.e., we have a boxarrayarray for each cbox, each
    *  fbox leading to a boxarray.
    *
    *  Algo.: For each cbox:
    *    1) refine & stretch by a unit in each dimension.
    *    2) boxmap_intersect with the fgrid map to get all fboxes contained
    *       or abutting this cbox.
    *    3) get the fgrid_cinterface_extents for each of these fboxes.
    *
    *  fgrid_cinterface_extents[var1][ci]
    *--------------------------------------------------------------------------*/
    A_pmatrix=  hypre_SStructMatrixPMatrix(fac_A, part_crse);
    nvars    =  hypre_SStructPMatrixNVars(A_pmatrix);

    fgrid_cinterface_extents= hypre_TAlloc(hypre_BoxArrayArray **, nvars);
    for (var1= 0; var1< nvars; var1++)
    {
       fmap= hypre_SStructGridMap(grid, part_fine, var1);
       stencils= hypre_SStructPMatrixSStencil(A_pmatrix, var1, var1);

       cgrid= hypre_SStructPGridSGrid(hypre_SStructPMatrixPGrid(A_pmatrix), var1);
       cgrid_boxes= hypre_StructGridBoxes(cgrid); 
       fgrid_cinterface_extents[var1]= hypre_TAlloc(hypre_BoxArrayArray *, 
                                                    hypre_BoxArraySize(cgrid_boxes));

       hypre_ForBoxI(ci, cgrid_boxes)
       {
          cgrid_box= hypre_BoxArrayBox(cgrid_boxes, ci);

          hypre_StructMapCoarseToFine(hypre_BoxIMin(cgrid_box), zero_index,
                                      refine_factors, hypre_BoxIMin(&refined_box));
          hypre_SetIndex(index1, refine_factors[0]-1, refine_factors[1]-1,
                         refine_factors[2]-1);
          hypre_StructMapCoarseToFine(hypre_BoxIMax(cgrid_box), index1,
                                      refine_factors, hypre_BoxIMax(&refined_box));

         /*------------------------------------------------------------------------
          * Stretch the refined_box so that a BoxMapIntersect will get abutting
          * fboxes.
          *------------------------------------------------------------------------*/
          for (i= 0; i< ndim; i++)
          {
             hypre_BoxIMin(&refined_box)[i]-= 1;
             hypre_BoxIMax(&refined_box)[i]+= 1;
          }

          hypre_BoxMapIntersect(fmap, hypre_BoxIMin(&refined_box),
                                hypre_BoxIMax(&refined_box), &map_entries,
                               &nmap_entries);

          fgrid_cinterface_extents[var1][ci]= hypre_BoxArrayArrayCreate(nmap_entries);

         /*------------------------------------------------------------------------
          * Get the  fgrid_cinterface_extents using var1-var1 stencil (only like-
          * variables couple).
          *------------------------------------------------------------------------*/
          if (stencils != NULL)
          {
             for (i= 0; i< nmap_entries; i++)
             {
                hypre_BoxMapEntryGetExtents(map_entries[i],
                                            hypre_BoxIMin(&map_entry_box),
                                            hypre_BoxIMax(&map_entry_box));
                hypre_CFInterfaceExtents2(&map_entry_box, cgrid_box, stencils, refine_factors,
                        hypre_BoxArrayArrayBoxArray(fgrid_cinterface_extents[var1][ci], i) );
             }
          }
          hypre_TFree(map_entries);

       }  /* hypre_ForBoxI(ci, cgrid_boxes) */
    }     /* for (var1= 0; var1< nvars; var1++) */

   /*--------------------------------------------------------------------------
    *  STEP 1:
    *        ADJUST THE ENTRIES ALONG THE C/F BOXES SO THAT THE COARSENED
    *        C/F CONNECTION HAS THE APPROPRIATE ROW SUM. 
    *        WE ARE ASSUMING ONLY LIKE VARIABLES COUPLE.
    *--------------------------------------------------------------------------*/
    for (var1= 0; var1< nvars; var1++)
    {
       cgrid= hypre_SStructPGridSGrid(hypre_SStructPMatrixPGrid(A_pmatrix), var1);
       cgrid_boxes= hypre_StructGridBoxes(cgrid);
       stencils=  hypre_SStructPMatrixSStencil(A_pmatrix, var1, var1);

       /*----------------------------------------------------------------------
        * Extract only where variables couple.
        *----------------------------------------------------------------------*/
       if (stencils != NULL)
       {
           stencil_size= hypre_StructStencilSize(stencils);

           /*------------------------------------------------------------------
            *  stencil_ranks[i]      =  rank of stencil entry i.
            *  rank_stencils[i]      =  stencil entry of rank i.
            *
            * These are needed in collapsing the unstructured connections to
            * a stencil connection.
            *------------------------------------------------------------------*/
           stencil_ranks= hypre_TAlloc(int, stencil_size);
           rank_stencils= hypre_TAlloc(int, max_stencil_size);
           for (i= 0; i< max_stencil_size; i++)
           {
              rank_stencils[i]= -1;
              if (i < stencil_size)
              {
                 stencil_ranks[i]= -1;
              }
           }

           for (i= 0; i< stencil_size; i++)
           {
               hypre_CopyIndex(hypre_StructStencilElement(stencils, i), stencil_shape_i);
               MapStencilRank(stencil_shape_i, j);
               stencil_ranks[i]= j;
               rank_stencils[stencil_ranks[i]] = i;            
           }
           centre= rank_stencils[0];

           smatrix_var = hypre_SStructPMatrixSMatrix(A_pmatrix, var1, var1);

           a_ptrs   = hypre_TAlloc(double *, stencil_size);
           hypre_ForBoxI(ci, cgrid_boxes)
           {
              cgrid_box= hypre_BoxArrayBox(cgrid_boxes, ci);

              cinterface_arrays= fgrid_cinterface_extents[var1][ci];
              A_dbox= hypre_BoxArrayBox(hypre_StructMatrixDataSpace(smatrix_var), ci);

              /*-----------------------------------------------------------------
               * Ptrs to the correct data location.
               *-----------------------------------------------------------------*/
              for (i= 0; i< stencil_size; i++)
              {
                 hypre_CopyIndex(hypre_StructStencilElement(stencils, i), stencil_shape_i);
                 a_ptrs[i]= hypre_StructMatrixExtractPointerByIndex(smatrix_var,
                                                                    ci,
                                                                    stencil_shape_i);
              }

              /*-------------------------------------------------------------------
               * Loop over the c/f interface boxes and set the centre to be the row
               * sum. Coarsen the c/f connection and set the centre to preserve
               * the row sum of the composite operator along the c/f interface.
               *-------------------------------------------------------------------*/
              hypre_ForBoxArrayI(fi, cinterface_arrays)
              {
                 cinterface_array= hypre_BoxArrayArrayBoxArray(cinterface_arrays, fi);
                 box_array_size  = hypre_BoxArraySize(cinterface_array);
                 for (boxi= stencil_size; boxi< box_array_size; boxi++)
                 {
                    fgrid_cinterface= hypre_BoxArrayBox(cinterface_array, boxi);
                    hypre_CopyIndex(hypre_BoxIMin(fgrid_cinterface), node_extents);
                    hypre_BoxGetSize(fgrid_cinterface, loop_size);
                    
                    hypre_BoxLoop1Begin(loop_size,
                                        A_dbox, node_extents, stridec, iA);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,iA
#include "hypre_box_smp_forloop.h"
                    hypre_BoxLoop1For(loopi, loopj, loopk, iA)
                    {
                       for (i= 0; i< stencil_size; i++)
                       {
                          if (i != centre)
                          {
                             a_ptrs[centre][iA]+= a_ptrs[i][iA];
                          }
                       }

                       /*-----------------------------------------------------------------
                        * Search for unstructured connections for this coarse node. Need
                        * to compute the index of the node. We will "collapse" the
                        * unstructured connections to the appropriate stencil entry. Thus
                        * we need to serch for the stencil entry. 
                        *-----------------------------------------------------------------*/
                       index_temp[0]= node_extents[0] + loopi;
                       index_temp[1]= node_extents[1] + loopj;
                       index_temp[2]= node_extents[2] + loopk;

                       hypre_SStructGridFindMapEntry(grid, part_crse, index_temp, var1,
                                                     &map_entry);
                       hypre_SStructMapEntryGetGlobalRank(map_entry, index_temp, &rank,
                                                          matrix_type);
                       if (nUventries > 0)
                       {
                          found= false;
                          if ((rank-startrank) >= hypre_SStructGraphIUVEntry(graph, 0) &&
                              (rank-startrank) <= hypre_SStructGraphIUVEntry(graph, nUventries-1))
                          {
                              found= true;
                          }
                       }

                       /*-----------------------------------------------------------------
                        * The graph has Uventries only if (nUventries > 0). Therefore,
                        * check this. Only like variables contribute to the row sum.
                        *-----------------------------------------------------------------*/
                       if (nUventries > 0 && found == true)
                       {
                           Uventry= hypre_SStructGraphUVEntry(graph, rank-startrank);

                           if (Uventry != NULL)
                           {
                               nUentries= hypre_SStructUVEntryNUEntries(Uventry);
                              
                              /*-----------------------------------------------------------
                               * extract only the connections to level part_fine and the
                               * correct variable.
                               *-----------------------------------------------------------*/
                               temp1= hypre_CTAlloc(int, nUentries);
                               cnt1= 0;
                               for (i=0; i< nUentries; i++)
                               {
                                  if (hypre_SStructUVEntryToPart(Uventry, i) == part_fine 
                                          &&  hypre_SStructUVEntryToVar(Uventry, i) == var1)
                                  {
                                     temp1[cnt1++]= i;
                                  }
                               }

                               ncols= hypre_TAlloc(int, cnt1);
                               rows = hypre_TAlloc(int, cnt1);
                               cols = hypre_TAlloc(int, cnt1);
                               temp2= hypre_TAlloc(int, cnt1);
                               vals = hypre_CTAlloc(double, cnt1);

                               for (i= 0; i< cnt1; i++)
                               {
                                  ncols[i]= 1;
                                  rows[i] = rank;
                                  cols[i] = hypre_SStructUVEntryRank(Uventry, temp1[i]);
                         
                                 /* determine the stencil connection pattern */
                                  hypre_StructMapFineToCoarse(
                                                  hypre_SStructUVEntryToIndex(Uventry, temp1[i]),
                                                  zero_index, stridef, index2);
                                  hypre_SubtractIndex(index2, index_temp, index1);
                                  MapStencilRank(index1, temp2[i]);

                                 /* zero off this stencil connection into the fbox */
                                  if (temp2[i] < max_stencil_size)
                                  {
                                     j= rank_stencils[temp2[i]];
                                     if (j > 0)
                                     {
                                        a_ptrs[j][iA]= 0.0;
                                     }
                                  }
                               }  /* for (i= 0; i< cnt1; i++) */

                               hypre_TFree(temp1);

                               HYPRE_IJMatrixGetValues(ij_A, cnt1, ncols, rows, cols, vals);
                               for (i= 0; i< cnt1; i++)
                               {
                                  a_ptrs[centre][iA]+= vals[i];
                               }

                               hypre_TFree(ncols);
                               hypre_TFree(rows);
                               hypre_TFree(cols);

                              /* compute the connection to the coarsened fine box */
                               for (i= 0; i< cnt1; i++)
                               {
                                  if (temp2[i] < max_stencil_size)
                                  {
                                      j= rank_stencils[temp2[i]];
                                      if (j > 0)
                                      {
                                         a_ptrs[j][iA]+= vals[i];
                                      }
                                  }
                               }
                               hypre_TFree(vals);
                               hypre_TFree(temp2);

                              /* centre connection which preserves the row sum */
                               for (i= 0; i< stencil_size; i++)
                               {
                                  if (i != centre)
                                  {
                                     a_ptrs[centre][iA]-= a_ptrs[i][iA];
                                  }
                               }

                           }   /* if (Uventry != NULL) */
                       }       /* if (nUventries > 0) */
                    }          /* hypre_BoxLoop1End(iA) */
                    hypre_BoxLoop1End(iA);
                 }  /* for (boxi= stencil_size; boxi< box_array_size; boxi++) */
              }     /* hypre_ForBoxArrayI(fi, cinterface_arrays) */
           }        /* hypre_ForBoxI(ci, cgrid_boxes) */

           hypre_TFree(a_ptrs);
           hypre_TFree(stencil_ranks);
           hypre_TFree(rank_stencils);
       }   /* if (stencils != NULL) */
    }      /* end var1 */


    for (var1= 0; var1< nvars; var1++)
    {
       cgrid= hypre_SStructPGridSGrid(hypre_SStructPMatrixPGrid(A_pmatrix), var1);
       cgrid_boxes= hypre_StructGridBoxes(cgrid);

       hypre_ForBoxI(ci, cgrid_boxes)
       {
          hypre_BoxArrayArrayDestroy(fgrid_cinterface_extents[var1][ci]);
       }
       hypre_TFree(fgrid_cinterface_extents[var1]);
    }
    hypre_TFree(fgrid_cinterface_extents);

    return 0;
}

