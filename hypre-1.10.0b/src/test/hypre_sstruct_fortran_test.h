/******************************************************************************
 *  Definitions of sstruct fortran interface routines
 *****************************************************************************/

#define HYPRE_SStructGraphCreate \
        hypre_F90_NAME(fhypre_sstructgraphcreate, FHYPRE_SSTRUCTGRAPHCREATE)
extern void hypre_F90_NAME(fhypre_sstructgraphcreate, FHYPRE_SSTRUCTGRAPHCREATE)
                          (int *, long int *, long int *);

#define HYPRE_SStructGraphDestroy \
        hypre_F90_NAME(fhypre_sstructgraphdestroy, FHYPRE_SSTRUCTGRAPHDESTROY)
extern void hypre_F90_NAME(fhypre_sstructgraphdestroy, FHYPRE_SSTRUCTGRAPHDESTROY)
                          (long int *);

#define HYPRE_SStructGraphSetStencil \
        hypre_F90_NAME(fhypre_sstructgraphsetstencil, FHYPRE_SSTRUCTGRAPHSETSTENCIL)
extern void hypre_F90_NAME(fhypre_sstructgraphsetstencil, FHYPRE_SSTRUCTGRAPHSETSTENCIL)
                          (long int *, int *, int *, long int *);

#define HYPRE_SStructGraphAddEntries \
        hypre_F90_NAME(fhypre_sstructgraphaddentries, FHYPRE_SSTRUCTGRAPHADDENTRIES)
extern void hypre_F90_NAME(fhypre_sstructgraphaddentries, FHYPRE_SSTRUCTGRAPHADDENTRIES)
                          (long int *, int *, int *, int *, int *, int *, int *);

#define HYPRE_SStructGraphAssemble \
        hypre_F90_NAME(fhypre_sstructgraphassemble, FHYPRE_SSTRUCTGRAPHASSEMBLE)
extern void hypre_F90_NAME(fhypre_sstructgraphassemble, FHYPRE_SSTRUCTGRAPHASSEMBLE)
                          (long int *);

#define HYPRE_SStructGraphSetObjectType \
        hypre_F90_NAME(fhypre_sstructgraphsetobjecttype, FHYPRE_SSTRUCTGRAPHSETOBJECTTYPE)

extern void hypre_F90_NAME(fhypre_sstructgraphsetobjecttype, FHYPRE_SSTRUCTGRAPHSETOBJECTTYPE)
                          (long int *, int *);



#define HYPRE_SStructGridCreate \
        hypre_F90_NAME(fhypre_sstructgridcreate, FHYPRE_SSTRUCTGRIDCREATE)
extern void hypre_F90_NAME(fhypre_sstructgridcreate, FHYPRE_SSTRUCTGRIDCREATE)
                          (int *, int *, int *, long int *);

#define HYPRE_SStructGridDestroy \
        hypre_F90_NAME(fhypre_sstructgriddestroy, FHYPRE_SSTRUCTGRIDDESTROY)
extern void hypre_F90_NAME(fhypre_sstructgriddestroy, FHYPRE_SSTRUCTGRIDDESTROY)
                          (long int *);

#define HYPRE_SStructGridSetExtents \
        hypre_F90_NAME(fhypre_sstructgridsetextents, FHYPRE_SSTRUCTGRIDSETEXTENTS)
extern void hypre_F90_NAME(fhypre_sstructgridsetextents, FHYPRE_SSTRUCTGRIDSETEXTENTS)
                          (long int *, int *, int *, int *);

#define HYPRE_SStructGridSetVariables \
        hypre_F90_NAME(fhypre_sstructgridsetvariables, FHYPRE_SSTRUCTGRIDSETVARIABLES)
extern void hypre_F90_NAME(fhypre_sstructgridsetvariables, FHYPRE_SSTRUCTGRIDSETVARIABLES)
                          (long int *, int *, int *, long int *);

#define HYPRE_SStructGridAddVariables \
        hypre_F90_NAME(fhypre_sstructgridaddvariables, FHYPRE_SSTRUCTGRIDADDVARIABLES)
extern void hypre_F90_NAME(fhypre_sstructgridaddvariables, FHYPRE_SSTRUCTGRIDADDVARIABLES)
                          (long int  *, int *, int *, int *, long int *);

#define HYPRE_SStructGridSetNeighborBox \
        hypre_F90_NAME(fhypre_sstructgridsetneighborbox, FHYPRE_SSTRUCTGRIDSETNEIGHBORBOX)
extern void hypre_F90_NAME(fhypre_sstructgridsetneighborbox, FHYPRE_SSTRUCTGRIDSETNEIGHBORBOX)
                          (long int *, int *, int *, int *, int *, int *, int *, int *);

#define HYPRE_SStructGridAddUnstructuredPart \
        hypre_F90_NAME(fhypre_sstructgridaddunstructure, FHYPRE_SSTRUCTGRIDADDUNSTRUCTURE)
extern void hypre_F90_NAME(fhypre_sstructgridaddunstructure, FHYPRE_SSTRUCTGRIDADDUNSTRUCTURE)
                          (long int *, int *, int *);

#define HYPRE_SStructGridAssemble \
        hypre_F90_NAME(fhypre_sstructgridassemble, FHYPRE_SSTRUCTGRIDASSEMBLE)
extern void hypre_F90_NAME(fhypre_sstructgridassemble, FHYPRE_SSTRUCTGRIDASSEMBLE)
                          (long int *);

#define HYPRE_SStructGridSetPeriodic \
        hypre_F90_NAME(fhypre_sstructgridsetperiodic, FHYPRE_SSTRUCTGRIDSETPERIODIC)
extern void hypre_F90_NAME(fhypre_sstructgridsetperiodic, FHYPRE_SSTRUCTGRIDSETPERIODIC)
                          (long int *, int *, int *);

#define HYPRE_SStructGridSetNumGhost \
        hypre_F90_NAME(fhypre_sstructgridsetnumghost, FHYPRE_SSTRUCTGRIDSETNUMGHOST)
extern void hypre_F90_NAME(fhypre_sstructgridsetnumghost, FHYPRE_SSTRUCTGRIDSETNUMGHOST)
                          (long int *, int *);



#define HYPRE_SStructMatrixCreate \
        hypre_F90_NAME(fhypre_sstructmatrixcreate, FHYPRE_SSTRUCTMATRIXCREATE)
extern void hypre_F90_NAME(fhypre_sstructmatrixcreate, FHYPRE_SSTRUCTMATRIXCREATE)
                          (int *, long int *, long int *);

#define HYPRE_SStructMatrixDestroy \
        hypre_F90_NAME(fhypre_sstructmatrixdestroy, FHYPRE_SSTRUCTMATRIXDESTROY)
extern void hypre_F90_NAME(fhypre_sstructmatrixdestroy, FHYPRE_SSTRUCTMATRIXDESTROY)
                          (long int *);

#define HYPRE_SStructMatrixInitialize \
        hypre_F90_NAME(fhypre_sstructmatrixinitialize, FHYPRE_SSTRUCTMATRIXINITIALIZE)
extern void hypre_F90_NAME(fhypre_sstructmatrixinitialize, FHYPRE_SSTRUCTMATRIXINITIALIZE)
                          (long int *);

#define HYPRE_SStructMatrixSetValues \
        hypre_F90_NAME(fhypre_sstructmatrixsetvalues, FHYPRE_SSTRUCTMATRIXSETVALUES)
extern void hypre_F90_NAME(fhypre_sstructmatrixsetvalues, FHYPRE_SSTRUCTMATRIXSETVALUES)
                          (long int *, int *, int *, int *, int *, int *, double *);

#define HYPRE_SStructMatrixSetBoxValues \
        hypre_F90_NAME(fhypre_sstructmatrixsetboxvalues, FHYPRE_SSTRUCTMATRIXSETBOXVALUES)
extern void hypre_F90_NAME(fhypre_sstructmatrixsetboxvalues, FHYPRE_SSTRUCTMATRIXSETBOXVALUES)
                          (long int *, int *, int *, int *, int *, int *, int *, double *);

#define HYPRE_SStructMatrixGetValues \
        hypre_F90_NAME(fhypre_sstructmatrixgetvalues, FHYPRE_SSTRUCTMATRIXGETVALUES
extern void hypre_F90_NAME(fhypre_sstructmatrixgetvalues, FHYPRE_SSTRUCTMATRIXGETVALUES)
                          (long int *, int *, int *, int *, int *, int *, double *);

#define HYPRE_SStructMatrixGetBoxValues \
        hypre_F90_NAME(fhypre_sstructmatrixgetboxvalues, FHYPRE_SSTRUCTMATRIXGETBOXVALUES)
extern void hypre_F90_NAME(fhypre_sstructmatrixgetboxvalues, FHYPRE_SSTRUCTMATRIXGETBOXVALUES)
                          (long int *, int *, int *, int *, int *, int *, int *, double *);

#define HYPRE_SStructMatrixAddToValues \
        hypre_F90_NAME(fhypre_sstructmatrixaddtovalues, FHYPRE_SSTRUCTMATRIXADDTOVALUES)
extern void hypre_F90_NAME(fhypre_sstructmatrixaddtovalues, FHYPRE_SSTRUCTMATRIXADDTOVALUES)
                          (long int *, int *, int *, int *, int *, int *, double *);

#define HYPRE_SStructMatrixAddToBoxValues \
        hypre_F90_NAME(fhypre_sstructmatrixaddtoboxvalue, FHYPRE_SSTRUCTMATRIXADDTOBOXVALUE)
extern void hypre_F90_NAME(fhypre_sstructmatrixaddtoboxvalue, FHYPRE_SSTRUCTMATRIXADDTOBOXVALUE)
                          (long int *, int *, int *, int *, int *, int *, int *, double *);

#define HYPRE_SStructMatrixAssemble \
        hypre_F90_NAME(fhypre_sstructmatrixassemble, FHYPRE_SSTRUCTMATRIXASSEMBLE)
extern void hypre_F90_NAME(fhypre_sstructmatrixassemble, FHYPRE_SSTRUCTMATRIXASSEMBLE)
                          (long int *);

#define HYPRE_SStructMatrixSetSymmetric \
        hypre_F90_NAME(fhypre_sstructmatrixsetsymmetric, FHYPRE_SSTRUCTMATRIXSETSYMMETRIC)
extern void hypre_F90_NAME(fhypre_sstructmatrixsetsymmetric, FHYPRE_SSTRUCTMATRIXSETSYMMETRIC)
                          (long int *, int *, int *, int *, int *);

#define HYPRE_SStructMatrixSetNSSymmetric \
        hypre_F90_NAME(fhypre_sstructmatrixsetnssymmetr, FHYPRE_SSTRUCTMATRIXSETNSSYMMETR)
extern void hypre_F90_NAME(fhypre_sstructmatrixsetnssymmetr, FHYPRE_SSTRUCTMATRIXSETNSSYMMETR)
                          (long int *, int *);

#define HYPRE_SStructMatrixSetObjectType \
        hypre_F90_NAME(fhypre_sstructmatrixsetobjecttyp, FHYPRE_SSTRUCTMATRIXSETOBJECTTYP)
extern void hypre_F90_NAME(fhypre_sstructmatrixsetobjecttyp, FHYPRE_SSTRUCTMATRIXSETOBJECTTYP)
                          (long int *, int *);

#define HYPRE_SStructMatrixGetObject \
        hypre_F90_NAME(fhypre_sstructmatrixgetobject, FHYPRE_SSTRUCTMATRIXGETOBJECT)
extern void hypre_F90_NAME(fhypre_sstructmatrixgetobject, FHYPRE_SSTRUCTMATRIXGETOBJECT)
                          (long int *, long int *);

#define HYPRE_SStructMatrixPrint \
        hypre_F90_NAME(fhypre_sstructmatrixprint, FHYPRE_SSTRUCTMATRIXPRINT)
extern void hypre_F90_NAME(fhypre_sstructmatrixprint, FHYPRE_SSTRUCTMATRIXPRINT)
                          (const char *, long int *, int *);



#define HYPRE_SStructStencilCreate \
        hypre_F90_NAME(fhypre_sstructstencilcreate, FHYPRE_SSTRUCTSTENCILCREATE)
extern void hypre_F90_NAME(fhypre_sstructstencilcreate, FHYPRE_SSTRUCTSTENCILCREATE)
                          (int *, int *, long int *);

#define HYPRE_SStructStencilDestroy \
        hypre_F90_NAME(fhypre_sstructstencildestroy, FHYPRE_SSTRUCTSTENCILDESTROY)
extern void hypre_F90_NAME(fhypre_sstructstencildestroy, FHYPRE_SSTRUCTSTENCILDESTROY)
                          (long int *);

#define HYPRE_SStructStencilSetEntry \
        hypre_F90_NAME(fhypre_sstructstencilsetentry, FHYPRE_SSTRUCTSTENCILSETENTRY)
extern void hypre_F90_NAME(fhypre_sstructstencilsetentry, FHYPRE_SSTRUCTSTENCILSETENTRY)
                          (long int *, int *, int *, int *);



#define HYPRE_SStructVectorCreate \
        hypre_F90_NAME(fhypre_sstructvectorcreate, FHYPRE_SSTRUCTVECTORCREATE)
extern void hypre_F90_NAME(fhypre_sstructvectorcreate, FHYPRE_SSTRUCTVECTORCREATE)
                          (int *, long int *, long int *);

#define HYPRE_SStructVectorDestroy \
        hypre_F90_NAME(fhypre_sstructvectordestroy, FHYPRE_SSTRUCTVECTORDESTROY)
extern void hypre_F90_NAME(fhypre_sstructvectordestroy, FHYPRE_SSTRUCTVECTORDESTROY)
                          (long int *);

#define HYPRE_SStructVectorInitialize \
        hypre_F90_NAME(fhypre_sstructvectorinitialize, FHYPRE_SSTRUCTVECTORINITIALIZE)
extern void hypre_F90_NAME(fhypre_sstructvectorinitialize, FHYPRE_SSTRUCTVECTORINITIALIZE)
                          (long int *);

#define HYPRE_SStructVectorSetValues \
        hypre_F90_NAME(fhypre_sstructvectorsetvalues, FHYPRE_SSTRUCTVECTORSETVALUES)
extern void hypre_F90_NAME(fhypre_sstructvectorsetvalues, FHYPRE_SSTRUCTVECTORSETVALUES)
                          (long int *, int *, int *, int *, double *);

#define HYPRE_SStructVectorSetBoxValues \
        hypre_F90_NAME(fhypre_sstructvectorsetboxvalues, FHYPRE_SSTRUCTVECTORSETBOXVALUES)
extern void hypre_F90_NAME(fhypre_sstructvectorsetboxvalues, FHYPRE_SSTRUCTVECTORSETBOXVALUES)
                          (long int *, int *, int *, int *, int *, double *);

#define HYPRE_SStructVectorAddToValues \
        hypre_F90_NAME(fhypre_sstructvectoraddtovalues, FHYPRE_SSTRUCTVECTORADDTOVALUES)
extern void hypre_F90_NAME(fhypre_sstructvectoraddtovalues, FHYPRE_SSTRUCTVECTORADDTOVALUES)
                          (long int *, int *, int *, int *, double *);

#define HYPRE_SStructVectorAddToBoxValues \
        hypre_F90_NAME(fhypre_sstructvectoraddtoboxvalu, FHYPRE_SSTRUCTVECTORADDTOBOXVALU)
extern void hypre_F90_NAME(fhypre_sstructvectoraddtoboxvalu, FHYPRE_SSTRUCTVECTORADDTOBOXVALU)
                          (long int *, int *, int *, int *, int *, double *);

#define HYPRE_SStructVectorAssemble \
        hypre_F90_NAME(fhypre_sstructvectorassemble, FHYPRE_SSTRUCTVECTORASSEMBLE)
extern void hypre_F90_NAME(fhypre_sstructvectorassemble, FHYPRE_SSTRUCTVECTORASSEMBLE)
                          (long int *);

#define HYPRE_SStructVectorGather \
        hypre_F90_NAME(fhypre_sstructvectorgather, FHYPRE_SSTRUCTVECTORGATHER)
extern void hypre_F90_NAME(fhypre_sstructvectorgather, FHYPRE_SSTRUCTVECTORGATHER)
                          (long int *);

#define HYPRE_SStructVectorGetValues \
        hypre_F90_NAME(fhypre_sstructvectorgetvalues, FHYPRE_SSTRUCTVECTORGETVALUES)
extern void hypre_F90_NAME(fhypre_sstructvectorgetvalues, FHYPRE_SSTRUCTVECTORGETVALUES)
                          (long int *, int *, int *, int *, double *);

#define HYPRE_SStructVectorGetBoxValues \
        hypre_F90_NAME(fhypre_sstructvectorgetboxvalues, FHYPRE_SSTRUCTVECTORGETBOXVALUES)
extern void hypre_F90_NAME(fhypre_sstructvectorgetboxvalues, FHYPRE_SSTRUCTVECTORGETBOXVALUES)
                          (long int *, int *, int *, int *, int *, double *);

#define HYPRE_SStructVectorSetObjectType \
        hypre_F90_NAME(fhypre_sstructvectorsetobjecttyp, FHYPRE_SSTRUCTVECTORSETOBJECTTYP)
extern void hypre_F90_NAME(fhypre_sstructvectorsetobjecttyp, FHYPRE_SSTRUCTVECTORSETOBJECTTYP)
                          (long int *, int *);

#define HYPRE_SStructVectorGetObject \
        hypre_F90_NAME(fhypre_sstructvectorgetobject, FHYPRE_SSTRUCTVECTORGETOBJECT)
extern void hypre_F90_NAME(fhypre_sstructvectorgetobject, FHYPRE_SSTRUCTVECTORGETOBJECT)
                          (long int *, void *);

#define HYPRE_SStructVectorPrint \
        hypre_F90_NAME(fhypre_sstructvectorprint, FHYPRE_SSTRUCTVECTORPRINT)
extern void hypre_F90_NAME(fhypre_sstructvectorprint, FHYPRE_SSTRUCTVECTORPRINT)
                          (const char *, long int *, int *);



#define HYPRE_SStructBiCGSTABCreate \
        hypre_F90_NAME(fhypre_sstructbicgstabcreate, FHYPRE_SSTRUCTBICGSTABCREATE)
extern void hypre_F90_NAME(fhypre_sstructbicgstabcreate, FHYPRE_SSTRUCTBICGSTABCREATE)
                          (int *, long int *);

#define HYPRE_SStructBiCGSTABDestroy \
        hypre_F90_NAME(fhypre_sstructbicgstabdestroy, FHYPRE_SSTRUCTBICGSTABDESTROY)
extern void hypre_F90_NAME(fhypre_sstructbicgstabdestroy, FHYPRE_SSTRUCTBICGSTABDESTROY)
                          (long int *);

#define HYPRE_SStructBiCGSTABSetup \
        hypre_F90_NAME(fhypre_sstructbicgstabsetup, FHYPRE_SSTRUCTBICGSTABSETUP)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetup, FHYPRE_SSTRUCTBICGSTABSETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructBiCGSTABSolve \
        hypre_F90_NAME(fhypre_sstructbicgstabsolve, FHYPRE_SSTRUCTBICGSTABSOLVE)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsolve, FHYPRE_SSTRUCTBICGSTABSOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructBiCGSTABSetTol \
        hypre_F90_NAME(fhypre_sstructbicgstabsettol, FHYPRE_SSTRUCTBICGSTABSETTOL)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsettol, FHYPRE_SSTRUCTBICGSTABSETTOL)
                          (long int *, double *);

#define HYPRE_SStructBiCGSTABSetMinIter \
        hypre_F90_NAME(fhypre_sstructbicgstabsetminiter, FHYPRE_SSTRUCTBICGSTABSETMINITER)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetminiter, FHYPRE_SSTRUCTBICGSTABSETMINITER)
                          (long int *, int *);

#define HYPRE_SStructBiCGSTABSetMaxIter \
        hypre_F90_NAME(fhypre_sstructbicgstabsetmaxiter, FHYPRE_SSTRUCTBICGSTABSETMAXITER)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetmaxiter, FHYPRE_SSTRUCTBICGSTABSETMAXITER)
                          (long int *, int *);

#define HYPRE_SStructBiCGSTABSetStopCrit \
        hypre_F90_NAME(fhypre_sstructbicgstabsetstopcri, FHYPRE_SSTRUCTBICGSTABSETSTOPCRI)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetstopcri, FHYPRE_SSTRUCTBICGSTABSETSTOPCRI)
                          (long int *, int *);

#define HYPRE_SStructBiCGSTABSetPrecond \
        hypre_F90_NAME(fhypre_sstructbicgstabsetprecond, FHYPRE_SSTRUCTBICGSTABSETPRECOND)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetprecond, FHYPRE_SSTRUCTBICGSTABSETPRECOND)
                          (long int *, int *, long int *);

#define HYPRE_SStructBiCGSTABSetLogging \
        hypre_F90_NAME(fhypre_sstructbicgstabsetlogging, FHYPRE_SSTRUCTBICGSTABSETLOGGING)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetlogging, FHYPRE_SSTRUCTBICGSTABSETLOGGING)
                          (long int *, int *);

#define HYPRE_SStructBiCGSTABSetPrintLevel \
        hypre_F90_NAME(fhypre_sstructbicgstabsetprintle, FHYPRE_SSTRUCTBICGSTABSETPRINTLE)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetprintle, FHYPRE_SSTRUCTBICGSTABSETPRINTLE)
                          (long int *, int *);

#define HYPRE_SStructBiCGSTABGetNumIterations \
        hypre_F90_NAME(fhypre_sstructbicgstabgetnumiter, FHYPRE_SSTRUCTBICGSTABGETNUMITER)
extern void hypre_F90_NAME(fhypre_sstructbicgstabgetnumiter, FHYPRE_SSTRUCTBICGSTABGETNUMITER)
                          (long int *, int *);

#define HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_sstructbicgstabgetfinalre, FHYPRE_SSTRUCTBICGSTABGETFINALRE)
extern void hypre_F90_NAME(fhypre_sstructbicgstabgetfinalre, FHYPRE_SSTRUCTBICGSTABGETFINALRE)
                          (long int *, double *);

#define HYPRE_SStructBiCGSTABGetResidual \
        hypre_F90_NAME(fhypre_sstructbicgstabgetresidua, FHYPRE_SSTRUCTBICGSTABGETRESIDUA)
extern void hypre_F90_NAME(fhypre_sstructbicgstabgetresidua, FHYPRE_SSTRUCTBICGSTABGETRESIDUA)
                          (long int *, long int *);



#define HYPRE_SStructGMRESCreate \
        hypre_F90_NAME(fhypre_sstructgmrescreate, FHYPRE_SSTRUCTGMRESCREATE)
extern void hypre_F90_NAME(fhypre_sstructgmrescreate, FHYPRE_SSTRUCTGMRESCREATE)
                          (long int *, long int *);

#define HYPRE_SStructGMRESDestroy \
        hypre_F90_NAME(fhypre_sstructgmresdestroy, FHYPRE_SSTRUCTGMRESDESTROY)
extern void hypre_F90_NAME(fhypre_sstructgmresdestroy, FHYPRE_SSTRUCTGMRESDESTROY)
                          (long int *);

#define HYPRE_SStructGMRESSetup \
        hypre_F90_NAME(fhypre_sstructgmressetup, FHYPRE_SSTRUCTGMRESSETUP)
extern void hypre_F90_NAME(fhypre_sstructgmressetup, FHYPRE_SSTRUCTGMRESSETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructGMRESSolve \
        hypre_F90_NAME(fhypre_sstructgmressolve, FHYPRE_SSTRUCTGMRESSOLVE)
extern void hypre_F90_NAME(fhypre_sstructgmressolve, FHYPRE_SSTRUCTGMRESSOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructGMRESSetKDim \
        hypre_F90_NAME(fhypre_sstructgmressetkdim, FHYPRE_SSTRUCTGMRESSETKDIM)
extern void hypre_F90_NAME(fhypre_sstructgmressetkdim, FHYPRE_SSTRUCTGMRESSETKDIM)
                          (long int *, int *);

#define HYPRE_SStructGMRESSetTol \
        hypre_F90_NAME(fhypre_sstructgmressettol, FHYPRE_SSTRUCTGMRESSETTOL)
extern void hypre_F90_NAME(fhypre_sstructgmressettol, FHYPRE_SSTRUCTGMRESSETTOL)
                          (long int *, double *);

#define HYPRE_SStructGMRESSetMinIter \
        hypre_F90_NAME(fhypre_sstructgmressetminiter, FHYPRE_SSTRUCTGMRESSETMINITER)
extern void hypre_F90_NAME(fhypre_sstructgmressetminiter, FHYPRE_SSTRUCTGMRESSETMINITER)
                          (long int *, int *);

#define HYPRE_SStructGMRESSetMaxIter \
        hypre_F90_NAME(fhypre_sstructgmressetmaxiter, FHYPRE_SSTRUCTGMRESSETMAXITER)
extern void hypre_F90_NAME(fhypre_sstructgmressetmaxiter, FHYPRE_SSTRUCTGMRESSETMAXITER)
                          (long int *, int *);

#define HYPRE_SStructGMRESSetStopCrit \
        hypre_F90_NAME(fhypre_sstructgmressetstopcrit, FHYPRE_SSTRUCTGMRESSETSTOPCRIT)
extern void hypre_F90_NAME(fhypre_sstructgmressetstopcrit, FHYPRE_SSTRUCTGMRESSETSTOPCRIT)
                          (long int *, int *);

#define HYPRE_SStructGMRESSetPrecond \
        hypre_F90_NAME(fhypre_sstructgmressetprecond, FHYPRE_SSTRUCTGMRESSETPRECOND)
extern void hypre_F90_NAME(fhypre_sstructgmressetprecond, FHYPRE_SSTRUCTGMRESSETPRECOND)
                          (long int *, int *, long int *);


#define HYPRE_SStructGMRESSetLogging \
        hypre_F90_NAME(fhypre_sstructgmressetlogging, FHYPRE_SSTRUCTGMRESSETLOGGING)
extern void hypre_F90_NAME(fhypre_sstructgmressetlogging, FHYPRE_SSTRUCTGMRESSETLOGGING)
                          (long int *, int *);

#define HYPRE_SStructGMRESSetPrintLevel \
        hypre_F90_NAME(fhypre_sstructgmressetprintlevel, FHYPRE_SSTRUCTGMRESSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_sstructgmressetprintlevel, FHYPRE_SSTRUCTGMRESSETPRINTLEVEL)
                          (long int *, int *);

#define HYPRE_SStructGMRESGetNumIterations \
      hypre_F90_NAME(fhypre_sstructgmresgetnumiterati, FHYPRE_SSTRUCTGMRESGETNUMITERATI)
extern void hypre_F90_NAME(fhypre_sstructgmresgetnumiterati, FHYPRE_SSTRUCTGMRESGETNUMITERATI)
                          (long int *, int *);

#define HYPRE_SStructGMRESGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_sstructgmresgetfinalrelat, FHYPRE_SSTRUCTGMRESGETFINALRELAT)
extern void hypre_F90_NAME(fhypre_sstructgmresgetfinalrelat, FHYPRE_SSTRUCTGMRESGETFINALRELAT)
                          (long int *, double  *);

#define HYPRE_SStructGMRESGetResidual \
        hypre_F90_NAME(fhypre_sstructgmresgetresidual, FHYPRE_SSTRUCTGMRESGETRESIDUAL)
extern void hypre_F90_NAME(fhypre_sstructgmresgetresidual, FHYPRE_SSTRUCTGMRESGETRESIDUAL)
                          (long int *, long int *);



#define HYPRE_SStructPCGCreate \
        hypre_F90_NAME(fhypre_sstructpcgcreate, FHYPRE_SSTRUCTPCGCREATE)
extern void hypre_F90_NAME(fhypre_sstructpcgcreate, FHYPRE_SSTRUCTPCGCREATE)
                          (long int *, long int *);

#define HYPRE_SStructPCGDestroy \
        hypre_F90_NAME(fhypre_sstructpcgdestroy, FHYPRE_SSTRUCTPCGDESTROY)
extern void hypre_F90_NAME(fhypre_sstructpcgdestroy, FHYPRE_SSTRUCTPCGDESTROY)
                          (long int *);

#define HYPRE_SStructPCGSetup \
        hypre_F90_NAME(fhypre_sstructpcgsetup, FHYPRE_SSTRUCTPCGDESTROY)
extern void hypre_F90_NAME(fhypre_sstructpcgsetup, FHYPRE_SSTRUCTPCGDESTROY)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructPCGSolve \
        hypre_F90_NAME(fhypre_sstructpcgsolve, FHYPRE_SSTRUCTPCGSOLVE)
extern void hypre_F90_NAME(fhypre_sstructpcgsolve, FHYPRE_SSTRUCTPCGSOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructPCGSetTol \
        hypre_F90_NAME(fhypre_sstructpcgsettol, FHYPRE_SSTRUCTPCGSETTOL)
extern void hypre_F90_NAME(fhypre_sstructpcgsettol, FHYPRE_SSTRUCTPCGSETTOL)
                          (long int *, double *);

#define HYPRE_SStructPCGSetMaxIter \
        hypre_F90_NAME(fhypre_sstructpcgsetmaxiter, FHYPRE_SSTRUCTPCGSETMAXITER)
extern void hypre_F90_NAME(fhypre_sstructpcgsetmaxiter, FHYPRE_SSTRUCTPCGSETMAXITER)
                          (long int *, int  *);

#define HYPRE_SStructPCGSetTwoNorm \
        hypre_F90_NAME(fhypre_sstructpcgsettwonorm, FHYPRE_SSTRUCTPCGSETTWONORM)
extern void hypre_F90_NAME(fhypre_sstructpcgsettwonorm, FHYPRE_SSTRUCTPCGSETTWONORM)
                          (long int *, int  *);

#define HYPRE_SStructPCGSetRelChange \
        hypre_F90_NAME(fhypre_sstructpcgsetrelchange, FHYPRE_SSTRUCTPCGSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_sstructpcgsetrelchange, FHYPRE_SSTRUCTPCGSETRELCHANGE)
                          (long int *, int  *);

#define HYPRE_SStructPCGSetPrecond \
        hypre_F90_NAME(fhypre_sstructpcgsetprecond, FHYPRE_SSTRUCTPCGSETPRECOND)
extern void hypre_F90_NAME(fhypre_sstructpcgsetprecond, FHYPRE_SSTRUCTPCGSETPRECOND)
                          (long int *, int  *, long int *);


#define HYPRE_SStructPCGSetLogging \
        hypre_F90_NAME(fhypre_sstructpcgsetlogging, FHYPRE_SSTRUCTPCGSETLOGGING)
extern void hypre_F90_NAME(fhypre_sstructpcgsetlogging, FHYPRE_SSTRUCTPCGSETLOGGING)
                          (long int *, int  *);

#define HYPRE_SStructPCGSetPrintLevel \
        hypre_F90_NAME(fhypre_sstructpcgsetprintlevel, FHYPRE_SSTRUCTPCGSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_sstructpcgsetprintlevel, FHYPRE_SSTRUCTPCGSETPRINTLEVEL)
                          (long int *, int  *);

#define HYPRE_SStructPCGGetNumIterations \
        hypre_F90_NAME(fhypre_sstructpcggetnumiteration, FHYPRE_SSTRUCTPCGGETNUMITERATION)
extern void hypre_F90_NAME(fhypre_sstructpcggetnumiteration, FHYPRE_SSTRUCTPCGGETNUMITERATION)
                          (long int *, int  *);

#define HYPRE_SStructPCGGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_sstructpcggetfinalrelativ, FHYPRE_SSTRUCTPCGGETFINALRELATIV)
extern void hypre_F90_NAME(fhypre_sstructpcggetfinalrelativ, FHYPRE_SSTRUCTPCGGETFINALRELATIV)
                          (long int *, double *);

#define HYPRE_SStructPCGGetResidual \
        hypre_F90_NAME(fhypre_sstructpcggetresidual, FHYPRE_SSTRUCTPCGGETRESIDUAL)
extern void hypre_F90_NAME(fhypre_sstructpcggetresidual, FHYPRE_SSTRUCTPCGGETRESIDUAL)
                          (long int *, long int *);

#define HYPRE_SStructDiagScaleSetup \
        hypre_F90_NAME(fhypre_sstructdiagscalesetup, FHYPRE_SSTRUCTDIAGSCALESETUP)
extern void hypre_F90_NAME(fhypre_sstructdiagscalesetup, FHYPRE_SSTRUCTDIAGSCALESETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructDiagScale \
        hypre_F90_NAME(fhypre_sstructdiagscale, FHYPRE_SSTRUCTDIAGSCALE)
extern void hypre_F90_NAME(fhypre_sstructdiagscale, FHYPRE_SSTRUCTDIAGSCALE)
                          (long int *, long int *, long int *, long int *);


#define HYPRE_SStructSplitCreate \
        hypre_F90_NAME(fhypre_sstructsplitcreate, FHYPRE_SSTRUCTSPLITCREATE)
extern void hypre_F90_NAME(fhypre_sstructsplitcreate, FHYPRE_SSTRUCTSPLITCREATE)
                          (long int *, long int *);

#define HYPRE_SStructSplitDestroy \
        hypre_F90_NAME(fhypre_sstructsplitdestroy, FHYPRE_SSTRUCTSPLITDESTROY)
extern void hypre_F90_NAME(fhypre_sstructsplitdestroy, FHYPRE_SSTRUCTSPLITDESTROY)
                          (long int *);

#define HYPRE_SStructSplitSetup \
        hypre_F90_NAME(fhypre_sstructsplitsetup, FHYPRE_SSTRUCTSPLITSETUP)
extern void hypre_F90_NAME(fhypre_sstructsplitsetup, FHYPRE_SSTRUCTSPLITSETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructSplitSolve \
        hypre_F90_NAME(fhypre_sstructsplitsolve, FHYPRE_SSTRUCTSPLITSOLVE)
extern void hypre_F90_NAME(fhypre_sstructsplitsolve, FHYPRE_SSTRUCTSPLITSOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructSplitSetTol \
        hypre_F90_NAME(fhypre_sstructsplitsettol, FHYPRE_SSTRUCTSPLITSETTOL)
extern void hypre_F90_NAME(fhypre_sstructsplitsettol, FHYPRE_SSTRUCTSPLITSETTOL)
                          (long int *, double *);

#define HYPRE_SStructSplitSetMaxIter \
        hypre_F90_NAME(fhypre_sstructsplitsetmaxiter, FHYPRE_SSTRUCTSPLITSETMAXITER)
extern void hypre_F90_NAME(fhypre_sstructsplitsetmaxiter, FHYPRE_SSTRUCTSPLITSETMAXITER)
                          (long int *, int  *);

#define HYPRE_SStructSplitSetZeroGuess \
        hypre_F90_NAME(fhypre_sstructsplitsetzeroguess, FHYPRE_SSTRUCTSPLITSETZEROGUESS)
extern void hypre_F90_NAME(fhypre_sstructsplitsetzeroguess, FHYPRE_SSTRUCTSPLITSETZEROGUESS)
                          (long int *);

#define HYPRE_SStructSplitSetNonZeroGuess \
        hypre_F90_NAME(fhypre_sstructsplitsetnonzerogue, FHYPRE_SSTRUCTSPLITSETNONZEROGUE)
extern void hypre_F90_NAME(fhypre_sstructsplitsetnonzerogue, FHYPRE_SSTRUCTSPLITSETNONZEROGUE)
                          (long int *);

#define HYPRE_SStructSplitSetStructSolver \
        hypre_F90_NAME(fhypre_sstructsplitsetstructsolv, FHYPRE_SSTRUCTSPLITSETSTRUCTSOLV)
extern void hypre_F90_NAME(fhypre_sstructsplitsetstructsolv, FHYPRE_SSTRUCTSPLITSETSTRUCTSOLV)
                          (long int *, int  *);

#define HYPRE_SStructSplitGetNumIterations \
        hypre_F90_NAME(fhypre_sstructsplitgetnumiterati, FHYPRE_SSTRUCTSPLITGETNUMITERATI)
extern void hypre_F90_NAME(fhypre_sstructsplitgetnumiterati, FHYPRE_SSTRUCTSPLITGETNUMITERATI)
                          (long int *, int  *);

#define HYPRE_SStructSplitGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_sstructsplitgetfinalrelat, FHYPRE_SSTRUCTSPLITGETFINALRELAT)
extern void hypre_F90_NAME(fhypre_sstructsplitgetfinalrelat, FHYPRE_SSTRUCTSPLITGETFINALRELAT)
                          (long int *, double *);



#define HYPRE_SStructSysPFMGCreate \
        hypre_F90_NAME(fhypre_sstructsyspfmgcreate, FHYPRE_SSTRUCTSYSPFMGCREATE)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgcreate, FHYPRE_SSTRUCTSYSPFMGCREATE)
                          (long int *, long int *);

#define HYPRE_SStructSysPFMGDestroy \
        hypre_F90_NAME(fhypre_sstructsyspfmgdestroy, FHYPRE_SSTRUCTSYSPFMGDESTROY)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgdestroy, FHYPRE_SSTRUCTSYSPFMGDESTROY)
                          (long int *);

#define HYPRE_SStructSysPFMGSetup \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetup, FHYPRE_SSTRUCTSYSPFMGSETUP)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetup, FHYPRE_SSTRUCTSYSPFMGSETUP)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructSysPFMGSolve \
        hypre_F90_NAME(fhypre_sstructsyspfmgsolve, FHYPRE_SSTRUCTSYSPFMGSOLVE)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsolve, FHYPRE_SSTRUCTSYSPFMGSOLVE)
                          (long int *, long int *, long int *, long int *);

#define HYPRE_SStructSysPFMGSetTol \
        hypre_F90_NAME(fhypre_sstructsyspfmgsettol, FHYPRE_SSTRUCTSYSPFMGSETTOL)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsettol, FHYPRE_SSTRUCTSYSPFMGSETTOL)
                          (long int *, double *);

#define HYPRE_SStructSysPFMGSetMaxIter \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetmaxiter, FHYPRE_SSTRUCTSYSPFMGSETMAXITER)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetmaxiter, FHYPRE_SSTRUCTSYSPFMGSETMAXITER)
                          (long int *, int *);

#define HYPRE_SStructSysPFMGSetRelChange \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetrelchang, FHYPRE_SSTRUCTSYSPFMGSETRELCHANG)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetrelchang, FHYPRE_SSTRUCTSYSPFMGSETRELCHANG)
                          (long int *, int *);

#define HYPRE_SStructSysPFMGSetZeroGuess \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetzerogues, FHYPRE_SSTRUCTSYSPFMGSETZEROGUES)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetzerogues, FHYPRE_SSTRUCTSYSPFMGSETZEROGUES)
                          (long int *);

#define HYPRE_SStructSysPFMGSetNonZeroGuess \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetnonzerog, FHYPRE_SSTRUCTSYSPFMGSETNONZEROG)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetnonzerog, FHYPRE_SSTRUCTSYSPFMGSETNONZEROG)
                          (long int *);

#define HYPRE_SStructSysPFMGSetRelaxType \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetrelaxtyp, FHYPRE_SSTRUCTSYSPFMGSETRELAXTYP)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetrelaxtyp, FHYPRE_SSTRUCTSYSPFMGSETRELAXTYP)
                          (long int *, int *);

#define HYPRE_SStructSysPFMGSetNumPreRelax \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetnumprere, FHYPRE_SSTRUCTSYSPFMGSETNUMPRERE)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetnumprere, FHYPRE_SSTRUCTSYSPFMGSETNUMPRERE)
                          (long int *, int *);

#define HYPRE_SStructSysPFMGSetNumPostRelax \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetnumpostr, FHYPRE_SSTRUCTSYSPFMGSETNUMPOSTR)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetnumpostr, FHYPRE_SSTRUCTSYSPFMGSETNUMPOSTR)
                          (long int *, int *);


#define HYPRE_SStructSysPFMGSetSkipRelax \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetskiprela, FHYPRE_SSTRUCTSYSPFMGSETSKIPRELA)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetskiprela, FHYPRE_SSTRUCTSYSPFMGSETSKIPRELA)
                          (long int *, int *);

#define HYPRE_SStructSysPFMGSetDxyz \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetdxyz, FHYPRE_SSTRUCTSYSPFMGSETDXYZ)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetdxyz, FHYPRE_SSTRUCTSYSPFMGSETDXYZ)
                          (long int *, double *);

#define HYPRE_SStructSysPFMGSetLogging \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetlogging, FHYPRE_SSTRUCTSYSPFMGSETLOGGING)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetlogging, FHYPRE_SSTRUCTSYSPFMGSETLOGGING)
                          (long int *, int *);

#define HYPRE_SStructSysPFMGSetPrintLevel \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetprintlev, FHYPRE_SSTRUCTSYSPFMGSETPRINTLEV)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetprintlev, FHYPRE_SSTRUCTSYSPFMGSETPRINTLEV)
                          (long int *, int *);

#define HYPRE_SStructSysPFMGGetNumIterations \
        hypre_F90_NAME(fhypre_sstructsyspfmggetnumitera, FHYPRE_SSTRUCTSYSPFMGGETNUMITERA)
extern void hypre_F90_NAME(fhypre_sstructsyspfmggetnumitera, FHYPRE_SSTRUCTSYSPFMGGETNUMITERA)
                          (long int *, int *);


#define HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_sstructsyspfmggetfinalrel, FHYPRE_SSTRUCTSYSPFMGGETFINALREL)
extern void hypre_F90_NAME(fhypre_sstructsyspfmggetfinalrel, FHYPRE_SSTRUCTSYSPFMGGETFINALREL)
                          (long int *, double *);
