
#include <HYPRE_config.h>

#include "HYPRE_sstruct_ls.h"

#ifndef hypre_SSTRUCT_LS_HEADER
#define hypre_SSTRUCT_LS_HEADER

#include "utilities.h"
#include "krylov.h"
#include "struct_ls.h"
#include "sstruct_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * hypre_SStructOwnInfo data structure
 * This structure is for the coarsen fboxes that are on this processor,
 * and the cboxes of cgrid/(all coarsened fboxes) on this processor (i.e.,
 * the coarse boxes of the composite cgrid (no underlying) on this processor).
 *--------------------------------------------------------------------------*/
#ifndef hypre_OWNINFODATA_HEADER
#define hypre_OWNINFODATA_HEADER


typedef struct 
{
   int                   size;

   hypre_BoxArrayArray  *own_boxes;    /* size of fgrid */
   int                 **own_cboxnums; /* local cbox number- each fbox
                                          leads to an array of cboxes */

   hypre_BoxArrayArray  *own_composite_cboxes;  /* size of cgrid */
   int                   own_composite_size;
} hypre_SStructOwnInfoData;


/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructOwnInfoData;
 *--------------------------------------------------------------------------*/

#define hypre_SStructOwnInfoDataSize(own_data)       ((own_data) -> size)
#define hypre_SStructOwnInfoDataOwnBoxes(own_data)   ((own_data) -> own_boxes)
#define hypre_SStructOwnInfoDataOwnBoxNums(own_data) \
((own_data) -> own_cboxnums)
#define hypre_SStructOwnInfoDataCompositeCBoxes(own_data) \
((own_data) -> own_composite_cboxes)
#define hypre_SStructOwnInfoDataCompositeSize(own_data) \
((own_data) -> own_composite_size)

#endif
/*--------------------------------------------------------------------------
 * hypre_SStructRecvInfo data structure
 *--------------------------------------------------------------------------*/
#ifndef hypre_RECVINFODATA_HEADER
#define hypre_RECVINFODATA_HEADER


typedef struct 
{
   int                   size;

   hypre_BoxArrayArray  *recv_boxes;
   int                 **recv_procs;

} hypre_SStructRecvInfoData;

#endif
/*--------------------------------------------------------------------------
 * hypre_SStructSendInfo data structure
 *--------------------------------------------------------------------------*/
#ifndef hypre_SENDINFODATA_HEADER
#define hypre_SENDINFODATA_HEADER


typedef struct 
{
   int                   size;

   hypre_BoxArrayArray  *send_boxes;
   int                 **send_procs;
   int                 **send_remote_boxnums;

} hypre_SStructSendInfoData;

#endif

/* fac_amr_fcoarsen.c */
int hypre_AMR_FCoarsen( hypre_SStructMatrix *A , hypre_SStructMatrix *fac_A , hypre_SStructPMatrix *A_crse , hypre_Index refine_factors , int level );

/* fac_amr_rap.c */
int hypre_AMR_RAP( hypre_SStructMatrix *A , hypre_Index *rfactors , hypre_SStructMatrix **fac_A_ptr );

/* fac_amr_zero_data.c */
int hypre_ZeroAMRVectorData( hypre_SStructVector *b , int *plevels , hypre_Index *rfactors );
int hypre_ZeroAMRMatrixData( hypre_SStructMatrix *A , int part_crse , hypre_Index rfactors );

/* fac.c */
void *hypre_FACCreate( MPI_Comm comm );
int hypre_FACDestroy2( void *fac_vdata );
int hypre_FACSetTol( void *fac_vdata , double tol );
int hypre_FACSetPLevels( void *fac_vdata , int nparts , int *plevels );
int hypre_FACSetPRefinements( void *fac_vdata , int nparts , int (*prefinements )[3 ]);
int hypre_FACSetMaxLevels( void *fac_vdata , int nparts );
int hypre_FACSetMaxIter( void *fac_vdata , int max_iter );
int hypre_FACSetRelChange( void *fac_vdata , int rel_change );
int hypre_FACSetZeroGuess( void *fac_vdata , int zero_guess );
int hypre_FACSetRelaxType( void *fac_vdata , int relax_type );
int hypre_FACSetNumPreSmooth( void *fac_vdata , int num_pre_smooth );
int hypre_FACSetNumPostSmooth( void *fac_vdata , int num_post_smooth );
int hypre_FACSetCoarseSolverType( void *fac_vdata , int csolver_type );
int hypre_FACSetLogging( void *fac_vdata , int logging );
int hypre_FACGetNumIterations( void *fac_vdata , int *num_iterations );
int hypre_FACPrintLogging( void *fac_vdata , int myid );
int hypre_FACGetFinalRelativeResidualNorm( void *fac_vdata , double *relative_residual_norm );

/* fac_cf_coarsen.c */
int hypre_AMR_CFCoarsen( hypre_SStructMatrix *A , hypre_SStructMatrix *fac_A , hypre_Index refine_factors , int level );

/* fac_CFInterfaceExtents.c */
hypre_BoxArray *hypre_CFInterfaceExtents( hypre_Box *fgrid_box , hypre_Box *cgrid_box , hypre_StructStencil *stencils , hypre_Index rfactors );
int hypre_CFInterfaceExtents2( hypre_Box *fgrid_box , hypre_Box *cgrid_box , hypre_StructStencil *stencils , hypre_Index rfactors , hypre_BoxArray *cf_interface );

/* fac_cfstencil_box.c */
hypre_Box *hypre_CF_StenBox( hypre_Box *fgrid_box , hypre_Box *cgrid_box , hypre_Index stencil_shape , hypre_Index rfactors );

/* fac_interp2.c */
int hypre_FacSemiInterpCreate2( void **fac_interp_vdata_ptr );
int hypre_FacSemiInterpDestroy2( void *fac_interp_vdata );
int hypre_FacSemiInterpSetup2( void *fac_interp_vdata , hypre_SStructVector *e , hypre_SStructPVector *ec , hypre_Index rfactors );
int hypre_FAC_IdentityInterp2( void *fac_interp_vdata , hypre_SStructPVector *xc , hypre_SStructVector *e );
int hypre_FAC_WeightedInterp2( void *fac_interp_vdata , hypre_SStructPVector *xc , hypre_SStructVector *e_parts );

/* fac_relax.c */
int hypre_FacLocalRelax( void *relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *x , hypre_SStructPVector *b , int num_relax , int *zero_guess );

/* fac_restrict2.c */
int hypre_FacSemiRestrictCreate2( void **fac_restrict_vdata_ptr );
int hypre_FacSemiRestrictSetup2( void *fac_restrict_vdata , hypre_SStructVector *r , int part_crse , int part_fine , hypre_SStructPVector *rc , hypre_Index rfactors );
int hypre_FACRestrict2( void *fac_restrict_vdata , hypre_SStructVector *xf , hypre_SStructPVector *xc );
int hypre_FacSemiRestrictDestroy2( void *fac_restrict_vdata );

/* fac_setup2.c */
int hypre_FacSetup2( void *fac_vdata , hypre_SStructMatrix *A , hypre_SStructVector *b , hypre_SStructVector *x );

/* fac_solve3.c */
int hypre_FACSolve3( void *fac_vdata , hypre_SStructMatrix *A_in , hypre_SStructVector *b_in , hypre_SStructVector *x_in );

/* fac_zero_cdata.c */
int hypre_FacZeroCData( void *fac_vdata , hypre_SStructMatrix *A , hypre_SStructVector *b , hypre_SStructVector *x );

/* fac_zero_stencilcoef.c */
int hypre_FacZeroCFSten( hypre_SStructPMatrix *Af , hypre_SStructPMatrix *Ac , hypre_SStructGrid *grid , int fine_part , hypre_Index rfactors );
int hypre_FacZeroFCSten( hypre_SStructPMatrix *A , hypre_SStructGrid *grid , int fine_part );

/* hypre_bsearch.c */
int hypre_LowerBinarySearch( int *list , int value , int list_length );
int hypre_UpperBinarySearch( int *list , int value , int list_length );

/* HYPRE_sstruct_bicgstab.c */
int HYPRE_SStructBiCGSTABCreate( MPI_Comm comm , HYPRE_SStructSolver *solver );
int HYPRE_SStructBiCGSTABDestroy( HYPRE_SStructSolver solver );
int HYPRE_SStructBiCGSTABSetup( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructBiCGSTABSolve( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructBiCGSTABSetTol( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructBiCGSTABSetMinIter( HYPRE_SStructSolver solver , int min_iter );
int HYPRE_SStructBiCGSTABSetMaxIter( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructBiCGSTABSetStopCrit( HYPRE_SStructSolver solver , int stop_crit );
int HYPRE_SStructBiCGSTABSetPrecond( HYPRE_SStructSolver solver , HYPRE_PtrToSStructSolverFcn precond , HYPRE_PtrToSStructSolverFcn precond_setup , void *precond_data );
int HYPRE_SStructBiCGSTABSetLogging( HYPRE_SStructSolver solver , int logging );
int HYPRE_SStructBiCGSTABSetPrintLevel( HYPRE_SStructSolver solver , int print_level );
int HYPRE_SStructBiCGSTABGetNumIterations( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm( HYPRE_SStructSolver solver , double *norm );
int HYPRE_SStructBiCGSTABGetResidual( HYPRE_SStructSolver solver , void **residual );

/* HYPRE_sstruct_gmres.c */
int HYPRE_SStructGMRESCreate( MPI_Comm comm , HYPRE_SStructSolver *solver );
int HYPRE_SStructGMRESDestroy( HYPRE_SStructSolver solver );
int HYPRE_SStructGMRESSetup( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructGMRESSolve( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructGMRESSetKDim( HYPRE_SStructSolver solver , int k_dim );
int HYPRE_SStructGMRESSetTol( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructGMRESSetMinIter( HYPRE_SStructSolver solver , int min_iter );
int HYPRE_SStructGMRESSetMaxIter( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructGMRESSetStopCrit( HYPRE_SStructSolver solver , int stop_crit );
int HYPRE_SStructGMRESSetPrecond( HYPRE_SStructSolver solver , HYPRE_PtrToSStructSolverFcn precond , HYPRE_PtrToSStructSolverFcn precond_setup , void *precond_data );
int HYPRE_SStructGMRESSetLogging( HYPRE_SStructSolver solver , int logging );
int HYPRE_SStructGMRESSetPrintLevel( HYPRE_SStructSolver solver , int level );
int HYPRE_SStructGMRESGetNumIterations( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructGMRESGetFinalRelativeResidualNorm( HYPRE_SStructSolver solver , double *norm );
int HYPRE_SStructGMRESGetResidual( HYPRE_SStructSolver solver , void **residual );

/* HYPRE_sstruct_InterFAC.c */
int HYPRE_SStructFACCreate( MPI_Comm comm , HYPRE_SStructSolver *solver );
int HYPRE_SStructFACDestroy2( HYPRE_SStructSolver solver );
int HYPRE_SStructFACSetup2( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructFACSolve3( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructFACSetTol( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructFACSetPLevels( HYPRE_SStructSolver solver , int nparts , int *plevels );
int HYPRE_SStructFACSetPRefinements( HYPRE_SStructSolver solver , int nparts , int (*rfactors )[3 ]);
int HYPRE_SStructFACSetMaxLevels( HYPRE_SStructSolver solver , int max_levels );
int HYPRE_SStructFACSetMaxIter( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructFACSetRelChange( HYPRE_SStructSolver solver , int rel_change );
int HYPRE_SStructFACSetZeroGuess( HYPRE_SStructSolver solver );
int HYPRE_SStructFACSetNonZeroGuess( HYPRE_SStructSolver solver );
int HYPRE_SStructFACSetRelaxType( HYPRE_SStructSolver solver , int relax_type );
int HYPRE_SStructFACSetNumPreRelax( HYPRE_SStructSolver solver , int num_pre_relax );
int HYPRE_SStructFACSetNumPostRelax( HYPRE_SStructSolver solver , int num_post_relax );
int HYPRE_SStructFACSetCoarseSolverType( HYPRE_SStructSolver solver , int csolver_type );
int HYPRE_SStructFACSetLogging( HYPRE_SStructSolver solver , int logging );
int HYPRE_SStructFACGetNumIterations( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructFACGetFinalRelativeResidualNorm( HYPRE_SStructSolver solver , double *norm );

/* HYPRE_sstruct_pcg.c */
int HYPRE_SStructPCGCreate( MPI_Comm comm , HYPRE_SStructSolver *solver );
int HYPRE_SStructPCGDestroy( HYPRE_SStructSolver solver );
int HYPRE_SStructPCGSetup( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructPCGSolve( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructPCGSetTol( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructPCGSetMaxIter( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructPCGSetTwoNorm( HYPRE_SStructSolver solver , int two_norm );
int HYPRE_SStructPCGSetRelChange( HYPRE_SStructSolver solver , int rel_change );
int HYPRE_SStructPCGSetPrecond( HYPRE_SStructSolver solver , HYPRE_PtrToSStructSolverFcn precond , HYPRE_PtrToSStructSolverFcn precond_setup , void *precond_data );
int HYPRE_SStructPCGSetLogging( HYPRE_SStructSolver solver , int logging );
int HYPRE_SStructPCGSetPrintLevel( HYPRE_SStructSolver solver , int level );
int HYPRE_SStructPCGGetNumIterations( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructPCGGetFinalRelativeResidualNorm( HYPRE_SStructSolver solver , double *norm );
int HYPRE_SStructPCGGetResidual( HYPRE_SStructSolver solver , void **residual );
int HYPRE_SStructDiagScaleSetup( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector y , HYPRE_SStructVector x );
int HYPRE_SStructDiagScale( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector y , HYPRE_SStructVector x );

/* HYPRE_sstruct_split.c */
int HYPRE_SStructSplitCreate( MPI_Comm comm , HYPRE_SStructSolver *solver_ptr );
int HYPRE_SStructSplitDestroy( HYPRE_SStructSolver solver );
int HYPRE_SStructSplitSetup( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructSplitSolve( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructSplitSetTol( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructSplitSetMaxIter( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructSplitSetZeroGuess( HYPRE_SStructSolver solver );
int HYPRE_SStructSplitSetNonZeroGuess( HYPRE_SStructSolver solver );
int HYPRE_SStructSplitSetStructSolver( HYPRE_SStructSolver solver , int ssolver );
int HYPRE_SStructSplitGetNumIterations( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructSplitGetFinalRelativeResidualNorm( HYPRE_SStructSolver solver , double *norm );

/* HYPRE_sstruct_sys_pfmg.c */
int HYPRE_SStructSysPFMGCreate( MPI_Comm comm , HYPRE_SStructSolver *solver );
int HYPRE_SStructSysPFMGDestroy( HYPRE_SStructSolver solver );
int HYPRE_SStructSysPFMGSetup( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructSysPFMGSolve( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructSysPFMGSetTol( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructSysPFMGSetMaxIter( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructSysPFMGSetRelChange( HYPRE_SStructSolver solver , int rel_change );
int HYPRE_SStructSysPFMGSetZeroGuess( HYPRE_SStructSolver solver );
int HYPRE_SStructSysPFMGSetNonZeroGuess( HYPRE_SStructSolver solver );
int HYPRE_SStructSysPFMGSetRelaxType( HYPRE_SStructSolver solver , int relax_type );
int HYPRE_SStructSysPFMGSetNumPreRelax( HYPRE_SStructSolver solver , int num_pre_relax );
int HYPRE_SStructSysPFMGSetNumPostRelax( HYPRE_SStructSolver solver , int num_post_relax );
int HYPRE_SStructSysPFMGSetSkipRelax( HYPRE_SStructSolver solver , int skip_relax );
int HYPRE_SStructSysPFMGSetDxyz( HYPRE_SStructSolver solver , double *dxyz );
int HYPRE_SStructSysPFMGSetLogging( HYPRE_SStructSolver solver , int logging );
int HYPRE_SStructSysPFMGSetPrintLevel( HYPRE_SStructSolver solver , int print_level );
int HYPRE_SStructSysPFMGGetNumIterations( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm( HYPRE_SStructSolver solver , double *norm );

/* krylov.c */
int hypre_SStructKrylovIdentitySetup( void *vdata , void *A , void *b , void *x );
int hypre_SStructKrylovIdentity( void *vdata , void *A , void *b , void *x );

/* krylov_sstruct.c */
char *hypre_SStructKrylovCAlloc( int count , int elt_size );
int hypre_SStructKrylovFree( char *ptr );
void *hypre_SStructKrylovCreateVector( void *vvector );
void *hypre_SStructKrylovCreateVectorArray( int n , void *vvector );
int hypre_SStructKrylovDestroyVector( void *vvector );
void *hypre_SStructKrylovMatvecCreate( void *A , void *x );
int hypre_SStructKrylovMatvec( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_SStructKrylovMatvecDestroy( void *matvec_data );
double hypre_SStructKrylovInnerProd( void *x , void *y );
int hypre_SStructKrylovCopyVector( void *x , void *y );
int hypre_SStructKrylovClearVector( void *x );
int hypre_SStructKrylovScaleVector( double alpha , void *x );
int hypre_SStructKrylovAxpy( double alpha , void *x , void *y );
int hypre_SStructKrylovCommInfo( void *A , int *my_id , int *num_procs );

/* node_relax.c */
void *hypre_NodeRelaxCreate( MPI_Comm comm );
int hypre_NodeRelaxDestroy( void *relax_vdata );
int hypre_NodeRelaxSetup( void *relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *b , hypre_SStructPVector *x );
int hypre_NodeRelax( void *relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *b , hypre_SStructPVector *x );
int hypre_NodeRelaxSetTol( void *relax_vdata , double tol );
int hypre_NodeRelaxSetMaxIter( void *relax_vdata , int max_iter );
int hypre_NodeRelaxSetZeroGuess( void *relax_vdata , int zero_guess );
int hypre_NodeRelaxSetWeight( void *relax_vdata , double weight );
int hypre_NodeRelaxSetNumNodesets( void *relax_vdata , int num_nodesets );
int hypre_NodeRelaxSetNodeset( void *relax_vdata , int nodeset , int nodeset_size , hypre_Index nodeset_stride , hypre_Index *nodeset_indices );
int hypre_NodeRelaxSetNodesetRank( void *relax_vdata , int nodeset , int nodeset_rank );
int hypre_NodeRelaxSetTempVec( void *relax_vdata , hypre_SStructPVector *t );

/* sstruct_amr_intercommunication.c */
int hypre_SStructAMRInterCommunication( hypre_SStructSendInfoData *sendinfo , hypre_SStructRecvInfoData *recvinfo , hypre_BoxArray *send_data_space , hypre_BoxArray *recv_data_space , int num_values , MPI_Comm comm , hypre_CommPkg **comm_pkg_ptr );

/* sstruct_owninfo.c */
int hypre_SStructIndexScaleF_C( hypre_Index findex , hypre_Index index , hypre_Index stride , hypre_Index cindex );
int hypre_SStructIndexScaleC_F( hypre_Index cindex , hypre_Index index , hypre_Index stride , hypre_Index findex );
hypre_SStructOwnInfoData *hypre_SStructOwnInfo( hypre_StructGrid *fgrid , hypre_StructGrid *cgrid , hypre_BoxMap *cmap , hypre_BoxMap *fmap , hypre_Index rfactor );
int hypre_SStructOwnInfoDataDestroy( hypre_SStructOwnInfoData *owninfo_data );

/* sstruct_recvinfo.c */
hypre_SStructRecvInfoData *hypre_SStructRecvInfo( hypre_StructGrid *cgrid , hypre_BoxMap *fmap , hypre_Index rfactor );
int hypre_SStructRecvInfoDataDestroy( hypre_SStructRecvInfoData *recvinfo_data );

/* sstruct_sendinfo.c */
hypre_SStructSendInfoData *hypre_SStructSendInfo( hypre_StructGrid *fgrid , hypre_BoxMap *cmap , hypre_Index rfactor );
int hypre_SStructSendInfoDataDestroy( hypre_SStructSendInfoData *sendinfo_data );

/* sys_pfmg.c */
void *hypre_SysPFMGCreate( MPI_Comm comm );
int hypre_SysPFMGDestroy( void *sys_pfmg_vdata );
int hypre_SysPFMGSetTol( void *sys_pfmg_vdata , double tol );
int hypre_SysPFMGSetMaxIter( void *sys_pfmg_vdata , int max_iter );
int hypre_SysPFMGSetRelChange( void *sys_pfmg_vdata , int rel_change );
int hypre_SysPFMGSetZeroGuess( void *sys_pfmg_vdata , int zero_guess );
int hypre_SysPFMGSetRelaxType( void *sys_pfmg_vdata , int relax_type );
int hypre_SysPFMGSetNumPreRelax( void *sys_pfmg_vdata , int num_pre_relax );
int hypre_SysPFMGSetNumPostRelax( void *sys_pfmg_vdata , int num_post_relax );
int hypre_SysPFMGSetSkipRelax( void *sys_pfmg_vdata , int skip_relax );
int hypre_SysPFMGSetDxyz( void *sys_pfmg_vdata , double *dxyz );
int hypre_SysPFMGSetLogging( void *sys_pfmg_vdata , int logging );
int hypre_SysPFMGSetPrintLevel( void *sys_pfmg_vdata , int print_level );
int hypre_SysPFMGGetNumIterations( void *sys_pfmg_vdata , int *num_iterations );
int hypre_SysPFMGPrintLogging( void *sys_pfmg_vdata , int myid );
int hypre_SysPFMGGetFinalRelativeResidualNorm( void *sys_pfmg_vdata , double *relative_residual_norm );

/* sys_pfmg_relax.c */
void *hypre_SysPFMGRelaxCreate( MPI_Comm comm );
int hypre_SysPFMGRelaxDestroy( void *sys_pfmg_relax_vdata );
int hypre_SysPFMGRelax( void *sys_pfmg_relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *b , hypre_SStructPVector *x );
int hypre_SysPFMGRelaxSetup( void *sys_pfmg_relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *b , hypre_SStructPVector *x );
int hypre_SysPFMGRelaxSetType( void *sys_pfmg_relax_vdata , int relax_type );
int hypre_SysPFMGRelaxSetPreRelax( void *sys_pfmg_relax_vdata );
int hypre_SysPFMGRelaxSetPostRelax( void *sys_pfmg_relax_vdata );
int hypre_SysPFMGRelaxSetTol( void *sys_pfmg_relax_vdata , double tol );
int hypre_SysPFMGRelaxSetMaxIter( void *sys_pfmg_relax_vdata , int max_iter );
int hypre_SysPFMGRelaxSetZeroGuess( void *sys_pfmg_relax_vdata , int zero_guess );
int hypre_SysPFMGRelaxSetTempVec( void *sys_pfmg_relax_vdata , hypre_SStructPVector *t );

/* sys_pfmg_setup.c */
int hypre_SysPFMGSetup( void *sys_pfmg_vdata , hypre_SStructMatrix *A_in , hypre_SStructVector *b_in , hypre_SStructVector *x_in );
int hypre_SysStructCoarsen( hypre_SStructPGrid *fgrid , hypre_Index index , hypre_Index stride , int prune , hypre_SStructPGrid **cgrid_ptr );

/* sys_pfmg_setup_interp.c */
hypre_SStructPMatrix *hypre_SysPFMGCreateInterpOp( hypre_SStructPMatrix *A , hypre_SStructPGrid *cgrid , int cdir );
int hypre_SysPFMGSetupInterpOp( hypre_SStructPMatrix *A , int cdir , hypre_Index findex , hypre_Index stride , hypre_SStructPMatrix *P );

/* sys_pfmg_setup_rap.c */
hypre_SStructPMatrix *hypre_SysPFMGCreateRAPOp( hypre_SStructPMatrix *R , hypre_SStructPMatrix *A , hypre_SStructPMatrix *P , hypre_SStructPGrid *coarse_grid , int cdir );
int hypre_SysPFMGSetupRAPOp( hypre_SStructPMatrix *R , hypre_SStructPMatrix *A , hypre_SStructPMatrix *P , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_SStructPMatrix *Ac );

/* sys_pfmg_solve.c */
int hypre_SysPFMGSolve( void *sys_pfmg_vdata , hypre_SStructMatrix *A_in , hypre_SStructVector *b_in , hypre_SStructVector *x_in );

/* sys_semi_interp.c */
int hypre_SysSemiInterpCreate( void **sys_interp_vdata_ptr );
int hypre_SysSemiInterpSetup( void *sys_interp_vdata , hypre_SStructPMatrix *P , int P_stored_as_transpose , hypre_SStructPVector *xc , hypre_SStructPVector *e , hypre_Index cindex , hypre_Index findex , hypre_Index stride );
int hypre_SysSemiInterp( void *sys_interp_vdata , hypre_SStructPMatrix *P , hypre_SStructPVector *xc , hypre_SStructPVector *e );
int hypre_SysSemiInterpDestroy( void *sys_interp_vdata );

/* sys_semi_restrict.c */
int hypre_SysSemiRestrictCreate( void **sys_restrict_vdata_ptr );
int hypre_SysSemiRestrictSetup( void *sys_restrict_vdata , hypre_SStructPMatrix *R , int R_stored_as_transpose , hypre_SStructPVector *r , hypre_SStructPVector *rc , hypre_Index cindex , hypre_Index findex , hypre_Index stride );
int hypre_SysSemiRestrict( void *sys_restrict_vdata , hypre_SStructPMatrix *R , hypre_SStructPVector *r , hypre_SStructPVector *rc );
int hypre_SysSemiRestrictDestroy( void *sys_restrict_vdata );


#ifdef __cplusplus
}
#endif

#endif

