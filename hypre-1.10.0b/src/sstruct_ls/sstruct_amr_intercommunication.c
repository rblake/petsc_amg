#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SStructAMRInterCommunication: Given the sendinfo, recvinfo, etc.,
 * a communication pkg is formed. This pkg may be used for amr inter_level
 * communication.
 *--------------------------------------------------------------------------*/

int
hypre_SStructAMRInterCommunication( hypre_SStructSendInfoData *sendinfo,
                                    hypre_SStructRecvInfoData *recvinfo,
                                    hypre_BoxArray            *send_data_space,
                                    hypre_BoxArray            *recv_data_space,
                                    int                        num_values,
                                    MPI_Comm                   comm,
                                    hypre_CommPkg            **comm_pkg_ptr )
{
   hypre_CommInfo         *comm_info;
   hypre_CommPkg          *comm_pkg;

   hypre_BoxArrayArray    *sendboxes;
   int                   **sprocesses;
   hypre_BoxArrayArray    *send_rboxes;
   int                   **send_rboxnums;

   hypre_BoxArrayArray    *recvboxes;
   int                   **rprocesses;

   hypre_BoxArray         *boxarray;

   int                     i, j;
   int                     ierr = 0;

   /*------------------------------------------------------------------------
    *  The communication info is copied from sendinfo & recvinfo.
    *------------------------------------------------------------------------*/
   sendboxes  = hypre_BoxArrayArrayDuplicate(sendinfo -> send_boxes);
   send_rboxes= hypre_BoxArrayArrayDuplicate(sendinfo -> send_boxes);

   sprocesses   = hypre_CTAlloc(int *, hypre_BoxArrayArraySize(send_rboxes));
   send_rboxnums= hypre_CTAlloc(int *, hypre_BoxArrayArraySize(send_rboxes));

   hypre_ForBoxArrayI(i, sendboxes)
   {
      boxarray= hypre_BoxArrayArrayBoxArray(sendboxes, i);
      sprocesses[i]   = hypre_CTAlloc(int, hypre_BoxArraySize(boxarray));
      send_rboxnums[i]= hypre_CTAlloc(int, hypre_BoxArraySize(boxarray));

      hypre_ForBoxI(j, boxarray)
      {
         sprocesses[i][j]   = (sendinfo -> send_procs)[i][j];
         send_rboxnums[i][j]= (sendinfo -> send_remote_boxnums)[i][j];
      }
   }

   recvboxes  = hypre_BoxArrayArrayDuplicate(recvinfo -> recv_boxes);
   rprocesses = hypre_CTAlloc(int *, hypre_BoxArrayArraySize(recvboxes));

   hypre_ForBoxArrayI(i, recvboxes)
   {
      boxarray= hypre_BoxArrayArrayBoxArray(recvboxes, i);
      rprocesses[i]= hypre_CTAlloc(int, hypre_BoxArraySize(boxarray));

      hypre_ForBoxI(j, boxarray)
      {
         rprocesses[i][j]   = (recvinfo -> recv_procs)[i][j];
      }
   }


   hypre_CommInfoCreate(sendboxes, recvboxes, sprocesses, rprocesses,
                        send_rboxnums, send_rboxes, &comm_info);

   hypre_CommPkgCreate(comm_info,
                       send_data_space,
                       recv_data_space,
                       num_values,
                       comm,
                      &comm_pkg);

  *comm_pkg_ptr = comm_pkg;

   return ierr;
}


