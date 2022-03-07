
#pragma once

#include "const.h"


class Exchange {

protected:

  int nx;
  int ny;
  int hs;
  int max_pack;
  bool exchW;
  bool exchE;
  bool exchS;
  bool exchN;
  SArray<int,2,3,3> neigh;

  #ifdef __ENABLE_MPI__
    MPI_Datatype mpi_dtype;
  #endif

  int num_pack;
  int num_unpack;

  MPI_Request sReq [2];
  MPI_Request rReq [2];

  MPI_Status  sStat[2];
  MPI_Status  rStat[2];

  real3d haloSendBufS;
  real3d haloSendBufN;
  real3d haloSendBufW;
  real3d haloSendBufE;
  real3d haloRecvBufS;
  real3d haloRecvBufN;
  real3d haloRecvBufW;
  real3d haloRecvBufE;

  real2d edgeRecvBufE;
  real2d edgeRecvBufW;
  real2d edgeSendBufE;
  real2d edgeSendBufW;
  real2d edgeRecvBufN;
  real2d edgeRecvBufS;
  real2d edgeSendBufN;
  real2d edgeSendBufS;

  realHost3d haloSendBufS_host;
  realHost3d haloSendBufN_host;
  realHost3d haloSendBufW_host;
  realHost3d haloSendBufE_host;
  realHost3d haloRecvBufS_host;
  realHost3d haloRecvBufN_host;
  realHost3d haloRecvBufW_host;
  realHost3d haloRecvBufE_host;

  realHost2d edgeRecvBufE_host;
  realHost2d edgeRecvBufW_host;
  realHost2d edgeSendBufE_host;
  realHost2d edgeSendBufW_host;
  realHost2d edgeRecvBufN_host;
  realHost2d edgeRecvBufS_host;
  realHost2d edgeSendBufN_host;
  realHost2d edgeSendBufS_host;

public:


  void mpiwrap( int err , int line ) {
    #ifdef __ENABLE_MPI__
      if (err != MPI_SUCCESS) {
        std::cerr << "ERROR: MPI error at line" << line << std::endl;
        endrun("");
      }
    #endif
  }


  Exchange() {
    nx = -1;
    ny = -1;
    max_pack = -1;
    #ifdef __ENABLE_MPI__
      mpi_dtype = MPI_DOUBLE;
      if (std::is_same<real,float>::value) mpi_dtype = MPI_FLOAT;
    #endif
  }


  void allocate(int max_pack, int nx, int ny, int px, int py, int nproc_x, int nproc_y,
                bool periodic_x, bool periodic_y, SArray<int,2,3,3> &neigh, int hs) {
    this->max_pack = max_pack;
    this->nx       = nx      ;
    this->ny       = ny      ;
    this->neigh    = neigh   ;
    this->hs       = hs      ;

    this->exchW = ( px != 0         || (px == 0         && periodic_x) );
    this->exchE = ( px != nproc_x-1 || (px == nproc_x-1 && periodic_x) );

    this->exchS = ( py != 0         || (py == 0         && periodic_y) );
    this->exchN = ( py != nproc_y-1 || (py == nproc_y-1 && periodic_y) );

    haloSendBufS = real3d("haloSendBufS",max_pack,hs,nx);
    haloSendBufN = real3d("haloSendBufN",max_pack,hs,nx);
    haloSendBufW = real3d("haloSendBufW",max_pack,ny,hs);
    haloSendBufE = real3d("haloSendBufE",max_pack,ny,hs);
    haloRecvBufS = real3d("haloRecvBufS",max_pack,hs,nx);
    haloRecvBufN = real3d("haloRecvBufN",max_pack,hs,nx);
    haloRecvBufW = real3d("haloRecvBufW",max_pack,ny,hs);
    haloRecvBufE = real3d("haloRecvBufE",max_pack,ny,hs);

    edgeSendBufS = real2d("edgeSendBufS",max_pack,nx);
    edgeSendBufN = real2d("edgeSendBufN",max_pack,nx);
    edgeSendBufW = real2d("edgeSendBufW",max_pack,ny);
    edgeSendBufE = real2d("edgeSendBufE",max_pack,ny);
    edgeRecvBufS = real2d("edgeRecvBufS",max_pack,nx);
    edgeRecvBufN = real2d("edgeRecvBufN",max_pack,nx);
    edgeRecvBufW = real2d("edgeRecvBufW",max_pack,ny);
    edgeRecvBufE = real2d("edgeRecvBufE",max_pack,ny);

    haloSendBufS_host = haloSendBufS.createHostCopy(); 
    haloSendBufN_host = haloSendBufN.createHostCopy(); 
    haloSendBufW_host = haloSendBufW.createHostCopy(); 
    haloSendBufE_host = haloSendBufE.createHostCopy(); 
    haloRecvBufS_host = haloRecvBufS.createHostCopy(); 
    haloRecvBufN_host = haloRecvBufN.createHostCopy(); 
    haloRecvBufW_host = haloRecvBufW.createHostCopy(); 
    haloRecvBufE_host = haloRecvBufE.createHostCopy(); 

    edgeSendBufS_host = edgeSendBufS.createHostCopy(); 
    edgeSendBufN_host = edgeSendBufN.createHostCopy(); 
    edgeSendBufW_host = edgeSendBufW.createHostCopy(); 
    edgeSendBufE_host = edgeSendBufE.createHostCopy(); 
    edgeRecvBufS_host = edgeRecvBufS.createHostCopy(); 
    edgeRecvBufN_host = edgeRecvBufN.createHostCopy(); 
    edgeRecvBufW_host = edgeRecvBufW.createHostCopy(); 
    edgeRecvBufE_host = edgeRecvBufE.createHostCopy(); 
  }


  ~Exchange() {
    nx = -1;
    ny = -1;
    max_pack = -1;

    haloSendBufS = real3d();
    haloSendBufN = real3d();
    haloSendBufW = real3d();
    haloSendBufE = real3d();
    haloRecvBufS = real3d();
    haloRecvBufN = real3d();
    haloRecvBufW = real3d();
    haloRecvBufE = real3d();

    edgeSendBufS = real2d();
    edgeSendBufN = real2d();
    edgeSendBufW = real2d();
    edgeSendBufE = real2d();
    edgeRecvBufS = real2d();
    edgeRecvBufN = real2d();
    edgeRecvBufW = real2d();
    edgeRecvBufE = real2d();

    haloSendBufS_host = realHost3d(); 
    haloSendBufN_host = realHost3d(); 
    haloSendBufW_host = realHost3d(); 
    haloSendBufE_host = realHost3d(); 
    haloRecvBufS_host = realHost3d(); 
    haloRecvBufN_host = realHost3d(); 
    haloRecvBufW_host = realHost3d(); 
    haloRecvBufE_host = realHost3d(); 

    edgeSendBufS_host = realHost2d(); 
    edgeSendBufN_host = realHost2d(); 
    edgeSendBufW_host = realHost2d(); 
    edgeSendBufE_host = realHost2d(); 
    edgeRecvBufS_host = realHost2d(); 
    edgeRecvBufN_host = realHost2d(); 
    edgeRecvBufW_host = realHost2d(); 
    edgeRecvBufE_host = realHost2d(); 
  }


  void halo_init() {
    num_pack   = 0;
    num_unpack = 0;
  }


  void halo_finalize() {
    if (num_unpack != num_pack) {
      endrun("ERROR: You did not unpack everything you packed");
    }
  }


  void halo_pack_x(real3d const &arr) {
    YAKL_SCOPE( haloSendBufW , this->haloSendBufW );
    YAKL_SCOPE( haloSendBufE , this->haloSendBufE );
    YAKL_SCOPE( nx           , this->nx           );
    YAKL_SCOPE( ny           , this->ny           );
    YAKL_SCOPE( exchW        , this->exchW        );
    YAKL_SCOPE( exchE        , this->exchE        );
    YAKL_SCOPE( num_pack     , this->num_pack     );
    int num_vars = arr.dimension[0];
    if (num_pack + num_vars > max_pack) endrun("ERROR: Packing too many variables. Increase max_pack");
    if (arr.dimension[1] != ny+2*hs) endrun("ERROR: Array y-dimension not valid");
    if (arr.dimension[2] != nx+2*hs) endrun("ERROR: Array x-dimension not valid");
    parallel_for( SimpleBounds<3>(num_vars,ny,hs) , YAKL_LAMBDA (int v, int j, int ii) {
      if (exchW) haloSendBufW(num_pack+v,j,ii) = arr(v,hs+j,hs+ii);
      if (exchE) haloSendBufE(num_pack+v,j,ii) = arr(v,hs+j,nx+ii);
    });
    num_pack += num_vars;
  }
  void halo_pack_x(real2d const &arr) {
    YAKL_SCOPE( haloSendBufW , this->haloSendBufW );
    YAKL_SCOPE( haloSendBufE , this->haloSendBufE );
    YAKL_SCOPE( nx           , this->nx           );
    YAKL_SCOPE( ny           , this->ny           );
    YAKL_SCOPE( exchW        , this->exchW        );
    YAKL_SCOPE( exchE        , this->exchE        );
    YAKL_SCOPE( num_pack     , this->num_pack     );
    if (num_pack + 1 > max_pack) endrun("ERROR: Packing too many variables. Increase max_pack");
    if (arr.dimension[0] != ny+2*hs) endrun("ERROR: Array y-dimension not valid");
    if (arr.dimension[1] != nx+2*hs) endrun("ERROR: Array x-dimension not valid");
    parallel_for( SimpleBounds<2>(ny,hs) , YAKL_LAMBDA (int j, int ii) {
      if (exchW) haloSendBufW(num_pack,j,ii) = arr(hs+j,hs+ii);
      if (exchE) haloSendBufE(num_pack,j,ii) = arr(hs+j,nx+ii);
    });
    num_pack++;
  }


  void halo_pack_y(real3d const &arr) {
    YAKL_SCOPE( haloSendBufS , this->haloSendBufS );
    YAKL_SCOPE( haloSendBufN , this->haloSendBufN );
    YAKL_SCOPE( nx           , this->nx           );
    YAKL_SCOPE( ny           , this->ny           );
    YAKL_SCOPE( exchS        , this->exchS        );
    YAKL_SCOPE( exchN        , this->exchN        );
    YAKL_SCOPE( num_pack     , this->num_pack     );
    int num_vars = arr.dimension[0];
    if (num_pack + num_vars > max_pack) endrun("ERROR: Packing too many variables. Increase max_pack");
    if (arr.dimension[1] != ny+2*hs) endrun("ERROR: Array y-dimension not valid");
    if (arr.dimension[2] != nx+2*hs) endrun("ERROR: Array x-dimension not valid");
    parallel_for( SimpleBounds<3>(num_vars,hs,nx) , YAKL_LAMBDA (int v, int jj, int i) {
      if (exchS) haloSendBufS(num_pack+v,jj,i) = arr(v,hs+jj,hs+i);
      if (exchN) haloSendBufN(num_pack+v,jj,i) = arr(v,ny+jj,hs+i);
    });
    num_pack += num_vars;
  }
  void halo_pack_y(real2d const &arr) {
    YAKL_SCOPE( haloSendBufS , this->haloSendBufS );
    YAKL_SCOPE( haloSendBufN , this->haloSendBufN );
    YAKL_SCOPE( nx           , this->nx           );
    YAKL_SCOPE( ny           , this->ny           );
    YAKL_SCOPE( exchS        , this->exchS        );
    YAKL_SCOPE( exchN        , this->exchN        );
    YAKL_SCOPE( num_pack     , this->num_pack     );
    if (num_pack + 1 > max_pack) endrun("ERROR: Packing too many variables. Increase max_pack");
    if (arr.dimension[0] != ny+2*hs) endrun("ERROR: Array y-dimension not valid");
    if (arr.dimension[1] != nx+2*hs) endrun("ERROR: Array x-dimension not valid");
    parallel_for( SimpleBounds<2>(hs,nx) , YAKL_LAMBDA (int jj, int i) {
      if (exchS) haloSendBufS(num_pack,jj,i) = arr(hs+jj,hs+i);
      if (exchN) haloSendBufN(num_pack,jj,i) = arr(ny+jj,hs+i);
    });
    num_pack++;
  }


  void halo_unpack_x(real3d &arr) {
    YAKL_SCOPE( haloRecvBufW , this->haloRecvBufW );
    YAKL_SCOPE( haloRecvBufE , this->haloRecvBufE );
    YAKL_SCOPE( nx           , this->nx           );
    YAKL_SCOPE( ny           , this->ny           );
    YAKL_SCOPE( exchW        , this->exchW        );
    YAKL_SCOPE( exchE        , this->exchE        );
    int num_vars = arr.dimension[0];
    if (num_unpack + num_vars > num_pack) endrun("ERROR: Unpacking more items than you packed.");
    if (arr.dimension[1] != ny+2*hs) endrun("ERROR: Array y-dimension not valid");
    if (arr.dimension[2] != nx+2*hs) endrun("ERROR: Array x-dimension not valid");
    parallel_for( SimpleBounds<3>(num_vars,ny,hs) , YAKL_LAMBDA (int v, int j, int ii) {
      if (exchW) arr(v,hs+j,      ii) = haloRecvBufW(num_unpack+v,j,ii);
      if (exchE) arr(v,hs+j,nx+hs+ii) = haloRecvBufE(num_unpack+v,j,ii);
    });
    num_unpack += num_vars;
  }
  void halo_unpack_x(real2d &arr) {
    YAKL_SCOPE( haloRecvBufW , this->haloRecvBufW );
    YAKL_SCOPE( haloRecvBufE , this->haloRecvBufE );
    YAKL_SCOPE( nx           , this->nx           );
    YAKL_SCOPE( ny           , this->ny           );
    YAKL_SCOPE( exchW        , this->exchW        );
    YAKL_SCOPE( exchE        , this->exchE        );
    if (num_unpack + 1 > num_pack) endrun("ERROR: Unpacking more items than you packed.");
    if (arr.dimension[0] != ny+2*hs) endrun("ERROR: Array y-dimension not valid");
    if (arr.dimension[1] != nx+2*hs) endrun("ERROR: Array x-dimension not valid");
    parallel_for( SimpleBounds<2>(ny,hs) , YAKL_LAMBDA (int j, int ii) {
      if (exchW) arr(hs+j,      ii) = haloRecvBufW(num_unpack,j,ii);
      if (exchE) arr(hs+j,nx+hs+ii) = haloRecvBufE(num_unpack,j,ii);
    });
    num_unpack++;
  }


  void halo_unpack_y(real3d &arr) {
    YAKL_SCOPE( haloRecvBufS , this->haloRecvBufS );
    YAKL_SCOPE( haloRecvBufN , this->haloRecvBufN );
    YAKL_SCOPE( nx           , this->nx           );
    YAKL_SCOPE( ny           , this->ny           );
    YAKL_SCOPE( exchS        , this->exchS        );
    YAKL_SCOPE( exchN        , this->exchN        );
    int num_vars = arr.dimension[0];
    if (num_unpack + num_vars > num_pack) endrun("ERROR: Unpacking more items than you packed.");
    if (arr.dimension[1] != ny+2*hs) endrun("ERROR: Array y-dimension not valid");
    if (arr.dimension[2] != nx+2*hs) endrun("ERROR: Array x-dimension not valid");
    parallel_for( SimpleBounds<3>(num_vars,hs,nx) , YAKL_LAMBDA (int v, int jj, int i) {
      if (exchS) arr(v,      jj,hs+i) = haloRecvBufS(num_unpack+v,jj,i);
      if (exchN) arr(v,ny+hs+jj,hs+i) = haloRecvBufN(num_unpack+v,jj,i);
    });
    num_unpack += num_vars;
  }
  void halo_unpack_y(real2d &arr) {
    YAKL_SCOPE( haloRecvBufS , this->haloRecvBufS );
    YAKL_SCOPE( haloRecvBufN , this->haloRecvBufN );
    YAKL_SCOPE( nx           , this->nx           );
    YAKL_SCOPE( ny           , this->ny           );
    YAKL_SCOPE( exchS        , this->exchS        );
    YAKL_SCOPE( exchN        , this->exchN        );
    if (num_unpack + 1 > num_pack) endrun("ERROR: Unpacking more items than you packed.");
    if (arr.dimension[0] != ny+2*hs) endrun("ERROR: Array y-dimension not valid");
    if (arr.dimension[1] != nx+2*hs) endrun("ERROR: Array x-dimension not valid");
    parallel_for( SimpleBounds<2>(hs,nx) , YAKL_LAMBDA (int jj, int i) {
      if (exchS) arr(      jj,hs+i) = haloRecvBufS(num_unpack,jj,i);
      if (exchN) arr(ny+hs+jj,hs+i) = haloRecvBufN(num_unpack,jj,i);
    });
    num_unpack++;
  }


  void halo_exchange_x() {
    #ifdef __ENABLE_MPI__
      int ierr;

      yakl::fence();

      //Pre-post the receives
      if (exchW) mpiwrap( MPI_Irecv( haloRecvBufW_host.data() , num_pack*ny*hs , mpi_dtype , neigh(1,0) , 0 ,
                                     MPI_COMM_WORLD , &rReq[0] ) , __LINE__ );
      if (exchE) mpiwrap( MPI_Irecv( haloRecvBufE_host.data() , num_pack*ny*hs , mpi_dtype , neigh(1,2) , 1 ,
                                     MPI_COMM_WORLD , &rReq[1] ) , __LINE__ );

      if (exchW) haloSendBufW.deep_copy_to(haloSendBufW_host);
      if (exchE) haloSendBufE.deep_copy_to(haloSendBufE_host);
      yakl::fence();

      //Send the data
      if (exchW) mpiwrap( MPI_Isend( haloSendBufW_host.data() , num_pack*ny*hs , mpi_dtype , neigh(1,0) , 1 ,
                                     MPI_COMM_WORLD , &sReq[0] ) , __LINE__ );
      if (exchE) mpiwrap( MPI_Isend( haloSendBufE_host.data() , num_pack*ny*hs , mpi_dtype , neigh(1,2) , 0 ,
                                     MPI_COMM_WORLD , &sReq[1] ) , __LINE__ );

      //Wait for the sends and receives to finish
      if (exchW) {
        mpiwrap( MPI_Wait(&sReq[0], &sStat[0]) , __LINE__ );
        mpiwrap( MPI_Wait(&rReq[0], &rStat[0]) , __LINE__ );
      }
      if (exchE) {
        mpiwrap( MPI_Wait(&sReq[1], &sStat[1]) , __LINE__ );
        mpiwrap( MPI_Wait(&rReq[1], &rStat[1]) , __LINE__ );
      }

      if (exchW) haloRecvBufW_host.deep_copy_to(haloRecvBufW);
      if (exchE) haloRecvBufE_host.deep_copy_to(haloRecvBufE);
    #endif
  }


  void halo_exchange_y() {
    #ifdef __ENABLE_MPI__
      int ierr;

      yakl::fence();

      //Pre-post the receives
      if (exchS) mpiwrap( MPI_Irecv( haloRecvBufS_host.data() , num_pack*hs*nx , mpi_dtype , neigh(0,1) , 0 ,
                                     MPI_COMM_WORLD , &rReq[0] ) , __LINE__ );
      if (exchN) mpiwrap( MPI_Irecv( haloRecvBufN_host.data() , num_pack*hs*nx , mpi_dtype , neigh(2,1) , 1 ,
                                     MPI_COMM_WORLD , &rReq[1] ) , __LINE__ );

      if (exchS) haloSendBufS.deep_copy_to(haloSendBufS_host);
      if (exchN) haloSendBufN.deep_copy_to(haloSendBufN_host);
      yakl::fence();

      //Send the data
      if (exchS) mpiwrap( MPI_Isend( haloSendBufS_host.data() , num_pack*hs*nx , mpi_dtype , neigh(0,1) , 1 ,
                                     MPI_COMM_WORLD , &sReq[0] ) , __LINE__ );
      if (exchN) mpiwrap( MPI_Isend( haloSendBufN_host.data() , num_pack*hs*nx , mpi_dtype , neigh(2,1) , 0 ,
                                     MPI_COMM_WORLD , &sReq[1] ) , __LINE__ );

      //Wait for the sends and receives to finish
      if (exchS) {
        mpiwrap( MPI_Wait(&sReq[0], &sStat[0]) , __LINE__ );
        mpiwrap( MPI_Wait(&rReq[0], &rStat[0]) , __LINE__ );
      }
      if (exchN) {
        mpiwrap( MPI_Wait(&sReq[1], &sStat[1]) , __LINE__ );
        mpiwrap( MPI_Wait(&rReq[1], &rStat[1]) , __LINE__ );
      }

      if (exchS) haloRecvBufS_host.deep_copy_to(haloRecvBufS);
      if (exchN) haloRecvBufN_host.deep_copy_to(haloRecvBufN);
    #endif
  }


  void edge_init() {
    num_pack = 0;
    num_unpack = 0;
  }


  void edge_finalize() {
  }


  void edge_pack_x(real4d const &fwaves, real3d const &surf, real3d const &h_u, real3d const &u_u) {
    YAKL_SCOPE( edgeSendBufW , this->edgeSendBufW );
    YAKL_SCOPE( edgeSendBufE , this->edgeSendBufE );
    YAKL_SCOPE( nx           , this->nx           );
    YAKL_SCOPE( ny           , this->ny           );
    YAKL_SCOPE( exchW        , this->exchW        );
    YAKL_SCOPE( exchE        , this->exchE        );
    int numState = fwaves.dimension[0];
    parallel_for( SimpleBounds<2>(numState+3,ny) , YAKL_LAMBDA (int v, int j) {
      if (v < numState) {
        if (exchW) edgeSendBufW(v,j) = fwaves(v,1,j,0 );
        if (exchE) edgeSendBufE(v,j) = fwaves(v,0,j,nx);
      } else if (v == numState) { 
        if (exchW) edgeSendBufW(v,j) = surf(1,j,0 );
        if (exchE) edgeSendBufE(v,j) = surf(0,j,nx);
      } else if (v == numState+1) { 
        if (exchW) edgeSendBufW(v,j) = h_u(1,j,0 );
        if (exchE) edgeSendBufE(v,j) = h_u(0,j,nx);
      } else if (v == numState+2) { 
        if (exchW) edgeSendBufW(v,j) = u_u(1,j,0 );
        if (exchE) edgeSendBufE(v,j) = u_u(0,j,nx);
      }
    });
    num_pack += numState+3;
  }


  void edge_pack_y(real4d const &fwaves, real3d const &surf, real3d const &h_v, real3d const &v_v) {
    YAKL_SCOPE( edgeSendBufS , this->edgeSendBufS );
    YAKL_SCOPE( edgeSendBufN , this->edgeSendBufN );
    YAKL_SCOPE( nx           , this->nx           );
    YAKL_SCOPE( ny           , this->ny           );
    YAKL_SCOPE( exchS        , this->exchS        );
    YAKL_SCOPE( exchN        , this->exchN        );
    int numState = fwaves.dimension[0];
    parallel_for( SimpleBounds<2>(numState+3,nx) , YAKL_LAMBDA (int v, int i) {
      if (v < numState) {
        if (exchS) edgeSendBufS(v,i) = fwaves(v,1,0 ,i);
        if (exchN) edgeSendBufN(v,i) = fwaves(v,0,ny,i);
      } else if (v == numState) {
        if (exchS) edgeSendBufS(v,i) = surf(1,0 ,i);
        if (exchN) edgeSendBufN(v,i) = surf(0,ny,i);
      } else if (v == numState+1) { 
        if (exchS) edgeSendBufS(v,i) = h_v(1,0 ,i);
        if (exchN) edgeSendBufN(v,i) = h_v(0,ny,i);
      } else if (v == numState+2) { 
        if (exchS) edgeSendBufS(v,i) = v_v(1,0 ,i);
        if (exchN) edgeSendBufN(v,i) = v_v(0,ny,i);
      }
    });
    num_pack += numState+3;
  }


  void edge_unpack_x(real4d &fwaves, real3d &surf, real3d &h_u, real3d &u_u) {
    YAKL_SCOPE( edgeRecvBufW , this->edgeRecvBufW );
    YAKL_SCOPE( edgeRecvBufE , this->edgeRecvBufE );
    YAKL_SCOPE( nx           , this->nx           );
    YAKL_SCOPE( ny           , this->ny           );
    YAKL_SCOPE( exchW        , this->exchW        );
    YAKL_SCOPE( exchE        , this->exchE        );
    int numState = fwaves.dimension[0];
    parallel_for( SimpleBounds<2>(numState+3,ny) , YAKL_LAMBDA (int v, int j) {
      if (v < numState) {
        if (exchW) fwaves(v,0,j,0 ) = edgeRecvBufW(v,j);
        if (exchE) fwaves(v,1,j,nx) = edgeRecvBufE(v,j);
      } else if (v == numState) {
        if (exchW) surf(0,j,0 ) = edgeRecvBufW(v,j);
        if (exchE) surf(1,j,nx) = edgeRecvBufE(v,j);
      } else if (v == numState+1) {
        if (exchW) h_u(0,j,0 ) = edgeRecvBufW(v,j);
        if (exchE) h_u(1,j,nx) = edgeRecvBufE(v,j);
      } else if (v == numState+2) {
        if (exchW) u_u(0,j,0 ) = edgeRecvBufW(v,j);
        if (exchE) u_u(1,j,nx) = edgeRecvBufE(v,j);
      }
    });
    num_unpack += numState+3;
  }


  void edge_unpack_y(real4d &fwaves, real3d &surf, real3d &h_v, real3d &v_v) {
    YAKL_SCOPE( edgeRecvBufS , this->edgeRecvBufS );
    YAKL_SCOPE( edgeRecvBufN , this->edgeRecvBufN );
    YAKL_SCOPE( nx           , this->nx           );
    YAKL_SCOPE( ny           , this->ny           );
    YAKL_SCOPE( exchS        , this->exchS        );
    YAKL_SCOPE( exchN        , this->exchN        );
    int numState = fwaves.dimension[0];
    parallel_for( SimpleBounds<2>(numState+3,nx) , YAKL_LAMBDA (int v, int i) {
      if (v < numState) {
        if (exchS) fwaves(v,0,0 ,i) = edgeRecvBufS(v,i);
        if (exchN) fwaves(v,1,ny,i) = edgeRecvBufN(v,i);
      } else if (v == numState) {
        if (exchS) surf(0,0 ,i) = edgeRecvBufS(v,i);
        if (exchN) surf(1,ny,i) = edgeRecvBufN(v,i);
      } else if (v == numState+1) {
        if (exchS) h_v(0,0 ,i) = edgeRecvBufS(v,i);
        if (exchN) h_v(1,ny,i) = edgeRecvBufN(v,i);
      } else if (v == numState+2) {
        if (exchS) v_v(0,0 ,i) = edgeRecvBufS(v,i);
        if (exchN) v_v(1,ny,i) = edgeRecvBufN(v,i);
      }
    });
    num_unpack += numState+3;
  }


  void edge_exchange_x() {
    #ifdef __ENABLE_MPI__
      int ierr;

      yakl::fence();

      //Pre-post the receives
      if (exchW) mpiwrap( MPI_Irecv( edgeRecvBufW_host.data() , num_pack*ny , mpi_dtype , neigh(1,0) , 0 ,
                                     MPI_COMM_WORLD , &rReq[0] ) , __LINE__ );
      if (exchE) mpiwrap( MPI_Irecv( edgeRecvBufE_host.data() , num_pack*ny , mpi_dtype , neigh(1,2) , 1 ,
                                     MPI_COMM_WORLD , &rReq[1] ) , __LINE__ );

      if (exchW) edgeSendBufW.deep_copy_to(edgeSendBufW_host);
      if (exchE) edgeSendBufE.deep_copy_to(edgeSendBufE_host);
      yakl::fence();

      //Send the data
      if (exchW) mpiwrap( MPI_Isend( edgeSendBufW_host.data() , num_pack*ny , mpi_dtype , neigh(1,0) , 1 ,
                                     MPI_COMM_WORLD , &sReq[0] ) , __LINE__ );
      if (exchE) mpiwrap( MPI_Isend( edgeSendBufE_host.data() , num_pack*ny , mpi_dtype , neigh(1,2) , 0 ,
                                     MPI_COMM_WORLD , &sReq[1] ) , __LINE__ );

      //Wait for the sends and receives to finish
      if (exchW) {
        mpiwrap( MPI_Wait(&sReq[0], &sStat[0]) , __LINE__ );
        mpiwrap( MPI_Wait(&rReq[0], &rStat[0]) , __LINE__ );
      }
      if (exchE) {
        mpiwrap( MPI_Wait(&sReq[1], &sStat[1]) , __LINE__ );
        mpiwrap( MPI_Wait(&rReq[1], &rStat[1]) , __LINE__ );
      }

      if (exchW) edgeRecvBufW_host.deep_copy_to(edgeRecvBufW);
      if (exchE) edgeRecvBufE_host.deep_copy_to(edgeRecvBufE);
    #endif
  }


  void edge_exchange_y() {
    #ifdef __ENABLE_MPI__
      int ierr;

      yakl::fence();

      //Pre-post the receives
      if (exchS) mpiwrap( MPI_Irecv( edgeRecvBufS_host.data() , num_pack*nx , mpi_dtype , neigh(0,1) , 0 ,
                                     MPI_COMM_WORLD , &rReq[0] ) , __LINE__ );
      if (exchN) mpiwrap( MPI_Irecv( edgeRecvBufN_host.data() , num_pack*nx , mpi_dtype , neigh(2,1) , 1 ,
                                     MPI_COMM_WORLD , &rReq[1] ) , __LINE__ );

      if (exchS) edgeSendBufS.deep_copy_to(edgeSendBufS_host);
      if (exchN) edgeSendBufN.deep_copy_to(edgeSendBufN_host);
      yakl::fence();

      //Send the data
      if (exchS) mpiwrap( MPI_Isend( edgeSendBufS_host.data() , num_pack*nx , mpi_dtype , neigh(0,1) , 1 ,
                                     MPI_COMM_WORLD , &sReq[0] ) , __LINE__ );
      if (exchN) mpiwrap( MPI_Isend( edgeSendBufN_host.data() , num_pack*nx , mpi_dtype , neigh(2,1) , 0 ,
                                     MPI_COMM_WORLD , &sReq[1] ) , __LINE__ );

      //Wait for the sends and receives to finish
      if (exchS) {
        mpiwrap( MPI_Wait(&sReq[0], &sStat[0]) , __LINE__ );
        mpiwrap( MPI_Wait(&rReq[0], &rStat[0]) , __LINE__ );
      }
      if (exchN) {
        mpiwrap( MPI_Wait(&sReq[1], &sStat[1]) , __LINE__ );
        mpiwrap( MPI_Wait(&rReq[1], &rStat[1]) , __LINE__ );
      }

      if (exchS) edgeRecvBufS_host.deep_copy_to(edgeRecvBufS);
      if (exchN) edgeRecvBufN_host.deep_copy_to(edgeRecvBufN);
    #endif
  }


};

