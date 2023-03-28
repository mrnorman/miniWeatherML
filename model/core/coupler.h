
#pragma once

#include "main_header.h"
#include "DataManager.h"
#include "YAKL_pnetcdf.h"
#include "YAKL_netcdf.h"
#include "MultipleFields.h"
#include "Options.h"

// The Coupler class holds everything a component or module of this model would need in order to perform its
// changes to the model state


namespace core {

  class Coupler {
  protected:
    Options options;

    real xlen;   // Domain length in the x-direction in meters
    real ylen;   // Domain length in the y-direction in meters
    real zlen;   // Domain length in the z-direction in meters
    real dt_gcm; // Time step of the GCM for this MMF invocation

    // MPI parallelization information
    int    nranks;           // Total number of MPI ranks / processes
    int    myrank;           // My rank # in [0,nranks-1]
    int    nens;             // Number of ensembles
    size_t nx_glob;          // Total global number of cells in the x-direction (summing all MPI Processes)
    size_t ny_glob;          // Total global number of cells in the y-direction (summing all MPI Processes)
    int    nproc_x;          // Number of parallel processes distributed over the x-dimension
    int    nproc_y;          // Number of parallel processes distributed over the y-dimension
                             // nproc_x * nproc_y  must equal  nranks
    int    px;               // My process ID in the x-direction
    int    py;               // My process ID in the y-direction
    size_t i_beg;            // Beginning of my x-direction global index
    size_t j_beg;            // Beginning of my y-direction global index
    size_t i_end;            // End of my x-direction global index
    size_t j_end;            // End of my y-direction global index
    bool   mainproc;         // myrank == 0
    SArray<int,2,3,3> neigh; // List of neighboring rank IDs;  1st index: y;  2nd index: x
                             // Y: 0 = south;  1 = middle;  2 = north
                             // X: 0 = west ;  1 = center;  3 = east 

    DataManager dm;

    struct Tracer {
      std::string name;
      std::string desc;
      bool        positive;
      bool        adds_mass;
    };
    std::vector<Tracer> tracers;


  public:

    Coupler() {
      this->xlen   = -1;
      this->ylen   = -1;
      this->zlen   = -1;
      this->dt_gcm = -1;
    }


    Coupler(Coupler &&) = default;
    Coupler &operator=(Coupler &&) = default;
    Coupler(Coupler const &) = delete;
    Coupler &operator=(Coupler const &) = delete;


    ~Coupler() {
      yakl::fence();
      dm.finalize();
      options.finalize();
      tracers = std::vector<Tracer>();
      this->xlen   = -1;
      this->ylen   = -1;
      this->zlen   = -1;
      this->dt_gcm = -1;
    }


    void clone_into( Coupler &coupler ) {
      coupler.xlen     = this->xlen    ;
      coupler.ylen     = this->ylen    ;
      coupler.zlen     = this->zlen    ;
      coupler.dt_gcm   = this->dt_gcm  ;
      coupler.tracers  = this->tracers ;
      coupler.nranks   = this->nranks  ;
      coupler.myrank   = this->myrank  ;
      coupler.nens     = this->nens    ;
      coupler.nx_glob  = this->nx_glob ;
      coupler.ny_glob  = this->ny_glob ;
      coupler.nproc_x  = this->nproc_x ;
      coupler.nproc_y  = this->nproc_y ;
      coupler.px       = this->px      ;
      coupler.py       = this->py      ;
      coupler.i_beg    = this->i_beg   ;
      coupler.j_beg    = this->j_beg   ;
      coupler.i_end    = this->i_end   ;
      coupler.j_end    = this->j_end   ;
      coupler.mainproc = this->mainproc;
      coupler.neigh    = this->neigh   ;
      this->dm.clone_into( coupler.dm );
    }


    void distribute_mpi_and_allocate_coupled_state(int nz, size_t ny_glob, size_t nx_glob, int nens,
                                                   int nproc_x_in = -1 , int nproc_y_in = -1 ,
                                                   int px_in      = -1 , int py_in      = -1 ,
                                                   int i_beg_in   = -1 , int i_end_in   = -1 ,
                                                   int j_beg_in   = -1 , int j_end_in   = -1 ) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      this->nens    = nens   ;
      this->nx_glob = nx_glob;
      this->ny_glob = ny_glob;

      MPI_Comm_size( MPI_COMM_WORLD , &nranks );
      MPI_Comm_rank( MPI_COMM_WORLD , &myrank );

      mainproc = (myrank == 0);

      bool sim2d = ny_glob == 1;

      if (sim2d) {
        nproc_x = nranks;
        nproc_y = 1;
      } else {
        // Find integer nproc_y * nproc_x == nranks such that nproc_y and nproc_x are as close as possible
        nproc_y = (int) std::ceil( std::sqrt((double) nranks) );
        while (nproc_y >= 1) {
          if (nranks % nproc_y == 0) { break; }
          nproc_y--;
        }
        nproc_x = nranks / nproc_y;
      }

      // Get my ID within each dimension's number of ranks
      py = myrank / nproc_x;
      px = myrank % nproc_x;

      // Get my beginning and ending indices in the x- and y- directions
      double nper;
      nper = ((double) nx_glob)/nproc_x;
      i_beg = static_cast<size_t>( round( nper* px    )   );
      i_end = static_cast<size_t>( round( nper*(px+1) )-1 );
      nper = ((double) ny_glob)/nproc_y;
      j_beg = static_cast<size_t>( round( nper* py    )   );
      j_end = static_cast<size_t>( round( nper*(py+1) )-1 );

      // For multi-resolution experiments, the user might want to set these manually to ensure that
      //   grids match up properly when decomposed into ranks
      if (nproc_x_in > 0) nproc_x = nproc_x_in;
      if (nproc_y_in > 0) nproc_y = nproc_y_in;
      if (px_in      > 0) px      = px_in     ;
      if (py_in      > 0) py      = py_in     ;
      if (i_beg_in   > 0) i_beg   = i_beg_in  ;
      if (i_end_in   > 0) i_end   = i_end_in  ;
      if (j_beg_in   > 0) j_beg   = j_beg_in  ;
      if (j_end_in   > 0) j_end   = j_end_in  ;

      //Determine my number of grid cells
      int nx = i_end - i_beg + 1;
      int ny = j_end - j_beg + 1;
      for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
          int pxloc = px+i-1;
          while (pxloc < 0        ) { pxloc = pxloc + nproc_x; }
          while (pxloc > nproc_x-1) { pxloc = pxloc - nproc_x; }
          int pyloc = py+j-1;
          while (pyloc < 0        ) { pyloc = pyloc + nproc_y; }
          while (pyloc > nproc_y-1) { pyloc = pyloc - nproc_y; }
          neigh(j,i) = pyloc * nproc_x + pxloc;
        }
      }

      // Debug output for the parallel decomposition
      #if 0
        if (mainproc) {
          std::cout << "There are " << nranks << " ranks, with " << nproc_x << " in the x-direction and "
                                                                 << nproc_y << " in the y-direction.\n\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
        for (int rr=0; rr < nranks; rr++) {
          MPI_Barrier(MPI_COMM_WORLD);
          if (rr == myrank) {
            std::cout << "Hello! My Rank is    : " << myrank << std::endl;
            std::cout << "My proc grid ID is   : " << px << " , " << py << std::endl;
            std::cout << "I have               : " << nx << " x " << ny << " x " << nz <<  " columns." << std::endl;
            std::cout << "I start at index     : " << i_beg << " x " << j_beg << std::endl;
            std::cout << "I end at index       : " << i_end << " x " << j_end << std::endl;
            std::cout << "My neighbor matrix is:" << std::endl;
            for (int j = 2; j >= 0; j--) {
              for (int i = 0; i < 3; i++) {
                std::cout << std::setw(6) << neigh(j,i) << " ";
              }
              std::cout << std::endl;
            }
            std::cout << std::endl;
          }
          MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      #endif

      dm.add_dimension( "nens" , nens );
      dm.add_dimension( "x"    , nx   );
      dm.add_dimension( "y"    , ny   );
      dm.add_dimension( "z"    , nz   );
    }


    void set_dt_gcm(real dt_gcm) { this->dt_gcm = dt_gcm; }

    real                      get_xlen                  () const { return this->xlen        ; }
    real                      get_ylen                  () const { return this->ylen        ; }
    real                      get_zlen                  () const { return this->zlen        ; }
    real                      get_dt_gcm                () const { return this->dt_gcm      ; }
    int                       get_nranks                () const { return this->nranks      ; }
    int                       get_myrank                () const { return this->myrank      ; }
    int                       get_nens                  () const { return this->nens        ; }
    size_t                    get_nx_glob               () const { return this->nx_glob     ; }
    size_t                    get_ny_glob               () const { return this->ny_glob     ; }
    int                       get_nproc_x               () const { return this->nproc_x     ; }
    int                       get_nproc_y               () const { return this->nproc_y     ; }
    int                       get_px                    () const { return this->px          ; }
    int                       get_py                    () const { return this->py          ; }
    size_t                    get_i_beg                 () const { return this->i_beg       ; }
    size_t                    get_j_beg                 () const { return this->j_beg       ; }
    size_t                    get_i_end                 () const { return this->i_end       ; }
    size_t                    get_j_end                 () const { return this->j_end       ; }
    bool                      is_sim2d                  () const { return this->ny_glob == 1; }
    bool                      is_mainproc               () const { return this->mainproc    ; }
    SArray<int,2,3,3> const & get_neighbor_rankid_matrix() const { return this->neigh       ; }
    DataManager       const & get_data_manager_readonly () const { return this->dm          ; }
    DataManager             & get_data_manager_readwrite()       { return this->dm          ; }


    int get_nx() const {
      if (dm.find_dimension("x") == -1) return -1;
      return dm.get_dimension_size("x");
    }


    int get_ny() const {
      if (dm.find_dimension("y") == -1) return -1;
      return dm.get_dimension_size("y");
    }


    int get_nz() const {
      if (dm.find_dimension("z") == -1) return -1;
      return dm.get_dimension_size("z");
    }


    real get_dx() const { return get_xlen() / nx_glob; }


    real get_dy() const { return get_ylen() / ny_glob; }


    real get_dz() const { return get_zlen() / get_nz(); }


    int get_num_tracers() const { return tracers.size(); }


    MPI_Datatype get_mpi_data_type() const {
      if      constexpr (std::is_same<real,float >()) { return MPI_FLOAT; }
      else if constexpr (std::is_same<real,double>()) { return MPI_DOUBLE; }
      else { endrun("ERROR: Invalid type for 'real'"); }
      return MPI_FLOAT;
    }


    template <class T>
    void add_option( std::string key , T value ) {
      options.add_option<T>(key,value);
    }


    template <class T>
    void set_option( std::string key , T value ) {
      options.set_option<T>(key,value);
    }


    template <class T>
    T get_option( std::string key ) const {
      return options.get_option<T>(key);
    }


    template <class T>
    T get_option( std::string key , T val ) const {
      if (option_exists(key)) return options.get_option<T>(key);
      return val;
    }


    bool option_exists( std::string key ) const {
      return options.option_exists(key);
    }


    void delete_option( std::string key ) {
      options.delete_option(key);
    }


    void set_grid(real xlen, real ylen, real zlen) {
      this->xlen = xlen;
      this->ylen = ylen;
      this->zlen = zlen;
    }

    
    void add_tracer( std::string tracer_name , std::string tracer_desc , bool positive , bool adds_mass ) {
      int nz   = get_nz();
      int ny   = get_ny();
      int nx   = get_nx();
      int nens = get_nens();
      dm.register_and_allocate<real>( tracer_name , tracer_desc , {nz,ny,nx,nens} , {"z","y","x","nens"} );
      tracers.push_back( { tracer_name , tracer_desc , positive , adds_mass } );
    }

    
    std::vector<std::string> get_tracer_names() const {
      std::vector<std::string> ret;
      for (int i=0; i < tracers.size(); i++) { ret.push_back( tracers[i].name ); }
      return ret;
    }

    
    void get_tracer_info(std::string tracer_name , std::string &tracer_desc, bool &tracer_found ,
                         bool &positive , bool &adds_mass) const {
      std::vector<std::string> ret;
      for (int i=0; i < tracers.size(); i++) {
        if (tracer_name == tracers[i].name) {
          positive     = tracers[i].positive ;
          tracer_desc  = tracers[i].desc     ;
          adds_mass    = tracers[i].adds_mass;
          tracer_found = true;
          return;
        }
      }
      tracer_found = false;
    }

    
    bool tracer_exists( std::string tracer_name ) const {
      for (int i=0; i < tracers.size(); i++) {
        if (tracer_name == tracers[i].name) return true;
      }
      return false;
    }


    template <class T>
    MultiField<typename std::remove_cv<T>::type,4> create_halos( MultiField<T,4> const &fields_in , int hs ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;

      int num_fields = fields_in.get_num_fields();
      int nens = fields_in.get_field(0).extent(3);
      int nx   = fields_in.get_field(0).extent(2);
      int ny   = fields_in.get_field(0).extent(1);
      int nz   = fields_in.get_field(0).extent(0);

      int hs_y = ny > 1 ? hs : 0;

      MultiField<real,4> fields_out;

      for (int ifield = 0; ifield < num_fields; ifield++) {
        // Allocate
        fields_out.add_field( real4d(fields_in.get_field(ifield).label(),nz+2*hs,ny+2*hs_y,nx+2*hs,nens) );
        fields_out.get_field(ifield) = 0;
        // Fill internal domain
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          fields_out(ifield,hs+k,hs_y+j,hs+i,iens) = fields_in(ifield,k,j,i,iens);
        });
      }

      return fields_out;
    }


    void fill_horizontal_halos_periodic( MultiField<real,4> const &fields , int hs ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      int tag0 = 10;

      int hs_y = fields.get_field(0).extent(1) > 1 ? hs : 0;

      int num_fields = fields.get_num_fields();
      int nens = fields.get_field(0).extent(3);
      int nx   = fields.get_field(0).extent(2) - 2*hs;
      int ny   = fields.get_field(0).extent(1) - 2*hs_y;
      int nz   = fields.get_field(0).extent(0) - 2*hs;

      { // x-direction (east-west)

        // Allocate send buffers and then pack send buffers
        real5d halo_send_buf_W("coupler_halo_send_buf_W",num_fields,nz+2*hs,ny+2*hs_y,hs,nens);
        real5d halo_send_buf_E("coupler_halo_send_buf_E",num_fields,nz+2*hs,ny+2*hs_y,hs,nens);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_fields,nz+2*hs,ny+2*hs_y,hs,nens) ,
                                          YAKL_LAMBDA (int v, int k, int j, int ii, int iens) {
          halo_send_buf_W(v,k,j,ii,iens) = fields(v,k,j,hs+ii,iens);
          halo_send_buf_E(v,k,j,ii,iens) = fields(v,k,j,nx+ii,iens);
        });

        // Allocate host receive buffers and receive the receive buffers
        realHost5d halo_recv_buf_W_host("coupler_halo_recv_buf_W_host",num_fields,nz+2*hs,ny+2*hs_y,hs,nens);
        realHost5d halo_recv_buf_E_host("coupler_halo_recv_buf_E_host",num_fields,nz+2*hs,ny+2*hs_y,hs,nens);
        auto data_type = get_mpi_data_type();
        MPI_Request rReq[2];
        MPI_Irecv( halo_recv_buf_W_host.data() , halo_recv_buf_W_host.size() , data_type , neigh(1,0) , tag0+0 , MPI_COMM_WORLD , &rReq[0] );
        MPI_Irecv( halo_recv_buf_E_host.data() , halo_recv_buf_E_host.size() , data_type , neigh(1,2) , tag0+1 , MPI_COMM_WORLD , &rReq[1] );

        // Copy send buffers to host and send the send buffers
        auto halo_send_buf_W_host = halo_send_buf_W.createHostCopy();
        auto halo_send_buf_E_host = halo_send_buf_E.createHostCopy();
        MPI_Request sReq[2];
        MPI_Isend( halo_send_buf_W_host.data() , halo_send_buf_W_host.size() , data_type , neigh(1,0) , tag0+1 , MPI_COMM_WORLD , &sReq[0] );
        MPI_Isend( halo_send_buf_E_host.data() , halo_send_buf_E_host.size() , data_type , neigh(1,2) , tag0+0 , MPI_COMM_WORLD , &sReq[1] );

        // Wait for sends and receives to complete
        MPI_Status sStat[2];
        MPI_Status rStat[2];
        MPI_Waitall(2, sReq, sStat);
        MPI_Waitall(2, rReq, rStat);

        // Copy receive buffers to device and then copy into halos
        auto halo_recv_buf_W = halo_recv_buf_W_host.createDeviceCopy();
        auto halo_recv_buf_E = halo_recv_buf_E_host.createDeviceCopy();
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_fields,nz+2*hs,ny+2*hs_y,hs,nens) ,
                                          YAKL_LAMBDA (int v, int k, int j, int ii, int iens) {
          fields(v,k,j,      ii,iens) = halo_recv_buf_W(v,k,j,ii,iens);
          fields(v,k,j,nx+hs+ii,iens) = halo_recv_buf_E(v,k,j,ii,iens);
        });

      } // x-direction (east-west)

      if (ny > 1) { // y-direction (north-south)
        // Allocate send buffers and then pack send buffers
        real5d halo_send_buf_S("coupler_halo_send_buf_S",num_fields,nz+2*hs,hs,nx+2*hs,nens);
        real5d halo_send_buf_N("coupler_halo_send_buf_N",num_fields,nz+2*hs,hs,nx+2*hs,nens);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_fields,nz+2*hs,hs,nx+2*hs,nens) ,
                                          YAKL_LAMBDA (int v, int k, int jj, int i, int iens) {
          halo_send_buf_S(v,k,jj,i,iens) = fields(v,k,hs+jj,i,iens);
          halo_send_buf_N(v,k,jj,i,iens) = fields(v,k,ny+jj,i,iens);
        });

        // Allocate host receive buffers and receive the receive buffers
        realHost5d halo_recv_buf_S_host("coupler_halo_recv_buf_S_host",num_fields,nz+2*hs,hs,nx+2*hs,nens);
        realHost5d halo_recv_buf_N_host("coupler_halo_recv_buf_N_host",num_fields,nz+2*hs,hs,nx+2*hs,nens);
        auto data_type = get_mpi_data_type();
        MPI_Request rReq[2];
        MPI_Irecv( halo_recv_buf_S_host.data() , halo_recv_buf_S_host.size() , data_type , neigh(0,1) , tag0+2 , MPI_COMM_WORLD , &rReq[0] );
        MPI_Irecv( halo_recv_buf_N_host.data() , halo_recv_buf_N_host.size() , data_type , neigh(2,1) , tag0+3 , MPI_COMM_WORLD , &rReq[1] );

        // Copy send buffers to host and send the send buffers
        auto halo_send_buf_S_host = halo_send_buf_S.createHostCopy();
        auto halo_send_buf_N_host = halo_send_buf_N.createHostCopy();
        MPI_Request sReq[2];
        MPI_Isend( halo_send_buf_S_host.data() , halo_send_buf_S_host.size() , data_type , neigh(0,1) , tag0+3 , MPI_COMM_WORLD , &sReq[0] );
        MPI_Isend( halo_send_buf_N_host.data() , halo_send_buf_N_host.size() , data_type , neigh(2,1) , tag0+2 , MPI_COMM_WORLD , &sReq[1] );

        // Wait for sends and receives to complete
        MPI_Status sStat[2];
        MPI_Status rStat[2];
        MPI_Waitall(2, sReq, sStat);
        MPI_Waitall(2, rReq, rStat);

        // Copy receive buffers to device and then copy into halos
        auto halo_recv_buf_S = halo_recv_buf_S_host.createDeviceCopy();
        auto halo_recv_buf_N = halo_recv_buf_N_host.createDeviceCopy();
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_fields,nz+2*hs,hs,nx+2*hs,nens) ,
                                          YAKL_LAMBDA (int v, int k, int jj, int i, int iens) {
          fields(v,k,      jj,i,iens) = halo_recv_buf_S(v,k,jj,i,iens);
          fields(v,k,ny+hs+jj,i,iens) = halo_recv_buf_N(v,k,jj,i,iens);
        });

      } // y-direction (north-south)

    }

  };

}


