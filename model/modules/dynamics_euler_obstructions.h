
#pragma once

#include "main_header.h"
#include "MultipleFields.h"
#include "TransformMatrices.h"
#include "WenoLimiter.h"

namespace modules {

  // This clas simplements an A-grid (collocated) cell-centered Finite-Volume method with an upwind Godunov Riemanns
  // solver at cell edges, high-order-accurate reconstruction, Weighted Essentially Non-Oscillatory (WENO) limiting,
  // and a third-order-accurate three-stage Strong Stability Preserving Runge-Kutta time stepping.
  // The dycore prognoses full density, u-, v-, and w-momenta, and mass-weighted potential temperature
  // Since the coupler state is dry density, u-, v-, and w-velocity, and temperature, we need to convert to and from
  // the coupler state.

  class Dynamics_Euler_Stratified_WenoFV {
    public:

    // Order of accuracy (numerical convergence for smooth flows) for the dynamical core
    #ifndef MW_ORD
      int  static constexpr ord = 3;
    #else
      int  static constexpr ord = MW_ORD;
    #endif
    int  static constexpr hs  = (ord-1)/2; // Number of halo cells ("hs" == "halo size")

    int  static constexpr num_state = 5;

    // IDs for the variables in the state vector
    int  static constexpr idR = 0;  // Density
    int  static constexpr idU = 1;  // u-momentum
    int  static constexpr idV = 2;  // v-momentum
    int  static constexpr idW = 3;  // w-momentum
    int  static constexpr idT = 4;  // Density * potential temperature

    // IDs for the test cases
    int  static constexpr DATA_SIMPLE_CITY = 0;

    int  static constexpr MATERIAL_NONE = 0;
    int  static constexpr MATERIAL_WALL = 1;
    int  static constexpr MATERIAL_OPEN = 2;

    int  static constexpr DIR_X = 0;
    int  static constexpr DIR_Y = 1;
    int  static constexpr DIR_Z = 2;

    // Hydrostatic background profiles for density and potential temperature as cell averages and cell edge values
    real        etime;         // Elapsed time
    real        out_freq;      // Frequency out file output
    int         num_out;       // Number of outputs produced thus far
    std::string fname;         // File name for file output
    int         init_data_int; // Integer representation of the type of initial data to use (test case)
    bool        periodic_x, periodic_y, periodic_z;

    // Physical constants
    real        R_d;    // Dry air ideal gas constant
    real        cp_d;   // Specific heat of dry air at constant pressure
    real        cp_v;   // Specific heat of water vapor at constant pressure
    real        p0;     // Reference pressure (Pa); also typically surface pressure for dry simulations
    real        kappa;  // R_d / c_p
    real        gamma;  // cp_d / (cp_d - R_d)
    real        C0;     // pow( R_d * pow( p0 , -kappa ) , gamma )

    bool1d      tracer_adds_mass;  // Whether a tracer adds mass to the full density
    bool1d      tracer_positive;   // Whether a tracer needs to remain non-negative

    SArray<real,1,ord>            gll_pts;          // GLL point locations in domain [-0.5 , 0.5]
    SArray<real,1,ord>            gll_wts;          // GLL weights normalized to sum to 1
    SArray<real,2,ord,ord>        sten_to_coefs;    // Matrix to convert ord stencil avgs to ord poly coefs
    SArray<real,2,ord,2  >        coefs_to_gll;     // Matrix to convert ord poly coefs to two GLL points
    SArray<real,3,hs+1,hs+1,hs+1> weno_recon_lower; // WENO's lower-order reconstruction matrices (sten_to_coefs)
    SArray<real,1,hs+2>           weno_idl;         // Ideal weights for WENO
    real                          weno_sigma;       // WENO sigma parameter (handicap high-order TV estimate)

    int3d material;


    // Compute the maximum stable time step using very conservative assumptions about max wind speed
    real compute_time_step( core::Coupler const &coupler ) const {
      auto dx = coupler.get_dx();
      auto dy = coupler.get_dy();
      auto dz = coupler.get_dz();
      real constexpr maxwave = 350 + 80;
      real cfl = 0.3;
      if (coupler.get_ny_glob() == 1) cfl = 0.5;
      return cfl * std::min( std::min( dx , dy ) , dz ) / maxwave;
    }


    // Perform a single time step using SSPRK3 time stepping
    void time_step(core::Coupler &coupler, real &dt_phys) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;
      using yakl::intrinsics::maxval;
      using yakl::intrinsics::abs;

      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      auto num_tracers = coupler.get_num_tracers();

      // Create arrays to hold state and tracers with halos on the left and right of the domain
      // Cells [0:hs-1] are the left halos, and cells [nx+hs:nx+2*hs-1] are the right halos
      real4d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
      real4d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);

      // Populate the state and tracers arrays using data from the coupler, convert to the dycore's desired state
      convert_coupler_to_dynamics( coupler , state , tracers );

      // Get the max stable time step for the dynamics. dt_phys might be > dt_dyn, meaning we would need to sub-cycle
      real dt_dyn = compute_time_step( coupler );

      // Get the number of sub-cycles we need, and set the dynamics time step accordingly
      int ncycles = (int) std::ceil( dt_phys / dt_dyn );
      dt_dyn = dt_phys / ncycles;

      YAKL_SCOPE( tracer_positive , this->tracer_positive );
      
      for (int icycle = 0; icycle < ncycles; icycle++) {
        // SSPRK3 requires temporary arrays to hold intermediate state and tracers arrays
        real4d state_tmp   ("state_tmp"   ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
        real4d state_tend  ("state_tend"  ,num_state  ,nz     ,ny     ,nx     );
        real4d tracers_tmp ("tracers_tmp" ,num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);
        real4d tracers_tend("tracers_tend",num_tracers,nz     ,ny     ,nx     );
        //////////////
        // Stage 1
        //////////////
        compute_tendencies( coupler , state     , state_tend , tracers     , tracers_tend , dt_dyn );
        // Apply tendencies
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          for (int l = 0; l < num_state  ; l++) {
            state_tmp  (l,hs+k,hs+j,hs+i) = state  (l,hs+k,hs+j,hs+i) + dt_dyn * state_tend  (l,k,j,i);
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers_tmp(l,hs+k,hs+j,hs+i) = tracers(l,hs+k,hs+j,hs+i) + dt_dyn * tracers_tend(l,k,j,i);
            // For machine precision negative values after FCT-enforced positivity application
            if (tracer_positive(l)) {
              tracers_tmp(l,hs+k,hs+j,hs+i) = std::max( 0._fp , tracers_tmp(l,hs+k,hs+j,hs+i) );
            }
          }
        });
        //////////////
        // Stage 2
        //////////////
        compute_tendencies( coupler , state_tmp , state_tend , tracers_tmp , tracers_tend , (1._fp/4._fp) * dt_dyn );
        // Apply tendencies
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          for (int l = 0; l < num_state  ; l++) {
            state_tmp  (l,hs+k,hs+j,hs+i) = (3._fp/4._fp) * state      (l,hs+k,hs+j,hs+i) + 
                                            (1._fp/4._fp) * state_tmp  (l,hs+k,hs+j,hs+i) +
                                            (1._fp/4._fp) * dt_dyn * state_tend  (l,k,j,i);
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers_tmp(l,hs+k,hs+j,hs+i) = (3._fp/4._fp) * tracers    (l,hs+k,hs+j,hs+i) + 
                                            (1._fp/4._fp) * tracers_tmp(l,hs+k,hs+j,hs+i) +
                                            (1._fp/4._fp) * dt_dyn * tracers_tend(l,k,j,i);
            // For machine precision negative values after FCT-enforced positivity application
            if (tracer_positive(l)) {
              tracers_tmp(l,hs+k,hs+j,hs+i) = std::max( 0._fp , tracers_tmp(l,hs+k,hs+j,hs+i) );
            }
          }
        });
        //////////////
        // Stage 3
        //////////////
        compute_tendencies( coupler , state_tmp , state_tend , tracers_tmp , tracers_tend , (2._fp/3._fp) * dt_dyn );
        // Apply tendencies
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          for (int l = 0; l < num_state  ; l++) {
            state      (l,hs+k,hs+j,hs+i) = (1._fp/3._fp) * state      (l,hs+k,hs+j,hs+i) +
                                            (2._fp/3._fp) * state_tmp  (l,hs+k,hs+j,hs+i) +
                                            (2._fp/3._fp) * dt_dyn * state_tend  (l,k,j,i);
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers    (l,hs+k,hs+j,hs+i) = (1._fp/3._fp) * tracers    (l,hs+k,hs+j,hs+i) +
                                            (2._fp/3._fp) * tracers_tmp(l,hs+k,hs+j,hs+i) +
                                            (2._fp/3._fp) * dt_dyn * tracers_tend(l,k,j,i);
            // For machine precision negative values after FCT-enforced positivity application
            if (tracer_positive(l)) {
              tracers    (l,hs+k,hs+j,hs+i) = std::max( 0._fp , tracers    (l,hs+k,hs+j,hs+i) );
            }
          }
        });
      }

      // Convert the dycore's state back to the coupler's state
      convert_dynamics_to_coupler( coupler , state , tracers );

      // Advance the dycore's tracking of total ellapsed time
      etime += dt_phys;
      // Do output and inform the user if it's time to do output
      if (out_freq >= 0. && etime / out_freq >= num_out+1) {
        output( coupler , etime );
        num_out++;
        // Let the user know what the max vertical velocity is to ensure the model hasn't crashed
        auto &dm = coupler.get_data_manager_readonly();
        real maxw_loc = maxval(abs(dm.get_collapsed<real const>("wvel")));
        real maxw;
        auto mpi_data_type = coupler.get_mpi_data_type();
        MPI_Reduce( &maxw_loc , &maxw , 1 , mpi_data_type , MPI_MAX , 0 , MPI_COMM_WORLD );
        if (coupler.is_mainproc()) {
          std::cout << "Etime , dtphys, maxw: " << std::scientific << std::setw(10) << etime   << " , " 
                                                << std::scientific << std::setw(10) << dt_phys << " , "
                                                << std::scientific << std::setw(10) << maxw    << std::endl;
        }
      }
    }


    template <int D>
    YAKL_INLINE static void gather_state_stencil( SArray<real,1,ord> &stencil ,
                                                  int l , int k , int j , int i ,
                                                  int3d const &material ,
                                                  real4d const &state ) {
      bool normal_vel = ( (D==DIR_X) && (l==idU) ) ||
                        ( (D==DIR_Y) && (l==idV) ) ||
                        ( (D==DIR_Z) && (l==idW) ); // Is this variable normal velocity?
      stencil(hs) = state(l,hs+k,hs+j,hs+i);  // Store center stencil value
      { // Traverse left; once you hit a material boundary, apply BC's
        bool hit_material = false;
        real sticky_value;
        for (int s=hs-1; s >= 0; s--) {
          int ind_i, ind_j, ind_k;
          if constexpr (D == DIR_X) { ind_i = i+s ;  ind_j = hs+j;  ind_k = hs+k; }
          if constexpr (D == DIR_Y) { ind_i = hs+i;  ind_j = j+s ;  ind_k = hs+k; }
          if constexpr (D == DIR_Z) { ind_i = hs+i;  ind_j = hs+j;  ind_k = k+s ; }
          int mat = material(ind_k,ind_j,ind_i);
          if (mat == MATERIAL_NONE) {
            stencil(s) = state(l,ind_k,ind_j,ind_i);
          } else { // Must be MATERIAL_OPEN or MATERIAL_WALL
            if (! hit_material) {
              hit_material = true;
              if ( (mat == MATERIAL_WALL) && normal_vel ) { sticky_value = 0; }
              else                                        { sticky_value = stencil(s+1); }
            }
            stencil(s) = sticky_value;
          }
        }
      }
      { // Traverse right; once you hit a material boundary, apply BC's
        bool hit_material = false;
        real sticky_value;
        for (int s=hs+1; s < ord; s++) {
          int ind_i, ind_j, ind_k;
          if constexpr (D == DIR_X) { ind_i = i+s ;  ind_j = hs+j;  ind_k = hs+k; }
          if constexpr (D == DIR_Y) { ind_i = hs+i;  ind_j = j+s ;  ind_k = hs+k; }
          if constexpr (D == DIR_Z) { ind_i = hs+i;  ind_j = hs+j;  ind_k = k+s ; }
          int mat = material(ind_k,ind_j,ind_i);
          if (mat == MATERIAL_NONE) {
            stencil(s) = state(l,ind_k,ind_j,ind_i);
          } else { // Must be MATERIAL_OPEN or MATERIAL_WALL
            if (! hit_material) {
              hit_material = true;
              if ( (mat == MATERIAL_WALL) && normal_vel ) { sticky_value = 0; }
              else                                        { sticky_value = stencil(s-1); }
            }
            stencil(s) = sticky_value;
          }
        }
      }
    }


    template <int D>
    YAKL_INLINE static void set_state_limits( real5d const &state_limits ,
                                              int l , int k , int j , int i ,
                                              int3d const &material ,
                                              SArray<real,1,2> &gll ) {
      bool normal_vel = ( (D==DIR_X) && (l==idU) ) ||
                        ( (D==DIR_Y) && (l==idV) ) ||
                        ( (D==DIR_Z) && (l==idW) ); // Is this variable normal velocity?
      { // Handle the left interface of this cell
        int k_ind, j_ind, i_ind;
        if constexpr (D == DIR_X) { k_ind = k  ;  j_ind = j  ;  i_ind = i-1; }
        if constexpr (D == DIR_Y) { k_ind = k  ;  j_ind = j-1;  i_ind = i  ; }
        if constexpr (D == DIR_Z) { k_ind = k-1;  j_ind = j  ;  i_ind = i  ; }
        int mat = material(hs+k_ind,hs+j_ind,hs+i_ind);
        if ( (mat == MATERIAL_WALL) && normal_vel ) gll(0) = 0;
        state_limits(l,1,k,j,i) = gll(0);
        if ( mat != MATERIAL_NONE ) state_limits(l,0,k,j,i) = gll(0);
      }
      { // Handle the right interface of this cell
        int k_ind, j_ind, i_ind;
        if constexpr (D == DIR_X) { k_ind = k  ;  j_ind = j  ;  i_ind = i+1; }
        if constexpr (D == DIR_Y) { k_ind = k  ;  j_ind = j+1;  i_ind = i  ; }
        if constexpr (D == DIR_Z) { k_ind = k+1;  j_ind = j  ;  i_ind = i  ; }
        int mat = material(hs+k_ind,hs+j_ind,hs+i_ind);
        if ( (mat == MATERIAL_WALL) && normal_vel ) gll(1) = 0;
        state_limits(l,0,k_ind,j_ind,i_ind) = gll(1);
        if ( mat != MATERIAL_NONE ) state_limits(l,1,k_ind,j_ind,i_ind) = gll(1);
      }
    }


    template <int D>
    YAKL_INLINE static void gather_tracer_stencil( SArray<real,1,ord> &stencil ,
                                                   int l , int k , int j , int i ,
                                                   int3d const &material ,
                                                   real4d const &tracers ) {
      stencil(hs) = tracers(l,hs+k,hs+j,hs+i);  // Store center stencil value
      { // Traverse left; once you hit a material boundary, apply BC's
        bool hit_material = false;
        real sticky_value;
        for (int s=hs-1; s >= 0; s--) {
          int ind_i, ind_j, ind_k;
          if constexpr (D == DIR_X) { ind_i = i+s ;  ind_j = hs+j;  ind_k = hs+k; }
          if constexpr (D == DIR_Y) { ind_i = hs+i;  ind_j = j+s ;  ind_k = hs+k; }
          if constexpr (D == DIR_Z) { ind_i = hs+i;  ind_j = hs+j;  ind_k = k+s ; }
          int mat = material(ind_k,ind_j,ind_i);
          if (mat == MATERIAL_NONE) {
            stencil(s) = tracers(l,ind_k,ind_j,ind_i);
          } else { // Must be MATERIAL_OPEN or MATERIAL_WALL
            if (! hit_material) {
              hit_material = true;
              sticky_value = stencil(s+1);
            }
            stencil(s) = sticky_value;
          }
        }
      }
      { // Traverse right; once you hit a material boundary, apply BC's
        bool hit_material = false;
        real sticky_value;
        for (int s=hs+1; s < ord; s++) {
          int ind_i, ind_j, ind_k;
          if constexpr (D == DIR_X) { ind_i = i+s ;  ind_j = hs+j;  ind_k = hs+k; }
          if constexpr (D == DIR_Y) { ind_i = hs+i;  ind_j = j+s ;  ind_k = hs+k; }
          if constexpr (D == DIR_Z) { ind_i = hs+i;  ind_j = hs+j;  ind_k = k+s ; }
          int mat = material(ind_k,ind_j,ind_i);
          if (mat == MATERIAL_NONE) {
            stencil(s) = tracers(l,ind_k,ind_j,ind_i);
          } else { // Must be MATERIAL_OPEN or MATERIAL_WALL
            if (! hit_material) {
              hit_material = true;
              sticky_value = stencil(s-1);
            }
            stencil(s) = sticky_value;
          }
        }
      }
    }


    template <int D>
    YAKL_INLINE static void set_tracer_limits( real5d const &tracers_limits ,
                                               int l , int k , int j , int i ,
                                               int3d const &material ,
                                               SArray<real,1,2> &gll ) {
      { // Handle the left interface of this cell
        int k_ind, j_ind, i_ind;
        if constexpr (D == DIR_X) { k_ind = k  ;  j_ind = j  ;  i_ind = i-1; }
        if constexpr (D == DIR_Y) { k_ind = k  ;  j_ind = j-1;  i_ind = i  ; }
        if constexpr (D == DIR_Z) { k_ind = k-1;  j_ind = j  ;  i_ind = i  ; }
        int mat = material(hs+k_ind,hs+j_ind,hs+i_ind);
        tracers_limits(l,1,k,j,i) = gll(0);
        if ( mat != MATERIAL_NONE ) tracers_limits(l,0,k,j,i) = gll(0);
      }
      { // Handle the right interface of this cell
        int k_ind, j_ind, i_ind;
        if constexpr (D == DIR_X) { k_ind = k  ;  j_ind = j  ;  i_ind = i+1; }
        if constexpr (D == DIR_Y) { k_ind = k  ;  j_ind = j+1;  i_ind = i  ; }
        if constexpr (D == DIR_Z) { k_ind = k+1;  j_ind = j  ;  i_ind = i  ; }
        int mat = material(hs+k_ind,hs+j_ind,hs+i_ind);
        tracers_limits(l,0,k_ind,j_ind,i_ind) = gll(1);
        if ( mat != MATERIAL_NONE ) tracers_limits(l,1,k_ind,j_ind,i_ind) = gll(1);
      }
    }



    // Compute the tendencies for state and tracers for one semi-discretized step inside the RK integrator
    // Tendencies are the time rate of change for a quantity
    // Coupler is non-const because we are writing to the flux variables
    void compute_tendencies( core::Coupler &coupler , real4d const &state   , real4d const &state_tend   ,
                                                      real4d const &tracers , real4d const &tracers_tend , real dt ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;
      using std::min;
      using std::max;

      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      auto dx = coupler.get_dx();
      auto dy = coupler.get_dy();
      auto dz = coupler.get_dz();
      auto sim2d = coupler.is_sim2d();
      auto num_tracers = coupler.get_num_tracers();

      // A slew of things to bring from class scope into local scope so that lambdas copy them by value to the GPU
      YAKL_SCOPE( tracer_positive            , this->tracer_positive            );
      YAKL_SCOPE( coefs_to_gll               , this->coefs_to_gll               );
      YAKL_SCOPE( sten_to_coefs              , this->sten_to_coefs              );
      YAKL_SCOPE( weno_recon_lower           , this->weno_recon_lower           );
      YAKL_SCOPE( weno_idl                   , this->weno_idl                   );
      YAKL_SCOPE( weno_sigma                 , this->weno_sigma                 );
      YAKL_SCOPE( C0                         , this->C0                         );
      YAKL_SCOPE( gamma                      , this->gamma                      );
      YAKL_SCOPE( periodic_z                 , this->periodic_z                 );
      YAKL_SCOPE( material                   , this->material                   );

      halo_exchange( coupler , state , tracers );
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(hs,ny,nx) , YAKL_LAMBDA (int s, int j, int i ) {
        for (int l=0; l < num_state; l++) {
          state(l,      s,hs+j,hs+i) = state(l,      s+nz,hs+j,hs+i);
          state(l,hs+nz+s,hs+j,hs+i) = state(l,hs+nz+s-nz,hs+j,hs+i);
        }
        for (int l=0; l < num_tracers; l++) {
          tracers(l,      s,hs+j,hs+i) = tracers(l,      s+nz,hs+j,hs+i);
          tracers(l,hs+nz+s,hs+j,hs+i) = tracers(l,hs+nz+s-nz,hs+j,hs+i);
        }
      });


      // These arrays store high-order-accurate samples of the state and tracers at cell edges after cell-centered recon
      real5d state_limits_x  ("state_limits_x"  ,num_state  ,2,nz  ,ny  ,nx+1);
      real5d state_limits_y  ("state_limits_y"  ,num_state  ,2,nz  ,ny+1,nx  );
      real5d state_limits_z  ("state_limits_z"  ,num_state  ,2,nz+1,ny  ,nx  );
      real5d tracers_limits_x("tracers_limits_x",num_tracers,2,nz  ,ny  ,nx+1);
      real5d tracers_limits_y("tracers_limits_y",num_tracers,2,nz  ,ny+1,nx  );
      real5d tracers_limits_z("tracers_limits_z",num_tracers,2,nz+1,ny  ,nx  );

      // Compute samples of state and tracers at cell edges using cell-centered reconstructions at high-order with WENO
      // At the end of this, we will have two samples per cell edge in each dimension, one from each adjacent cell.
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i ) {
        if ( material(hs+k,hs+j,hs+i) == MATERIAL_NONE ) { // Don't reconstruct for material cells
          ////////////////////////////////////////////////////////
          // X-direction
          ////////////////////////////////////////////////////////
          // State
          for (int l=0; l < num_state; l++) {
            SArray<real,1,ord> stencil;
            gather_state_stencil<DIR_X>( stencil , l , k , j , i , material , state );
            SArray<real,1,2>   gll;
            reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
            set_state_limits<DIR_X>( state_limits_x , l , k , j , i , material , gll );
          }

          // Tracers
          for (int l=0; l < num_tracers; l++) {
            SArray<real,1,ord> stencil;
            gather_tracer_stencil<DIR_X>( stencil , l , k , j , i , material , tracers );
            SArray<real,1,2>   gll;
            reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
            set_tracer_limits<DIR_X>( tracers_limits_x , l , k , j , i , material , gll );
          }

          ////////////////////////////////////////////////////////
          // Y-direction
          ////////////////////////////////////////////////////////
          // If we're simulating in only 2-D, then do not compute y-direction tendencies
          if (!sim2d) {
            // State
            for (int l=0; l < num_state; l++) {
              SArray<real,1,ord> stencil;
              gather_state_stencil<DIR_Y>( stencil , l , k , j , i , material , state );
              SArray<real,1,2>   gll;
              for (int s=0; s < ord; s++) { stencil(s) = state(l,hs+k,j+s,hs+i); }
              reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
              set_state_limits<DIR_Y>( state_limits_y , l , k , j , i , material , gll );
            }

            // Tracers
            for (int l=0; l < num_tracers; l++) {
              SArray<real,1,ord> stencil;
              gather_tracer_stencil<DIR_Y>( stencil , l , k , j , i , material , tracers );
              SArray<real,1,2>   gll;
              reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
              set_tracer_limits<DIR_Y>( tracers_limits_y , l , k , j , i , material , gll );
            }
          } else {
            for (int l=0; l < num_state; l++) {
              state_limits_y(l,1,k,j  ,i) = 0;
              state_limits_y(l,0,k,j+1,i) = 0;
            }
            for (int l=0; l < num_tracers; l++) {
              tracers_limits_y(l,1,k,j  ,i) = 0;
              tracers_limits_y(l,0,k,j+1,i) = 0;
            }
          }

          ////////////////////////////////////////////////////////
          // Z-direction
          ////////////////////////////////////////////////////////
          // State
          for (int l=0; l < num_state; l++) {
            SArray<real,1,ord> stencil;
            gather_state_stencil<DIR_Z>( stencil , l , k , j , i , material , state );
            SArray<real,1,2>   gll;
            reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
            set_state_limits<DIR_Z>( state_limits_z , l , k , j , i , material , gll );
          }

          // Tracers
          for (int l=0; l < num_tracers; l++) {
            SArray<real,1,ord> stencil;
            gather_tracer_stencil<DIR_Z>( stencil , l , k , j , i , material , tracers );
            SArray<real,1,2>   gll;
            reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
            set_tracer_limits<DIR_Z>( tracers_limits_z , l , k , j , i , material , gll );
          }
        }
      });

      edge_exchange( coupler , state_limits_x , tracers_limits_x , state_limits_y , tracers_limits_y );
      if (periodic_z) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i ) {
          for (int l=0; l < num_state; l++) {
            state_limits_z(l,0,0 ,j,i) = state_limits_z(l,0,nz,j,i);
            state_limits_z(l,1,nz,j,i) = state_limits_z(l,1,0 ,j,i);
          }
          for (int l=0; l < num_tracers; l++) {
            tracers_limits_z(l,0,0 ,j,i) = tracers_limits_z(l,0,nz,j,i);
            tracers_limits_z(l,1,nz,j,i) = tracers_limits_z(l,1,0 ,j,i);
          }
        });
      }


      // The store a single values flux at cell edges
      auto &dm = coupler.get_data_manager_readwrite();
      auto state_flux_x   = dm.get<real,4>("state_flux_x"  );
      auto state_flux_y   = dm.get<real,4>("state_flux_y"  );
      auto state_flux_z   = dm.get<real,4>("state_flux_z"  );
      auto tracers_flux_x = dm.get<real,4>("tracers_flux_x");
      auto tracers_flux_y = dm.get<real,4>("tracers_flux_y");
      auto tracers_flux_z = dm.get<real,4>("tracers_flux_z");

      // Use upwind Riemann solver to reconcile discontinuous limits of state and tracers at each cell edges
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz+1,ny+1,nx+1) , YAKL_LAMBDA (int k, int j, int i ) {
        ////////////////////////////////////////////////////////
        // X-direction
        ////////////////////////////////////////////////////////
        if (j < ny && k < nz) {
          if ( (material(hs+k,hs+j,hs+i-1) == MATERIAL_NONE) || (material(hs+k,hs+j,hs+i) == MATERIAL_NONE) ) {
            // Get left and right state
            real r_L = state_limits_x(idR,0,k,j,i)    ;   real r_R = state_limits_x(idR,1,k,j,i)    ;
            real u_L = state_limits_x(idU,0,k,j,i)/r_L;   real u_R = state_limits_x(idU,1,k,j,i)/r_R;
            real v_L = state_limits_x(idV,0,k,j,i)/r_L;   real v_R = state_limits_x(idV,1,k,j,i)/r_R;
            real w_L = state_limits_x(idW,0,k,j,i)/r_L;   real w_R = state_limits_x(idW,1,k,j,i)/r_R;
            real t_L = state_limits_x(idT,0,k,j,i)/r_L;   real t_R = state_limits_x(idT,1,k,j,i)/r_R;
            // Compute average state
            real r = 0.5_fp * (r_L + r_R);
            real u = 0.5_fp * (u_L + u_R);
            real v = 0.5_fp * (v_L + v_R);
            real w = 0.5_fp * (w_L + w_R);
            real t = 0.5_fp * (t_L + t_R);
            real p = C0 * pow(r*t,gamma);
            real cs2 = gamma*p/r;
            real cs  = sqrt(cs2);

            // COMPUTE UPWIND STATE FLUXES
            // Get left and right fluxes
            real q1_L = state_limits_x(idR,0,k,j,i);   real q1_R = state_limits_x(idR,1,k,j,i);
            real q2_L = state_limits_x(idU,0,k,j,i);   real q2_R = state_limits_x(idU,1,k,j,i);
            real q3_L = state_limits_x(idV,0,k,j,i);   real q3_R = state_limits_x(idV,1,k,j,i);
            real q4_L = state_limits_x(idW,0,k,j,i);   real q4_R = state_limits_x(idW,1,k,j,i);
            real q5_L = state_limits_x(idT,0,k,j,i);   real q5_R = state_limits_x(idT,1,k,j,i);
            // Compute upwind characteristics
            // Waves 1-3, velocity: u
            real w1, w2, w3;
            if (u > 0) {
              w1 = q1_L - q5_L/t;
              w2 = q3_L - v*q5_L/t;
              w3 = q4_L - w*q5_L/t;
            } else {
              w1 = q1_R - q5_R/t;
              w2 = q3_R - v*q5_R/t;
              w3 = q4_R - w*q5_R/t;
            }
            // Wave 5, velocity: u-cs
            real w5 =  u*q1_R/(2*cs) - q2_R/(2*cs) + q5_R/(2*t);
            // Wave 6, velocity: u+cs
            real w6 = -u*q1_L/(2*cs) + q2_L/(2*cs) + q5_L/(2*t);
            // Use right eigenmatrix to compute upwind flux
            real q1 = w1 + w5 + w6;
            real q2 = u*w1 + (u-cs)*w5 + (u+cs)*w6;
            real q3 = w2 + v*w5 + v*w6;
            real q4 = w3 + w*w5 + w*w6;
            real q5 =      t*w5 + t*w6;

            state_flux_x(idR,k,j,i) = q2;
            state_flux_x(idU,k,j,i) = q2*q2/q1 + C0*pow(q5,gamma);
            state_flux_x(idV,k,j,i) = q2*q3/q1;
            state_flux_x(idW,k,j,i) = q2*q4/q1;
            state_flux_x(idT,k,j,i) = q2*q5/q1;

            // COMPUTE UPWIND TRACER FLUXES
            // Handle it one tracer at a time
            for (int tr=0; tr < num_tracers; tr++) {
              if (u > 0) {
                tracers_flux_x(tr,k,j,i) = q2 * tracers_limits_x(tr,0,k,j,i) / r_L;
              } else {
                tracers_flux_x(tr,k,j,i) = q2 * tracers_limits_x(tr,1,k,j,i) / r_R;
              }
            }
          }
        }

        ////////////////////////////////////////////////////////
        // Y-direction
        ////////////////////////////////////////////////////////
        // If we are simulating in 2-D, then do not do Riemann in the y-direction
        if ( (! sim2d) && i < nx && k < nz) {
          if ( (material(hs+k,hs+j-1,hs+i) == MATERIAL_NONE) || (material(hs+k,hs+j,hs+i) == MATERIAL_NONE) ) {
            // Get left and right state
            real r_L = state_limits_y(idR,0,k,j,i)    ;   real r_R = state_limits_y(idR,1,k,j,i)    ;
            real u_L = state_limits_y(idU,0,k,j,i)/r_L;   real u_R = state_limits_y(idU,1,k,j,i)/r_R;
            real v_L = state_limits_y(idV,0,k,j,i)/r_L;   real v_R = state_limits_y(idV,1,k,j,i)/r_R;
            real w_L = state_limits_y(idW,0,k,j,i)/r_L;   real w_R = state_limits_y(idW,1,k,j,i)/r_R;
            real t_L = state_limits_y(idT,0,k,j,i)/r_L;   real t_R = state_limits_y(idT,1,k,j,i)/r_R;
            // Compute average state
            real r = 0.5_fp * (r_L + r_R);
            real u = 0.5_fp * (u_L + u_R);
            real v = 0.5_fp * (v_L + v_R);
            real w = 0.5_fp * (w_L + w_R);
            real t = 0.5_fp * (t_L + t_R);
            real p = C0 * pow(r*t,gamma);
            real cs2 = gamma*p/r;
            real cs  = sqrt(cs2);

            // COMPUTE UPWIND STATE FLUXES
            // Get left and right fluxes
            real q1_L = state_limits_y(idR,0,k,j,i);   real q1_R = state_limits_y(idR,1,k,j,i);
            real q2_L = state_limits_y(idU,0,k,j,i);   real q2_R = state_limits_y(idU,1,k,j,i);
            real q3_L = state_limits_y(idV,0,k,j,i);   real q3_R = state_limits_y(idV,1,k,j,i);
            real q4_L = state_limits_y(idW,0,k,j,i);   real q4_R = state_limits_y(idW,1,k,j,i);
            real q5_L = state_limits_y(idT,0,k,j,i);   real q5_R = state_limits_y(idT,1,k,j,i);
            // Compute upwind characteristics
            // Waves 1-3, velocity: v
            real w1, w2, w3;
            if (v > 0) {
              w1 = q1_L - q5_L/t;
              w2 = q2_L - u*q5_L/t;
              w3 = q4_L - w*q5_L/t;
            } else {
              w1 = q1_R - q5_R/t;
              w2 = q2_R - u*q5_R/t;
              w3 = q4_R - w*q5_R/t;
            }
            // Wave 5, velocity: v-cs
            real w5 =  v*q1_R/(2*cs) - q3_R/(2*cs) + q5_R/(2*t);
            // Wave 6, velocity: v+cs
            real w6 = -v*q1_L/(2*cs) + q3_L/(2*cs) + q5_L/(2*t);
            // Use right eigenmatrix to compute upwind flux
            real q1 = w1 + w5 + w6;
            real q2 = w2 + u*w5 + u*w6;
            real q3 = v*w1 + (v-cs)*w5 + (v+cs)*w6;
            real q4 = w3 + w*w5 + w*w6;
            real q5 =      t*w5 + t*w6;

            state_flux_y(idR,k,j,i) = q3;
            state_flux_y(idU,k,j,i) = q3*q2/q1;
            state_flux_y(idV,k,j,i) = q3*q3/q1 + C0*pow(q5,gamma);
            state_flux_y(idW,k,j,i) = q3*q4/q1;
            state_flux_y(idT,k,j,i) = q3*q5/q1;

            // COMPUTE UPWIND TRACER FLUXES
            // Handle it one tracer at a time
            for (int tr=0; tr < num_tracers; tr++) {
              if (v > 0) {
                tracers_flux_y(tr,k,j,i) = q3 * tracers_limits_y(tr,0,k,j,i) / r_L;
              } else {
                tracers_flux_y(tr,k,j,i) = q3 * tracers_limits_y(tr,1,k,j,i) / r_R;
              }
            }
          }
        } else if (i < nx && k < nz) {
          state_flux_y(idR,k,j,i) = 0;
          state_flux_y(idU,k,j,i) = 0;
          state_flux_y(idV,k,j,i) = 0;
          state_flux_y(idW,k,j,i) = 0;
          state_flux_y(idT,k,j,i) = 0;
          for (int tr=0; tr < num_tracers; tr++) { tracers_flux_y(tr,k,j,i) = 0; }
        }

        ////////////////////////////////////////////////////////
        // Z-direction
        ////////////////////////////////////////////////////////
        if (i < nx && j < ny) {
          if ( (material(hs+k-1,hs+j,hs+i) == MATERIAL_NONE) || (material(hs+k,hs+j,hs+i) == MATERIAL_NONE) ) {
            // Boundary conditions
            if (k == 0) {
              for (int l = 0; l < num_state  ; l++) { state_limits_z  (l,0,0 ,j,i) = state_limits_z  (l,1,0 ,j,i); }
              for (int l = 0; l < num_tracers; l++) { tracers_limits_z(l,0,0 ,j,i) = tracers_limits_z(l,1,0 ,j,i); }
            }
            if (k == nz) {
              for (int l = 0; l < num_state  ; l++) { state_limits_z  (l,1,nz,j,i) = state_limits_z  (l,0,nz,j,i); }
              for (int l = 0; l < num_tracers; l++) { tracers_limits_z(l,1,nz,j,i) = tracers_limits_z(l,0,nz,j,i); }
            }
            // Get left and right state
            real r_L = state_limits_z(idR,0,k,j,i)    ;   real r_R = state_limits_z(idR,1,k,j,i)    ;
            real u_L = state_limits_z(idU,0,k,j,i)/r_L;   real u_R = state_limits_z(idU,1,k,j,i)/r_R;
            real v_L = state_limits_z(idV,0,k,j,i)/r_L;   real v_R = state_limits_z(idV,1,k,j,i)/r_R;
            real w_L = state_limits_z(idW,0,k,j,i)/r_L;   real w_R = state_limits_z(idW,1,k,j,i)/r_R;
            real t_L = state_limits_z(idT,0,k,j,i)/r_L;   real t_R = state_limits_z(idT,1,k,j,i)/r_R;
            // Compute average state
            real r = 0.5_fp * (r_L + r_R);
            real u = 0.5_fp * (u_L + u_R);
            real v = 0.5_fp * (v_L + v_R);
            real w = 0.5_fp * (w_L + w_R);
            real t = 0.5_fp * (t_L + t_R);
            real p = C0 * pow(r*t,gamma);
            real cs2 = gamma*p/r;
            real cs  = sqrt(cs2);
            // Get left and right fluxes
            real q1_L = state_limits_z(idR,0,k,j,i);   real q1_R = state_limits_z(idR,1,k,j,i);
            real q2_L = state_limits_z(idU,0,k,j,i);   real q2_R = state_limits_z(idU,1,k,j,i);
            real q3_L = state_limits_z(idV,0,k,j,i);   real q3_R = state_limits_z(idV,1,k,j,i);
            real q4_L = state_limits_z(idW,0,k,j,i);   real q4_R = state_limits_z(idW,1,k,j,i);
            real q5_L = state_limits_z(idT,0,k,j,i);   real q5_R = state_limits_z(idT,1,k,j,i);
            // Compute upwind characteristics
            // Waves 1-3, velocity: w
            real w1, w2, w3;
            if (w > 0) {
              w1 = q1_L - q5_L/t;
              w2 = q2_L - u*q5_L/t;
              w3 = q3_L - v*q5_L/t;
            } else {
              w1 = q1_R - q5_R/t;
              w2 = q2_R - u*q5_R/t;
              w3 = q3_R - v*q5_R/t;
            }
            // Wave 5, velocity: w-cs
            real w5 =  w*q1_R/(2*cs) - q4_R/(2*cs) + q5_R/(2*t);
            // Wave 6, velocity: w+cs
            real w6 = -w*q1_L/(2*cs) + q4_L/(2*cs) + q5_L/(2*t);
            // Use right eigenmatrix to compute upwind flux
            real q1 = w1 + w5 + w6;
            real q2 = w2 + u*w5 + u*w6;
            real q3 = w3 + v*w5 + v*w6;
            real q4 = w*w1 + (w-cs)*w5 + (w+cs)*w6;
            real q5 =      t*w5 + t*w6;

            state_flux_z(idR,k,j,i) = q4;
            state_flux_z(idU,k,j,i) = q4*q2/q1;
            state_flux_z(idV,k,j,i) = q4*q3/q1;
            state_flux_z(idW,k,j,i) = q4*q4/q1 + C0*pow(q5,gamma);
            state_flux_z(idT,k,j,i) = q4*q5/q1;

            // COMPUTE UPWIND TRACER FLUXES
            // Handle it one tracer at a time
            for (int tr=0; tr < num_tracers; tr++) {
              if (w > 0) {
                tracers_flux_z(tr,k,j,i) = q4 * tracers_limits_z(tr,0,k,j,i) / r_L;
              } else {
                tracers_flux_z(tr,k,j,i) = q4 * tracers_limits_z(tr,1,k,j,i) / r_R;
              }
            }
          }
        }
      });

      // Deallocate state and tracer limits because they are no longer needed
      state_limits_x   = real5d();
      state_limits_y   = real5d();
      state_limits_z   = real5d();
      tracers_limits_x = real5d();
      tracers_limits_y = real5d();
      tracers_limits_z = real5d();

      // Flux Corrected Transport to enforce positivity for tracer species that must remain non-negative
      // This looks like it has a race condition, but it does not. Only one of the adjacent cells can ever change
      // a given edge flux because it's only changed if its sign oriented outward from a cell.
      parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(num_tracers,nz,ny,nx) , YAKL_LAMBDA (int tr, int k, int j, int i ) {
        if (tracer_positive(tr)) {
          real mass_available = max(tracers(tr,hs+k,hs+j,hs+i),0._fp) * dx * dy * dz;
          real flux_out_x = ( max(tracers_flux_x(tr,k,j,i+1),0._fp) - min(tracers_flux_x(tr,k,j,i),0._fp) ) / dx;
          real flux_out_y = ( max(tracers_flux_y(tr,k,j+1,i),0._fp) - min(tracers_flux_y(tr,k,j,i),0._fp) ) / dy;
          real flux_out_z = ( max(tracers_flux_z(tr,k+1,j,i),0._fp) - min(tracers_flux_z(tr,k,j,i),0._fp) ) / dz;
          real mass_out = (flux_out_x + flux_out_y + flux_out_z) * dt * dx * dy * dz;
          if (mass_out > mass_available) {
            real mult = mass_available / mass_out;
            if (tracers_flux_x(tr,k,j,i+1) > 0) tracers_flux_x(tr,k,j,i+1) *= mult;
            if (tracers_flux_x(tr,k,j,i  ) < 0) tracers_flux_x(tr,k,j,i  ) *= mult;
            if (tracers_flux_y(tr,k,j+1,i) > 0) tracers_flux_y(tr,k,j+1,i) *= mult;
            if (tracers_flux_y(tr,k,j  ,i) < 0) tracers_flux_y(tr,k,j  ,i) *= mult;
            if (tracers_flux_z(tr,k+1,j,i) > 0) tracers_flux_z(tr,k+1,j,i) *= mult;
            if (tracers_flux_z(tr,k  ,j,i) < 0) tracers_flux_z(tr,k  ,j,i) *= mult;
          }
        }
      });

      // Compute tendencies as the flux divergence
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        if (material(hs+k,hs+j,hs+i) == MATERIAL_NONE) {
          for (int l = 0; l < num_state; l++) {
            state_tend  (l,k,j,i) = -( state_flux_x  (l,k  ,j  ,i+1) - state_flux_x  (l,k,j,i) ) / dx
                                    -( state_flux_y  (l,k  ,j+1,i  ) - state_flux_y  (l,k,j,i) ) / dy
                                    -( state_flux_z  (l,k+1,j  ,i  ) - state_flux_z  (l,k,j,i) ) / dz;
            if (l == idV && sim2d) state_tend(l,k,j,i) = 0;
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers_tend(l,k,j,i) = -( tracers_flux_x(l,k  ,j  ,i+1) - tracers_flux_x(l,k,j,i) ) / dx
                                    -( tracers_flux_y(l,k  ,j+1,i  ) - tracers_flux_y(l,k,j,i) ) / dy
                                    -( tracers_flux_z(l,k+1,j  ,i  ) - tracers_flux_z(l,k,j,i) ) / dz;
          }
        } else {
          for (int l = 0; l < num_state  ; l++) { state_tend  (l,k,j,i) = 0; }
          for (int l = 0; l < num_tracers; l++) { tracers_tend(l,k,j,i) = 0; }
        }
      });
    }


    // Initialize the class data as well as the state and tracers arrays and convert them back into the coupler state
    void init(core::Coupler &coupler) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      // Set class data from # grid points, grid spacing, domain sizes, whether it's 2-D, and physical constants
      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();

      auto dx    = coupler.get_dx();
      auto dy    = coupler.get_dy();
      auto dz    = coupler.get_dz();

      auto xlen  = coupler.get_xlen();
      auto ylen  = coupler.get_ylen();
      auto zlen  = coupler.get_zlen();

      material = int3d ("material",nz+2*hs,ny+2*hs,nx+2*hs);
      material = MATERIAL_NONE;

      auto &dm = coupler.get_data_manager_readwrite();

      dm.register_and_allocate<real>("density_dry","",{nz,nx,ny});
      dm.register_and_allocate<real>("uvel","",{nz,nx,ny});
      dm.register_and_allocate<real>("vvel","",{nz,nx,ny});
      dm.register_and_allocate<real>("wvel","",{nz,nx,ny});
      dm.register_and_allocate<real>("temp","",{nz,nx,ny});

      coupler.add_tracer( "pollution" , "pollution" , true , true );

      auto sim2d = (coupler.get_ny_glob() == 1);

      R_d   = coupler.get_option<real>("R_d" ,287);
      cp_d  = coupler.get_option<real>("cp_d",1004);
      cp_v  = coupler.get_option<real>("cp_v",1004-287);
      p0    = coupler.get_option<real>("p0"  ,1.e5);
      kappa = R_d / cp_d;
      gamma = cp_d / (cp_d - R_d);
      C0    = pow( R_d * pow( p0 , -kappa ) , gamma );

      // Use TransformMatrices class to create matrices & GLL points to convert degrees of freedom as needed
      TransformMatrices::get_gll_points          (gll_pts         );
      TransformMatrices::get_gll_weights         (gll_wts         );
      TransformMatrices::sten_to_coefs           (sten_to_coefs   );
      TransformMatrices::coefs_to_gll_lower      (coefs_to_gll    );
      TransformMatrices::weno_lower_sten_to_coefs(weno_recon_lower);
      weno::wenoSetIdealSigma<ord>( weno_idl , weno_sigma );

      // Create arrays to determine whether we should add mass for a tracer or whether it should remain non-negative
      auto num_tracers = coupler.get_num_tracers();
      tracer_adds_mass = bool1d("tracer_adds_mass",num_tracers);
      tracer_positive  = bool1d("tracer_positive" ,num_tracers);

      // Must assign on the host to avoid segfaults
      auto tracer_adds_mass_host = tracer_adds_mass.createHostCopy();
      auto tracer_positive_host  = tracer_positive .createHostCopy();

      auto tracer_names = coupler.get_tracer_names();  // Get a list of tracer names
      for (int tr=0; tr < num_tracers; tr++) {
        std::string tracer_desc;
        bool        tracer_found, positive, adds_mass;
        coupler.get_tracer_info( tracer_names[tr] , tracer_desc, tracer_found , positive , adds_mass);
        tracer_positive_host (tr) = positive;
        tracer_adds_mass_host(tr) = adds_mass;
      }
      tracer_positive_host .deep_copy_to(tracer_positive );
      tracer_adds_mass_host.deep_copy_to(tracer_adds_mass);

      auto init_data = coupler.get_option<std::string>("init_data");
      fname          = coupler.get_option<std::string>("out_fname");
      out_freq       = coupler.get_option<real       >("out_freq" );

      dm.register_and_allocate<bool>("tracer_adds_mass","",{num_tracers});
      auto dm_tracer_adds_mass = dm.get<bool,1>("tracer_adds_mass");
      tracer_adds_mass.deep_copy_to(dm_tracer_adds_mass);

      // Set an integer version of the input_data so we can test it inside GPU kernels
      if      (init_data == "simple_city"  ) { init_data_int = DATA_SIMPLE_CITY;   }
      else { endrun("ERROR: Invalid init_data in yaml input file"); }

      etime   = 0;
      num_out = 0;

      // Allocate temp arrays to hold state and tracers before we convert it back to the coupler state
      real4d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
      real4d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);

      // Define quadrature weights and points for 3-point rules
      const int nqpoints = 9;
      SArray<real,1,nqpoints> qpoints;
      SArray<real,1,nqpoints> qweights;
      TransformMatrices::get_gll_points (qpoints );
      TransformMatrices::get_gll_weights(qweights);

      YAKL_SCOPE( init_data_int       , this->init_data_int       );
      YAKL_SCOPE( R_d                 , this->R_d                 );
      YAKL_SCOPE( cp_d                , this->cp_d                );
      YAKL_SCOPE( p0                  , this->p0                  );
      YAKL_SCOPE( gamma               , this->gamma               );
      YAKL_SCOPE( C0                  , this->C0                  );
      YAKL_SCOPE( material            , this->material            );

      auto px      = coupler.get_px();
      auto py      = coupler.get_py();
      auto nproc_x = coupler.get_nproc_x();
      auto nproc_y = coupler.get_nproc_y();
      auto i_beg   = coupler.get_i_beg();
      auto j_beg   = coupler.get_j_beg();
      auto nx_glob = coupler.get_nx_glob();
      auto ny_glob = coupler.get_ny_glob();

      if ( init_data_int == DATA_SIMPLE_CITY ) {
        periodic_x = false;
        periodic_y = true;
        periodic_z = false;

        // Use quadrature to initialize state and tracer data
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          for (int l=0; l < num_state  ; l++) { state  (l,hs+k,hs+j,hs+i) = 0.; }
          for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+i) = 0.; }
          //Use Gauss-Legendre quadrature
          for (int kk=0; kk<nqpoints; kk++) {
            for (int jj=0; jj<nqpoints; jj++) {
              for (int ii=0; ii<nqpoints; ii++) {
                real x = (i+i_beg+0.5)*dx + (qpoints(ii)-0.5)*dx;
                real y = (j+j_beg+0.5)*dy + (qpoints(jj)-0.5)*dy;   if (sim2d) y = ylen/2;
                real z = (k      +0.5)*dz + (qpoints(kk)-0.5)*dz;
                real T = 300;
                real u = 20;
                real v = 0;
                real w = 0;
                real p = p0;
                real rho = p/(R_d*T);
                real theta = std::pow(p/C0,1._fp/gamma)/rho;
                real wt = qweights(ii)*qweights(jj)*qweights(kk);
                state(idR,hs+k,hs+j,hs+i) += rho       * wt;
                state(idU,hs+k,hs+j,hs+i) += rho*u     * wt;
                state(idV,hs+k,hs+j,hs+i) += rho*v     * wt;
                state(idW,hs+k,hs+j,hs+i) += rho*w     * wt;
                state(idT,hs+k,hs+j,hs+i) += rho*theta * wt;
                for (int tr=0; tr < num_tracers; tr++) {
                  tracers(tr,hs+k,hs+j,hs+i) += 0      * wt;
                }
              }
            }
          }
        });
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,hs) , YAKL_LAMBDA (int k, int j, int ii) {
          if (px == 0        ) material(hs+k,hs+j,      ii) = MATERIAL_OPEN;
          if (px == nproc_x-1) material(hs+k,hs+j,hs+nx+ii) = MATERIAL_OPEN;
        });
        // parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,hs,nx) , YAKL_LAMBDA (int k, int jj, int i) {
        //   if (py == 0        ) material(hs+k,      jj,hs+i) = MATERIAL_OPEN;
        //   if (py == nproc_y-1) material(hs+k,hs+ny+jj,hs+i) = MATERIAL_OPEN;
        // });
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(hs,ny,nx) , YAKL_LAMBDA (int kk, int j, int i) {
          material(      kk,hs+j,hs+i) = MATERIAL_WALL;
          material(hs+nz+kk,hs+j,hs+i) = MATERIAL_WALL;
        });

        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          int x0 = 0.3*nx_glob;
          int y0 = ny_glob/2;
          int xr = 0.1*ny_glob;
          int yr = 0.1*ny_glob;
          int ktop = nz / 4;
          if ( (std::abs((int) (j_beg+j-y0)) < yr) &&
               (std::abs((int) (i_beg+i-x0)) < xr) &&
               k < ktop ) {
            material(hs+k,hs+j,hs+i) = MATERIAL_WALL;
          }
        });
      }

      // Convert the initialized state and tracers arrays back to the coupler state
      convert_dynamics_to_coupler( coupler , state , tracers );

      // Output the initial state
      if (out_freq >= 0. ) output( coupler , etime );

      // Register the tracers in the coupler so the user has access if they want (and init to zero)
      dm.register_and_allocate<real>("state_flux_x"  ,"state_flux_x"  ,{num_state  ,nz  ,ny  ,nx+1},{"num_state"  ,"z"  ,"y"  ,"xp1"});
      dm.register_and_allocate<real>("state_flux_y"  ,"state_flux_y"  ,{num_state  ,nz  ,ny+1,nx  },{"num_state"  ,"z"  ,"yp1","x"  });
      dm.register_and_allocate<real>("state_flux_z"  ,"state_flux_z"  ,{num_state  ,nz+1,ny  ,nx  },{"num_state"  ,"zp1","y"  ,"x"  });
      dm.register_and_allocate<real>("tracers_flux_x","tracers_flux_x",{num_tracers,nz  ,ny  ,nx+1},{"num_tracers","z"  ,"y"  ,"xp1"});
      dm.register_and_allocate<real>("tracers_flux_y","tracers_flux_y",{num_tracers,nz  ,ny+1,nx  },{"num_tracers","z"  ,"yp1","x"  });
      dm.register_and_allocate<real>("tracers_flux_z","tracers_flux_z",{num_tracers,nz+1,ny  ,nx  },{"num_tracers","zp1","y"  ,"x"  });
      auto state_flux_x   = dm.get<real,4>("state_flux_x"  );
      auto state_flux_y   = dm.get<real,4>("state_flux_y"  );
      auto state_flux_z   = dm.get<real,4>("state_flux_z"  );
      auto tracers_flux_x = dm.get<real,4>("tracers_flux_x");
      auto tracers_flux_y = dm.get<real,4>("tracers_flux_y");
      auto tracers_flux_z = dm.get<real,4>("tracers_flux_z");
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz+1,ny+1,nx+1) , YAKL_LAMBDA (int k, int j, int i) {
        for (int l=0; l < num_state; l++) {
          if (j < ny && k < nz) state_flux_x(l,k,j,i) = 0;
          if (i < nx && k < nz) state_flux_y(l,k,j,i) = 0;
          if (i < nx && j < ny) state_flux_z(l,k,j,i) = 0;
        }
        for (int l=0; l < num_tracers; l++) {
          if (j < ny && k < nz) tracers_flux_x(l,k,j,i) = 0;
          if (i < nx && k < nz) tracers_flux_y(l,k,j,i) = 0;
          if (i < nx && j < ny) tracers_flux_z(l,k,j,i) = 0;
        }
      });
    }


    // Convert dynamics state and tracers arrays to the coupler state and write to the coupler's data
    void convert_dynamics_to_coupler( core::Coupler &coupler , realConst4d state , realConst4d tracers ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      auto num_tracers = coupler.get_num_tracers();

      auto &dm = coupler.get_data_manager_readwrite();

      // Get state from the coupler
      auto dm_rho_d = dm.get<real,3>("density_dry");
      auto dm_uvel  = dm.get<real,3>("uvel"       );
      auto dm_vvel  = dm.get<real,3>("vvel"       );
      auto dm_wvel  = dm.get<real,3>("wvel"       );
      auto dm_temp  = dm.get<real,3>("temp"       );

      // Get tracers from the coupler
      core::MultiField<real,3> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) {
        dm_tracers.add_field( dm.get<real,3>(tracer_names[tr]) );
      }

      YAKL_SCOPE( R_d                 , this->R_d                 );
      YAKL_SCOPE( gamma               , this->gamma               );
      YAKL_SCOPE( C0                  , this->C0                  );
      YAKL_SCOPE( tracer_adds_mass    , this->tracer_adds_mass    );

      // Convert from state and tracers arrays to the coupler's data
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        real rho   = state(idR,hs+k,hs+j,hs+i);
        real u     = state(idU,hs+k,hs+j,hs+i) / rho;
        real v     = state(idV,hs+k,hs+j,hs+i) / rho;
        real w     = state(idW,hs+k,hs+j,hs+i) / rho;
        real theta = state(idT,hs+k,hs+j,hs+i) / rho;
        real press = C0 * pow( rho*theta , gamma );

        real rho_d = rho;
        for (int tr=0; tr < num_tracers; tr++) {
          if (tracer_adds_mass(tr)) rho_d -= tracers(tr,hs+k,hs+j,hs+i);
        }
        real temp = press / ( rho_d * R_d );

        dm_rho_d(k,j,i) = rho_d;
        dm_uvel (k,j,i) = u;
        dm_vvel (k,j,i) = v;
        dm_wvel (k,j,i) = w;
        dm_temp (k,j,i) = temp;
        for (int tr=0; tr < num_tracers; tr++) {
          dm_tracers(tr,k,j,i) = tracers(tr,hs+k,hs+j,hs+i);
        }
      });
    }


    // Convert coupler's data to state and tracers arrays
    void convert_coupler_to_dynamics( core::Coupler const &coupler , real4d &state , real4d &tracers ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      auto num_tracers = coupler.get_num_tracers();

      auto &dm = coupler.get_data_manager_readonly();

      // Get the coupler's state (as const because it's read-only)
      auto dm_rho_d = dm.get<real const,3>("density_dry");
      auto dm_uvel  = dm.get<real const,3>("uvel"       );
      auto dm_vvel  = dm.get<real const,3>("vvel"       );
      auto dm_wvel  = dm.get<real const,3>("wvel"       );
      auto dm_temp  = dm.get<real const,3>("temp"       );

      // Get the coupler's tracers (as const because it's read-only)
      core::MultiField<real const,3> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) {
        dm_tracers.add_field( dm.get<real const,3>(tracer_names[tr]) );
      }

      YAKL_SCOPE( R_d                 , this->R_d                 );
      YAKL_SCOPE( gamma               , this->gamma               );
      YAKL_SCOPE( C0                  , this->C0                  );
      YAKL_SCOPE( tracer_adds_mass    , this->tracer_adds_mass    );

      // Convert from the coupler's state to the dycore's state and tracers arrays
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        real rho_d = dm_rho_d(k,j,i);
        real u     = dm_uvel (k,j,i);
        real v     = dm_vvel (k,j,i);
        real w     = dm_wvel (k,j,i);
        real temp  = dm_temp (k,j,i);
        real press = rho_d * R_d * temp;

        real rho = rho_d;
        for (int tr=0; tr < num_tracers; tr++) {
          if (tracer_adds_mass(tr)) rho += dm_tracers(tr,k,j,i);
        }
        real theta = pow( press/C0 , 1._fp / gamma ) / rho;

        state(idR,hs+k,hs+j,hs+i) = rho;
        state(idU,hs+k,hs+j,hs+i) = rho * u;
        state(idV,hs+k,hs+j,hs+i) = rho * v;
        state(idW,hs+k,hs+j,hs+i) = rho * w;
        state(idT,hs+k,hs+j,hs+i) = rho * theta;
        for (int tr=0; tr < num_tracers; tr++) {
          tracers(tr,hs+k,hs+j,hs+i) = dm_tracers(tr,k,j,i);
        }
      });
    }


    // Perform file output
    void output( core::Coupler const &coupler , real etime ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;
      yakl::timer_start("output");

      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      auto dx = coupler.get_dx();
      auto dy = coupler.get_dy();
      auto dz = coupler.get_dz();
      auto num_tracers = coupler.get_num_tracers();

      real4d state  ("state"  ,num_state  ,hs+nz,hs+ny,hs+nx);
      real4d tracers("tracers",num_tracers,hs+nz,hs+ny,hs+nx);
      convert_coupler_to_dynamics( coupler , state , tracers );

      yakl::SimplePNetCDF nc;
      MPI_Offset ulIndex = 0; // Unlimited dimension index to place this data at

      int i_beg = coupler.get_i_beg();
      int j_beg = coupler.get_j_beg();

      if (etime == 0) {
        nc.create(fname , NC_CLOBBER | NC_64BIT_DATA);

        nc.create_dim( "x" , coupler.get_nx_glob() );
        nc.create_dim( "y" , coupler.get_ny_glob() );
        nc.create_dim( "z" , nz );
        nc.create_unlim_dim( "t" );

        nc.create_var<real>( "x" , {"x"} );
        nc.create_var<real>( "y" , {"y"} );
        nc.create_var<real>( "z" , {"z"} );
        nc.create_var<real>( "t" , {"t"} );
        nc.create_var<real>( "density_dry" , {"t","z","y","x"} );
        nc.create_var<real>( "uvel"        , {"t","z","y","x"} );
        nc.create_var<real>( "vvel"        , {"t","z","y","x"} );
        nc.create_var<real>( "wvel"        , {"t","z","y","x"} );
        nc.create_var<real>( "temperature" , {"t","z","y","x"} );
        auto tracer_names = coupler.get_tracer_names();
        for (int tr = 0; tr < num_tracers; tr++) {
          nc.create_var<real>( tracer_names[tr] , {"t","z","y","x"} );
        }

        nc.enddef();

        // x-coordinate
        real1d xloc("xloc",nx);
        parallel_for( YAKL_AUTO_LABEL() , nx , YAKL_LAMBDA (int i) { xloc(i) = (i+i_beg+0.5)*dx; });
        nc.write_all( xloc.createHostCopy() , "x" , {i_beg} );

        // y-coordinate
        real1d yloc("yloc",ny);
        parallel_for( YAKL_AUTO_LABEL() , ny , YAKL_LAMBDA (int j) { yloc(j) = (j+j_beg+0.5)*dy; });
        nc.write_all( yloc.createHostCopy() , "y" , {j_beg} );

        // z-coordinate
        real1d zloc("zloc",nz);
        parallel_for( YAKL_AUTO_LABEL() , nz , YAKL_LAMBDA (int k) { zloc(k) = (k      +0.5)*dz; });
        nc.begin_indep_data();
        if (coupler.is_mainproc()) {
          nc.write( zloc.createHostCopy() , "z" );
          nc.write1( 0._fp , "t" , 0 , "t" );
        }
        nc.end_indep_data();

      } else {

        nc.open(fname);
        ulIndex = nc.get_dim_size("t");

        // Write the elapsed time
        nc.begin_indep_data();
        if (coupler.is_mainproc()) {
          nc.write1(etime,"t",ulIndex,"t");
        }
        nc.end_indep_data();

      }

      auto &dm = coupler.get_data_manager_readonly();
      nc.write1_all(dm.get<real const,3>("density_dry").createHostCopy(),"density_dry",ulIndex,{0,j_beg,i_beg},"t");
      nc.write1_all(dm.get<real const,3>("uvel"       ).createHostCopy(),"uvel"       ,ulIndex,{0,j_beg,i_beg},"t");
      nc.write1_all(dm.get<real const,3>("vvel"       ).createHostCopy(),"vvel"       ,ulIndex,{0,j_beg,i_beg},"t");
      nc.write1_all(dm.get<real const,3>("wvel"       ).createHostCopy(),"wvel"       ,ulIndex,{0,j_beg,i_beg},"t");
      nc.write1_all(dm.get<real const,3>("temp"       ).createHostCopy(),"temperature",ulIndex,{0,j_beg,i_beg},"t");
      // Write the tracers to file
      auto tracer_names = coupler.get_tracer_names();
      for (int tr = 0; tr < num_tracers; tr++) {
        nc.write1_all(dm.get<real const,3>(tracer_names[tr]).createHostCopy(),tracer_names[tr],ulIndex,{0,j_beg,i_beg},"t");
      }

      nc.close();
      yakl::timer_stop("output");
    }


    void halo_exchange(core::Coupler const &coupler , real4d const &state , real4d const &tracers) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      auto num_tracers = coupler.get_num_tracers();
      auto sim2d = coupler.is_sim2d();

      int npack = num_state + num_tracers;

      realHost4d halo_send_buf_W_host("halo_send_buf_W_host",npack,nz,ny,hs);
      realHost4d halo_send_buf_E_host("halo_send_buf_E_host",npack,nz,ny,hs);
      realHost4d halo_send_buf_S_host("halo_send_buf_S_host",npack,nz,hs,nx);
      realHost4d halo_send_buf_N_host("halo_send_buf_N_host",npack,nz,hs,nx);
      realHost4d halo_recv_buf_S_host("halo_recv_buf_S_host",npack,nz,hs,nx);
      realHost4d halo_recv_buf_N_host("halo_recv_buf_N_host",npack,nz,hs,nx);
      realHost4d halo_recv_buf_W_host("halo_recv_buf_W_host",npack,nz,ny,hs);
      realHost4d halo_recv_buf_E_host("halo_recv_buf_E_host",npack,nz,ny,hs);

      real4d halo_send_buf_W("halo_send_buf_W",npack,nz,ny,hs);
      real4d halo_send_buf_E("halo_send_buf_E",npack,nz,ny,hs);

      parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(npack,nz,ny,hs) , YAKL_LAMBDA (int v, int k, int j, int ii) {
        if (v < num_state) {
          halo_send_buf_W(v,k,j,ii) = state  (v          ,hs+k,hs+j,hs+ii);
          halo_send_buf_E(v,k,j,ii) = state  (v          ,hs+k,hs+j,nx+ii);
        } else {
          halo_send_buf_W(v,k,j,ii) = tracers(v-num_state,hs+k,hs+j,hs+ii);
          halo_send_buf_E(v,k,j,ii) = tracers(v-num_state,hs+k,hs+j,nx+ii);
        }
      });

      real4d halo_send_buf_S("halo_send_buf_S",npack,nz,hs,nx);
      real4d halo_send_buf_N("halo_send_buf_N",npack,nz,hs,nx);

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(npack,nz,hs,nx) , YAKL_LAMBDA (int v, int k, int jj, int i) {
          if (v < num_state) {
            halo_send_buf_S(v,k,jj,i) = state  (v          ,hs+k,hs+jj,hs+i);
            halo_send_buf_N(v,k,jj,i) = state  (v          ,hs+k,ny+jj,hs+i);
          } else {
            halo_send_buf_S(v,k,jj,i) = tracers(v-num_state,hs+k,hs+jj,hs+i);
            halo_send_buf_N(v,k,jj,i) = tracers(v-num_state,hs+k,ny+jj,hs+i);
          }
        });
      }

      yakl::fence();
      yakl::timer_start("halo_exchange_mpi");

      real4d halo_recv_buf_W("halo_recv_buf_W",npack,nz,ny,hs);
      real4d halo_recv_buf_E("halo_recv_buf_E",npack,nz,ny,hs);
      real4d halo_recv_buf_S("halo_recv_buf_S",npack,nz,hs,nx);
      real4d halo_recv_buf_N("halo_recv_buf_N",npack,nz,hs,nx);

      MPI_Request sReq[4];
      MPI_Request rReq[4];

      auto &neigh = coupler.get_neighbor_rankid_matrix();
      auto mpi_data_type = coupler.get_mpi_data_type();

      //Pre-post the receives
      MPI_Irecv( halo_recv_buf_W_host.data() , npack*nz*ny*hs , mpi_data_type , neigh(1,0) , 0 , MPI_COMM_WORLD , &rReq[0] );
      MPI_Irecv( halo_recv_buf_E_host.data() , npack*nz*ny*hs , mpi_data_type , neigh(1,2) , 1 , MPI_COMM_WORLD , &rReq[1] );
      if (!sim2d) {
        MPI_Irecv( halo_recv_buf_S_host.data() , npack*nz*hs*nx , mpi_data_type , neigh(0,1) , 2 , MPI_COMM_WORLD , &rReq[2] );
        MPI_Irecv( halo_recv_buf_N_host.data() , npack*nz*hs*nx , mpi_data_type , neigh(2,1) , 3 , MPI_COMM_WORLD , &rReq[3] );
      }

      halo_send_buf_W.deep_copy_to(halo_send_buf_W_host);
      halo_send_buf_E.deep_copy_to(halo_send_buf_E_host);
      if (!sim2d) {
        halo_send_buf_S.deep_copy_to(halo_send_buf_S_host);
        halo_send_buf_N.deep_copy_to(halo_send_buf_N_host);
      }

      yakl::fence();

      //Send the data
      MPI_Isend( halo_send_buf_W_host.data() , npack*nz*ny*hs , mpi_data_type , neigh(1,0) , 1 , MPI_COMM_WORLD , &sReq[0] );
      MPI_Isend( halo_send_buf_E_host.data() , npack*nz*ny*hs , mpi_data_type , neigh(1,2) , 0 , MPI_COMM_WORLD , &sReq[1] );
      if (!sim2d) {
        MPI_Isend( halo_send_buf_S_host.data() , npack*nz*hs*nx , mpi_data_type , neigh(0,1) , 3 , MPI_COMM_WORLD , &sReq[2] );
        MPI_Isend( halo_send_buf_N_host.data() , npack*nz*hs*nx , mpi_data_type , neigh(2,1) , 2 , MPI_COMM_WORLD , &sReq[3] );
      }

      MPI_Status  sStat[4];
      MPI_Status  rStat[4];

      //Wait for the sends and receives to finish
      if (sim2d) {
        MPI_Waitall(2, sReq, sStat);
        MPI_Waitall(2, rReq, rStat);
      } else {
        MPI_Waitall(4, sReq, sStat);
        MPI_Waitall(4, rReq, rStat);
      }
      yakl::timer_stop("halo_exchange_mpi");

      halo_recv_buf_W_host.deep_copy_to(halo_recv_buf_W);
      halo_recv_buf_E_host.deep_copy_to(halo_recv_buf_E);
      if (!sim2d) {
        halo_recv_buf_S_host.deep_copy_to(halo_recv_buf_S);
        halo_recv_buf_N_host.deep_copy_to(halo_recv_buf_N);
      }

      parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(npack,nz,ny,hs) , YAKL_LAMBDA (int v, int k, int j, int ii) {
        if (v < num_state) {
          state  (v          ,hs+k,hs+j,      ii) = halo_recv_buf_W(v,k,j,ii);
          state  (v          ,hs+k,hs+j,nx+hs+ii) = halo_recv_buf_E(v,k,j,ii);
        } else {
          tracers(v-num_state,hs+k,hs+j,      ii) = halo_recv_buf_W(v,k,j,ii);
          tracers(v-num_state,hs+k,hs+j,nx+hs+ii) = halo_recv_buf_E(v,k,j,ii);
        }
      });

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(npack,nz,hs,nx) , YAKL_LAMBDA (int v, int k, int jj, int i) {
          if (v < num_state) {
            state  (v          ,hs+k,      jj,hs+i) = halo_recv_buf_S(v,k,jj,i);
            state  (v          ,hs+k,ny+hs+jj,hs+i) = halo_recv_buf_N(v,k,jj,i);
          } else {
            tracers(v-num_state,hs+k,      jj,hs+i) = halo_recv_buf_S(v,k,jj,i);
            tracers(v-num_state,hs+k,ny+hs+jj,hs+i) = halo_recv_buf_N(v,k,jj,i);
          }
        });
      }
    }


    void edge_exchange(core::Coupler const &coupler , real5d const &state_limits_x , real5d const &tracers_limits_x ,
                                                      real5d const &state_limits_y , real5d const &tracers_limits_y ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      auto num_tracers = coupler.get_num_tracers();
      auto sim2d = coupler.is_sim2d();

      int npack = num_state + num_tracers;

      realHost3d edge_send_buf_S_host("edge_send_buf_S_host",num_state+num_tracers,nz,nx);
      realHost3d edge_send_buf_N_host("edge_send_buf_N_host",num_state+num_tracers,nz,nx);
      realHost3d edge_send_buf_W_host("edge_send_buf_W_host",num_state+num_tracers,nz,ny);
      realHost3d edge_send_buf_E_host("edge_send_buf_E_host",num_state+num_tracers,nz,ny);
      realHost3d edge_recv_buf_S_host("edge_recv_buf_S_host",num_state+num_tracers,nz,nx);
      realHost3d edge_recv_buf_N_host("edge_recv_buf_N_host",num_state+num_tracers,nz,nx);
      realHost3d edge_recv_buf_W_host("edge_recv_buf_W_host",num_state+num_tracers,nz,ny);
      realHost3d edge_recv_buf_E_host("edge_recv_buf_E_host",num_state+num_tracers,nz,ny);

      real3d edge_send_buf_W("edge_send_buf_W",npack,nz,ny);
      real3d edge_send_buf_E("edge_send_buf_E",npack,nz,ny);

      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(npack,nz,ny) , YAKL_LAMBDA (int v, int k, int j) {
        if (v < num_state) {
          edge_send_buf_W(v,k,j) = state_limits_x  (v          ,1,k,j,0 );
          edge_send_buf_E(v,k,j) = state_limits_x  (v          ,0,k,j,nx);
        } else {
          edge_send_buf_W(v,k,j) = tracers_limits_x(v-num_state,1,k,j,0 );
          edge_send_buf_E(v,k,j) = tracers_limits_x(v-num_state,0,k,j,nx);
        }
      });

      real3d edge_send_buf_S("edge_send_buf_S",npack,nz,nx);
      real3d edge_send_buf_N("edge_send_buf_N",npack,nz,nx);

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(npack,nz,nx) , YAKL_LAMBDA (int v, int k, int i) {
          if (v < num_state) {
            edge_send_buf_S(v,k,i) = state_limits_y  (v          ,1,k,0 ,i);
            edge_send_buf_N(v,k,i) = state_limits_y  (v          ,0,k,ny,i);
          } else {
            edge_send_buf_S(v,k,i) = tracers_limits_y(v-num_state,1,k,0 ,i);
            edge_send_buf_N(v,k,i) = tracers_limits_y(v-num_state,0,k,ny,i);
          }
        });
      }

      yakl::fence();
      yakl::timer_start("edge_exchange_mpi");

      real3d edge_recv_buf_W("edge_recv_buf_W",npack,nz,ny);
      real3d edge_recv_buf_E("edge_recv_buf_E",npack,nz,ny);
      real3d edge_recv_buf_S("edge_recv_buf_S",npack,nz,nx);
      real3d edge_recv_buf_N("edge_recv_buf_N",npack,nz,nx);

      MPI_Request sReq[4];
      MPI_Request rReq[4];

      auto &neigh = coupler.get_neighbor_rankid_matrix();
      auto mpi_data_type = coupler.get_mpi_data_type();

      //Pre-post the receives
      MPI_Irecv( edge_recv_buf_W_host.data() , npack*nz*ny , mpi_data_type , neigh(1,0) , 4 , MPI_COMM_WORLD , &rReq[0] );
      MPI_Irecv( edge_recv_buf_E_host.data() , npack*nz*ny , mpi_data_type , neigh(1,2) , 5 , MPI_COMM_WORLD , &rReq[1] );
      if (!sim2d) {
        MPI_Irecv( edge_recv_buf_S_host.data() , npack*nz*nx , mpi_data_type , neigh(0,1) , 6 , MPI_COMM_WORLD , &rReq[2] );
        MPI_Irecv( edge_recv_buf_N_host.data() , npack*nz*nx , mpi_data_type , neigh(2,1) , 7 , MPI_COMM_WORLD , &rReq[3] );
      }

      edge_send_buf_W.deep_copy_to(edge_send_buf_W_host);
      edge_send_buf_E.deep_copy_to(edge_send_buf_E_host);
      if (!sim2d) {
        edge_send_buf_S.deep_copy_to(edge_send_buf_S_host);
        edge_send_buf_N.deep_copy_to(edge_send_buf_N_host);
      }

      yakl::fence();

      //Send the data
      MPI_Isend( edge_send_buf_W_host.data() , npack*nz*ny , mpi_data_type , neigh(1,0) , 5 , MPI_COMM_WORLD , &sReq[0] );
      MPI_Isend( edge_send_buf_E_host.data() , npack*nz*ny , mpi_data_type , neigh(1,2) , 4 , MPI_COMM_WORLD , &sReq[1] );
      if (!sim2d) {
        MPI_Isend( edge_send_buf_S_host.data() , npack*nz*nx , mpi_data_type , neigh(0,1) , 7 , MPI_COMM_WORLD , &sReq[2] );
        MPI_Isend( edge_send_buf_N_host.data() , npack*nz*nx , mpi_data_type , neigh(2,1) , 6 , MPI_COMM_WORLD , &sReq[3] );
      }

      MPI_Status  sStat[4];
      MPI_Status  rStat[4];

      //Wait for the sends and receives to finish
      if (sim2d) {
        MPI_Waitall(2, sReq, sStat);
        MPI_Waitall(2, rReq, rStat);
      } else {
        MPI_Waitall(4, sReq, sStat);
        MPI_Waitall(4, rReq, rStat);
      }
      yakl::timer_stop("edge_exchange_mpi");

      edge_recv_buf_W_host.deep_copy_to(edge_recv_buf_W);
      edge_recv_buf_E_host.deep_copy_to(edge_recv_buf_E);
      if (!sim2d) {
        edge_recv_buf_S_host.deep_copy_to(edge_recv_buf_S);
        edge_recv_buf_N_host.deep_copy_to(edge_recv_buf_N);
      }

      auto px   = coupler.get_px();
      auto py   = coupler.get_py();
      auto nproc_x = coupler.get_nproc_x();
      auto nproc_y = coupler.get_nproc_y();
      YAKL_SCOPE( periodic_x , this->periodic_x );
      YAKL_SCOPE( periodic_y , this->periodic_y );

      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(npack,nz,ny) , YAKL_LAMBDA (int v, int k, int j) {
        if (v < num_state) {
          if ( periodic_x || (px > 0        ) ) state_limits_x  (v          ,0,k,j,0 ) = edge_recv_buf_W(v,k,j);
          if ( periodic_x || (px < nproc_x-1) ) state_limits_x  (v          ,1,k,j,nx) = edge_recv_buf_E(v,k,j);
        } else {
          if ( periodic_x || (px > 0        ) ) tracers_limits_x(v-num_state,0,k,j,0 ) = edge_recv_buf_W(v,k,j);
          if ( periodic_x || (px < nproc_x-1) ) tracers_limits_x(v-num_state,1,k,j,nx) = edge_recv_buf_E(v,k,j);
        }
      });

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(npack,nz,nx) , YAKL_LAMBDA (int v, int k, int i) {
          if (v < num_state) {
            if ( periodic_y || (py > 0        ) ) state_limits_y  (v          ,0,k,0 ,i) = edge_recv_buf_S(v,k,i);
            if ( periodic_y || (py < nproc_y-1) ) state_limits_y  (v          ,1,k,ny,i) = edge_recv_buf_N(v,k,i);
          } else {
            if ( periodic_y || (py > 0        ) ) tracers_limits_y(v-num_state,0,k,0 ,i) = edge_recv_buf_S(v,k,i);
            if ( periodic_y || (py < nproc_y-1) ) tracers_limits_y(v-num_state,1,k,ny,i) = edge_recv_buf_N(v,k,i);
          }
        });
      }
    }


    // ord stencil cell averages to two GLL point values via high-order reconstruction and WENO limiting
    YAKL_INLINE static void reconstruct_gll_values( SArray<real,1,ord> const stencil                      ,
                                                    SArray<real,1,2> &gll                                 ,
                                                    SArray<real,2,ord,2> const &coefs_to_gll              ,
                                                    SArray<real,2,ord,ord>  const &sten_to_coefs          ,
                                                    SArray<real,3,hs+1,hs+1,hs+1> const &weno_recon_lower ,
                                                    SArray<real,1,hs+2> const &idl                        ,
                                                    real sigma ) {
      // Reconstruct values
      SArray<real,1,ord> wenoCoefs;
      weno::compute_weno_coefs<ord>( weno_recon_lower , sten_to_coefs , stencil , wenoCoefs , idl , sigma );
      // Transform ord weno coefficients into 2 GLL points
      for (int ii=0; ii<2; ii++) {
        real tmp = 0;
        for (int s=0; s < ord; s++) {
          tmp += coefs_to_gll(s,ii) * wenoCoefs(s);
        }
        gll(ii) = tmp;
      }
    }

  };

}


