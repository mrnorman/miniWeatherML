
#pragma once

#include "main_header.h"
#include "MultipleFields.h"
#include "TransformMatrices.h"
#include "WenoLimiter.h"

namespace modules {


  class Dynamics_compr_nh_energy {
    public:

    // Order of accuracy (numerical convergence for smooth flows) for the dynamical core
    #ifndef MW_ORD
      int  static constexpr ord = 5;
    #else
      int  static constexpr ord = MW_ORD;
    #endif
    int  static constexpr hs  = (ord-1)/2; // Number of halo cells ("hs" == "halo size")

    int  static constexpr num_state = 5;

    // IDs for the variables in the state vector
    int  static constexpr idR = 0;  // rho
    int  static constexpr idU = 1;  // rho*u
    int  static constexpr idV = 2;  // rho*v
    int  static constexpr idW = 3;  // rho*w
    int  static constexpr idT = 4;  // rho*energy

    // IDs for the test cases
    int  static constexpr DATA_THERMAL   = 0;
    int  static constexpr DATA_SUPERCELL = 1;

    // Hydrostatic background profiles for density and potential temperature as cell averages and cell edge values
    real1d      hy_dens_cells;
    real1d      hy_pressure_cells;
    real1d      hy_dens_edges;
    real1d      hy_pressure_edges;
    real        etime;         // Elapsed time
    real        out_freq;      // Frequency out file output
    int         num_out;       // Number of outputs produced thus far
    std::string fname;         // File name for file output
    int         init_data_int; // Integer representation of the type of initial data to use (test case)

    int         nx  , ny  , nz  ;  // # cells in each dimension
    real        dx  , dy  , dz  ;  // grid spacing in each dimension
    real        xlen, ylen, zlen;  // length of domain in each dimension
    bool        sim2d;             // Whether we're simulating in 2-D

    // Physical constants
    real        R_d;    // Dry air ideal gas constant
    real        R_v;    // Water vapor ideal gas constant
    real        cp_d;   // Specific heat of dry air at constant pressure
    real        cp_v;   // Specific heat of water vapor at constant pressure
    real        cv_d;   // Specific heat of dry air at constant volume
    real        cv_v;   // Specific heat of water vapor at constant volume
    real        p0;     // Reference pressure (Pa); also typically surface pressure for dry simulations
    real        grav;   // Acceleration due to gravity
    real        kappa;  // R_d / c_p
    real        gamma;  // cp_d / (cp_d - R_d)
    real        C0;     // pow( R_d * pow( p0 , -kappa ) , gamma )

    int         num_tracers;       // Number of tracers we are using
    int         idWV;              // Index number for water vapor in the tracers array
    bool1d      tracer_adds_mass;  // Whether a tracer adds mass to the full density
    bool1d      tracer_positive;   // Whether a tracer needs to remain non-negative

    SArray<real,1,ord>            gll_pts;          // GLL point locations in domain [-0.5 , 0.5]
    SArray<real,1,ord>            gll_wts;          // GLL weights normalized to sum to 1
    SArray<real,2,ord,ord>        sten_to_coefs;    // Matrix to convert ord stencil avgs to ord poly coefs
    SArray<real,2,ord,2  >        coefs_to_gll;     // Matrix to convert ord poly coefs to two GLL points
    SArray<real,3,hs+1,hs+1,hs+1> weno_recon_lower; // WENO's lower-order reconstruction matrices (sten_to_coefs)
    SArray<real,1,hs+2>           weno_idl;         // Ideal weights for WENO
    real                          weno_sigma;       // WENO sigma parameter (handicap high-order TV estimate)

    realHost4d halo_send_buf_S_host;
    realHost4d halo_send_buf_N_host;
    realHost4d halo_send_buf_W_host;
    realHost4d halo_send_buf_E_host;
    realHost4d halo_recv_buf_S_host;
    realHost4d halo_recv_buf_N_host;
    realHost4d halo_recv_buf_W_host;
    realHost4d halo_recv_buf_E_host;

    realHost3d edge_send_buf_S_host;
    realHost3d edge_send_buf_N_host;
    realHost3d edge_send_buf_W_host;
    realHost3d edge_send_buf_E_host;
    realHost3d edge_recv_buf_S_host;
    realHost3d edge_recv_buf_N_host;
    realHost3d edge_recv_buf_W_host;
    realHost3d edge_recv_buf_E_host;


    // Compute the maximum stable time step using very conservative assumptions about max wind speed
    real compute_time_step( core::Coupler const &coupler ) const {
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

      YAKL_SCOPE( num_tracers     , this->num_tracers     );
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
        MPI_Reduce( &maxw_loc , &maxw , 1 , MPI_DOUBLE , MPI_MAX , 0 , MPI_COMM_WORLD );
        if (coupler.is_mainproc()) {
          std::cout << "Etime , dtphys, maxw: " << std::scientific << std::setw(10) << etime   << " , " 
                                                << std::scientific << std::setw(10) << dt_phys << " , "
                                                << std::scientific << std::setw(10) << maxw    << std::endl;
        }
      }
    }


    // Compute the tendencies for state and tracers for one semi-discretized step inside the RK integrator
    // Tendencies are the time rate of change for a quantity
    // Coupler is non-const because we are writing to the flux variables
    void compute_tendencies( core::Coupler &coupler , real4d const &state   , real4d const &state_tend  ,
                                                      real4d const &tracers , real4d const &tracer_tend , real dt ) const {
      state_tend  = 0;
      tracer_tend = 0;
      using yakl::c::parallel_for;
      using yakl::c::Bounds;
      using std::min;
      using std::max;

      // A slew of things to bring from class scope into local scope so that lambdas copy them by value to the GPU
      YAKL_SCOPE( hy_dens_cells              , this->hy_dens_cells              );
      YAKL_SCOPE( hy_pressure_cells          , this->hy_pressure_cells          );
      YAKL_SCOPE( hy_dens_edges              , this->hy_dens_edges              );
      YAKL_SCOPE( hy_pressure_edges          , this->hy_pressure_edges          );
      YAKL_SCOPE( num_tracers                , this->num_tracers                );
      YAKL_SCOPE( tracer_positive            , this->tracer_positive            );
      YAKL_SCOPE( coefs_to_gll               , this->coefs_to_gll               );
      YAKL_SCOPE( sten_to_coefs              , this->sten_to_coefs              );
      YAKL_SCOPE( weno_recon_lower           , this->weno_recon_lower           );
      YAKL_SCOPE( weno_idl                   , this->weno_idl                   );
      YAKL_SCOPE( weno_sigma                 , this->weno_sigma                 );
      YAKL_SCOPE( nx                         , this->nx                         );
      YAKL_SCOPE( ny                         , this->ny                         );
      YAKL_SCOPE( nz                         , this->nz                         );
      YAKL_SCOPE( dx                         , this->dx                         );
      YAKL_SCOPE( dy                         , this->dy                         );
      YAKL_SCOPE( dz                         , this->dz                         );
      YAKL_SCOPE( sim2d                      , this->sim2d                      );
      YAKL_SCOPE( C0                         , this->C0                         );
      YAKL_SCOPE( gamma                      , this->gamma                      );
      YAKL_SCOPE( grav                       , this->grav                       );

      halo_exchange( coupler , state , tracers );

      real3d pressure("pressure",nz+2*hs,ny+2*hs,nx+2*hs);

      // Divide all mass weighted quantities by density for better reconstruction
      // Compute perturbation pressure in all cells
      // Fill in the top and bottom boundaries with free-slip solid wall BC's
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny+2*hs,nx+2*hs) , YAKL_LAMBDA (int k, int j, int i ) {
        real rho_v = tracers(idWV,hs+k,j,i);
        real rho_d = state(idR,hs+k,j,i);
        for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho_d -= tracers(tr,hs+k,j,i); }

        state(idU,hs+k,j,i) /= state(idR,hs+k,j,i);
        state(idV,hs+k,j,i) /= state(idR,hs+k,j,i);
        state(idW,hs+k,j,i) /= state(idR,hs+k,j,i);
        state(idT,hs+k,j,i) /= state(idR,hs+k,j,i);
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,j,i) /= state(idR,hs+k,j,i); }

        real r = state(idR,hs+k,j,i);
        real u = state(idU,hs+k,j,i);
        real v = state(idV,hs+k,j,i);
        real w = state(idW,hs+k,j,i);
        real e = state(idT,hs+k,j,i);
        real z = (k+0.5)*dz;

        real temp = ( e - (u*u+v*v+w*w)/2 - grav*z ) / (rho_d/r*cv_d+rho_v/r*cv_v);
        pressure(hs+k,j,i) = (rho_d*R_d + rho_v*R_v)*temp - hy_pressure_cells(k);

        state(idR,hs+k,j,i) -= hy_dens_cells(k);

        if (k == 0) {
          for (int kk=0; kk < hs; kk++) {
            state(idR,kk,j,i) = state(idR,hs,j,i);
            state(idU,kk,j,i) = state(idU,hs,j,i);
            state(idV,kk,j,i) = state(idV,hs,j,i);
            state(idW,kk,j,i) = 0;
            state(idT,kk,j,i) = state(idT,hs,j,i);
            pressure (kk,j,i) = pressure (hs,j,i);
            for (int tr=0; tr < num_tracers; tr++) { tracers(tr,kk,j,i) = tracers(tr,hs,j,i); }
          }
        }
        if (k == nz-1) {
          for (int kk=hs+nz; kk < nz+2*hs; kk++) {
            state(idR,kk,j,i) = state(idR,hs+nz-1,j,i);
            state(idU,kk,j,i) = state(idU,hs+nz-1,j,i);
            state(idV,kk,j,i) = state(idV,hs+nz-1,j,i);
            state(idW,kk,j,i) = 0;
            state(idT,kk,j,i) = state(idT,hs+nz-1,j,i);
            pressure (kk,j,i) = pressure (hs+nz-1,j,i);
            for (int tr=0; tr < num_tracers; tr++) { tracers(tr,kk,j,i) = tracers(tr,hs+nz-1,j,i); }
          }
        }
      });

      // These arrays store high-order-accurate samples of the state and tracers at cell edges after cell-centered recon
      real5d state_limits_x ("state_limits_x" ,2,num_state  ,nz,ny,nx+1);
      real5d tracer_limits_x("tracer_limits_x",2,num_tracers,nz,ny,nx+1);
      real5d state_limits_y ("state_limits_y" ,2,num_state  ,nz,ny+1,nx);
      real5d tracer_limits_y("tracer_limits_y",2,num_tracers,nz,ny+1,nx);
      real5d state_limits_z ("state_limits_z" ,2,num_state  ,nz+1,ny,nx);
      real5d tracer_limits_z("tracer_limits_z",2,num_tracers,nz+1,ny,nx);

      // Compute samples of state and tracers at cell edges using cell-centered reconstructions at high-order with WENO
      // At the end of this, we will have two samples per cell edge in each dimension, one from each adjacent cell.
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i ) {
        ///////////
        // Density
        ///////////
        {
          yakl::SArray<real,1,2> gll;
          yakl::SArray<real,1,ord> stencil;
          // x-direction
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idR,hs+k,hs+j,i+ii); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          state_limits_x(1,idR,k,j,i  ) = gll(0) + hy_dens_cells(k);
          state_limits_x(0,idR,k,j,i+1) = gll(1) + hy_dens_cells(k);
          // y-direction
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idR,hs+k,j+jj,hs+i); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          state_limits_y(1,idR,k,j  ,i) = gll(0) + hy_dens_cells(k);
          state_limits_y(0,idR,k,j+1,i) = gll(1) + hy_dens_cells(k);
          // z-direction
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idR,k+kk,hs+j,hs+i); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          state_limits_z(1,idR,k  ,j,i) = gll(0) + hy_dens_edges(k  );
          state_limits_z(0,idR,k+1,j,i) = gll(1) + hy_dens_edges(k+1);
          if (k == 0   ) state_limits_z(0,idR,k  ,j,i) = state_limits_z(1,idR,k  ,j,i);
          if (k == nz-1) state_limits_z(1,idR,k+1,j,i) = state_limits_z(0,idR,k+1,j,i);
        }

        ///////////
        // uvel
        ///////////
        {
          yakl::SArray<real,1,2> gll;
          yakl::SArray<real,1,ord> stencil;
          // x-direction
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idU,hs+k,hs+j,i+ii); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          state_limits_x(1,idU,k,j,i  ) = gll(0);
          state_limits_x(0,idU,k,j,i+1) = gll(1);
          // y-direction
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idU,hs+k,j+jj,hs+i); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          state_limits_y(1,idU,k,j  ,i) = gll(0);
          state_limits_y(0,idU,k,j+1,i) = gll(1);
          // z-direction
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idU,k+kk,hs+j,hs+i); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          state_limits_z(1,idU,k  ,j,i) = gll(0);
          state_limits_z(0,idU,k+1,j,i) = gll(1);
          if (k == 0   ) state_limits_z(0,idU,k  ,j,i) = state_limits_z(1,idU,k  ,j,i);
          if (k == nz-1) state_limits_z(1,idU,k+1,j,i) = state_limits_z(0,idU,k+1,j,i);
        }

        ///////////
        // vvel
        ///////////
        if (! sim2d) {
          yakl::SArray<real,1,2> gll;
          yakl::SArray<real,1,ord> stencil;
          // x-direction
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idV,hs+k,hs+j,i+ii); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          state_limits_x(1,idV,k,j,i  ) = gll(0);
          state_limits_x(0,idV,k,j,i+1) = gll(1);
          // y-direction
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idV,hs+k,j+jj,hs+i); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          state_limits_y(1,idV,k,j  ,i) = gll(0);
          state_limits_y(0,idV,k,j+1,i) = gll(1);
          // z-direction
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idV,k+kk,hs+j,hs+i); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          state_limits_z(1,idV,k  ,j,i) = gll(0);
          state_limits_z(0,idV,k+1,j,i) = gll(1);
          if (k == 0   ) state_limits_z(0,idV,k  ,j,i) = state_limits_z(1,idV,k  ,j,i);
          if (k == nz-1) state_limits_z(1,idV,k+1,j,i) = state_limits_z(0,idV,k+1,j,i);
        } else {
          state_limits_x(1,idV,k,j,i  ) = 0;
          state_limits_x(0,idV,k,j,i+1) = 0;
          state_limits_y(1,idV,k,j  ,i) = 0;
          state_limits_y(0,idV,k,j+1,i) = 0;
          state_limits_z(1,idV,k  ,j,i) = 0;
          state_limits_z(0,idV,k+1,j,i) = 0;
          if (k == 0   ) state_limits_z(0,idV,k  ,j,i) = 0;
          if (k == nz-1) state_limits_z(1,idV,k+1,j,i) = 0;
        }

        ///////////
        // wvel
        ///////////
        {
          yakl::SArray<real,1,2> gll;
          yakl::SArray<real,1,ord> stencil;
          // x-direction
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idW,hs+k,hs+j,i+ii); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          state_limits_x(1,idW,k,j,i  ) = gll(0);
          state_limits_x(0,idW,k,j,i+1) = gll(1);
          // y-direction
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idW,hs+k,j+jj,hs+i); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          state_limits_y(1,idW,k,j  ,i) = gll(0);
          state_limits_y(0,idW,k,j+1,i) = gll(1);
          // z-direction
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idW,k+kk,hs+j,hs+i); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          state_limits_z(1,idW,k  ,j,i) = gll(0);
          state_limits_z(0,idW,k+1,j,i) = gll(1);
          if (k == 0   ) { state_limits_z(0,idW,k  ,j,i) = 0;   state_limits_z(1,idW,k  ,j,i) = 0; }
          if (k == nz-1) { state_limits_z(0,idW,k+1,j,i) = 0;   state_limits_z(1,idW,k+1,j,i) = 0; }
        }

        ///////////
        // pressure
        ///////////
        {
          yakl::SArray<real,1,2> gll;
          yakl::SArray<real,1,ord> stencil;
          // x-direction
          for (int ii=0; ii < ord; ii++) { stencil(ii) = pressure(hs+k,hs+j,i+ii); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          state_limits_x(1,idT,k,j,i  ) = gll(0) + hy_pressure_cells(k);
          state_limits_x(0,idT,k,j,i+1) = gll(1) + hy_pressure_cells(k);
          // y-direction
          for (int jj=0; jj < ord; jj++) { stencil(jj) = pressure(hs+k,j+jj,hs+i); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          state_limits_y(1,idT,k,j  ,i) = gll(0) + hy_pressure_cells(k);
          state_limits_y(0,idT,k,j+1,i) = gll(1) + hy_pressure_cells(k);
          // z-direction
          for (int kk=0; kk < ord; kk++) { stencil(kk) = pressure(k+kk,hs+j,hs+i); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          state_limits_z(1,idT,k  ,j,i) = gll(0) + hy_pressure_edges(k  );
          state_limits_z(0,idT,k+1,j,i) = gll(1) + hy_pressure_edges(k+1);
          if (k == 0   ) state_limits_z(0,idT,k  ,j,i) = state_limits_z(1,idT,k  ,j,i);
          if (k == nz-1) state_limits_z(1,idT,k+1,j,i) = state_limits_z(0,idT,k+1,j,i);
        }

        ///////////
        // tracers
        ///////////
        // x-direction
        for (int tr=0; tr < num_tracers; tr++) {
          yakl::SArray<real,1,2> gll;
          yakl::SArray<real,1,ord> stencil;
          for (int ii=0; ii < ord; ii++) { stencil(ii) = tracers(tr,hs+k,hs+j,i+ii); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          tracer_limits_x(1,tr,k,j,i  ) = gll(0) * state_limits_x(1,idR,k,j,i  );
          tracer_limits_x(0,tr,k,j,i+1) = gll(1) * state_limits_x(0,idR,k,j,i+1);
          // y-direction
          for (int jj=0; jj < ord; jj++) { stencil(jj) = tracers(tr,hs+k,j+jj,hs+i); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          tracer_limits_y(1,tr,k,j  ,i) = gll(0) * state_limits_y(1,idR,k,j  ,i);
          tracer_limits_y(0,tr,k,j+1,i) = gll(1) * state_limits_y(0,idR,k,j+1,i);
          // z-direction
          for (int kk=0; kk < ord; kk++) { stencil(kk) = tracers(tr,k+kk,hs+j,hs+i); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          tracer_limits_z(1,tr,k  ,j,i) = gll(0) * state_limits_z(1,idR,k  ,j,i);
          tracer_limits_z(0,tr,k+1,j,i) = gll(1) * state_limits_z(0,idR,k+1,j,i);
          if (k == 0   ) tracer_limits_z(0,tr,k  ,j,i) = tracer_limits_z(1,tr,k  ,j,i);
          if (k == nz-1) tracer_limits_z(1,tr,k+1,j,i) = tracer_limits_z(0,tr,k+1,j,i);
        }
      });

      edge_exchange( coupler , state_limits_x , tracer_limits_x , state_limits_y , tracer_limits_y );

      // The store a single values flux at cell edges
      auto &dm = coupler.get_data_manager_readwrite();
      auto state_flux_x  = dm.get<real,4>("state_flux_x" );
      auto state_flux_y  = dm.get<real,4>("state_flux_y" );
      auto state_flux_z  = dm.get<real,4>("state_flux_z" );
      auto tracer_flux_x = dm.get<real,4>("tracers_flux_x");
      auto tracer_flux_y = dm.get<real,4>("tracers_flux_y");
      auto tracer_flux_z = dm.get<real,4>("tracers_flux_z");

      // Use upwind Riemann solver to reconcile discontinuous limits of state and tracers at each cell edges
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz+1,ny+1,nx+1) , YAKL_LAMBDA (int k, int j, int i ) {
        ////////////////////////////////////////////////////////
        // X-direction
        ////////////////////////////////////////////////////////
        if (j < ny && k < nz) {
          // Get left and right state
          real r_L = state_limits_x(0,idR,k,j,i);     real r_R = state_limits_x(1,idR,k,j,i);
          real u_L = state_limits_x(0,idU,k,j,i);     real u_R = state_limits_x(1,idU,k,j,i);
          real v_L = state_limits_x(0,idV,k,j,i);     real v_R = state_limits_x(1,idV,k,j,i);
          real w_L = state_limits_x(0,idW,k,j,i);     real w_R = state_limits_x(1,idW,k,j,i);
          real p_L = state_limits_x(0,idT,k,j,i);     real p_R = state_limits_x(1,idT,k,j,i);
          real e_L                              ;     real e_R                              ;
          // Compute e_L
          {
            real rho_d = r_L;
            for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho_d -= tracer_limits_x(0,tr,k,j,i); }
            real rho_v = tracer_limits_x(0,idWV,k,j,i);
            real T = p_L/r_L/(rho_d/r_L*R_d+rho_v/r_L*R_v);
            real z = (k+0.5_fp)*dz;
            e_L = (rho_d/r_L*cv_d+rho_v/r_L*cv_v)*T + (u_L*u_L+v_L*v_L+w_L*w_L)/2 + grav*z;
          }
          // Compute e_R
          {
            real rho_d = r_R;
            for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho_d -= tracer_limits_x(1,tr,k,j,i); }
            real rho_v = tracer_limits_x(1,idWV,k,j,i);
            real T = p_R/r_R/(rho_d/r_R*R_d+rho_v/r_R*R_v);
            real z = (k+0.5_fp)*dz;
            e_R = (rho_d/r_R*cv_d+rho_v/r_R*cv_v)*T + (u_R*u_R+v_R*v_R+w_R*w_R)/2 + grav*z;
          }
          // Compute average state
          real r = 0.5_fp * (r_L + r_R);
          real u = 0.5_fp * (u_L + u_R);
          real v = 0.5_fp * (v_L + v_R);
          real w = 0.5_fp * (w_L + w_R);
          real p = 0.5_fp * (p_L + p_R);
          real e = 0.5_fp * (e_L + e_R);
          real cs2 = gamma*p/r;
          real cs  = sqrt(cs2);

          // Rename variables & convert to conserved state variables: r, r*u, r*v, r*w, r*e, p
          auto &q1_L=r_L;  auto &q2_L=u_L;  auto &q3_L=v_L;  auto &q4_L=w_L;  auto &q5_L=e_L;  auto &q6_L=p_L;
          auto &q1_R=r_R;  auto &q2_R=u_R;  auto &q3_R=v_R;  auto &q4_R=w_R;  auto &q5_R=e_R;  auto &q6_R=p_R;
          q2_L *= r_L;  q3_L *= r_L;  q4_L *= r_L;  q5_L *= r_L;
          q2_R *= r_R;  q3_R *= r_R;  q4_R *= r_R;  q5_R *= r_R;

          // Waves 1-4 (u)
          real q1, q2, q3, q4, q5, q6;
          if (u > 0) { q1=q1_L;  q2=q2_L;  q3=q3_L;  q4=q4_L;  q5=q5_L;  q6=q6_L; }
          else       { q1=q1_R;  q2=q2_R;  q3=q3_R;  q4=q4_R;  q5=q5_R;  q6=q6_R; }
          real w1 = q1 - q6/cs2;
          real w2 = q3 - q6*v/cs2;
          real w3 = q4 - q6*w/cs2;
          real w4 = q1*u*u - q2*u + q5 - q6*(cs2+e*gamma)/(cs2*gamma);
          // Wave 5 (u-cs)
          real w5 =  q1_R*u/(2*cs) - q2_R/(2*cs) + q6_R/(2*cs2);
          // Wave 6 (u+cs)
          real w6 = -q1_L*u/(2*cs) + q2_L/(2*cs) + q6_L/(2*cs2);

          q1 = w1            + w5                                + w6                               ;
          q2 = w1*u          + w5*(u-cs)                         + w6*(u+cs)                        ;
          q3 =      w2       + w5*v                              + w6*v                             ;
          q4 =         w3    + w5*w                              + w6*w                             ;
          q5 =            w4 - w5*(cs*gamma*u-cs2-e*gamma)/gamma + w6*(cs*gamma*u+cs2+e*gamma)/gamma;
          q6 =                 w5*cs2                            + w6*cs2                           ;

          r = q1;
          u = q2/r;
          v = q3/r;
          w = q4/r;
          e = q5/r;
          p = q6;

          state_flux_x(idR,k,j,i) = r*u;
          state_flux_x(idU,k,j,i) = r*u*u+p;
          state_flux_x(idV,k,j,i) = r*u*v;
          state_flux_x(idW,k,j,i) = r*u*w;
          state_flux_x(idT,k,j,i) = r*u*e+u*p;
          for (int tr=0; tr < num_tracers; tr++) {
            if (u > 0) { tracer_flux_x(tr,k,j,i) = r*u*tracer_limits_x(0,tr,k,j,i)/r_L; }
            else       { tracer_flux_x(tr,k,j,i) = r*u*tracer_limits_x(1,tr,k,j,i)/r_R; }
          }
        }

        ////////////////////////////////////////////////////////
        // Y-direction
        ////////////////////////////////////////////////////////
        if (i < nx && k < nz) {
          if (! sim2d) {
            // Get left and right state
            real r_L = state_limits_y(0,idR,k,j,i);     real r_R = state_limits_y(1,idR,k,j,i);
            real u_L = state_limits_y(0,idU,k,j,i);     real u_R = state_limits_y(1,idU,k,j,i);
            real v_L = state_limits_y(0,idV,k,j,i);     real v_R = state_limits_y(1,idV,k,j,i);
            real w_L = state_limits_y(0,idW,k,j,i);     real w_R = state_limits_y(1,idW,k,j,i);
            real p_L = state_limits_y(0,idT,k,j,i);     real p_R = state_limits_y(1,idT,k,j,i);
            real e_L                              ;     real e_R                              ;
            // Compute e_L
            {
              real rho_d = r_L;
              for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho_d -= tracer_limits_y(0,tr,k,j,i); }
              real rho_v = tracer_limits_y(0,idWV,k,j,i);
              real T = p_L/r_L/(rho_d/r_L*R_d+rho_v/r_L*R_v);
              real z = (k+0.5_fp)*dz;
              e_L = (rho_d/r_L*cv_d+rho_v/r_L*cv_v)*T + (u_L*u_L+v_L*v_L+w_L*w_L)/2 + grav*z;
            }
            // Compute e_R
            {
              real rho_d = r_R;
              for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho_d -= tracer_limits_y(1,tr,k,j,i); }
              real rho_v = tracer_limits_y(1,idWV,k,j,i);
              real T = p_R/r_R/(rho_d/r_R*R_d+rho_v/r_R*R_v);
              real z = (k+0.5_fp)*dz;
              e_R = (rho_d/r_R*cv_d+rho_v/r_R*cv_v)*T + (u_R*u_R+v_R*v_R+w_R*w_R)/2 + grav*z;
            }
            // Compute average state
            real r = 0.5_fp * (r_L + r_R);
            real u = 0.5_fp * (u_L + u_R);
            real v = 0.5_fp * (v_L + v_R);
            real w = 0.5_fp * (w_L + w_R);
            real p = 0.5_fp * (p_L + p_R);
            real e = 0.5_fp * (e_L + e_R);
            real cs2 = gamma*p/r;
            real cs  = sqrt(cs2);

            // Rename variables & convert to conserved state variables: r, r*u, r*v, r*w, r*e, p
            auto &q1_L=r_L;  auto &q2_L=u_L;  auto &q3_L=v_L;  auto &q4_L=w_L;  auto &q5_L=e_L;  auto &q6_L=p_L;
            auto &q1_R=r_R;  auto &q2_R=u_R;  auto &q3_R=v_R;  auto &q4_R=w_R;  auto &q5_R=e_R;  auto &q6_R=p_R;
            q2_L *= r_L;  q3_L *= r_L;  q4_L *= r_L;  q5_L *= r_L;
            q2_R *= r_R;  q3_R *= r_R;  q4_R *= r_R;  q5_R *= r_R;

            // Waves 1-4 (v)
            real q1, q2, q3, q4, q5, q6;
            if (v > 0) { q1=q1_L;  q2=q2_L;  q3=q3_L;  q4=q4_L;  q5=q5_L;  q6=q6_L; }
            else       { q1=q1_R;  q2=q2_R;  q3=q3_R;  q4=q4_R;  q5=q5_R;  q6=q6_R; }
            real w1 = q1 - q6/cs2;
            real w2 = q2 - q6*u/cs2;
            real w3 = q4 - q6*w/cs2;
            real w4 = q1*v*v - q3*v + q5 - q6*(cs2+e*gamma)/(cs2*gamma);
            // Wave 5 (v-cs)
            real w5 =  q1_R*v/(2*cs) - q3_R/(2*cs) + q6_R/(2*cs2);
            // Wave 6 (v+cs)
            real w6 = -q1_L*v/(2*cs) + q3_L/(2*cs) + q6_L/(2*cs2);

            q1 = w1            + w5                                + w6                               ;
            q2 =      w2       + w5*u                              + w6*u                             ;
            q3 = w1*v          + w5*(v-cs)                         + w6*(v+cs)                        ;
            q4 =         w3    + w5*w                              + w6*w                             ;
            q5 =            w4 - w5*(cs*gamma*v-cs2-e*gamma)/gamma + w6*(cs*gamma*v+cs2+e*gamma)/gamma;
            q6 =                 w5*cs2                            + w6*cs2                           ;

            r = q1;
            u = q2/r;
            v = q3/r;
            w = q4/r;
            e = q5/r;
            p = q6;

            state_flux_y(idR,k,j,i) = r*v;
            state_flux_y(idU,k,j,i) = r*v*u;
            state_flux_y(idV,k,j,i) = r*v*v+p;
            state_flux_y(idW,k,j,i) = r*v*w;
            state_flux_y(idT,k,j,i) = r*v*e+v*p;
            for (int tr=0; tr < num_tracers; tr++) {
              if (v > 0) { tracer_flux_y(tr,k,j,i) = r*v*tracer_limits_y(0,tr,k,j,i)/r_L; }
              else       { tracer_flux_y(tr,k,j,i) = r*v*tracer_limits_y(1,tr,k,j,i)/r_R; }
            }
          } else {
            state_flux_y(idR,k,j,i) = 0;
            state_flux_y(idU,k,j,i) = 0;
            state_flux_y(idV,k,j,i) = 0;
            state_flux_y(idW,k,j,i) = 0;
            state_flux_y(idT,k,j,i) = 0;
            for (int tr=0; tr < num_tracers; tr++) { tracer_flux_y(tr,k,j,i) = 0; }
          }
        }

        ////////////////////////////////////////////////////////
        // Z-direction
        ////////////////////////////////////////////////////////
        if (i < nx && j < ny) {
          // Get left and right state
          real r_L = state_limits_z(0,idR,k,j,i);     real r_R = state_limits_z(1,idR,k,j,i);
          real u_L = state_limits_z(0,idU,k,j,i);     real u_R = state_limits_z(1,idU,k,j,i);
          real v_L = state_limits_z(0,idV,k,j,i);     real v_R = state_limits_z(1,idV,k,j,i);
          real w_L = state_limits_z(0,idW,k,j,i);     real w_R = state_limits_z(1,idW,k,j,i);
          real p_L = state_limits_z(0,idT,k,j,i);     real p_R = state_limits_z(1,idT,k,j,i);
          real e_L                              ;     real e_R                              ;
          // Compute e_L
          {
            real rho_d = r_L;
            for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho_d -= tracer_limits_z(0,tr,k,j,i); }
            real rho_v = tracer_limits_z(0,idWV,k,j,i);
            real T = p_L/r_L/(rho_d/r_L*R_d+rho_v/r_L*R_v);
            real z = k*dz;
            e_L = (rho_d/r_L*cv_d+rho_v/r_L*cv_v)*T + (u_L*u_L+v_L*v_L+w_L*w_L)/2 + grav*z;
          }
          // Compute e_R
          {
            real rho_d = r_R;
            for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho_d -= tracer_limits_z(1,tr,k,j,i); }
            real rho_v = tracer_limits_z(1,idWV,k,j,i);
            real T = p_R/r_R/(rho_d/r_R*R_d+rho_v/r_R*R_v);
            real z = k*dz;
            e_R = (rho_d/r_R*cv_d+rho_v/r_R*cv_v)*T + (u_R*u_R+v_R*v_R+w_R*w_R)/2 + grav*z;
          }
          // Compute average state
          real r = 0.5_fp * (r_L + r_R);
          real u = 0.5_fp * (u_L + u_R);
          real v = 0.5_fp * (v_L + v_R);
          real w = 0.5_fp * (w_L + w_R);
          real p = 0.5_fp * (p_L + p_R);
          real e = 0.5_fp * (e_L + e_R);
          real cs2 = gamma*p/r;
          real cs  = sqrt(cs2);

          // Rename variables & convert to conserved state variables: r, r*u, r*v, r*w, r*e, p
          auto &q1_L=r_L;  auto &q2_L=u_L;  auto &q3_L=v_L;  auto &q4_L=w_L;  auto &q5_L=e_L;  auto &q6_L=p_L;
          auto &q1_R=r_R;  auto &q2_R=u_R;  auto &q3_R=v_R;  auto &q4_R=w_R;  auto &q5_R=e_R;  auto &q6_R=p_R;
          q2_L *= r_L;  q3_L *= r_L;  q4_L *= r_L;  q5_L *= r_L;
          q2_R *= r_R;  q3_R *= r_R;  q4_R *= r_R;  q5_R *= r_R;

          // Waves 1-4 (w)
          real q1, q2, q3, q4, q5, q6;
          if (w > 0) { q1=q1_L;  q2=q2_L;  q3=q3_L;  q4=q4_L;  q5=q5_L;  q6=q6_L; }
          else       { q1=q1_R;  q2=q2_R;  q3=q3_R;  q4=q4_R;  q5=q5_R;  q6=q6_R; }
          real w1 = q1 - q6/cs2;
          real w2 = q2 - q6*u/cs2;
          real w3 = q3 - q6*v/cs2;
          real w4 = q1*w*w - q4*w + q5 - q6*(cs2+e*gamma)/(cs2*gamma);
          // Wave 5 (w-cs)
          real w5 =  q1_R*w/(2*cs) - q4_R/(2*cs) + q6_R/(2*cs2);
          // Wave 6 (w+cs)
          real w6 = -q1_L*w/(2*cs) + q4_L/(2*cs) + q6_L/(2*cs2);

          q1 = w1            + w5                                + w6                               ;
          q2 =      w2       + w5*u                              + w6*u                             ;
          q3 =         w3    + w5*v                              + w6*v                             ;
          q4 = w1*w          + w5*(w-cs)                         + w6*(w+cs)                        ;
          q5 =            w4 - w5*(cs*gamma*w-cs2-e*gamma)/gamma + w6*(cs*gamma*w+cs2+e*gamma)/gamma;
          q6 =                 w5*cs2                            + w6*cs2                           ;

          r = q1;
          u = q2/r;
          v = q3/r;
          w = q4/r;
          e = q5/r;
          p = q6;

          state_flux_z(idR,k,j,i) = r*w;
          state_flux_z(idU,k,j,i) = r*w*u;
          state_flux_z(idV,k,j,i) = r*w*v;
          state_flux_z(idW,k,j,i) = r*w*w+p;
          state_flux_z(idT,k,j,i) = r*w*e+w*p;
          for (int tr=0; tr < num_tracers; tr++) {
            if (w > 0) { tracer_flux_z(tr,k,j,i) = r*w*tracer_limits_z(0,tr,k,j,i)/r_L; }
            else       { tracer_flux_z(tr,k,j,i) = r*w*tracer_limits_z(1,tr,k,j,i)/r_R; }
          }
        }

        ////////////////////////////////////////////////////////////////////////////////////////
        // Add hydrostatsis back to density, and re-multiply the density to other variables
        ////////////////////////////////////////////////////////////////////////////////////////
        if (k < nz && j < ny && i < nx) {
          state(idR,hs+k,hs+j,hs+i) += hy_dens_cells(k);
          state(idU,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i);
          state(idV,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i);
          state(idW,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i);
          state(idT,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i);
          for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i); }
        }
      });

      // Deallocate state and tracer limits because they are no longer needed
      state_limits_x  = real5d();
      state_limits_y  = real5d();
      state_limits_z  = real5d();
      tracer_limits_x = real5d();
      tracer_limits_y = real5d();
      tracer_limits_z = real5d();

      // Flux Corrected Transport to enforce positivity for tracer species that must remain non-negative
      // This looks like it has a race condition, but it does not. Only one of the adjacent cells can ever change
      // a given edge flux because it's only changed if its sign oriented outward from a cell.
      parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(num_tracers,nz,ny,nx) , YAKL_LAMBDA (int tr, int k, int j, int i ) {
        if (tracer_positive(tr)) {
          real mass_available = max(tracers(tr,hs+k,hs+j,hs+i),0._fp) * dx * dy * dz;
          real flux_out_x = ( max(tracer_flux_x(tr,k,j,i+1),0._fp) - min(tracer_flux_x(tr,k,j,i),0._fp) ) / dx;
          real flux_out_y = ( max(tracer_flux_y(tr,k,j+1,i),0._fp) - min(tracer_flux_y(tr,k,j,i),0._fp) ) / dy;
          real flux_out_z = ( max(tracer_flux_z(tr,k+1,j,i),0._fp) - min(tracer_flux_z(tr,k,j,i),0._fp) ) / dz;
          real mass_out = (flux_out_x + flux_out_y + flux_out_z) * dt * dx * dy * dz;
          if (mass_out > mass_available) {
            real mult = mass_available / mass_out;
            if (tracer_flux_x(tr,k,j,i+1) > 0) tracer_flux_x(tr,k,j,i+1) *= mult;
            if (tracer_flux_x(tr,k,j,i  ) < 0) tracer_flux_x(tr,k,j,i  ) *= mult;
            if (tracer_flux_y(tr,k,j+1,i) > 0) tracer_flux_y(tr,k,j+1,i) *= mult;
            if (tracer_flux_y(tr,k,j  ,i) < 0) tracer_flux_y(tr,k,j  ,i) *= mult;
            if (tracer_flux_z(tr,k+1,j,i) > 0) tracer_flux_z(tr,k+1,j,i) *= mult;
            if (tracer_flux_z(tr,k  ,j,i) < 0) tracer_flux_z(tr,k  ,j,i) *= mult;
          }
        }
      });

      // Compute tendencies as the flux divergence + gravity source term
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        for (int l = 0; l < num_state; l++) {
          state_tend(l,k,j,i) = -( state_flux_x  (l,k  ,j  ,i+1) - state_flux_x  (l,k,j,i) ) / dx
                                -( state_flux_y  (l,k  ,j+1,i  ) - state_flux_y  (l,k,j,i) ) / dy
                                -( state_flux_z  (l,k+1,j  ,i  ) - state_flux_z  (l,k,j,i) ) / dz;
          if (l == idW) state_tend(l,k,j,i) += -grav * state(idR,hs+k,hs+j,hs+i);
          if (l == idV && sim2d) state_tend(l,k,j,i) = 0;
        }
        for (int l = 0; l < num_tracers; l++) {
          tracer_tend(l,k,j,i) = -( tracer_flux_x(l,k  ,j  ,i+1) - tracer_flux_x(l,k,j,i) ) / dx
                                 -( tracer_flux_y(l,k  ,j+1,i  ) - tracer_flux_y(l,k,j,i) ) / dy
                                 -( tracer_flux_z(l,k+1,j  ,i  ) - tracer_flux_z(l,k,j,i) ) / dz;
        }
      });
    }


    // Initialize the class data as well as the state and tracers arrays and convert them back into the coupler state
    void init(core::Coupler &coupler) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      // Set class data from # grid points, grid spacing, domain sizes, whether it's 2-D, and physical constants
      nx    = coupler.get_nx();
      ny    = coupler.get_ny();
      nz    = coupler.get_nz();

      dx    = coupler.get_dx();
      dy    = coupler.get_dy();
      dz    = coupler.get_dz();

      xlen  = coupler.get_xlen();
      ylen  = coupler.get_ylen();
      zlen  = coupler.get_zlen();

      sim2d = (coupler.get_ny_glob() == 1);

      R_d   = coupler.get_option<real>("R_d" ,287.042);
      R_v   = coupler.get_option<real>("R_v" ,461.505);
      cp_d  = coupler.get_option<real>("cp_d",1004.64);
      cp_v  = coupler.get_option<real>("cp_v",1859   );
      p0    = coupler.get_option<real>("p0"  ,1.e5   );
      grav  = coupler.get_option<real>("grav",9.80616);
      cv_d  = cp_d - R_d;
      cv_v  = cp_v - R_v;
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
      num_tracers = coupler.get_num_tracers();
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
        if (tracer_names[tr] == "water_vapor") idWV = tr;  // Be sure to track which index belongs to water vapor
      }
      tracer_positive_host .deep_copy_to(tracer_positive );
      tracer_adds_mass_host.deep_copy_to(tracer_adds_mass);

      auto init_data = coupler.get_option<std::string>("init_data");
      fname          = coupler.get_option<std::string>("out_fname");
      out_freq       = coupler.get_option<real       >("out_freq" );

      // Set an integer version of the input_data so we can test it inside GPU kernels
      if      (init_data == "thermal"  ) { init_data_int = DATA_THERMAL;   }
      else if (init_data == "supercell") { init_data_int = DATA_SUPERCELL; }
      else { endrun("ERROR: Invalid init_data in yaml input file"); }

      etime   = 0;
      num_out = 0;

      // Allocate temp arrays to hold state and tracers before we convert it back to the coupler state
      real4d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
      real4d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);

      // Allocate arrays for hydrostatic background states
      hy_dens_cells       = real1d("hy_dens_cells"      ,nz  );
      hy_pressure_cells   = real1d("hy_pressure_cells"  ,nz  );
      hy_dens_edges       = real1d("hy_dens_edges"      ,nz+1);
      hy_pressure_edges   = real1d("hy_pressure_edges"  ,nz+1);

      if (init_data_int == DATA_SUPERCELL) {

        init_supercell( coupler , state , tracers );

      } else {

        // Define quadrature weights and points for 3-point rules
        const int nqpoints = 9;
        SArray<real,1,nqpoints> qpoints, qweights;

        TransformMatrices::get_gll_points(qpoints);
        TransformMatrices::get_gll_weights(qweights);

        YAKL_SCOPE( init_data_int       , this->init_data_int       );
        YAKL_SCOPE( hy_dens_cells       , this->hy_dens_cells       );
        YAKL_SCOPE( hy_pressure_cells   , this->hy_pressure_cells   );
        YAKL_SCOPE( hy_dens_edges       , this->hy_dens_edges       );
        YAKL_SCOPE( hy_pressure_edges   , this->hy_pressure_edges   );
        YAKL_SCOPE( dx                  , this->dx                  );
        YAKL_SCOPE( dy                  , this->dy                  );
        YAKL_SCOPE( dz                  , this->dz                  );
        YAKL_SCOPE( xlen                , this->xlen                );
        YAKL_SCOPE( ylen                , this->ylen                );
        YAKL_SCOPE( sim2d               , this->sim2d               );
        YAKL_SCOPE( R_d                 , this->R_d                 );
        YAKL_SCOPE( R_v                 , this->R_v                 );
        YAKL_SCOPE( cv_d                , this->cv_d                );
        YAKL_SCOPE( cv_v                , this->cv_v                );
        YAKL_SCOPE( cp_d                , this->cp_d                );
        YAKL_SCOPE( p0                  , this->p0                  );
        YAKL_SCOPE( grav                , this->grav                );
        YAKL_SCOPE( gamma               , this->gamma               );
        YAKL_SCOPE( C0                  , this->C0                  );
        YAKL_SCOPE( num_state           , this->num_state           );
        YAKL_SCOPE( num_tracers         , this->num_tracers         );
        YAKL_SCOPE( idWV                , this->idWV                );

        size_t i_beg = coupler.get_i_beg();
        size_t j_beg = coupler.get_j_beg();

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
                real rho, u, v, w, temp, p, rho_v, hr, ht;

                if (init_data_int == DATA_THERMAL) {
                  thermal(x,y,z,xlen,ylen,grav,C0,gamma,cp_d,p0,R_d,R_v,rho,u,v,w,temp,p,rho_v,hr,ht);
                }
                real rho_d = rho - rho_v;
                real e = (rho_d/rho*cv_d+rho_v/rho*cv_v)*temp + (u*u+v*v+w*w)/2 + grav*z;

                if (sim2d) v = 0;

                real wt = qweights(ii)*qweights(jj)*qweights(kk);
                state  (idR ,hs+k,hs+j,hs+i) += rho   * wt;
                state  (idU ,hs+k,hs+j,hs+i) += rho*u * wt;
                state  (idV ,hs+k,hs+j,hs+i) += rho*v * wt;
                state  (idW ,hs+k,hs+j,hs+i) += rho*w * wt;
                state  (idT ,hs+k,hs+j,hs+i) += rho*e * wt;
                tracers(idWV,hs+k,hs+j,hs+i) += rho_v * wt;
              }
            }
          }
        });


        // Compute hydrostatic background cell averages using quadrature
        parallel_for( YAKL_AUTO_LABEL() , Bounds<1>(nz) , YAKL_LAMBDA (int k) {
          hy_dens_cells    (k) = 0.;
          hy_pressure_cells(k) = 0.;
          for (int kk=0; kk<nqpoints; kk++) {
            real z = (k+0.5)*dz + (qpoints(kk)-0.5)*dz;
            real hr, ht;

            if (init_data_int == DATA_THERMAL) { hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht); }
            real p_d  = C0 * pow( hr*ht , gamma );

            hy_dens_cells    (k) += hr  * qweights(kk);
            hy_pressure_cells(k) += p_d * qweights(kk);
          }
        });

        // Compute hydrostatic background cell edge values
        parallel_for( YAKL_AUTO_LABEL() , Bounds<1>(nz+1) , YAKL_LAMBDA (int k) {
          real z = k*dz;
          real hr, ht;

          if (init_data_int == DATA_THERMAL) { hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht); }
          real p_d  = C0 * pow( hr*ht , gamma );

          hy_dens_edges    (k) = hr ;
          hy_pressure_edges(k) = p_d;
        });

      }

      halo_send_buf_W_host = realHost4d("halo_send_buf_W_host",num_state+num_tracers,nz,ny,hs);
      halo_send_buf_E_host = realHost4d("halo_send_buf_E_host",num_state+num_tracers,nz,ny,hs);
      halo_send_buf_S_host = realHost4d("halo_send_buf_S_host",num_state+num_tracers,nz,hs,nx);
      halo_send_buf_N_host = realHost4d("halo_send_buf_N_host",num_state+num_tracers,nz,hs,nx);
      halo_recv_buf_S_host = realHost4d("halo_recv_buf_S_host",num_state+num_tracers,nz,hs,nx);
      halo_recv_buf_N_host = realHost4d("halo_recv_buf_N_host",num_state+num_tracers,nz,hs,nx);
      halo_recv_buf_W_host = realHost4d("halo_recv_buf_W_host",num_state+num_tracers,nz,ny,hs);
      halo_recv_buf_E_host = realHost4d("halo_recv_buf_E_host",num_state+num_tracers,nz,ny,hs);

      edge_send_buf_S_host = realHost3d("edge_send_buf_S_host",num_state+num_tracers,nz,nx);
      edge_send_buf_N_host = realHost3d("edge_send_buf_N_host",num_state+num_tracers,nz,nx);
      edge_send_buf_W_host = realHost3d("edge_send_buf_W_host",num_state+num_tracers,nz,ny);
      edge_send_buf_E_host = realHost3d("edge_send_buf_E_host",num_state+num_tracers,nz,ny);
      edge_recv_buf_S_host = realHost3d("edge_recv_buf_S_host",num_state+num_tracers,nz,nx);
      edge_recv_buf_N_host = realHost3d("edge_recv_buf_N_host",num_state+num_tracers,nz,nx);
      edge_recv_buf_W_host = realHost3d("edge_recv_buf_W_host",num_state+num_tracers,nz,ny);
      edge_recv_buf_E_host = realHost3d("edge_recv_buf_E_host",num_state+num_tracers,nz,ny);

      // Convert the initialized state and tracers arrays back to the coupler state
      convert_dynamics_to_coupler( coupler , state , tracers );

      // Output the initial state
      if (out_freq >= 0. ) output( coupler , etime );

      // Some modules might need to use hydrostasis to project values into material boundaries
      // So let's put it into the coupler's data manager just in case
      auto &dm = coupler.get_data_manager_readwrite();
      dm.register_and_allocate<real>("hy_dens_cells"    ,"hydrostatic density cell averages"      ,{nz});
      dm.register_and_allocate<real>("hy_pressure_cells","hydrostatic density*theta cell averages",{nz});
      auto dm_hy_dens_cells     = dm.get<real,1>("hy_dens_cells"    );
      auto dm_hy_pressure_cells = dm.get<real,1>("hy_pressure_cells");
      hy_dens_cells    .deep_copy_to( dm_hy_dens_cells    );
      hy_pressure_cells.deep_copy_to( dm_hy_pressure_cells);

      // Register the tracers in the coupler so the user has access if they want (and init to zero)
      YAKL_SCOPE( nx          , this->nx          );
      YAKL_SCOPE( ny          , this->ny          );
      YAKL_SCOPE( nz          , this->nz          );
      YAKL_SCOPE( num_state   , this->num_state   );
      YAKL_SCOPE( num_tracers , this->num_tracers );
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


    // Initialize the supercell test case
    void init_supercell( core::Coupler &coupler , real4d &state , real4d &tracers ) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;
      real constexpr z_0    = 0;
      real constexpr z_trop = 12000;
      real constexpr T_0    = 300;
      real constexpr T_trop = 213;
      real constexpr T_top  = 213;
      real constexpr p_0    = 100000;

      int constexpr ngpt = 9;
      SArray<real,1,ngpt> gll_pts, gll_wts;

      TransformMatrices::get_gll_points (gll_pts);
      TransformMatrices::get_gll_weights(gll_wts);

      // Temporary arrays used to compute the initial state for high-CAPE supercell conditions
      real3d quad_temp       ("quad_temp"       ,nz,ngpt-1,ngpt);
      real2d hyDensGLL       ("hyDensGLL"       ,nz,ngpt);
      real2d hyDensThetaGLL  ("hyDensThetaGLL"  ,nz,ngpt);
      real2d hyDensVapGLL    ("hyDensVapGLL"    ,nz,ngpt);
      real2d hyPressureGLL   ("hyPressureGLL"   ,nz,ngpt);
      real1d hyDensCells     ("hyDensCells"     ,nz);
      real1d hyDensThetaCells("hyDensThetaCells",nz);

      real ztop = coupler.get_zlen();

      YAKL_SCOPE( hy_dens_cells       , this->hy_dens_cells       );
      YAKL_SCOPE( hy_pressure_cells   , this->hy_pressure_cells   );
      YAKL_SCOPE( hy_dens_edges       , this->hy_dens_edges       );
      YAKL_SCOPE( hy_pressure_edges   , this->hy_pressure_edges   );
      YAKL_SCOPE( nx                  , this->nx                  );
      YAKL_SCOPE( ny                  , this->ny                  );
      YAKL_SCOPE( nz                  , this->nz                  );
      YAKL_SCOPE( dx                  , this->dx                  );
      YAKL_SCOPE( dy                  , this->dy                  );
      YAKL_SCOPE( dz                  , this->dz                  );
      YAKL_SCOPE( xlen                , this->xlen                );
      YAKL_SCOPE( ylen                , this->ylen                );
      YAKL_SCOPE( sim2d               , this->sim2d               );
      YAKL_SCOPE( R_d                 , this->R_d                 );
      YAKL_SCOPE( R_v                 , this->R_v                 );
      YAKL_SCOPE( cv_d                , this->cv_d                );
      YAKL_SCOPE( cv_v                , this->cv_v                );
      YAKL_SCOPE( grav                , this->grav                );
      YAKL_SCOPE( num_tracers         , this->num_tracers         );
      YAKL_SCOPE( idWV                , this->idWV                );

      // Compute quadrature term to integrate to get pressure at GLL points
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ngpt-1,ngpt) ,
                    YAKL_LAMBDA (int k, int kk, int kkk) {
        // Middle of this cell
        real cellmid   = (k+0.5_fp) * dz;
        // Bottom, top, and middle of the space between these two ngpt GLL points
        real ngpt_b    = cellmid + gll_pts(kk  )*dz;
        real ngpt_t    = cellmid + gll_pts(kk+1)*dz;
        real ngpt_m    = 0.5_fp * (ngpt_b + ngpt_t);
        // Compute grid spacing between these ngpt GLL points
        real ngpt_dz   = dz * ( gll_pts(kk+1) - gll_pts(kk) );
        // Compute the locate of this GLL point within the ngpt GLL points
        real zloc      = ngpt_m + ngpt_dz * gll_pts(kkk);
        // Compute full density at this location
        real temp      = init_supercell_temperature (zloc, z_0, z_trop, ztop, T_0, T_trop, T_top);
        real press_dry = init_supercell_pressure_dry(zloc, z_0, z_trop, ztop, T_0, T_trop, T_top, p_0, R_d, grav);
        real qvs       = init_supercell_sat_mix_dry(press_dry, temp);
        real relhum    = init_supercell_relhum(zloc, z_0, z_trop);
        if (relhum * qvs > 0.014_fp) relhum = 0.014_fp / qvs;
        real qv        = std::min( 0.014_fp , qvs*relhum );
        quad_temp(k,kk,kkk) = -(1+qv)*grav/(R_d+qv*R_v)/temp;
      });

      // Compute pressure at GLL points
      parallel_for( YAKL_AUTO_LABEL() , 1 , YAKL_LAMBDA (int dummy) {
        hyPressureGLL(0,0) = p_0;
        for (int k=0; k < nz; k++) {
          for (int kk=0; kk < ngpt-1; kk++) {
            real tot = 0;
            for (int kkk=0; kkk < ngpt; kkk++) {
              tot += quad_temp(k,kk,kkk) * gll_wts(kkk);
            }
            tot *= dz * ( gll_pts(kk+1) - gll_pts(kk) );
            hyPressureGLL(k,kk+1) = hyPressureGLL(k,kk) * exp( tot );
            if (kk == ngpt-2 && k < nz-1) {
              hyPressureGLL(k+1,0) = hyPressureGLL(k,ngpt-1);
            }
          }
        }
      });

      // Compute hydrostatic background state at GLL points
      parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz,ngpt) , YAKL_LAMBDA (int k, int kk) {
        real zloc = (k+0.5_fp)*dz + gll_pts(kk)*dz;
        real temp       = init_supercell_temperature (zloc, z_0, z_trop, ztop, T_0, T_trop, T_top);
        real press_tmp  = init_supercell_pressure_dry(zloc, z_0, z_trop, ztop, T_0, T_trop, T_top, p_0, R_d, grav);
        real qvs        = init_supercell_sat_mix_dry(press_tmp, temp);
        real relhum     = init_supercell_relhum(zloc, z_0, z_trop);
        if (relhum * qvs > 0.014_fp) relhum = 0.014_fp / qvs;
        real qv         = std::min( 0.014_fp , qvs*relhum );
        real press      = hyPressureGLL(k,kk);
        real dens_dry   = press / (R_d+qv*R_v) / temp;
        real dens_vap   = qv * dens_dry;
        real dens       = dens_dry + dens_vap;
        hyDensGLL     (k,kk) = dens;
        hyDensVapGLL  (k,kk) = dens_vap;
        if (kk == 0) {
          hy_dens_edges    (k) = dens;
          hy_pressure_edges(k) = press;
        }
        if (k == nz-1 && kk == ngpt-1) {
          hy_dens_edges    (k+1) = dens;
          hy_pressure_edges(k+1) = press;
        }
      });

      // Compute hydrostatic background state over cells
      parallel_for( YAKL_AUTO_LABEL() , Bounds<1>(nz) , YAKL_LAMBDA (int k) {
        real press_tot = 0;
        real dens_tot  = 0;
        for (int kk=0; kk < ngpt; kk++) {
          press_tot += hyPressureGLL(k,kk) * gll_wts(kk);
          dens_tot  += hyDensGLL    (k,kk) * gll_wts(kk);
        }
        // These are used in the rest of the model
        hy_dens_cells    (k) = dens_tot;
        hy_pressure_cells(k) = press_tot;
      });

      size_t i_beg = coupler.get_i_beg();
      size_t j_beg = coupler.get_j_beg();

      // Initialize the state
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        for (int l=0; l < num_state  ; l++) { state  (l,hs+k,hs+j,hs+i) = 0; }
        for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+i) = 0; }
        for (int kk=0; kk < ngpt; kk++) {
          for (int jj=0; jj < ngpt; jj++) {
            for (int ii=0; ii < ngpt; ii++) {
              real xloc = (i+i_beg+0.5_fp)*dx + gll_pts(ii)*dx;
              real yloc = (j+j_beg+0.5_fp)*dy + gll_pts(jj)*dy;
              real zloc = (k      +0.5_fp)*dz + gll_pts(kk)*dz;

              if (sim2d) yloc = ylen/2;

              real rho   = hyDensGLL    (k,kk);
              real p     = hyPressureGLL(k,kk);
              real rho_v = hyDensVapGLL (k,kk);
              real temp = init_supercell_temperature(zloc, z_0, z_trop, ztop, T_0, T_trop, T_top);
              real rho_d = rho - rho_v;

              real u;
              real constexpr zs = 5000;
              real constexpr us = 30;
              real constexpr uc = 15;
              if (zloc < zs) {
                u = us * (zloc / zs) - uc;
              } else {
                u = us - uc;
              }

              real v = 0;
              real w = 0;
              real e = (rho_d/rho*cv_d+rho_v/rho*cv_v)*temp + (u*u+v*v+w*w)/2 + grav*zloc;

              real factor = gll_wts(ii) * gll_wts(jj) * gll_wts(kk);
              state  (idR ,hs+k,hs+j,hs+i) += rho     * factor;
              state  (idU ,hs+k,hs+j,hs+i) += rho * u * factor;
              state  (idV ,hs+k,hs+j,hs+i) += rho * v * factor;
              state  (idW ,hs+k,hs+j,hs+i) += rho * w * factor;
              state  (idT ,hs+k,hs+j,hs+i) += rho * e * factor;
              tracers(idWV,hs+k,hs+j,hs+i) += rho_v   * factor;
            }
          }
        }
      });
    }


    // Convert dynamics state and tracers arrays to the coupler state and write to the coupler's data
    void convert_dynamics_to_coupler( core::Coupler &coupler , realConst4d state , realConst4d tracers ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

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

      YAKL_SCOPE( cv_d                , this->cv_d                );
      YAKL_SCOPE( cv_v                , this->cv_v                );
      YAKL_SCOPE( grav                , this->grav                );
      YAKL_SCOPE( dz                  , this->dz                  );
      YAKL_SCOPE( num_tracers         , this->num_tracers         );
      YAKL_SCOPE( idWV                , this->idWV                );
      YAKL_SCOPE( tracer_adds_mass    , this->tracer_adds_mass    );

      // Convert from state and tracers arrays to the coupler's data
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        real rho = state(idR,hs+k,hs+j,hs+i);
        real u   = state(idU,hs+k,hs+j,hs+i) / rho;
        real v   = state(idV,hs+k,hs+j,hs+i) / rho;
        real w   = state(idW,hs+k,hs+j,hs+i) / rho;
        real e   = state(idT,hs+k,hs+j,hs+i) / rho;

        real rho_v = tracers(idWV,hs+k,hs+j,hs+i);
        real rho_d = rho;
        for (int tr=0; tr < num_tracers; tr++) {
          if (tracer_adds_mass(tr)) rho_d -= tracers(tr,hs+k,hs+j,hs+i);
        }
        real temp = (e - (u*u+v*v+w*w)/2 - grav*(k+0.5_fp)*dz) / (rho_d/rho*cv_d+rho_v/rho*cv_v);

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

      YAKL_SCOPE( cv_d                , this->cv_d                );
      YAKL_SCOPE( cv_v                , this->cv_v                );
      YAKL_SCOPE( grav                , this->grav                );
      YAKL_SCOPE( dz                  , this->dz                  );
      YAKL_SCOPE( num_tracers         , this->num_tracers         );
      YAKL_SCOPE( idWV                , this->idWV                );
      YAKL_SCOPE( tracer_adds_mass    , this->tracer_adds_mass    );

      // Convert from the coupler's state to the dycore's state and tracers arrays
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        real rho_d = dm_rho_d(k,j,i);
        real u     = dm_uvel (k,j,i);
        real v     = dm_vvel (k,j,i);
        real w     = dm_wvel (k,j,i);
        real temp  = dm_temp (k,j,i);
        real rho_v = dm_tracers(idWV,k,j,i);

        real rho = rho_d;
        for (int tr=0; tr < num_tracers; tr++) {
          if (tracer_adds_mass(tr)) rho += dm_tracers(tr,k,j,i);
        }
        real e = (rho_d/rho*cv_d+rho_v/rho*cv_v)*temp + (u*u+v*v+w*w)/2 + grav*(k+0.5_fp)*dz;

        state(idR,hs+k,hs+j,hs+i) = rho;
        state(idU,hs+k,hs+j,hs+i) = rho * u;
        state(idV,hs+k,hs+j,hs+i) = rho * v;
        state(idW,hs+k,hs+j,hs+i) = rho * w;
        state(idT,hs+k,hs+j,hs+i) = rho * e;
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

      YAKL_SCOPE( dx                  , this->dx                  );
      YAKL_SCOPE( dy                  , this->dy                  );
      YAKL_SCOPE( dz                  , this->dz                  );

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

      YAKL_SCOPE( nx    , this->nx    );
      YAKL_SCOPE( ny    , this->ny    );

      int npack = num_state + num_tracers;

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

      //Pre-post the receives
      MPI_Irecv( halo_recv_buf_W_host.data() , npack*nz*ny*hs , MPI_DOUBLE , neigh(1,0) , 0 , MPI_COMM_WORLD , &rReq[0] );
      MPI_Irecv( halo_recv_buf_E_host.data() , npack*nz*ny*hs , MPI_DOUBLE , neigh(1,2) , 1 , MPI_COMM_WORLD , &rReq[1] );
      if (!sim2d) {
        MPI_Irecv( halo_recv_buf_S_host.data() , npack*nz*hs*nx , MPI_DOUBLE , neigh(0,1) , 2 , MPI_COMM_WORLD , &rReq[2] );
        MPI_Irecv( halo_recv_buf_N_host.data() , npack*nz*hs*nx , MPI_DOUBLE , neigh(2,1) , 3 , MPI_COMM_WORLD , &rReq[3] );
      }

      halo_send_buf_W.deep_copy_to(halo_send_buf_W_host);
      halo_send_buf_E.deep_copy_to(halo_send_buf_E_host);
      if (!sim2d) {
        halo_send_buf_S.deep_copy_to(halo_send_buf_S_host);
        halo_send_buf_N.deep_copy_to(halo_send_buf_N_host);
      }

      yakl::fence();

      //Send the data
      MPI_Isend( halo_send_buf_W_host.data() , npack*nz*ny*hs , MPI_DOUBLE , neigh(1,0) , 1 , MPI_COMM_WORLD , &sReq[0] );
      MPI_Isend( halo_send_buf_E_host.data() , npack*nz*ny*hs , MPI_DOUBLE , neigh(1,2) , 0 , MPI_COMM_WORLD , &sReq[1] );
      if (!sim2d) {
        MPI_Isend( halo_send_buf_S_host.data() , npack*nz*hs*nx , MPI_DOUBLE , neigh(0,1) , 3 , MPI_COMM_WORLD , &sReq[2] );
        MPI_Isend( halo_send_buf_N_host.data() , npack*nz*hs*nx , MPI_DOUBLE , neigh(2,1) , 2 , MPI_COMM_WORLD , &sReq[3] );
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

      YAKL_SCOPE( nx    , this->nx    );
      YAKL_SCOPE( ny    , this->ny    );

      int npack = num_state + num_tracers;

      real3d edge_send_buf_W("edge_send_buf_W",npack,nz,ny);
      real3d edge_send_buf_E("edge_send_buf_E",npack,nz,ny);

      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(npack,nz,ny) , YAKL_LAMBDA (int v, int k, int j) {
        if (v < num_state) {
          edge_send_buf_W(v,k,j) = state_limits_x  (1,v          ,k,j,0 );
          edge_send_buf_E(v,k,j) = state_limits_x  (0,v          ,k,j,nx);
        } else {                                      
          edge_send_buf_W(v,k,j) = tracers_limits_x(1,v-num_state,k,j,0 );
          edge_send_buf_E(v,k,j) = tracers_limits_x(0,v-num_state,k,j,nx);
        }
      });

      real3d edge_send_buf_S("edge_send_buf_S",npack,nz,nx);
      real3d edge_send_buf_N("edge_send_buf_N",npack,nz,nx);

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(npack,nz,nx) , YAKL_LAMBDA (int v, int k, int i) {
          if (v < num_state) {
            edge_send_buf_S(v,k,i) = state_limits_y  (1,v          ,k,0 ,i);
            edge_send_buf_N(v,k,i) = state_limits_y  (0,v          ,k,ny,i);
          } else {                                      
            edge_send_buf_S(v,k,i) = tracers_limits_y(1,v-num_state,k,0 ,i);
            edge_send_buf_N(v,k,i) = tracers_limits_y(0,v-num_state,k,ny,i);
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

      //Pre-post the receives
      MPI_Irecv( edge_recv_buf_W_host.data() , npack*nz*ny , MPI_DOUBLE , neigh(1,0) , 4 , MPI_COMM_WORLD , &rReq[0] );
      MPI_Irecv( edge_recv_buf_E_host.data() , npack*nz*ny , MPI_DOUBLE , neigh(1,2) , 5 , MPI_COMM_WORLD , &rReq[1] );
      if (!sim2d) {
        MPI_Irecv( edge_recv_buf_S_host.data() , npack*nz*nx , MPI_DOUBLE , neigh(0,1) , 6 , MPI_COMM_WORLD , &rReq[2] );
        MPI_Irecv( edge_recv_buf_N_host.data() , npack*nz*nx , MPI_DOUBLE , neigh(2,1) , 7 , MPI_COMM_WORLD , &rReq[3] );
      }

      edge_send_buf_W.deep_copy_to(edge_send_buf_W_host);
      edge_send_buf_E.deep_copy_to(edge_send_buf_E_host);
      if (!sim2d) {
        edge_send_buf_S.deep_copy_to(edge_send_buf_S_host);
        edge_send_buf_N.deep_copy_to(edge_send_buf_N_host);
      }

      yakl::fence();

      //Send the data
      MPI_Isend( edge_send_buf_W_host.data() , npack*nz*ny , MPI_DOUBLE , neigh(1,0) , 5 , MPI_COMM_WORLD , &sReq[0] );
      MPI_Isend( edge_send_buf_E_host.data() , npack*nz*ny , MPI_DOUBLE , neigh(1,2) , 4 , MPI_COMM_WORLD , &sReq[1] );
      if (!sim2d) {
        MPI_Isend( edge_send_buf_S_host.data() , npack*nz*nx , MPI_DOUBLE , neigh(0,1) , 7 , MPI_COMM_WORLD , &sReq[2] );
        MPI_Isend( edge_send_buf_N_host.data() , npack*nz*nx , MPI_DOUBLE , neigh(2,1) , 6 , MPI_COMM_WORLD , &sReq[3] );
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

      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(npack,nz,ny) , YAKL_LAMBDA (int v, int k, int j) {
        if (v < num_state) {
          state_limits_x  (0,v          ,k,j,0 ) = edge_recv_buf_W(v,k,j);
          state_limits_x  (1,v          ,k,j,nx) = edge_recv_buf_E(v,k,j);
        } else {             
          tracers_limits_x(0,v-num_state,k,j,0 ) = edge_recv_buf_W(v,k,j);
          tracers_limits_x(1,v-num_state,k,j,nx) = edge_recv_buf_E(v,k,j);
        }
      });

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(npack,nz,nx) , YAKL_LAMBDA (int v, int k, int i) {
          if (v < num_state) {
            state_limits_y  (0,v          ,k,0 ,i) = edge_recv_buf_S(v,k,i);
            state_limits_y  (1,v          ,k,ny,i) = edge_recv_buf_N(v,k,i);
          } else {             
            tracers_limits_y(0,v-num_state,k,0 ,i) = edge_recv_buf_S(v,k,i);
            tracers_limits_y(1,v-num_state,k,ny,i) = edge_recv_buf_N(v,k,i);
          }
        });
      }
    }


    // Creates initial data at a point in space for the rising moist thermal test case
    YAKL_INLINE static void thermal(real x, real y, real z, real xlen, real ylen, real grav, real C0, real gamma,
                                    real cp, real p0, real R_d, real R_v, real &rho, real &u, real &v, real &w,
                                    real &temp, real &p, real &rho_v, real &hr, real &ht) {
      hydro_const_theta(z,grav,C0,cp,p0,gamma,R_d,hr,ht);
      real rho_d   = hr;
      u            = 0.;
      v            = 0.;
      w            = 0.;
      real theta_d = ht; // + sample_ellipse_cosine(2._fp  ,  x,y,z  ,  xlen/2,ylen/2,2000.  ,  2000.,2000.,2000.);
      real p_d     = C0 * pow( rho_d*theta_d , gamma );
      temp         = p_d / rho_d / R_d;
      real sat_pv  = saturation_vapor_pressure(temp);
      real sat_rv  = sat_pv / R_v / temp;
      rho_v        = 0; // sample_ellipse_cosine(0.8_fp  ,  x,y,z  ,  xlen/2,ylen/2,2000.  ,  2000.,2000.,2000.) * sat_rv;
      p            = rho_d * R_d * temp + rho_v * R_v * temp;
      rho          = rho_d + rho_v;
    }


    // Computes a hydrostatic background density and potential temperature using c constant potential temperature
    // backgrounda for a single vertical location
    YAKL_INLINE static void hydro_const_theta( real z, real grav, real C0, real cp, real p0, real gamma, real rd,
                                               real &r, real &t ) {
      const real theta0 = 300.;  //Background potential temperature
      const real exner0 = 1.;    //Surface-level Exner pressure
      t = theta0;                                       //Potential Temperature at z
      real exner = exner0 - grav * z / (cp * theta0);   //Exner pressure at z
      real p = p0 * std::pow(exner,(cp/rd));            //Pressure at z
      real rt = std::pow((p / C0),(1._fp / gamma));     //rho*theta at z
      r = rt / t;                                       //Density at z
    }


    // Samples a 3-D ellipsoid at a point in space
    YAKL_INLINE static real sample_ellipse_cosine(real amp, real x   , real y   , real z   ,
                                                            real x0  , real y0  , real z0  ,
                                                            real xrad, real yrad, real zrad) {
      //Compute distance from bubble center
      real dist = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) +
                        ((y-y0)/yrad)*((y-y0)/yrad) +
                        ((z-z0)/zrad)*((z-z0)/zrad) ) * M_PI / 2.;
      //If the distance from bubble center is less than the radius, create a cos**2 profile
      if (dist <= M_PI / 2.) {
        return amp * std::pow(cos(dist),2._fp);
      } else {
        return 0.;
      }
    }


    YAKL_INLINE static real saturation_vapor_pressure(real temp) {
      real tc = temp - 273.15;
      return 610.94 * std::exp( 17.625*tc / (243.04+tc) );
    }


    // Compute supercell temperature profile at a vertical location
    YAKL_INLINE static real init_supercell_temperature(real z, real z_0, real z_trop, real z_top,
                                                       real T_0, real T_trop, real T_top) {
      if (z <= z_trop) {
        real lapse = - (T_trop - T_0) / (z_trop - z_0);
        return T_0 - lapse * (z - z_0);
      } else {
        real lapse = - (T_top - T_trop) / (z_top - z_trop);
        return T_trop - lapse * (z - z_trop);
      }
    }


    // Compute supercell dry pressure profile at a vertical location
    YAKL_INLINE static real init_supercell_pressure_dry(real z, real z_0, real z_trop, real z_top,
                                                        real T_0, real T_trop, real T_top,
                                                        real p_0, real R_d, real grav) {
      if (z <= z_trop) {
        real lapse = - (T_trop - T_0) / (z_trop - z_0);
        real T = init_supercell_temperature(z, z_0, z_trop, z_top, T_0, T_trop, T_top);
        return p_0 * pow( T / T_0 , grav/(R_d*lapse) );
      } else {
        // Get pressure at the tropopause
        real lapse = - (T_trop - T_0) / (z_trop - z_0);
        real p_trop = p_0 * pow( T_trop / T_0 , grav/(R_d*lapse) );
        // Get pressure at requested height
        lapse = - (T_top - T_trop) / (z_top - z_trop);
        if (lapse != 0) {
          real T = init_supercell_temperature(z, z_0, z_trop, z_top, T_0, T_trop, T_top);
          return p_trop * pow( T / T_trop , grav/(R_d*lapse) );
        } else {
          return p_trop * exp(-grav*(z-z_trop)/(R_d*T_trop));
        }
      }
    }

    
    // Compute supercell relative humidity profile at a vertical location
    YAKL_INLINE static real init_supercell_relhum(real z, real z_0, real z_trop) {
      if (z <= z_trop) {
        return 1._fp - 0.75_fp * pow(z / z_trop , 1.25_fp );
      } else {
        return 0.25_fp;
      }
    }


    // Computes dry saturation mixing ratio
    YAKL_INLINE static real init_supercell_sat_mix_dry( real press , real T ) {
      return 380/(press) * exp( 17.27_fp * (T-273)/(T-36) );
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


