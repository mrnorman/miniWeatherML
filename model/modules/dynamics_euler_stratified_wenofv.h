
#pragma once

#include "main_header.h"
#include "MultipleFields.h"
#include "TransformMatrices.h"
#include "WenoLimiter.h"
#include <random>

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
      int  static constexpr ord = 5;
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
    int  static constexpr DATA_THERMAL   = 0;
    int  static constexpr DATA_SUPERCELL = 1;
    int  static constexpr DATA_CITY      = 2;

    int  static constexpr BC_PERIODIC = 0;
    int  static constexpr BC_OPEN     = 1;
    int  static constexpr BC_WALL     = 2;

    // Hydrostatic background profiles for density and potential temperature as cell averages and cell edge values
    real1d      hy_dens_cells;
    real1d      hy_dens_theta_cells;
    real1d      hy_dens_edges;
    real1d      hy_dens_theta_edges;
    real        etime;         // Elapsed time
    real        out_freq;      // Frequency out file output
    int         num_out;       // Number of outputs produced thus far
    std::string fname;         // File name for file output
    int         init_data_int; // Integer representation of the type of initial data to use (test case)

    int         nx  , ny  , nz  ;  // # cells in each dimension
    real        dx  , dy  , dz  ;  // grid spacing in each dimension
    real        xlen, ylen, zlen;  // length of domain in each dimension
    int         bc_x, bc_y, bc_z;
    bool        sim2d;             // Whether we're simulating in 2-D

    // Physical constants
    real        R_d;    // Dry air ideal gas constant
    real        R_v;    // Water vapor ideal gas constant
    real        cp_d;   // Specific heat of dry air at constant pressure
    real        cp_v;   // Specific heat of water vapor at constant pressure
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

    bool use_immersed_boundaries;

    real3d immersed_proportion;


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
        auto mpi_data_type = coupler.get_mpi_data_type();
        MPI_Reduce( &maxw_loc , &maxw , 1 , mpi_data_type , MPI_MAX , 0 , MPI_COMM_WORLD );
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
    void compute_tendencies( core::Coupler &coupler , real4d const &state   , real4d const &state_tend   ,
                                                      real4d const &tracers , real4d const &tracers_tend , real dt ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;
      using std::min;
      using std::max;

      // A slew of things to bring from class scope into local scope so that lambdas copy them by value to the GPU
      YAKL_SCOPE( hy_dens_cells              , this->hy_dens_cells              );
      YAKL_SCOPE( hy_dens_theta_cells        , this->hy_dens_theta_cells        );
      YAKL_SCOPE( hy_dens_edges              , this->hy_dens_edges              );
      YAKL_SCOPE( hy_dens_theta_edges        , this->hy_dens_theta_edges        );
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
      YAKL_SCOPE( use_immersed_boundaries    , this->use_immersed_boundaries    );
      YAKL_SCOPE( immersed_proportion        , this->immersed_proportion        );

      // Since tracers are full mass, it's helpful before reconstruction to remove the background density for potentially
      // more accurate reconstructions of tracer concentrations
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i ) {
        state(idU,hs+k,hs+j,hs+i) /= ( state(idR,hs+k,hs+j,hs+i) + hy_dens_cells(k) );
        state(idV,hs+k,hs+j,hs+i) /= ( state(idR,hs+k,hs+j,hs+i) + hy_dens_cells(k) );
        state(idW,hs+k,hs+j,hs+i) /= ( state(idR,hs+k,hs+j,hs+i) + hy_dens_cells(k) );
        for (int tr=0; tr < num_tracers; tr++) {
          tracers(tr,hs+k,hs+j,hs+i) /= ( state(idR,hs+k,hs+j,hs+i) + hy_dens_cells(k) );
        }
      });

      halo_exchange( coupler , state , tracers );

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
        ////////////////////////////////////////////////////////
        // X-direction
        ////////////////////////////////////////////////////////
        // State
        for (int l=0; l < num_state; l++) {
          // Gather the stencil of cell averages, and use WENO to compute values at the cell edges (i.e., 2 GLL points)
          SArray<real,1,ord> stencil;
          SArray<real,1,2>   gll;
          for (int s=0; s < ord; s++) { stencil(s) = state(l,hs+k,hs+j,i+s); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          state_limits_x(l,1,k,j,i  ) = gll(0);
          state_limits_x(l,0,k,j,i+1) = gll(1);
        }
        // Add back hydrostatic backgrounds to density and density*theta because only perturbations were reconstructed
        state_limits_x(idR,1,k,j,i  ) += hy_dens_cells(k);
        state_limits_x(idR,0,k,j,i+1) += hy_dens_cells(k);
        state_limits_x(idU,1,k,j,i  ) *= state_limits_x(idR,1,k,j,i  );
        state_limits_x(idU,0,k,j,i+1) *= state_limits_x(idR,0,k,j,i+1);
        state_limits_x(idV,1,k,j,i  ) *= state_limits_x(idR,1,k,j,i  );
        state_limits_x(idV,0,k,j,i+1) *= state_limits_x(idR,0,k,j,i+1);
        state_limits_x(idW,1,k,j,i  ) *= state_limits_x(idR,1,k,j,i  );
        state_limits_x(idW,0,k,j,i+1) *= state_limits_x(idR,0,k,j,i+1);
        state_limits_x(idT,1,k,j,i  ) += hy_dens_theta_cells(k);
        state_limits_x(idT,0,k,j,i+1) += hy_dens_theta_cells(k);

        // Tracers
        for (int l=0; l < num_tracers; l++) {
          // Gather the stencil of cell averages, and use WENO to compute values at the cell edges (i.e., 2 GLL points)
          SArray<real,1,ord> stencil;
          SArray<real,1,2>   gll;
          for (int s=0; s < ord; s++) { stencil(s) = tracers(l,hs+k,hs+j,i+s); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          tracers_limits_x(l,1,k,j,i  ) = gll(0) * state_limits_x(idR,1,k,j,i  );
          tracers_limits_x(l,0,k,j,i+1) = gll(1) * state_limits_x(idR,0,k,j,i+1);
        }

        ////////////////////////////////////////////////////////
        // Y-direction
        ////////////////////////////////////////////////////////
        // If we're simulating in only 2-D, then do not compute y-direction tendencies
        if (!sim2d) {
          // State
          for (int l=0; l < num_state; l++) {
            // Gather the stencil of cell averages, and use WENO to compute values at the cell edges (i.e., 2 GLL points)
            SArray<real,1,ord> stencil;
            SArray<real,1,2>   gll;
            for (int s=0; s < ord; s++) { stencil(s) = state(l,hs+k,j+s,hs+i); }
            reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
            state_limits_y(l,1,k,j  ,i) = gll(0);
            state_limits_y(l,0,k,j+1,i) = gll(1);
          }
          // Add back hydrostatic backgrounds to density and density*theta because only perturbations were reconstructed
          state_limits_y(idR,1,k,j  ,i) += hy_dens_cells(k);
          state_limits_y(idR,0,k,j+1,i) += hy_dens_cells(k);
          state_limits_y(idU,1,k,j  ,i) *= state_limits_y(idR,1,k,j  ,i);
          state_limits_y(idU,0,k,j+1,i) *= state_limits_y(idR,0,k,j+1,i);
          state_limits_y(idV,1,k,j  ,i) *= state_limits_y(idR,1,k,j  ,i);
          state_limits_y(idV,0,k,j+1,i) *= state_limits_y(idR,0,k,j+1,i);
          state_limits_y(idW,1,k,j  ,i) *= state_limits_y(idR,1,k,j  ,i);
          state_limits_y(idW,0,k,j+1,i) *= state_limits_y(idR,0,k,j+1,i);
          state_limits_y(idT,1,k,j  ,i) += hy_dens_theta_cells(k);
          state_limits_y(idT,0,k,j+1,i) += hy_dens_theta_cells(k);

          // Tracers
          for (int l=0; l < num_tracers; l++) {
            // Gather the stencil of cell averages, and use WENO to compute values at the cell edges (i.e., 2 GLL points)
            SArray<real,1,ord> stencil;
            SArray<real,1,2>   gll;
            for (int s=0; s < ord; s++) { stencil(s) = tracers(l,hs+k,j+s,hs+i); }
            reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
            tracers_limits_y(l,1,k,j  ,i) = gll(0) * state_limits_y(idR,1,k,j  ,i);
            tracers_limits_y(l,0,k,j+1,i) = gll(1) * state_limits_y(idR,0,k,j+1,i);
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
          // Gather the stencil of cell averages, and use WENO to compute values at the cell edges (i.e., 2 GLL points)
          SArray<real,1,ord> stencil;
          SArray<real,1,2>   gll;
          for (int s=0; s < ord; s++) {
            // We wet w-momentum to zero in the boundaries
            if ( l == idW && ((k+s < hs) || (k+s >= nz+hs)) ) {
              stencil(s) = 0;
            } else {
              int ind = min( nz+hs-1 , max( (int) hs , k+s ) );
              stencil(s) = state(l,ind,hs+j,hs+i);
            }
          }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          state_limits_z(l,1,k  ,j,i) = gll(0);
          state_limits_z(l,0,k+1,j,i) = gll(1);
        }
        // Add back hydrostatic backgrounds to density and density*theta because only perturbations were reconstructed
        state_limits_z(idR,1,k  ,j,i) += hy_dens_edges(k  );
        state_limits_z(idR,0,k+1,j,i) += hy_dens_edges(k+1);
        state_limits_z(idU,1,k  ,j,i) *= state_limits_z(idR,1,k  ,j,i);
        state_limits_z(idU,0,k+1,j,i) *= state_limits_z(idR,0,k+1,j,i);
        state_limits_z(idV,1,k  ,j,i) *= state_limits_z(idR,1,k  ,j,i);
        state_limits_z(idV,0,k+1,j,i) *= state_limits_z(idR,0,k+1,j,i);
        state_limits_z(idW,1,k  ,j,i) *= state_limits_z(idR,1,k  ,j,i);
        state_limits_z(idW,0,k+1,j,i) *= state_limits_z(idR,0,k+1,j,i);
        state_limits_z(idT,1,k  ,j,i) += hy_dens_theta_edges(k  );
        state_limits_z(idT,0,k+1,j,i) += hy_dens_theta_edges(k+1);
        // We wet w-momentum to zero at the boundaries
        if (k == 0   ) { state_limits_z(idW,1,k  ,j,i) = 0; }
        if (k == nz-1) { state_limits_z(idW,0,k+1,j,i) = 0; }

        // Tracers
        for (int l=0; l < num_tracers; l++) {
          // Gather the stencil of cell averages, and use WENO to compute values at the cell edges (i.e., 2 GLL points)
          SArray<real,1,ord> stencil;
          SArray<real,1,2>   gll;
          for (int s=0; s < ord; s++) {
            int ind = min( nz+hs-1 , max( (int) hs , k+s ) );
            stencil(s) = tracers(l,ind,hs+j,hs+i);
          }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          tracers_limits_z(l,1,k  ,j,i) = gll(0) * state_limits_z(idR,1,k  ,j,i);
          tracers_limits_z(l,0,k+1,j,i) = gll(1) * state_limits_z(idR,0,k+1,j,i);
        }
      });

      edge_exchange( coupler , state_limits_x , tracers_limits_x ,
                               state_limits_y , tracers_limits_y ,
                               state_limits_z , tracers_limits_z );

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

        ////////////////////////////////////////////////////////
        // Y-direction
        ////////////////////////////////////////////////////////
        // If we are simulating in 2-D, then do not do Riemann in the y-direction
        if ( (! sim2d) && i < nx && k < nz) {
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

        if (i < nx && j < ny && k < nz) {
          state(idU,hs+k,hs+j,hs+i) *= ( state(idR,hs+k,hs+j,hs+i) + hy_dens_cells(k) );
          state(idV,hs+k,hs+j,hs+i) *= ( state(idR,hs+k,hs+j,hs+i) + hy_dens_cells(k) );
          state(idW,hs+k,hs+j,hs+i) *= ( state(idR,hs+k,hs+j,hs+i) + hy_dens_cells(k) );
          for (int tr=0; tr < num_tracers; tr++) {
            tracers(tr,hs+k,hs+j,hs+i) *= ( state(idR,hs+k,hs+j,hs+i) + hy_dens_cells(k) );
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

      // Compute tendencies as the flux divergence + gravity source term
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        for (int l = 0; l < num_state; l++) {
          state_tend  (l,k,j,i) = -( state_flux_x  (l,k  ,j  ,i+1) - state_flux_x  (l,k,j,i) ) / dx
                                  -( state_flux_y  (l,k  ,j+1,i  ) - state_flux_y  (l,k,j,i) ) / dy
                                  -( state_flux_z  (l,k+1,j  ,i  ) - state_flux_z  (l,k,j,i) ) / dz;
          if (l == idW) state_tend(l,k,j,i) += -grav * ( state(idR,hs+k,hs+j,hs+i) + hy_dens_cells(k) );
          if (l == idV && sim2d) state_tend(l,k,j,i) = 0;
        }
        for (int l = 0; l < num_tracers; l++) {
          tracers_tend(l,k,j,i) = -( tracers_flux_x(l,k  ,j  ,i+1) - tracers_flux_x(l,k,j,i) ) / dx
                                  -( tracers_flux_y(l,k  ,j+1,i  ) - tracers_flux_y(l,k,j,i) ) / dy
                                  -( tracers_flux_z(l,k+1,j  ,i  ) - tracers_flux_z(l,k,j,i) ) / dz;
        }
        if (use_immersed_boundaries) {
          real delta     = std::pow( dx*dy*dz , 1._fp/3._fp );
          real beta      = immersed_proportion(k,j,i);
          real C_d       = 1.e3*beta/delta;
          real C_t       = 1.e1*beta/delta;
          real rho       = state(idR,hs+k,hs+j,hs+i) + hy_dens_cells(k);
          real uvel      = state(idU,hs+k,hs+j,hs+i) / rho;
          real vvel      = state(idV,hs+k,hs+j,hs+i) / rho;
          real wvel      = state(idW,hs+k,hs+j,hs+i) / rho;
          real rho_theta = state(idT,hs+k,hs+j,hs+i) + hy_dens_theta_cells(k);
          real wind_mag  = sqrt(uvel*uvel+vvel*vvel+wvel*wvel);
          state_tend(idR,k,j,i) += -C_t * (rho-hy_dens_cells(k)) * wind_mag;
          state_tend(idU,k,j,i) += -C_d * rho * std::abs(uvel) * uvel;
          state_tend(idV,k,j,i) += -C_d * rho * std::abs(vvel) * vvel;
          state_tend(idW,k,j,i) += -C_d * rho * std::abs(wvel) * wvel;
          state_tend(idT,k,j,i) += -C_t * (rho_theta-hy_dens_theta_cells(k)) * wind_mag;
          // if (immersed_proportion(k,j,i) == 1) {
          //   state_tend(idR,k,j,i) = -state(idR,hs+k,hs+j,hs+i)/dt;
          //   state_tend(idU,k,j,i) = -state(idU,hs+k,hs+j,hs+i)/dt;
          //   state_tend(idV,k,j,i) = -state(idV,hs+k,hs+j,hs+i)/dt;
          //   state_tend(idW,k,j,i) = -state(idW,hs+k,hs+j,hs+i)/dt;
          //   state_tend(idT,k,j,i) = -state(idT,hs+k,hs+j,hs+i)/dt;
          // }
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

      auto &dm = coupler.get_data_manager_readwrite();

      dm.register_and_allocate<real>("density_dry","",{nz,ny,nx});
      dm.register_and_allocate<real>("uvel","",{nz,ny,nx});
      dm.register_and_allocate<real>("vvel","",{nz,ny,nx});
      dm.register_and_allocate<real>("wvel","",{nz,ny,nx});
      dm.register_and_allocate<real>("temp","",{nz,ny,nx});

      sim2d = (coupler.get_ny_glob() == 1);

      R_d   = coupler.get_option<real>("R_d" ,287 );
      R_v   = coupler.get_option<real>("R_v" ,461 );
      cp_d  = coupler.get_option<real>("cp_d",1003);
      cp_v  = coupler.get_option<real>("cp_v",1859);
      p0    = coupler.get_option<real>("p0"  ,1.e5);
      grav  = coupler.get_option<real>("grav",9.81);
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

      coupler.set_option<int>("idWV",idWV);
      dm.register_and_allocate<bool>("tracer_adds_mass","",{num_tracers});
      auto dm_tracer_adds_mass = dm.get<bool,1>("tracer_adds_mass");
      tracer_adds_mass.deep_copy_to(dm_tracer_adds_mass);

      // Set an integer version of the input_data so we can test it inside GPU kernels
      if      (init_data == "thermal"  ) { init_data_int = DATA_THERMAL;   }
      else if (init_data == "supercell") { init_data_int = DATA_SUPERCELL; }
      else if (init_data == "city"     ) { init_data_int = DATA_CITY;      }
      else { endrun("ERROR: Invalid init_data in yaml input file"); }

      use_immersed_boundaries = false;
      immersed_proportion = real3d("immersed_proportion",nz,ny,nx);
      immersed_proportion = 0;

      etime   = 0;
      num_out = 0;

      // Allocate temp arrays to hold state and tracers before we convert it back to the coupler state
      real4d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
      real4d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);

      // Allocate arrays for hydrostatic background states
      hy_dens_cells       = real1d("hy_dens_cells"      ,nz  );
      hy_dens_theta_cells = real1d("hy_dens_theta_cells",nz  );
      hy_dens_edges       = real1d("hy_dens_edges"      ,nz+1);
      hy_dens_theta_edges = real1d("hy_dens_theta_edges",nz+1);

      if (init_data_int == DATA_SUPERCELL) {

        bc_x = BC_PERIODIC;
        bc_y = BC_PERIODIC;
        bc_z = BC_WALL;
        init_supercell( coupler , state , tracers );

      } else if (init_data_int == DATA_THERMAL) {

        bc_x = BC_PERIODIC;
        bc_y = BC_PERIODIC;
        bc_z = BC_WALL;
        // Define quadrature weights and points for 3-point rules
        const int nqpoints = 9;
        SArray<real,1,nqpoints> qpoints;
        SArray<real,1,nqpoints> qweights;

        TransformMatrices::get_gll_points (qpoints );
        TransformMatrices::get_gll_weights(qweights);

        YAKL_SCOPE( init_data_int       , this->init_data_int       );
        YAKL_SCOPE( hy_dens_cells       , this->hy_dens_cells       );
        YAKL_SCOPE( hy_dens_theta_cells , this->hy_dens_theta_cells );
        YAKL_SCOPE( hy_dens_edges       , this->hy_dens_edges       );
        YAKL_SCOPE( hy_dens_theta_edges , this->hy_dens_theta_edges );
        YAKL_SCOPE( dx                  , this->dx                  );
        YAKL_SCOPE( dy                  , this->dy                  );
        YAKL_SCOPE( dz                  , this->dz                  );
        YAKL_SCOPE( xlen                , this->xlen                );
        YAKL_SCOPE( ylen                , this->ylen                );
        YAKL_SCOPE( sim2d               , this->sim2d               );
        YAKL_SCOPE( R_d                 , this->R_d                 );
        YAKL_SCOPE( R_v                 , this->R_v                 );
        YAKL_SCOPE( cp_d                , this->cp_d                );
        YAKL_SCOPE( p0                  , this->p0                  );
        YAKL_SCOPE( grav                , this->grav                );
        YAKL_SCOPE( gamma               , this->gamma               );
        YAKL_SCOPE( C0                  , this->C0                  );
        YAKL_SCOPE( num_tracers         , this->num_tracers         );
        YAKL_SCOPE( idWV                , this->idWV                );

        size_t i_beg = coupler.get_i_beg();
        size_t j_beg = coupler.get_j_beg();

        // Use quadrature to initialize state and tracer data
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          for (int l=0; l < num_state; l++) {
            state(l,hs+k,hs+j,hs+i) = 0.;
          }
          for (int l=0; l < num_tracers; l++) {
            tracers(l,hs+k,hs+j,hs+i) = 0.;
          }
          //Use Gauss-Legendre quadrature
          for (int kk=0; kk<nqpoints; kk++) {
            for (int jj=0; jj<nqpoints; jj++) {
              for (int ii=0; ii<nqpoints; ii++) {
                real x = (i+i_beg+0.5)*dx + (qpoints(ii)-0.5)*dx;
                real y = (j+j_beg+0.5)*dy + (qpoints(jj)-0.5)*dy;   if (sim2d) y = ylen/2;
                real z = (k      +0.5)*dz + (qpoints(kk)-0.5)*dz;
                real rho, u, v, w, theta, rho_v, hr, ht;

                thermal(x,y,z,xlen,ylen,grav,C0,gamma,cp_d,p0,R_d,R_v,rho,u,v,w,theta,rho_v,hr,ht);

                if (sim2d) v = 0;

                real wt = qweights(ii)*qweights(jj)*qweights(kk);
                state(idR,hs+k,hs+j,hs+i) += ( rho - hr )          * wt;
                state(idU,hs+k,hs+j,hs+i) += rho*u                 * wt;
                state(idV,hs+k,hs+j,hs+i) += rho*v                 * wt;
                state(idW,hs+k,hs+j,hs+i) += rho*w                 * wt;
                state(idT,hs+k,hs+j,hs+i) += ( rho*theta - hr*ht ) * wt;
                for (int tr=0; tr < num_tracers; tr++) {
                  if (tr == idWV) { tracers(tr,hs+k,hs+j,hs+i) += rho_v * wt; }
                  else            { tracers(tr,hs+k,hs+j,hs+i) += 0     * wt; }
                }
              }
            }
          }
        });


        // Compute hydrostatic background cell averages using quadrature
        parallel_for( YAKL_AUTO_LABEL() , Bounds<1>(nz) , YAKL_LAMBDA (int k) {
          hy_dens_cells      (k) = 0.;
          hy_dens_theta_cells(k) = 0.;
          for (int kk=0; kk<nqpoints; kk++) {
            real z = (k+0.5)*dz + (qpoints(kk)-0.5)*dz;
            real hr, ht;

            hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht);

            hy_dens_cells      (k) += hr    * qweights(kk);
            hy_dens_theta_cells(k) += hr*ht * qweights(kk);
          }
        });

        // Compute hydrostatic background cell edge values
        parallel_for( YAKL_AUTO_LABEL() , Bounds<1>(nz+1) , YAKL_LAMBDA (int k) {
          real z = k*dz;
          real hr, ht;

          hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht);

          hy_dens_edges      (k) = hr   ;
          hy_dens_theta_edges(k) = hr*ht;
        });

      } else if (init_data_int == DATA_CITY) {

        bc_x = BC_OPEN;
        bc_y = BC_OPEN;
        bc_z = BC_WALL;
        use_immersed_boundaries = true;
        immersed_proportion = 0;

        real height_mean = 30;
        real height_std  = 5;

        int pad_x1 = 3;
        int pad_x2 = 4;
        int pad_y1 = 1;
        int pad_y2 = 1;

        int nblocks_x = floor(xlen/90 -pad_x1-pad_x2);
        int nblocks_y = floor(ylen/270-pad_y1-pad_y2);

        int nbuildings_x = nblocks_x * 2;
        int nbuildings_y = nblocks_y * 8;

        int cells_per_building_x = floor(30/dx);
        int cells_per_building_y = floor(30/dy);

        realHost2d building_heights_host("building_heights",nbuildings_y,nbuildings_x);
        {
          std::random_device rd{};
          std::mt19937 gen{rd()};
          std::normal_distribution<> d{height_mean, height_std};
          for (int j=0; j < nbuildings_y; j++) {
            for (int i=0; i < nbuildings_x; i++) {
              building_heights_host(j,i) = d(gen);
            }
          }
        }
        auto building_heights = building_heights_host.createDeviceCopy();

        // Define quadrature weights and points for 3-point rules
        const int nqpoints = 9;
        SArray<real,1,nqpoints> qpoints;
        SArray<real,1,nqpoints> qweights;

        TransformMatrices::get_gll_points (qpoints );
        TransformMatrices::get_gll_weights(qweights);

        YAKL_SCOPE( init_data_int       , this->init_data_int       );
        YAKL_SCOPE( hy_dens_cells       , this->hy_dens_cells       );
        YAKL_SCOPE( hy_dens_theta_cells , this->hy_dens_theta_cells );
        YAKL_SCOPE( hy_dens_edges       , this->hy_dens_edges       );
        YAKL_SCOPE( hy_dens_theta_edges , this->hy_dens_theta_edges );
        YAKL_SCOPE( dx                  , this->dx                  );
        YAKL_SCOPE( dy                  , this->dy                  );
        YAKL_SCOPE( dz                  , this->dz                  );
        YAKL_SCOPE( xlen                , this->xlen                );
        YAKL_SCOPE( ylen                , this->ylen                );
        YAKL_SCOPE( sim2d               , this->sim2d               );
        YAKL_SCOPE( R_d                 , this->R_d                 );
        YAKL_SCOPE( R_v                 , this->R_v                 );
        YAKL_SCOPE( cp_d                , this->cp_d                );
        YAKL_SCOPE( p0                  , this->p0                  );
        YAKL_SCOPE( grav                , this->grav                );
        YAKL_SCOPE( gamma               , this->gamma               );
        YAKL_SCOPE( C0                  , this->C0                  );
        YAKL_SCOPE( num_tracers         , this->num_tracers         );
        YAKL_SCOPE( idWV                , this->idWV                );
        YAKL_SCOPE( immersed_proportion , this->immersed_proportion );
        YAKL_SCOPE( nz                  , this->nz                  );

        auto i_beg   = coupler.get_i_beg();
        auto j_beg   = coupler.get_j_beg();
        auto nx_glob = coupler.get_nx_glob();
        auto ny_glob = coupler.get_ny_glob();

        // Use quadrature to initialize state and tracer data
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          for (int l=0; l < num_state; l++) {
            state(l,hs+k,hs+j,hs+i) = 0.;
          }
          for (int l=0; l < num_tracers; l++) {
            tracers(l,hs+k,hs+j,hs+i) = 0.;
          }
          //Use Gauss-Legendre quadrature
          for (int kk=0; kk<nqpoints; kk++) {
            for (int jj=0; jj<nqpoints; jj++) {
              for (int ii=0; ii<nqpoints; ii++) {
                real x = (i+i_beg+0.5)*dx + (qpoints(ii)-0.5)*dx;
                real y = (j+j_beg+0.5)*dy + (qpoints(jj)-0.5)*dy;   if (sim2d) y = ylen/2;
                real z = (k      +0.5)*dz + (qpoints(kk)-0.5)*dz;
                real rho, u, v, w, theta, rho_v, hr, ht;

                hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht);

                rho   = hr;
                u     = 20;
                v     = 0;
                w     = 0;
                theta = ht;
                rho_v = 0;

                if (sim2d) v = 0;

                real wt = qweights(ii)*qweights(jj)*qweights(kk);
                state(idR,hs+k,hs+j,hs+i) += ( rho - hr )          * wt;
                state(idU,hs+k,hs+j,hs+i) += rho*u                 * wt;
                state(idV,hs+k,hs+j,hs+i) += rho*v                 * wt;
                state(idW,hs+k,hs+j,hs+i) += rho*w                 * wt;
                state(idT,hs+k,hs+j,hs+i) += ( rho*theta - hr*ht ) * wt;
                for (int tr=0; tr < num_tracers; tr++) {
                  if (tr == idWV) { tracers(tr,hs+k,hs+j,hs+i) += rho_v * wt; }
                  else            { tracers(tr,hs+k,hs+j,hs+i) += 0     * wt; }
                }
              }
            }
          }
          int inorm = (i_beg+i-cells_per_building_x*3*pad_x1)/cells_per_building_x;
          int jnorm = (j_beg+j-cells_per_building_y*9*pad_y1)/cells_per_building_y;
          int iblock = inorm / 3;
          int jblock = jnorm / 9;
          if ( ( inorm >= 0 && inorm < nblocks_x*3 && inorm%3 < 2 ) &&
               ( jnorm >= 0 && jnorm < nblocks_y*9 && jnorm%9 < 8 ) ) {
            if ( k <= std::ceil( building_heights(jblock*8+jnorm%8,iblock*2+inorm%2) / dz ) ) {
              immersed_proportion(k,j,i) = 1;
              state(idU,hs+k,hs+j,hs+i) = 0;
              state(idV,hs+k,hs+j,hs+i) = 0;
              state(idW,hs+k,hs+j,hs+i) = 0;
            }
          }
          if ( k == 0 ) {
            immersed_proportion(k,j,i) = 1;
            state(idU,hs+k,hs+j,hs+i) = 0;
            state(idV,hs+k,hs+j,hs+i) = 0;
            state(idW,hs+k,hs+j,hs+i) = 0;
          }
          // To generate turbulence
          int i1 = 10;
          int i2 = 10;
          int scale = 4;
          real strength = 0.2;
          if (i_beg+i >= i1 && i_beg+i <= i2) {
            if ( ((j_beg+j)/scale)%2 == 0 && (k/scale)%2 == 1 ) {
            immersed_proportion(k,j,i) = strength;
            state(idU,hs+k,hs+j,hs+i) = 0;
            state(idV,hs+k,hs+j,hs+i) = 0;
            state(idW,hs+k,hs+j,hs+i) = 0;
            }
          }
        });


        // Compute hydrostatic background cell averages using quadrature
        parallel_for( YAKL_AUTO_LABEL() , Bounds<1>(nz) , YAKL_LAMBDA (int k) {
          hy_dens_cells      (k) = 0.;
          hy_dens_theta_cells(k) = 0.;
          for (int kk=0; kk<nqpoints; kk++) {
            real z = (k+0.5)*dz + (qpoints(kk)-0.5)*dz;
            real hr, ht;

            hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht);

            hy_dens_cells      (k) += hr    * qweights(kk);
            hy_dens_theta_cells(k) += hr*ht * qweights(kk);
          }
        });

        // Compute hydrostatic background cell edge values
        parallel_for( YAKL_AUTO_LABEL() , Bounds<1>(nz+1) , YAKL_LAMBDA (int k) {
          real z = k*dz;
          real hr, ht;

          hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht);

          hy_dens_edges      (k) = hr   ;
          hy_dens_theta_edges(k) = hr*ht;
        });

      }

      // Convert the initialized state and tracers arrays back to the coupler state
      convert_dynamics_to_coupler( coupler , state , tracers );

      // Output the initial state
      if (out_freq >= 0. ) output( coupler , etime );

      // Some modules might need to use hydrostasis to project values into material boundaries
      // So let's put it into the coupler's data manager just in case
      dm.register_and_allocate<real>("hy_dens_cells"      ,"hydrostatic density cell averages"      ,{nz});
      dm.register_and_allocate<real>("hy_dens_theta_cells","hydrostatic density*theta cell averages",{nz});
      auto dm_hy_dens_cells       = dm.get<real,1>("hy_dens_cells"      );
      auto dm_hy_dens_theta_cells = dm.get<real,1>("hy_dens_theta_cells");
      hy_dens_cells      .deep_copy_to( dm_hy_dens_cells      );
      hy_dens_theta_cells.deep_copy_to( dm_hy_dens_theta_cells);

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

      // Temporary arrays used to compute the initial state for high-CAPE supercell conditions
      real3d quad_temp       ("quad_temp"       ,nz,ord-1,ord);
      real2d hyDensGLL       ("hyDensGLL"       ,nz,ord);
      real2d hyDensThetaGLL  ("hyDensThetaGLL"  ,nz,ord);
      real2d hyDensVapGLL    ("hyDensVapGLL"    ,nz,ord);
      real2d hyPressureGLL   ("hyPressureGLL"   ,nz,ord);
      real1d hyDensCells     ("hyDensCells"     ,nz);
      real1d hyDensThetaCells("hyDensThetaCells",nz);

      real ztop = coupler.get_zlen();

      YAKL_SCOPE( hy_dens_cells       , this->hy_dens_cells       );
      YAKL_SCOPE( hy_dens_theta_cells , this->hy_dens_theta_cells );
      YAKL_SCOPE( hy_dens_edges       , this->hy_dens_edges       );
      YAKL_SCOPE( hy_dens_theta_edges , this->hy_dens_theta_edges );
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
      YAKL_SCOPE( grav                , this->grav                );
      YAKL_SCOPE( gamma               , this->gamma               );
      YAKL_SCOPE( C0                  , this->C0                  );
      YAKL_SCOPE( num_tracers         , this->num_tracers         );
      YAKL_SCOPE( idWV                , this->idWV                );
      YAKL_SCOPE( gll_pts             , this->gll_pts             );
      YAKL_SCOPE( gll_wts             , this->gll_wts             );

      // Compute quadrature term to integrate to get pressure at GLL points
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ord-1,ord) ,
                    YAKL_LAMBDA (int k, int kk, int kkk) {
        // Middle of this cell
        real cellmid   = (k+0.5_fp) * dz;
        // Bottom, top, and middle of the space between these two ord GLL points
        real ord_b    = cellmid + gll_pts(kk  )*dz;
        real ord_t    = cellmid + gll_pts(kk+1)*dz;
        real ord_m    = 0.5_fp * (ord_b + ord_t);
        // Compute grid spacing between these ord GLL points
        real ord_dz   = dz * ( gll_pts(kk+1) - gll_pts(kk) );
        // Compute the locate of this GLL point within the ord GLL points
        real zloc      = ord_m + ord_dz * gll_pts(kkk);
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
          for (int kk=0; kk < ord-1; kk++) {
            real tot = 0;
            for (int kkk=0; kkk < ord; kkk++) {
              tot += quad_temp(k,kk,kkk) * gll_wts(kkk);
            }
            tot *= dz * ( gll_pts(kk+1) - gll_pts(kk) );
            hyPressureGLL(k,kk+1) = hyPressureGLL(k,kk) * exp( tot );
            if (kk == ord-2 && k < nz-1) {
              hyPressureGLL(k+1,0) = hyPressureGLL(k,ord-1);
            }
          }
        }
      });

      // Compute hydrostatic background state at GLL points
      parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz,ord) , YAKL_LAMBDA (int k, int kk) {
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
        real dens_theta = pow( press / C0 , 1._fp / gamma );
        hyDensGLL     (k,kk) = dens;
        hyDensThetaGLL(k,kk) = dens_theta;
        hyDensVapGLL  (k,kk) = dens_vap;
        if (kk == 0) {
          hy_dens_edges      (k) = dens;
          hy_dens_theta_edges(k) = dens_theta;
        }
        if (k == nz-1 && kk == ord-1) {
          hy_dens_edges      (k+1) = dens;
          hy_dens_theta_edges(k+1) = dens_theta;
        }
      });

      // Compute hydrostatic background state over cells
      parallel_for( YAKL_AUTO_LABEL() , Bounds<1>(nz) , YAKL_LAMBDA (int k) {
        real press_tot      = 0;
        real dens_tot       = 0;
        real dens_vap_tot   = 0;
        real dens_theta_tot = 0;
        for (int kk=0; kk < ord; kk++) {
          press_tot      += hyPressureGLL (k,kk) * gll_wts(kk);
          dens_tot       += hyDensGLL     (k,kk) * gll_wts(kk);
          dens_vap_tot   += hyDensVapGLL  (k,kk) * gll_wts(kk);
          dens_theta_tot += hyDensThetaGLL(k,kk) * gll_wts(kk);
        }
        real press      = press_tot;
        real dens       = dens_tot;
        real dens_vap   = dens_vap_tot;
        real dens_theta = dens_theta_tot;
        real dens_dry   = dens - dens_vap;
        real R          = dens_dry / dens * R_d + dens_vap / dens * R_v;
        real temp       = press / (dens * R);
        real qv         = dens_vap / dens_dry;
        real zloc       = (k+0.5_fp)*dz;
        real press_tmp  = init_supercell_pressure_dry(zloc, z_0, z_trop, ztop, T_0, T_trop, T_top, p_0, R_d, grav);
        real qvs        = init_supercell_sat_mix_dry(press_tmp, temp);
        real relhum     = qv / qvs;
        real T          = temp - 273;
        real a          = 17.27;
        real b          = 237.7;
        real tdew       = b * ( a*T / (b + T) + log(relhum) ) / ( a - ( a*T / (b+T) + log(relhum) ) );
        // These are used in the rest of the model
        hy_dens_cells      (k) = dens;
        hy_dens_theta_cells(k) = dens_theta;
      });

      size_t i_beg = coupler.get_i_beg();
      size_t j_beg = coupler.get_j_beg();





      // Initialize the state
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        state(idR,hs+k,hs+j,hs+i) = 0;
        state(idU,hs+k,hs+j,hs+i) = 0;
        state(idV,hs+k,hs+j,hs+i) = 0;
        state(idW,hs+k,hs+j,hs+i) = 0;
        state(idT,hs+k,hs+j,hs+i) = 0;
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i) = 0; }
        for (int kk=0; kk < ord; kk++) {
          for (int jj=0; jj < ord; jj++) {
            for (int ii=0; ii < ord; ii++) {
              real xloc = (i+i_beg+0.5_fp)*dx + gll_pts(ii)*dx;
              real yloc = (j+j_beg+0.5_fp)*dy + gll_pts(jj)*dy;
              real zloc = (k      +0.5_fp)*dz + gll_pts(kk)*dz;

              if (sim2d) yloc = ylen/2;

              real dens = hyDensGLL(k,kk);

              real uvel;
              real constexpr zs = 5000;
              real constexpr us = 30;
              real constexpr uc = 15;
              if (zloc < zs) {
                uvel = us * (zloc / zs) - uc;
              } else {
                uvel = us - uc;
              }

              real vvel       = 0;
              real wvel       = 0;
              real dens_vap   = hyDensVapGLL  (k,kk);
              real dens_theta = hyDensThetaGLL(k,kk);

              real factor = gll_wts(ii) * gll_wts(jj) * gll_wts(kk);
              state  (idR ,hs+k,hs+j,hs+i) += (dens - hyDensGLL(k,kk))            * factor;
              state  (idU ,hs+k,hs+j,hs+i) += dens * uvel                         * factor;
              state  (idV ,hs+k,hs+j,hs+i) += dens * vvel                         * factor;
              state  (idW ,hs+k,hs+j,hs+i) += dens * wvel                         * factor;
              state  (idT ,hs+k,hs+j,hs+i) += (dens_theta - hyDensThetaGLL(k,kk)) * factor;
              tracers(idWV,hs+k,hs+j,hs+i) += dens_vap                            * factor;
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

      YAKL_SCOPE( hy_dens_cells       , this->hy_dens_cells       );
      YAKL_SCOPE( hy_dens_theta_cells , this->hy_dens_theta_cells );
      YAKL_SCOPE( R_d                 , this->R_d                 );
      YAKL_SCOPE( R_v                 , this->R_v                 );
      YAKL_SCOPE( gamma               , this->gamma               );
      YAKL_SCOPE( C0                  , this->C0                  );
      YAKL_SCOPE( num_tracers         , this->num_tracers         );
      YAKL_SCOPE( idWV                , this->idWV                );
      YAKL_SCOPE( tracer_adds_mass    , this->tracer_adds_mass    );

      // Convert from state and tracers arrays to the coupler's data
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        real rho   = state(idR,hs+k,hs+j,hs+i) + hy_dens_cells(k);
        real u     = state(idU,hs+k,hs+j,hs+i) / rho;
        real v     = state(idV,hs+k,hs+j,hs+i) / rho;
        real w     = state(idW,hs+k,hs+j,hs+i) / rho;
        real theta = ( state(idT,hs+k,hs+j,hs+i) + hy_dens_theta_cells(k) ) / rho;
        real press = C0 * pow( rho*theta , gamma );

        real rho_v = tracers(idWV,hs+k,hs+j,hs+i);
        real rho_d = rho;
        for (int tr=0; tr < num_tracers; tr++) {
          if (tracer_adds_mass(tr)) rho_d -= tracers(tr,hs+k,hs+j,hs+i);
        }
        real temp = press / ( rho_d * R_d + rho_v * R_v );

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

      YAKL_SCOPE( hy_dens_cells       , this->hy_dens_cells       );
      YAKL_SCOPE( hy_dens_theta_cells , this->hy_dens_theta_cells );
      YAKL_SCOPE( R_d                 , this->R_d                 );
      YAKL_SCOPE( R_v                 , this->R_v                 );
      YAKL_SCOPE( gamma               , this->gamma               );
      YAKL_SCOPE( C0                  , this->C0                  );
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
        real press = rho_d * R_d * temp + rho_v * R_v * temp;

        real rho = rho_d;
        for (int tr=0; tr < num_tracers; tr++) {
          if (tracer_adds_mass(tr)) rho += dm_tracers(tr,k,j,i);
        }
        real theta = pow( press/C0 , 1._fp / gamma ) / rho;

        state(idR,hs+k,hs+j,hs+i) = rho - hy_dens_cells(k);
        state(idU,hs+k,hs+j,hs+i) = rho * u;
        state(idV,hs+k,hs+j,hs+i) = rho * v;
        state(idW,hs+k,hs+j,hs+i) = rho * w;
        state(idT,hs+k,hs+j,hs+i) = rho * theta - hy_dens_theta_cells(k);
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

      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto num_tracers = coupler.get_num_tracers();
      auto sim2d       = coupler.is_sim2d();
      auto px          = coupler.get_px();
      auto py          = coupler.get_py();
      auto nproc_x     = coupler.get_nproc_x();
      auto nproc_y     = coupler.get_nproc_y();

      YAKL_SCOPE( bc_x , this->bc_x );
      YAKL_SCOPE( bc_y , this->bc_y );
      YAKL_SCOPE( bc_z , this->bc_z );

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

      ////////////////////////////////////
      // Begin boundary conditions
      ////////////////////////////////////
      if (bc_z == BC_PERIODIC) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(hs,ny,nx) , YAKL_LAMBDA (int kk, int j, int i) {
          for (int l=0; l < num_state; l++) {
            state(l,      kk,hs+j,hs+i) = state(l,      kk+nz,hs+j,hs+i);
            state(l,hs+nz+kk,hs+j,hs+i) = state(l,hs+nz+kk-nz,hs+j,hs+i);
          }
          for (int l=0; l < num_tracers; l++) {
            tracers(l,      kk,hs+j,hs+i) = tracers(l,      kk+nz,hs+j,hs+i);
            tracers(l,hs+nz+kk,hs+j,hs+i) = tracers(l,hs+nz+kk-nz,hs+j,hs+i);
          }
        });
      } else if (bc_z == BC_WALL || bc_z == BC_OPEN) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(hs,ny,nx) , YAKL_LAMBDA (int kk, int j, int i) {
          for (int l=0; l < num_state; l++) {
            if (l == idW && bc_z == BC_WALL) {
              state(l,      kk,hs+j,hs+i) = 0;
              state(l,hs+nz+kk,hs+j,hs+i) = 0;
            } else {
              state(l,      kk,hs+j,hs+i) = state(l,hs+0   ,hs+j,hs+i);
              state(l,hs+nz+kk,hs+j,hs+i) = state(l,hs+nz-1,hs+j,hs+i);
            }
          }
          for (int l=0; l < num_tracers; l++) {
            tracers(l,      kk,hs+j,hs+i) = tracers(l,hs+0   ,hs+j,hs+i);
            tracers(l,hs+nz+kk,hs+j,hs+i) = tracers(l,hs+nz-1,hs+j,hs+i);
          }
        });
      }
      if (bc_x == BC_WALL || bc_x == BC_OPEN) {
        if (px == 0) {
          parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,hs) , YAKL_LAMBDA (int k, int j, int ii) {
            for (int l=0; l < num_state; l++) {
              if (l == idU && bc_x == BC_WALL) { state(l,hs+k,hs+j,ii) = 0; }
              else                             { state(l,hs+k,hs+j,ii) = state(l,hs+k,hs+j,hs+0); }
            }
            for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,ii) = tracers(l,hs+k,hs+j,hs+0); }
          });
        }
        if (px == nproc_x-1) {
          parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,hs) , YAKL_LAMBDA (int k, int j, int ii) {
            for (int l=0; l < num_state; l++) {
              if (l == idU && bc_x == BC_WALL) { state(l,hs+k,hs+j,hs+nx+ii) = 0; }
              else                             { state(l,hs+k,hs+j,hs+nx+ii) = state(l,hs+k,hs+j,hs+nx-1); }
            }
            for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+nx+ii) = tracers(l,hs+k,hs+j,hs+nx-1); }
          });
        }
      }
      if (bc_y == BC_WALL || bc_y == BC_OPEN) {
        if (py == 0) {
          parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,hs,nx) , YAKL_LAMBDA (int k, int jj, int i) {
            for (int l=0; l < num_state; l++) {
              if (l == idV && bc_y == BC_WALL) { state(l,hs+k,jj,hs+i) = 0; }
              else                             { state(l,hs+k,jj,hs+i) = state(l,hs+k,hs+0,hs+i); }
            }
            for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,jj,hs+i) = tracers(l,hs+k,hs+0,hs+i); }
          });
        }
        if (py == nproc_y-1) {
          parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,hs,nx) , YAKL_LAMBDA (int k, int jj, int i) {
            for (int l=0; l < num_state; l++) {
              if (l == idV && bc_y == BC_WALL) { state(l,hs+k,hs+ny+jj,hs+i) = 0; }
              else                             { state(l,hs+k,hs+ny+jj,hs+i) = state(l,hs+k,hs+ny-1,hs+i); }
            }
            for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+ny+jj,hs+i) = tracers(l,hs+k,hs+ny-1,hs+i); }
          });
        }
      }
    }


    void edge_exchange(core::Coupler const &coupler , real5d const &state_limits_x , real5d const &tracers_limits_x ,
                                                      real5d const &state_limits_y , real5d const &tracers_limits_y ,
                                                      real5d const &state_limits_z , real5d const &tracers_limits_z ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto num_tracers = coupler.get_num_tracers();
      auto sim2d       = coupler.is_sim2d();
      auto px          = coupler.get_px();
      auto py          = coupler.get_py();
      auto nproc_x     = coupler.get_nproc_x();
      auto nproc_y     = coupler.get_nproc_y();

      YAKL_SCOPE( bc_x , this->bc_x );
      YAKL_SCOPE( bc_y , this->bc_y );
      YAKL_SCOPE( bc_z , this->bc_z );

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

      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(npack,nz,ny) , YAKL_LAMBDA (int v, int k, int j) {
        if (v < num_state) {
          state_limits_x  (v          ,0,k,j,0 ) = edge_recv_buf_W(v,k,j);
          state_limits_x  (v          ,1,k,j,nx) = edge_recv_buf_E(v,k,j);
        } else {
          tracers_limits_x(v-num_state,0,k,j,0 ) = edge_recv_buf_W(v,k,j);
          tracers_limits_x(v-num_state,1,k,j,nx) = edge_recv_buf_E(v,k,j);
        }
      });

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(npack,nz,nx) , YAKL_LAMBDA (int v, int k, int i) {
          if (v < num_state) {
            state_limits_y  (v          ,0,k,0 ,i) = edge_recv_buf_S(v,k,i);
            state_limits_y  (v          ,1,k,ny,i) = edge_recv_buf_N(v,k,i);
          } else {
            tracers_limits_y(v-num_state,0,k,0 ,i) = edge_recv_buf_S(v,k,i);
            tracers_limits_y(v-num_state,1,k,ny,i) = edge_recv_buf_N(v,k,i);
          }
        });
      }

      /////////////////////////////////
      // Begin boundary conditions
      /////////////////////////////////
      if (bc_z == BC_PERIODIC) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
          for (int l=0; l < num_state; l++) {
            state_limits_z(l,0,0 ,j,i) = state_limits_z(l,0,nz,j,i);
            state_limits_z(l,1,nz,j,i) = state_limits_z(l,1,0 ,j,i);
          }
          for (int l=0; l < num_tracers; l++) {
            tracers_limits_z(l,0,0 ,j,i) = tracers_limits_z(l,0,nz,j,i);
            tracers_limits_z(l,1,nz,j,i) = tracers_limits_z(l,1,0 ,j,i);
          }
        });
      } else if (bc_z == BC_WALL || bc_z == BC_OPEN) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
          for (int l=0; l < num_state; l++) {
            if (l == idW && bc_z == BC_WALL) {
              state_limits_z(l,0,0 ,j,i) = 0;
              state_limits_z(l,1,0 ,j,i) = 0;
              state_limits_z(l,0,nz,j,i) = 0;
              state_limits_z(l,1,nz,j,i) = 0;
            } else {
              state_limits_z(l,0,0 ,j,i) = state_limits_z(l,1,0 ,j,i);
              state_limits_z(l,1,nz,j,i) = state_limits_z(l,0,nz,j,i);
            }
          }
          for (int l=0; l < num_tracers; l++) {
            tracers_limits_z(l,0,0 ,j,i) = tracers_limits_z(l,1,0 ,j,i);
            tracers_limits_z(l,1,nz,j,i) = tracers_limits_z(l,0,nz,j,i);
          }
        });
      }
      if (bc_x == BC_WALL || bc_x == BC_OPEN) {
        if (px == 0) {
          parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz,ny) , YAKL_LAMBDA (int k, int j) {
            for (int l=0; l < num_state; l++) {
              if (l == idU && bc_x == BC_WALL) { state_limits_x(l,0,k,j,0) = 0; state_limits_x(l,1,k,j,0) = 0; }
              else                             { state_limits_x(l,0,k,j,0) = state_limits_x(l,1,k,j,0); }
            }
            for (int l=0; l < num_tracers; l++) { tracers_limits_x(l,0,k,j,0) = tracers_limits_x(l,1,k,j,0); }
          });
        } else if (px == nproc_x-1) {
          parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz,ny) , YAKL_LAMBDA (int k, int j) {
            for (int l=0; l < num_state; l++) {
              if (l == idU && bc_x == BC_WALL) { state_limits_x(l,0,k,j,nx) = 0; state_limits_x(l,1,k,j,nx) = 0; }
              else                             { state_limits_x(l,1,k,j,nx) = state_limits_x(l,0,k,j,nx); }
            }
            for (int l=0; l < num_tracers; l++) { tracers_limits_x(l,1,k,j,nx) = tracers_limits_x(l,0,k,j,nx); }
          });
        }
      }
      if (bc_y == BC_WALL || bc_y == BC_OPEN) {
        if (py == 0) {
          parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz,nx) , YAKL_LAMBDA (int k, int i) {
            for (int l=0; l < num_state; l++) {
              if (l == idV && bc_y == BC_WALL) { state_limits_y(l,0,k,0,i) = 0; state_limits_y(l,1,k,0,i) = 0; }
              else                             { state_limits_y(l,0,k,0,i) = state_limits_y(l,1,k,0,i); }
            }
            for (int l=0; l < num_tracers; l++) { tracers_limits_y(l,0,k,0,i) = tracers_limits_y(l,1,k,0,i); }
          });
        } else if (py == nproc_y-1) {
          parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz,nx) , YAKL_LAMBDA (int k, int i) {
            for (int l=0; l < num_state; l++) {
              if (l == idV && bc_y == BC_WALL) { state_limits_y(l,0,k,ny,i) = 0; state_limits_y(l,1,k,ny,i) = 0; }
              else                             { state_limits_y(l,1,k,ny,i) = state_limits_y(l,0,k,ny,i); }
            }
            for (int l=0; l < num_tracers; l++) { tracers_limits_y(l,1,k,ny,i) = tracers_limits_y(l,0,k,ny,i); }
          });
        }
      }
    }


    // Creates initial data at a point in space for the rising moist thermal test case
    YAKL_INLINE static void thermal(real x, real y, real z, real xlen, real ylen, real grav, real C0, real gamma,
                                    real cp, real p0, real R_d, real R_v, real &rho, real &u, real &v, real &w,
                                    real &theta, real &rho_v, real &hr, real &ht) {
      hydro_const_theta(z,grav,C0,cp,p0,gamma,R_d,hr,ht);
      real rho_d   = hr;
      u            = 0.;
      v            = 0.;
      w            = 0.;
      real theta_d = ht + sample_ellipse_cosine(2._fp  ,  x,y,z  ,  xlen/2,ylen/2,2000.  ,  2000.,2000.,2000.);
      real p_d     = C0 * pow( rho_d*theta_d , gamma );
      real temp    = p_d / rho_d / R_d;
      real sat_pv  = saturation_vapor_pressure(temp);
      real sat_rv  = sat_pv / R_v / temp;
      rho_v        = sample_ellipse_cosine(0.8_fp  ,  x,y,z  ,  xlen/2,ylen/2,2000.  ,  2000.,2000.,2000.) * sat_rv;
      real p       = rho_d * R_d * temp + rho_v * R_v * temp;
      rho          = rho_d + rho_v;
      theta        = std::pow( p / C0 , 1._fp / gamma ) / rho;
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


