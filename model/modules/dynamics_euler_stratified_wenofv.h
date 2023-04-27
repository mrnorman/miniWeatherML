
#pragma once

#include "main_header.h"
#include "MultipleFields.h"
#include "TransformMatrices.h"
#include "WenoLimiter.h"
#include <random>
#include <sstream>

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
    int  static constexpr DATA_BUILDING  = 3;

    int  static constexpr BC_PERIODIC = 0;
    int  static constexpr BC_OPEN     = 1;
    int  static constexpr BC_WALL     = 2;

    // Hydrostatic background profiles for density and potential temperature as cell averages and cell edge values
    real2d      hy_dens_cells;
    real2d      hy_dens_theta_cells;
    real2d      hy_dens_edges;
    real2d      hy_dens_theta_edges;
    real        etime;         // Elapsed time
    real        out_freq;      // Frequency out file output
    int         num_out;       // Number of outputs produced thus far
    int         init_data_int; // Integer representation of the type of initial data to use (test case)

    int         idWV;              // Index number for water vapor in the tracers array
    bool1d      tracer_adds_mass;  // Whether a tracer adds mass to the full density
    bool1d      tracer_positive;   // Whether a tracer needs to remain non-negative

    SArray<real,1,ord>            gll_pts;          // GLL point locations in domain [-0.5 , 0.5]
    SArray<real,1,ord>            gll_wts;          // GLL weights normalized to sum to 1
    SArray<real,2,ord,2  >        coefs_to_gll;     // Matrix to convert ord poly coefs to two GLL points


    // Compute the maximum stable time step using very conservative assumptions about max wind speed
    real compute_time_step( core::Coupler const &coupler ) const {
      auto dx = coupler.get_dx();
      auto dy = coupler.get_dy();
      auto dz = coupler.get_dz();
      real constexpr maxwave = 350 + 80;
      real cfl = 0.6;
      return cfl * std::min( std::min( dx , dy ) , dz ) / maxwave;
    }


    // Perform a single time step using SSPRK3 time stepping
    void time_step(core::Coupler &coupler, real &dt_phys) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;
      using yakl::intrinsics::maxval;
      using yakl::intrinsics::abs;

      YAKL_SCOPE( tracer_positive , this->tracer_positive );

      auto num_tracers = coupler.get_num_tracers();
      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();

      // Create arrays to hold state and tracers with halos on the left and right of the domain
      // Cells [0:hs-1] are the left halos, and cells [nx+hs:nx+2*hs-1] are the right halos
      real5d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real5d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs,nens);

      // Populate the state and tracers arrays using data from the coupler, convert to the dycore's desired state
      convert_coupler_to_dynamics( coupler , state , tracers );

      // Get the max stable time step for the dynamics. dt_phys might be > dt_dyn, meaning we would need to sub-cycle
      real dt_dyn = compute_time_step( coupler );

      // Get the number of sub-cycles we need, and set the dynamics time step accordingly
      int ncycles = (int) std::ceil( dt_phys / dt_dyn );
      dt_dyn = dt_phys / ncycles;

      for (int icycle = 0; icycle < ncycles; icycle++) {
        // SSPRK3 requires temporary arrays to hold intermediate state and tracers arrays
        real5d state_tmp   ("state_tmp"   ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs,nens);
        real5d state_tend  ("state_tend"  ,num_state  ,nz     ,ny     ,nx     ,nens);
        real5d tracers_tmp ("tracers_tmp" ,num_tracers,nz+2*hs,ny+2*hs,nx+2*hs,nens);
        real5d tracers_tend("tracers_tend",num_tracers,nz     ,ny     ,nx     ,nens);
        //////////////
        // Stage 1
        //////////////
        compute_tendencies( coupler , state     , state_tend , tracers     , tracers_tend , dt_dyn );
        // Apply tendencies
        parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state_tmp  (l,hs+k,hs+j,hs+i,iens) = state  (l,hs+k,hs+j,hs+i,iens) + dt_dyn * state_tend  (l,k,j,i,iens);
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers_tmp(l,hs+k,hs+j,hs+i,iens) = tracers(l,hs+k,hs+j,hs+i,iens) + dt_dyn * tracers_tend(l,k,j,i,iens);
            // For machine precision negative values after FCT-enforced positivity application
            if (tracer_positive(l)) {
              tracers_tmp(l,hs+k,hs+j,hs+i,iens) = std::max( 0._fp , tracers_tmp(l,hs+k,hs+j,hs+i,iens) );
            }
          }
        });
        //////////////
        // Stage 2
        //////////////
        compute_tendencies( coupler , state_tmp , state_tend , tracers_tmp , tracers_tend , (1._fp/4._fp) * dt_dyn );
        // Apply tendencies
        parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state_tmp  (l,hs+k,hs+j,hs+i,iens) = (3._fp/4._fp) * state      (l,hs+k,hs+j,hs+i,iens) + 
                                                 (1._fp/4._fp) * state_tmp  (l,hs+k,hs+j,hs+i,iens) +
                                                 (1._fp/4._fp) * dt_dyn * state_tend  (l,k,j,i,iens);
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers_tmp(l,hs+k,hs+j,hs+i,iens) = (3._fp/4._fp) * tracers    (l,hs+k,hs+j,hs+i,iens) + 
                                                 (1._fp/4._fp) * tracers_tmp(l,hs+k,hs+j,hs+i,iens) +
                                                 (1._fp/4._fp) * dt_dyn * tracers_tend(l,k,j,i,iens);
            // For machine precision negative values after FCT-enforced positivity application
            if (tracer_positive(l)) {
              tracers_tmp(l,hs+k,hs+j,hs+i,iens) = std::max( 0._fp , tracers_tmp(l,hs+k,hs+j,hs+i,iens) );
            }
          }
        });
        //////////////
        // Stage 3
        //////////////
        compute_tendencies( coupler , state_tmp , state_tend , tracers_tmp , tracers_tend , (2._fp/3._fp) * dt_dyn );
        // Apply tendencies
        parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state      (l,hs+k,hs+j,hs+i,iens) = (1._fp/3._fp) * state      (l,hs+k,hs+j,hs+i,iens) +
                                                 (2._fp/3._fp) * state_tmp  (l,hs+k,hs+j,hs+i,iens) +
                                                 (2._fp/3._fp) * dt_dyn * state_tend  (l,k,j,i,iens);
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers    (l,hs+k,hs+j,hs+i,iens) = (1._fp/3._fp) * tracers    (l,hs+k,hs+j,hs+i,iens) +
                                                 (2._fp/3._fp) * tracers_tmp(l,hs+k,hs+j,hs+i,iens) +
                                                 (2._fp/3._fp) * dt_dyn * tracers_tend(l,k,j,i,iens);
            // For machine precision negative values after FCT-enforced positivity application
            if (tracer_positive(l)) {
              tracers    (l,hs+k,hs+j,hs+i,iens) = std::max( 0._fp , tracers    (l,hs+k,hs+j,hs+i,iens) );
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
    void compute_tendencies( core::Coupler &coupler , real5d const &state   , real5d const &state_tend   ,
                                                      real5d const &tracers , real5d const &tracers_tend , real dt ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;
      using std::min;
      using std::max;

      auto use_immersed_boundaries = coupler.get_option<bool>("use_immersed_boundaries");
      auto earthrot                = coupler.get_option<real>("earthrot");
      auto fcor                    = 2*earthrot*sin(coupler.get_option<real>("latitude"));
      auto nens                    = coupler.get_nens();
      auto nx                      = coupler.get_nx();
      auto ny                      = coupler.get_ny();
      auto nz                      = coupler.get_nz();
      auto dx                      = coupler.get_dx();
      auto dy                      = coupler.get_dy();
      auto dz                      = coupler.get_dz();
      auto sim2d                   = coupler.is_sim2d();
      auto C0                      = coupler.get_option<real>("C0"     );
      auto gamma                   = coupler.get_option<real>("gamma_d");
      auto grav                    = coupler.get_option<real>("grav"   );
      auto enable_gravity = coupler.get_option<bool>("enable_gravity",true);
      auto num_tracers             = coupler.get_num_tracers();

      // The store a single values flux at cell edges
      auto &dm = coupler.get_data_manager_readwrite();
      auto state_flux_x        = dm.get<real,5>("state_flux_x"  );
      auto state_flux_y        = dm.get<real,5>("state_flux_y"  );
      auto state_flux_z        = dm.get<real,5>("state_flux_z"  );
      auto tracers_flux_x      = dm.get<real,5>("tracers_flux_x");
      auto tracers_flux_y      = dm.get<real,5>("tracers_flux_y");
      auto tracers_flux_z      = dm.get<real,5>("tracers_flux_z");
      auto immersed_proportion = dm.get<real,4>("immersed_proportion");

      // A slew of things to bring from class scope into local scope so that lambdas copy them by value to the GPU
      YAKL_SCOPE( hy_dens_cells              , this->hy_dens_cells              );
      YAKL_SCOPE( hy_dens_theta_cells        , this->hy_dens_theta_cells        );
      YAKL_SCOPE( hy_dens_edges              , this->hy_dens_edges              );
      YAKL_SCOPE( hy_dens_theta_edges        , this->hy_dens_theta_edges        );
      YAKL_SCOPE( tracer_positive            , this->tracer_positive            );
      YAKL_SCOPE( coefs_to_gll               , this->coefs_to_gll               );

      // Since tracers are full mass, it's helpful before reconstruction to remove the background density for potentially
      // more accurate reconstructions of tracer concentrations
      parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        state(idU,hs+k,hs+j,hs+i,iens) /= ( state(idR,hs+k,hs+j,hs+i,iens) + hy_dens_cells(k,iens) );
        state(idV,hs+k,hs+j,hs+i,iens) /= ( state(idR,hs+k,hs+j,hs+i,iens) + hy_dens_cells(k,iens) );
        state(idW,hs+k,hs+j,hs+i,iens) /= ( state(idR,hs+k,hs+j,hs+i,iens) + hy_dens_cells(k,iens) );
        for (int tr=0; tr < num_tracers; tr++) {
          tracers(tr,hs+k,hs+j,hs+i,iens) /= ( state(idR,hs+k,hs+j,hs+i,iens) + hy_dens_cells(k,iens) );
        }
      });

      halo_exchange( coupler , state , tracers );

      // These arrays store high-order-accurate samples of the state and tracers at cell edges after cell-centered recon
      real6d state_limits_x  ("state_limits_x"  ,num_state  ,2,nz  ,ny  ,nx+1,nens);
      real6d state_limits_y  ("state_limits_y"  ,num_state  ,2,nz  ,ny+1,nx  ,nens);
      real6d state_limits_z  ("state_limits_z"  ,num_state  ,2,nz+1,ny  ,nx  ,nens);
      real6d tracers_limits_x("tracers_limits_x",num_tracers,2,nz  ,ny  ,nx+1,nens);
      real6d tracers_limits_y("tracers_limits_y",num_tracers,2,nz  ,ny+1,nx  ,nens);
      real6d tracers_limits_z("tracers_limits_z",num_tracers,2,nz+1,ny  ,nx  ,nens);

      weno::WenoLimiter<ord> limiter;

      // Compute samples of state and tracers at cell edges using cell-centered reconstructions at high-order with WENO
      // At the end of this, we will have two samples per cell edge in each dimension, one from each adjacent cell.
      parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        ////////////////////////////////////////////////////////
        // X-direction
        ////////////////////////////////////////////////////////
        // State
        for (int l=0; l < num_state; l++) {
          // Gather the stencil of cell averages, and use WENO to compute values at the cell edges (i.e., 2 GLL points)
          SArray<real,1,ord> stencil;
          SArray<real,1,2>   gll;
          for (int s=0; s < ord; s++) { stencil(s) = state(l,hs+k,hs+j,i+s,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter);
          state_limits_x(l,1,k,j,i  ,iens) = gll(0);
          state_limits_x(l,0,k,j,i+1,iens) = gll(1);
        }
        // Add back hydrostatic backgrounds to density and density*theta because only perturbations were reconstructed
        state_limits_x(idR,1,k,j,i  ,iens) += hy_dens_cells(k,iens);
        state_limits_x(idR,0,k,j,i+1,iens) += hy_dens_cells(k,iens);
        state_limits_x(idU,1,k,j,i  ,iens) *= state_limits_x(idR,1,k,j,i  ,iens);
        state_limits_x(idU,0,k,j,i+1,iens) *= state_limits_x(idR,0,k,j,i+1,iens);
        state_limits_x(idV,1,k,j,i  ,iens) *= state_limits_x(idR,1,k,j,i  ,iens);
        state_limits_x(idV,0,k,j,i+1,iens) *= state_limits_x(idR,0,k,j,i+1,iens);
        state_limits_x(idW,1,k,j,i  ,iens) *= state_limits_x(idR,1,k,j,i  ,iens);
        state_limits_x(idW,0,k,j,i+1,iens) *= state_limits_x(idR,0,k,j,i+1,iens);
        state_limits_x(idT,1,k,j,i  ,iens) += hy_dens_theta_cells(k,iens);
        state_limits_x(idT,0,k,j,i+1,iens) += hy_dens_theta_cells(k,iens);
        // Tracers
        for (int l=0; l < num_tracers; l++) {
          // Gather the stencil of cell averages, and use WENO to compute values at the cell edges (i.e., 2 GLL points)
          SArray<real,1,ord> stencil;
          SArray<real,1,2>   gll;
          for (int s=0; s < ord; s++) { stencil(s) = tracers(l,hs+k,hs+j,i+s,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter);
          tracers_limits_x(l,1,k,j,i  ,iens) = gll(0) * state_limits_x(idR,1,k,j,i  ,iens);
          tracers_limits_x(l,0,k,j,i+1,iens) = gll(1) * state_limits_x(idR,0,k,j,i+1,iens);
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
            for (int s=0; s < ord; s++) { stencil(s) = state(l,hs+k,j+s,hs+i,iens); }
            reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter);
            state_limits_y(l,1,k,j  ,i,iens) = gll(0);
            state_limits_y(l,0,k,j+1,i,iens) = gll(1);
          }
          // Add back hydrostatic backgrounds to density and density*theta because only perturbations were reconstructed
          state_limits_y(idR,1,k,j  ,i,iens) += hy_dens_cells(k,iens);
          state_limits_y(idR,0,k,j+1,i,iens) += hy_dens_cells(k,iens);
          state_limits_y(idU,1,k,j  ,i,iens) *= state_limits_y(idR,1,k,j  ,i,iens);
          state_limits_y(idU,0,k,j+1,i,iens) *= state_limits_y(idR,0,k,j+1,i,iens);
          state_limits_y(idV,1,k,j  ,i,iens) *= state_limits_y(idR,1,k,j  ,i,iens);
          state_limits_y(idV,0,k,j+1,i,iens) *= state_limits_y(idR,0,k,j+1,i,iens);
          state_limits_y(idW,1,k,j  ,i,iens) *= state_limits_y(idR,1,k,j  ,i,iens);
          state_limits_y(idW,0,k,j+1,i,iens) *= state_limits_y(idR,0,k,j+1,i,iens);
          state_limits_y(idT,1,k,j  ,i,iens) += hy_dens_theta_cells(k,iens);
          state_limits_y(idT,0,k,j+1,i,iens) += hy_dens_theta_cells(k,iens);
          // Tracers
          for (int l=0; l < num_tracers; l++) {
            // Gather the stencil of cell averages, and use WENO to compute values at the cell edges (i.e., 2 GLL points)
            SArray<real,1,ord> stencil;
            SArray<real,1,2>   gll;
            for (int s=0; s < ord; s++) { stencil(s) = tracers(l,hs+k,j+s,hs+i,iens); }
            reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter);
            tracers_limits_y(l,1,k,j  ,i,iens) = gll(0) * state_limits_y(idR,1,k,j  ,i,iens);
            tracers_limits_y(l,0,k,j+1,i,iens) = gll(1) * state_limits_y(idR,0,k,j+1,i,iens);
          }
        } else {
          for (int l=0; l < num_state; l++) {
            state_limits_y(l,1,k,j  ,i,iens) = 0;
            state_limits_y(l,0,k,j+1,i,iens) = 0;
          }
          for (int l=0; l < num_tracers; l++) {
            tracers_limits_y(l,1,k,j  ,i,iens) = 0;
            tracers_limits_y(l,0,k,j+1,i,iens) = 0;
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
          for (int s=0; s < ord; s++) { stencil(s) = state(l,k+s,hs+j,hs+i,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter);
          state_limits_z(l,1,k  ,j,i,iens) = gll(0);
          state_limits_z(l,0,k+1,j,i,iens) = gll(1);
        }
        // Add back hydrostatic backgrounds to density and density*theta because only perturbations were reconstructed
        state_limits_z(idR,1,k  ,j,i,iens) += hy_dens_edges(k  ,iens);
        state_limits_z(idR,0,k+1,j,i,iens) += hy_dens_edges(k+1,iens);
        state_limits_z(idU,1,k  ,j,i,iens) *= state_limits_z(idR,1,k  ,j,i,iens);
        state_limits_z(idU,0,k+1,j,i,iens) *= state_limits_z(idR,0,k+1,j,i,iens);
        state_limits_z(idV,1,k  ,j,i,iens) *= state_limits_z(idR,1,k  ,j,i,iens);
        state_limits_z(idV,0,k+1,j,i,iens) *= state_limits_z(idR,0,k+1,j,i,iens);
        state_limits_z(idW,1,k  ,j,i,iens) *= state_limits_z(idR,1,k  ,j,i,iens);
        state_limits_z(idW,0,k+1,j,i,iens) *= state_limits_z(idR,0,k+1,j,i,iens);
        state_limits_z(idT,1,k  ,j,i,iens) += hy_dens_theta_edges(k  ,iens);
        state_limits_z(idT,0,k+1,j,i,iens) += hy_dens_theta_edges(k+1,iens);
        // Tracers
        for (int l=0; l < num_tracers; l++) {
          // Gather the stencil of cell averages, and use WENO to compute values at the cell edges (i.e., 2 GLL points)
          SArray<real,1,ord> stencil;
          SArray<real,1,2>   gll;
          for (int s=0; s < ord; s++) { stencil(s) = tracers(l,k+s,hs+j,hs+i,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter);
          tracers_limits_z(l,1,k  ,j,i,iens) = gll(0) * state_limits_z(idR,1,k  ,j,i,iens);
          tracers_limits_z(l,0,k+1,j,i,iens) = gll(1) * state_limits_z(idR,0,k+1,j,i,iens);
        }
      });

      edge_exchange( coupler , state_limits_x , tracers_limits_x ,
                               state_limits_y , tracers_limits_y ,
                               state_limits_z , tracers_limits_z );

      // Use upwind Riemann solver to reconcile discontinuous limits of state and tracers at each cell edges
      parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz+1,ny+1,nx+1,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        // X-direction
        if (j < ny && k < nz) {
          // Acoustically upwind mass flux and pressure
          real ru_L = state_limits_x(idU,0,k,j,i,iens);   real ru_R = state_limits_x(idU,1,k,j,i,iens);
          real rt_L = state_limits_x(idT,0,k,j,i,iens);   real rt_R = state_limits_x(idT,1,k,j,i,iens);
          real p_L  = C0*std::pow(rt_L,gamma)         ;   real p_R  = C0*std::pow(rt_R,gamma)         ;
          real constexpr cs = 350;
          real w1 = 0.5_fp * (p_R-cs*ru_R);
          real w2 = 0.5_fp * (p_L+cs*ru_L);
          real p_upw  = w1 + w2;
          real ru_upw = (w2-w1)/cs;
          // Advectively upwind everything else
          int ind = ru_L+ru_R > 0 ? 0 : 1;
          real r_upw = state_limits_x(idR,ind,k,j,i,iens);
          state_flux_x(idR,k,j,i,iens) = ru_upw;
          state_flux_x(idU,k,j,i,iens) = ru_upw*state_limits_x(idU,ind,k,j,i,iens)/r_upw + p_upw;
          state_flux_x(idV,k,j,i,iens) = ru_upw*state_limits_x(idV,ind,k,j,i,iens)/r_upw;
          state_flux_x(idW,k,j,i,iens) = ru_upw*state_limits_x(idW,ind,k,j,i,iens)/r_upw;
          state_flux_x(idT,k,j,i,iens) = ru_upw*state_limits_x(idT,ind,k,j,i,iens)/r_upw;
          for (int tr=0; tr < num_tracers; tr++) {
            tracers_flux_x(tr,k,j,i,iens) = ru_upw*tracers_limits_x(tr,ind,k,j,i,iens)/r_upw;
          }
        }

        // Y-direction
        // If we are simulating in 2-D, then do not do Riemann in the y-direction
        if ( (! sim2d) && i < nx && k < nz) {
          // Acoustically upwind mass flux and pressure
          real rv_L = state_limits_y(idV,0,k,j,i,iens);   real rv_R = state_limits_y(idV,1,k,j,i,iens);
          real rt_L = state_limits_y(idT,0,k,j,i,iens);   real rt_R = state_limits_y(idT,1,k,j,i,iens);
          real p_L  = C0*std::pow(rt_L,gamma)         ;   real p_R  = C0*std::pow(rt_R,gamma)         ;
          real constexpr cs = 350;
          real w1 = 0.5_fp * (p_R-cs*rv_R);
          real w2 = 0.5_fp * (p_L+cs*rv_L);
          real p_upw  = w1 + w2;
          real rv_upw = (w2-w1)/cs;
          // Advectively upwind everything else
          int ind = rv_L+rv_R > 0 ? 0 : 1;
          real r_upw = state_limits_y(idR,ind,k,j,i,iens);
          state_flux_y(idR,k,j,i,iens) = rv_upw;
          state_flux_y(idU,k,j,i,iens) = rv_upw*state_limits_y(idU,ind,k,j,i,iens)/r_upw;
          state_flux_y(idV,k,j,i,iens) = rv_upw*state_limits_y(idV,ind,k,j,i,iens)/r_upw + p_upw;
          state_flux_y(idW,k,j,i,iens) = rv_upw*state_limits_y(idW,ind,k,j,i,iens)/r_upw;
          state_flux_y(idT,k,j,i,iens) = rv_upw*state_limits_y(idT,ind,k,j,i,iens)/r_upw;
          for (int tr=0; tr < num_tracers; tr++) {
            tracers_flux_y(tr,k,j,i,iens) = rv_upw*tracers_limits_y(tr,ind,k,j,i,iens)/r_upw;
          }
        } else if (i < nx && k < nz) {
          state_flux_y(idR,k,j,i,iens) = 0;
          state_flux_y(idU,k,j,i,iens) = 0;
          state_flux_y(idV,k,j,i,iens) = 0;
          state_flux_y(idW,k,j,i,iens) = 0;
          state_flux_y(idT,k,j,i,iens) = 0;
          for (int tr=0; tr < num_tracers; tr++) { tracers_flux_y(tr,k,j,i,iens) = 0; }
        }

        // Z-direction
        if (i < nx && j < ny) {
          // Acoustically upwind mass flux and pressure
          real rw_L = state_limits_z(idW,0,k,j,i,iens);   real rw_R = state_limits_z(idW,1,k,j,i,iens);
          real rt_L = state_limits_z(idT,0,k,j,i,iens);   real rt_R = state_limits_z(idT,1,k,j,i,iens);
          real p_L  = C0*std::pow(rt_L,gamma)         ;   real p_R  = C0*std::pow(rt_R,gamma)         ;
          real constexpr cs = 350;
          real w1 = 0.5_fp * (p_R-cs*rw_R);
          real w2 = 0.5_fp * (p_L+cs*rw_L);
          real p_upw  = w1 + w2;
          real rw_upw = (w2-w1)/cs;
          // Advectively upwind everything else
          int ind = rw_L+rw_R > 0 ? 0 : 1;
          real r_upw = state_limits_z(idR,ind,k,j,i,iens);
          state_flux_z(idR,k,j,i,iens) = rw_upw;
          state_flux_z(idU,k,j,i,iens) = rw_upw*state_limits_z(idU,ind,k,j,i,iens)/r_upw;
          state_flux_z(idV,k,j,i,iens) = rw_upw*state_limits_z(idV,ind,k,j,i,iens)/r_upw;
          state_flux_z(idW,k,j,i,iens) = rw_upw*state_limits_z(idW,ind,k,j,i,iens)/r_upw + p_upw;
          state_flux_z(idT,k,j,i,iens) = rw_upw*state_limits_z(idT,ind,k,j,i,iens)/r_upw;
          for (int tr=0; tr < num_tracers; tr++) {
            tracers_flux_z(tr,k,j,i,iens) = rw_upw*tracers_limits_z(tr,ind,k,j,i,iens)/r_upw;
          }
        }

        // Multiply density back to other variables
        if (i < nx && j < ny && k < nz) {
          state(idU,hs+k,hs+j,hs+i,iens) *= ( state(idR,hs+k,hs+j,hs+i,iens) + hy_dens_cells(k,iens) );
          state(idV,hs+k,hs+j,hs+i,iens) *= ( state(idR,hs+k,hs+j,hs+i,iens) + hy_dens_cells(k,iens) );
          state(idW,hs+k,hs+j,hs+i,iens) *= ( state(idR,hs+k,hs+j,hs+i,iens) + hy_dens_cells(k,iens) );
          for (int tr=0; tr < num_tracers; tr++) {
            tracers(tr,hs+k,hs+j,hs+i,iens) *= ( state(idR,hs+k,hs+j,hs+i,iens) + hy_dens_cells(k,iens) );
          }
        }
      });

      // Deallocate state and tracer limits because they are no longer needed
      state_limits_x   = real6d();
      state_limits_y   = real6d();
      state_limits_z   = real6d();
      tracers_limits_x = real6d();
      tracers_limits_y = real6d();
      tracers_limits_z = real6d();

      // Flux Corrected Transport to enforce positivity for tracer species that must remain non-negative
      // This looks like it has a race condition, but it does not. Only one of the adjacent cells can ever change
      // a given edge flux because it's only changed if its sign oriented outward from a cell.
      parallel_for( YAKL_AUTO_LABEL() , Bounds<5>(num_tracers,nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int tr, int k, int j, int i, int iens) {
        if (tracer_positive(tr)) {
          real mass_available = max(tracers(tr,hs+k,hs+j,hs+i,iens),0._fp) * dx * dy * dz;
          real flux_out_x = ( max(tracers_flux_x(tr,k,j,i+1,iens),0._fp) - min(tracers_flux_x(tr,k,j,i,iens),0._fp) ) / dx;
          real flux_out_y = ( max(tracers_flux_y(tr,k,j+1,i,iens),0._fp) - min(tracers_flux_y(tr,k,j,i,iens),0._fp) ) / dy;
          real flux_out_z = ( max(tracers_flux_z(tr,k+1,j,i,iens),0._fp) - min(tracers_flux_z(tr,k,j,i,iens),0._fp) ) / dz;
          real mass_out = (flux_out_x + flux_out_y + flux_out_z) * dt * dx * dy * dz;
          if (mass_out > mass_available) {
            real mult = mass_available / mass_out;
            if (tracers_flux_x(tr,k,j,i+1,iens) > 0) tracers_flux_x(tr,k,j,i+1,iens) *= mult;
            if (tracers_flux_x(tr,k,j,i  ,iens) < 0) tracers_flux_x(tr,k,j,i  ,iens) *= mult;
            if (tracers_flux_y(tr,k,j+1,i,iens) > 0) tracers_flux_y(tr,k,j+1,i,iens) *= mult;
            if (tracers_flux_y(tr,k,j  ,i,iens) < 0) tracers_flux_y(tr,k,j  ,i,iens) *= mult;
            if (tracers_flux_z(tr,k+1,j,i,iens) > 0) tracers_flux_z(tr,k+1,j,i,iens) *= mult;
            if (tracers_flux_z(tr,k  ,j,i,iens) < 0) tracers_flux_z(tr,k  ,j,i,iens) *= mult;
          }
        }
      });

      // Compute tendencies as the flux divergence + gravity source term
      parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        for (int l = 0; l < num_state; l++) {
          state_tend  (l,k,j,i,iens) = -( state_flux_x  (l,k  ,j  ,i+1,iens) - state_flux_x  (l,k,j,i,iens) ) / dx
                                       -( state_flux_y  (l,k  ,j+1,i  ,iens) - state_flux_y  (l,k,j,i,iens) ) / dy
                                       -( state_flux_z  (l,k+1,j  ,i  ,iens) - state_flux_z  (l,k,j,i,iens) ) / dz;
          if (l == idW && enable_gravity) state_tend(l,k,j,i,iens) += -grav * ( state(idR,hs+k,hs+j,hs+i,iens) + hy_dens_cells(k,iens) );
          if (l == idU) state_tend(l,k,j,i,iens) += fcor*state(idV,hs+k,hs+j,hs+i,iens);
          if (l == idV) state_tend(l,k,j,i,iens) -= fcor*state(idU,hs+k,hs+j,hs+i,iens);
          if (l == idV && sim2d) state_tend(l,k,j,i,iens) = 0;
        }
        for (int l = 0; l < num_tracers; l++) {
          tracers_tend(l,k,j,i,iens) = -( tracers_flux_x(l,k  ,j  ,i+1,iens) - tracers_flux_x(l,k,j,i,iens) ) / dx
                                       -( tracers_flux_y(l,k  ,j+1,i  ,iens) - tracers_flux_y(l,k,j,i,iens) ) / dy
                                       -( tracers_flux_z(l,k+1,j  ,i  ,iens) - tracers_flux_z(l,k,j,i,iens) ) / dz;
        }
        if (use_immersed_boundaries) {
          // Determine the time scale of damping
          real tau = 1.e3*dt;
          // Compute immersed material tendencies (zero velocity, reference density & temperature)
          real imm_tend_idR = -std::min(1._fp,dt/tau)*state(idR,hs+k,hs+j,hs+i,iens)/dt;
          real imm_tend_idU = -std::min(1._fp,dt/tau)*state(idU,hs+k,hs+j,hs+i,iens)/dt;
          real imm_tend_idV = -std::min(1._fp,dt/tau)*state(idV,hs+k,hs+j,hs+i,iens)/dt;
          real imm_tend_idW = -std::min(1._fp,dt/tau)*state(idW,hs+k,hs+j,hs+i,iens)/dt;
          real imm_tend_idT = -std::min(1._fp,dt/tau)*state(idT,hs+k,hs+j,hs+i,iens)/dt;
          // immersed proportion has immersed tendnecies. Other proportion has free tendencies
          real prop = immersed_proportion(k,j,i,iens);
          state_tend(idR,k,j,i,iens) = prop*imm_tend_idR + (1-prop)*state_tend(idR,k,j,i,iens);
          state_tend(idU,k,j,i,iens) = prop*imm_tend_idU + (1-prop)*state_tend(idU,k,j,i,iens);
          state_tend(idV,k,j,i,iens) = prop*imm_tend_idV + (1-prop)*state_tend(idV,k,j,i,iens);
          state_tend(idW,k,j,i,iens) = prop*imm_tend_idW + (1-prop)*state_tend(idW,k,j,i,iens);
          state_tend(idT,k,j,i,iens) = prop*imm_tend_idT + (1-prop)*state_tend(idT,k,j,i,iens);
        }
      });
    }


    // ord stencil cell averages to two GLL point values via high-order reconstruction and WENO limiting
    YAKL_INLINE static void reconstruct_gll_values( SArray<real,1,ord>     const &stencil      ,
                                                    SArray<real,1,2>             &gll          ,
                                                    SArray<real,2,ord,2>   const &coefs_to_gll ,
                                                    weno::WenoLimiter<ord> const &limiter    ) {
      // Reconstruct values
      SArray<real,1,ord> wenoCoefs;
      limiter.compute_limited_coefs( stencil , wenoCoefs );
      // Transform ord weno coefficients into 2 GLL points
      for (int ii=0; ii<2; ii++) {
        real tmp = 0;
        for (int s=0; s < ord; s++) {
          tmp += coefs_to_gll(s,ii) * wenoCoefs(s);
        }
        gll(ii) = tmp;
      }
    }


    void halo_exchange(core::Coupler const &coupler , real5d const &state , real5d const &tracers) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto num_tracers = coupler.get_num_tracers();
      auto sim2d       = coupler.is_sim2d();
      auto px          = coupler.get_px();
      auto py          = coupler.get_py();
      auto nproc_x     = coupler.get_nproc_x();
      auto nproc_y     = coupler.get_nproc_y();
      auto bc_x        = coupler.get_option<int>("bc_x");
      auto bc_y        = coupler.get_option<int>("bc_y");
      auto bc_z        = coupler.get_option<int>("bc_z");

      int npack = num_state + num_tracers;

      realHost5d halo_send_buf_W_host("halo_send_buf_W_host",npack,nz,ny,hs,nens);
      realHost5d halo_send_buf_E_host("halo_send_buf_E_host",npack,nz,ny,hs,nens);
      realHost5d halo_send_buf_S_host("halo_send_buf_S_host",npack,nz,hs,nx,nens);
      realHost5d halo_send_buf_N_host("halo_send_buf_N_host",npack,nz,hs,nx,nens);
      realHost5d halo_recv_buf_S_host("halo_recv_buf_S_host",npack,nz,hs,nx,nens);
      realHost5d halo_recv_buf_N_host("halo_recv_buf_N_host",npack,nz,hs,nx,nens);
      realHost5d halo_recv_buf_W_host("halo_recv_buf_W_host",npack,nz,ny,hs,nens);
      realHost5d halo_recv_buf_E_host("halo_recv_buf_E_host",npack,nz,ny,hs,nens);

      real5d halo_send_buf_W("halo_send_buf_W",npack,nz,ny,hs,nens);
      real5d halo_send_buf_E("halo_send_buf_E",npack,nz,ny,hs,nens);

      parallel_for( YAKL_AUTO_LABEL() , Bounds<5>(npack,nz,ny,hs,nens) ,
                                        YAKL_LAMBDA (int v, int k, int j, int ii, int iens) {
        if (v < num_state) {
          halo_send_buf_W(v,k,j,ii,iens) = state  (v          ,hs+k,hs+j,hs+ii,iens);
          halo_send_buf_E(v,k,j,ii,iens) = state  (v          ,hs+k,hs+j,nx+ii,iens);
        } else {
          halo_send_buf_W(v,k,j,ii,iens) = tracers(v-num_state,hs+k,hs+j,hs+ii,iens);
          halo_send_buf_E(v,k,j,ii,iens) = tracers(v-num_state,hs+k,hs+j,nx+ii,iens);
        }
      });

      real5d halo_send_buf_S("halo_send_buf_S",npack,nz,hs,nx,nens);
      real5d halo_send_buf_N("halo_send_buf_N",npack,nz,hs,nx,nens);

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<5>(npack,nz,hs,nx,nens) ,
                                          YAKL_LAMBDA (int v, int k, int jj, int i, int iens) {
          if (v < num_state) {
            halo_send_buf_S(v,k,jj,i,iens) = state  (v          ,hs+k,hs+jj,hs+i,iens);
            halo_send_buf_N(v,k,jj,i,iens) = state  (v          ,hs+k,ny+jj,hs+i,iens);
          } else {
            halo_send_buf_S(v,k,jj,i,iens) = tracers(v-num_state,hs+k,hs+jj,hs+i,iens);
            halo_send_buf_N(v,k,jj,i,iens) = tracers(v-num_state,hs+k,ny+jj,hs+i,iens);
          }
        });
      }

      yakl::fence();
      yakl::timer_start("halo_exchange_mpi");

      real5d halo_recv_buf_W("halo_recv_buf_W",npack,nz,ny,hs,nens);
      real5d halo_recv_buf_E("halo_recv_buf_E",npack,nz,ny,hs,nens);
      real5d halo_recv_buf_S("halo_recv_buf_S",npack,nz,hs,nx,nens);
      real5d halo_recv_buf_N("halo_recv_buf_N",npack,nz,hs,nx,nens);

      MPI_Request sReq[4];
      MPI_Request rReq[4];

      auto &neigh = coupler.get_neighbor_rankid_matrix();
      auto mpi_data_type = coupler.get_mpi_data_type();

      #ifdef MW_GPU_AWARE_MPI
        yakl::fence();

        //Pre-post the receives
        MPI_Irecv( halo_recv_buf_W.data() , halo_recv_buf_W.size() , mpi_data_type , neigh(1,0) , 0 , MPI_COMM_WORLD , &rReq[0] );
        MPI_Irecv( halo_recv_buf_E.data() , halo_recv_buf_E.size() , mpi_data_type , neigh(1,2) , 1 , MPI_COMM_WORLD , &rReq[1] );
        if (!sim2d) {
          MPI_Irecv( halo_recv_buf_S.data() , halo_recv_buf_S.size() , mpi_data_type , neigh(0,1) , 2 , MPI_COMM_WORLD , &rReq[2] );
          MPI_Irecv( halo_recv_buf_N.data() , halo_recv_buf_N.size() , mpi_data_type , neigh(2,1) , 3 , MPI_COMM_WORLD , &rReq[3] );
        }

        //Send the data
        MPI_Isend( halo_send_buf_W.data() , halo_send_buf_W.size() , mpi_data_type , neigh(1,0) , 1 , MPI_COMM_WORLD , &sReq[0] );
        MPI_Isend( halo_send_buf_E.data() , halo_send_buf_E.size() , mpi_data_type , neigh(1,2) , 0 , MPI_COMM_WORLD , &sReq[1] );
        if (!sim2d) {
          MPI_Isend( halo_send_buf_S.data() , halo_send_buf_S.size() , mpi_data_type , neigh(0,1) , 3 , MPI_COMM_WORLD , &sReq[2] );
          MPI_Isend( halo_send_buf_N.data() , halo_send_buf_N.size() , mpi_data_type , neigh(2,1) , 2 , MPI_COMM_WORLD , &sReq[3] );
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
      #else
        //Pre-post the receives
        MPI_Irecv( halo_recv_buf_W_host.data() , halo_recv_buf_W_host.size() , mpi_data_type , neigh(1,0) , 0 , MPI_COMM_WORLD , &rReq[0] );
        MPI_Irecv( halo_recv_buf_E_host.data() , halo_recv_buf_E_host.size() , mpi_data_type , neigh(1,2) , 1 , MPI_COMM_WORLD , &rReq[1] );
        if (!sim2d) {
          MPI_Irecv( halo_recv_buf_S_host.data() , halo_recv_buf_S_host.size() , mpi_data_type , neigh(0,1) , 2 , MPI_COMM_WORLD , &rReq[2] );
          MPI_Irecv( halo_recv_buf_N_host.data() , halo_recv_buf_N_host.size() , mpi_data_type , neigh(2,1) , 3 , MPI_COMM_WORLD , &rReq[3] );
        }

        halo_send_buf_W.deep_copy_to(halo_send_buf_W_host);
        halo_send_buf_E.deep_copy_to(halo_send_buf_E_host);
        if (!sim2d) {
          halo_send_buf_S.deep_copy_to(halo_send_buf_S_host);
          halo_send_buf_N.deep_copy_to(halo_send_buf_N_host);
        }

        yakl::fence();

        //Send the data
        MPI_Isend( halo_send_buf_W_host.data() , halo_send_buf_W_host.size() , mpi_data_type , neigh(1,0) , 1 , MPI_COMM_WORLD , &sReq[0] );
        MPI_Isend( halo_send_buf_E_host.data() , halo_send_buf_E_host.size() , mpi_data_type , neigh(1,2) , 0 , MPI_COMM_WORLD , &sReq[1] );
        if (!sim2d) {
          MPI_Isend( halo_send_buf_S_host.data() , halo_send_buf_S_host.size() , mpi_data_type , neigh(0,1) , 3 , MPI_COMM_WORLD , &sReq[2] );
          MPI_Isend( halo_send_buf_N_host.data() , halo_send_buf_N_host.size() , mpi_data_type , neigh(2,1) , 2 , MPI_COMM_WORLD , &sReq[3] );
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
      #endif

      parallel_for( YAKL_AUTO_LABEL() , Bounds<5>(npack,nz,ny,hs,nens) ,
                                        YAKL_LAMBDA (int v, int k, int j, int ii, int iens) {
        if (v < num_state) {
          state  (v          ,hs+k,hs+j,      ii,iens) = halo_recv_buf_W(v,k,j,ii,iens);
          state  (v          ,hs+k,hs+j,nx+hs+ii,iens) = halo_recv_buf_E(v,k,j,ii,iens);
        } else {
          tracers(v-num_state,hs+k,hs+j,      ii,iens) = halo_recv_buf_W(v,k,j,ii,iens);
          tracers(v-num_state,hs+k,hs+j,nx+hs+ii,iens) = halo_recv_buf_E(v,k,j,ii,iens);
        }
      });

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<5>(npack,nz,hs,nx,nens) ,
                                          YAKL_LAMBDA (int v, int k, int jj, int i, int iens) {
          if (v < num_state) {
            state  (v          ,hs+k,      jj,hs+i,iens) = halo_recv_buf_S(v,k,jj,i,iens);
            state  (v          ,hs+k,ny+hs+jj,hs+i,iens) = halo_recv_buf_N(v,k,jj,i,iens);
          } else {
            tracers(v-num_state,hs+k,      jj,hs+i,iens) = halo_recv_buf_S(v,k,jj,i,iens);
            tracers(v-num_state,hs+k,ny+hs+jj,hs+i,iens) = halo_recv_buf_N(v,k,jj,i,iens);
          }
        });
      }

      ////////////////////////////////////
      // Begin boundary conditions
      ////////////////////////////////////
      if (bc_z == BC_PERIODIC) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(hs,ny,nx,nens) ,
                                          YAKL_LAMBDA (int kk, int j, int i, int iens) {
          for (int l=0; l < num_state; l++) {
            state(l,      kk,hs+j,hs+i,iens) = state(l,      kk+nz,hs+j,hs+i,iens);
            state(l,hs+nz+kk,hs+j,hs+i,iens) = state(l,hs+nz+kk-nz,hs+j,hs+i,iens);
          }
          for (int l=0; l < num_tracers; l++) {
            tracers(l,      kk,hs+j,hs+i,iens) = tracers(l,      kk+nz,hs+j,hs+i,iens);
            tracers(l,hs+nz+kk,hs+j,hs+i,iens) = tracers(l,hs+nz+kk-nz,hs+j,hs+i,iens);
          }
        });
      } else if (bc_z == BC_WALL || bc_z == BC_OPEN) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(hs,ny,nx,nens) ,
                                          YAKL_LAMBDA (int kk, int j, int i, int iens) {
          for (int l=0; l < num_state; l++) {
            if (l == idW && bc_z == BC_WALL) {
              state(l,      kk,hs+j,hs+i,iens) = 0;
              state(l,hs+nz+kk,hs+j,hs+i,iens) = 0;
            } else {
              state(l,      kk,hs+j,hs+i,iens) = state(l,hs+0   ,hs+j,hs+i,iens);
              state(l,hs+nz+kk,hs+j,hs+i,iens) = state(l,hs+nz-1,hs+j,hs+i,iens);
            }
          }
          for (int l=0; l < num_tracers; l++) {
            tracers(l,      kk,hs+j,hs+i,iens) = tracers(l,hs+0   ,hs+j,hs+i,iens);
            tracers(l,hs+nz+kk,hs+j,hs+i,iens) = tracers(l,hs+nz-1,hs+j,hs+i,iens);
          }
        });
      }
      if (bc_x == BC_WALL || bc_x == BC_OPEN) {
        if (px == 0) {
          parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,hs,nens) ,
                                            YAKL_LAMBDA (int k, int j, int ii, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idU && bc_x == BC_WALL) { state(l,hs+k,hs+j,ii,iens) = 0; }
              else                             { state(l,hs+k,hs+j,ii,iens) = state(l,hs+k,hs+j,hs+0,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,ii,iens) = tracers(l,hs+k,hs+j,hs+0,iens); }
          });
        }
        if (px == nproc_x-1) {
          parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,hs,nens) ,
                                            YAKL_LAMBDA (int k, int j, int ii, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idU && bc_x == BC_WALL) { state(l,hs+k,hs+j,hs+nx+ii,iens) = 0; }
              else                             { state(l,hs+k,hs+j,hs+nx+ii,iens) = state(l,hs+k,hs+j,hs+nx-1,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+nx+ii,iens) = tracers(l,hs+k,hs+j,hs+nx-1,iens); }
          });
        }
      }
      if (bc_y == BC_WALL || bc_y == BC_OPEN) {
        if (py == 0) {
          parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,hs,nx,nens) ,
                                            YAKL_LAMBDA (int k, int jj, int i, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idV && bc_y == BC_WALL) { state(l,hs+k,jj,hs+i,iens) = 0; }
              else                             { state(l,hs+k,jj,hs+i,iens) = state(l,hs+k,hs+0,hs+i,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,jj,hs+i,iens) = tracers(l,hs+k,hs+0,hs+i,iens); }
          });
        }
        if (py == nproc_y-1) {
          parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,hs,nx,nens) ,
                                            YAKL_LAMBDA (int k, int jj, int i, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idV && bc_y == BC_WALL) { state(l,hs+k,hs+ny+jj,hs+i,iens) = 0; }
              else                             { state(l,hs+k,hs+ny+jj,hs+i,iens) = state(l,hs+k,hs+ny-1,hs+i,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+ny+jj,hs+i,iens) = tracers(l,hs+k,hs+ny-1,hs+i,iens); }
          });
        }
      }

    }


    void edge_exchange(core::Coupler const &coupler , real6d const &state_limits_x , real6d const &tracers_limits_x ,
                                                      real6d const &state_limits_y , real6d const &tracers_limits_y ,
                                                      real6d const &state_limits_z , real6d const &tracers_limits_z ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto num_tracers = coupler.get_num_tracers();
      auto sim2d       = coupler.is_sim2d();
      auto px          = coupler.get_px();
      auto py          = coupler.get_py();
      auto nproc_x     = coupler.get_nproc_x();
      auto nproc_y     = coupler.get_nproc_y();
      auto bc_x        = coupler.get_option<int>("bc_x");
      auto bc_y        = coupler.get_option<int>("bc_y");
      auto bc_z        = coupler.get_option<int>("bc_z");

      int npack = num_state + num_tracers;

      realHost4d edge_send_buf_S_host("edge_send_buf_S_host",num_state+num_tracers,nz,nx,nens);
      realHost4d edge_send_buf_N_host("edge_send_buf_N_host",num_state+num_tracers,nz,nx,nens);
      realHost4d edge_send_buf_W_host("edge_send_buf_W_host",num_state+num_tracers,nz,ny,nens);
      realHost4d edge_send_buf_E_host("edge_send_buf_E_host",num_state+num_tracers,nz,ny,nens);
      realHost4d edge_recv_buf_S_host("edge_recv_buf_S_host",num_state+num_tracers,nz,nx,nens);
      realHost4d edge_recv_buf_N_host("edge_recv_buf_N_host",num_state+num_tracers,nz,nx,nens);
      realHost4d edge_recv_buf_W_host("edge_recv_buf_W_host",num_state+num_tracers,nz,ny,nens);
      realHost4d edge_recv_buf_E_host("edge_recv_buf_E_host",num_state+num_tracers,nz,ny,nens);

      real4d edge_send_buf_W("edge_send_buf_W",npack,nz,ny,nens);
      real4d edge_send_buf_E("edge_send_buf_E",npack,nz,ny,nens);

      parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(npack,nz,ny,nens) , YAKL_LAMBDA (int v, int k, int j, int iens) {
        if (v < num_state) {
          edge_send_buf_W(v,k,j,iens) = state_limits_x  (v          ,1,k,j,0 ,iens);
          edge_send_buf_E(v,k,j,iens) = state_limits_x  (v          ,0,k,j,nx,iens);
        } else {
          edge_send_buf_W(v,k,j,iens) = tracers_limits_x(v-num_state,1,k,j,0 ,iens);
          edge_send_buf_E(v,k,j,iens) = tracers_limits_x(v-num_state,0,k,j,nx,iens);
        }
      });

      real4d edge_send_buf_S("edge_send_buf_S",npack,nz,nx,nens);
      real4d edge_send_buf_N("edge_send_buf_N",npack,nz,nx,nens);

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(npack,nz,nx,nens) , YAKL_LAMBDA (int v, int k, int i, int iens) {
          if (v < num_state) {
            edge_send_buf_S(v,k,i,iens) = state_limits_y  (v          ,1,k,0 ,i,iens);
            edge_send_buf_N(v,k,i,iens) = state_limits_y  (v          ,0,k,ny,i,iens);
          } else {
            edge_send_buf_S(v,k,i,iens) = tracers_limits_y(v-num_state,1,k,0 ,i,iens);
            edge_send_buf_N(v,k,i,iens) = tracers_limits_y(v-num_state,0,k,ny,i,iens);
          }
        });
      }

      yakl::fence();
      yakl::timer_start("edge_exchange_mpi");

      real4d edge_recv_buf_W("edge_recv_buf_W",npack,nz,ny,nens);
      real4d edge_recv_buf_E("edge_recv_buf_E",npack,nz,ny,nens);
      real4d edge_recv_buf_S("edge_recv_buf_S",npack,nz,nx,nens);
      real4d edge_recv_buf_N("edge_recv_buf_N",npack,nz,nx,nens);

      MPI_Request sReq[4];
      MPI_Request rReq[4];

      auto &neigh = coupler.get_neighbor_rankid_matrix();
      auto mpi_data_type = coupler.get_mpi_data_type();

      #ifdef MW_GPU_AWARE_MPI
        yakl::fence();

        //Pre-post the receives
        MPI_Irecv( edge_recv_buf_W.data() , edge_recv_buf_W.size() , mpi_data_type , neigh(1,0) , 4 , MPI_COMM_WORLD , &rReq[0] );
        MPI_Irecv( edge_recv_buf_E.data() , edge_recv_buf_E.size() , mpi_data_type , neigh(1,2) , 5 , MPI_COMM_WORLD , &rReq[1] );
        if (!sim2d) {
          MPI_Irecv( edge_recv_buf_S.data() , edge_recv_buf_S.size() , mpi_data_type , neigh(0,1) , 6 , MPI_COMM_WORLD , &rReq[2] );
          MPI_Irecv( edge_recv_buf_N.data() , edge_recv_buf_N.size() , mpi_data_type , neigh(2,1) , 7 , MPI_COMM_WORLD , &rReq[3] );
        }

        //Send the data
        MPI_Isend( edge_send_buf_W.data() , edge_send_buf_W.size() , mpi_data_type , neigh(1,0) , 5 , MPI_COMM_WORLD , &sReq[0] );
        MPI_Isend( edge_send_buf_E.data() , edge_send_buf_E.size() , mpi_data_type , neigh(1,2) , 4 , MPI_COMM_WORLD , &sReq[1] );
        if (!sim2d) {
          MPI_Isend( edge_send_buf_S.data() , edge_send_buf_S.size() , mpi_data_type , neigh(0,1) , 7 , MPI_COMM_WORLD , &sReq[2] );
          MPI_Isend( edge_send_buf_N.data() , edge_send_buf_N.size() , mpi_data_type , neigh(2,1) , 6 , MPI_COMM_WORLD , &sReq[3] );
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
      #else
        //Pre-post the receives
        MPI_Irecv( edge_recv_buf_W_host.data() , edge_recv_buf_W_host.size() , mpi_data_type , neigh(1,0) , 4 , MPI_COMM_WORLD , &rReq[0] );
        MPI_Irecv( edge_recv_buf_E_host.data() , edge_recv_buf_E_host.size() , mpi_data_type , neigh(1,2) , 5 , MPI_COMM_WORLD , &rReq[1] );
        if (!sim2d) {
          MPI_Irecv( edge_recv_buf_S_host.data() , edge_recv_buf_S_host.size() , mpi_data_type , neigh(0,1) , 6 , MPI_COMM_WORLD , &rReq[2] );
          MPI_Irecv( edge_recv_buf_N_host.data() , edge_recv_buf_N_host.size() , mpi_data_type , neigh(2,1) , 7 , MPI_COMM_WORLD , &rReq[3] );
        }

        edge_send_buf_W.deep_copy_to(edge_send_buf_W_host);
        edge_send_buf_E.deep_copy_to(edge_send_buf_E_host);
        if (!sim2d) {
          edge_send_buf_S.deep_copy_to(edge_send_buf_S_host);
          edge_send_buf_N.deep_copy_to(edge_send_buf_N_host);
        }

        yakl::fence();

        //Send the data
        MPI_Isend( edge_send_buf_W_host.data() , edge_send_buf_W_host.size() , mpi_data_type , neigh(1,0) , 5 , MPI_COMM_WORLD , &sReq[0] );
        MPI_Isend( edge_send_buf_E_host.data() , edge_send_buf_E_host.size() , mpi_data_type , neigh(1,2) , 4 , MPI_COMM_WORLD , &sReq[1] );
        if (!sim2d) {
          MPI_Isend( edge_send_buf_S_host.data() , edge_send_buf_S_host.size() , mpi_data_type , neigh(0,1) , 7 , MPI_COMM_WORLD , &sReq[2] );
          MPI_Isend( edge_send_buf_N_host.data() , edge_send_buf_N_host.size() , mpi_data_type , neigh(2,1) , 6 , MPI_COMM_WORLD , &sReq[3] );
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
      #endif

      parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(npack,nz,ny,nens) ,
                                        YAKL_LAMBDA (int v, int k, int j, int iens) {
        if (v < num_state) {
          state_limits_x  (v          ,0,k,j,0 ,iens) = edge_recv_buf_W(v,k,j,iens);
          state_limits_x  (v          ,1,k,j,nx,iens) = edge_recv_buf_E(v,k,j,iens);
        } else {
          tracers_limits_x(v-num_state,0,k,j,0 ,iens) = edge_recv_buf_W(v,k,j,iens);
          tracers_limits_x(v-num_state,1,k,j,nx,iens) = edge_recv_buf_E(v,k,j,iens);
        }
      });

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(npack,nz,nx,nens) ,
                                          YAKL_LAMBDA (int v, int k, int i, int iens) {
          if (v < num_state) {
            state_limits_y  (v          ,0,k,0 ,i,iens) = edge_recv_buf_S(v,k,i,iens);
            state_limits_y  (v          ,1,k,ny,i,iens) = edge_recv_buf_N(v,k,i,iens);
          } else {
            tracers_limits_y(v-num_state,0,k,0 ,i,iens) = edge_recv_buf_S(v,k,i,iens);
            tracers_limits_y(v-num_state,1,k,ny,i,iens) = edge_recv_buf_N(v,k,i,iens);
          }
        });
      }

      /////////////////////////////////
      // Begin boundary conditions
      /////////////////////////////////
      if (bc_z == BC_PERIODIC) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(ny,nx,nens) ,
                                          YAKL_LAMBDA (int j, int i, int iens) {
          for (int l=0; l < num_state; l++) {
            state_limits_z(l,0,0 ,j,i,iens) = state_limits_z(l,0,nz,j,i,iens);
            state_limits_z(l,1,nz,j,i,iens) = state_limits_z(l,1,0 ,j,i,iens);
          }
          for (int l=0; l < num_tracers; l++) {
            tracers_limits_z(l,0,0 ,j,i,iens) = tracers_limits_z(l,0,nz,j,i,iens);
            tracers_limits_z(l,1,nz,j,i,iens) = tracers_limits_z(l,1,0 ,j,i,iens);
          }
        });
      } else if (bc_z == BC_WALL || bc_z == BC_OPEN) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(ny,nx,nens) ,
                                          YAKL_LAMBDA (int j, int i, int iens) {
          for (int l=0; l < num_state; l++) {
            if (l == idW && bc_z == BC_WALL) {
              state_limits_z(l,0,0 ,j,i,iens) = 0;
              state_limits_z(l,1,0 ,j,i,iens) = 0;
              state_limits_z(l,0,nz,j,i,iens) = 0;
              state_limits_z(l,1,nz,j,i,iens) = 0;
            } else {
              state_limits_z(l,0,0 ,j,i,iens) = state_limits_z(l,1,0 ,j,i,iens);
              state_limits_z(l,1,nz,j,i,iens) = state_limits_z(l,0,nz,j,i,iens);
            }
          }
          for (int l=0; l < num_tracers; l++) {
            tracers_limits_z(l,0,0 ,j,i,iens) = tracers_limits_z(l,1,0 ,j,i,iens);
            tracers_limits_z(l,1,nz,j,i,iens) = tracers_limits_z(l,0,nz,j,i,iens);
          }
        });
      }
      if (bc_x == BC_WALL || bc_x == BC_OPEN) {
        if (px == 0) {
          parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nens) ,
                                            YAKL_LAMBDA (int k, int j, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idU && bc_x == BC_WALL) { state_limits_x(l,0,k,j,0,iens) = 0; state_limits_x(l,1,k,j,0,iens) = 0; }
              else                             { state_limits_x(l,0,k,j,0,iens) = state_limits_x(l,1,k,j,0,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers_limits_x(l,0,k,j,0,iens) = tracers_limits_x(l,1,k,j,0,iens); }
          });
        } else if (px == nproc_x-1) {
          parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nens) ,
                                            YAKL_LAMBDA (int k, int j, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idU && bc_x == BC_WALL) { state_limits_x(l,0,k,j,nx,iens) = 0; state_limits_x(l,1,k,j,nx,iens) = 0; }
              else                             { state_limits_x(l,1,k,j,nx,iens) = state_limits_x(l,0,k,j,nx,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers_limits_x(l,1,k,j,nx,iens) = tracers_limits_x(l,0,k,j,nx,iens); }
          });
        }
      }
      if (bc_y == BC_WALL || bc_y == BC_OPEN) {
        if (py == 0) {
          parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,nx,nens) ,
                                            YAKL_LAMBDA (int k, int i, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idV && bc_y == BC_WALL) { state_limits_y(l,0,k,0,i,iens) = 0; state_limits_y(l,1,k,0,i,iens) = 0; }
              else                             { state_limits_y(l,0,k,0,i,iens) = state_limits_y(l,1,k,0,i,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers_limits_y(l,0,k,0,i,iens) = tracers_limits_y(l,1,k,0,i,iens); }
          });
        } else if (py == nproc_y-1) {
          parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,nx,nens) ,
                                            YAKL_LAMBDA (int k, int i, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idV && bc_y == BC_WALL) { state_limits_y(l,0,k,ny,i,iens) = 0; state_limits_y(l,1,k,ny,i,iens) = 0; }
              else                             { state_limits_y(l,1,k,ny,i,iens) = state_limits_y(l,0,k,ny,i,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers_limits_y(l,1,k,ny,i,iens) = tracers_limits_y(l,0,k,ny,i,iens); }
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


    // Initialize the class data as well as the state and tracers arrays and convert them back into the coupler state
    void init(core::Coupler &coupler) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      YAKL_SCOPE( init_data_int       , this->init_data_int       );
      YAKL_SCOPE( hy_dens_cells       , this->hy_dens_cells       );
      YAKL_SCOPE( hy_dens_theta_cells , this->hy_dens_theta_cells );
      YAKL_SCOPE( hy_dens_edges       , this->hy_dens_edges       );
      YAKL_SCOPE( hy_dens_theta_edges , this->hy_dens_theta_edges );
      YAKL_SCOPE( idWV                , this->idWV                );

      // Set class data from # grid points, grid spacing, domain sizes, whether it's 2-D, and physical constants
      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto dx          = coupler.get_dx();
      auto dy          = coupler.get_dy();
      auto dz          = coupler.get_dz();
      auto xlen        = coupler.get_xlen();
      auto ylen        = coupler.get_ylen();
      auto zlen        = coupler.get_zlen();
      auto i_beg       = coupler.get_i_beg();
      auto j_beg       = coupler.get_j_beg();
      auto nx_glob     = coupler.get_nx_glob();
      auto ny_glob     = coupler.get_ny_glob();
      auto sim2d       = coupler.is_sim2d();
      auto num_tracers = coupler.get_num_tracers();
      auto enable_gravity = coupler.get_option<bool>("enable_gravity",true);

      if (! coupler.option_exists("R_d"     )) coupler.set_option<real>("R_d"     ,287.       );
      if (! coupler.option_exists("cp_d"    )) coupler.set_option<real>("cp_d"    ,1003.      );
      if (! coupler.option_exists("R_v"     )) coupler.set_option<real>("R_v"     ,461.       );
      if (! coupler.option_exists("cp_v"    )) coupler.set_option<real>("cp_v"    ,1859       );
      if (! coupler.option_exists("p0"      )) coupler.set_option<real>("p0"      ,1.e5       );
      if (! coupler.option_exists("grav"    )) coupler.set_option<real>("grav"    ,9.81       );
      if (! coupler.option_exists("earthrot")) coupler.set_option<real>("earthrot",7.292115e-5);
      auto R_d  = coupler.get_option<real>("R_d" );
      auto cp_d = coupler.get_option<real>("cp_d");
      auto R_v  = coupler.get_option<real>("R_v" );
      auto cp_v = coupler.get_option<real>("cp_v");
      auto p0   = coupler.get_option<real>("p0"  );
      auto grav = coupler.get_option<real>("grav");
      if (! coupler.option_exists("cv_d"   )) coupler.set_option<real>("cv_d"   ,cp_d - R_d );
      auto cv_d = coupler.get_option<real>("cv_d");
      if (! coupler.option_exists("gamma_d")) coupler.set_option<real>("gamma_d",cp_d / cv_d);
      if (! coupler.option_exists("kappa_d")) coupler.set_option<real>("kappa_d",R_d  / cp_d);
      if (! coupler.option_exists("cv_v"   )) coupler.set_option<real>("cv_v"   ,R_v - cp_v );
      auto gamma = coupler.get_option<real>("gamma_d");
      auto kappa = coupler.get_option<real>("kappa_d");
      if (! coupler.option_exists("C0")) coupler.set_option<real>("C0" , pow( R_d * pow( p0 , -kappa ) , gamma ));
      auto C0    = coupler.get_option<real>("C0");
      coupler.set_option<real>("latitude",0);

      auto &dm = coupler.get_data_manager_readwrite();

      dm.register_and_allocate<real>("density_dry","",{nz,ny,nx,nens});
      dm.register_and_allocate<real>("uvel"       ,"",{nz,ny,nx,nens});
      dm.register_and_allocate<real>("vvel"       ,"",{nz,ny,nx,nens});
      dm.register_and_allocate<real>("wvel"       ,"",{nz,ny,nx,nens});
      dm.register_and_allocate<real>("temp"       ,"",{nz,ny,nx,nens});

      sim2d = (coupler.get_ny_glob() == 1);

      R_d   = coupler.get_option<real>("R_d"    );
      R_v   = coupler.get_option<real>("R_v"    );
      cp_d  = coupler.get_option<real>("cp_d"   );
      cp_v  = coupler.get_option<real>("cp_v"   );
      p0    = coupler.get_option<real>("p0"     );
      grav  = coupler.get_option<real>("grav"   );
      kappa = coupler.get_option<real>("kappa_d");
      gamma = coupler.get_option<real>("gamma_d");
      C0    = coupler.get_option<real>("C0"     );

      // Use TransformMatrices class to create matrices & GLL points to convert degrees of freedom as needed
      TransformMatrices::get_gll_points          (gll_pts         );
      TransformMatrices::get_gll_weights         (gll_wts         );
      TransformMatrices::coefs_to_gll_lower      (coefs_to_gll    );

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
      out_freq       = coupler.get_option<real       >("out_freq" );

      coupler.set_option<int>("idWV",idWV);
      dm.register_and_allocate<bool>("tracer_adds_mass","",{num_tracers});
      auto dm_tracer_adds_mass = dm.get<bool,1>("tracer_adds_mass");
      tracer_adds_mass.deep_copy_to(dm_tracer_adds_mass);

      // Set an integer version of the input_data so we can test it inside GPU kernels
      if      (init_data == "thermal"  ) { init_data_int = DATA_THERMAL;   }
      else if (init_data == "supercell") { init_data_int = DATA_SUPERCELL; }
      else if (init_data == "city"     ) { init_data_int = DATA_CITY;      }
      else if (init_data == "building" ) { init_data_int = DATA_BUILDING;  }
      else { endrun("ERROR: Invalid init_data in yaml input file"); }

      coupler.set_option<bool>("use_immersed_boundaries",false);
      dm.register_and_allocate<real>("immersed_proportion","",{nz,ny,nx,nens});
      auto immersed_proportion = dm.get<real,4>("immersed_proportion");
      immersed_proportion = 0;

      etime   = 0;
      num_out = 0;

      // Allocate temp arrays to hold state and tracers before we convert it back to the coupler state
      real5d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real5d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs,nens);

      // Allocate arrays for hydrostatic background states
      hy_dens_cells       = real2d("hy_dens_cells"      ,nz  ,nens);
      hy_dens_theta_cells = real2d("hy_dens_theta_cells",nz  ,nens);
      hy_dens_edges       = real2d("hy_dens_edges"      ,nz+1,nens);
      hy_dens_theta_edges = real2d("hy_dens_theta_edges",nz+1,nens);

      if (init_data_int == DATA_SUPERCELL) {

        coupler.add_option<int>("bc_x",BC_PERIODIC);
        coupler.add_option<int>("bc_y",BC_PERIODIC);
        coupler.add_option<int>("bc_z",BC_WALL);
        coupler.add_option<real>("latitude",0);
        init_supercell( coupler , state , tracers );

      } else if (init_data_int == DATA_THERMAL) {

        coupler.add_option<int>("bc_x",BC_PERIODIC);
        coupler.add_option<int>("bc_y",BC_PERIODIC);
        coupler.add_option<int>("bc_z",BC_WALL    );
        coupler.add_option<real>("latitude",0);
        // Define quadrature weights and points for 3-point rules
        const int nqpoints = 3;
        SArray<real,1,nqpoints> qpoints;
        SArray<real,1,nqpoints> qweights;

        qpoints(0) = 0.112701665379258311482073460022;
        qpoints(1) = 0.500000000000000000000000000000;
        qpoints(2) = 0.887298334620741688517926539980;

        qweights(0) = 0.277777777777777777777777777779;
        qweights(1) = 0.444444444444444444444444444444;
        qweights(2) = 0.277777777777777777777777777779;

        size_t i_beg = coupler.get_i_beg();
        size_t j_beg = coupler.get_j_beg();

        // Use quadrature to initialize state and tracer data
        parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l=0; l < num_state  ; l++) { state  (l,hs+k,hs+j,hs+i,iens) = 0.; }
          for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+i,iens) = 0.; }
          //Use Gauss-Legendre quadrature
          for (int kk=0; kk<nqpoints; kk++) {
            for (int jj=0; jj<nqpoints; jj++) {
              for (int ii=0; ii<nqpoints; ii++) {
                real x = (i+i_beg+0.5)*dx + (qpoints(ii)-0.5)*dx;
                real y = (j+j_beg+0.5)*dy + (qpoints(jj)-0.5)*dy;   if (sim2d) y = ylen/2;
                real z = (k      +0.5)*dz + (qpoints(kk)-0.5)*dz;
                real rho, u, v, w, theta, rho_v, hr, ht;

                if (init_data_int == DATA_THERMAL) {
                  thermal(x,y,z,xlen,ylen,grav,C0,gamma,cp_d,p0,R_d,R_v,rho,u,v,w,theta,rho_v,hr,ht);
                }

                if (sim2d) v = 0;

                real wt = qweights(ii)*qweights(jj)*qweights(kk);
                state(idR,hs+k,hs+j,hs+i,iens) += ( rho - hr )          * wt;
                state(idU,hs+k,hs+j,hs+i,iens) += rho*u                 * wt;
                state(idV,hs+k,hs+j,hs+i,iens) += rho*v                 * wt;
                state(idW,hs+k,hs+j,hs+i,iens) += rho*w                 * wt;
                state(idT,hs+k,hs+j,hs+i,iens) += ( rho*theta - hr*ht ) * wt;
                for (int tr=0; tr < num_tracers; tr++) {
                  if (tr == idWV) { tracers(tr,hs+k,hs+j,hs+i,iens) += rho_v * wt; }
                  else            { tracers(tr,hs+k,hs+j,hs+i,iens) += 0     * wt; }
                }
              }
            }
          }
        });


        // Compute hydrostatic background cell averages using quadrature
        parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz,nens) , YAKL_LAMBDA (int k, int iens) {
          hy_dens_cells      (k,iens) = 0.;
          hy_dens_theta_cells(k,iens) = 0.;
          for (int kk=0; kk<nqpoints; kk++) {
            real z = (k+0.5)*dz + (qpoints(kk)-0.5)*dz;
            real hr, ht;

            if (init_data_int == DATA_THERMAL) { hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht); }

            hy_dens_cells      (k,iens) += hr    * qweights(kk);
            hy_dens_theta_cells(k,iens) += hr*ht * qweights(kk);
          }
        });

        // Compute hydrostatic background cell edge values
        parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz+1,nens) , YAKL_LAMBDA (int k, int iens) {
          real z = k*dz;
          real hr, ht;

          if (init_data_int == DATA_THERMAL) { hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht); }

          hy_dens_edges      (k,iens) = hr   ;
          hy_dens_theta_edges(k,iens) = hr*ht;
        });

      } else if (init_data_int == DATA_CITY) {

        coupler.add_option<int>("bc_x",BC_PERIODIC);
        coupler.add_option<int>("bc_y",BC_PERIODIC);
        coupler.add_option<int>("bc_z",BC_WALL    );
        coupler.set_option<bool>("use_immersed_boundaries",true);
        immersed_proportion = 0;

        real height_mean = 60;
        real height_std  = 10;

        int building_length = 30;
        int cells_per_building = (int) std::round(building_length / dx);
        int buildings_pad = 20;
        int nblocks_x = (static_cast<int>(xlen)/building_length - 2*buildings_pad)/3;
        int nblocks_y = (static_cast<int>(ylen)/building_length - 2*buildings_pad)/9;
        int nbuildings_x = nblocks_x * 3;
        int nbuildings_y = nblocks_y * 9;

        realHost2d building_heights_host("building_heights",nbuildings_y,nbuildings_x);
        if (coupler.is_mainproc()) {
          std::mt19937 gen{17};
          std::normal_distribution<> d{height_mean, height_std};
          for (int j=0; j < nbuildings_y; j++) {
            for (int i=0; i < nbuildings_x; i++) {
              building_heights_host(j,i) = d(gen);
            }
          }
        }
        auto type = coupler.get_mpi_data_type();
        MPI_Bcast( building_heights_host.data() , building_heights_host.size() , type , 0 , MPI_COMM_WORLD);
        auto building_heights = building_heights_host.createDeviceCopy();

        // Define quadrature weights and points for 3-point rules
        const int nqpoints = 9;
        SArray<real,1,nqpoints> qpoints;
        SArray<real,1,nqpoints> qweights;

        TransformMatrices::get_gll_points (qpoints );
        TransformMatrices::get_gll_weights(qweights);

        // Use quadrature to initialize state and tracer data
        parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l=0; l < num_state  ; l++) { state  (l,hs+k,hs+j,hs+i,iens) = 0.; }
          for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+i,iens) = 0.; }
          //Use Gauss-Legendre quadrature
          for (int kk=0; kk<nqpoints; kk++) {
            for (int jj=0; jj<nqpoints; jj++) {
              for (int ii=0; ii<nqpoints; ii++) {
                real x = (i+i_beg+0.5)*dx + (qpoints(ii)-0.5)*dx;
                real y = (j+j_beg+0.5)*dy + (qpoints(jj)-0.5)*dy;   if (sim2d) y = ylen/2;
                real z = (k      +0.5)*dz + (qpoints(kk)-0.5)*dz;
                real rho, u, v, w, theta, rho_v, hr, ht;

                if (enable_gravity) {
                  hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht);
                } else {
                  hr = 1.15;
                  ht = 300;
                }

                rho   = hr;
                u     = 20;
                v     = 0;
                w     = 0;
                theta = ht;
                rho_v = 0;

                if (sim2d) v = 0;

                real wt = qweights(ii)*qweights(jj)*qweights(kk);
                state(idR,hs+k,hs+j,hs+i,iens) += ( rho - hr )          * wt;
                state(idU,hs+k,hs+j,hs+i,iens) += rho*u                 * wt;
                state(idV,hs+k,hs+j,hs+i,iens) += rho*v                 * wt;
                state(idW,hs+k,hs+j,hs+i,iens) += rho*w                 * wt;
                state(idT,hs+k,hs+j,hs+i,iens) += ( rho*theta - hr*ht ) * wt;
                for (int tr=0; tr < num_tracers; tr++) {
                  if (tr == idWV) { tracers(tr,hs+k,hs+j,hs+i,iens) += rho_v * wt; }
                  else            { tracers(tr,hs+k,hs+j,hs+i,iens) += 0     * wt; }
                }
              }
            }
          }
          int inorm = (static_cast<int>(i_beg)+i)/cells_per_building - buildings_pad;
          int jnorm = (static_cast<int>(j_beg)+j)/cells_per_building - buildings_pad;
          if ( ( inorm >= 0 && inorm < nblocks_x*3 && inorm%3 < 2 ) &&
               ( jnorm >= 0 && jnorm < nblocks_y*9 && jnorm%9 < 8 ) ) {
            if ( k <= std::ceil( building_heights(jnorm,inorm) / dz ) ) {
              immersed_proportion(k,j,i,iens) = 1;
              // state(idU,hs+k,hs+j,hs+i,iens) = 0;
              // state(idV,hs+k,hs+j,hs+i,iens) = 0;
              // state(idW,hs+k,hs+j,hs+i,iens) = 0;
            }
          }
        });
        if (enable_gravity) {
          // Compute hydrostatic background cell averages using quadrature
          parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz,nens) , YAKL_LAMBDA (int k, int iens) {
            hy_dens_cells      (k,iens) = 0.;
            hy_dens_theta_cells(k,iens) = 0.;
            for (int kk=0; kk<nqpoints; kk++) {
              real z = (k+0.5)*dz + (qpoints(kk)-0.5)*dz;
              real hr, ht;

              hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht);

              hy_dens_cells      (k,iens) += hr    * qweights(kk);
              hy_dens_theta_cells(k,iens) += hr*ht * qweights(kk);
            }
          });

          // Compute hydrostatic background cell edge values
          parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz+1,nens) , YAKL_LAMBDA (int k, int iens) {
            real z = k*dz;
            real hr, ht;

            hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht);

            hy_dens_edges      (k,iens) = hr   ;
            hy_dens_theta_edges(k,iens) = hr*ht;
          });
        } else {
          hy_dens_cells = 1.15;
          hy_dens_edges = 1.15;
          hy_dens_theta_cells = 1.15*300;
          hy_dens_theta_edges = 1.15*300;
        }

      } else if (init_data_int == DATA_BUILDING) {

        coupler.add_option<int>("bc_x",BC_PERIODIC);
        coupler.add_option<int>("bc_y",BC_PERIODIC);
        coupler.add_option<int>("bc_z",BC_WALL    );
        coupler.set_option<bool>("use_immersed_boundaries",true);
        immersed_proportion = 0;

        // Define quadrature weights and points for 3-point rules
        const int nqpoints = 9;
        SArray<real,1,nqpoints> qpoints;
        SArray<real,1,nqpoints> qweights;

        TransformMatrices::get_gll_points (qpoints );
        TransformMatrices::get_gll_weights(qweights);

        // Use quadrature to initialize state and tracer data
        parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l=0; l < num_state  ; l++) { state  (l,hs+k,hs+j,hs+i,iens) = 0.; }
          for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+i,iens) = 0.; }
          //Use Gauss-Legendre quadrature
          for (int kk=0; kk<nqpoints; kk++) {
            for (int jj=0; jj<nqpoints; jj++) {
              for (int ii=0; ii<nqpoints; ii++) {
                real x = (i+i_beg+0.5)*dx + (qpoints(ii)-0.5)*dx;
                real y = (j+j_beg+0.5)*dy + (qpoints(jj)-0.5)*dy;   if (sim2d) y = ylen/2;
                real z = (k      +0.5)*dz + (qpoints(kk)-0.5)*dz;
                real rho, u, v, w, theta, rho_v, hr, ht;

                if (enable_gravity) {
                  hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht);
                } else {
                  hr = 1.15;
                  ht = 300;
                }

                rho   = hr;
                u     = 20;
                v     = 0;
                w     = 0;
                theta = ht;
                rho_v = 0;

                if (sim2d) v = 0;

                real wt = qweights(ii)*qweights(jj)*qweights(kk);
                state(idR,hs+k,hs+j,hs+i,iens) += ( rho - hr )          * wt;
                state(idU,hs+k,hs+j,hs+i,iens) += rho*u                 * wt;
                state(idV,hs+k,hs+j,hs+i,iens) += rho*v                 * wt;
                state(idW,hs+k,hs+j,hs+i,iens) += rho*w                 * wt;
                state(idT,hs+k,hs+j,hs+i,iens) += ( rho*theta - hr*ht ) * wt;
                for (int tr=0; tr < num_tracers; tr++) {
                  if (tr == idWV) { tracers(tr,hs+k,hs+j,hs+i,iens) += rho_v * wt; }
                  else            { tracers(tr,hs+k,hs+j,hs+i,iens) += 0     * wt; }
                }
              }
            }
          }
          real x0 = 0.3*nx_glob;
          real y0 = 0.5*ny_glob;
          real xr = 0.05*ny_glob;
          real yr = 0.05*ny_glob;
          if ( std::abs(i_beg+i-x0) <= xr && std::abs(j_beg+j-y0) <= yr && k <= 0.2*nz ) {
            immersed_proportion(k,j,i,iens) = 1;
            // state(idU,hs+k,hs+j,hs+i,iens) = 0;
            // state(idV,hs+k,hs+j,hs+i,iens) = 0;
            // state(idW,hs+k,hs+j,hs+i,iens) = 0;
          }
        });

        if (enable_gravity) {
          // Compute hydrostatic background cell averages using quadrature
          parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz,nens) , YAKL_LAMBDA (int k, int iens) {
            hy_dens_cells      (k,iens) = 0.;
            hy_dens_theta_cells(k,iens) = 0.;
            for (int kk=0; kk<nqpoints; kk++) {
              real z = (k+0.5)*dz + (qpoints(kk)-0.5)*dz;
              real hr, ht;

              hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht);

              hy_dens_cells      (k,iens) += hr    * qweights(kk);
              hy_dens_theta_cells(k,iens) += hr*ht * qweights(kk);
            }
          });

          // Compute hydrostatic background cell edge values
          parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz+1,nens) , YAKL_LAMBDA (int k, int iens) {
            real z = k*dz;
            real hr, ht;

            hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht);

            hy_dens_edges      (k,iens) = hr   ;
            hy_dens_theta_edges(k,iens) = hr*ht;
          });
        } else {
          hy_dens_cells = 1.15;
          hy_dens_edges = 1.15;
          hy_dens_theta_cells = 1.15*300;
          hy_dens_theta_edges = 1.15*300;
        }

      }

      // Convert the initialized state and tracers arrays back to the coupler state
      convert_dynamics_to_coupler( coupler , state , tracers );

      // Output the initial state
      if (out_freq >= 0. ) output( coupler , etime );

      // Some modules might need to use hydrostasis to project values into material boundaries
      // So let's put it into the coupler's data manager just in case
      dm.register_and_allocate<real>("hy_dens_cells"      ,"hydrostatic density cell averages"      ,{nz,nens});
      dm.register_and_allocate<real>("hy_dens_theta_cells","hydrostatic density*theta cell averages",{nz,nens});
      auto dm_hy_dens_cells       = dm.get<real,2>("hy_dens_cells"      );
      auto dm_hy_dens_theta_cells = dm.get<real,2>("hy_dens_theta_cells");
      hy_dens_cells      .deep_copy_to( dm_hy_dens_cells      );
      hy_dens_theta_cells.deep_copy_to( dm_hy_dens_theta_cells);

      // Register the tracers in the coupler so the user has access if they want (and init to zero)
      dm.register_and_allocate<real>("state_flux_x"  ,"state_flux_x"  ,{num_state  ,nz  ,ny  ,nx+1,nens},{"num_state"  ,"z"  ,"y"  ,"xp1","nens"});
      dm.register_and_allocate<real>("state_flux_y"  ,"state_flux_y"  ,{num_state  ,nz  ,ny+1,nx  ,nens},{"num_state"  ,"z"  ,"yp1","x"  ,"nens"});
      dm.register_and_allocate<real>("state_flux_z"  ,"state_flux_z"  ,{num_state  ,nz+1,ny  ,nx  ,nens},{"num_state"  ,"zp1","y"  ,"x"  ,"nens"});
      dm.register_and_allocate<real>("tracers_flux_x","tracers_flux_x",{num_tracers,nz  ,ny  ,nx+1,nens},{"num_tracers","z"  ,"y"  ,"xp1","nens"});
      dm.register_and_allocate<real>("tracers_flux_y","tracers_flux_y",{num_tracers,nz  ,ny+1,nx  ,nens},{"num_tracers","z"  ,"yp1","x"  ,"nens"});
      dm.register_and_allocate<real>("tracers_flux_z","tracers_flux_z",{num_tracers,nz+1,ny  ,nx  ,nens},{"num_tracers","zp1","y"  ,"x"  ,"nens"});
      dm.get<real,5>("state_flux_x"  ) = 0;
      dm.get<real,5>("state_flux_y"  ) = 0;
      dm.get<real,5>("state_flux_z"  ) = 0;
      dm.get<real,5>("tracers_flux_x") = 0;
      dm.get<real,5>("tracers_flux_y") = 0;
      dm.get<real,5>("tracers_flux_z") = 0;
    }


    // Initialize the supercell test case
    void init_supercell( core::Coupler &coupler , real5d &state , real5d &tracers ) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;
      real constexpr z_0    = 0;
      real constexpr z_trop = 12000;
      real constexpr T_0    = 300;
      real constexpr T_trop = 213;
      real constexpr T_top  = 213;
      real constexpr p_0    = 100000;

      YAKL_SCOPE( hy_dens_cells       , this->hy_dens_cells       );
      YAKL_SCOPE( hy_dens_theta_cells , this->hy_dens_theta_cells );
      YAKL_SCOPE( hy_dens_edges       , this->hy_dens_edges       );
      YAKL_SCOPE( hy_dens_theta_edges , this->hy_dens_theta_edges );
      YAKL_SCOPE( idWV                , this->idWV                );
      YAKL_SCOPE( gll_pts             , this->gll_pts             );
      YAKL_SCOPE( gll_wts             , this->gll_wts             );

      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto dx          = coupler.get_dx();
      auto dy          = coupler.get_dy();
      auto dz          = coupler.get_dz();
      auto xlen        = coupler.get_xlen();
      auto ylen        = coupler.get_ylen();
      auto sim2d       = coupler.is_sim2d();
      auto R_d         = coupler.get_option<real>("R_d"    );
      auto R_v         = coupler.get_option<real>("R_v"    );
      auto grav        = coupler.get_option<real>("grav"   );
      auto gamma       = coupler.get_option<real>("gamma_d");
      auto C0          = coupler.get_option<real>("C0"     );
      auto num_tracers = coupler.get_num_tracers();
      auto i_beg       = coupler.get_i_beg();
      auto j_beg       = coupler.get_j_beg();

      // Temporary arrays used to compute the initial state for high-CAPE supercell conditions
      real3d quad_temp       ("quad_temp"       ,nz,ord-1,ord);
      real2d hyDensGLL       ("hyDensGLL"       ,nz,ord);
      real2d hyDensThetaGLL  ("hyDensThetaGLL"  ,nz,ord);
      real2d hyDensVapGLL    ("hyDensVapGLL"    ,nz,ord);
      real2d hyPressureGLL   ("hyPressureGLL"   ,nz,ord);
      real1d hyDensCells     ("hyDensCells"     ,nz);
      real1d hyDensThetaCells("hyDensThetaCells",nz);

      real ztop = coupler.get_zlen();

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
          for (int iens=0; iens < nens; iens++) {
            hy_dens_edges      (k,iens) = dens;
            hy_dens_theta_edges(k,iens) = dens_theta;
          }
        }
        if (k == nz-1 && kk == ord-1) {
          for (int iens=0; iens < nens; iens++) {
            hy_dens_edges      (k+1,iens) = dens;
            hy_dens_theta_edges(k+1,iens) = dens_theta;
          }
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
        for (int iens=0; iens < nens; iens++) {
          hy_dens_cells      (k,iens) = dens;
          hy_dens_theta_cells(k,iens) = dens_theta;
        }
      });

      // Initialize the state
      parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        state(idR,hs+k,hs+j,hs+i,iens) = 0;
        state(idU,hs+k,hs+j,hs+i,iens) = 0;
        state(idV,hs+k,hs+j,hs+i,iens) = 0;
        state(idW,hs+k,hs+j,hs+i,iens) = 0;
        state(idT,hs+k,hs+j,hs+i,iens) = 0;
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i,iens) = 0; }
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
              state  (idR ,hs+k,hs+j,hs+i,iens) += (dens - hyDensGLL(k,kk))            * factor;
              state  (idU ,hs+k,hs+j,hs+i,iens) += dens * uvel                         * factor;
              state  (idV ,hs+k,hs+j,hs+i,iens) += dens * vvel                         * factor;
              state  (idW ,hs+k,hs+j,hs+i,iens) += dens * wvel                         * factor;
              state  (idT ,hs+k,hs+j,hs+i,iens) += (dens_theta - hyDensThetaGLL(k,kk)) * factor;
              tracers(idWV,hs+k,hs+j,hs+i,iens) += dens_vap                            * factor;
            }
          }
        }
      });
    }


    // Convert dynamics state and tracers arrays to the coupler state and write to the coupler's data
    void convert_dynamics_to_coupler( core::Coupler &coupler , realConst5d state , realConst5d tracers ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      YAKL_SCOPE( hy_dens_cells       , this->hy_dens_cells       );
      YAKL_SCOPE( hy_dens_theta_cells , this->hy_dens_theta_cells );
      YAKL_SCOPE( idWV                , this->idWV                );
      YAKL_SCOPE( tracer_adds_mass    , this->tracer_adds_mass    );

      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto R_d         = coupler.get_option<real>("R_d"    );
      auto R_v         = coupler.get_option<real>("R_v"    );
      auto gamma       = coupler.get_option<real>("gamma_d");
      auto C0          = coupler.get_option<real>("C0"     );
      auto num_tracers = coupler.get_num_tracers();

      auto &dm = coupler.get_data_manager_readwrite();

      // Get state from the coupler
      auto dm_rho_d = dm.get<real,4>("density_dry");
      auto dm_uvel  = dm.get<real,4>("uvel"       );
      auto dm_vvel  = dm.get<real,4>("vvel"       );
      auto dm_wvel  = dm.get<real,4>("wvel"       );
      auto dm_temp  = dm.get<real,4>("temp"       );

      // Get tracers from the coupler
      core::MultiField<real,4> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) {
        dm_tracers.add_field( dm.get<real,4>(tracer_names[tr]) );
      }

      // Convert from state and tracers arrays to the coupler's data
      parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        real rho   = state(idR,hs+k,hs+j,hs+i,iens) + hy_dens_cells(k,iens);
        real u     = state(idU,hs+k,hs+j,hs+i,iens) / rho;
        real v     = state(idV,hs+k,hs+j,hs+i,iens) / rho;
        real w     = state(idW,hs+k,hs+j,hs+i,iens) / rho;
        real theta = ( state(idT,hs+k,hs+j,hs+i,iens) + hy_dens_theta_cells(k,iens) ) / rho;
        real press = C0 * pow( rho*theta , gamma );

        real rho_v = tracers(idWV,hs+k,hs+j,hs+i,iens);
        real rho_d = rho;
        for (int tr=0; tr < num_tracers; tr++) {
          if (tracer_adds_mass(tr)) rho_d -= tracers(tr,hs+k,hs+j,hs+i,iens);
        }
        real temp = press / ( rho_d * R_d + rho_v * R_v );

        dm_rho_d(k,j,i,iens) = rho_d;
        dm_uvel (k,j,i,iens) = u;
        dm_vvel (k,j,i,iens) = v;
        dm_wvel (k,j,i,iens) = w;
        dm_temp (k,j,i,iens) = temp;
        for (int tr=0; tr < num_tracers; tr++) {
          dm_tracers(tr,k,j,i,iens) = tracers(tr,hs+k,hs+j,hs+i,iens);
        }
      });
    }


    // Convert coupler's data to state and tracers arrays
    void convert_coupler_to_dynamics( core::Coupler const &coupler , real5d &state , real5d &tracers ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      YAKL_SCOPE( hy_dens_cells       , this->hy_dens_cells       );
      YAKL_SCOPE( hy_dens_theta_cells , this->hy_dens_theta_cells );
      YAKL_SCOPE( idWV                , this->idWV                );
      YAKL_SCOPE( tracer_adds_mass    , this->tracer_adds_mass    );

      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto R_d         = coupler.get_option<real>("R_d"    );
      auto R_v         = coupler.get_option<real>("R_v"    );
      auto gamma       = coupler.get_option<real>("gamma_d");
      auto C0          = coupler.get_option<real>("C0"     );
      auto num_tracers = coupler.get_num_tracers();

      auto &dm = coupler.get_data_manager_readonly();

      // Get the coupler's state (as const because it's read-only)
      auto dm_rho_d = dm.get<real const,4>("density_dry");
      auto dm_uvel  = dm.get<real const,4>("uvel"       );
      auto dm_vvel  = dm.get<real const,4>("vvel"       );
      auto dm_wvel  = dm.get<real const,4>("wvel"       );
      auto dm_temp  = dm.get<real const,4>("temp"       );

      // Get the coupler's tracers (as const because it's read-only)
      core::MultiField<real const,4> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) {
        dm_tracers.add_field( dm.get<real const,4>(tracer_names[tr]) );
      }

      // Convert from the coupler's state to the dycore's state and tracers arrays
      parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        real rho_d = dm_rho_d(k,j,i,iens);
        real u     = dm_uvel (k,j,i,iens);
        real v     = dm_vvel (k,j,i,iens);
        real w     = dm_wvel (k,j,i,iens);
        real temp  = dm_temp (k,j,i,iens);
        real rho_v = dm_tracers(idWV,k,j,i,iens);
        real press = rho_d * R_d * temp + rho_v * R_v * temp;

        real rho = rho_d;
        for (int tr=0; tr < num_tracers; tr++) {
          if (tracer_adds_mass(tr)) rho += dm_tracers(tr,k,j,i,iens);
        }
        real theta = pow( press/C0 , 1._fp / gamma ) / rho;

        state(idR,hs+k,hs+j,hs+i,iens) = rho - hy_dens_cells(k,iens);
        state(idU,hs+k,hs+j,hs+i,iens) = rho * u;
        state(idV,hs+k,hs+j,hs+i,iens) = rho * v;
        state(idW,hs+k,hs+j,hs+i,iens) = rho * w;
        state(idT,hs+k,hs+j,hs+i,iens) = rho * theta - hy_dens_theta_cells(k,iens);
        for (int tr=0; tr < num_tracers; tr++) {
          tracers(tr,hs+k,hs+j,hs+i,iens) = dm_tracers(tr,k,j,i,iens);
        }
      });
    }


    // Perform file output
    void output( core::Coupler const &coupler , real etime ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;
      yakl::timer_start("output");

      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto dx          = coupler.get_dx();
      auto dy          = coupler.get_dy();
      auto dz          = coupler.get_dz();
      auto num_tracers = coupler.get_num_tracers();
      int i_beg = coupler.get_i_beg();
      int j_beg = coupler.get_j_beg();
      int iens = 0;

      if (coupler.get_option<bool>("file_per_process",false)) {

        yakl::SimpleNetCDF nc;
        MPI_Offset ulIndex = 0; // Unlimited dimension index to place this data at

        
        std::stringstream fname;
        fname << coupler.get_option<std::string>("out_prefix") << "_" << std::setw(8) << std::setfill('0')
              << coupler.get_myrank() << ".nc";

        if (etime == 0) {
          nc.create( fname.str() , yakl::NETCDF_MODE_REPLACE );

          nc.createDim( "x" , coupler.get_nx() );
          nc.createDim( "y" , coupler.get_ny() );
          nc.createDim( "z" , nz );
          nc.createDim( "t" );

          // x-coordinate
          real1d xloc("xloc",nx);
          parallel_for( YAKL_AUTO_LABEL() , nx , YAKL_LAMBDA (int i) { xloc(i) = (i+i_beg+0.5)*dx; });
          nc.write( xloc , "x" , {"x"} );

          // y-coordinate
          real1d yloc("yloc",ny);
          parallel_for( YAKL_AUTO_LABEL() , ny , YAKL_LAMBDA (int j) { yloc(j) = (j+j_beg+0.5)*dy; });
          nc.write( yloc , "y" , {"y"} );

          // z-coordinate
          real1d zloc("zloc",nz);
          parallel_for( YAKL_AUTO_LABEL() , nz , YAKL_LAMBDA (int k) { zloc(k) = (k      +0.5)*dz; });
          nc.write( zloc , "z" , {"z"} );
          nc.write1( 0._fp , "t" , 0 , "t" );

        } else {

          nc.open( fname.str() , yakl::NETCDF_MODE_WRITE );
          ulIndex = nc.getDimSize("t");

          // Write the elapsed time
          nc.write1(etime,"t",ulIndex,"t");

        }

        std::vector<std::string> varnames(num_state+num_tracers);
        varnames[0] = "density_dry";
        varnames[1] = "uvel";
        varnames[2] = "vvel";
        varnames[3] = "wvel";
        varnames[4] = "temp";
        auto tracer_names = coupler.get_tracer_names();
        for (int tr=0; tr < num_tracers; tr++) { varnames[num_state+tr] = tracer_names[tr]; }

        auto &dm = coupler.get_data_manager_readonly();
        real3d data("data",nz,ny,nx);
        for (int i=0; i < varnames.size(); i++) {
          auto var = dm.get<real const,4>(varnames[i]);
          parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) { data(k,j,i) = var(k,j,i,iens); });
          nc.write1(data,varnames[i],{"z","y","x"},ulIndex,"t");
        }

        nc.close();

      } else { // if file_per_process

        yakl::SimplePNetCDF nc;
        MPI_Offset ulIndex = 0; // Unlimited dimension index to place this data at

        std::stringstream fname;
        fname << coupler.get_option<std::string>("out_prefix") << ".nc";
        if (etime == 0) {
          nc.create(fname.str() , NC_CLOBBER | NC_64BIT_DATA);

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
          nc.create_var<real>( "temp"        , {"t","z","y","x"} );
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

          nc.open(fname.str());
          ulIndex = nc.get_dim_size("t");

          // Write the elapsed time
          nc.begin_indep_data();
          if (coupler.is_mainproc()) {
            nc.write1(etime,"t",ulIndex,"t");
          }
          nc.end_indep_data();

        }

        std::vector<std::string> varnames(num_state+num_tracers);
        varnames[0] = "density_dry";
        varnames[1] = "uvel";
        varnames[2] = "vvel";
        varnames[3] = "wvel";
        varnames[4] = "temp";
        auto tracer_names = coupler.get_tracer_names();
        for (int tr=0; tr < num_tracers; tr++) { varnames[num_state+tr] = tracer_names[tr]; }

        auto &dm = coupler.get_data_manager_readonly();
        real3d data("data",nz,ny,nx);
        for (int i=0; i < varnames.size(); i++) {
          auto var = dm.get<real const,4>(varnames[i]);
          parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) { data(k,j,i) = var(k,j,i,iens); });
          nc.write1_all(data.createHostCopy(),varnames[i],ulIndex,{0,j_beg,i_beg},"t");
        }

        nc.close();

      } // if file_per_process

      yakl::timer_stop("output");
    }


  };

}


