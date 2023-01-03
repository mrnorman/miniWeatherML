
#pragma once

#include "main_header.h"
#include "MultipleFields.h"
#include "TransformMatrices.h"
#include "WenoLimiter.h"

namespace modules {

  // Solves the energy-conserving Euler equations without a gravity source term for ideal gas. No tracers for now.
  // Assuming periodic in all dimensions for now

  class Dynamics_Euler_Energy {
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
    int  static constexpr idT = 4;  // Density * (internal + kinetic energy)

    real static constexpr gamma = 1.4;
    real static constexpr R_d   = 287;
    real static constexpr c_v   = 717.5;
    real static constexpr c_p   = 1004.5;

    // IDs for the test cases
    int  static constexpr DATA_KELVIN_HELMHOLTZ = 0;

    int  static constexpr RIEMANN_NATIVE   = 1;
    int  static constexpr RIEMANN_LLF      = 2;
    int  static constexpr RIEMANN_PRESSURE = 3;

    int  static constexpr riemann_choice = RIEMANN_PRESSURE;

    real        etime;         // Elapsed time
    real        out_freq;      // Frequency out file output
    int         num_out;       // Number of outputs produced thus far
    std::string fname;         // File name for file output
    int         init_data_int; // Integer representation of the type of initial data to use (test case)

    SArray<real,2,ord,ord>        sten_to_coefs;    // Matrix to convert ord stencil avgs to ord poly coefs
    SArray<real,2,ord,2  >        coefs_to_gll;     // Matrix to convert ord poly coefs to two GLL points
    SArray<real,3,hs+1,hs+1,hs+1> weno_recon_lower; // WENO's lower-order reconstruction matrices (sten_to_coefs)
    SArray<real,1,hs+2>           weno_idl;         // Ideal weights for WENO
    real                          weno_sigma;       // WENO sigma parameter (handicap high-order TV estimate)


    // Compute the maximum stable time step using very conservative assumptions about max wind speed
    real compute_time_step( core::Coupler const &coupler ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto &dm = coupler.get_data_manager_readonly();
      auto rho    = dm.get<real const,3>("density");
      auto umom   = dm.get<real const,3>("umom");
      auto vmom   = dm.get<real const,3>("vmom");
      auto wmom   = dm.get<real const,3>("wmom");
      auto energy = dm.get<real const,3>("energy");
      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      auto dx = coupler.get_dx();
      auto dy = coupler.get_dy();
      auto dz = coupler.get_dz();
      real cfl = 0.5;
      real3d dt3d("dt3d",nz,ny,nx);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        real r = rho (k,j,i);
        real u = umom(k,j,i)/r;
        real v = vmom(k,j,i)/r;
        real w = wmom(k,j,i)/r;
        real re = energy(k,j,i);
        real K = (u*u + v*v + w*w)/2;
        real p = (gamma-1)*(re - r*K);
        real cs = sqrt(gamma*p/r);
        dt3d(k,j,i) = std::min( { dx*cfl/(std::abs(u)+cs) ,
                                  dy*cfl/(std::abs(v)+cs) ,
                                  dz*cfl/(std::abs(w)+cs) } );
      });
      real dt_loc = yakl::intrinsics::minval(dt3d);
      real dt_glob;
      auto data_type = coupler.get_mpi_data_type();
      MPI_Allreduce(&dt_loc, &dt_glob, 1, data_type, MPI_MIN, MPI_COMM_WORLD);
      return dt_glob;
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

      // Cells [0:hs-1] are the left halos, and cells [nx+hs:nx+2*hs-1] are the right halos
      real4d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);

      // Populate the state arrays using data from the coupler, convert to the dycore's desired state
      convert_coupler_to_dynamics( coupler , state );

      real etime_loc = 0;
      
      while (etime_loc < dt_phys) {
        real dt_dyn = compute_time_step( coupler );
        if (etime_loc + dt_dyn > dt_phys) dt_dyn = dt_phys - etime_loc;

        // SSPRK3 requires temporary arrays to hold intermediate state
        real4d state_tmp   ("state_tmp"   ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
        real4d state_tend  ("state_tend"  ,num_state  ,nz     ,ny     ,nx     );
        //////////////
        // Stage 1
        //////////////
        compute_tendencies( coupler , state     , state_tend , dt_dyn );
        // Apply tendencies
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          for (int l = 0; l < num_state  ; l++) {
            state_tmp  (l,hs+k,hs+j,hs+i) = state  (l,hs+k,hs+j,hs+i) + dt_dyn * state_tend  (l,k,j,i);
          }
        });
        //////////////
        // Stage 2
        //////////////
        compute_tendencies( coupler , state_tmp , state_tend , (1._fp/4._fp) * dt_dyn );
        // Apply tendencies
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          for (int l = 0; l < num_state  ; l++) {
            state_tmp  (l,hs+k,hs+j,hs+i) = (3._fp/4._fp) * state      (l,hs+k,hs+j,hs+i) + 
                                            (1._fp/4._fp) * state_tmp  (l,hs+k,hs+j,hs+i) +
                                            (1._fp/4._fp) * dt_dyn * state_tend  (l,k,j,i);
          }
        });
        //////////////
        // Stage 3
        //////////////
        compute_tendencies( coupler , state_tmp , state_tend , (2._fp/3._fp) * dt_dyn );
        // Apply tendencies
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          for (int l = 0; l < num_state  ; l++) {
            state      (l,hs+k,hs+j,hs+i) = (1._fp/3._fp) * state      (l,hs+k,hs+j,hs+i) +
                                            (2._fp/3._fp) * state_tmp  (l,hs+k,hs+j,hs+i) +
                                            (2._fp/3._fp) * dt_dyn * state_tend  (l,k,j,i);
          }
        });

        etime_loc += dt_dyn;
      }

      // Convert the dycore's state back to the coupler's state
      convert_dynamics_to_coupler( coupler , state );

      // Advance the dycore's tracking of total ellapsed time
      etime += dt_phys;
      // Do output and inform the user if it's time to do output
      if (out_freq >= 0. && etime / out_freq >= num_out+1) {
        output( coupler , etime );
        num_out++;
        // Let the user know what the max vertical velocity is to ensure the model hasn't crashed
        auto &dm = coupler.get_data_manager_readonly();
        using yakl::componentwise::operator/;
        using yakl::componentwise::operator*;
        using yakl::componentwise::operator+;
        auto uvel = dm.get_collapsed<real const>("umom") / dm.get_collapsed<real const>("density");
        auto vvel = dm.get_collapsed<real const>("vmom") / dm.get_collapsed<real const>("density");
        auto wvel = dm.get_collapsed<real const>("wmom") / dm.get_collapsed<real const>("density");
        real maxw_loc = sqrt( maxval( uvel*uvel + vvel*vvel + wvel*wvel ) );
        real maxw;
        auto data_type = coupler.get_mpi_data_type();
        MPI_Reduce( &maxw_loc , &maxw , 1 , data_type , MPI_MAX , 0 , MPI_COMM_WORLD );
        if (coupler.is_mainproc()) {
          std::cout << "Etime , dtphys, max_wind: " << std::scientific << std::setw(10) << etime   << " , " 
                                                    << std::scientific << std::setw(10) << dt_phys << " , "
                                                    << std::scientific << std::setw(10) << maxw    << std::endl;
        }
      }
    }


    // Compute the tendencies for state for one semi-discretized step inside the RK integrator
    // Tendencies are the time rate of change for a quantity
    // Coupler is non-const because we are writing to the flux variables
    void compute_tendencies( core::Coupler &coupler , real4d const &state , real4d const &state_tend , real dt ) const {
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
      bool sim2d = ny == 1;

      // A slew of things to bring from class scope into local scope so that lambdas copy them by value to the GPU
      YAKL_SCOPE( coefs_to_gll     , this->coefs_to_gll     );
      YAKL_SCOPE( sten_to_coefs    , this->sten_to_coefs    );
      YAKL_SCOPE( weno_recon_lower , this->weno_recon_lower );
      YAKL_SCOPE( weno_idl         , this->weno_idl         );
      YAKL_SCOPE( weno_sigma       , this->weno_sigma       );

      halo_exchange( coupler , state );

      // These arrays store high-order-accurate samples of the state at cell edges after cell-centered recon
      real5d state_limits_x  ("state_limits_x",num_state,2,nz  ,ny  ,nx+1);
      real5d state_limits_y  ("state_limits_y",num_state,2,nz  ,ny+1,nx  );
      real5d state_limits_z  ("state_limits_z",num_state,2,nz+1,ny  ,nx  );

      // Compute samples of state at cell edges using cell-centered reconstructions at high-order with WENO
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
        } else {
          for (int l=0; l < num_state; l++) {
            state_limits_y(l,1,k,j  ,i) = 0;
            state_limits_y(l,0,k,j+1,i) = 0;
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
            int ind_k = k+s;
            if (ind_k < hs     ) ind_k += nz;
            if (ind_k > hs+nz-1) ind_k -= nz;
            stencil(s) = state(l,ind_k,hs+j,hs+i);
          }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
          state_limits_z(l,1,k  ,j,i) = gll(0);
          state_limits_z(l,0,k+1,j,i) = gll(1);
        }
      });

      edge_exchange( coupler , state_limits_x , state_limits_y );

      // The store a single values flux at cell edges
      auto &dm = coupler.get_data_manager_readwrite();
      auto state_flux_x = dm.get<real,4>("state_flux_x");
      auto state_flux_y = dm.get<real,4>("state_flux_y");
      auto state_flux_z = dm.get<real,4>("state_flux_z");

      // Use upwind Riemann solver to reconcile discontinuous limits of state at each cell edges
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz+1,ny+1,nx+1) , YAKL_LAMBDA (int k, int j, int i) {
        ////////////////////////////////////////////////////////
        // X-direction
        ////////////////////////////////////////////////////////
        if (j < ny && k < nz) {
          real q1_L = state_limits_x(idR,0,k,j,i);   real q1_R = state_limits_x(idR,1,k,j,i);
          real q2_L = state_limits_x(idU,0,k,j,i);   real q2_R = state_limits_x(idU,1,k,j,i);
          real q3_L = state_limits_x(idV,0,k,j,i);   real q3_R = state_limits_x(idV,1,k,j,i);
          real q4_L = state_limits_x(idW,0,k,j,i);   real q4_R = state_limits_x(idW,1,k,j,i);
          real q5_L = state_limits_x(idT,0,k,j,i);   real q5_R = state_limits_x(idT,1,k,j,i);

          real K_L = (q2_L*q2_L + q3_L*q3_L + q4_L*q4_L) / (2*q1_L*q1_L);
          real p_L = (gamma-1)*(q5_L - q1_L*K_L);
          real K_R = (q2_R*q2_R + q3_R*q3_R + q4_R*q4_R) / (2*q1_R*q1_R);
          real p_R = (gamma-1)*(q5_R - q1_R*K_R);

          real r = 0.5_fp * (q1_L + q1_R);
          real u = 0.5_fp * (q2_L + q2_R)/r;
          real v = 0.5_fp * (q3_L + q3_R)/r;
          real w = 0.5_fp * (q4_L + q4_R)/r;
          real e = 0.5_fp * (q5_L + q5_R)/r;
          real K = (u*u+v*v+w*w)/2;
          real p = (gamma-1)*(r*e - r*K);
          real h = e+p/r;
          real cs = sqrt(gamma*p/r);
          real cs2 = cs*cs;

          if (riemann_choice == RIEMANN_NATIVE) {
            real f1_L = q2_L                          ;   real f1_R = q2_R                          ;
            real f2_L = q2_L*q2_L/q1_L + p_L          ;   real f2_R = q2_R*q2_R/q1_R + p_R          ;
            real f3_L = q2_L*q3_L/q1_L                ;   real f3_R = q2_R*q3_R/q1_R                ;
            real f4_L = q2_L*q4_L/q1_L                ;   real f4_R = q2_R*q4_R/q1_R                ;
            real f5_L = q2_L*q5_L/q1_L + q2_L*p_L/q1_L;   real f5_R = q2_R*q5_R/q1_R + q2_R*p_R/q1_R;

            real f1_U, f2_U, f3_U, f4_U, f5_U;
            real rden1 = 1._fp / (2*(K*cs-cs*h));
            real rden2 = 1._fp / (K-h);
            real rden3 = 1._fp / (2*(K-h));

            // Wave 1
            if (u-cs > 0) { f1_U=f1_L;  f2_U=f2_L;  f3_U=f3_L;  f4_U=f4_L;  f5_U=f5_L; }
            else          { f1_U=f1_R;  f2_U=f2_R;  f3_U=f3_R;  f4_U=f4_R;  f5_U=f5_R; }
            real w1 = -f1_U*(K*cs-(K-h)*u)*rden1 + f2_U*(cs*u-K+h)*rden1 + f3_U*v*rden3 + f4_U*w*rden3 - f5_U*rden3;
            // Wave 2
            if (u+cs > 0) { f1_U=f1_L;  f2_U=f2_L;  f3_U=f3_L;  f4_U=f4_L;  f5_U=f5_L; }
            else          { f1_U=f1_R;  f2_U=f2_R;  f3_U=f3_R;  f4_U=f4_R;  f5_U=f5_R; }
            real w2 = -f1_U*(K*cs+(K-h)*u)*rden1 + f2_U*(cs*u+K-h)*rden1 + f3_U*v*rden3 + f4_U*w*rden3 - f5_U*rden3;
            // Waves 3-5
            if (u    > 0) { f1_U=f1_L;  f2_U=f2_L;  f3_U=f3_L;  f4_U=f4_L;  f5_U=f5_L; }
            else          { f1_U=f1_R;  f2_U=f2_R;  f3_U=f3_R;  f4_U=f4_R;  f5_U=f5_R; }
            real w3 = f1_U*(2*K-h)*rden2             - f2_U*u*rden2   - f3_U*v*rden2             - f4_U*w*rden2         + f5_U*rden2;
            real w4 = f1_U*(u*u*v+v*v*v+v*w*w)*rden3 - f2_U*u*v*rden2 + f3_U*(u*u+w*w-K-h)*rden2 - f4_U*v*w*rden2       + f5_U*v*rden2;
            real w5 = f1_U*(K*w)*rden2               - f2_U*u*w*rden2 - f3_U*v*w*rden2           - f4_U*(w*w-K+h)*rden2 + f5_U*w*rden2;

            state_flux_x(idR,k,j,i) = w1          + w2          + w3;
            state_flux_x(idU,k,j,i) = w1*(u-cs)   + w2*(u+cs)   + w3*u;
            state_flux_x(idV,k,j,i) = w1*v        + w2*v                     + w4;
            state_flux_x(idW,k,j,i) = w1*w        + w2*w                            + w5;
            state_flux_x(idT,k,j,i) = w1*(h-u*cs) + w2*(h+u*cs) + w3*(u*u-K) + w4*v + w5*w;
          } else if (riemann_choice == RIEMANN_LLF) {
            real f1_L = q2_L                          ;   real f1_R = q2_R                          ;
            real f2_L = q2_L*q2_L/q1_L + p_L          ;   real f2_R = q2_R*q2_R/q1_R + p_R          ;
            real f3_L = q2_L*q3_L/q1_L                ;   real f3_R = q2_R*q3_R/q1_R                ;
            real f4_L = q2_L*q4_L/q1_L                ;   real f4_R = q2_R*q4_R/q1_R                ;
            real f5_L = q2_L*q5_L/q1_L + q2_L*p_L/q1_L;   real f5_R = q2_R*q5_R/q1_R + q2_R*p_R/q1_R;
            real maxwave = std::max( std::abs(q2_L/q1_L) + sqrt( gamma*p_L/q1_L ) , 
                                     std::abs(q2_R/q1_R) + sqrt( gamma*p_R/q1_R ) );
            state_flux_x(idR,k,j,i) = 0.5_fp * ( f1_L + f1_R - maxwave * ( q1_R - q1_L ) );
            state_flux_x(idU,k,j,i) = 0.5_fp * ( f2_L + f2_R - maxwave * ( q2_R - q2_L ) );
            state_flux_x(idV,k,j,i) = 0.5_fp * ( f3_L + f3_R - maxwave * ( q3_R - q3_L ) );
            state_flux_x(idW,k,j,i) = 0.5_fp * ( f4_L + f4_R - maxwave * ( q4_R - q4_L ) );
            state_flux_x(idT,k,j,i) = 0.5_fp * ( f5_L + f5_R - maxwave * ( q5_R - q5_L ) );
          } else if (riemann_choice == RIEMANN_PRESSURE) {
            real q1_U, q2_U, q3_U, q4_U, q5_U, q6_U;
            // Waves 1-4
            if (u    > 0) { q1_U=q1_L;  q2_U=q2_L;  q3_U=q3_L;  q4_U=q4_L;  q5_U=q5_L;  q6_U=p_L; }
            else          { q1_U=q1_R;  q2_U=q2_R;  q3_U=q3_R;  q4_U=q4_R;  q5_U=q5_R;  q6_U=p_R; }
            real w1 = q1_U - q6_U/cs2;
            real w2 = q3_U - q6_U*v/cs2;
            real w3 = q4_U - q6_U*w/cs2;
            real w4 = q1_U*u*u - q2_U*u + q5_U - q6_U*(cs2+e*gamma)/(cs2*gamma);
            // Wave 5
            if (u-cs > 0) { q1_U=q1_L;  q2_U=q2_L;  q3_U=q3_L;  q4_U=q4_L;  q5_U=q5_L;  q6_U=p_L; }
            else          { q1_U=q1_R;  q2_U=q2_R;  q3_U=q3_R;  q4_U=q4_R;  q5_U=q5_R;  q6_U=p_R; }
            real w5 =  q1_U*u/(2*cs) - q2_U/(2*cs) + q6_U/(2*cs2);
            // Wave 6
            if (u+cs > 0) { q1_U=q1_L;  q2_U=q2_L;  q3_U=q3_L;  q4_U=q4_L;  q5_U=q5_L;  q6_U=p_L; }
            else          { q1_U=q1_R;  q2_U=q2_R;  q3_U=q3_R;  q4_U=q4_R;  q5_U=q5_R;  q6_U=p_R; }
            real w6 = -q1_U*u/(2*cs) + q2_U/(2*cs) + q6_U/(2*cs2);
            q1_U = w1   + w5                                + w6;
            q2_U = w1*u + w5*(u-cs)                         + w6*(u+cs);
            q3_U = w2   + w5*v                              + w6*v;
            q4_U = w3   + w5*w                              + w6*w;
            q5_U = w4   - w5*(cs*gamma*u-cs2-e*gamma)/gamma + w6*(cs*gamma*u+cs2+e*gamma)/gamma;
            q6_U =        w5*cs2                            + w6*cs2;
            r = q1_U;
            u = q2_U/r;
            v = q3_U/r;
            w = q4_U/r;
            e = q5_U/r;
            p = q6_U;
            state_flux_x(idR,k,j,i) = r*u;
            state_flux_x(idU,k,j,i) = r*u*u + p;
            state_flux_x(idV,k,j,i) = r*u*v;
            state_flux_x(idW,k,j,i) = r*u*w;
            state_flux_x(idT,k,j,i) = r*u*e + u*p;
          }
        }

        ////////////////////////////////////////////////////////
        // Y-direction
        ////////////////////////////////////////////////////////
        // If we are simulating in 2-D, then do not do Riemann in the y-direction
        if ( (! sim2d) && i < nx && k < nz) {
          real q1_L = state_limits_y(idR,0,k,j,i);   real q1_R = state_limits_y(idR,1,k,j,i);
          real q2_L = state_limits_y(idU,0,k,j,i);   real q2_R = state_limits_y(idU,1,k,j,i);
          real q3_L = state_limits_y(idV,0,k,j,i);   real q3_R = state_limits_y(idV,1,k,j,i);
          real q4_L = state_limits_y(idW,0,k,j,i);   real q4_R = state_limits_y(idW,1,k,j,i);
          real q5_L = state_limits_y(idT,0,k,j,i);   real q5_R = state_limits_y(idT,1,k,j,i);

          real K_L = (q2_L*q2_L + q3_L*q3_L + q4_L*q4_L) / (2*q1_L*q1_L);
          real p_L = (gamma-1)*(q5_L - q1_L*K_L);
          real K_R = (q2_R*q2_R + q3_R*q3_R + q4_R*q4_R) / (2*q1_R*q1_R);
          real p_R = (gamma-1)*(q5_R - q1_R*K_R);

          real r = 0.5_fp * (q1_L + q1_R);
          real u = 0.5_fp * (q2_L + q2_R)/r;
          real v = 0.5_fp * (q3_L + q3_R)/r;
          real w = 0.5_fp * (q4_L + q4_R)/r;
          real e = 0.5_fp * (q5_L + q5_R)/r;
          real K = (u*u+v*v+w*w)/2;
          real p = (gamma-1)*(r*e - r*K);
          real h = e+p/r;
          real cs = sqrt(gamma*p/r);
          real cs2 = cs*cs;

          if (riemann_choice == RIEMANN_NATIVE) {
            real f1_L = q3_L                          ;   real f1_R = q3_R                          ;
            real f2_L = q3_L*q2_L/q1_L                ;   real f2_R = q3_R*q2_R/q1_R                ;
            real f3_L = q3_L*q3_L/q1_L + p_L          ;   real f3_R = q3_R*q3_R/q1_R + p_R          ;
            real f4_L = q3_L*q4_L/q1_L                ;   real f4_R = q3_R*q4_R/q1_R                ;
            real f5_L = q3_L*q5_L/q1_L + q3_L*p_L/q1_L;   real f5_R = q3_R*q5_R/q1_R + q3_R*p_R/q1_R;

            real f1_U, f2_U, f3_U, f4_U, f5_U;
            real rden1 = 1._fp / (2*(K*cs-cs*h));
            real rden2 = 1._fp / (K-h);
            real rden3 = 1._fp / (2*(K-h));

            // Wave 1
            if (v-cs > 0) { f1_U=f1_L;  f2_U=f2_L;  f3_U=f3_L;  f4_U=f4_L;  f5_U=f5_L; }
            else          { f1_U=f1_R;  f2_U=f2_R;  f3_U=f3_R;  f4_U=f4_R;  f5_U=f5_R; }
            real w1 = -f1_U*(K*cs-(K-h)*v)*rden1 + f2_U*u*rden3 + f3_U*(cs*v-K+h)*rden1 + f4_U*w*rden3 - f5_U*rden3;
            // Wave 2
            if (v+cs > 0) { f1_U=f1_L;  f2_U=f2_L;  f3_U=f3_L;  f4_U=f4_L;  f5_U=f5_L; }
            else          { f1_U=f1_R;  f2_U=f2_R;  f3_U=f3_R;  f4_U=f4_R;  f5_U=f5_R; }
            real w2 = -f1_U*(K*cs+(K-h)*v)*rden1 + f2_U*u*rden3 + f3_U*(cs*v+K-h)*rden1 + f4_U*w*rden3 - f5_U*rden3;
            // Waves 3-5
            if (v    > 0) { f1_U=f1_L;  f2_U=f2_L;  f3_U=f3_L;  f4_U=f4_L;  f5_U=f5_L; }
            else          { f1_U=f1_R;  f2_U=f2_R;  f3_U=f3_R;  f4_U=f4_R;  f5_U=f5_R; }
            real w3 = f1_U*(2*K-h)*rden2             - f2_U*u*rden2             - f3_U*v*rden2   - f4_U*w*rden2         + f5_U*rden2;
            real w4 = f1_U*(u*u*u+u*v*v+u*w*w)*rden3 + f2_U*(v*v+w*w-K-h)*rden2 - f3_U*u*v*rden2 - f4_U*u*w*rden2       + f5_U*u*rden2;
            real w5 = f1_U*K*w*rden2                 - f2_U*u*w*rden2           - f3_U*v*w*rden2 - f4_U*(w*w-K+h)*rden2 + f5_U*w*rden2;

            state_flux_y(idR,k,j,i) = w1          + w2          + w3;
            state_flux_y(idU,k,j,i) = w1*u        + w2*u                     + w4;
            state_flux_y(idV,k,j,i) = w1*(v-cs)   + w2*(v+cs)   + w3*v;
            state_flux_y(idW,k,j,i) = w1*w        + w2*w                            + w5;
            state_flux_y(idT,k,j,i) = w1*(h-v*cs) + w2*(h+v*cs) + w3*(v*v-K) + w4*u + w5*w;
          } else if (riemann_choice == RIEMANN_LLF) {
            real f1_L = q3_L                          ;   real f1_R = q3_R                          ;
            real f2_L = q3_L*q2_L/q1_L                ;   real f2_R = q3_R*q2_R/q1_R                ;
            real f3_L = q3_L*q3_L/q1_L + p_L          ;   real f3_R = q3_R*q3_R/q1_R + p_R          ;
            real f4_L = q3_L*q4_L/q1_L                ;   real f4_R = q3_R*q4_R/q1_R                ;
            real f5_L = q3_L*q5_L/q1_L + q3_L*p_L/q1_L;   real f5_R = q3_R*q5_R/q1_R + q3_R*p_R/q1_R;
            real maxwave = std::max( std::abs(q3_L/q1_L) + sqrt( gamma*p_L/q1_L ) , 
                                     std::abs(q3_R/q1_R) + sqrt( gamma*p_R/q1_R ) );
            state_flux_y(idR,k,j,i) = 0.5_fp * ( f1_L + f1_R - maxwave * ( q1_R - q1_L ) );
            state_flux_y(idU,k,j,i) = 0.5_fp * ( f2_L + f2_R - maxwave * ( q2_R - q2_L ) );
            state_flux_y(idV,k,j,i) = 0.5_fp * ( f3_L + f3_R - maxwave * ( q3_R - q3_L ) );
            state_flux_y(idW,k,j,i) = 0.5_fp * ( f4_L + f4_R - maxwave * ( q4_R - q4_L ) );
            state_flux_y(idT,k,j,i) = 0.5_fp * ( f5_L + f5_R - maxwave * ( q5_R - q5_L ) );
          } else if (riemann_choice == RIEMANN_PRESSURE) {
            real q1_U, q2_U, q3_U, q4_U, q5_U, q6_U;
            // Waves 1-4
            if (v    > 0) { q1_U=q1_L;  q2_U=q2_L;  q3_U=q3_L;  q4_U=q4_L;  q5_U=q5_L;  q6_U=p_L; }
            else          { q1_U=q1_R;  q2_U=q2_R;  q3_U=q3_R;  q4_U=q4_R;  q5_U=q5_R;  q6_U=p_R; }
            real w1 = q1_U - q6_U/cs2;
            real w2 = q2_U - q6_U*u/cs2;
            real w3 = q4_U - q6_U*w/cs2;
            real w4 = q1_U*v*v - q3_U*v + q5_U - q6_U*(cs2+e*gamma)/(cs2*gamma);
            // Wave 5
            if (v-cs > 0) { q1_U=q1_L;  q2_U=q2_L;  q3_U=q3_L;  q4_U=q4_L;  q5_U=q5_L;  q6_U=p_L; }
            else          { q1_U=q1_R;  q2_U=q2_R;  q3_U=q3_R;  q4_U=q4_R;  q5_U=q5_R;  q6_U=p_R; }
            real w5 =  q1_U*v/(2*cs) - q3_U/(2*cs) + q6_U/(2*cs2);
            // Wave 6
            if (v+cs > 0) { q1_U=q1_L;  q2_U=q2_L;  q3_U=q3_L;  q4_U=q4_L;  q5_U=q5_L;  q6_U=p_L; }
            else          { q1_U=q1_R;  q2_U=q2_R;  q3_U=q3_R;  q4_U=q4_R;  q5_U=q5_R;  q6_U=p_R; }
            real w6 = -q1_U*v/(2*cs) + q3_U/(2*cs) + q6_U/(2*cs2);
            q1_U = w1   + w5                                + w6;
            q2_U = w2   + w5*u                              + w6*u;
            q3_U = w1*v + w5*(v-cs)                         + w6*(v+cs);
            q4_U = w3   + w5*w                              + w6*w;
            q5_U = w4   - w5*(cs*gamma*v-cs2-e*gamma)/gamma + w6*(cs*gamma*v+cs2+e*gamma)/gamma;
            q6_U =        w5*cs2                            + w6*cs2;
            r = q1_U;
            u = q2_U/r;
            v = q3_U/r;
            w = q4_U/r;
            e = q5_U/r;
            p = q6_U;
            state_flux_y(idR,k,j,i) = r*v;
            state_flux_y(idU,k,j,i) = r*v*u;
            state_flux_y(idV,k,j,i) = r*v*v + p;
            state_flux_y(idW,k,j,i) = r*v*w;
            state_flux_y(idT,k,j,i) = r*v*e + v*p;
          }
        } else if (i < nx && k < nz) {
          state_flux_y(idR,k,j,i) = 0;
          state_flux_y(idU,k,j,i) = 0;
          state_flux_y(idV,k,j,i) = 0;
          state_flux_y(idW,k,j,i) = 0;
          state_flux_y(idT,k,j,i) = 0;
        }

        ////////////////////////////////////////////////////////
        // Z-direction
        ////////////////////////////////////////////////////////
        if (i < nx && j < ny) {
          // Boundary conditions
          if (k == 0) {
            for (int l = 0; l < num_state; l++) { state_limits_z(l,0,0 ,j,i) = state_limits_z(l,0,nz,j,i); }
          }
          if (k == nz) {
            for (int l = 0; l < num_state; l++) { state_limits_z(l,1,nz,j,i) = state_limits_z(l,1,0 ,j,i); }
          }
          real q1_L = state_limits_z(idR,0,k,j,i);   real q1_R = state_limits_z(idR,1,k,j,i);
          real q2_L = state_limits_z(idU,0,k,j,i);   real q2_R = state_limits_z(idU,1,k,j,i);
          real q3_L = state_limits_z(idV,0,k,j,i);   real q3_R = state_limits_z(idV,1,k,j,i);
          real q4_L = state_limits_z(idW,0,k,j,i);   real q4_R = state_limits_z(idW,1,k,j,i);
          real q5_L = state_limits_z(idT,0,k,j,i);   real q5_R = state_limits_z(idT,1,k,j,i);

          real K_L = (q2_L*q2_L + q3_L*q3_L + q4_L*q4_L) / (2*q1_L*q1_L);
          real p_L = (gamma-1)*(q5_L - q1_L*K_L);
          real K_R = (q2_R*q2_R + q3_R*q3_R + q4_R*q4_R) / (2*q1_R*q1_R);
          real p_R = (gamma-1)*(q5_R - q1_R*K_R);

          real r = 0.5_fp * (q1_L + q1_R);
          real u = 0.5_fp * (q2_L + q2_R)/r;
          real v = 0.5_fp * (q3_L + q3_R)/r;
          real w = 0.5_fp * (q4_L + q4_R)/r;
          real e = 0.5_fp * (q5_L + q5_R)/r;
          real K = (u*u+v*v+w*w)/2;
          real p = (gamma-1)*(r*e - r*K);
          real h = e+p/r;
          real cs = sqrt(gamma*p/r);
          real cs2 = cs*cs;

          if (riemann_choice == RIEMANN_NATIVE) {
            real f1_L = q4_L                          ;   real f1_R = q4_R                          ;
            real f2_L = q4_L*q2_L/q1_L                ;   real f2_R = q4_R*q2_R/q1_R                ;
            real f3_L = q4_L*q3_L/q1_L                ;   real f3_R = q4_R*q3_R/q1_R                ;
            real f4_L = q4_L*q4_L/q1_L + p_L          ;   real f4_R = q4_R*q4_R/q1_R + p_R          ;
            real f5_L = q4_L*q5_L/q1_L + q4_L*p_L/q1_L;   real f5_R = q4_R*q5_R/q1_R + q4_R*p_R/q1_R;

            real f1_U, f2_U, f3_U, f4_U, f5_U;
            real rden1 = 1._fp / (2*(K*cs-cs*h));
            real rden2 = 1._fp / (K-h);
            real rden3 = 1._fp / (2*(K-h));

            // Wave 1
            if (w-cs > 0) { f1_U=f1_L;  f2_U=f2_L;  f3_U=f3_L;  f4_U=f4_L;  f5_U=f5_L; }
            else          { f1_U=f1_R;  f2_U=f2_R;  f3_U=f3_R;  f4_U=f4_R;  f5_U=f5_R; }
            real w1 = -f1_U*(K*cs-(K-h)*w)*rden1 + f2_U*u*rden3 + f3_U*v*rden3 + f4_U*(cs*w-K+h)*rden1 - f5_U*rden3;
            // Wave 2
            if (w+cs > 0) { f1_U=f1_L;  f2_U=f2_L;  f3_U=f3_L;  f4_U=f4_L;  f5_U=f5_L; }
            else          { f1_U=f1_R;  f2_U=f2_R;  f3_U=f3_R;  f4_U=f4_R;  f5_U=f5_R; }
            real w2 = -f1_U*(K*cs+(K-h)*w)*rden1 + f2_U*u*rden3 + f3_U*v*rden3 + f4_U*(cs*w+K-h)*rden1 - f5_U*rden3;
            // Waves 3-5
            if (w    > 0) { f1_U=f1_L;  f2_U=f2_L;  f3_U=f3_L;  f4_U=f4_L;  f5_U=f5_L; }
            else          { f1_U=f1_R;  f2_U=f2_R;  f3_U=f3_R;  f4_U=f4_R;  f5_U=f5_R; }
            real w3 = f1_U*(2*K-h)*rden2             - f2_U*u*rden2             - f3_U*v*rden2         - f4_U*w*rden2   + f5_U*rden2;
            real w4 = f1_U*(u*u*u+u*v*v+u*w*w)*rden3 + f2_U*(v*v+w*w-K-h)*rden2 - f3_U*u*v*rden2       - f4_U*u*w*rden2 + f5_U*u*rden2;
            real w5 = f1_U*K*v*rden2                 - f2_U*u*v*rden2           - f3_U*(v*v-K+h)*rden2 - f4_U*v*w*rden2 + f5_U*v*rden2;

            state_flux_z(idR,k,j,i) = w1          + w2          + w3;
            state_flux_z(idU,k,j,i) = w1*u        + w2*u                     + w4;
            state_flux_z(idV,k,j,i) = w1*v        + w2*v                            + w4;
            state_flux_z(idW,k,j,i) = w1*(w-cs)   + w2*(w+cs)   + w3*w;
            state_flux_z(idT,k,j,i) = w1*(h-w*cs) + w2*(h+w*cs) + w3*(w*w-K) + w4*u + w5*v;
          } else if (riemann_choice == RIEMANN_LLF) {
            real f1_L = q4_L                          ;   real f1_R = q4_R                          ;
            real f2_L = q4_L*q2_L/q1_L                ;   real f2_R = q4_R*q2_R/q1_R                ;
            real f3_L = q4_L*q3_L/q1_L                ;   real f3_R = q4_R*q3_R/q1_R                ;
            real f4_L = q4_L*q4_L/q1_L + p_L          ;   real f4_R = q4_R*q4_R/q1_R + p_R          ;
            real f5_L = q4_L*q5_L/q1_L + q4_L*p_L/q1_L;   real f5_R = q4_R*q5_R/q1_R + q4_R*p_R/q1_R;
            real maxwave = std::max( std::abs(q4_L/q1_L) + sqrt( gamma*p_L/q1_L ) , 
                                     std::abs(q4_R/q1_R) + sqrt( gamma*p_R/q1_R ) );
            state_flux_z(idR,k,j,i) = 0.5_fp * ( f1_L + f1_R - maxwave * ( q1_R - q1_L ) );
            state_flux_z(idU,k,j,i) = 0.5_fp * ( f2_L + f2_R - maxwave * ( q2_R - q2_L ) );
            state_flux_z(idV,k,j,i) = 0.5_fp * ( f3_L + f3_R - maxwave * ( q3_R - q3_L ) );
            state_flux_z(idW,k,j,i) = 0.5_fp * ( f4_L + f4_R - maxwave * ( q4_R - q4_L ) );
            state_flux_z(idT,k,j,i) = 0.5_fp * ( f5_L + f5_R - maxwave * ( q5_R - q5_L ) );
          } else if (riemann_choice == RIEMANN_PRESSURE) {
            real q1_U, q2_U, q3_U, q4_U, q5_U, q6_U;
            // Waves 1-4
            if (w    > 0) { q1_U=q1_L;  q2_U=q2_L;  q3_U=q3_L;  q4_U=q4_L;  q5_U=q5_L;  q6_U=p_L; }
            else          { q1_U=q1_R;  q2_U=q2_R;  q3_U=q3_R;  q4_U=q4_R;  q5_U=q5_R;  q6_U=p_R; }
            real w1 = q1_U - q6_U/cs2;
            real w2 = q2_U - q6_U*u/cs2;
            real w3 = q3_U - q6_U*v/cs2;
            real w4 = q1_U*w*w - q4_U*w + q5_U - q6_U*(cs2+e*gamma)/(cs2*gamma);
            // Wave 5
            if (w-cs > 0) { q1_U=q1_L;  q2_U=q2_L;  q3_U=q3_L;  q4_U=q4_L;  q5_U=q5_L;  q6_U=p_L; }
            else          { q1_U=q1_R;  q2_U=q2_R;  q3_U=q3_R;  q4_U=q4_R;  q5_U=q5_R;  q6_U=p_R; }
            real w5 =  q1_U*w/(2*cs) - q4_U/(2*cs) + q6_U/(2*cs2);
            // Wave 6
            if (w+cs > 0) { q1_U=q1_L;  q2_U=q2_L;  q3_U=q3_L;  q4_U=q4_L;  q5_U=q5_L;  q6_U=p_L; }
            else          { q1_U=q1_R;  q2_U=q2_R;  q3_U=q3_R;  q4_U=q4_R;  q5_U=q5_R;  q6_U=p_R; }
            real w6 = -q1_U*w/(2*cs) + q4_U/(2*cs) + q6_U/(2*cs2);
            q1_U = w1   + w5                                + w6;
            q2_U = w2   + w5*u                              + w6*u;
            q3_U = w3   + w5*v                              + w6*v;
            q4_U = w1*w + w5*(w-cs)                         + w6*(w+cs);
            q5_U = w4   - w5*(cs*gamma*w-cs2-e*gamma)/gamma + w6*(cs*gamma*w+cs2+e*gamma)/gamma;
            q6_U =        w5*cs2                            + w6*cs2;
            r = q1_U;
            u = q2_U/r;
            v = q3_U/r;
            w = q4_U/r;
            e = q5_U/r;
            p = q6_U;
            state_flux_z(idR,k,j,i) = r*w;
            state_flux_z(idU,k,j,i) = r*w*u;
            state_flux_z(idV,k,j,i) = r*w*v;
            state_flux_z(idW,k,j,i) = r*w*w + p;
            state_flux_z(idT,k,j,i) = r*w*e + w*p;
          }
        }
      });

      // Deallocate state because they are no longer needed
      state_limits_x = real5d();
      state_limits_y = real5d();
      state_limits_z = real5d();

      // Compute tendencies as the flux divergence + gravity source term
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        for (int l = 0; l < num_state; l++) {
          state_tend(l,k,j,i) = -( state_flux_x(l,k  ,j  ,i+1) - state_flux_x(l,k,j,i) ) / dx
                                -( state_flux_y(l,k  ,j+1,i  ) - state_flux_y(l,k,j,i) ) / dy
                                -( state_flux_z(l,k+1,j  ,i  ) - state_flux_z(l,k,j,i) ) / dz;
          if (l == idV && sim2d) state_tend(l,k,j,i) = 0;
        }
      });
    }


    // Initialize the class data as well as the state arrays and convert them back into the coupler state
    void init(core::Coupler &coupler) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      // Set class data from # grid points, grid spacing, domain sizes, whether it's 2-D, and physical constants
      auto nx    = coupler.get_nx();
      auto ny    = coupler.get_ny();
      auto nz    = coupler.get_nz();

      auto dx    = coupler.get_dx();
      auto dy    = coupler.get_dy();
      auto dz    = coupler.get_dz();

      auto xlen  = coupler.get_xlen();
      auto ylen  = coupler.get_ylen();
      auto zlen  = coupler.get_zlen();

      bool sim2d = (coupler.get_ny_glob() == 1);

      coupler.set_option<real>("gamma",gamma);

      // Use TransformMatrices class to create matrices & GLL points to convert degrees of freedom as needed
      TransformMatrices::sten_to_coefs           (sten_to_coefs   );
      TransformMatrices::coefs_to_gll_lower      (coefs_to_gll    );
      TransformMatrices::weno_lower_sten_to_coefs(weno_recon_lower);
      weno::wenoSetIdealSigma<ord>( weno_idl , weno_sigma );

      auto input_fname = coupler.get_option<std::string>("standalone_input_file");
      YAML::Node config = YAML::LoadFile(input_fname);
      auto init_data = config["init_data"].as<std::string>();
      fname          = coupler.get_option<std::string>("out_fname");
      out_freq       = config["out_freq" ].as<real       >();

      auto &dm = coupler.get_data_manager_readwrite();

      // Set an integer version of the input_data so we can test it inside GPU kernels
      if      (init_data == "kelvin_helmholtz") { init_data_int = DATA_KELVIN_HELMHOLTZ; }
      else { endrun("ERROR: Invalid init_data in yaml input file"); }

      etime   = 0;
      num_out = 0;

      // Allocate temp arrays to hold state before we convert it back to the coupler state
      real4d state  ("state",num_state,nz+2*hs,ny+2*hs,nx+2*hs);

      if (init_data_int == DATA_KELVIN_HELMHOLTZ) {

        int constexpr nqpoints = 9;
        SArray<real,1,nqpoints> qpoints;
        SArray<real,1,nqpoints> qweights;
        TransformMatrices::get_gll_points (qpoints );
        TransformMatrices::get_gll_weights(qweights);

        auto i_beg = coupler.get_i_beg();
        auto j_beg = coupler.get_j_beg();

        // Use quadrature to initialize state data
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          for (int l=0; l < num_state; l++) { state(l,hs+k,hs+j,hs+i) = 0.; }
          //Use Gauss-Legendre quadrature
          for (int kk=0; kk<nqpoints; kk++) {
            for (int jj=0; jj<nqpoints; jj++) {
              for (int ii=0; ii<nqpoints; ii++) {
                real x = (i+i_beg+0.5)*dx + (qpoints(ii)-0.5)*dx;
                real y = (j+j_beg+0.5)*dy + (qpoints(jj)-0.5)*dy;   if (sim2d) y = ylen/2;
                real z = (k      +0.5)*dz + (qpoints(kk)-0.5)*dz;

                real alpha  = 0.25;
                real lambda = 0.01;
                int  n      = 2;
                int  L      = 1;
                
                real rho = std::abs(z-zlen/2) >= zlen/4 ? 1 : 2;
                real u   = std::abs(z-zlen/2) >= zlen/4 ? alpha : -alpha;
                real w   =             lambda * sin(2*M_PI*n*(x-xlen/2)/L);
                real v   = sim2d ? 0 : lambda * sin(2*M_PI*n*(y-ylen/2)/L);
                real p   = 2.5;

                real K = (u*u+v*v+w*w)/2;

                real wt = qweights(ii)*qweights(jj)*qweights(kk);
                state(idR,hs+k,hs+j,hs+i) += rho                     * wt;
                state(idU,hs+k,hs+j,hs+i) += rho*u                   * wt;
                state(idV,hs+k,hs+j,hs+i) += rho*v                   * wt;
                state(idW,hs+k,hs+j,hs+i) += rho*w                   * wt;
                state(idT,hs+k,hs+j,hs+i) += ( p/(gamma-1) + rho*K ) * wt;
              }
            }
          }
        });
      }

      // Convert the initialized state arrays back to the coupler state
      convert_dynamics_to_coupler( coupler , state );

      // Output the initial state
      if (out_freq >= 0. ) output( coupler , etime );

      int num_state = 5;
      dm.register_and_allocate<real>("state_flux_x","",{num_state,nz  ,ny  ,nx+1},{"num_state","z"  ,"y"  ,"xp1"});
      dm.register_and_allocate<real>("state_flux_y","",{num_state,nz  ,ny+1,nx  },{"num_state","z"  ,"yp1","x"  });
      dm.register_and_allocate<real>("state_flux_z","",{num_state,nz+1,ny  ,nx  },{"num_state","zp1","y"  ,"x"  });
      auto state_flux_x = dm.get<real,4>("state_flux_x");
      auto state_flux_y = dm.get<real,4>("state_flux_y");
      auto state_flux_z = dm.get<real,4>("state_flux_z");
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz+1,ny+1,nx+1) , YAKL_LAMBDA (int k, int j, int i) {
        for (int l=0; l < num_state; l++) {
          if (j < ny && k < nz) state_flux_x(l,k,j,i) = 0;
          if (i < nx && k < nz) state_flux_y(l,k,j,i) = 0;
          if (i < nx && j < ny) state_flux_z(l,k,j,i) = 0;
        }
      });
    }


    // Convert dynamics state arrays to the coupler state and write to the coupler's data
    void convert_dynamics_to_coupler( core::Coupler &coupler , realConst4d state ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      auto &dm = coupler.get_data_manager_readwrite();
      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();

      if (! dm.entry_exists("density")) {
        dm.register_and_allocate<real>("density","rho"          ,{nz,ny,nx},{"z","y","x"});
        dm.register_and_allocate<real>("umom"   ,"rho*u"        ,{nz,ny,nx},{"z","y","x"});
        dm.register_and_allocate<real>("vmom"   ,"rho*v"        ,{nz,ny,nx},{"z","y","x"});
        dm.register_and_allocate<real>("wmom"   ,"rho*w"        ,{nz,ny,nx},{"z","y","x"});
        dm.register_and_allocate<real>("energy" ,"rho*(c_v*T+K)",{nz,ny,nx},{"z","y","x"});
      }

      auto dm_rho  = dm.get<real,3>("density");
      auto dm_rhou = dm.get<real,3>("umom");
      auto dm_rhov = dm.get<real,3>("vmom");
      auto dm_rhow = dm.get<real,3>("wmom");
      auto dm_ener = dm.get<real,3>("energy");

      // Convert from state arrays to the coupler's data
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        dm_rho (k,j,i) = state(idR,hs+k,hs+j,hs+i);
        dm_rhou(k,j,i) = state(idU,hs+k,hs+j,hs+i);
        dm_rhov(k,j,i) = state(idV,hs+k,hs+j,hs+i);
        dm_rhow(k,j,i) = state(idW,hs+k,hs+j,hs+i);
        dm_ener(k,j,i) = state(idT,hs+k,hs+j,hs+i);
      });
    }


    // Convert coupler's data to state arrays
    void convert_coupler_to_dynamics( core::Coupler const &coupler , real4d &state ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      auto &dm = coupler.get_data_manager_readonly();
      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();

      auto dm_rho  = dm.get<real const,3>("density");
      auto dm_rhou = dm.get<real const,3>("umom");
      auto dm_rhov = dm.get<real const,3>("vmom");
      auto dm_rhow = dm.get<real const,3>("wmom");
      auto dm_ener = dm.get<real const,3>("energy");

      // Convert from state arrays to the coupler's data
      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        state(idR,hs+k,hs+j,hs+i) = dm_rho (k,j,i);
        state(idU,hs+k,hs+j,hs+i) = dm_rhou(k,j,i);
        state(idV,hs+k,hs+j,hs+i) = dm_rhov(k,j,i);
        state(idW,hs+k,hs+j,hs+i) = dm_rhow(k,j,i);
        state(idT,hs+k,hs+j,hs+i) = dm_ener(k,j,i);
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
        nc.create_var<real>( "density"  , {"t","z","y","x"} );
        nc.create_var<real>( "uvel"     , {"t","z","y","x"} );
        nc.create_var<real>( "vvel"     , {"t","z","y","x"} );
        nc.create_var<real>( "wvel"     , {"t","z","y","x"} );
        nc.create_var<real>( "energy"   , {"t","z","y","x"} );
        nc.create_var<real>( "pressure" , {"t","z","y","x"} );

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
      auto rho  = dm.get<real const,3>("density");
      auto rhou = dm.get<real const,3>("umom");
      auto rhov = dm.get<real const,3>("vmom");
      auto rhow = dm.get<real const,3>("wmom");
      auto ener = dm.get<real const,3>("energy");
      using yakl::componentwise::operator/;
      using yakl::componentwise::operator*;
      using yakl::componentwise::operator+;
      using yakl::componentwise::operator-;
      auto uvel = rhou/rho;
      auto vvel = rhov/rho;
      auto wvel = rhow/rho;
      auto K = (uvel*uvel + vvel*vvel + wvel*wvel)/2;
      auto pressure = (gamma-1)*(ener - rho*K);
      nc.write1_all(rho     .createHostCopy(),"density" ,ulIndex,{0,j_beg,i_beg},"t");
      nc.write1_all(uvel    .createHostCopy(),"uvel"    ,ulIndex,{0,j_beg,i_beg},"t");
      nc.write1_all(vvel    .createHostCopy(),"vvel"    ,ulIndex,{0,j_beg,i_beg},"t");
      nc.write1_all(wvel    .createHostCopy(),"wvel"    ,ulIndex,{0,j_beg,i_beg},"t");
      nc.write1_all(ener    .createHostCopy(),"energy"  ,ulIndex,{0,j_beg,i_beg},"t");
      nc.write1_all(pressure.createHostCopy(),"pressure",ulIndex,{0,j_beg,i_beg},"t");

      nc.close();
      yakl::timer_stop("output");
    }


    void halo_exchange(core::Coupler const &coupler , real4d const &state) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      bool sim2d = ny == 1;

      int npack = num_state;

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
        halo_send_buf_W(v,k,j,ii) = state(v,hs+k,hs+j,hs+ii);
        halo_send_buf_E(v,k,j,ii) = state(v,hs+k,hs+j,nx+ii);
      });

      real4d halo_send_buf_S("halo_send_buf_S",npack,nz,hs,nx);
      real4d halo_send_buf_N("halo_send_buf_N",npack,nz,hs,nx);

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(npack,nz,hs,nx) , YAKL_LAMBDA (int v, int k, int jj, int i) {
          halo_send_buf_S(v,k,jj,i) = state(v,hs+k,hs+jj,hs+i);
          halo_send_buf_N(v,k,jj,i) = state(v,hs+k,ny+jj,hs+i);
        });
      }

      yakl::fence();
      yakl::timer_start("halo_exchange_mpi");

      real4d halo_recv_buf_W("halo_recv_buf_W",npack,nz,ny,hs);
      real4d halo_recv_buf_E("halo_recv_buf_E",npack,nz,ny,hs);
      real4d halo_recv_buf_S("halo_recv_buf_S",npack,nz,hs,nx);
      real4d halo_recv_buf_N("halo_recv_buf_N",npack,nz,hs,nx);

      auto data_type = coupler.get_mpi_data_type();

      MPI_Request sReq[4];
      MPI_Request rReq[4];

      auto &neigh = coupler.get_neighbor_rankid_matrix();

      //Pre-post the receives
      MPI_Irecv( halo_recv_buf_W_host.data() , npack*nz*ny*hs , data_type , neigh(1,0) , 0 , MPI_COMM_WORLD , &rReq[0] );
      MPI_Irecv( halo_recv_buf_E_host.data() , npack*nz*ny*hs , data_type , neigh(1,2) , 1 , MPI_COMM_WORLD , &rReq[1] );
      if (!sim2d) {
        MPI_Irecv( halo_recv_buf_S_host.data() , npack*nz*hs*nx , data_type , neigh(0,1) , 2 , MPI_COMM_WORLD , &rReq[2] );
        MPI_Irecv( halo_recv_buf_N_host.data() , npack*nz*hs*nx , data_type , neigh(2,1) , 3 , MPI_COMM_WORLD , &rReq[3] );
      }

      halo_send_buf_W.deep_copy_to(halo_send_buf_W_host);
      halo_send_buf_E.deep_copy_to(halo_send_buf_E_host);
      if (!sim2d) {
        halo_send_buf_S.deep_copy_to(halo_send_buf_S_host);
        halo_send_buf_N.deep_copy_to(halo_send_buf_N_host);
      }

      yakl::fence();

      //Send the data
      MPI_Isend( halo_send_buf_W_host.data() , npack*nz*ny*hs , data_type , neigh(1,0) , 1 , MPI_COMM_WORLD , &sReq[0] );
      MPI_Isend( halo_send_buf_E_host.data() , npack*nz*ny*hs , data_type , neigh(1,2) , 0 , MPI_COMM_WORLD , &sReq[1] );
      if (!sim2d) {
        MPI_Isend( halo_send_buf_S_host.data() , npack*nz*hs*nx , data_type , neigh(0,1) , 3 , MPI_COMM_WORLD , &sReq[2] );
        MPI_Isend( halo_send_buf_N_host.data() , npack*nz*hs*nx , data_type , neigh(2,1) , 2 , MPI_COMM_WORLD , &sReq[3] );
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
        state(v,hs+k,hs+j,      ii) = halo_recv_buf_W(v,k,j,ii);
        state(v,hs+k,hs+j,nx+hs+ii) = halo_recv_buf_E(v,k,j,ii);
      });

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(npack,nz,hs,nx) , YAKL_LAMBDA (int v, int k, int jj, int i) {
          state(v,hs+k,      jj,hs+i) = halo_recv_buf_S(v,k,jj,i);
          state(v,hs+k,ny+hs+jj,hs+i) = halo_recv_buf_N(v,k,jj,i);
        });
      }
    }


    void edge_exchange(core::Coupler const &coupler , real5d const &state_limits_x ,
                                                      real5d const &state_limits_y ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      bool sim2d = ny == 1;

      int npack = num_state;

      realHost3d edge_send_buf_S_host("edge_send_buf_S_host",npack,nz,nx);
      realHost3d edge_send_buf_N_host("edge_send_buf_N_host",npack,nz,nx);
      realHost3d edge_send_buf_W_host("edge_send_buf_W_host",npack,nz,ny);
      realHost3d edge_send_buf_E_host("edge_send_buf_E_host",npack,nz,ny);
      realHost3d edge_recv_buf_S_host("edge_recv_buf_S_host",npack,nz,nx);
      realHost3d edge_recv_buf_N_host("edge_recv_buf_N_host",npack,nz,nx);
      realHost3d edge_recv_buf_W_host("edge_recv_buf_W_host",npack,nz,ny);
      realHost3d edge_recv_buf_E_host("edge_recv_buf_E_host",npack,nz,ny);

      real3d edge_send_buf_W("edge_send_buf_W",npack,nz,ny);
      real3d edge_send_buf_E("edge_send_buf_E",npack,nz,ny);

      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(npack,nz,ny) , YAKL_LAMBDA (int v, int k, int j) {
        edge_send_buf_W(v,k,j) = state_limits_x(v,1,k,j,0 );
        edge_send_buf_E(v,k,j) = state_limits_x(v,0,k,j,nx);
      });

      real3d edge_send_buf_S("edge_send_buf_S",npack,nz,nx);
      real3d edge_send_buf_N("edge_send_buf_N",npack,nz,nx);

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(npack,nz,nx) , YAKL_LAMBDA (int v, int k, int i) {
          edge_send_buf_S(v,k,i) = state_limits_y(v,1,k,0 ,i);
          edge_send_buf_N(v,k,i) = state_limits_y(v,0,k,ny,i);
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

      auto data_type = coupler.get_mpi_data_type();

      //Pre-post the receives
      MPI_Irecv( edge_recv_buf_W_host.data() , npack*nz*ny , data_type , neigh(1,0) , 4 , MPI_COMM_WORLD , &rReq[0] );
      MPI_Irecv( edge_recv_buf_E_host.data() , npack*nz*ny , data_type , neigh(1,2) , 5 , MPI_COMM_WORLD , &rReq[1] );
      if (!sim2d) {
        MPI_Irecv( edge_recv_buf_S_host.data() , npack*nz*nx , data_type , neigh(0,1) , 6 , MPI_COMM_WORLD , &rReq[2] );
        MPI_Irecv( edge_recv_buf_N_host.data() , npack*nz*nx , data_type , neigh(2,1) , 7 , MPI_COMM_WORLD , &rReq[3] );
      }

      edge_send_buf_W.deep_copy_to(edge_send_buf_W_host);
      edge_send_buf_E.deep_copy_to(edge_send_buf_E_host);
      if (!sim2d) {
        edge_send_buf_S.deep_copy_to(edge_send_buf_S_host);
        edge_send_buf_N.deep_copy_to(edge_send_buf_N_host);
      }

      yakl::fence();

      //Send the data
      MPI_Isend( edge_send_buf_W_host.data() , npack*nz*ny , data_type , neigh(1,0) , 5 , MPI_COMM_WORLD , &sReq[0] );
      MPI_Isend( edge_send_buf_E_host.data() , npack*nz*ny , data_type , neigh(1,2) , 4 , MPI_COMM_WORLD , &sReq[1] );
      if (!sim2d) {
        MPI_Isend( edge_send_buf_S_host.data() , npack*nz*nx , data_type , neigh(0,1) , 7 , MPI_COMM_WORLD , &sReq[2] );
        MPI_Isend( edge_send_buf_N_host.data() , npack*nz*nx , data_type , neigh(2,1) , 6 , MPI_COMM_WORLD , &sReq[3] );
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
        state_limits_x(v,0,k,j,0 ) = edge_recv_buf_W(v,k,j);
        state_limits_x(v,1,k,j,nx) = edge_recv_buf_E(v,k,j);
      });

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(npack,nz,nx) , YAKL_LAMBDA (int v, int k, int i) {
          state_limits_y(v,0,k,0 ,i) = edge_recv_buf_S(v,k,i);
          state_limits_y(v,1,k,ny,i) = edge_recv_buf_N(v,k,i);
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


