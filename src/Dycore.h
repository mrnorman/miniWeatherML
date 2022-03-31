
#pragma once

#include "main_header.h"
#include "MultipleFields.h"

class Dycore {
  public:

  int  static constexpr num_state = 5;
  int  static constexpr hs        = 2;
  int  static constexpr sten_size = 4;
  real static constexpr hv_beta   = 0.03;

  int  static constexpr idR = 0;
  int  static constexpr idU = 1;
  int  static constexpr idV = 2;
  int  static constexpr idW = 3;
  int  static constexpr idT = 4;

  int  static constexpr DATA_THERMAL   = 0;
  int  static constexpr DATA_SUPERCELL = 1;

  real1d      hy_dens_cells;
  real1d      hy_dens_theta_cells;
  real1d      hy_dens_edges;
  real1d      hy_dens_theta_edges;
  real        etime;
  real        out_freq;
  int         num_out;
  std::string fname;
  int         init_data_int;

  int         nx  , ny  , nz  ;
  real        dx  , dy  , dz  ;
  real        xlen, ylen, zlen;
  bool        sim2d;

  real        R_d, R_v, cp_d, cp_v, p0, grav, kappa, gamma, C0;

  int         num_tracers;
  int         idWV;
  bool1d      tracer_adds_mass;
  bool1d      tracer_positive;

  real compute_time_step( core::Coupler const &coupler ) {
    real constexpr maxwave = 350 + 80;
    real cfl = 0.3;
    if (coupler.get_ny() == 1) cfl = 0.5;
    return cfl * std::min( std::min( dx , dy ) , dz ) / maxwave;
  }


  void time_step(core::Coupler &coupler, real &dt_phys) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    real4d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
    real4d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);

    convert_coupler_to_dynamics( coupler , state , tracers );

    real dt_dyn = compute_time_step( coupler );

    int ncycles = (int) std::ceil( dt_phys / dt_dyn );
    dt_dyn = dt_phys / ncycles;
    
    for (int icycle = 0; icycle < ncycles; icycle++) {
      real4d state_tmp   ("state_tmp"   ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
      real4d state_tend  ("state_tend"  ,num_state  ,nz     ,ny     ,nx     );
      real4d tracers_tmp ("tracers_tmp" ,num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);
      real4d tracers_tend("tracers_tend",num_tracers,nz     ,ny     ,nx     );
      // Stage 1
      compute_tendencies( coupler , state     , state_tend , tracers     , tracers_tend , dt_dyn );
      parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
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
      // Stage 2
      compute_tendencies( coupler , state_tmp , state_tend , tracers_tmp , tracers_tend , (1._fp/4._fp) * dt_dyn );
      parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
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
      // Stage 3
      compute_tendencies( coupler , state_tmp , state_tend , tracers_tmp , tracers_tend , (2._fp/3._fp) * dt_dyn );
      parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
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

    convert_dynamics_to_coupler( coupler , state , tracers );

    etime += dt_phys;
    if (out_freq >= 0. && etime / out_freq >= num_out+1) {
      yakl::timer_start("output");
      output( coupler , etime );
      yakl::timer_stop("output");
      num_out++;
    }
  }


  void compute_tendencies( core::Coupler const &coupler , real4d const &state   , real4d const &state_tend   ,
                                                          real4d const &tracers , real4d const &tracers_tend , real dt ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    real4d state_flux_x  ("state_flux_x"  ,num_state  ,nz  ,ny  ,nx+1);
    real4d state_flux_y  ("state_flux_y"  ,num_state  ,nz  ,ny+1,nx  );
    real4d state_flux_z  ("state_flux_z"  ,num_state  ,nz+1,ny  ,nx  );

    real4d tracers_flux_x("tracers_flux_x",num_tracers,nz  ,ny  ,nx+1);
    real4d tracers_flux_y("tracers_flux_y",num_tracers,nz  ,ny+1,nx  );
    real4d tracers_flux_z("tracers_flux_z",num_tracers,nz+1,ny  ,nx  );

    YAKL_SCOPE( hy_dens_cells       , this->hy_dens_cells       );
    YAKL_SCOPE( hy_dens_theta_cells , this->hy_dens_theta_cells );
    YAKL_SCOPE( hy_dens_edges       , this->hy_dens_edges       );
    YAKL_SCOPE( hy_dens_theta_edges , this->hy_dens_theta_edges );

    // Compute fluxes in the x, y, and z directions for state and tracers
    parallel_for( Bounds<3>(nz+1,ny+1,nx+1) , YAKL_LAMBDA (int k, int j, int i ) {
      ////////////////////////////////////////////////////////
      // X-direction
      ////////////////////////////////////////////////////////
      if (j < ny && k < nz) {
        SArray<real,1,sten_size> stencil;
        SArray<real,1,num_state> d3_vals;
        SArray<real,1,num_state> vals;
        //Compute the hyperviscosity coeficient
        real hv_coef = -hv_beta * dx / (16*dt);

        //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
        for (int l=0; l < num_state; l++) {
          for (int s=0; s < sten_size; s++) {
            int ind = i+s;   if (ind < hs) ind += nx;   if (ind >= nx+hs) ind -= nx;
            stencil(s) = state(l,hs+k,hs+j,ind);
          }
          //Fourth-order-accurate interpolation of the state
          vals   (l) = -stencil(0)/12 + 7*stencil(1)/12 + 7*stencil(2)/12 - stencil(3)/12;
          //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
          d3_vals(l) = -stencil(0)    + 3*stencil(1)    - 3*stencil(2)    + stencil(3);
        }

        //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
        real r = vals(idR) + hy_dens_cells(k);
        real u = vals(idU) / r;
        real v = vals(idV) / r;
        real w = vals(idW) / r;
        real t = ( vals(idT) + hy_dens_theta_cells(k) ) / r;
        real p = C0*pow( r*t  , gamma );

        //Compute the flux vector
        state_flux_x(idR,k,j,i) = r*u     - hv_coef*d3_vals(idR);
        state_flux_x(idU,k,j,i) = r*u*u+p - hv_coef*d3_vals(idU);
        state_flux_x(idV,k,j,i) = r*u*v   - hv_coef*d3_vals(idV);
        state_flux_x(idW,k,j,i) = r*u*w   - hv_coef*d3_vals(idW);
        state_flux_x(idT,k,j,i) = r*u*t   - hv_coef*d3_vals(idT);

        for (int l=0; l < num_tracers; l++) {
          for (int s=0; s < sten_size; s++) {
            int ind = i+s;   if (ind < hs) ind += nx;   if (ind >= nx+hs) ind -= nx;
            stencil(s) = tracers(l,hs+k,hs+j,ind);
          }
          //Fourth-order-accurate interpolation of the state
          real val    = -stencil(0)/12 + 7*stencil(1)/12 + 7*stencil(2)/12 - stencil(3)/12;
          //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
          real d3_val = -stencil(0)    + 3*stencil(1)    - 3*stencil(2)    + stencil(3);
          if (tracer_positive(l)) val = std::max( 0._fp , val );
          tracers_flux_x(l,k,j,i) = u*val - hv_coef*d3_val;
        }
      }

      ////////////////////////////////////////////////////////
      // Y-direction
      ////////////////////////////////////////////////////////
      if ( (! sim2d) && i < nx && k < nz) {
        SArray<real,1,sten_size> stencil;
        SArray<real,1,num_state> d3_vals;
        SArray<real,1,num_state> vals;
        //Compute the hyperviscosity coeficient
        real hv_coef = -hv_beta * dy / (16*dt);

        //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
        for (int l=0; l < num_state; l++) {
          for (int s=0; s < sten_size; s++) {
            int ind = j+s;   if (ind < hs) ind += ny;   if (ind >= ny+hs) ind -= ny;
            stencil(s) = state(l,hs+k,ind,hs+i);
          }
          //Fourth-order-accurate interpolation of the state
          vals   (l) = -stencil(0)/12 + 7*stencil(1)/12 + 7*stencil(2)/12 - stencil(3)/12;
          //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
          d3_vals(l) = -stencil(0)    + 3*stencil(1)    - 3*stencil(2)    + stencil(3);
        }

        //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
        real r = vals(idR) + hy_dens_cells(k);
        real u = vals(idU) / r;
        real v = vals(idV) / r;
        real w = vals(idW) / r;
        real t = ( vals(idT) + hy_dens_theta_cells(k) ) / r;
        real p = C0*pow( r*t  , gamma );

        //Compute the flux vector
        state_flux_y(idR,k,j,i) = r*v     - hv_coef*d3_vals(idR);
        state_flux_y(idU,k,j,i) = r*v*u   - hv_coef*d3_vals(idU);
        state_flux_y(idV,k,j,i) = r*v*v+p - hv_coef*d3_vals(idV);
        state_flux_y(idW,k,j,i) = r*v*w   - hv_coef*d3_vals(idW);
        state_flux_y(idT,k,j,i) = r*v*t   - hv_coef*d3_vals(idT);

        for (int l=0; l < num_tracers; l++) {
          for (int s=0; s < sten_size; s++) {
            int ind = j+s;   if (ind < hs) ind += ny;   if (ind >= ny+hs) ind -= ny;
            stencil(s) = tracers(l,hs+k,ind,hs+i);
          }
          //Fourth-order-accurate interpolation of the state
          real val    = -stencil(0)/12 + 7*stencil(1)/12 + 7*stencil(2)/12 - stencil(3)/12;
          //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
          real d3_val = -stencil(0)    + 3*stencil(1)    - 3*stencil(2)    + stencil(3);
          if (tracer_positive(l)) val = std::max( 0._fp , val );
          tracers_flux_y(l,k,j,i) = v*val - hv_coef*d3_val;
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
        SArray<real,1,sten_size> stencil;
        SArray<real,1,num_state> d3_vals;
        SArray<real,1,num_state> vals;
        //Compute the hyperviscosity coeficient
        real hv_coef = -hv_beta * dz / (16*dt);

        //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
        for (int l=0; l < num_state; l++) {
          for (int s=0; s < sten_size; s++) {
            int ind = std::min( nz+hs-1 , std::max( (int) hs , k+s ) );
            stencil(s) = state(l,ind,hs+j,hs+i);
          }
          //Fourth-order-accurate interpolation of the state
          vals   (l) = -stencil(0)/12 + 7*stencil(1)/12 + 7*stencil(2)/12 - stencil(3)/12;
          //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
          d3_vals(l) = -stencil(0)    + 3*stencil(1)    - 3*stencil(2)    + stencil(3);
        }
        if (k == 0 || k == nz) { vals(idW) = 0;   d3_vals(idW) = 0; }

        //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
        real r = vals(idR) + hy_dens_edges(k);
        real u = vals(idU) / r;
        real v = vals(idV) / r;
        real w = vals(idW) / r;
        real t = ( vals(idT) + hy_dens_theta_edges(k) ) / r;
        real p = C0*pow( r*t  , gamma );

        //Compute the flux vector
        state_flux_z(idR,k,j,i) = r*w     - hv_coef*d3_vals(idR);
        state_flux_z(idU,k,j,i) = r*w*u   - hv_coef*d3_vals(idU);
        state_flux_z(idV,k,j,i) = r*w*v   - hv_coef*d3_vals(idV);
        state_flux_z(idW,k,j,i) = r*w*w+p - hv_coef*d3_vals(idW);
        state_flux_z(idT,k,j,i) = r*w*t   - hv_coef*d3_vals(idT);

        for (int l=0; l < num_tracers; l++) {
          for (int s=0; s < sten_size; s++) {
            int ind = std::min( nz+hs-1 , std::max( (int) hs , k+s ) );
            stencil(s) = tracers(l,ind,hs+j,hs+i);
          }
          //Fourth-order-accurate interpolation of the state
          real val    = -stencil(0)/12 + 7*stencil(1)/12 + 7*stencil(2)/12 - stencil(3)/12;
          //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
          real d3_val = -stencil(0)    + 3*stencil(1)    - 3*stencil(2)    + stencil(3);
          if (tracer_positive(l)) val = std::max( 0._fp , val );
          tracers_flux_z(l,k,j,i) = w*val - hv_coef*d3_val;
        }
      }
    });

    // Flux Corrected Transport to enforce positivity for tracer species that must remain non-negative
    // This looks like it has a race condition, but it does not. Only one of the adjacent cells can ever change
    // a given edge flux because it's only changed if its sign oriented outward from a cell.
    parallel_for( Bounds<4>(num_tracers,nz,ny,nx) , YAKL_LAMBDA (int tr, int k, int j, int i ) {
      if (tracer_positive(tr)) {
        real mass_available = std::max(tracers(tr,hs+k,hs+j,hs+i),0._fp) * dx * dy * dz;
        real flux_out_x = ( std::max(tracers_flux_x(tr,k,j,i+1),0._fp) - std::min(tracers_flux_x(tr,k,j,i),0._fp) ) / dx;
        real flux_out_y = ( std::max(tracers_flux_y(tr,k,j+1,i),0._fp) - std::min(tracers_flux_y(tr,k,j,i),0._fp) ) / dy;
        real flux_out_z = ( std::max(tracers_flux_z(tr,k+1,j,i),0._fp) - std::min(tracers_flux_z(tr,k,j,i),0._fp) ) / dz;
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
    parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
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
    });
  }


  void init(core::Coupler &coupler) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    nx    = coupler.get_nx();
    ny    = coupler.get_ny();
    nz    = coupler.get_nz();

    dx    = coupler.get_dx();
    dy    = coupler.get_dy();
    dz    = coupler.get_dz();

    xlen  = coupler.get_xlen();
    ylen  = coupler.get_ylen();
    zlen  = coupler.get_zlen();

    sim2d = (ny == 1);

    R_d   = coupler.R_d ;
    R_v   = coupler.R_v ;
    cp_d  = coupler.cp_d;
    cp_v  = coupler.cp_v;
    p0    = coupler.p0  ;
    grav  = coupler.grav;
    kappa = R_d / cp_d;
    gamma = cp_d / (cp_d - R_d);
    C0    = pow( R_d * pow( p0 , -kappa ) , gamma );

    num_tracers = coupler.get_num_tracers();
    tracer_adds_mass = bool1d("tracer_adds_mass",num_tracers);
    tracer_positive  = bool1d("tracer_positive" ,num_tracers);

    auto tracer_adds_mass_host = tracer_adds_mass.createHostCopy();
    auto tracer_positive_host  = tracer_positive .createHostCopy();

    auto tracer_names = coupler.get_tracer_names();
    for (int tr=0; tr < num_tracers; tr++) {
      std::string tracer_desc;
      bool        tracer_found, positive, adds_mass;
      coupler.get_tracer_info( tracer_names[tr] , tracer_desc, tracer_found , positive , adds_mass);
      tracer_positive_host (tr) = positive;
      tracer_adds_mass_host(tr) = adds_mass;
      if (tracer_names[tr] == "water_vapor") idWV = tr;
    }
    tracer_positive_host .deep_copy_to(tracer_positive );
    tracer_adds_mass_host.deep_copy_to(tracer_adds_mass);

    auto inFile = coupler.get_option<std::string>( "standalone_input_file" );
    YAML::Node config = YAML::LoadFile(inFile);
    auto init_data = config["init_data"].as<std::string>();
    fname          = config["out_fname"].as<std::string>();
    out_freq       = config["out_freq" ].as<real>();

    if      (init_data == "thermal"  ) { init_data_int = DATA_THERMAL;   }
    else if (init_data == "supercell") { init_data_int = DATA_SUPERCELL; }
    else { endrun("ERROR: Invalid init_data in yaml input file"); }

    auto rho_v = coupler.dm.get<real,3>("water_vapor");

    etime   = 0;
    num_out = 0;

    // Define quadrature weights and points
    const int nqpoints = 3;
    SArray<real,1,nqpoints> qpoints;
    SArray<real,1,nqpoints> qweights;

    qpoints(0) = 0.112701665379258311482073460022;
    qpoints(1) = 0.500000000000000000000000000000;
    qpoints(2) = 0.887298334620741688517926539980;

    qweights(0) = 0.277777777777777777777777777779;
    qweights(1) = 0.444444444444444444444444444444;
    qweights(2) = 0.277777777777777777777777777779;

    // Compute the state that the dycore uses
    real4d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
    real4d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);

    hy_dens_cells       = real1d("hy_dens_cells"      ,nz  );
    hy_dens_theta_cells = real1d("hy_dens_theta_cells",nz  );
    hy_dens_edges       = real1d("hy_dens_edges"      ,nz+1);
    hy_dens_theta_edges = real1d("hy_dens_theta_edges",nz+1);

    if (init_data_int == DATA_SUPERCELL) {

      init_supercell( coupler , state , tracers );

    } else {

      YAKL_SCOPE( init_data_int       , this->init_data_int       );
      YAKL_SCOPE( hy_dens_cells       , this->hy_dens_cells       );
      YAKL_SCOPE( hy_dens_theta_cells , this->hy_dens_theta_cells );
      YAKL_SCOPE( hy_dens_edges       , this->hy_dens_edges       );
      YAKL_SCOPE( hy_dens_theta_edges , this->hy_dens_theta_edges );

      parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
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
              real x = (i+0.5)*dx + (qpoints(ii)-0.5)*dx;
              real y = (j+0.5)*dy + (qpoints(jj)-0.5)*dy;   if (sim2d) y = ylen/2;
              real z = (k+0.5)*dz + (qpoints(kk)-0.5)*dz;
              real rho, u, v, w, theta, rho_v, hr, ht;

              if (init_data_int == DATA_THERMAL) {
                thermal(x,y,z,xlen,ylen,grav,C0,gamma,cp_d,p0,R_d,R_v,rho,u,v,w,theta,rho_v,hr,ht);
              }

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


      parallel_for( Bounds<1>(nz) , YAKL_LAMBDA (int k) {
        hy_dens_cells      (k) = 0.;
        hy_dens_theta_cells(k) = 0.;
        for (int kk=0; kk<nqpoints; kk++) {
          real z = (k+0.5)*dz + (qpoints(kk)-0.5)*dz;
          real hr, ht;

          if (init_data_int == DATA_THERMAL) { hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht); }

          hy_dens_cells      (k) += hr    * qweights(kk);
          hy_dens_theta_cells(k) += hr*ht * qweights(kk);
        }
      });

      parallel_for( Bounds<1>(nz+1) , YAKL_LAMBDA (int k) {
        real z = k*dz;
        real hr, ht;

        if (init_data_int == DATA_THERMAL) { hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht); }

        hy_dens_edges      (k) = hr   ;
        hy_dens_theta_edges(k) = hr*ht;
      });

    }

    convert_dynamics_to_coupler( coupler , state , tracers );

    output( coupler , etime );
  }


  void init_supercell( core::Coupler &coupler , real4d &state , real4d &tracers ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;
    real constexpr z_0    = 0;
    real constexpr z_trop = 12000;
    real constexpr T_0    = 300;
    real constexpr T_trop = 213;
    real constexpr T_top  = 213;
    real constexpr p_0    = 100000;

    int constexpr ngll = 5;

    SArray<real,1,ngll> gll_pts;
    SArray<real,1,ngll> gll_wts;

    gll_pts(0)=-0.50000000000000000000000000000000000000;
    gll_pts(1)=-0.32732683535398857189914622812342917778;
    gll_pts(2)=0.00000000000000000000000000000000000000;
    gll_pts(3)=0.32732683535398857189914622812342917778;
    gll_pts(4)=0.50000000000000000000000000000000000000;

    gll_wts(0)=0.050000000000000000000000000000000000000;
    gll_wts(1)=0.27222222222222222222222222222222222222;
    gll_wts(2)=0.35555555555555555555555555555555555556;
    gll_wts(3)=0.27222222222222222222222222222222222222;
    gll_wts(4)=0.050000000000000000000000000000000000000;

    real3d quad_temp       ("quad_temp"       ,nz,ngll-1,ngll);
    real2d hyDensGLL       ("hyDensGLL"       ,nz,ngll);
    real2d hyDensThetaGLL  ("hyDensThetaGLL"  ,nz,ngll);
    real2d hyDensVapGLL    ("hyDensVapGLL"    ,nz,ngll);
    real2d hyPressureGLL   ("hyPressureGLL"   ,nz,ngll);
    real1d hyDensCells     ("hyDensCells"     ,nz);
    real1d hyDensThetaCells("hyDensThetaCells",nz);

    real ztop = coupler.get_zlen();

    // Compute quadrature term to integrate to get pressure at GLL points
    parallel_for( "Spatial.h init_state 4" , Bounds<3>(nz,ngll-1,ngll) ,
                  YAKL_LAMBDA (int k, int kk, int kkk) {
      // Middle of this cell
      real cellmid   = (k+0.5_fp) * dz;
      // Bottom, top, and middle of the space between these two ngll GLL points
      real ngll_b    = cellmid + gll_pts(kk  )*dz;
      real ngll_t    = cellmid + gll_pts(kk+1)*dz;
      real ngll_m    = 0.5_fp * (ngll_b + ngll_t);
      // Compute grid spacing between these ngll GLL points
      real ngll_dz   = dz * ( gll_pts(kk+1) - gll_pts(kk) );
      // Compute the locate of this GLL point within the ngll GLL points
      real zloc      = ngll_m + ngll_dz * gll_pts(kkk);
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
    parallel_for( "Spatial.h init_state 5" , 1 , YAKL_LAMBDA (int dummy) {
      hyPressureGLL(0,0) = p_0;
      for (int k=0; k < nz; k++) {
        for (int kk=0; kk < ngll-1; kk++) {
          real tot = 0;
          for (int kkk=0; kkk < ngll; kkk++) {
            tot += quad_temp(k,kk,kkk) * gll_wts(kkk);
          }
          tot *= dz * ( gll_pts(kk+1) - gll_pts(kk) );
          hyPressureGLL(k,kk+1) = hyPressureGLL(k,kk) * exp( tot );
          if (kk == ngll-2 && k < nz-1) {
            hyPressureGLL(k+1,0) = hyPressureGLL(k,ngll-1);
          }
        }
      }
    });

    // Compute hydrostatic background state at GLL points
    parallel_for( "Spatial.h init_state 6" , Bounds<2>(nz,ngll) , YAKL_LAMBDA (int k, int kk) {
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
      if (k == nz-1 && kk == ngll-1) {
        hy_dens_edges      (k+1) = dens;
        hy_dens_theta_edges(k+1) = dens_theta;
      }
    });

    // Compute hydrostatic background state over cells
    parallel_for( "Spatial.h init_state 7" , Bounds<1>(nz) , YAKL_LAMBDA (int k) {
      real press_tot      = 0;
      real dens_tot       = 0;
      real dens_vap_tot   = 0;
      real dens_theta_tot = 0;
      for (int kk=0; kk < ngll; kk++) {
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

    // Initialize the state
    parallel_for( "Spatial.h init_state 12" , Bounds<3>(nz,ny,nx) ,
                  YAKL_LAMBDA (int k, int j, int i) {
      state  (idR ,hs+k,hs+j,hs+i) = 0;
      state  (idU ,hs+k,hs+j,hs+i) = 0;
      state  (idV ,hs+k,hs+j,hs+i) = 0;
      state  (idW ,hs+k,hs+j,hs+i) = 0;
      state  (idT ,hs+k,hs+j,hs+i) = 0;
      for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i) = 0; }
      for (int kk=0; kk < ngll; kk++) {
        for (int jj=0; jj < ngll; jj++) {
          for (int ii=0; ii < ngll; ii++) {
            real xloc = (i+0.5_fp)*dx + gll_pts(ii)*dx;
            real yloc = (j+0.5_fp)*dy + gll_pts(jj)*dy;
            real zloc = (k+0.5_fp)*dz + gll_pts(kk)*dz;

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

            // real x0 = xlen / 2;
            // real y0 = ylen / 2;
            // real z0 = 1500;
            // real radx = 10000;
            // real rady = 10000;
            // real radz = 1500;
            // real amp  = 3;

            // real xn = (xloc - x0) / radx;
            // real yn = (yloc - y0) / rady;
            // real zn = (zloc - z0) / radz;

            // real rad = sqrt( xn*xn + yn*yn + zn*zn );

            // TODO: enable this whenever you want the standalone idealize non-MMF test for supercell
            // real theta_pert = 0;
            // if (rad < 1) {
            //   theta_pert = amp * pow( cos(M_PI*rad/2) , 2._fp );
            // }
            // dens_theta += dens * theta_pert;

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


  void convert_dynamics_to_coupler( core::Coupler &coupler , realConst4d state , realConst4d tracers ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    YAKL_SCOPE( hy_dens_cells       , this->hy_dens_cells      );
    YAKL_SCOPE( hy_dens_theta_cells , this->hy_dens_theta_cells );

    auto dm_rho_d = coupler.dm.get<real,3>("density_dry");
    auto dm_uvel  = coupler.dm.get<real,3>("uvel"       );
    auto dm_vvel  = coupler.dm.get<real,3>("vvel"       );
    auto dm_wvel  = coupler.dm.get<real,3>("wvel"       );
    auto dm_temp  = coupler.dm.get<real,3>("temp"       );

    core::MultiField<real,3> dm_tracers;
    auto tracer_names = coupler.get_tracer_names();
    for (int tr=0; tr < num_tracers; tr++) {
      dm_tracers.add_field( coupler.dm.get<real,3>(tracer_names[tr]) );
    }

    parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
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


  void convert_coupler_to_dynamics( core::Coupler const &coupler , real4d &state , real4d &tracers ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    YAKL_SCOPE( hy_dens_cells       , this->hy_dens_cells       );
    YAKL_SCOPE( hy_dens_theta_cells , this->hy_dens_theta_cells );

    auto dm_rho_d = coupler.dm.get<real const,3>("density_dry");
    auto dm_uvel  = coupler.dm.get<real const,3>("uvel"       );
    auto dm_vvel  = coupler.dm.get<real const,3>("vvel"       );
    auto dm_wvel  = coupler.dm.get<real const,3>("wvel"       );
    auto dm_temp  = coupler.dm.get<real const,3>("temp"       );

    core::MultiField<real const,3> dm_tracers;
    auto tracer_names = coupler.get_tracer_names();
    for (int tr=0; tr < num_tracers; tr++) {
      dm_tracers.add_field( coupler.dm.get<real const,3>(tracer_names[tr]) );
    }

    parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
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


  void output( core::Coupler const &coupler , real etime ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    real4d state  ("state"  ,num_state  ,hs+nz,hs+ny,hs+nx);
    real4d tracers("tracers",num_tracers,hs+nz,hs+ny,hs+nx);
    convert_coupler_to_dynamics( coupler , state , tracers );

    yakl::SimpleNetCDF nc;
    int ulIndex = 0; // Unlimited dimension index to place this data at

    if (etime == 0) {
      nc.create(fname);

      // x-coordinate
      real1d xloc("xloc",nx);
      parallel_for( "Spatial.h output 1" , nx , YAKL_LAMBDA (int i) { xloc(i) = (i+0.5)*dx; });
      nc.write(xloc.createHostCopy(),"x",{"x"});

      // y-coordinate
      real1d yloc("yloc",ny);
      parallel_for( "Spatial.h output 2" , ny , YAKL_LAMBDA (int i) { yloc(i) = (i+0.5)*dy; });
      nc.write(yloc.createHostCopy(),"y",{"y"});

      // z-coordinate
      real1d zloc("zloc",nz);
      parallel_for( "Spatial.h output 3" , nz , YAKL_LAMBDA (int i) { zloc(i) = (i+0.5)*dz; });
      nc.write(zloc.createHostCopy(),"z",{"z"});

      nc.write(hy_dens_cells      .createHostCopy(),"hydrostatic_density"      ,{"z"});
      nc.write(hy_dens_theta_cells.createHostCopy(),"hydrostatic_density_theta",{"z"});

      // Elapsed time
      nc.write1(0._fp,"t",0,"t");
    } else {
      nc.open(fname,yakl::NETCDF_MODE_WRITE);
      ulIndex = nc.getDimSize("t");

      // Write the elapsed time
      nc.write1(etime,"t",ulIndex,"t");
    }

    auto &dm = coupler.dm;
    nc.write1(dm.get<real const,3>("density_dry").createHostCopy(),"density_dry",{"z","y","x"},ulIndex,"t");
    nc.write1(dm.get<real const,3>("uvel"       ).createHostCopy(),"uvel"       ,{"z","y","x"},ulIndex,"t");
    nc.write1(dm.get<real const,3>("vvel"       ).createHostCopy(),"vvel"       ,{"z","y","x"},ulIndex,"t");
    nc.write1(dm.get<real const,3>("wvel"       ).createHostCopy(),"wvel"       ,{"z","y","x"},ulIndex,"t");
    nc.write1(dm.get<real const,3>("temp"       ).createHostCopy(),"temperature",{"z","y","x"},ulIndex,"t");
    auto tracer_names = coupler.get_tracer_names();
    for (int tr = 0; tr < num_tracers; tr++) {
      nc.write1(dm.get<real const,3>(tracer_names[tr]).createHostCopy(),tracer_names[tr],{"z","y","x"},ulIndex,"t");
    }

    YAKL_SCOPE( hy_dens_cells       , this->hy_dens_cells       );
    YAKL_SCOPE( hy_dens_theta_cells , this->hy_dens_theta_cells );

    real3d data("data",nz,ny,nx);
    parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      data(k,j,i) = state(idR,hs+k,hs+j,hs+i);
    });
    nc.write1(data.createHostCopy(),"density_pert",{"z","y","x"},ulIndex,"t");

    parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      real hy_r  = hy_dens_cells      (k);
      real hy_rt = hy_dens_theta_cells(k);
      real r     = state(idR,hs+k,hs+j,hs+i) + hy_r;
      real rt    = state(idT,hs+k,hs+j,hs+i) + hy_rt;
      data(k,j,i) = rt / r - hy_rt / hy_r;
    });
    nc.write1(data.createHostCopy(),"theta_pert",{"z","y","x"},ulIndex,"t");

    nc.close();
  }


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


  YAKL_INLINE real init_supercell_temperature(real z, real z_0, real z_trop, real z_top,
                                                      real T_0, real T_trop, real T_top) {
    if (z <= z_trop) {
      real lapse = - (T_trop - T_0) / (z_trop - z_0);
      return T_0 - lapse * (z - z_0);
    } else {
      real lapse = - (T_top - T_trop) / (z_top - z_trop);
      return T_trop - lapse * (z - z_trop);
    }
  }


  YAKL_INLINE real init_supercell_pressure_dry(real z, real z_0, real z_trop, real z_top,
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

  
  YAKL_INLINE real init_supercell_relhum(real z, real z_0, real z_trop) {
    if (z <= z_trop) {
      return 1._fp - 0.75_fp * pow(z / z_trop , 1.25_fp );
    } else {
      return 0.25_fp;
    }
  }

  
  YAKL_INLINE real init_supercell_relhum_d_dz(real z, real z_0, real z_trop) {
    if (z <= z_trop) {
      return -0.9375_fp*pow(z/z_trop, 0.25_fp)/z_trop;
    } else {
      return 0;
    }
  }


  YAKL_INLINE real init_supercell_sat_mix_dry( real press , real T ) {
    return 380/(press) * exp( 17.27_fp * (T-273)/(T-36) );
  }


  YAKL_INLINE real init_supercell_sat_mix_dry_d_dT( real p , real T ) {
    return 2205033.6*exp(17.27*T/(T - 36) - 6424.44/(T - 36))/(p*(T*T - 72*T + 1296));
  }


  YAKL_INLINE real init_supercell_sat_mix_dry_d_dp( real p , real T ) {
    return -380*exp(17.27*T/(T - 36) - 6424.44/(T - 36))/(p*p);
  }

};


