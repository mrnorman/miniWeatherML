
#pragma once

#include "main_header.h"

class Dycore {
  public:

  int  static constexpr num_state = 5;
  int  static constexpr hs        = 2;
  int  static constexpr sten_size = 4;
  real static constexpr hv_beta   = 0.05;

  int  static constexpr idR       = 0;
  int  static constexpr idU       = 1;
  int  static constexpr idV       = 2;
  int  static constexpr idW       = 3;
  int  static constexpr idT       = 4;

  real1d      hy_dens_cells;
  real1d      hy_dens_theta_cells;
  real1d      hy_dens_edges;
  real1d      hy_dens_theta_edges;
  real        etime;
  real        out_freq;
  int         num_out;
  std::string fname;


  real compute_time_step( core::Coupler const &coupler ) {
    real constexpr maxwave = 350 + 30;
    real constexpr cfl = 0.3;
    real dt_dyn = cfl * std::min( std::min( coupler.get_dx() , coupler.get_dy() ) , coupler.get_dz() ) / maxwave;
    return dt_dyn;
  }


  void time_step(core::Coupler &coupler, real &dt_phys) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    int  nx, ny, nz;
    real xlen, ylen, zlen, dx, dy, dz;
    bool sim2d;
    get_grid(coupler,nx,ny,nz,xlen,ylen,zlen,dx,dy,dz,sim2d);

    real4d state("state",num_state,nz+2*hs,ny+2*hs,nx+2*hs);
    convert_coupler_to_dynamics( coupler , state );

    real dt_dyn = compute_time_step( coupler );

    int ncycles = (int) std::ceil( dt_phys / dt_dyn );
    dt_dyn = dt_phys / ncycles;
    
    for (int icycle = 0; icycle < ncycles; icycle++) {
      real4d state_tmp("state_tmp",num_state,nz+2*hs,ny+2*hs,nx+2*hs);
      real4d tend     ("tend"     ,num_state,nz     ,ny     ,nx     );
      // Stage 1
      compute_tendencies( coupler , state     , tend , dt_dyn/3._fp );
      parallel_for( Bounds<4>(num_state,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i) {
        state_tmp(l,hs+k,hs+j,hs+i) = state(l,hs+k,hs+j,hs+i) + dt_dyn/3._fp * tend(l,k,j,i);
      });
      // Stage 2
      compute_tendencies( coupler , state_tmp , tend , dt_dyn/2._fp );
      parallel_for( Bounds<4>(num_state,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i) {
        state_tmp(l,hs+k,hs+j,hs+i) = state(l,hs+k,hs+j,hs+i) + dt_dyn/2._fp * tend(l,k,j,i);
      });
      // Stage 3
      compute_tendencies( coupler , state_tmp , tend , dt_dyn/1._fp );
      parallel_for( Bounds<4>(num_state,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i) {
        state    (l,hs+k,hs+j,hs+i) = state(l,hs+k,hs+j,hs+i) + dt_dyn/1._fp * tend(l,k,j,i);
      });
    }

    convert_dynamics_to_coupler( coupler , state );

    etime += dt_phys;
    if (out_freq >= 0. && etime / out_freq >= num_out+1) {
      yakl::timer_start("output");
      output( coupler , etime );
      yakl::timer_stop("output");
      num_out++;
    }
  }


  void compute_tendencies( core::Coupler const &coupler , real4d const &state , real4d const &tend , real dt ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    int  nx, ny, nz;
    real xlen, ylen, zlen, dx, dy, dz;
    bool sim2d;
    get_grid(coupler,nx,ny,nz,xlen,ylen,zlen,dx,dy,dz,sim2d);

    real R_d, R_v, cp_d, cp_v, p0, grav, kappa, gamma, C0;
    get_constants(coupler, R_d, R_v, cp_d, cp_v, p0, grav, kappa, gamma, C0);

    real4d flux_x("flux_x",num_state,nz  ,ny  ,nx+1);
    real4d flux_y("flux_y",num_state,nz  ,ny+1,nx  );
    real4d flux_z("flux_z",num_state,nz+1,ny  ,nx  );

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
        flux_x(idR,k,j,i) = r*u     - hv_coef*d3_vals(idR);
        flux_x(idU,k,j,i) = r*u*u+p - hv_coef*d3_vals(idU);
        flux_x(idV,k,j,i) = r*u*v   - hv_coef*d3_vals(idV);
        flux_x(idW,k,j,i) = r*u*w   - hv_coef*d3_vals(idW);
        flux_x(idT,k,j,i) = r*u*t   - hv_coef*d3_vals(idT);
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
        flux_y(idR,k,j,i) = r*v     - hv_coef*d3_vals(idR);
        flux_y(idU,k,j,i) = r*v*u   - hv_coef*d3_vals(idU);
        flux_y(idV,k,j,i) = r*v*v+p - hv_coef*d3_vals(idV);
        flux_y(idW,k,j,i) = r*v*w   - hv_coef*d3_vals(idW);
        flux_y(idT,k,j,i) = r*v*t   - hv_coef*d3_vals(idT);
      } else if (i < nx && k < nz) {
        flux_y(idR,k,j,i) = 0;
        flux_y(idU,k,j,i) = 0;
        flux_y(idV,k,j,i) = 0;
        flux_y(idW,k,j,i) = 0;
        flux_y(idT,k,j,i) = 0;
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
            int ind = std::min( nz+hs-1 , std::max( hs , k+s ) );
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
        flux_z(idR,k,j,i) = r*w     - hv_coef*d3_vals(idR);
        flux_z(idU,k,j,i) = r*w*u   - hv_coef*d3_vals(idU);
        flux_z(idV,k,j,i) = r*w*v   - hv_coef*d3_vals(idV);
        flux_z(idW,k,j,i) = r*w*w+p - hv_coef*d3_vals(idW);
        flux_z(idT,k,j,i) = r*w*t   - hv_coef*d3_vals(idT);
      }
    });

    parallel_for( Bounds<4>(num_state,nz,ny,nx) , YAKL_LAMBDA ( int l, int k, int j, int i ) {
      tend(l,k,j,i) = -( flux_x(l,k  ,j  ,i+1) - flux_x(l,k,j,i) ) / dx
                      -( flux_y(l,k  ,j+1,i  ) - flux_y(l,k,j,i) ) / dy
                      -( flux_z(l,k+1,j  ,i  ) - flux_z(l,k,j,i) ) / dz;
      if (l == idW) tend(l,k,j,i) += -grav * ( state(idR,hs+k,hs+j,hs+i) + hy_dens_cells(k) );
      if (l == idV && sim2d) tend(l,k,j,i) = 0;
    });
  }


  void init(core::Coupler &coupler) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;
    auto inFile = coupler.get_option<std::string>( "standalone_input_file" );
    YAML::Node config = YAML::LoadFile(inFile);
    auto init_data = config["init_data"].as<std::string>();
    fname          = config["out_fname"].as<std::string>();
    out_freq       = config["out_freq" ].as<real>();

    int  nx, ny, nz;
    real xlen, ylen, zlen, dx, dy, dz;
    bool sim2d;
    get_grid(coupler,nx,ny,nz,xlen,ylen,zlen,dx,dy,dz,sim2d);

    real R_d, R_v, cp_d, cp_v, p0, grav, kappa, gamma, C0;
    get_constants(coupler, R_d, R_v, cp_d, cp_v, p0, grav, kappa, gamma, C0);

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
    real4d state("state",num_state,nz+2*hs,ny+2*hs,nx+2*hs);

    parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      for (int l=0; l < num_state; l++) {
        state(l,hs+k,hs+j,hs+i) = 0.;
      }
      //Use Gauss-Legendre quadrature
      for (int kk=0; kk<nqpoints; kk++) {
        for (int jj=0; jj<nqpoints; jj++) {
          for (int ii=0; ii<nqpoints; ii++) {
            real x = (i+0.5)*dx + (qpoints(ii)-0.5)*dx;
            real y = (j+0.5)*dy + (qpoints(jj)-0.5)*dy;   if (sim2d) y = ylen/2;
            real z = (k+0.5)*dz + (qpoints(kk)-0.5)*dz;
            real r, u, v, w, t, hr, ht;

            if (init_data == "thermal") { thermal(x,y,z,xlen,ylen,grav,C0,gamma,cp_d,p0,R_d,r,u,v,w,t,hr,ht); }
            else { endrun("ERROR: init_data not supported"); }

            if (sim2d) v = 0;

            real wt = qweights(ii)*qweights(jj)*qweights(kk);
            state(idR,hs+k,hs+j,hs+i) += r                         * wt;
            state(idU,hs+k,hs+j,hs+i) += (r+hr)*u                  * wt;
            state(idV,hs+k,hs+j,hs+i) += (r+hr)*v                  * wt;
            state(idW,hs+k,hs+j,hs+i) += (r+hr)*w                  * wt;
            state(idT,hs+k,hs+j,hs+i) += ( (r+hr)*(t+ht) - hr*ht ) * wt;
          }
        }
      }
    });

    hy_dens_cells       = real1d("hy_dens_cells"      ,nz  );
    hy_dens_theta_cells = real1d("hy_dens_theta_cells",nz  );
    hy_dens_edges       = real1d("hy_dens_edges"      ,nz+1);
    hy_dens_theta_edges = real1d("hy_dens_theta_edges",nz+1);

    parallel_for( Bounds<1>(nz) , YAKL_LAMBDA (int k) {
      hy_dens_cells      (k) = 0.;
      hy_dens_theta_cells(k) = 0.;
      for (int kk=0; kk<nqpoints; kk++) {
        real z = (k+0.5)*dz + (qpoints(kk)-0.5)*dz;
        real hr, ht;

        if (init_data == "thermal") { hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht); }

        hy_dens_cells      (k) += hr    * qweights(kk);
        hy_dens_theta_cells(k) += hr*ht * qweights(kk);
      }
    });

    parallel_for( Bounds<1>(nz+1) , YAKL_LAMBDA (int k) {
      real z = k*dz;
      real hr, ht;

      if (init_data == "thermal") { hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht); }

      hy_dens_edges      (k) = hr   ;
      hy_dens_theta_edges(k) = hr*ht;
    });

    convert_dynamics_to_coupler( coupler , state );

    output( coupler , etime );
  }


  void convert_dynamics_to_coupler( core::Coupler &coupler , realConst4d state ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;
    int  nx, ny, nz;
    real xlen, ylen, zlen, dx, dy, dz;
    bool sim2d;
    get_grid(coupler,nx,ny,nz,xlen,ylen,zlen,dx,dy,dz,sim2d);

    real R_d, R_v, cp_d, cp_v, p0, grav, kappa, gamma, C0;
    get_constants(coupler, R_d, R_v, cp_d, cp_v, p0, grav, kappa, gamma, C0);

    YAKL_SCOPE( hy_dens_cells       , this->hy_dens_cells      );
    YAKL_SCOPE( hy_dens_theta_cells , this->hy_dens_theta_cells );

    auto rho_d = coupler.dm.get<real,3>("density_dry");
    auto uvel  = coupler.dm.get<real,3>("uvel"       );
    auto vvel  = coupler.dm.get<real,3>("vvel"       );
    auto wvel  = coupler.dm.get<real,3>("wvel"       );
    auto temp  = coupler.dm.get<real,3>("temp"       );

    parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      real r  = state(idR,hs+k,hs+j,hs+i) + hy_dens_cells(k);
      real ru = state(idU,hs+k,hs+j,hs+i);
      real rv = state(idV,hs+k,hs+j,hs+i);
      real rw = state(idW,hs+k,hs+j,hs+i);
      real rt = state(idT,hs+k,hs+j,hs+i) + hy_dens_theta_cells(k);
      real p  = C0 * pow( rt , gamma );

      rho_d(k,j,i) = r;
      uvel (k,j,i) = ru / r;
      vvel (k,j,i) = rv / r;
      wvel (k,j,i) = rw / r;
      temp (k,j,i) = p / rho_d(k,j,i) / R_d;
    });
  }


  void convert_coupler_to_dynamics( core::Coupler const &coupler , real4d &state ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;
    int  nx, ny, nz;
    real xlen, ylen, zlen, dx, dy, dz;
    bool sim2d;
    get_grid(coupler,nx,ny,nz,xlen,ylen,zlen,dx,dy,dz,sim2d);

    real R_d, R_v, cp_d, cp_v, p0, grav, kappa, gamma, C0;
    get_constants(coupler, R_d, R_v, cp_d, cp_v, p0, grav, kappa, gamma, C0);

    YAKL_SCOPE( hy_dens_cells       , this->hy_dens_cells       );
    YAKL_SCOPE( hy_dens_theta_cells , this->hy_dens_theta_cells );

    auto rho_d = coupler.dm.get<real const,3>("density_dry");
    auto uvel  = coupler.dm.get<real const,3>("uvel"       );
    auto vvel  = coupler.dm.get<real const,3>("vvel"       );
    auto wvel  = coupler.dm.get<real const,3>("wvel"       );
    auto temp  = coupler.dm.get<real const,3>("temp"       );

    parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      real r  = rho_d(k,j,i);
      real ru = uvel(k,j,i) * r;
      real rv = vvel(k,j,i) * r;
      real rw = wvel(k,j,i) * r;
      real T  = temp(k,j,i);
      real p  = rho_d(k,j,i) * R_d * T;
      real rt = pow( p/C0 , 1._fp / gamma );

      state(idR,hs+k,hs+j,hs+i) = r - hy_dens_cells(k);
      state(idU,hs+k,hs+j,hs+i) = ru;
      state(idV,hs+k,hs+j,hs+i) = rv;
      state(idW,hs+k,hs+j,hs+i) = rw;
      state(idT,hs+k,hs+j,hs+i) = rt - hy_dens_theta_cells(k);
    });
  }


  void output( core::Coupler const &coupler , real etime ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    int  nz = coupler.get_nz();   int  ny = coupler.get_ny();   int  nx = coupler.get_nx();
    real dz = coupler.get_dz();   real dy = coupler.get_dy();   real dx = coupler.get_dx();

    real4d state("state",num_state,hs+nz,hs+ny,hs+nx);
    convert_coupler_to_dynamics( coupler , state );

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
    nc.write1(dm.get<real const,3>("density_dry"),"density_dry",{"z","y","x"},ulIndex,"t");
    nc.write1(dm.get<real const,3>("uvel"       ),"uvel"       ,{"z","y","x"},ulIndex,"t");
    nc.write1(dm.get<real const,3>("vvel"       ),"vvel"       ,{"z","y","x"},ulIndex,"t");
    nc.write1(dm.get<real const,3>("wvel"       ),"wvel"       ,{"z","y","x"},ulIndex,"t");
    nc.write1(dm.get<real const,3>("temp"       ),"temperature",{"z","y","x"},ulIndex,"t");

    real3d data("data",nz,ny,nx);
    parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      data(k,j,i) = state(idR,hs+k,hs+j,hs+i);
    });
    nc.write1(data,"density_pert",{"z","y","x"},ulIndex,"t");
    parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      real hy_r  = hy_dens_cells      (k);
      real hy_rt = hy_dens_theta_cells(k);
      real r     = state(idR,hs+k,hs+j,hs+i) + hy_r;
      real rt    = state(idT,hs+k,hs+j,hs+i) + hy_rt;
      data(k,j,i) = rt / r - hy_rt / hy_r;
    });
    nc.write1(data,"theta_pert",{"z","y","x"},ulIndex,"t");

    nc.close();
  }


  YAKL_INLINE static void thermal(real x, real y, real z, real xlen, real ylen, real grav, real C0, real gamma,
                                  real cp, real p0, real rd, real &r, real &u, real &v, real &w, real &t,
                                  real &hr, real &ht) {
    hydro_const_theta(z,grav,C0,cp,p0,gamma,rd,hr,ht);
    r = 0.;
    t = 0.;
    u = 0.;
    v = 0.;
    w = 0.;
    t = t + sample_ellipse_cosine(3._fp  ,  x,y,z  ,  xlen/2,ylen/2,2000.  ,  2000.,2000.,2000.);
  }


  YAKL_INLINE static void hydro_const_theta( real z, real grav, real C0, real cp, real p0, real gamma, real rd,
                                             real &r, real &t ) {
    const real theta0 = 300.;  //Background potential temperature
    const real exner0 = 1.;    //Surface-level Exner pressure
    t = theta0;                                       //Potential Temperature at z
    real exner = exner0 - grav * z / (cp * theta0);   //Exner pressure at z
    real p = p0 * pow(exner,(cp/rd));                 //Pressure at z
    real rt = pow((p / C0),(1._fp / gamma));          //rho*theta at z
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
      return amp * pow(cos(dist),2._fp);
    } else {
      return 0.;
    }
  }


  static void get_grid( core::Coupler const &coupler, int  &nx  , int  &ny  , int  &nz  ,
                                                      real &xlen, real &ylen, real &zlen,
                                                      real &dx  , real &dy  , real &dz  , bool &sim2d) {
    xlen = coupler.get_xlen();   ylen = coupler.get_ylen();   zlen = coupler.get_zlen();
    nx   = coupler.get_nx  ();   ny   = coupler.get_ny  ();   nz   = coupler.get_nz  ();
    dx   = coupler.get_dx  ();   dy   = coupler.get_dy  ();   dz   = coupler.get_dz  ();
    sim2d = (ny == 1);
  }


  static void get_constants(core::Coupler const &coupler, real &R_d, real &R_v, real &cp_d, real &cp_v, real &p0,
                                                          real &grav, real &kappa, real &gamma, real &C0) {
    R_d  = coupler.R_d ;
    R_v  = coupler.R_v ;
    cp_d = coupler.cp_d;
    cp_v = coupler.cp_v;
    p0   = coupler.p0  ;
    grav = coupler.grav;
    kappa = R_d / cp_d;
    gamma = cp_d / (cp_d - R_d);
    C0    = pow( R_d * pow( p0 , -kappa ) , gamma );
  }

};


