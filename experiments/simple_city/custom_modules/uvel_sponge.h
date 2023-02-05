
#pragma once

#include "coupler.h"

namespace custom_modules {
  inline void uvel_sponge( core::Coupler &coupler ) {

    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    real R_d   = coupler.get_option<real>("R_d" ,287);
    real cp_d  = coupler.get_option<real>("cp_d",1004);
    real cp_v  = coupler.get_option<real>("cp_v",1004-287);
    real p0    = coupler.get_option<real>("p0"  ,1.e5);
    real kappa = R_d / cp_d;
    real gamma = cp_d / (cp_d - R_d);
    real C0    = pow( R_d * pow( p0 , -kappa ) , gamma );

    auto nx      = coupler.get_nx();
    auto ny      = coupler.get_ny();
    auto nz      = coupler.get_nz();
    auto i_beg   = coupler.get_i_beg();
    auto nx_glob = coupler.get_nx_glob();

    auto &dm = coupler.get_data_manager_readwrite();
    auto rhod = dm.get<real,3>("density_dry");
    auto uvel = dm.get<real,3>("uvel");
    auto vvel = dm.get<real,3>("vvel");
    auto wvel = dm.get<real,3>("wvel");
    auto temp = dm.get<real,3>("temp");

    int num_cells = 0.10 * nx_glob;

    parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      real xloc   = (i_beg+i) / (num_cells-1._fp);
      real weight = i_beg+i < num_cells ? (cos(M_PI*xloc)+1)/2 : 0;
      real T = 300;
      real u = 20;
      real v = 0;
      real w = 0;
      real p = p0;
      real rho = p/(R_d*T);
      rhod(k,j,i) = weight*rho + (1-weight)*rhod(k,j,i);
      uvel(k,j,i) = weight*u   + (1-weight)*uvel(k,j,i);
      vvel(k,j,i) = weight*v   + (1-weight)*vvel(k,j,i);
      wvel(k,j,i) = weight*w   + (1-weight)*wvel(k,j,i);
      temp(k,j,i) = weight*T   + (1-weight)*temp(k,j,i);
    });
  }
}


