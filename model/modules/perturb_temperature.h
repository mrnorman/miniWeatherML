
#pragma once

#include "coupler.h"

namespace modules {

  inline void perturb_temperature( core::Coupler &coupler ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    int nz = coupler.get_nz();
    int ny = coupler.get_ny();
    int nx = coupler.get_nx();

    // {
    //   int  num_levels = nz / 4;
    //   real magnitude  = 3.;

    //   size_t seed = static_cast<size_t>(coupler.get_myrank()*nz*nx*ny);

    //   // ny*nx can all be globbed together for this routine
    //   auto &dm = coupler.get_data_manager_readwrite();
    //   auto temp = dm.get_lev_col<real>("temp");
    //   int ncol = ny*nx;

    //   parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(num_levels,ncol) , YAKL_LAMBDA (int k, int i) {
    //     yakl::Random prng(seed+k*ncol+i);  // seed + k*ncol + i  is a globally unique identifier
    //     real rand = prng.genFP<real>()*2._fp - 1._fp;  // Random number in [-1,1]
    //     real scaling = ( num_levels - static_cast<real>(k) ) / num_levels;  // Less effect at higher levels
    //     temp(k,i) += rand * magnitude * scaling;
    //   });
    // }

    {
      auto &dm = coupler.get_data_manager_readwrite();
      auto temp = dm.get<real,3>("temp");

      size_t i_beg = coupler.get_i_beg();
      size_t j_beg = coupler.get_j_beg();
      real   dx    = coupler.get_dx();
      real   dy    = coupler.get_dy();
      real   dz    = coupler.get_dz();
      real   xlen  = coupler.get_xlen();
      real   ylen  = coupler.get_ylen();

      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        real xloc = (i+i_beg+0.5_fp)*dx;
        real yloc = (j+j_beg+0.5_fp)*dy;
        real zloc = (k      +0.5_fp)*dz;
        real x0 = xlen / 2;
        real y0 = ylen / 2;
        real z0 = 1500;
        real radx = 10000;
        real rady = 10000;
        real radz = 1500;
        real amp  = 5;
        real xn = (xloc - x0) / radx;
        real yn = (yloc - y0) / rady;
        real zn = (zloc - z0) / radz;
        real rad = sqrt( xn*xn + yn*yn + zn*zn );
        if (rad < 1) {
          temp(k,j,i) += amp * pow( cos(M_PI*rad/2) , 2._fp );
        }
      });
    }
  }

}


