
#pragma once

#include "coupler.h"

namespace modules {

  inline void perturb_temperature( core::Coupler &coupler ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    int nz = coupler.get_nz();
    int ny = coupler.get_ny();
    int nx = coupler.get_nx();

    int  num_levels = nz / 4;
    real magnitude  = 10.;

    size_t seed = static_cast<size_t>(coupler.get_myrank()*nz*nx*ny);

    // ny*nx can all be globbed together for this routine
    auto &dm = coupler.get_data_manager_readwrite();
    auto temp = dm.get_lev_col<real>("temp");
    int ncol = ny*nx;

    parallel_for( "perturb temperature" , Bounds<2>(num_levels,ncol) , YAKL_LAMBDA (int k, int i) {
      yakl::Random prng(seed+k*ncol+i);  // seed + k*ncol + i  is a globally unique identifier
      real rand = prng.genFP<real>()*2._fp - 1._fp;  // Random number in [-1,1]
      real scaling = ( num_levels - static_cast<real>(k) ) / num_levels;  // Less effect at higher levels
      temp(k,i) += rand * magnitude * scaling;
    });
  }

}


