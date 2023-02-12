
#pragma once

#include "coupler.h"

namespace custom_modules {

  inline void nudge( core::Coupler &coupler , bool nudge_u , bool nudge_v ,
                     real uval , real vval , real dt , real time_scale ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    auto nx_glob = coupler.get_nx_glob();
    auto ny_glob = coupler.get_ny_glob();
    auto nx      = coupler.get_nx();
    auto ny      = coupler.get_ny();
    auto nz      = coupler.get_nz();

    auto dm = coupler.get_data_manager_readwrite();
    auto uvel = dm.get<real,3>("uvel");
    auto vvel = dm.get<real,3>("vvel");

    real vel_loc[2];
    real vel[2];
    auto vel_loc[0] = yakl::intrinsics::sum(uvel);
    auto vel_loc[1] = yakl::intrinsics::sum(vvel);

    MPI_Allreduce(vel_loc, vel, 2, coupler.get_mpi_data_type() , MPI_SUM , MPI_COMM_WORLD );

    vel_loc[0] /= nz*nx_glob*ny_glob;
    vel_loc[1] /= nz*nx_glob*ny_glob;

    real forcing_u = (uval - vel[0]) * dt / time_scale;
    real forcing_v = (vval - vel[1]) * dt / time_scale;

    parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
    });
  }
}


