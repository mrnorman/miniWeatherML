
#pragma once

#include "coupler.h"

namespace modules {

  inline void sponge_layer( core::Coupler &coupler , real dt ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    int  nz   = coupler.get_nz  ();
    int  ny   = coupler.get_ny  ();
    int  nx   = coupler.get_nx  ();
    real zlen = coupler.get_zlen();
    real dz   = coupler.get_dz  ();

    int num_layers = 5;  // Number of model top vertical layers that participate in the sponge relaxation

    int WFLD = 3; // fourth entry into "fields" is the "w velocity" field. Set the havg to zero for WFLD

    // Get a list of tracer names for retrieval
    std::vector<std::string> tracer_names = coupler.get_tracer_names();
    int num_tracers = coupler.get_num_tracers();

    auto &dm = coupler.get_data_manager_readwrite();

    // Create MultiField of all state and tracer full variables, since we're doing the same operation on each
    core::MultiField<real,3> full_fields;
    full_fields.add_field( dm.get<real,3>("density_dry") );
    full_fields.add_field( dm.get<real,3>("uvel"       ) );
    full_fields.add_field( dm.get<real,3>("vvel"       ) );
    full_fields.add_field( dm.get<real,3>("wvel"       ) );
    full_fields.add_field( dm.get<real,3>("temp"       ) );
    for (int tr=0; tr < num_tracers; tr++) {
      full_fields.add_field( dm.get<real,3>(tracer_names[tr]) );
    }

    int num_fields = full_fields.get_num_fields();

    // Compute the horizontal average for each vertical level (that we use for the sponge layer)
    real2d havg_fields("havg_fields",num_fields,num_layers);
    havg_fields = 0;
    real r_nx_ny = 1._fp / (nx*ny);
    parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(num_fields,num_layers,ny,nx) , YAKL_LAMBDA (int ifld, int kloc, int j, int i) {
      int k = nz - 1 - kloc;
      if (ifld != WFLD) yakl::atomicAdd( havg_fields(ifld,kloc) , full_fields(ifld,k,j,i) * r_nx_ny );
    });

    auto havg_fields_loc_host = havg_fields.createHostCopy();
    auto havg_fields_host = havg_fields_loc_host.createHostObject();
    auto mpi_data_type = coupler.get_mpi_data_type();
    MPI_Allreduce( havg_fields_loc_host.data() , havg_fields_host.data() , havg_fields_host.size() , mpi_data_type , MPI_SUM , MPI_COMM_WORLD );
    havg_fields_host.deep_copy_to(havg_fields);  // After this, havg_fields has the sum, not average, over tasks

    int num_tasks;
    MPI_Comm_size( MPI_COMM_WORLD , &num_tasks );

    real constexpr time_scale = 60;  // strength of each application is dt / time_scale  (same as SAM's tau_min)
    real time_factor = dt / time_scale;

    // use a cosine relaxation in space:  ((cos(pi*rel_dist)+1)/2)^2
    parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(num_fields,num_layers,ny,nx) , YAKL_LAMBDA (int ifld, int kloc, int j, int i) {
      int k = nz - 1 - kloc;
      real z = (k+0.5_fp)*dz;
      real rel_dist = ( zlen - z ) / ( num_layers * dz );
      real space_factor = ( cos(M_PI*rel_dist) + 1 ) / 2;
      real factor = space_factor * time_factor;
      full_fields(ifld,k,j,i) += ( havg_fields(ifld,kloc)/num_tasks - full_fields(ifld,k,j,i) ) * factor;
    });
  }

}

