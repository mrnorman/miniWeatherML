
#pragma once

#include "coupler.h"

namespace modules {

  inline void sponge_layer( core::Coupler &coupler , real dt , real time_scale = 60 ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    auto ny_glob = coupler.get_ny_glob();
    auto nx_glob = coupler.get_nx_glob();
    auto nz      = coupler.get_nz  ();
    auto ny      = coupler.get_ny  ();
    auto nx      = coupler.get_nx  ();
    auto nens    = coupler.get_nens();
    auto zlen    = coupler.get_zlen();
    auto dz      = coupler.get_dz  ();

    int num_layers = 10;  // Number of model top vertical layers that participate in the sponge relaxation

    int WFLD = 3; // fourth entry into "fields" is the "w velocity" field. Set the havg to zero for WFLD

    // Get a list of tracer names for retrieval
    std::vector<std::string> tracer_names = coupler.get_tracer_names();
    int num_tracers = coupler.get_num_tracers();

    auto &dm = coupler.get_data_manager_readwrite();

    // Create MultiField of all state and tracer full variables, since we're doing the same operation on each
    core::MultiField<real,4> full_fields;
    full_fields.add_field( dm.get<real,4>("density_dry") );
    full_fields.add_field( dm.get<real,4>("uvel"       ) );
    full_fields.add_field( dm.get<real,4>("vvel"       ) );
    full_fields.add_field( dm.get<real,4>("wvel"       ) );
    full_fields.add_field( dm.get<real,4>("temp"       ) );
    for (int tr=0; tr < num_tracers; tr++) {
      full_fields.add_field( dm.get<real,4>(tracer_names[tr]) );
    }

    int num_fields = full_fields.get_num_fields();

    // Compute the horizontal average for each vertical level (that we use for the sponge layer)
    real3d havg_fields("havg_fields",num_fields,num_layers,nens);
    havg_fields = 0;
    parallel_for( YAKL_AUTO_LABEL() , Bounds<5>(num_fields,num_layers,ny,nx,nens) ,
                                      YAKL_LAMBDA (int ifld, int kloc, int j, int i, int iens) {
      int k = nz - 1 - kloc;
      if (ifld != WFLD) yakl::atomicAdd( havg_fields(ifld,kloc,iens) , full_fields(ifld,k,j,i,iens) );
    });

    #ifdef MW_GPU_AWARE_MPI
      auto havg_fields_loc = havg_fields.createDeviceCopy(); // Has an implicit fence()
      MPI_Allreduce( havg_fields_loc.data() , havg_fields.data() , havg_fields.size() ,
                     coupler.get_mpi_data_type() , MPI_SUM , MPI_COMM_WORLD );
    #else
      auto havg_fields_loc_host = havg_fields.createHostCopy();
      auto havg_fields_host = havg_fields_loc_host.createHostObject();
      MPI_Allreduce( havg_fields_loc_host.data() , havg_fields_host.data() , havg_fields_host.size() ,
                     coupler.get_mpi_data_type() , MPI_SUM , MPI_COMM_WORLD );
      havg_fields_host.deep_copy_to(havg_fields);  // After this, havg_fields has the sum, not average, over tasks
    #endif

    real time_factor = dt / time_scale;

    // use a cosine relaxation in space:  ((cos(pi*rel_dist)+1)/2)^2
    parallel_for( YAKL_AUTO_LABEL() , Bounds<5>(num_fields,num_layers,ny,nx,nens) ,
                                      YAKL_LAMBDA (int ifld, int kloc, int j, int i, int iens) {
      int k = nz - 1 - kloc;
      real z = (k+0.5_fp)*dz;
      real rel_dist = ( zlen - z ) / ( num_layers * dz );
      real space_factor = ( cos(M_PI*rel_dist) + 1 ) / 2;
      real factor = space_factor * time_factor;
      full_fields(ifld,k,j,i,iens) += ( havg_fields(ifld,kloc,iens)/(nx_glob*ny_glob) - full_fields(ifld,k,j,i,iens) ) * factor;
    });
  }

}

