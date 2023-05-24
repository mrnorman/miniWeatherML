
#pragma once

namespace custom_modules {

  template <class MODEL>
  inline real2d calculate_loss_and_overwrite_lo( core::Coupler &coupler_lo ,
                                                 core::Coupler &coupler_hi ,
                                                 MODEL const &model       ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    using yakl::intrinsics::minval;
    using yakl::intrinsics::maxval;
    using yakl::intrinsics::sum;
    // lo-res grid
    auto nx_lo = coupler_lo.get_nx();
    auto ny_lo = coupler_lo.get_ny();
    auto nz_lo = coupler_lo.get_nz();
    auto nens  = coupler_lo.get_nens();
    // hi-res grid
    auto nx_hi = coupler_hi.get_nx();
    auto ny_hi = coupler_hi.get_ny();
    auto nz_hi = coupler_hi.get_nz();
    auto refine_factor = nx_hi / nx_lo;
    // Get tracer information
    auto num_tracers = coupler_lo.get_num_tracers();
    auto tracer_names = coupler_lo.get_tracer_names();
    // Accrue lo state
    auto &dm_lo = coupler_lo.get_data_manager_readwrite();
    core::MultiField<real,4> fields_lo;
    fields_lo.add_field( dm_lo.get<real,4>("density_dry") );
    fields_lo.add_field( dm_lo.get<real,4>("uvel") );
    fields_lo.add_field( dm_lo.get<real,4>("vvel") );
    fields_lo.add_field( dm_lo.get<real,4>("wvel") );
    fields_lo.add_field( dm_lo.get<real,4>("temp") );
    for (int tr = 0; tr < num_tracers; tr++) { fields_lo.add_field( dm_lo.get<real,4>(tracer_names[tr]) ); }
    // Accrue hi state
    auto &dm_hi = coupler_hi.get_data_manager_readonly();
    core::MultiField<real const,4> fields_hi;
    fields_hi.add_field( dm_hi.get<real const,4>("density_dry") );
    fields_hi.add_field( dm_hi.get<real const,4>("uvel"       ) );
    fields_hi.add_field( dm_hi.get<real const,4>("vvel"       ) );
    fields_hi.add_field( dm_hi.get<real const,4>("wvel"       ) );
    fields_hi.add_field( dm_hi.get<real const,4>("temp"       ) );
    for (int tr = 0; tr < num_tracers; tr++) { fields_hi.add_field( dm_hi.get<real const,4>(tracer_names[tr]) ); }
    // Get number of fields
    auto num_fields = fields_hi.get_num_fields();
    // Calculate min
    auto dtype = coupler_lo.get_mpi_data_type();
    realHost1d min_loc("min_loc",num_fields);
    for (int l=0; l < num_fields; l++) { min_loc(l) = minval( fields_lo.get_field(l) ); }
    realHost1d min_glob("min_glob",num_fields);
    MPI_Allreduce( min_loc.data() , min_glob.data() , min_loc.size() , dtype , MPI_MIN , MPI_COMM_WORLD );
    // Calculate max
    realHost1d max_loc("max_loc",num_fields);
    for (int l=0; l < num_fields; l++) { max_loc(l) = maxval( fields_lo.get_field(l) ); }
    realHost1d max_glob("max_glob",num_fields);
    MPI_Allreduce( max_loc.data() , max_glob.data() , max_loc.size() , dtype , MPI_MAX , MPI_COMM_WORLD );
    // Calculate range
    realHost1d range_host("range_host",num_fields);
    for (int l=0; l < num_fields; l++) { range_host(l) = std::max( 1.e-20 , max_glob(l)-min_glob(l) ); }
    auto range = range_host.createDeviceCopy();
    // Calculate the coarsened hi-res solution
    real4d coarsened("coarsened",num_fields,nz_lo,ny_lo,nx_lo);
    if (coupler_lo.is_sim2d()) {
      real rfac = 1._fp/(refine_factor*refine_factor);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz_lo,ny_lo,nx_lo) , 
                                        YAKL_LAMBDA (int l, int k_lo, int j_lo, int i_lo) {
        int constexpr iens = 0;
        real val = 0;
        for (int kk=0; kk < refine_factor; kk++) {
          int k = k_lo*refine_factor + kk;
          int j = j_lo;
          for (int ii=0; ii < refine_factor; ii++) {
            int i = i_lo*refine_factor + ii;
            val += fields_hi(l,k,j,i,iens);
          }
        }
        coarsened(l,k_lo,j_lo,i_lo) = val * rfac;
      });
    } else {
      real rfac = 1._fp/(refine_factor*refine_factor*refine_factor);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz_lo,ny_lo,nx_lo) , 
                                        YAKL_LAMBDA (int l, int k_lo, int j_lo, int i_lo) {
        int constexpr iens = 0;
        real val = 0;
        for (int kk=0; kk < refine_factor; kk++) {
          int k = k_lo*refine_factor + kk;
          for (int jj=0; jj < refine_factor; jj++) {
            int j = j_lo*refine_factor + jj;
            for (int ii=0; ii < refine_factor; ii++) {
              int i = i_lo*refine_factor + ii;
              val += fields_hi(l,k,j,i,iens);
            }
          }
        }
        coarsened(l,k_lo,j_lo,i_lo) = val * rfac;
      });
    }
    // Compute the losses (normalized MSE), and overwrite the lo-res solution for each ensemble member
    real4d loss4d("loss4d",nens,nz_lo,ny_lo,nx_lo);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz_lo,ny_lo,nx_lo,nens) ,
                                      YAKL_LAMBDA (int k, int j, int i, int iens) {
      real val = 0;
      for (int l = 0; l < num_fields; l++) {
        real adiff = ( coarsened(l,k,j,i) - fields_lo(l,k,j,i,iens) ) / range(l);
        val += adiff*adiff;
        fields_lo(l,k,j,i,iens) = coarsened(l,k,j,i);
      }
      loss4d(iens,k,j,i) = val / num_fields;
    });
    return loss4d.reshape(nens,nx_lo*ny_lo*nz_lo);
  }

}


