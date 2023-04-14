
#pragma once

#include "coupler.h"
#include "MultipleFields.h"

namespace modules {


  class ColumnNudger {
  public:
    int static constexpr num_fields = 5;
    real3d column;

    void set_column( core::Coupler &coupler ) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      int nens = coupler.get_nens();
      int nx   = coupler.get_nx();
      int ny   = coupler.get_ny();
      int nz   = coupler.get_nz();

      column = real3d("column",num_fields,nz,nens);

      auto &dm = coupler.get_data_manager_readonly();

      core::MultiField<real const,4> state;
      state.add_field( dm.get<real const,4>("density_dry") );
      state.add_field( dm.get<real const,4>("uvel"       ) );
      state.add_field( dm.get<real const,4>("vvel"       ) );
      state.add_field( dm.get<real const,4>("temp"       ) );
      state.add_field( dm.get<real const,4>("water_vapor") );

      column = get_column_average( coupler , state );
    }


    void nudge_to_column( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      int nens = coupler.get_nens();
      int nx   = coupler.get_nx();
      int ny   = coupler.get_ny();
      int nz   = coupler.get_nz();

      auto &dm = coupler.get_data_manager_readwrite();

      core::MultiField<real,4> state;
      state.add_field( dm.get<real,4>("density_dry") );
      state.add_field( dm.get<real,4>("uvel"       ) );
      state.add_field( dm.get<real,4>("vvel"       ) );
      state.add_field( dm.get<real,4>("temp"       ) );
      state.add_field( dm.get<real,4>("water_vapor") );

      auto state_col_avg = get_column_average( coupler , state );

      YAKL_SCOPE( column , this->column );

      real constexpr time_scale = 900;
      parallel_for( YAKL_AUTO_LABEL() , Bounds<5>(num_fields,nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
        state(l,k,j,i,iens) += dt * ( column(l,k,iens) - state_col_avg(l,k,iens) ) / time_scale;
      });
    }


    template <class T>
    real3d get_column_average( core::Coupler const &coupler , core::MultiField<T,4> &state ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      int nx_glob = coupler.get_nx_glob();
      int ny_glob = coupler.get_ny_glob();
      int nens    = coupler.get_nens();
      int nx      = coupler.get_nx();
      int ny      = coupler.get_ny();
      int nz      = coupler.get_nz();

      real3d column_loc("column_loc",num_fields,nz,nens);
      column_loc = 0;

      parallel_for( YAKL_AUTO_LABEL() , Bounds<5>(num_fields,nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
        yakl::atomicAdd( column_loc(l,k,iens) , state(l,k,j,i,iens) );
      });

      #ifdef MW_GPU_AWARE_MPI
        auto column_total = column_loc.createDeviceObject();
        yakl::fence();
        MPI_Allreduce( column_loc.data() , column_total.data() , column_total.size() ,
                       coupler.get_mpi_data_type() , MPI_SUM , MPI_COMM_WORLD );
      #else
        auto column_total_host = column_loc.createHostObject();
        MPI_Allreduce( column_loc.createHostCopy().data() , column_total_host.data() , column_total_host.size() ,
                       coupler.get_mpi_data_type() , MPI_SUM , MPI_COMM_WORLD );
        auto column_total = column_total_host.createDeviceCopy();
      #endif

      parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(num_fields,nz,nens) , YAKL_LAMBDA (int l, int k, int iens) {
        column_loc(l,k,iens) = column_total(l,k,iens) / (nx_glob*ny_glob);
      });

      return column_loc;
    }

  };

}


