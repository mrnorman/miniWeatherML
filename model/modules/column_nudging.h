
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
      using yakl::c::SimpleBounds;

      int nens = coupler.get_nens();
      int nx   = coupler.get_nx();
      int ny   = coupler.get_ny();
      int nz   = coupler.get_nz();

      column = real3d("column",num_fields,nz,nens);
      column = 0;

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
      using yakl::c::SimpleBounds;

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
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_fields,nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
        state(l,k,j,i,iens) += dt * ( column(l,k,iens) - state_col_avg(l,k,iens) ) / time_scale;
      });
    }


    template <class T>
    real3d get_column_average( core::Coupler const &coupler , core::MultiField<T,4> &state ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      using yakl::componentwise::operator/;

      auto nens    = coupler.get_nens();
      auto nx      = coupler.get_nx();
      auto ny      = coupler.get_ny();
      auto nz      = coupler.get_nz();
      auto nx_glob = coupler.get_nx_glob();
      auto ny_glob = coupler.get_ny_glob();
      auto nranks  = coupler.get_nranks();

      real3d column_loc("column_loc",num_fields,nz,nens);
      column_loc = 0;

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_fields,nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
        yakl::atomicAdd( column_loc(l,k,iens) , state(l,k,j,i,iens) );
      });

      realHost3d column_total_host("column_total_host",num_fields,nz,nens);
      MPI_Allreduce( column_loc.createHostCopy().data() , column_total_host.data() , column_total_host.size() ,
                     coupler.get_mpi_data_type() , MPI_SUM , MPI_COMM_WORLD );
      return column_total_host.createDeviceCopy() / (ny_glob*nx_glob*nranks);
    }

  };

}


