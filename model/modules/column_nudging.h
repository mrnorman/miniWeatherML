
#pragma once

#include "coupler.h"
#include "MultipleFields.h"

namespace modules {


  class ColumnNudger {
  public:
    int static constexpr num_fields = 5;
    real2d column;

    void set_column( core::Coupler &coupler ) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      int nx = coupler.get_nx();
      int ny = coupler.get_ny();
      int nz = coupler.get_nz();

      column = real2d("column",num_fields,nz);
      yakl::memset(column,0._fp);

      auto &dm = coupler.get_data_manager_readonly();

      core::MultiField<real const,3> state;
      state.add_field( dm.get<real const,3>("density_dry") );
      state.add_field( dm.get<real const,3>("uvel"       ) );
      state.add_field( dm.get<real const,3>("vvel"       ) );
      state.add_field( dm.get<real const,3>("temp"       ) );
      state.add_field( dm.get<real const,3>("water_vapor") );

      column = get_column_average( coupler , state );
    }


    void nudge_to_column( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      int nx = coupler.get_nx();
      int ny = coupler.get_ny();
      int nz = coupler.get_nz();

      auto &dm = coupler.get_data_manager_readwrite();

      core::MultiField<real,3> state;
      state.add_field( dm.get<real,3>("density_dry") );
      state.add_field( dm.get<real,3>("uvel"       ) );
      state.add_field( dm.get<real,3>("vvel"       ) );
      state.add_field( dm.get<real,3>("temp"       ) );
      state.add_field( dm.get<real,3>("water_vapor") );

      auto state_col_avg = get_column_average( coupler , state );

      YAKL_SCOPE( column , this->column );

      real constexpr time_scale = 900;
      parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(num_fields,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i) {
        state(l,k,j,i) += dt * ( column(l,k) - state_col_avg(l,k) ) / time_scale;
      });
    }


    template <class T>
    real2d get_column_average( core::Coupler const &coupler , core::MultiField<T,3> &state ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      int nz = coupler.get_nz();
      int ny = coupler.get_ny();
      int nx = coupler.get_nx();

      real2d column_loc("column_loc",num_fields,nz);
      yakl::memset( column_loc , 0._fp );

      real factor = 1._fp / (nx*ny);
      parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(num_fields,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i) {
        yakl::atomicAdd( column_loc(l,k) , state(l,k,j,i)*factor );
      });

      realHost2d column_total_host("column_total_host",num_fields,nz);
      MPI_Allreduce( column_loc.createHostCopy().data() , column_total_host.data() , num_fields*nz ,
                     MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD );
      
      auto column_total = column_total_host.createDeviceCopy();
      int nranks = coupler.get_nranks();
      parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(num_fields,nz) , YAKL_LAMBDA (int l, int k) {
        column_loc(l,k) = column_total(l,k) / nranks;
      });

      return column_loc;
    }

  };

}


