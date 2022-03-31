
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

      core::MultiField<real const,3> state;
      state.add_field( coupler.dm.get<real const,3>("density_dry") );
      state.add_field( coupler.dm.get<real const,3>("uvel"       ) );
      state.add_field( coupler.dm.get<real const,3>("vvel"       ) );
      state.add_field( coupler.dm.get<real const,3>("temp"       ) );
      state.add_field( coupler.dm.get<real const,3>("water_vapor") );

      YAKL_SCOPE( column , this->column );

      real factor = 1._fp / (nx*ny);
      parallel_for( Bounds<4>(num_fields,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i) {
        yakl::atomicAdd( column(l,k) , state(l,k,j,i)*factor );
      });
    }


    void nudge_to_column( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      int nx = coupler.get_nx();
      int ny = coupler.get_ny();
      int nz = coupler.get_nz();

      core::MultiField<real,3> state;
      state.add_field( coupler.dm.get<real,3>("density_dry") );
      state.add_field( coupler.dm.get<real,3>("uvel"       ) );
      state.add_field( coupler.dm.get<real,3>("vvel"       ) );
      state.add_field( coupler.dm.get<real,3>("temp"       ) );
      state.add_field( coupler.dm.get<real,3>("water_vapor") );

      real2d state_col_avg("state_col_avg",num_fields,nz);
      yakl::memset(state_col_avg,0._fp);
      real factor = 1._fp / (nx*ny);
      parallel_for( Bounds<4>(num_fields,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i) {
        yakl::atomicAdd( state_col_avg(l,k) , state(l,k,j,i)*factor );
      });

      YAKL_SCOPE( column , this->column );

      real constexpr time_scale = 900;
      parallel_for( Bounds<4>(num_fields,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i) {
        state(l,k,j,i) += dt * ( column(l,k) - state_col_avg(l,k) ) / time_scale;
      });
    }
  };

}


