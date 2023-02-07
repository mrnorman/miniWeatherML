
#pragma once

#include "coupler.h"

namespace custom_modules {
  
  struct Uvel_Sponge {
    real1d col_rho_d, col_uvel, col_vvel, col_wvel, col_temp, col_rho_v;

    int sponge_cells = 5;

    inline void init( core::Coupler &coupler ) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      YAKL_SCOPE( col_rho_d , this->col_rho_d );
      YAKL_SCOPE( col_uvel  , this->col_uvel  );
      YAKL_SCOPE( col_vvel  , this->col_vvel  );
      YAKL_SCOPE( col_wvel  , this->col_wvel  );
      YAKL_SCOPE( col_temp  , this->col_temp  );
      YAKL_SCOPE( col_rho_v , this->col_rho_v );

      auto nz = coupler.get_nz();

      auto &dm = coupler.get_data_manager_readonly();
      auto rho_d = dm.get<real const,3>("density_dry");
      auto uvel  = dm.get<real const,3>("uvel");
      auto vvel  = dm.get<real const,3>("vvel");
      auto wvel  = dm.get<real const,3>("wvel");
      auto temp  = dm.get<real const,3>("temp");
      auto rho_v = dm.get<real const,3>("water_vapor");

      col_rho_d = real1d("col_rho_d",nz);
      col_uvel  = real1d("col_uvel ",nz);
      col_vvel  = real1d("col_vvel ",nz);
      col_wvel  = real1d("col_wvel ",nz);
      col_temp  = real1d("col_temp ",nz);
      col_rho_v = real1d("col_rho_v",nz);

      parallel_for( YAKL_AUTO_LABEL() , nz , YAKL_LAMBDA (int k) {
        col_rho_d(k) = rho_d(k,0,0);
        col_uvel (k) = uvel (k,0,0);
        col_vvel (k) = vvel (k,0,0);
        col_wvel (k) = wvel (k,0,0);
        col_temp (k) = temp (k,0,0);
        col_rho_v(k) = rho_v(k,0,0);
      });
    }


    inline void apply( core::Coupler &coupler ) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      real strength = 1.0;

      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      auto i_beg   = coupler.get_i_beg();

      real R_d   = coupler.get_option<real>("R_d" ,287     );
      real cp_d  = coupler.get_option<real>("cp_d",1004    );
      real p0    = coupler.get_option<real>("p0"  ,1.e5    );
      real kappa = R_d / cp_d;
      real gamma = cp_d / (cp_d - R_d);
      real C0    = pow( R_d * pow( p0 , -kappa ) , gamma );


      auto &dm = coupler.get_data_manager_readwrite();
      auto rho_d = dm.get<real,3>("density_dry");
      auto uvel  = dm.get<real,3>("uvel");
      auto vvel  = dm.get<real,3>("vvel");
      auto wvel  = dm.get<real,3>("wvel");
      auto temp  = dm.get<real,3>("temp");
      auto rho_v = dm.get<real,3>("water_vapor");

      YAKL_SCOPE( sponge_cells , this->sponge_cells );
      YAKL_SCOPE( col_rho_d    , this->col_rho_d    );
      YAKL_SCOPE( col_uvel     , this->col_uvel     );
      YAKL_SCOPE( col_vvel     , this->col_vvel     );
      YAKL_SCOPE( col_wvel     , this->col_wvel     );
      YAKL_SCOPE( col_temp     , this->col_temp     );
      YAKL_SCOPE( col_rho_v    , this->col_rho_v    );

      if (coupler.get_px() == 0) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          real xloc   = i / (sponge_cells-1._fp);
          real weight = i < sponge_cells ? (cos(M_PI*xloc)+1)/2 : 0;
          weight *= strength;
          rho_d(k,j,i) = weight*col_rho_d(k) + (1-weight)*rho_d(k,j,i);
          uvel (k,j,i) = weight*col_uvel (k) + (1-weight)*uvel (k,j,i);
          vvel (k,j,i) = weight*col_vvel (k) + (1-weight)*vvel (k,j,i);
          wvel (k,j,i) = weight*col_wvel (k) + (1-weight)*wvel (k,j,i);
          temp (k,j,i) = weight*col_temp (k) + (1-weight)*temp (k,j,i);
          rho_v(k,j,i) = weight*col_rho_v(k) + (1-weight)*rho_v(k,j,i);
        });
      }
      if (coupler.get_px() == coupler.get_nproc_x()-1) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          real xloc   = (nx-1-i) / (sponge_cells-1._fp);
          real weight = nx-1-i < sponge_cells ? (cos(M_PI*xloc)+1)/2 : 0;
          weight *= strength;
          rho_d(k,j,i) = weight*col_rho_d(k) + (1-weight)*rho_d(k,j,i);
          uvel (k,j,i) = weight*col_uvel (k) + (1-weight)*uvel (k,j,i);
          vvel (k,j,i) = weight*col_vvel (k) + (1-weight)*vvel (k,j,i);
          wvel (k,j,i) = weight*col_wvel (k) + (1-weight)*wvel (k,j,i);
          temp (k,j,i) = weight*col_temp (k) + (1-weight)*temp (k,j,i);
          rho_v(k,j,i) = weight*col_rho_v(k) + (1-weight)*rho_v(k,j,i);
        });
      }
    }
  };
}


