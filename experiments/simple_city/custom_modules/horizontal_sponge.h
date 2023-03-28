
#pragma once

#include "coupler.h"

namespace custom_modules {
  
  struct Horizontal_Sponge {
    real2d col_rho_d;
    real2d col_uvel;
    real2d col_vvel;
    real2d col_wvel;
    real2d col_temp;
    real2d col_rho_v;
    int    sponge_cells;
    real   time_scale;

    inline void init( core::Coupler &coupler , int sponge_cells = 10 , real time_scale = 1 ) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      YAKL_SCOPE( col_rho_d , this->col_rho_d );
      YAKL_SCOPE( col_uvel  , this->col_uvel  );
      YAKL_SCOPE( col_vvel  , this->col_vvel  );
      YAKL_SCOPE( col_wvel  , this->col_wvel  );
      YAKL_SCOPE( col_temp  , this->col_temp  );
      YAKL_SCOPE( col_rho_v , this->col_rho_v );

      auto nens = coupler.get_nens();
      auto nz   = coupler.get_nz();

      auto &dm = coupler.get_data_manager_readonly();
      auto rho_d = dm.get<real const,4>("density_dry");
      auto uvel  = dm.get<real const,4>("uvel");
      auto vvel  = dm.get<real const,4>("vvel");
      auto wvel  = dm.get<real const,4>("wvel");
      auto temp  = dm.get<real const,4>("temp");
      auto rho_v = dm.get<real const,4>("water_vapor");

      col_rho_d = real2d("col_rho_d",nz,nens);
      col_uvel  = real2d("col_uvel ",nz,nens);
      col_vvel  = real2d("col_vvel ",nz,nens);
      col_wvel  = real2d("col_wvel ",nz,nens);
      col_temp  = real2d("col_temp ",nz,nens);
      col_rho_v = real2d("col_rho_v",nz,nens);

      auto col_rho_d_host = col_rho_d.createHostObject();
      auto col_uvel_host  = col_uvel .createHostObject();
      auto col_vvel_host  = col_vvel .createHostObject();
      auto col_wvel_host  = col_wvel .createHostObject();
      auto col_temp_host  = col_temp .createHostObject();
      auto col_rho_v_host = col_rho_v.createHostObject();

      if (coupler.is_mainproc()) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz,nens) , YAKL_LAMBDA (int k, int iens) {
          col_rho_d(k,iens) = rho_d(k,0,0,iens);
          col_uvel (k,iens) = uvel (k,0,0,iens);
          col_vvel (k,iens) = vvel (k,0,0,iens);
          col_wvel (k,iens) = wvel (k,0,0,iens);
          col_temp (k,iens) = temp (k,0,0,iens);
          col_rho_v(k,iens) = rho_v(k,0,0,iens);
        });
        col_rho_d.deep_copy_to(col_rho_d_host);
        col_uvel .deep_copy_to(col_uvel_host );
        col_vvel .deep_copy_to(col_vvel_host );
        col_wvel .deep_copy_to(col_wvel_host );
        col_temp .deep_copy_to(col_temp_host );
        col_rho_v.deep_copy_to(col_rho_v_host);
        yakl::fence();
      }

      MPI_Bcast( col_rho_d_host.data(), col_rho_d_host.size() , coupler.get_mpi_data_type() , 0 , MPI_COMM_WORLD );
      MPI_Bcast( col_uvel_host .data(), col_uvel_host .size() , coupler.get_mpi_data_type() , 0 , MPI_COMM_WORLD );
      MPI_Bcast( col_vvel_host .data(), col_vvel_host .size() , coupler.get_mpi_data_type() , 0 , MPI_COMM_WORLD );
      MPI_Bcast( col_wvel_host .data(), col_wvel_host .size() , coupler.get_mpi_data_type() , 0 , MPI_COMM_WORLD );
      MPI_Bcast( col_temp_host .data(), col_temp_host .size() , coupler.get_mpi_data_type() , 0 , MPI_COMM_WORLD );
      MPI_Bcast( col_rho_v_host.data(), col_rho_v_host.size() , coupler.get_mpi_data_type() , 0 , MPI_COMM_WORLD );

      if (! coupler.is_mainproc()) {
        col_rho_d_host.deep_copy_to(col_rho_d);
        col_uvel_host .deep_copy_to(col_uvel );
        col_vvel_host .deep_copy_to(col_vvel );
        col_wvel_host .deep_copy_to(col_wvel );
        col_temp_host .deep_copy_to(col_temp );
        col_rho_v_host.deep_copy_to(col_rho_v);
      }
      this->sponge_cells = sponge_cells;
      this->time_scale   = time_scale;
    }


    void override_rho_d(real val) { col_rho_d = val; }
    void override_uvel (real val) { col_uvel  = val; }
    void override_vvel (real val) { col_vvel  = val; }
    void override_wvel (real val) { col_wvel  = val; }
    void override_temp (real val) { col_temp  = val; }
    void override_rho_v(real val) { col_rho_v = val; }


    inline void apply( core::Coupler &coupler , real dt ,
                       bool x1=true , bool x2=true , bool y1=true , bool y2=true ) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      auto nens = coupler.get_nens();
      auto nx   = coupler.get_nx();
      auto ny   = coupler.get_ny();
      auto nz   = coupler.get_nz();

      real R_d   = coupler.get_option<real>("R_d" );
      real cp_d  = coupler.get_option<real>("cp_d");
      real p0    = coupler.get_option<real>("p0"  );
      real kappa = R_d / cp_d;
      real gamma = cp_d / (cp_d - R_d);
      real C0    = pow( R_d * pow( p0 , -kappa ) , gamma );


      auto &dm = coupler.get_data_manager_readwrite();
      auto rho_d = dm.get<real,4>("density_dry");
      auto uvel  = dm.get<real,4>("uvel");
      auto vvel  = dm.get<real,4>("vvel");
      auto wvel  = dm.get<real,4>("wvel");
      auto temp  = dm.get<real,4>("temp");
      auto rho_v = dm.get<real,4>("water_vapor");

      YAKL_SCOPE( sponge_cells , this->sponge_cells );
      YAKL_SCOPE( time_scale   , this->time_scale   );
      YAKL_SCOPE( col_rho_d    , this->col_rho_d    );
      YAKL_SCOPE( col_uvel     , this->col_uvel     );
      YAKL_SCOPE( col_vvel     , this->col_vvel     );
      YAKL_SCOPE( col_wvel     , this->col_wvel     );
      YAKL_SCOPE( col_temp     , this->col_temp     );
      YAKL_SCOPE( col_rho_v    , this->col_rho_v    );

      real time_factor = dt / time_scale;

      if (coupler.get_px() == 0 && x1) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          real xloc   = i / (sponge_cells-1._fp);
          real weight = i < sponge_cells ? (cos(M_PI*xloc)+1)/2 : 0;
          weight *= time_factor;
          rho_d(k,j,i,iens) = weight*col_rho_d(k,iens) + (1-weight)*rho_d(k,j,i,iens);
          uvel (k,j,i,iens) = weight*col_uvel (k,iens) + (1-weight)*uvel (k,j,i,iens);
          vvel (k,j,i,iens) = weight*col_vvel (k,iens) + (1-weight)*vvel (k,j,i,iens);
          wvel (k,j,i,iens) = weight*col_wvel (k,iens) + (1-weight)*wvel (k,j,i,iens);
          temp (k,j,i,iens) = weight*col_temp (k,iens) + (1-weight)*temp (k,j,i,iens);
          rho_v(k,j,i,iens) = weight*col_rho_v(k,iens) + (1-weight)*rho_v(k,j,i,iens);
        });
      }
      if (coupler.get_px() == coupler.get_nproc_x()-1 && x2) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          real xloc   = (nx-1-i) / (sponge_cells-1._fp);
          real weight = nx-1-i < sponge_cells ? (cos(M_PI*xloc)+1)/2 : 0;
          weight *= time_factor;
          rho_d(k,j,i,iens) = weight*col_rho_d(k,iens) + (1-weight)*rho_d(k,j,i,iens);
          uvel (k,j,i,iens) = weight*col_uvel (k,iens) + (1-weight)*uvel (k,j,i,iens);
          vvel (k,j,i,iens) = weight*col_vvel (k,iens) + (1-weight)*vvel (k,j,i,iens);
          wvel (k,j,i,iens) = weight*col_wvel (k,iens) + (1-weight)*wvel (k,j,i,iens);
          temp (k,j,i,iens) = weight*col_temp (k,iens) + (1-weight)*temp (k,j,i,iens);
          rho_v(k,j,i,iens) = weight*col_rho_v(k,iens) + (1-weight)*rho_v(k,j,i,iens);
        });
      }
      if (coupler.get_py() == 0 && y1) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          real yloc   = j / (sponge_cells-1._fp);
          real weight = j < sponge_cells ? (cos(M_PI*yloc)+1)/2 : 0;
          weight *= time_factor;
          rho_d(k,j,i,iens) = weight*col_rho_d(k,iens) + (1-weight)*rho_d(k,j,i,iens);
          uvel (k,j,i,iens) = weight*col_uvel (k,iens) + (1-weight)*uvel (k,j,i,iens);
          vvel (k,j,i,iens) = weight*col_vvel (k,iens) + (1-weight)*vvel (k,j,i,iens);
          wvel (k,j,i,iens) = weight*col_wvel (k,iens) + (1-weight)*wvel (k,j,i,iens);
          temp (k,j,i,iens) = weight*col_temp (k,iens) + (1-weight)*temp (k,j,i,iens);
          rho_v(k,j,i,iens) = weight*col_rho_v(k,iens) + (1-weight)*rho_v(k,j,i,iens);
        });
      }
      if (coupler.get_py() == coupler.get_nproc_y()-1 && y2) {
        parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          real yloc   = (ny-1-j) / (sponge_cells-1._fp);
          real weight = ny-1-j < sponge_cells ? (cos(M_PI*yloc)+1)/2 : 0;
          weight *= time_factor;
          rho_d(k,j,i,iens) = weight*col_rho_d(k,iens) + (1-weight)*rho_d(k,j,i,iens);
          uvel (k,j,i,iens) = weight*col_uvel (k,iens) + (1-weight)*uvel (k,j,i,iens);
          vvel (k,j,i,iens) = weight*col_vvel (k,iens) + (1-weight)*vvel (k,j,i,iens);
          wvel (k,j,i,iens) = weight*col_wvel (k,iens) + (1-weight)*wvel (k,j,i,iens);
          temp (k,j,i,iens) = weight*col_temp (k,iens) + (1-weight)*temp (k,j,i,iens);
          rho_v(k,j,i,iens) = weight*col_rho_v(k,iens) + (1-weight)*rho_v(k,j,i,iens);
        });
      }
    }
  };
}


