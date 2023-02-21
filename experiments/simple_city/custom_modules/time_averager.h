#pragma once

#include "coupler.h"

namespace custom_modules {
  
  struct Time_Averager {
    real etime;

    void init(core::Coupler &coupler) {
      auto nens = coupler.get_nens();
      auto nx   = coupler.get_nx();
      auto ny   = coupler.get_ny();
      auto nz   = coupler.get_nz();

      auto &dm = coupler.get_data_manager_readwrite();

      dm.register_and_allocate<real>("time_avg_density_dry","",{nz,ny,nx,nens});
      dm.register_and_allocate<real>("time_avg_uvel"       ,"",{nz,ny,nx,nens});
      dm.register_and_allocate<real>("time_avg_vvel"       ,"",{nz,ny,nx,nens});
      dm.register_and_allocate<real>("time_avg_wvel"       ,"",{nz,ny,nx,nens});
      dm.register_and_allocate<real>("time_avg_temp"       ,"",{nz,ny,nx,nens});
      dm.register_and_allocate<real>("time_avg_water_vapor","",{nz,ny,nx,nens});

      dm.get<real,4>("time_avg_density_dry") = 0;
      dm.get<real,4>("time_avg_uvel"       ) = 0;
      dm.get<real,4>("time_avg_vvel"       ) = 0;
      dm.get<real,4>("time_avg_wvel"       ) = 0;
      dm.get<real,4>("time_avg_temp"       ) = 0;
      dm.get<real,4>("time_avg_water_vapor") = 0;

      etime = 0.;
    }

    void accumulate(core::Coupler &coupler , real dt) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;

      auto nens = coupler.get_nens();
      auto nx   = coupler.get_nx();
      auto ny   = coupler.get_ny();
      auto nz   = coupler.get_nz();

      auto &dm = coupler.get_data_manager_readwrite();

      auto rho_d = dm.get<real const,4>("density_dry");
      auto uvel  = dm.get<real const,4>("uvel"       );
      auto vvel  = dm.get<real const,4>("vvel"       );
      auto wvel  = dm.get<real const,4>("wvel"       );
      auto temp  = dm.get<real const,4>("temp"       );
      auto rho_v = dm.get<real const,4>("water_vapor");

      auto tavg_rho_d = dm.get<real,4>("time_avg_density_dry");
      auto tavg_uvel  = dm.get<real,4>("time_avg_uvel"       );
      auto tavg_vvel  = dm.get<real,4>("time_avg_vvel"       );
      auto tavg_wvel  = dm.get<real,4>("time_avg_wvel"       );
      auto tavg_temp  = dm.get<real,4>("time_avg_temp"       );
      auto tavg_rho_v = dm.get<real,4>("time_avg_water_vapor");

      double inertia = etime / (etime + dt);

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        tavg_rho_d(k,j,i,iens) = inertia * tavg_rho_d(k,j,i,iens) + (1-inertia) * rho_d(k,j,i,iens);
        tavg_uvel (k,j,i,iens) = inertia * tavg_uvel (k,j,i,iens) + (1-inertia) * uvel (k,j,i,iens);
        tavg_vvel (k,j,i,iens) = inertia * tavg_vvel (k,j,i,iens) + (1-inertia) * vvel (k,j,i,iens);
        tavg_wvel (k,j,i,iens) = inertia * tavg_wvel (k,j,i,iens) + (1-inertia) * wvel (k,j,i,iens);
        tavg_temp (k,j,i,iens) = inertia * tavg_temp (k,j,i,iens) + (1-inertia) * temp (k,j,i,iens);
        tavg_rho_v(k,j,i,iens) = inertia * tavg_rho_v(k,j,i,iens) + (1-inertia) * rho_v(k,j,i,iens);
      });
      
      etime += dt;
    }

    void finalize(core::Coupler &coupler ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;

      int  nx_glob = coupler.get_nx_glob();
      int  ny_glob = coupler.get_ny_glob();
      auto nens    = coupler.get_nens();
      auto nx      = coupler.get_nx();
      auto ny      = coupler.get_ny();
      auto nz      = coupler.get_nz();
      auto dx      = coupler.get_dx();
      auto dy      = coupler.get_dy();
      auto dz      = coupler.get_dz();
      int  i_beg   = coupler.get_i_beg();
      int  j_beg   = coupler.get_j_beg();

      auto &dm = coupler.get_data_manager_readonly();

      yakl::SimplePNetCDF nc;
      nc.create("time_averaged_fields.nc" , NC_CLOBBER | NC_64BIT_DATA);

      nc.create_dim( "x" , nx_glob );
      nc.create_dim( "y" , ny_glob );
      nc.create_dim( "z" , nz );

      nc.create_var<real>( "x" , {"x"} );
      nc.create_var<real>( "y" , {"y"} );
      nc.create_var<real>( "z" , {"z"} );
      nc.create_var<real>( "density_dry" , {"z","y","x"} );
      nc.create_var<real>( "uvel"        , {"z","y","x"} );
      nc.create_var<real>( "vvel"        , {"z","y","x"} );
      nc.create_var<real>( "wvel"        , {"z","y","x"} );
      nc.create_var<real>( "temp"        , {"z","y","x"} );
      nc.create_var<real>( "water_vapor" , {"z","y","x"} );

      nc.enddef();

      // x-coordinate
      real1d xloc("xloc",nx);
      parallel_for( YAKL_AUTO_LABEL() , nx , YAKL_LAMBDA (int i) { xloc(i) = (i+i_beg+0.5)*dx; });
      nc.write_all( xloc.createHostCopy() , "x" , {i_beg} );

      // y-coordinate
      real1d yloc("yloc",ny);
      parallel_for( YAKL_AUTO_LABEL() , ny , YAKL_LAMBDA (int j) { yloc(j) = (j+j_beg+0.5)*dy; });
      nc.write_all( yloc.createHostCopy() , "y" , {j_beg} );

      // z-coordinate
      real1d zloc("zloc",nz);
      parallel_for( YAKL_AUTO_LABEL() , nz , YAKL_LAMBDA (int k) { zloc(k) = (k      +0.5)*dz; });
      nc.begin_indep_data();
      if (coupler.is_mainproc()) {
        nc.write( zloc.createHostCopy() , "z" );
      }
      nc.end_indep_data();

      std::vector<std::string> varnames = {"density_dry","uvel","vvel","wvel","temp","water_vapor"};

      for (int i=0; i < varnames.size(); i++) {
        real3d data("data",nz,ny,nx);
        auto var = dm.get<real const,4>(std::string("time_avg_")+varnames[i]);
        parallel_for( SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {  data(k,j,i) = var(k,j,i,0); });
        nc.write_all(data.createHostCopy(),varnames[i],{0,j_beg,i_beg});
      }

      nc.close();
    }
  };

}


