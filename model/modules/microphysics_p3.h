
#pragma once

#include "coupler.h"


extern "C"
void p3_main_fortran(double *qc , double *nc , double *qr , double *nr , double *th_atm , double *qv ,
                     double &dt , double *qi , double *qm , double *ni , double *bm , double *pres ,
                     double *dz , double *nc_nuceat_tend , double *nccn_prescribed , double *ni_activated ,
                     double *inv_qc_relvar , int &it , double *precip_liq_surf , double *precip_ice_surf ,
                     int &its , int &ite , int &kts , int &kte , double *diag_eff_radius_qc ,
                     double *diag_eff_radius_qi , double *rho_qi , bool &do_predict_nc , 
                     bool &do_prescribed_CCN ,double *dpres , double *exner , double *qv2qi_depos_tend ,
                     double *precip_total_tend , double *nevapr , double *qr_evap_tend ,
                     double *precip_liq_flux , double *precip_ice_flux , double *cld_frac_r ,
                     double *cld_frac_l , double *cld_frac_i , double *p3_tend_out , double *mu_c ,
                     double *lamc , double *liq_ice_exchange , double *vap_liq_exchange , 
                     double *vap_ice_exchange , double *qv_prev , double *t_prev , double *col_location ,
                     double *elapsed_s );


extern "C"
void micro_p3_utils_init_fortran(real &cpair , real &rair , real &rh2o , real &rhoh2o , real &mwh2o ,
                                 real &mwdry , real &gravit , real &latvap , real &latice , real &cpliq ,
                                 real &tmelt , real &pi , int &iulog , bool &mainproc );


extern "C"
void p3_init_fortran(char const *lookup_file_dir , int &dir_len , char const *version_p3 , int &ver_len );


namespace modules {

  class Microphysics_P3 {
  public:
    // Doesn't actually have to be static or constexpr. Could be assigned in the constructor
    int static constexpr num_tracers = 9;

    // You should set these in the constructor
    real R_d    ;
    real cp_d   ;
    real cv_d   ;
    real gamma_d;
    real kappa_d;
    real R_v    ;
    real cp_v   ;
    real cv_v   ;
    real p0     ;

    real grav;
    real cp_l;

    bool first_step;

    real etime;

    // Indices for all of your tracer quantities
    int static constexpr ID_C  = 0;  // Local index for Cloud Water Mass  
    int static constexpr ID_NC = 1;  // Local index for Cloud Water Number
    int static constexpr ID_R  = 2;  // Local index for Rain Water Mass   
    int static constexpr ID_NR = 3;  // Local index for Rain Water Number 
    int static constexpr ID_I  = 4;  // Local index for Ice Mass          
    int static constexpr ID_M  = 5;  // Local index for Ice Number        
    int static constexpr ID_NI = 6;  // Local index for Ice-Rime Mass     
    int static constexpr ID_BM = 7;  // Local index for Ice-Rime Volume   
    int static constexpr ID_V  = 8;  // Local index for Water Vapor       



    // Set constants and likely num_tracers as well, and anything else you can do immediately
    Microphysics_P3() {
      R_d        = 287.042;
      cp_d       = 1004.64;
      cv_d       = cp_d - R_d;
      gamma_d    = cp_d / cv_d;
      kappa_d    = R_d  / cp_d;
      R_v        = 461.505;
      cp_v       = 1859;
      cv_v       = R_v - cp_v;
      p0         = 1.e5;
      grav       = 9.80616;
      first_step = true;
      cp_l       = 4188.0;
    }



    // This must return the correct # of tracers **BEFORE** init(...) is called
    YAKL_INLINE static int get_num_tracers() {
      return num_tracers;
    }



    // Can do whatever you want, but mainly for registering tracers and allocating data
    void init(core::Coupler &coupler) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      int nens = coupler.get_nens();
      int nx   = coupler.get_nx();
      int ny   = coupler.get_ny();
      int nz   = coupler.get_nz();

      // Register tracers in the coupler
      //                 name                description            positive   adds mass
      coupler.add_tracer("cloud_water"     , "Cloud Water Mass"   , true     , true );
      coupler.add_tracer("cloud_water_num" , "Cloud Water Number" , true     , false);
      coupler.add_tracer("rain"            , "Rain Water Mass"    , true     , true );
      coupler.add_tracer("rain_num"        , "Rain Water Number"  , true     , false);
      coupler.add_tracer("ice"             , "Ice Mass"           , true     , true );
      coupler.add_tracer("ice_num"         , "Ice Number"         , true     , false);
      coupler.add_tracer("ice_rime"        , "Ice-Rime Mass"      , true     , false);
      coupler.add_tracer("ice_rime_vol"    , "Ice-Rime Volume"    , true     , false);
      coupler.add_tracer("water_vapor"     , "Water Vapor"        , true     , true );

      auto &dm = coupler.get_data_manager_readwrite();

      dm.register_and_allocate<real>("qv_prev","qv from prev step"         ,{nz,ny,nx,nens},{"z","y","x","nens"});
      dm.register_and_allocate<real>("t_prev" ,"Temperature from prev step",{nz,ny,nx,nens},{"z","y","x","nens"});

      dm.get<real,4>( "cloud_water"     ) = 0;
      dm.get<real,4>( "cloud_water_num" ) = 0;
      dm.get<real,4>( "rain"            ) = 0;
      dm.get<real,4>( "rain_num"        ) = 0;
      dm.get<real,4>( "ice"             ) = 0;
      dm.get<real,4>( "ice_num"         ) = 0;
      dm.get<real,4>( "ice_rime"        ) = 0;
      dm.get<real,4>( "ice_rime_vol"    ) = 0;
      dm.get<real,4>( "water_vapor"     ) = 0;
      dm.get<real,4>( "qv_prev"         ) = 0;
      dm.get<real,4>( "t_prev"          ) = 0;

      real rhoh2o = 1000.;
      real mwdry  = 28.966;
      real mwh2o  = 18.016;
      real latvap = 2501000.0;
      real latice = 333700.0;
      real tmelt  = 273.15;
      real pi     = 3.14159265;
      int  iulog  = 1;
      bool mainproc = true;
      micro_p3_utils_init_fortran( cp_d , R_d , R_v , rhoh2o , mwh2o , mwdry ,
                                   grav , latvap , latice, cp_l , tmelt , pi , iulog , mainproc );

      std::string dir = "../model/modules/helpers/microphysics_p3";
      std::string ver = "4.1.1";
      int dir_len = dir.length();
      int ver_len = ver.length();
      p3_init_fortran( dir.c_str() , dir_len , ver.c_str() , ver_len );

      coupler.set_option<std::string>("micro","p3");
      coupler.set_option<real>("R_d"    ,R_d    );
      coupler.set_option<real>("cp_d"   ,cp_d   );
      coupler.set_option<real>("cv_d"   ,cv_d   );
      coupler.set_option<real>("gamma_d",gamma_d);
      coupler.set_option<real>("kappa_d",kappa_d);
      coupler.set_option<real>("R_v"    ,R_v    );
      coupler.set_option<real>("cp_v"   ,cp_v   );
      coupler.set_option<real>("cv_v"   ,cv_v   );
      coupler.set_option<real>("p0"     ,p0     );
      coupler.set_option<real>("grav"   ,grav   );

      etime = 0;
    }


    void time_step( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      // Get the dimensions sizes
      int nz   = coupler.get_nz();
      int ny   = coupler.get_ny();
      int nx   = coupler.get_nx();
      int nens = coupler.get_nens();
      int ncol = ny*nx*nens;

      real crm_dx = coupler.get_dx();
      real crm_dy = coupler.get_ny_glob() == 1 ? crm_dx : coupler.get_dy();

      auto &dm = coupler.get_data_manager_readwrite();

      // Get tracers dimensioned as (nz,ny*nx*nens)
      auto rho_c  = dm.get_lev_col<real>("cloud_water"    );
      auto rho_nc = dm.get_lev_col<real>("cloud_water_num");
      auto rho_r  = dm.get_lev_col<real>("rain"           );
      auto rho_nr = dm.get_lev_col<real>("rain_num"       );
      auto rho_i  = dm.get_lev_col<real>("ice"            );
      auto rho_ni = dm.get_lev_col<real>("ice_num"        );
      auto rho_m  = dm.get_lev_col<real>("ice_rime"       );
      auto rho_bm = dm.get_lev_col<real>("ice_rime_vol"   );
      auto rho_v  = dm.get_lev_col<real>("water_vapor"    );

      // Get coupler state
      auto rho_dry = dm.get_lev_col<real>("density_dry");
      auto temp    = dm.get_lev_col<real>("temp"       );

      real dz = coupler.get_dz();

      // Calculate the grid spacing
      real2d dz_arr("dz_arr",nz,ncol);
      yakl::memset( dz_arr , dz );

      // Get everything from the DataManager that's not a tracer but is persistent across multiple micro calls
      auto qv_prev = dm.get_lev_col<real>("qv_prev");
      auto t_prev  = dm.get_lev_col<real>("t_prev" );

      // Allocates inputs and outputs
      int p3_nout = 49;
      real2d qc                ( "qc"                 ,           nz   , ncol );
      real2d nc                ( "nc"                 ,           nz   , ncol );
      real2d qr                ( "qr"                 ,           nz   , ncol );
      real2d nr                ( "nr"                 ,           nz   , ncol );
      real2d qi                ( "qi"                 ,           nz   , ncol );
      real2d ni                ( "ni"                 ,           nz   , ncol );
      real2d qm                ( "qm"                 ,           nz   , ncol );
      real2d bm                ( "bm"                 ,           nz   , ncol );
      real2d qv                ( "qv"                 ,           nz   , ncol );
      real2d pressure          ( "pressure"           ,           nz   , ncol );
      real2d theta             ( "theta"              ,           nz   , ncol );
      real2d exner             ( "exner"              ,           nz   , ncol );
      real2d inv_exner         ( "inv_exner"          ,           nz   , ncol );
      real2d dpres             ( "dpres"              ,           nz   , ncol );
      real2d nc_nuceat_tend    ( "nc_nuceat_tend"     ,           nz   , ncol );
      real2d nccn_prescribed   ( "nccn_prescribed"    ,           nz   , ncol );
      real2d ni_activated      ( "ni_activated"       ,           nz   , ncol );
      real2d cld_frac_i        ( "cld_frac_i"         ,           nz   , ncol );
      real2d cld_frac_l        ( "cld_frac_l"         ,           nz   , ncol );
      real2d cld_frac_r        ( "cld_frac_r"         ,           nz   , ncol );
      real2d inv_qc_relvar     ( "inv_qc_relvar"      ,           nz   , ncol );
      real2d col_location      ( "col_location"       ,           3    , ncol );
      real1d precip_liq_surf   ( "precip_liq_surf"    ,                  ncol );
      real1d precip_ice_surf   ( "precip_ice_surf"    ,                  ncol );
      real2d diag_eff_radius_qc( "diag_eff_radius_qc" ,           nz   , ncol );
      real2d diag_eff_radius_qi( "diag_eff_radius_qi" ,           nz   , ncol );
      real2d bulk_qi           ( "bulk_qi"            ,           nz   , ncol );
      real2d mu_c              ( "mu_c"               ,           nz   , ncol );
      real2d lamc              ( "lamc"               ,           nz   , ncol );
      real2d qv2qi_depos_tend  ( "qv2qi_depos_tend"   ,           nz   , ncol );
      real2d precip_total_tend ( "precip_total_tend"  ,           nz   , ncol );
      real2d nevapr            ( "nevapr"             ,           nz   , ncol );
      real2d qr_evap_tend      ( "qr_evap_tend"       ,           nz   , ncol );
      real2d precip_liq_flux   ( "precip_liq_flux"    ,           nz+1 , ncol );
      real2d precip_ice_flux   ( "precip_ice_flux"    ,           nz+1 , ncol );
      real2d liq_ice_exchange  ( "liq_ice_exchange"   ,           nz   , ncol );
      real2d vap_liq_exchange  ( "vap_liq_exchange"   ,           nz   , ncol );
      real2d vap_ice_exchange  ( "vap_ice_exchange"   ,           nz   , ncol );
      real3d p3_tend_out       ( "p3_tend_out"        , p3_nout , nz   , ncol );

      //////////////////////////////////////////////////////////////////////////////
      // Compute quantities needed for inputs to P3
      //////////////////////////////////////////////////////////////////////////////
      // Force constants into local scope
      real R_d     = this->R_d;
      real R_v     = this->R_v;
      real cp_d    = this->cp_d;
      real cp_v    = this->cp_v;
      real cp_l    = this->cp_l;
      real p0      = this->p0;

      YAKL_SCOPE( first_step , this->first_step );
      YAKL_SCOPE( grav       , this->grav       );

      // Save initial state, and compute inputs for p3(...)
      parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
        // Compute total density
        real rho = rho_dry(k,i) + rho_c(k,i) + rho_r(k,i) + rho_i(k,i) + rho_v(k,i);

        compute_adjusted_state( rho, rho_dry(k,i) , rho_v(k,i) , rho_c(k,i) , temp(k,i),
                                R_v , cp_d , cp_v , cp_l );

        // Compute quantities for P3
        qc       (k,i) = rho_c (k,i) / rho_dry(k,i);
        nc       (k,i) = rho_nc(k,i) / rho_dry(k,i);
        qr       (k,i) = rho_r (k,i) / rho_dry(k,i);
        nr       (k,i) = rho_nr(k,i) / rho_dry(k,i);
        qi       (k,i) = rho_i (k,i) / rho_dry(k,i);
        ni       (k,i) = rho_ni(k,i) / rho_dry(k,i);
        qm       (k,i) = rho_m (k,i) / rho_dry(k,i);
        bm       (k,i) = rho_bm(k,i) / rho_dry(k,i);
        qv       (k,i) = rho_v (k,i) / rho_dry(k,i);
        pressure (k,i) = R_d * rho_dry(k,i) * temp(k,i) + R_v * rho_v(k,i) * temp(k,i);
        exner    (k,i) = pow( pressure(k,i) / p0 , R_d / cp_d );
        inv_exner(k,i) = 1. / exner(k,i);
        theta    (k,i) = temp(k,i) / exner(k,i);
        // P3 uses dpres to calculate density via the hydrostatic assumption.
        // So we just reverse this to compute dpres to give true density
        dpres(k,i) = rho_dry(k,i) * grav * dz;
        // nc_nuceat_tend, nccn_prescribed, and ni_activated are not used
        nc_nuceat_tend (k,i) = 0;
        nccn_prescribed(k,i) = 0;
        ni_activated   (k,i) = 0;
        // col_location is for debugging only, and it will be ignored for now
        if (k < 3) { col_location(k,i) = 1; }

        if (first_step) {
          qv_prev(k,i) = qv  (k,i);
          t_prev (k,i) = temp(k,i);
        }
      });

      parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
        // Assume cloud fracton is always 1
        cld_frac_l(k,i) = 1;
        cld_frac_i(k,i) = 1;
        cld_frac_r(k,i) = 1;
        // inv_qc_relvar is always set to one
        inv_qc_relvar(k,i) = 1;
      });
      double elapsed_s;
      int its, ite, kts, kte;
      int it = 1;
      bool do_predict_nc = false;
      bool do_prescribed_CCN = false;

      its = 1;
      ite = ncol;
      kts = 1;
      kte = nz;
      auto qc_host                 = qc                .createHostCopy();
      auto nc_host                 = nc                .createHostCopy();
      auto qr_host                 = qr                .createHostCopy();
      auto nr_host                 = nr                .createHostCopy();
      auto theta_host              = theta             .createHostCopy();
      auto qv_host                 = qv                .createHostCopy();
      auto qi_host                 = qi                .createHostCopy();
      auto qm_host                 = qm                .createHostCopy();
      auto ni_host                 = ni                .createHostCopy();
      auto bm_host                 = bm                .createHostCopy();
      auto pressure_host           = pressure          .createHostCopy();
      auto dz_host                 = dz_arr            .createHostCopy();
      auto nc_nuceat_tend_host     = nc_nuceat_tend    .createHostCopy();
      auto nccn_prescribed_host    = nccn_prescribed   .createHostCopy();
      auto ni_activated_host       = ni_activated      .createHostCopy();
      auto inv_qc_relvar_host      = inv_qc_relvar     .createHostCopy();
      auto precip_liq_surf_host    = precip_liq_surf   .createHostCopy();
      auto precip_ice_surf_host    = precip_ice_surf   .createHostCopy();
      auto diag_eff_radius_qc_host = diag_eff_radius_qc.createHostCopy();
      auto diag_eff_radius_qi_host = diag_eff_radius_qi.createHostCopy();
      auto bulk_qi_host            = bulk_qi           .createHostCopy();
      auto dpres_host              = dpres             .createHostCopy();
      auto inv_exner_host          = inv_exner         .createHostCopy();
      auto qv2qi_depos_tend_host   = qv2qi_depos_tend  .createHostCopy();
      auto precip_total_tend_host  = precip_total_tend .createHostCopy();
      auto nevapr_host             = nevapr            .createHostCopy();
      auto qr_evap_tend_host       = qr_evap_tend      .createHostCopy();
      auto precip_liq_flux_host    = precip_liq_flux   .createHostCopy();
      auto precip_ice_flux_host    = precip_ice_flux   .createHostCopy();
      auto cld_frac_r_host         = cld_frac_r        .createHostCopy();
      auto cld_frac_l_host         = cld_frac_l        .createHostCopy();
      auto cld_frac_i_host         = cld_frac_i        .createHostCopy();
      auto p3_tend_out_host        = p3_tend_out       .createHostCopy();
      auto mu_c_host               = mu_c              .createHostCopy();
      auto lamc_host               = lamc              .createHostCopy();
      auto liq_ice_exchange_host   = liq_ice_exchange  .createHostCopy();
      auto vap_liq_exchange_host   = vap_liq_exchange  .createHostCopy();
      auto vap_ice_exchange_host   = vap_ice_exchange  .createHostCopy();
      auto qv_prev_host            = qv_prev           .createHostCopy();
      auto t_prev_host             = t_prev            .createHostCopy();
      auto col_location_host       = col_location      .createHostCopy();

      p3_main_fortran(qc_host.data() , nc_host.data() , qr_host.data() , nr_host.data() , theta_host.data() ,
                      qv_host.data() , dt , qi_host.data() , qm_host.data() , ni_host.data() , bm_host.data() ,
                      pressure_host.data() , dz_host.data() , nc_nuceat_tend_host.data() ,
                      nccn_prescribed_host.data() , ni_activated_host.data() , inv_qc_relvar_host.data() , it ,
                      precip_liq_surf_host.data() , precip_ice_surf_host.data() , its , ite , kts , kte ,
                      diag_eff_radius_qc_host.data() , diag_eff_radius_qi_host.data() , bulk_qi_host.data() ,
                      do_predict_nc , do_prescribed_CCN , dpres_host.data() , inv_exner_host.data() ,
                      qv2qi_depos_tend_host.data() , precip_total_tend_host.data() , nevapr_host.data() ,
                      qr_evap_tend_host.data() , precip_liq_flux_host.data() , precip_ice_flux_host.data() ,
                      cld_frac_r_host.data() , cld_frac_l_host.data() , cld_frac_i_host.data() ,
                      p3_tend_out_host.data() , mu_c_host.data() , lamc_host.data() , liq_ice_exchange_host.data() ,
                      vap_liq_exchange_host.data() , vap_ice_exchange_host.data() , qv_prev_host.data() ,
                      t_prev_host.data() , col_location_host.data() , &elapsed_s );

      qc_host                .deep_copy_to( qc                 );
      nc_host                .deep_copy_to( nc                 );
      qr_host                .deep_copy_to( qr                 );
      nr_host                .deep_copy_to( nr                 );
      theta_host             .deep_copy_to( theta              );
      qv_host                .deep_copy_to( qv                 );
      qi_host                .deep_copy_to( qi                 );
      qm_host                .deep_copy_to( qm                 );
      ni_host                .deep_copy_to( ni                 );
      bm_host                .deep_copy_to( bm                 );
      pressure_host          .deep_copy_to( pressure           );
      dz_host                .deep_copy_to( dz_arr             );
      nc_nuceat_tend_host    .deep_copy_to( nc_nuceat_tend     );
      nccn_prescribed_host   .deep_copy_to( nccn_prescribed    );
      ni_activated_host      .deep_copy_to( ni_activated       );
      inv_qc_relvar_host     .deep_copy_to( inv_qc_relvar      );
      precip_liq_surf_host   .deep_copy_to( precip_liq_surf    );
      precip_ice_surf_host   .deep_copy_to( precip_ice_surf    );
      diag_eff_radius_qc_host.deep_copy_to( diag_eff_radius_qc );
      diag_eff_radius_qi_host.deep_copy_to( diag_eff_radius_qi );
      bulk_qi_host           .deep_copy_to( bulk_qi            );
      dpres_host             .deep_copy_to( dpres              );
      inv_exner_host         .deep_copy_to( inv_exner          );
      qv2qi_depos_tend_host  .deep_copy_to( qv2qi_depos_tend   );
      precip_total_tend_host .deep_copy_to( precip_total_tend  );
      nevapr_host            .deep_copy_to( nevapr             );
      qr_evap_tend_host      .deep_copy_to( qr_evap_tend       );
      precip_liq_flux_host   .deep_copy_to( precip_liq_flux    );
      precip_ice_flux_host   .deep_copy_to( precip_ice_flux    );
      cld_frac_r_host        .deep_copy_to( cld_frac_r         );
      cld_frac_l_host        .deep_copy_to( cld_frac_l         );
      cld_frac_i_host        .deep_copy_to( cld_frac_i         );
      p3_tend_out_host       .deep_copy_to( p3_tend_out        );
      mu_c_host              .deep_copy_to( mu_c               );
      lamc_host              .deep_copy_to( lamc               );
      liq_ice_exchange_host  .deep_copy_to( liq_ice_exchange   );
      vap_liq_exchange_host  .deep_copy_to( vap_liq_exchange   );
      vap_ice_exchange_host  .deep_copy_to( vap_ice_exchange   );
      qv_prev_host           .deep_copy_to( qv_prev            );
      t_prev_host            .deep_copy_to( t_prev             );
      col_location_host      .deep_copy_to( col_location       );
      
      ///////////////////////////////////////////////////////////////////////////////
      // Convert P3 outputs into dynamics coupler state and tracer masses
      ///////////////////////////////////////////////////////////////////////////////
      parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
        rho_c  (k,i) = std::max( qc(k,i)*rho_dry(k,i) , 0._fp );
        rho_nc (k,i) = std::max( nc(k,i)*rho_dry(k,i) , 0._fp );
        rho_r  (k,i) = std::max( qr(k,i)*rho_dry(k,i) , 0._fp );
        rho_nr (k,i) = std::max( nr(k,i)*rho_dry(k,i) , 0._fp );
        rho_i  (k,i) = std::max( qi(k,i)*rho_dry(k,i) , 0._fp );
        rho_ni (k,i) = std::max( ni(k,i)*rho_dry(k,i) , 0._fp );
        rho_m  (k,i) = std::max( qm(k,i)*rho_dry(k,i) , 0._fp );
        rho_bm (k,i) = std::max( bm(k,i)*rho_dry(k,i) , 0._fp );
        rho_v  (k,i) = std::max( qv(k,i)*rho_dry(k,i) , 0._fp );
        // While micro changes total pressure, thus changing exner, the definition
        // of theta depends on the old exner pressure, so we'll use old exner here
        temp   (k,i) = theta(k,i) * exner(k,i);
        // Save qv and temperature for the next call to p3_main
        qv_prev(k,i) = std::max( qv(k,i) , 0._fp );
        t_prev (k,i) = temp(k,i);
      });

      first_step = false;
      etime += dt;
    }


    // Returns saturation vapor pressure
    YAKL_INLINE static real saturation_vapor_pressure(real temp) {
      real tc = temp - 273.15;
      return 610.94 * exp( 17.625*tc / (243.04+tc) );
    }


    YAKL_INLINE static real latent_heat_condensation(real temp) {
      real tc = temp - 273.15;
      return (2500.8 - 2.36*tc + 0.0016*tc*tc - 0.00006*tc*tc*tc)*1000;
    }


    YAKL_INLINE static real cp_moist(real rho_d, real rho_v, real rho_c, real cp_d, real cp_v, real cp_l) {
      // For the moist specific heat, ignore other species than water vapor and cloud droplets
      real rho = rho_d + rho_v + rho_c;
      return rho_d / rho * cp_d  +  rho_v / rho * cp_v  +  rho_c / rho * cp_l;
    }


    // Compute an instantaneous adjustment of sub or super saturation
    YAKL_INLINE static void compute_adjusted_state(real rho, real rho_d , real &rho_v , real &rho_c , real &temp,
                                                   real R_v , real cp_d , real cp_v , real cp_l) {
      // Define a tolerance for convergence
      real tol = 1.e-6;

      // Saturation vapor pressure at this temperature
      real svp = saturation_vapor_pressure( temp );

      // Vapor pressure at this temperature
      real pv = rho_v * R_v * temp;

      // If we're super-saturated, we need to condense until saturation is reached
      if        (pv > svp) {
        ////////////////////////////////////////////////////////
        // Bisection method
        ////////////////////////////////////////////////////////
        // Set bounds on how much mass to condense
        real cond1  = 0;     // Minimum amount we can condense out
        real cond2 = rho_v;  // Maximum amount we can condense out

        bool keep_iterating = true;
        while (keep_iterating) {
          real rho_cond = (cond1 + cond2) / 2;                    // How much water vapor to condense for this iteration
          real rv_loc = std::max( 0._fp , rho_v - rho_cond );          // New vapor density
          real rc_loc = std::max( 0._fp , rho_c + rho_cond );          // New cloud liquid density
          real Lv = latent_heat_condensation(temp);               // Compute latent heat of condensation
          real cp = cp_moist(rho_d,rv_loc,rc_loc,cp_d,cp_v,cp_l); // New moist specific heat at constant pressure
          real temp_loc = temp + rho_cond*Lv/(rho*cp);            // New temperature after condensation
          real svp_loc = saturation_vapor_pressure(temp_loc);     // New saturation vapor pressure after condensation
          real pv_loc = rv_loc * R_v * temp_loc;                  // New vapor pressure after condensation
          // If we're supersaturated still, we need to condense out more water vapor
          // otherwise, we need to condense out less water vapor
          if (pv_loc > svp_loc) {
            cond1 = rho_cond;
          } else {
            cond2 = rho_cond;
          }
          // If we've converged, then we can stop iterating
          if (abs(cond2-cond1) <= tol) {
            rho_v = rv_loc;
            rho_c = rc_loc;
            temp  = temp_loc;
            keep_iterating = false;
          }
        }

      // If we are unsaturated and have cloud liquid
      } else if (pv < svp && rho_c > 0) {
        // If there's cloud, evaporate enough to achieve saturation
        // or all of it if there isn't enough to reach saturation
        ////////////////////////////////////////////////////////
        // Bisection method
        ////////////////////////////////////////////////////////
        // Set bounds on how much mass to evaporate
        real evap1 = 0;     // minimum amount we can evaporate
        real evap2 = rho_c; // maximum amount we can evaporate

        bool keep_iterating = true;
        while (keep_iterating) {
          real rho_evap = (evap1 + evap2) / 2;                    // How much water vapor to evapense
          real rv_loc = std::max( 0._fp , rho_v + rho_evap );          // New vapor density
          real rc_loc = std::max( 0._fp , rho_c - rho_evap );          // New cloud liquid density
          real Lv = latent_heat_condensation(temp);               // Compute latent heat of condensation for water
          real cp = cp_moist(rho_d,rv_loc,rc_loc,cp_d,cp_v,cp_l); // New moist specific heat
          real temp_loc = temp - rho_evap*Lv/(rho*cp);            // New temperature after evaporation
          real svp_loc = saturation_vapor_pressure(temp_loc);     // New saturation vapor pressure after evaporation
          real pv_loc = rv_loc * R_v * temp_loc;                  // New vapor pressure after evaporation
          // If we're unsaturated still, we need to evaporate out more water vapor
          // otherwise, we need to evaporate out less water vapor
          if (pv_loc < svp_loc) {
            evap1 = rho_evap;
          } else {
            evap2 = rho_evap;
          }
          // If we've converged, then we can stop iterating
          if (abs(evap2-evap1) <= tol) {
            rho_v = rv_loc;
            rho_c = rc_loc;
            temp  = temp_loc;
            keep_iterating = false;
          }
        }
      }
    }


    std::string micro_name() const {
      return "p3";
    }

  };

}


