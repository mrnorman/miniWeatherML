
#pragma once

#include "main_header.h"
#include "coupler.h"
#include "ponni.h"

namespace custom_modules {

  class Microphysics_Kessler {
  public:
    int static constexpr num_tracers = 3;

    real R_d    ;
    real cp_d   ;
    real cv_d   ;
    real gamma_d;
    real kappa_d;
    real R_v    ;
    real cp_v   ;
    real cv_v   ;
    real p0     ;
    real grav   ;

    int static constexpr ID_V = 0;  // Local index for water vapor
    int static constexpr ID_C = 1;  // Local index for cloud liquid
    int static constexpr ID_R = 2;  // Local index for precipitated liquid (rain)

    int static constexpr MAX_LAYERS=10;

    real2d scl_out;
    real3d scl_in ;
    ponni::Sequential<MAX_LAYERS> model;



    Microphysics_Kessler() {
      R_d     = 287.;
      cp_d    = 1003.;
      cv_d    = cp_d - R_d;
      gamma_d = cp_d / cv_d;
      kappa_d = R_d  / cp_d;
      R_v     = 461.;
      cp_v    = 1859;
      cv_v    = R_v - cp_v;
      p0      = 1.e5;
      grav    = 9.81;
    }



    YAKL_INLINE static int get_num_tracers() {
      return num_tracers;
    }



    void init(core::Coupler &coupler) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      int nx = coupler.get_nx();
      int ny = coupler.get_ny();
      int nz = coupler.get_nz();

      // Register tracers in the coupler
      //                 name              description       positive   adds mass
      coupler.add_tracer("water_vapor"   , "Water Vapor"   , true     , true);
      coupler.add_tracer("cloud_liquid"  , "Cloud liquid"  , true     , true);
      coupler.add_tracer("precip_liquid" , "precip_liquid" , true     , true);

      auto &dm = coupler.get_data_manager_readwrite();

      // Register and allocation non-tracer quantities used by the microphysics
      dm.register_and_allocate<real>( "precl" , "precipitation rate" , {ny,nx} , {"y","x"} );

      // Initialize all micro data to zero
      auto rho_v = dm.get<real,3>("water_vapor"  );
      auto rho_c = dm.get<real,3>("cloud_liquid" );
      auto rho_p = dm.get<real,3>("precip_liquid");
      auto precl = dm.get<real,2>("precl"        );

      parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        rho_v(k,j,i) = 0;
        rho_c(k,j,i) = 0;
        rho_p(k,j,i) = 0;
        if (k == 0) precl(j,i) = 0;
      });

      coupler.set_option<std::string>("micro","kessler");

      auto inFile = coupler.get_option<std::string>("standalone_input_file");
      YAML::Node config = YAML::LoadFile(inFile);
      auto keras_weights_h5  = config["keras_weights_h5" ].as<std::string>();
      auto keras_model_json  = config["keras_model_json" ].as<std::string>();
      auto nn_input_scaling  = config["nn_input_scaling" ].as<std::string>();
      auto nn_output_scaling = config["nn_output_scaling"].as<std::string>();

      model = ponni::load_keras_model<MAX_LAYERS>( keras_model_json , keras_weights_h5 );
      model.print_verbose();

      // Load the data scaling arrays
      scl_out = real2d("scl_out",4,2);
      scl_in  = real3d("scl_in" ,4,3,2);

      auto scl_out_host = scl_out.createHostCopy();
      auto scl_in_host  = scl_in .createHostCopy();

      std::ifstream file1;
      // input scaler
      file1.open( nn_input_scaling );
      for (int l = 0; l < 4; l++) {
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 2; j++) {
            file1 >> scl_in_host(l,i,j);
          }
        }
      }
      file1.close();
      file1.open( nn_output_scaling );
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
          file1 >> scl_out_host(i,j);
        }
      }
      file1.close();

      scl_in_host .deep_copy_to(scl_in );
      scl_out_host.deep_copy_to(scl_out);
      yakl::fence();

      std::cout << scl_in ;
      std::cout << "\n";
      std::cout << scl_out;
      std::cout << "\n";
    }



    void time_step( core::Coupler &coupler , real dt ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      auto &dm = coupler.get_data_manager_readwrite();

      // Grab the data
      auto rho_v   = dm.get_lev_col<real      >("water_vapor"  );
      auto rho_c   = dm.get_lev_col<real      >("cloud_liquid" );
      auto rho_r   = dm.get_lev_col<real      >("precip_liquid");
      auto rho_dry = dm.get_lev_col<real const>("density_dry"  );
      auto temp    = dm.get_lev_col<real      >("temp"         );

      // Grab the dimension sizes
      real dz   = coupler.get_dz();
      int  nz   = coupler.get_nz();
      int  ny   = coupler.get_ny();
      int  nx   = coupler.get_nx();
      int  ncol = ny*nx;

      YAKL_SCOPE( scl_out , this->scl_out );
      YAKL_SCOPE( scl_in  , this->scl_in  );

      /////////////////////////////////////////////////////////////////////////
      // NEURAL NETWORK PONNI
      /////////////////////////////////////////////////////////////////////////
      // Build inputs
      int constexpr num_in    = 12;
      int constexpr sten_size = 3 ;
      Array<float,2,memDevice,styleC> ponni_in("ponni_in",num_in,nz*ncol);

      parallel_for( Bounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
        for (int kk=0; kk < sten_size; kk++) {
          int ind_k = std::min(nz-1,std::max(0,k+kk-1));
          int iglob = k*ncol+i;
          ponni_in( 0*sten_size+kk , iglob ) = ( temp (ind_k,i) - scl_in(0,kk,0) ) / ( scl_in(0,kk,1) - scl_in(0,kk,0) );
          ponni_in( 1*sten_size+kk , iglob ) = ( rho_v(ind_k,i) - scl_in(1,kk,0) ) / ( scl_in(1,kk,1) - scl_in(1,kk,0) );
          ponni_in( 2*sten_size+kk , iglob ) = ( rho_c(ind_k,i) - scl_in(2,kk,0) ) / ( scl_in(2,kk,1) - scl_in(2,kk,0) );
          ponni_in( 3*sten_size+kk , iglob ) = ( rho_r(ind_k,i) - scl_in(3,kk,0) ) / ( scl_in(3,kk,1) - scl_in(3,kk,0) );
        }
      });

      auto ponni_out = model.inference_batchparallel( ponni_in );

      real2d temp_tmp  = temp .createDeviceCopy();
      real2d rho_v_tmp = rho_v.createDeviceCopy();
      real2d rho_c_tmp = rho_c.createDeviceCopy();
      real2d rho_r_tmp = rho_r.createDeviceCopy();

      parallel_for( Bounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
        int iglob = k*ncol+i;
        temp_tmp (k,i) =                   ponni_out( 0 , iglob ) * (scl_out(0,1) - scl_out(0,0)) + scl_out(0,0)  ;
        rho_v_tmp(k,i) = std::max( 0._fp , ponni_out( 1 , iglob ) * (scl_out(1,1) - scl_out(1,0)) + scl_out(1,0) );
        rho_c_tmp(k,i) = std::max( 0._fp , ponni_out( 2 , iglob ) * (scl_out(2,1) - scl_out(2,0)) + scl_out(2,0) );
        rho_r_tmp(k,i) = std::max( 0._fp , ponni_out( 3 , iglob ) * (scl_out(3,1) - scl_out(3,0)) + scl_out(3,0) );
      });

      // std::cout << yakl::intrinsics::maxval( temp_tmp  ) << "\n:";
      // std::cout << yakl::intrinsics::maxval( rho_v_tmp ) << "\n:";
      // std::cout << yakl::intrinsics::maxval( rho_c_tmp ) << "\n:";
      // std::cout << yakl::intrinsics::maxval( rho_r_tmp ) << "\n:";

      /////////////////////////////////////////////////////////////////////////
      // NORMAL
      /////////////////////////////////////////////////////////////////////////

      // These are inputs to kessler(...)
      real2d qv      ("qv"      ,nz,ncol);
      real2d qc      ("qc"      ,nz,ncol);
      real2d qr      ("qr"      ,nz,ncol);
      real2d pressure("pressure",nz,ncol);
      real2d theta   ("theta"   ,nz,ncol);
      real2d exner   ("exner"   ,nz,ncol);
      real2d zmid    ("zmid"    ,nz,ncol);

      // Force constants into local scope
      real R_d  = this->R_d;
      real R_v  = this->R_v;
      real cp_d = this->cp_d;
      real p0   = this->p0;

      // Save initial state, and compute inputs for kessler(...)
      parallel_for( "kessler timeStep 2" , Bounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
        zmid    (k,i) = (k+0.5_fp) * dz;
        qv      (k,i) = rho_v(k,i) / rho_dry(k,i);
        qc      (k,i) = rho_c(k,i) / rho_dry(k,i);
        qr      (k,i) = rho_r(k,i) / rho_dry(k,i);
        pressure(k,i) = R_d * rho_dry(k,i) * temp(k,i) + R_v * rho_v(k,i) * temp(k,i);
        exner   (k,i) = pow( pressure(k,i) / p0 , R_d / cp_d );
        theta   (k,i) = temp(k,i) / exner(k,i);
      });

      auto precl = dm.get_collapsed<real>("precl");

      ////////////////////////////////////////////
      // Call Kessler code
      ////////////////////////////////////////////
      kessler(theta, qv, qc, qr, rho_dry, precl, zmid, exner, dt, R_d, cp_d, p0);

      auto rho_v_diff = rho_v.createDeviceCopy();
      auto rho_c_diff = rho_c.createDeviceCopy();
      auto rho_r_diff = rho_r.createDeviceCopy();
      auto temp_diff  = temp .createDeviceCopy();

      // Post-process microphysics changes back to the coupler state
      parallel_for( "kessler timeStep 3" , Bounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
        rho_v   (k,i) = qv(k,i)*rho_dry(k,i);
        rho_c   (k,i) = qc(k,i)*rho_dry(k,i);
        rho_r   (k,i) = qr(k,i)*rho_dry(k,i);
        // While micro changes total pressure, thus changing exner, the definition
        // of theta depends on the old exner pressure, so we'll use old exner here
        temp    (k,i) = theta(k,i) * exner(k,i);

        rho_v_diff(k,i) = rho_v_tmp(k,i) - rho_v(k,i);
        rho_c_diff(k,i) = rho_c_tmp(k,i) - rho_c(k,i);
        rho_r_diff(k,i) = rho_r_tmp(k,i) - rho_r(k,i);
        temp_diff (k,i) = temp_tmp (k,i) - temp (k,i);
      });

      std::cout << "Relative diff rho_v: " << yakl::intrinsics::sum( rho_v_diff ) / ny / nx / nz << "\n";
      std::cout << "Relative diff rho_c: " << yakl::intrinsics::sum( rho_c_diff ) / ny / nx / nz << "\n";
      std::cout << "Relative diff rho_r: " << yakl::intrinsics::sum( rho_r_diff ) / ny / nx / nz << "\n";
      std::cout << "Relative diff temp : " << yakl::intrinsics::sum( temp_diff  ) / ny / nx / nz << "\n";

      temp_tmp .deep_copy_to( temp  );
      rho_v_tmp.deep_copy_to( rho_v );
      rho_c_tmp.deep_copy_to( rho_c );
      rho_r_tmp.deep_copy_to( rho_r );
      yakl::fence();
    }



    ///////////////////////////////////////////////////////////////////////////////
    //
    //  Version:  2.0
    //
    //  Date:  January 22nd, 2015
    //
    //  Change log:
    //  v2 - Added sub-cycling of rain sedimentation so as not to violate
    //       CFL condition.
    //
    //  The KESSLER subroutine implements the Kessler (1969) microphysics
    //  parameterization as described by Soong and Ogura (1973) and Klemp
    //  and Wilhelmson (1978, KW). KESSLER is called at the end of each
    //  time step and makes the final adjustments to the potential
    //  temperature and moisture variables due to microphysical processes
    //  occurring during that time step. KESSLER is called once for each
    //  vertical column of grid cells. Increments are computed and added
    //  into the respective variables. The Kessler scheme contains three
    //  moisture categories: water vapor, cloud water (liquid water that
    //  moves with the flow), and rain water (liquid water that falls
    //  relative to the surrounding air). There  are no ice categories.
    //  
    //  Variables in the column are ordered from the surface to the top.
    //
    //  Parameters:
    //     theta (inout) - dry potential temperature (K)
    //     qv    (inout) - water vapor mixing ratio (gm/gm) (dry mixing ratio)
    //     qc    (inout) - cloud water mixing ratio (gm/gm) (dry mixing ratio)
    //     qr    (inout) - rain  water mixing ratio (gm/gm) (dry mixing ratio)
    //     rho   (in   ) - dry air density (not mean state as in KW) (kg/m^3)
    //     pk    (in   ) - Exner function  (not mean state as in KW) (p/p0)**(R/cp)
    //     dt    (in   ) - time step (s)
    //     z     (in   ) - heights of thermodynamic levels in the grid column (m)
    //     precl (  out) - Precipitation rate (m_water/s)
    //     Rd    (in   ) - Dry air ideal gas constant
    //     cp    (in   ) - Specific heat of dry air at constant pressure
    //     p0    (in   ) - Reference pressure (Pa)
    //
    // Output variables:
    //     Increments are added into t, qv, qc, qr, and precl which are
    //     returned to the routine from which KESSLER was called. To obtain
    //     the total precip qt, after calling the KESSLER routine, compute:
    //
    //       qt = sum over surface grid cells of (precl * cell area)  (kg)
    //       [here, the conversion to kg uses (10^3 kg/m^3)*(10^-3 m/mm) = 1]
    //
    //
    //  Written in Fortran by: Paul Ullrich
    //                         University of California, Davis
    //                         Email: paullrich@ucdavis.edu
    //
    //  Ported to C++ / YAKL by: Matt Norman
    //                           Oak Ridge National Laboratory
    //                           normanmr@ornl.gov
    //                           https://mrnorman.github.io
    //
    //  Based on a code by Joseph Klemp
    //  (National Center for Atmospheric Research)
    //
    //  Reference:
    //
    //    Klemp, J. B., W. C. Skamarock, W. C., and S.-H. Park, 2015:
    //    Idealized Global Nonhydrostatic Atmospheric Test Cases on a Reduced
    //    Radius Sphere. Journal of Advances in Modeling Earth Systems. 
    //    doi:10.1002/2015MS000435
    //
    ///////////////////////////////////////////////////////////////////////////////

    void kessler(real2d const &theta, real2d const &qv, real2d const &qc, real2d const &qr, realConst2d rho,
                 real1d const &precl, realConst2d z, realConst2d pk, real dt, real Rd, real cp, real p0) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      int nz   = theta.dimension[0];
      int ncol = theta.dimension[1];

      // Maximum time step size in accordance with CFL condition
      if (dt <= 0) { endrun("kessler.f90 called with nonpositive dt"); }

      real psl    = p0 / 100;  //  pressure at sea level (mb)
      real rhoqr  = 1000._fp;  //  density of liquid water (kg/m^3)
      real lv     = 2.5e6_fp;  //  latent heat of vaporization (J/kg)

      real2d r    ("r"    ,nz  ,ncol);
      real2d rhalf("rhalf",nz  ,ncol);
      real2d pc   ("pc"   ,nz  ,ncol);
      real2d velqr("velqr",nz  ,ncol);
      real2d dt2d ("dt2d" ,nz-1,ncol);

      parallel_for( "kessler main 1" , Bounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
        r    (k,i) = 0.001_fp * rho(k,i);
        rhalf(k,i) = sqrt( rho(0,i) / rho(k,i) );
        pc   (k,i) = 3.8_fp / ( pow( pk(k,i) , cp/Rd ) * psl );
        // Liquid water terminal velocity (m/s) following KW eq. 2.15
        velqr(k,i) = 36.34_fp * pow( qr(k,i)*r(k,i) , 0.1364_fp ) * rhalf(k,i);
        // Compute maximum stable time step for each cell
        if (k < nz-1) {
          if (velqr(k,i) > 1.e-10_fp) {
            dt2d(k,i) = 0.8_fp * (z(k+1,i)-z(k,i))/velqr(k,i);
          } else {
            dt2d(k,i) = dt;
          }
        }
        // Initialize precip rate to zero
        if (k == 0) {
          precl(i) = 0;
        }
      });

      // Reduce down the minimum time step among the cells
      real dt_max = yakl::intrinsics::minval(dt2d);

      // Number of subcycles
      int rainsplit = ceil(dt / dt_max);
      real dt0 = dt / static_cast<real>(rainsplit);

      real2d sed("sed",nz,ncol);

      // Subcycle through rain process
      for (int nt=0; nt < rainsplit; nt++) {

        // Sedimentation term using upstream differencing
        parallel_for( "kessler main 2" , Bounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
          if (k == 0) {
            // Precipitation rate (m/s)
            precl(i) = precl(i) + rho(0,i) * qr(0,i) * velqr(0,i) / rhoqr;
          }
          if (k == nz-1) {
            sed(nz-1,i) = -dt0*qr(nz-1,i)*velqr(nz-1,i)/(0.5_fp * (z(nz-1,i)-z(nz-2,i)));
          } else {
            sed(k,i) = dt0 * ( r(k+1,i)*qr(k+1,i)*velqr(k+1,i) - 
                               r(k  ,i)*qr(k  ,i)*velqr(k  ,i) ) / ( r(k,i)*(z(k+1,i)-z(k,i)) );
          }
        });

        // Adjustment terms
        parallel_for( "kessler main 3" , Bounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
          // Autoconversion and accretion rates following KW eq. 2.13a,b
          real qrprod = qc(k,i) - ( qc(k,i)-dt0*std::max( 0.001_fp * (qc(k,i)-0.001_fp) , 0._fp ) ) /
                                  ( 1 + dt0 * 2.2_fp * pow( qr(k,i) , 0.875_fp ) );
          qc(k,i) = std::max( qc(k,i)-qrprod , 0._fp );
          qr(k,i) = std::max( qr(k,i)+qrprod+sed(k,i) , 0._fp );

          // Saturation vapor mixing ratio (gm/gm) following KW eq. 2.11
          real tmp = pk(k,i)*theta(k,i)-36._fp;
          real qvs = pc(k,i)*exp( 17.27_fp * (pk(k,i)*theta(k,i)-273._fp) / tmp );
          real prod = (qv(k,i)-qvs) / (1._fp + qvs*(4093._fp * lv/cp)/(tmp*tmp));

          // Evaporation rate following KW eq. 2.14a,b
          real tmp1 = dt0*( ( ( 1.6_fp + 124.9_fp * pow( r(k,i)*qr(k,i) , 0.2046_fp ) ) *
                              pow( r(k,i)*qr(k,i) , 0.525_fp ) ) /
                            ( 2550000._fp * pc(k,i) / (3.8_fp * qvs)+540000._fp) ) * 
                          ( std::max(qvs-qv(k,i),0._fp) / (r(k,i)*qvs) );
          real tmp2 = std::max( -prod-qc(k,i) , 0._fp );
          real tmp3 = qr(k,i);
          real ern = std::min( tmp1 , std::min( tmp2 , tmp3 ) );

          // Saturation adjustment following KW eq. 3.10
          theta(k,i)= theta(k,i) + lv / (cp*pk(k,i)) * 
                                   ( std::max( prod , -qc(k,i) ) - ern );
          qv(k,i) = std::max( qv(k,i) - std::max( prod , -qc(k,i) ) + ern , 0._fp );
          qc(k,i) = qc(k,i) + std::max( prod , -qc(k,i) );
          qr(k,i) = qr(k,i) - ern;

          // Recalculate liquid water terminal velocity
          velqr(k,i)  = 36.34_fp * pow( qr(k,i)*r(k,i) , 0.1364_fp ) * rhalf(k,i);
          if (k == 0 && nt == rainsplit-1) {
            precl(i) = precl(i) / static_cast<real>(rainsplit);
          }
        });

      }

    }


    std::string micro_name() const { return "kessler"; }

  };

}


