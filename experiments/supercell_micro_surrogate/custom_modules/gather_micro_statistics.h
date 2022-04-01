
#pragma once

#include "coupler.h"
#include "YAKL_netcdf.h"
#include <time.h>

namespace custom_modules {

  class StatisticsGatherer {
  public:
    double numer;
    double denom;
    int    num_out;

    StatisticsGatherer() { numer = 0; denom = 0; num_out = 0; }


    void gather_micro_statistics( core::Coupler &input , core::Coupler &output , real dt , real etime ) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;
      using std::min;
      using std::abs;

      auto temp_in   = input .dm.get<real const,3>("temp"         );
      auto temp_out  = output.dm.get<real const,3>("temp"         );
      auto rho_v_in  = input .dm.get<real const,3>("water_vapor"  );
      auto rho_v_out = output.dm.get<real const,3>("water_vapor"  );
      auto rho_c_in  = input .dm.get<real const,3>("cloud_liquid" );
      auto rho_c_out = output.dm.get<real const,3>("cloud_liquid" );
      auto rho_p_in  = input .dm.get<real const,3>("precip_liquid");
      auto rho_p_out = output.dm.get<real const,3>("precip_liquid");

      int nx = input.get_nx();
      int ny = input.get_ny();
      int nz = input.get_nz();

      int3d active("active",nz,ny,nx);

      parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        if ( is_active( temp_in (k,j,i) , temp_out (k,j,i) ,
                        rho_v_in(k,j,i) , rho_v_out(k,j,i) , 
                        rho_c_in(k,j,i) , rho_c_out(k,j,i) , 
                        rho_p_in(k,j,i) , rho_p_out(k,j,i) ) ) {
          active(k,j,i) = 1;
        } else {
          active(k,j,i) = 0;
        }
      });

      if (etime > (num_out+1)*3600) { print(); num_out++; }

      numer += yakl::intrinsics::sum( active );
      denom += nx*ny*nz;
    }


    YAKL_INLINE static bool is_active( real temp_in  , real temp_out  ,
                                real rho_v_in , real rho_v_out ,
                                real rho_c_in , real rho_c_out ,
                                real rho_p_in , real rho_p_out ) {
      real tol = 1.e-10;
      real temp_diff  = std::abs( temp_out  - temp_in  ) / temp_in;
      real rho_v_diff = std::abs( rho_v_out - rho_v_in ) / ( rho_v_in + 1.e-20 );
      real rho_c_diff = std::abs( rho_c_out - rho_c_in ) / ( rho_c_in + 1.e-20 );
      real rho_p_diff = std::abs( rho_p_out - rho_p_in ) / ( rho_p_in + 1.e-20 );
      rho_v_diff = std::min( rho_v_diff , std::abs(rho_v_out - rho_v_in) );
      rho_c_diff = std::min( rho_c_diff , std::abs(rho_c_out - rho_c_in) );
      rho_p_diff = std::min( rho_p_diff , std::abs(rho_p_out - rho_p_in) );
      rho_v_diff = std::min( rho_v_diff , rho_v_in );
      rho_c_diff = std::min( rho_c_diff , rho_c_in );
      rho_p_diff = std::min( rho_p_diff , rho_p_in );

      if (temp_diff > tol && rho_v_diff > tol && rho_c_diff > tol && rho_p_diff > tol) {  return true;  }
      return false;
    }


    void print() {
      std::cout << "*** Ratio Active ***:  " << std::scientific << std::setw(10) << numer / denom << std::endl;
    }


    void finalize( core::Coupler &coupler ) {  print();  }

  };


}


