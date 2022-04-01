
#pragma once

#include "coupler.h"
#include "YAKL_netcdf.h"
#include <time.h>

namespace custom_modules {

  class DataGenerator {
  public:
    
    void init( core::Coupler &coupler ) {
      yakl::SimpleNetCDF nc;
      nc.create("supercell_micro_surrogate_data.nc");
      nc.createDim("nsamples");
      nc.close();
    }


    void generate_samples( core::Coupler &input , core::Coupler &output , real dt , real etime ) {
      using std::min;
      using std::max;

      auto temp_in   = input .dm.get<real const,3>("temp"         ).createHostCopy();
      auto temp_out  = output.dm.get<real const,3>("temp"         ).createHostCopy();
      auto rho_v_in  = input .dm.get<real const,3>("water_vapor"  ).createHostCopy();
      auto rho_v_out = output.dm.get<real const,3>("water_vapor"  ).createHostCopy();
      auto rho_c_in  = input .dm.get<real const,3>("cloud_liquid" ).createHostCopy();
      auto rho_c_out = output.dm.get<real const,3>("cloud_liquid" ).createHostCopy();
      auto rho_p_in  = input .dm.get<real const,3>("precip_liquid").createHostCopy();
      auto rho_p_out = output.dm.get<real const,3>("precip_liquid").createHostCopy();

      int nx = input.get_nx();
      int ny = input.get_ny();
      int nz = input.get_nz();

      srand(time(nullptr));

      yakl::SimpleNetCDF nc;
      nc.open("supercell_micro_surrogate_data.nc",yakl::NETCDF_MODE_WRITE);
      int ulindex = nc.getDimSize("nsamples");

      for (int k=0; k < nz; k++) {
        for (int j=0; j < ny; j++) {
          for (int i=0; i < nx; i++) {
            real rand_num = (double) rand() / ((double) RAND_MAX);
            if (rand_num < 0.0001_fp) {
              realHost2d input ("input" ,4,3);
              realHost1d output("output",4);
              for (int kk = -1; kk <= 1; kk++) {
                int ind = min(nz,max(0,k+kk));
                input(0,kk+1) = temp_in (ind,j,i);
                input(1,kk+1) = rho_v_in(ind,j,i);
                input(2,kk+1) = rho_c_in(ind,j,i);
                input(3,kk+1) = rho_p_in(ind,j,i);
              }
              output(0) = temp_in (k,j,i);
              output(1) = rho_v_in(k,j,i);
              output(2) = rho_c_in(k,j,i);
              output(3) = rho_p_in(k,j,i);
              nc.write1( input  , "inputs"  , {"num_vars","stencil_size"} , ulindex , "nsamples" );
              nc.write1( output , "outputs" , {"num_vars"               } , ulindex , "nsamples" );
              ulindex++;
            }
          }
        }
      }
      nc.close();
    }

  };

}


