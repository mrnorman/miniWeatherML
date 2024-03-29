
#pragma once

#include "coupler.h"
#include "YAKL_netcdf.h"
#include <time.h>
#include "gather_micro_statistics.h"
#include <fstream>

namespace custom_modules {

  class DataGenerator {
  public:

    std::string fname;
    
    void init( core::Coupler &coupler ) {
      yakl::SimpleNetCDF nc;
      fname = std::string("supercell_kessler_data_task_") + std::to_string(coupler.get_myrank()) + 
              std::string(".nc");
      nc.create(fname);
      nc.createDim("nsamples");
      nc.close();

      std::ofstream myfile;
      myfile.open ("supercell_kessler_metadata.txt");
      myfile << "This dataset contains data for training a surrogate model to emulate Kessler microphysics.\n\n";
      myfile << "vars_in : temperature, dry air density, water vapor density, cloud liquid density, precipitation density\n";
      myfile << "vars_out: temperature, water vapor density, cloud liquid density, precipitation density\n";
      myfile.close();
      
    }


    void generate_samples_stencil( core::Coupler &input , core::Coupler &output , real dt , real etime ) {
      using std::min;
      using std::max;
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      int nx     = input.get_nx    ();
      int ny     = input.get_ny    ();
      int nz     = input.get_nz    ();
      int nranks = input.get_nranks();
      int myrank = input.get_myrank();

      // This was gathered from the gather_statistics.cpp driver
      // On average, about 40% of the cells experience active microphysics at any given time
      double ratio_active = 0.4;
      double expected_num_active   =    ratio_active  * nx*ny*nz;
      double expected_num_inactive = (1-ratio_active) * nx*ny*nz;

      double desired_samples_per_time_step = 50;

      double desired_ratio_active = 0.5;


      double desired_samples_active   =    desired_ratio_active  * desired_samples_per_time_step / nranks;
      double desired_samples_inactive = (1-desired_ratio_active) * desired_samples_per_time_step / nranks;

      double active_threshold   = desired_samples_active   / expected_num_active;
      double inactive_threshold = desired_samples_inactive / expected_num_inactive;

      auto &dm_in  = input .get_data_manager_readonly();
      auto &dm_out = output.get_data_manager_readonly();

      auto rho_d     = dm_in .get<real const,4>("density_dry"  );

      auto temp_in   = dm_in .get<real const,4>("temp"         );
      auto temp_out  = dm_out.get<real const,4>("temp"         );
      auto rho_v_in  = dm_in .get<real const,4>("water_vapor"  );
      auto rho_v_out = dm_out.get<real const,4>("water_vapor"  );
      auto rho_c_in  = dm_in .get<real const,4>("cloud_liquid" );
      auto rho_c_out = dm_out.get<real const,4>("cloud_liquid" );
      auto rho_p_in  = dm_in .get<real const,4>("precip_liquid");
      auto rho_p_out = dm_out.get<real const,4>("precip_liquid");

      bool3d do_sample("do_sample",nz,ny,nx);

      // Seed with time, but make sure the seed magnitude will not lead to an overflow of size_t's max value
      size_t max_mag = std::numeric_limits<size_t>::max() / ( (size_t)nranks + (size_t)nz * (size_t)ny * (size_t)nx );
      size_t seed = ( (size_t) time(nullptr) ) % max_mag;

      parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        double thresh;
        if ( StatisticsGatherer::is_active( temp_in (k,j,i,0) , temp_out (k,j,i,0) ,
                                            rho_v_in(k,j,i,0) , rho_v_out(k,j,i,0) ,
                                            rho_c_in(k,j,i,0) , rho_c_out(k,j,i,0) ,
                                            rho_p_in(k,j,i,0) , rho_p_out(k,j,i,0) ) ) {
          thresh = active_threshold;
        } else {
          thresh = inactive_threshold;
        }

        double rand_num = yakl::Random((seed+myrank)*nz*ny*nx + k*ny*nx + j*nx + i).genFP<double>();
        if (rand_num < thresh) { do_sample(k,j,i) = true;  }
        else                   { do_sample(k,j,i) = false; }
      });

      auto host_rho_d     = rho_d    .createHostCopy();
      auto host_temp_in   = temp_in  .createHostCopy();
      auto host_temp_out  = temp_out .createHostCopy();
      auto host_rho_v_in  = rho_v_in .createHostCopy();
      auto host_rho_v_out = rho_v_out.createHostCopy();
      auto host_rho_c_in  = rho_c_in .createHostCopy();
      auto host_rho_c_out = rho_c_out.createHostCopy();
      auto host_rho_p_in  = rho_p_in .createHostCopy();
      auto host_rho_p_out = rho_p_out.createHostCopy();
      auto host_do_sample = do_sample.createHostCopy();

      yakl::SimpleNetCDF nc;
      nc.open(fname,yakl::NETCDF_MODE_WRITE);
      int ulindex = nc.getDimSize("nsamples");

      if ( ! nc.varExists("time_step_size"     ) ) nc.write(dt                            ,"time_step_size"     );
      if ( ! nc.varExists("only_two_dimensions") ) nc.write(input.get_ny_glob()==1 ? 0 : 1,"only_two_dimensions");
      if ( ! nc.varExists("dx"                 ) ) nc.write(input.get_dx  ()              ,"dx"                 );
      if ( ! nc.varExists("dy"                 ) ) nc.write(input.get_dy  ()              ,"dy"                 );
      if ( ! nc.varExists("dz"                 ) ) nc.write(input.get_dz  ()              ,"dz"                 );
      if ( ! nc.varExists("xlen"               ) ) nc.write(input.get_xlen()              ,"xlen"               );
      if ( ! nc.varExists("ylen"               ) ) nc.write(input.get_ylen()              ,"ylen"               );
      if ( ! nc.varExists("zlen"               ) ) nc.write(input.get_zlen()              ,"zlen"               );

      // Save in 32-bit to reduce file / memory size when training
      floatHost2d gen_input ("gen_input" ,5,2);
      floatHost1d gen_output("gen_output",4);
      
      for (int k=0; k < nz; k++) {
        for (int j=0; j < ny; j++) {
          for (int i=0; i < nx; i++) {
            if (host_do_sample(k,j,i)) {
              gen_input (0,0) = host_temp_in (k,j,i,0);
              gen_input (1,0) = host_rho_d   (k,j,i,0);
              gen_input (2,0) = host_rho_v_in(k,j,i,0);
              gen_input (3,0) = host_rho_c_in(k,j,i,0);
              gen_input (4,0) = host_rho_p_in(k,j,i,0);
              gen_input (0,1) = host_temp_in (std::min(nz-1,k+1),j,i,0);
              gen_input (1,1) = host_rho_v_in(std::min(nz-1,k+1),j,i,0);
              gen_input (2,1) = host_rho_c_in(std::min(nz-1,k+1),j,i,0);
              gen_input (3,1) = host_rho_p_in(std::min(nz-1,k+1),j,i,0);
              gen_output(0) = host_temp_out (k,j,i,0);
              gen_output(1) = host_rho_v_out(k,j,i,0);
              gen_output(2) = host_rho_c_out(k,j,i,0);
              gen_output(3) = host_rho_p_out(k,j,i,0);
              nc.write1( gen_input  , "inputs"  , {"num_vars_in" ,"sten_size"} , ulindex , "nsamples" );
              nc.write1( gen_output , "outputs" , {"num_vars_out"            } , ulindex , "nsamples" );
              ulindex++;
            }
          }
        }
      }
      nc.close();
    }

  };

}


