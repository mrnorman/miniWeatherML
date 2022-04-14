
#pragma once

#include "main_header.h"
#include "coupler.h"
#include "YAKL_netcdf.h"
// ===================Torch================
#include "torch_interface.h"

namespace custom_modules {

  class Microphysics_NN {
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



    Microphysics_NN() {
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
    }



    void time_step( core::Coupler &coupler , real dt , real3d scl_in , real2d scl_out ,
        int devicenum , int mod_id ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      auto &dm = coupler.get_data_manager_readwrite();

      // Grab the data
      auto rho_v   = dm.get_lev_col<real      >("water_vapor"  );
      auto rho_c   = dm.get_lev_col<real      >("cloud_liquid" );
      auto rho_r   = dm.get_lev_col<real      >("precip_liquid");
      auto temp    = dm.get_lev_col<real      >("temp"         );

      ////////////////////////////////////////////
      // Use NN surrogate instead of Kessler
      ////////////////////////////////////////////
      torch_micro(temp, rho_v, rho_c, rho_r, scl_in, scl_out, devicenum, mod_id);

    }

    //////////////////////////////////////////////////////////
    // Machine learned model for (2D) microphysics evolution
    //////////////////////////////////////////////////////////
    void torch_micro(real2d const &temp, real2d const &rho_v, real2d const &rho_c, real2d const &rho_r,
        real3d scl_in , real2d scl_out , int devicenum , int mod_id ) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;
      // Get dimensions of snapshot
      int nz   = temp.dimension[0];
      int ncol = temp.dimension[1];
      // Create a Device Array of inputs (currently bachsize = total grid elements)
      int batchsize = nz*ncol;
      int num_state = 4;
      int num_stencil = 3;
      real3d input("input",batchsize,num_state,num_stencil);
      parallel_for( Bounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
         int ll = k + (i*nz);   // subscript to linear index
         int jj = 0;            // index for stencil
         for (int kk=-1; kk <= 1; kk++) {
           int ind = min(nz-1,max(0,k+kk));
           input(ll,0,jj) = ( temp (ind,i) - scl_in(0,jj,0) ) / ( scl_in(0,jj,1) - scl_in(0,jj,0) );
           input(ll,1,jj) = ( rho_v(ind,i) - scl_in(1,jj,0) ) / ( scl_in(1,jj,1) - scl_in(1,jj,0) );
           input(ll,2,jj) = ( rho_c(ind,i) - scl_in(2,jj,0) ) / ( scl_in(2,jj,1) - scl_in(2,jj,0) );
           input(ll,3,jj) = ( rho_r(ind,i) - scl_in(3,jj,0) ) / ( scl_in(3,jj,1) - scl_in(3,jj,0) );
           jj += 1;
         }
       });

      // Create tensor of inputs
      int tensor_in_id = torch_add_tensor( input.data() , {batchsize,num_state*num_stencil});
      
      // Move tensor of inputs to the GPU
      if (devicenum >= 0) {
        torch_move_tensor_to_gpu(tensor_in_id , devicenum);
      }

      // Execute the model and turn its output into a tensor.
      at::Tensor tensor_output = torch_module_forward( mod_id , {tensor_in_id} );

      // To access the data in the torch tensor, we need to use the function `mytensor.item<datatype>()`
      // PyTorch copies data from device to host during this. 
      // =========Copying torch tensor (GPU) to YAKL device array (does not work)==================
      // real2d outputDevice("outputDevice", batchsize, num_state);
      // std::cout<< "Starting device copy of tensor..." << std::endl;
      // parallel_for( Bounds<2>(batchsize, num_state) , YAKL_LAMBDA (int k, int l) {
      //     // outputDevice(k,l) = tensor_output[k][l].item<real>();     // Does not work as PyTorch copies from device to host
      // });
      // =========================================================================================
      // Thus, we make a host copy of the tensor data and then make a YAKL device array copy.
      // To do this, we copy the torch tensor to cpu and then get the data pointer.
      // We use the data pointer to do std::memcpy

      realHost2d outputHost("outputHost",batchsize,num_state);
      // create cpu tensor & then pointer so that it does not go out of scope
      auto tensorCPU = tensor_output.cpu();               
      auto tensorCPU_ptr = tensorCPU.data_ptr<float>();   // PyTorch needs this to be float!!!!!
      std::memcpy(outputHost.data(), tensorCPU_ptr, sizeof(real)*batchsize*num_state );
      // Create Device copy
      auto outputNN = outputHost.createDeviceCopy();
      
      // Update the microphysics
      parallel_for( Bounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
          int ll = k + (i*nz);   // subscript to linear index

          // NN-based q_fine (re-scale output from NN-model) 
          temp (k,i) = ( outputNN(ll,0) * (scl_out(0,1) - scl_out(0,0)) + scl_out(0,0) );
          rho_v(k,i) = ( outputNN(ll,1) * (scl_out(1,1) - scl_out(1,0)) + scl_out(1,0) );
          rho_c(k,i) = ( outputNN(ll,2) * (scl_out(2,1) - scl_out(2,0)) + scl_out(2,0) );
          rho_r(k,i) = ( outputNN(ll,3) * (scl_out(3,1) - scl_out(3,0)) + scl_out(3,0) );  
      });
    
    }   // end torch_micro()

    std::string micro_name() const { return "NN surrogate for Kessler"; }

  };

}


