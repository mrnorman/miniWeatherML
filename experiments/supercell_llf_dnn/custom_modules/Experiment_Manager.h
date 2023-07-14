
#pragma once

#include "coupler.h"
#include "MultipleFields.h"

namespace custom_modules {

  struct Experiment_Manager {
    std::vector<real>              loss_history;
    std::vector<std::vector<real>> parameter_history;


    void overwrite_lo(core::Coupler &coupler_lo, core::Coupler const &coupler_hi) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx_lo           = coupler_lo.get_nx();
      auto ny_lo           = coupler_lo.get_ny();
      auto nz_lo           = coupler_lo.get_nz();
      auto nens_lo         = coupler_lo.get_nens();
      auto sim2d           = coupler_lo.is_sim2d();
      auto num_tracers     = coupler_lo.get_num_tracers();
      auto tracer_names    = coupler_lo.get_tracer_names();
      auto refine_factor   = coupler_hi.get_nx() / nx_lo;
      auto refine_factor_y = sim2d ? 1 : refine_factor;
      auto &dm_lo          = coupler_lo.get_data_manager_readwrite();
      auto &dm_hi          = coupler_hi.get_data_manager_readonly ();

      // Assemble hi-res fields
      core::MultiField<real const,4> fields_hi;
      fields_hi.add_field( dm_hi.get<real const,4>("density_dry") );
      fields_hi.add_field( dm_hi.get<real const,4>("uvel"       ) );
      fields_hi.add_field( dm_hi.get<real const,4>("vvel"       ) );
      fields_hi.add_field( dm_hi.get<real const,4>("wvel"       ) );
      fields_hi.add_field( dm_hi.get<real const,4>("temp"       ) );
      for (int tr=0; tr < num_tracers; tr++) { fields_hi.add_field( dm_hi.get<real const,4>(tracer_names[tr]) ); }

      // Assemble lo-res fields
      core::MultiField<real,4> fields_lo;
      fields_lo.add_field( dm_lo.get<real,4>("density_dry") );
      fields_lo.add_field( dm_lo.get<real,4>("uvel"       ) );
      fields_lo.add_field( dm_lo.get<real,4>("vvel"       ) );
      fields_lo.add_field( dm_lo.get<real,4>("wvel"       ) );
      fields_lo.add_field( dm_lo.get<real,4>("temp"       ) );
      for (int tr=0; tr < num_tracers; tr++) { fields_lo.add_field( dm_lo.get<real,4>(tracer_names[tr]) ); }
      auto num_fields = fields_lo.get_num_fields();

      // Coarsen hi-res data to lo-res grid, and calculate min and max for each column
      real4d fields_hi_coarsened("fields_hi_coarsened",num_fields,nz_lo,ny_lo,nx_lo);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz_lo,ny_lo,nx_lo) ,
                                        YAKL_LAMBDA (int l, int k_lo, int j_lo, int i_lo) {
        int constexpr iens_hi = 0;
        real tmp = 0;
        for (int kk=0; kk < refine_factor; kk++) {
          for (int jj=0; jj < refine_factor_y; jj++) {
            for (int ii=0; ii < refine_factor; ii++) {
              int k_hi = k_lo*refine_factor   + kk;
              int j_hi = j_lo*refine_factor_y + jj;
              int i_hi = i_lo*refine_factor   + ii;
              tmp += fields_hi(l,k_hi,j_hi,i_hi,iens_hi);
            }
          }
        }
        fields_hi_coarsened(l,k_lo,j_lo,i_lo) = tmp / (refine_factor * refine_factor_y * refine_factor);
      });

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_fields,nz_lo,ny_lo,nx_lo,nens_lo) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
        fields_lo(l,k,j,i,iens) = fields_hi_coarsened(l,k,j,i);
      });
    }


    real1d compute_loss_and_overwrite_lo(core::Coupler &coupler_lo, core::Coupler const &coupler_hi) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx_lo           = coupler_lo.get_nx();
      auto ny_lo           = coupler_lo.get_ny();
      auto nz_lo           = coupler_lo.get_nz();
      auto nens_lo         = coupler_lo.get_nens();
      auto sim2d           = coupler_lo.is_sim2d();
      auto num_tracers     = coupler_lo.get_num_tracers();
      auto tracer_names    = coupler_lo.get_tracer_names();
      auto refine_factor   = coupler_hi.get_nx() / nx_lo;
      auto refine_factor_y = sim2d ? 1 : refine_factor;
      auto &dm_lo          = coupler_lo.get_data_manager_readwrite();
      auto &dm_hi          = coupler_hi.get_data_manager_readonly ();

      // Assemble hi-res fields
      core::MultiField<real const,4> fields_hi;
      fields_hi.add_field( dm_hi.get<real const,4>("density_dry") );
      fields_hi.add_field( dm_hi.get<real const,4>("uvel"       ) );
      fields_hi.add_field( dm_hi.get<real const,4>("vvel"       ) );
      fields_hi.add_field( dm_hi.get<real const,4>("wvel"       ) );
      fields_hi.add_field( dm_hi.get<real const,4>("temp"       ) );
      for (int tr=0; tr < num_tracers; tr++) { fields_hi.add_field( dm_hi.get<real const,4>(tracer_names[tr]) ); }

      // Assemble lo-res fields
      core::MultiField<real,4> fields_lo;
      fields_lo.add_field( dm_lo.get<real,4>("density_dry") );
      fields_lo.add_field( dm_lo.get<real,4>("uvel"       ) );
      fields_lo.add_field( dm_lo.get<real,4>("vvel"       ) );
      fields_lo.add_field( dm_lo.get<real,4>("wvel"       ) );
      fields_lo.add_field( dm_lo.get<real,4>("temp"       ) );
      for (int tr=0; tr < num_tracers; tr++) { fields_lo.add_field( dm_lo.get<real,4>(tracer_names[tr]) ); }
      auto num_fields = fields_lo.get_num_fields();

      // Coarsen hi-res data to lo-res grid, and calculate min and max for each column
      real4d fields_hi_coarsened("fields_hi_coarsened",num_fields,nz_lo,ny_lo,nx_lo);
      real2d col_min("col_min",num_fields,nz_lo);
      real2d col_max("col_max",num_fields,nz_lo);
      col_min = std::numeric_limits<real>::max();
      col_max = std::numeric_limits<real>::lowest();
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz_lo,ny_lo,nx_lo) ,
                                        YAKL_LAMBDA (int l, int k_lo, int j_lo, int i_lo) {
        int constexpr iens_hi = 0;
        real tmp = 0;
        for (int kk=0; kk < refine_factor; kk++) {
          for (int jj=0; jj < refine_factor_y; jj++) {
            for (int ii=0; ii < refine_factor; ii++) {
              int k_hi = k_lo*refine_factor   + kk;
              int j_hi = j_lo*refine_factor_y + jj;
              int i_hi = i_lo*refine_factor   + ii;
              tmp += fields_hi(l,k_hi,j_hi,i_hi,iens_hi);
            }
          }
        }
        fields_hi_coarsened(l,k_lo,j_lo,i_lo) = tmp / (refine_factor * refine_factor_y * refine_factor);
        yakl::atomicMin( col_min(l,k_lo) , fields_hi_coarsened(l,k_lo,j_lo,i_lo) );
        yakl::atomicMax( col_max(l,k_lo) , fields_hi_coarsened(l,k_lo,j_lo,i_lo) );
      });

      // Compute loss for each lo-res ensemble, and overwrite the solution for the lo-res ensemble
      real1d loss("loss",nens_lo);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_fields,nz_lo,ny_lo,nx_lo,nens_lo) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
        real normalizer = std::max(1.e-10 , col_max(l,k)-col_min(l,k));
        real diff = (fields_hi_coarsened(l,k,j,i) - fields_lo(l,k,j,i,iens)) / normalizer;
        yakl::atomicAdd( loss(iens) , diff*diff );
        fields_lo(l,k,j,i,iens) = fields_hi_coarsened(l,k,j,i);
      });

      return loss;
    }
  };

}


