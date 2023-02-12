
#pragma once

#include "coupler.h"

namespace custom_modules {
  
  inline void perturb( core::Coupler &coupler , std::vector<std::string> varnames , real mag ,
                       int i1 , int i2 , int j1 , int j2 , int k1 , int k2 ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    auto &dm = coupler.get_data_manager_readwrite();

    core::MultiField<real,3> fields;
    for (int i=0; i < varnames.size(); i++) { fields.add_field(dm.get<real,3>(varnames[i])); }

    int num_fields = varnames.size();

    int i_beg   = coupler.get_i_beg();
    int j_beg   = coupler.get_j_beg();
    int nx_glob = coupler.get_nx_glob();
    int ny_glob = coupler.get_ny_glob();
    int nx      = coupler.get_nx();
    int ny      = coupler.get_ny();
    int nz      = coupler.get_nz();
    int nranks  = coupler.get_nranks();
    int myrank  = coupler.get_myrank();

    parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(num_fields,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i) {
      if ( i_beg+i >= i1 && i_beg+i <= i2 &&
           j_beg+j >= j1 && j_beg+j <= j2 &&
           k       >= k1 && k       <= k2 ) {
        yakl::Random rand(myrank*num_fields*nz*ny_glob*nx_glob + 
                                          l*nz*ny_glob*nx_glob +
                                             k*ny_glob*nx_glob +
                                             (j_beg+j)*nx_glob +
                                                       i_beg+i);
        fields(l,k,j,i) += rand.genFP<real>(-1.,1.)*mag;
      }
    });
  }
}


