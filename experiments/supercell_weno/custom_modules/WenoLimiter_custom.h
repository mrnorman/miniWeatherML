
#pragma once

#include "WenoLimiter_recon_custom.h"


namespace custom_modules {
namespace weno {

  template <int ord> struct WenoLimiter;


  template <> struct WenoLimiter<3> {
    real cutoff, idl_1, idl_3;

    WenoLimiter() = default;
    YAKL_INLINE WenoLimiter(real cutoff ,
                            real mult   ) {
      this->cutoff = cutoff;
      this->idl_1  = 1;
      this->idl_3  = mult;
      convexify( idl_1 , idl_3 );
    }

    YAKL_INLINE void compute_limited_coefs( real s0, real s1, real s2, SArray<real,1,3> &coefs_3 ) const {
      coefs3_shift2( coefs_3 , s0 , s1 , s2 );
      real w_1 = 1.e-4;  // We need something non-zero but small here for this to ever work
      real w_3 = TV( coefs_3 );
      convexify( w_1 , w_3 );
      w_1 = idl_1 / std::max( w_1*w_1 , 1.e-20 );
      w_3 = idl_3 / std::max( w_3*w_3 , 1.e-20 );
      convexify( w_1 , w_3 );
      if (w_1 <= cutoff) w_1 = 0;
      convexify( w_1 , w_3  );
      coefs_3(0) = coefs_3(0)*w_3 + s1*w_1;
      coefs_3(1) = coefs_3(1)*w_3         ;
      coefs_3(2) = coefs_3(2)*w_3;
    }
  };



  template <> struct WenoLimiter<5> {
    real cutoff, idl_3, idl_5;
    WenoLimiter<3> weno3;

    WenoLimiter() = default;
    YAKL_INLINE WenoLimiter(real weno3_cutoff ,
                            real weno3_mult   ,
                            real weno5_cutoff ,
                            real weno5_mult   ) {
      this->weno3 = WenoLimiter<3>(weno3_cutoff,weno3_mult);
      this->cutoff = weno5_cutoff;
      this->idl_3  = 1;
      this->idl_5  = weno5_mult;
      convexify( idl_3 , idl_5 );
    }

    YAKL_INLINE void compute_limited_coefs( real s0, real s1, real s2, real s3, real s4 , SArray<real,1,5> &coefs_5 ) const {
      SArray<real,1,3> coefs_3;
      weno3.compute_limited_coefs( s1 , s2 , s3 , coefs_3 );
      coefs5_shift3( coefs_5 , s0 , s1 , s2 , s3 , s4 );
      real w_3 = std::max( 1.e-4 , TV( coefs_3 ) );
      real w_5 = TV( coefs_5 );
      convexify( w_3 , w_5 );
      w_3 = idl_3 / std::max(w_3*w_3 , 1.e-20);
      w_5 = idl_5 / std::max(w_5*w_5 , 1.e-20);
      convexify( w_3 , w_5 );
      if (w_3 <= cutoff) w_3 = 0;
      convexify( w_3 , w_5 );
      coefs_5(0) = coefs_5(0)*w_5 + coefs_3(0)*w_3;
      coefs_5(1) = coefs_5(1)*w_5 + coefs_3(1)*w_3;
      coefs_5(2) = coefs_5(2)*w_5 + coefs_3(2)*w_3;
      coefs_5(3) = coefs_5(3)*w_5;
      coefs_5(4) = coefs_5(4)*w_5;
    }
  };



  template <> struct WenoLimiter<7> {
    real cutoff, idl_5, idl_7;
    WenoLimiter<5> weno5;

    WenoLimiter() = default;
    YAKL_INLINE WenoLimiter(real weno3_cutoff ,
                            real weno3_mult   ,
                            real weno5_cutoff ,
                            real weno5_mult   ,
                            real weno7_cutoff ,
                            real weno7_mult   ) {
      this->weno5 = WenoLimiter<5>(weno3_cutoff,weno3_mult,weno5_cutoff,weno5_mult);
      this->cutoff = weno7_cutoff;
      this->idl_5  = 1;
      this->idl_7  = weno7_mult;
      convexify( idl_5 , idl_7 );
    }

    YAKL_INLINE void compute_limited_coefs( real s0, real s1, real s2, real s3, real s4, real s5, real s6 ,
                                            SArray<real,1,7> &coefs_7 ) const {
      SArray<real,1,5> coefs_5;
      weno5.compute_limited_coefs( s1 , s2 , s3 , s4 , s5 , coefs_5);
      coefs7( coefs_7 , s0 , s1 , s2 , s3 , s4 , s5 , s6 );
      real w_5 = std::max( 1.e-4 , TV( coefs_5 ) );
      real w_7 = TV( coefs_7 );
      convexify( w_5 , w_7 );
      w_5 = idl_5 / std::max( w_5*w_5 , 1.e-20 );
      w_7 = idl_7 / std::max( w_7*w_7 , 1.e-20 );
      convexify( w_5 , w_7 );
      if (w_5 <= cutoff) w_5 = 0;
      convexify( w_5 , w_7 );
      coefs_7(0) = coefs_7(0)*w_7 + coefs_5(0)*w_5;
      coefs_7(1) = coefs_7(1)*w_7 + coefs_5(1)*w_5;
      coefs_7(2) = coefs_7(2)*w_7 + coefs_5(2)*w_5;
      coefs_7(3) = coefs_7(3)*w_7 + coefs_5(3)*w_5;
      coefs_7(4) = coefs_7(4)*w_7 + coefs_5(4)*w_5;
      coefs_7(5) = coefs_7(5)*w_7;
      coefs_7(6) = coefs_7(6)*w_7;
    }
  };



  template <> struct WenoLimiter<9> {
    real cutoff, idl_7, idl_9;
    WenoLimiter<7> weno7;

    WenoLimiter() = default;
    YAKL_INLINE WenoLimiter(real weno3_cutoff ,
                            real weno3_mult   ,
                            real weno5_cutoff ,
                            real weno5_mult   ,
                            real weno7_cutoff ,
                            real weno7_mult   ,
                            real weno9_cutoff ,
                            real weno9_mult   ) {
      this->weno7 = WenoLimiter<7>(weno3_cutoff,weno3_mult,weno5_cutoff,weno5_mult,weno7_cutoff,weno7_mult);
      this->cutoff = weno9_cutoff;
      this->idl_7  = 1;
      this->idl_9  = weno9_mult;
      convexify( idl_7 , idl_9 );
    }

    YAKL_INLINE void compute_limited_coefs(real s0, real s1, real s2, real s3, real s4, real s5, real s6, real s7, real s8,
                                           SArray<real,1,9> &coefs_9 ) const {
      SArray<real,1,7> coefs_7;
      weno7.compute_limited_coefs( s1 , s2 , s3 , s4 , s5 , s6 , s7 , coefs_7 );
      coefs9( coefs_9 , s0 , s1 , s2 , s3 , s4 , s5 , s6 , s7 , s8 );
      real w_7 = std::max( 1.e-4 , TV( coefs_7 ) );
      real w_9 = TV( coefs_9 );
      convexify( w_7 , w_9 );
      w_7 = idl_7 / (w_7*w_7 + 1.e-20);
      w_9 = idl_9 / (w_9*w_9 + 1.e-20);
      convexify( w_7 , w_9 );
      if (w_7 <= cutoff) w_7 = 0;
      convexify( w_7 , w_9 );
      coefs_9(0) = coefs_9(0)*w_9 + coefs_7(0)*w_7;
      coefs_9(1) = coefs_9(1)*w_9 + coefs_7(1)*w_7;
      coefs_9(2) = coefs_9(2)*w_9 + coefs_7(2)*w_7;
      coefs_9(3) = coefs_9(3)*w_9 + coefs_7(3)*w_7;
      coefs_9(4) = coefs_9(4)*w_9 + coefs_7(4)*w_7;
      coefs_9(5) = coefs_9(5)*w_9 + coefs_7(5)*w_7;
      coefs_9(6) = coefs_9(6)*w_9 + coefs_7(6)*w_7;
      coefs_9(7) = coefs_9(7)*w_9;
      coefs_9(8) = coefs_9(8)*w_9;
    }
  };

}
}


