
#pragma once

#include "main_header.h"
#include "WenoLimiter_recon.h"


namespace weno {

  template <int ord> struct WenoLimiter;



  template <> struct WenoLimiter<3> {
    real cutoff, idl_L, idl_R, idl_H;

    YAKL_INLINE WenoLimiter(real cutoff_in = 0.0,
                            real idl_L_in  = 1,
                            real idl_R_in  = 1,
                            real idl_H_in  = 5.e2) {
      cutoff = cutoff_in;
      idl_L  = idl_L_in;
      idl_R  = idl_R_in;
      idl_H  = idl_H_in;
      convexify( idl_L , idl_R , idl_H );
    }

    YAKL_INLINE void compute_limited_coefs( SArray<real,1,3> const &s , SArray<real,1,3> &coefs_H ) const {
      SArray<real,1,2> coefs_L, coefs_R;
      coefs2_shift1( coefs_L , s(0) , s(1) );
      coefs2_shift2( coefs_R , s(1) , s(2) );
      coefs3_shift2( coefs_H , s(0) , s(1) , s(2) );
      real w_L = TV( coefs_L );
      real w_R = TV( coefs_R );
      real w_H = TV( coefs_H );
      convexify( w_L , w_R , w_H );
      w_L = idl_L / (w_L*w_L + 1.e-20);
      w_R = idl_R / (w_R*w_R + 1.e-20);
      w_H = idl_H / (w_H*w_H + 1.e-20);
      convexify( w_L , w_R , w_H );
      if (w_L <= cutoff) w_L = 0;
      if (w_R <= cutoff) w_R = 0;
      convexify( w_L , w_R , w_H );
      coefs_H(0) = coefs_H(0)*w_H + coefs_L(0)*w_L + coefs_R(0)*w_R;
      coefs_H(1) = coefs_H(1)*w_H + coefs_L(1)*w_L + coefs_R(1)*w_R;
      coefs_H(2) = coefs_H(2)*w_H;
    }
  };



  template <> struct WenoLimiter<5> {
    real cutoff, idl_L, idl_C, idl_R, idl_H;

    YAKL_INLINE WenoLimiter(real cutoff_in = 0,
                            real idl_L_in  = 1,
                            real idl_C_in  = 2,
                            real idl_R_in  = 1,
                            real idl_H_in  = 1.e3) {
      cutoff = cutoff_in;
      idl_L  = idl_L_in;
      idl_C  = idl_C_in;
      idl_R  = idl_R_in;
      idl_H  = idl_H_in;
      convexify( idl_L , idl_C , idl_R , idl_H );
    }

    YAKL_INLINE void compute_limited_coefs( SArray<real,1,5> const &s , SArray<real,1,5> &coefs_H ) const {
      SArray<real,1,3> coefs_L, coefs_C, coefs_R;
      coefs3_shift1( coefs_L , s(0) , s(1) , s(2) );
      coefs3_shift2( coefs_C , s(1) , s(2) , s(3) );
      coefs3_shift3( coefs_R , s(2) , s(3) , s(4) );
      coefs5_shift3( coefs_H , s(0) , s(1) , s(2) , s(3) , s(4) );
      real w_L = TV( coefs_L );
      real w_C = TV( coefs_C );
      real w_R = TV( coefs_R );
      real w_H = TV( coefs_H );
      convexify( w_L , w_C , w_R , w_H );
      w_L = idl_L / (w_L*w_L + 1.e-20);
      w_C = idl_C / (w_C*w_C + 1.e-20);
      w_R = idl_R / (w_R*w_R + 1.e-20);
      w_H = idl_H / (w_H*w_H + 1.e-20);
      convexify( w_L , w_C , w_R , w_H );
      if (w_L <= cutoff) w_L = 0;
      if (w_C <= cutoff) w_C = 0;
      if (w_R <= cutoff) w_R = 0;
      convexify( w_L , w_C , w_R , w_H );
      coefs_H(0) = coefs_H(0)*w_H + coefs_L(0)*w_L + coefs_C(0)*w_C + coefs_R(0)*w_R;
      coefs_H(1) = coefs_H(1)*w_H + coefs_L(1)*w_L + coefs_C(1)*w_C + coefs_R(1)*w_R;
      coefs_H(2) = coefs_H(2)*w_H + coefs_L(2)*w_L + coefs_C(2)*w_C + coefs_R(2)*w_R;
      coefs_H(3) = coefs_H(3)*w_H;
      coefs_H(4) = coefs_H(4)*w_H;
    }
  };



  template <> struct WenoLimiter<7> {
    real cutoff, idl_L, idl_C, idl_R, idl_H;

    YAKL_INLINE WenoLimiter(real cutoff_in = 0,
                            real idl_L_in  = 1,
                            real idl_C_in  = 2,
                            real idl_R_in  = 1,
                            real idl_H_in  = 1.e5) {
      cutoff = cutoff_in;
      idl_L  = idl_L_in;
      idl_C  = idl_C_in;
      idl_R  = idl_R_in;
      idl_H  = idl_H_in;
      convexify( idl_L , idl_C , idl_R , idl_H );
    }

    YAKL_INLINE void compute_limited_coefs( SArray<real,1,7> const &s , SArray<real,1,7> &coefs_H ) const {
      SArray<real,1,3> coefs_L, coefs_C, coefs_R;
      coefs3_shift1( coefs_L , s(1) , s(2) , s(3) );
      coefs3_shift2( coefs_C , s(2) , s(3) , s(4) );
      coefs3_shift3( coefs_R , s(3) , s(4) , s(5) );
      coefs7       ( coefs_H , s(0) , s(1) , s(2) , s(3) , s(4) , s(5) , s(6) );
      real w_L = TV( coefs_L );
      real w_C = TV( coefs_C );
      real w_R = TV( coefs_R );
      real w_H = TV( coefs_H );
      convexify( w_L , w_C , w_R , w_H );
      w_L = idl_L / (w_L*w_L + 1.e-20);
      w_C = idl_C / (w_C*w_C + 1.e-20);
      w_R = idl_R / (w_R*w_R + 1.e-20);
      w_H = idl_H / (w_H*w_H + 1.e-20);
      convexify( w_L , w_C , w_R , w_H );
      if (w_L <= cutoff) w_L = 0;
      if (w_C <= cutoff) w_C = 0;
      if (w_R <= cutoff) w_R = 0;
      convexify( w_L , w_C , w_R , w_H );
      coefs_H(0) = coefs_H(0)*w_H + coefs_L(0)*w_L + coefs_C(0)*w_C + coefs_R(0)*w_R;
      coefs_H(1) = coefs_H(1)*w_H + coefs_L(1)*w_L + coefs_C(1)*w_C + coefs_R(1)*w_R;
      coefs_H(2) = coefs_H(2)*w_H + coefs_L(2)*w_L + coefs_C(2)*w_C + coefs_R(2)*w_R;
      coefs_H(3) = coefs_H(3)*w_H;
      coefs_H(4) = coefs_H(4)*w_H;
      coefs_H(5) = coefs_H(5)*w_H;
      coefs_H(6) = coefs_H(6)*w_H;
    }
  };



  template <> struct WenoLimiter<9> {
    real cutoff, idl_L, idl_C, idl_R, idl_H;

    YAKL_INLINE WenoLimiter(real cutoff_in = 0,
                            real idl_L_in  = 1,
                            real idl_C_in  = 2,
                            real idl_R_in  = 1,
                            real idl_H_in  = 1.e8) {
      cutoff = cutoff_in;
      idl_L  = idl_L_in;
      idl_C  = idl_C_in;
      idl_R  = idl_R_in;
      idl_H  = idl_H_in;
      convexify( idl_L , idl_C , idl_R , idl_H );
    }

    YAKL_INLINE void compute_limited_coefs( SArray<real,1,9> const &s , SArray<real,1,9> &coefs_H ) const {
      SArray<real,1,3> coefs_L, coefs_C, coefs_R;
      coefs3_shift1( coefs_L , s(2) , s(3) , s(4) );
      coefs3_shift2( coefs_C , s(3) , s(4) , s(5) );
      coefs3_shift3( coefs_R , s(4) , s(5) , s(6) );
      coefs9       ( coefs_H , s(0) , s(1) , s(2) , s(3) , s(4) , s(5) , s(6) , s(7) , s(8) );
      real w_L = TV( coefs_L );
      real w_C = TV( coefs_C );
      real w_R = TV( coefs_R );
      real w_H = TV( coefs_H );
      convexify( w_L , w_C , w_R , w_H );
      w_L = idl_L / (w_L*w_L + 1.e-20);
      w_C = idl_C / (w_C*w_C + 1.e-20);
      w_R = idl_R / (w_R*w_R + 1.e-20);
      w_H = idl_H / (w_H*w_H + 1.e-20);
      convexify( w_L , w_C , w_R , w_H );
      if (w_L <= cutoff) w_L = 0;
      if (w_C <= cutoff) w_C = 0;
      if (w_R <= cutoff) w_R = 0;
      convexify( w_L , w_C , w_R , w_H );
      coefs_H(0) = coefs_H(0)*w_H + coefs_L(0)*w_L + coefs_C(0)*w_C + coefs_R(0)*w_R;
      coefs_H(1) = coefs_H(1)*w_H + coefs_L(1)*w_L + coefs_C(1)*w_C + coefs_R(1)*w_R;
      coefs_H(2) = coefs_H(2)*w_H + coefs_L(2)*w_L + coefs_C(2)*w_C + coefs_R(2)*w_R;
      coefs_H(3) = coefs_H(3)*w_H;
      coefs_H(4) = coefs_H(4)*w_H;
      coefs_H(5) = coefs_H(5)*w_H;
      coefs_H(6) = coefs_H(6)*w_H;
      coefs_H(7) = coefs_H(7)*w_H;
      coefs_H(8) = coefs_H(8)*w_H;
    }
  };

}


