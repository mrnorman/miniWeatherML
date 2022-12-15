
#pragma once

#include "main_header.h"

YAKL_INLINE void riemann_central_x( real5d const &state_limits_x , real5d const &tracers_limits_x ,
                                    int k , int j , int i , int num_tracers, real C0, real gamma,
                                    real &ru_upw , real &r_upw , real &u_upw , real &v_upw , real &w_upw , 
                                    real &theta_upw , real &p_upw , real4d const &tracers_upw ) {
  int  static constexpr idR = 0;  // Density
  int  static constexpr idU = 1;  // u-momentum
  int  static constexpr idV = 2;  // v-momentum
  int  static constexpr idW = 3;  // w-momentum
  int  static constexpr idT = 4;  // Density * potential temperature
  r_upw     = 0.5_fp * ( state_limits_x(idR,0,k,j,i) + state_limits_x(idR,1,k,j,i) );
  u_upw     = 0.5_fp * ( state_limits_x(idU,0,k,j,i) + state_limits_x(idU,1,k,j,i) ) / r_upw;
  v_upw     = 0.5_fp * ( state_limits_x(idV,0,k,j,i) + state_limits_x(idV,1,k,j,i) ) / r_upw;
  w_upw     = 0.5_fp * ( state_limits_x(idW,0,k,j,i) + state_limits_x(idW,1,k,j,i) ) / r_upw;
  theta_upw = 0.5_fp * ( state_limits_x(idT,0,k,j,i) + state_limits_x(idT,1,k,j,i) ) / r_upw;
  p_upw     = C0 * pow( r_upw * theta_upw , gamma );
  ru_upw    = r_upw * u_upw;
  for (int tr=0; tr < num_tracers; tr++) {
    tracers_upw(tr,k,j,i) = 0.5_fp * ( tracers_limits_x(tr,0,k,j,i) + tracers_limits_x(tr,1,k,j,i) ) / r_upw;
  }
}

YAKL_INLINE void riemann_central_y( real5d const &state_limits_y , real5d const &tracers_limits_y ,
                                    int k , int j , int i , int num_tracers, real C0, real gamma,
                                    real &rv_upw , real &r_upw , real &u_upw , real &v_upw , real &w_upw , 
                                    real &theta_upw , real &p_upw , real4d const &tracers_upw ) {
  int  static constexpr idR = 0;  // Density
  int  static constexpr idU = 1;  // u-momentum
  int  static constexpr idV = 2;  // v-momentum
  int  static constexpr idW = 3;  // w-momentum
  int  static constexpr idT = 4;  // Density * potential temperature
  r_upw     = 0.5_fp * ( state_limits_y(idR,0,k,j,i) + state_limits_y(idR,1,k,j,i) );
  u_upw     = 0.5_fp * ( state_limits_y(idU,0,k,j,i) + state_limits_y(idU,1,k,j,i) ) / r_upw;
  v_upw     = 0.5_fp * ( state_limits_y(idV,0,k,j,i) + state_limits_y(idV,1,k,j,i) ) / r_upw;
  w_upw     = 0.5_fp * ( state_limits_y(idW,0,k,j,i) + state_limits_y(idW,1,k,j,i) ) / r_upw;
  theta_upw = 0.5_fp * ( state_limits_y(idT,0,k,j,i) + state_limits_y(idT,1,k,j,i) ) / r_upw;
  p_upw     = C0 * pow( r_upw * theta_upw , gamma );
  rv_upw    = r_upw * v_upw;
  for (int tr=0; tr < num_tracers; tr++) {
    tracers_upw(tr,k,j,i) = 0.5_fp * ( tracers_limits_y(tr,0,k,j,i) + tracers_limits_y(tr,1,k,j,i) ) / r_upw;
  }
}

YAKL_INLINE void riemann_central_z( real5d const &state_limits_z , real5d const &tracers_limits_z ,
                                    int k , int j , int i , int num_tracers, real C0, real gamma,
                                    real &rw_upw , real &r_upw , real &u_upw , real &v_upw , real &w_upw , 
                                    real &theta_upw , real &p_upw , real4d const &tracers_upw ) {
  int  static constexpr idR = 0;  // Density
  int  static constexpr idU = 1;  // u-momentum
  int  static constexpr idV = 2;  // v-momentum
  int  static constexpr idW = 3;  // w-momentum
  int  static constexpr idT = 4;  // Density * potential temperature
  r_upw     = 0.5_fp * ( state_limits_z(idR,0,k,j,i) + state_limits_z(idR,1,k,j,i) );
  u_upw     = 0.5_fp * ( state_limits_z(idU,0,k,j,i) + state_limits_z(idU,1,k,j,i) ) / r_upw;
  v_upw     = 0.5_fp * ( state_limits_z(idV,0,k,j,i) + state_limits_z(idV,1,k,j,i) ) / r_upw;
  w_upw     = 0.5_fp * ( state_limits_z(idW,0,k,j,i) + state_limits_z(idW,1,k,j,i) ) / r_upw;
  theta_upw = 0.5_fp * ( state_limits_z(idT,0,k,j,i) + state_limits_z(idT,1,k,j,i) ) / r_upw;
  p_upw     = C0 * pow( r_upw * theta_upw , gamma );
  rw_upw    = r_upw * w_upw;
  for (int tr=0; tr < num_tracers; tr++) {
    tracers_upw(tr,k,j,i) = 0.5_fp * ( tracers_limits_z(tr,0,k,j,i) + tracers_limits_z(tr,1,k,j,i) ) / r_upw;
  }
}




YAKL_INLINE void riemann_advec_x( real5d const &state_limits_x , real5d const &tracers_limits_x ,
                                  int k , int j , int i , int num_tracers, real C0, real gamma,
                                  real &ru_upw , real &r_upw , real &u_upw , real &v_upw , real &w_upw , 
                                  real &theta_upw , real &p_upw , real4d const &tracers_upw ) {
  int  static constexpr idR = 0;  // Density
  int  static constexpr idU = 1;  // u-momentum
  int  static constexpr idV = 2;  // v-momentum
  int  static constexpr idW = 3;  // w-momentum
  int  static constexpr idT = 4;  // Density * potential temperature
  real ru2 = state_limits_x(idU,0,k,j,i) + state_limits_x(idU,1,k,j,i);
  if (ru2 == 0) {
    riemann_central_x(state_limits_x,tracers_limits_x,k,j,i,num_tracers,C0,gamma,
                      ru_upw,r_upw,u_upw,v_upw,w_upw,theta_upw,p_upw,tracers_upw);
    return;
  }
  int ind = ru2 > 0 ? 0 : 1;
  r_upw     = state_limits_x(idR,ind,k,j,i);
  u_upw     = state_limits_x(idU,ind,k,j,i) / r_upw;
  v_upw     = state_limits_x(idV,ind,k,j,i) / r_upw;
  w_upw     = state_limits_x(idW,ind,k,j,i) / r_upw;
  theta_upw = state_limits_x(idT,ind,k,j,i) / r_upw;
  p_upw     = C0 * pow( r_upw * theta_upw , gamma );
  ru_upw    = r_upw * u_upw;
  for (int tr=0; tr < num_tracers; tr++) { tracers_upw(tr,k,j,i) = tracers_limits_x(tr,ind,k,j,i) / r_upw; }
}

YAKL_INLINE void riemann_advec_y( real5d const &state_limits_y , real5d const &tracers_limits_y ,
                                  int k , int j , int i , int num_tracers, real C0, real gamma,
                                  real &rv_upw , real &r_upw , real &u_upw , real &v_upw , real &w_upw , 
                                  real &theta_upw , real &p_upw , real4d const &tracers_upw ) {
  int  static constexpr idR = 0;  // Density
  int  static constexpr idU = 1;  // u-momentum
  int  static constexpr idV = 2;  // v-momentum
  int  static constexpr idW = 3;  // w-momentum
  int  static constexpr idT = 4;  // Density * potential temperature
  real rv2 = state_limits_y(idV,0,k,j,i) + state_limits_y(idV,1,k,j,i);
  if (rv2 == 0) {
    riemann_central_y(state_limits_y,tracers_limits_y,k,j,i,num_tracers,C0,gamma,
                      rv_upw,r_upw,u_upw,v_upw,w_upw,theta_upw,p_upw,tracers_upw);
    return;
  }
  int ind = rv2 > 0 ? 0 : 1;
  r_upw     = state_limits_y(idR,ind,k,j,i);
  u_upw     = state_limits_y(idU,ind,k,j,i) / r_upw;
  v_upw     = state_limits_y(idV,ind,k,j,i) / r_upw;
  w_upw     = state_limits_y(idW,ind,k,j,i) / r_upw;
  theta_upw = state_limits_y(idT,ind,k,j,i) / r_upw;
  p_upw     = C0 * pow( r_upw * theta_upw , gamma );
  rv_upw    = r_upw * v_upw;
  for (int tr=0; tr < num_tracers; tr++) { tracers_upw(tr,k,j,i) = tracers_limits_y(tr,ind,k,j,i) / r_upw; }
}

YAKL_INLINE void riemann_advec_z( real5d const &state_limits_z , real5d const &tracers_limits_z ,
                                  int k , int j , int i , int num_tracers, real C0, real gamma,
                                  real &rw_upw , real &r_upw , real &u_upw , real &v_upw , real &w_upw , 
                                  real &theta_upw , real &p_upw , real4d const &tracers_upw ) {
  int  static constexpr idR = 0;  // Density
  int  static constexpr idU = 1;  // u-momentum
  int  static constexpr idV = 2;  // v-momentum
  int  static constexpr idW = 3;  // w-momentum
  int  static constexpr idT = 4;  // Density * potential temperature
  real rw2 = state_limits_z(idV,0,k,j,i) + state_limits_z(idV,1,k,j,i);
  if (rw2 == 0) {
    riemann_central_z(state_limits_z,tracers_limits_z,k,j,i,num_tracers,C0,gamma,
                      rw_upw,r_upw,u_upw,v_upw,w_upw,theta_upw,p_upw,tracers_upw);
    return;
  }
  int ind = rw2 > 0 ? 0 : 1;
  r_upw     = state_limits_z(idR,ind,k,j,i);
  u_upw     = state_limits_z(idU,ind,k,j,i) / r_upw;
  v_upw     = state_limits_z(idV,ind,k,j,i) / r_upw;
  w_upw     = state_limits_z(idW,ind,k,j,i) / r_upw;
  theta_upw = state_limits_z(idT,ind,k,j,i) / r_upw;
  p_upw     = C0 * pow( r_upw * theta_upw , gamma );
  rw_upw    = r_upw * w_upw;
  for (int tr=0; tr < num_tracers; tr++) { tracers_upw(tr,k,j,i) = tracers_limits_z(tr,ind,k,j,i) / r_upw; }
}




YAKL_INLINE void riemann_native_x( real5d const &state_limits_x , real5d const &tracers_limits_x ,
                                   int k , int j , int i , int num_tracers, real C0, real gamma,
                                   real &ru_upw , real &r_upw , real &u_upw , real &v_upw , real &w_upw , 
                                   real &theta_upw , real &p_upw , real4d const &tracers_upw ) {
  int  static constexpr idR = 0;  // Density
  int  static constexpr idU = 1;  // u-momentum
  int  static constexpr idV = 2;  // v-momentum
  int  static constexpr idW = 3;  // w-momentum
  int  static constexpr idT = 4;  // Density * potential temperature
  // Get left and right state
  real r_L = state_limits_x(idR,0,k,j,i)    ;   real r_R = state_limits_x(idR,1,k,j,i)    ;
  real u_L = state_limits_x(idU,0,k,j,i)/r_L;   real u_R = state_limits_x(idU,1,k,j,i)/r_R;
  real v_L = state_limits_x(idV,0,k,j,i)/r_L;   real v_R = state_limits_x(idV,1,k,j,i)/r_R;
  real w_L = state_limits_x(idW,0,k,j,i)/r_L;   real w_R = state_limits_x(idW,1,k,j,i)/r_R;
  real t_L = state_limits_x(idT,0,k,j,i)/r_L;   real t_R = state_limits_x(idT,1,k,j,i)/r_R;
  // Compute average state
  real r = 0.5_fp * (r_L + r_R);
  real u = 0.5_fp * (u_L + u_R);
  real v = 0.5_fp * (v_L + v_R);
  real w = 0.5_fp * (w_L + w_R);
  real t = 0.5_fp * (t_L + t_R);
  real p = C0 * pow(r*t,gamma);
  real cs2 = gamma*p/r;
  real cs  = sqrt(cs2);
  // Get left and right state
  real q1_L = r_L    ;   real q1_R = r_R    ;
  real q2_L = r_L*u_L;   real q2_R = r_R*u_R;
  real q3_L = r_L*v_L;   real q3_R = r_R*v_R;
  real q4_L = r_L*w_L;   real q4_R = r_R*w_R;
  real q5_L = r_L*t_L;   real q5_R = r_R*t_R;
  // Compute upwind characteristics
  // Waves 1-3, velocity: u
  real w1, w2, w3;
  if (u > 0) {
    w1 = q1_L - q5_L/t;
    w2 = q3_L - v*q5_L/t;
    w3 = q4_L - w*q5_L/t;
  } else if (u < 0) {
    w1 = q1_R - q5_R/t;
    w2 = q3_R - v*q5_R/t;
    w3 = q4_R - w*q5_R/t;
  } else {
    w1 = 0.5_fp * ( (q1_R - q5_R/t  ) + (q1_L - q5_L/t  ) );
    w2 = 0.5_fp * ( (q3_R - v*q5_R/t) + (q3_L - v*q5_L/t) );
    w3 = 0.5_fp * ( (q4_R - w*q5_R/t) + (q4_L - w*q5_L/t) );
  }
  // Wave 5, velocity: u-cs
  real w5 =  u*q1_R/(2*cs) - q2_R/(2*cs) + q5_R/(2*t);
  // Wave 6, velocity: u+cs
  real w6 = -u*q1_L/(2*cs) + q2_L/(2*cs) + q5_L/(2*t);
  // Use right eigenmatrix to compute upwind state
  real q1 = w1 + w5 + w6;
  real q2 = u*w1 + (u-cs)*w5 + (u+cs)*w6;
  real q3 = w2 + v*w5 + v*w6;
  real q4 = w3 + w*w5 + w*w6;
  real q5 =      t*w5 + t*w6;

  r_upw     = q1;
  u_upw     = q2 / q1;
  v_upw     = q3 / q1;
  w_upw     = q4 / q1;
  theta_upw = q5 / q1;
  p_upw     = C0 * pow( r_upw * theta_upw , gamma );
  ru_upw    = r_upw * u_upw;
  for (int tr=0; tr < num_tracers; tr++) {
    if        (u > 0) {
      tracers_upw(tr,k,j,i) = tracers_limits_x(tr,0,k,j,i) / state_limits_x(idR,0,k,j,i);
    } else if (u < 0) { 
      tracers_upw(tr,k,j,i) = tracers_limits_x(tr,1,k,j,i) / state_limits_x(idR,1,k,j,i);
    } else {
      tracers_upw(tr,k,j,i) = 0.5_fp * ( tracers_limits_x(tr,0,k,j,i) / state_limits_x(idR,0,k,j,i) +
                                         tracers_limits_x(tr,1,k,j,i) / state_limits_x(idR,1,k,j,i) );
    }
  }
}

YAKL_INLINE void riemann_native_y( real5d const &state_limits_y , real5d const &tracers_limits_y ,
                                   int k , int j , int i , int num_tracers, real C0, real gamma,
                                   real &rv_upw , real &r_upw , real &u_upw , real &v_upw , real &w_upw , 
                                   real &theta_upw , real &p_upw , real4d const &tracers_upw ) {
  int  static constexpr idR = 0;  // Density
  int  static constexpr idU = 1;  // u-momentum
  int  static constexpr idV = 2;  // v-momentum
  int  static constexpr idW = 3;  // w-momentum
  int  static constexpr idT = 4;  // Density * potential temperature
  // Get left and right state
  real r_L = state_limits_y(idR,0,k,j,i)    ;   real r_R = state_limits_y(idR,1,k,j,i)    ;
  real u_L = state_limits_y(idU,0,k,j,i)/r_L;   real u_R = state_limits_y(idU,1,k,j,i)/r_R;
  real v_L = state_limits_y(idV,0,k,j,i)/r_L;   real v_R = state_limits_y(idV,1,k,j,i)/r_R;
  real w_L = state_limits_y(idW,0,k,j,i)/r_L;   real w_R = state_limits_y(idW,1,k,j,i)/r_R;
  real t_L = state_limits_y(idT,0,k,j,i)/r_L;   real t_R = state_limits_y(idT,1,k,j,i)/r_R;
  // Compute average state
  real r = 0.5_fp * (r_L + r_R);
  real u = 0.5_fp * (u_L + u_R);
  real v = 0.5_fp * (v_L + v_R);
  real w = 0.5_fp * (w_L + w_R);
  real t = 0.5_fp * (t_L + t_R);
  real p = C0 * pow(r*t,gamma);
  real cs2 = gamma*p/r;
  real cs  = sqrt(cs2);
  // Get left and right state
  real q1_L = r_L    ;   real q1_R = r_R    ;
  real q2_L = r_L*u_L;   real q2_R = r_R*u_R;
  real q3_L = r_L*v_L;   real q3_R = r_R*v_R;
  real q4_L = r_L*w_L;   real q4_R = r_R*w_R;
  real q5_L = r_L*t_L;   real q5_R = r_R*t_R;
  // Compute upwind characteristics
  // Waves 1-3, velocity: v
  real w1, w2, w3;
  if (v > 0) {
    w1 = q1_L - q5_L/t;
    w2 = q2_L - u*q5_L/t;
    w3 = q4_L - w*q5_L/t;
  } else if (v < 0) {
    w1 = q1_R - q5_R/t;
    w2 = q2_R - u*q5_R/t;
    w3 = q4_R - w*q5_R/t;
  } else {
    w1 = 0.5_fp * ( (q1_R - q5_R/t  ) + (q1_L - q5_L/t  ) );
    w2 = 0.5_fp * ( (q2_R - u*q5_R/t) + (q2_L - u*q5_L/t) );
    w3 = 0.5_fp * ( (q4_R - w*q5_R/t) + (q4_L - w*q5_L/t) );
  }
  // Wave 5, velocity: v-cs
  real w5 =  v*q1_R/(2*cs) - q3_R/(2*cs) + q5_R/(2*t);
  // Wave 6, velocity: v+cs
  real w6 = -v*q1_L/(2*cs) + q3_L/(2*cs) + q5_L/(2*t);
  // Use right eigenmatrix to compute upwind state
  real q1 = w1 + w5 + w6;
  real q2 = w2 + u*w5 + u*w6;
  real q3 = v*w1 + (v-cs)*w5 + (v+cs)*w6;
  real q4 = w3 + w*w5 + w*w6;
  real q5 =      t*w5 + t*w6;

  r_upw     = q1;
  u_upw     = q2 / q1;
  v_upw     = q3 / q1;
  w_upw     = q4 / q1;
  theta_upw = q5 / q1;
  p_upw     = C0 * pow( r_upw * theta_upw , gamma );
  rv_upw    = r_upw * v_upw;
  for (int tr=0; tr < num_tracers; tr++) {
    if        (v > 0) {
      tracers_upw(tr,k,j,i) = tracers_limits_y(tr,0,k,j,i) / state_limits_y(idR,0,k,j,i);
    } else if (v < 0) { 
      tracers_upw(tr,k,j,i) = tracers_limits_y(tr,1,k,j,i) / state_limits_y(idR,1,k,j,i);
    } else {
      tracers_upw(tr,k,j,i) = 0.5_fp * ( tracers_limits_y(tr,0,k,j,i) / state_limits_y(idR,0,k,j,i) +
                                         tracers_limits_y(tr,1,k,j,i) / state_limits_y(idR,1,k,j,i) );
    }
  }
}

YAKL_INLINE void riemann_native_z( real5d const &state_limits_z , real5d const &tracers_limits_z ,
                                   int k , int j , int i , int num_tracers, real C0, real gamma,
                                   real &rw_upw , real &r_upw , real &u_upw , real &v_upw , real &w_upw , 
                                   real &theta_upw , real &p_upw , real4d const &tracers_upw ) {
  int  static constexpr idR = 0;  // Density
  int  static constexpr idU = 1;  // u-momentum
  int  static constexpr idV = 2;  // v-momentum
  int  static constexpr idW = 3;  // w-momentum
  int  static constexpr idT = 4;  // Density * potential temperature
  // Get left and right state
  real r_L = state_limits_z(idR,0,k,j,i)    ;   real r_R = state_limits_z(idR,1,k,j,i)    ;
  real u_L = state_limits_z(idU,0,k,j,i)/r_L;   real u_R = state_limits_z(idU,1,k,j,i)/r_R;
  real v_L = state_limits_z(idV,0,k,j,i)/r_L;   real v_R = state_limits_z(idV,1,k,j,i)/r_R;
  real w_L = state_limits_z(idW,0,k,j,i)/r_L;   real w_R = state_limits_z(idW,1,k,j,i)/r_R;
  real t_L = state_limits_z(idT,0,k,j,i)/r_L;   real t_R = state_limits_z(idT,1,k,j,i)/r_R;
  // Compute average state
  real r = 0.5_fp * (r_L + r_R);
  real u = 0.5_fp * (u_L + u_R);
  real v = 0.5_fp * (v_L + v_R);
  real w = 0.5_fp * (w_L + w_R);
  real t = 0.5_fp * (t_L + t_R);
  real p = C0 * pow(r*t,gamma);
  real cs2 = gamma*p/r;
  real cs  = sqrt(cs2);
  // Get left and right state
  real q1_L = r_L    ;   real q1_R = r_R    ;
  real q2_L = r_L*u_L;   real q2_R = r_R*u_R;
  real q3_L = r_L*v_L;   real q3_R = r_R*v_R;
  real q4_L = r_L*w_L;   real q4_R = r_R*w_R;
  real q5_L = r_L*t_L;   real q5_R = r_R*t_R;
  // Compute upwind characteristics
  // Waves 1-3, velocity: w
  real w1, w2, w3;
  if (w > 0) {
    w1 = q1_L - q5_L/t;
    w2 = q2_L - u*q5_L/t;
    w3 = q3_L - v*q5_L/t;
  } else if (w < 0) {
    w1 = q1_R - q5_R/t;
    w2 = q2_R - u*q5_R/t;
    w3 = q3_R - v*q5_R/t;
  } else {
    w1 = 0.5_fp * ( (q1_R - q5_R/t  ) + (q1_L - q5_L/t  ) );
    w2 = 0.5_fp * ( (q2_R - u*q5_R/t) + (q2_L - u*q5_L/t) );
    w3 = 0.5_fp * ( (q3_R - v*q5_R/t) + (q3_L - v*q5_L/t) );
  }
  // Wave 5, velocity: w-cs
  real w5 =  w*q1_R/(2*cs) - q4_R/(2*cs) + q5_R/(2*t);
  // Wave 6, velocity: w+cs
  real w6 = -w*q1_L/(2*cs) + q4_L/(2*cs) + q5_L/(2*t);
  // Use right eigenmatrix to compute upwind state
  real q1 = w1 + w5 + w6;
  real q2 = w2 + u*w5 + u*w6;
  real q3 = w3 + v*w5 + v*w6;
  real q4 = w*w1 + (w-cs)*w5 + (w+cs)*w6;
  real q5 =      t*w5 + t*w6;

  r_upw     = q1;
  u_upw     = q2 / q1;
  v_upw     = q3 / q1;
  w_upw     = q4 / q1;
  theta_upw = q5 / q1;
  p_upw     = C0 * pow( r_upw * theta_upw , gamma );
  rw_upw    = r_upw * w_upw;
  for (int tr=0; tr < num_tracers; tr++) {
    if        (w > 0) {
      tracers_upw(tr,k,j,i) = tracers_limits_z(tr,0,k,j,i) / state_limits_z(idR,0,k,j,i);
    } else if (w < 0) { 
      tracers_upw(tr,k,j,i) = tracers_limits_z(tr,1,k,j,i) / state_limits_z(idR,1,k,j,i);
    } else {
      tracers_upw(tr,k,j,i) = 0.5_fp * ( tracers_limits_z(tr,0,k,j,i) / state_limits_z(idR,0,k,j,i) +
                                         tracers_limits_z(tr,1,k,j,i) / state_limits_z(idR,1,k,j,i) );
    }
  }
}




YAKL_INLINE void riemann_acoust_x( real5d const &state_limits_x , real5d const &tracers_limits_x ,
                                   int k , int j , int i , int num_tracers, real C0, real gamma,
                                   real &ru_upw , real &p_upw ) {
  int  static constexpr idR = 0;  // Density
  int  static constexpr idU = 1;  // u-momentum
  int  static constexpr idV = 2;  // v-momentum
  int  static constexpr idW = 3;  // w-momentum
  int  static constexpr idT = 4;  // Density * potential temperature
  real r_L  = state_limits_x(idR,0,k,j,i);        real r_R  = state_limits_x(idR,1,k,j,i);
  real ru_L = state_limits_x(idU,0,k,j,i);        real ru_R = state_limits_x(idU,1,k,j,i);
  real rt_L = state_limits_x(idT,0,k,j,i);        real rt_R = state_limits_x(idT,1,k,j,i);
  real p_L  = C0*pow(rt_L,gamma);                 real p_R  = C0*pow(rt_R,gamma);
  real p = 0.5_fp * (p_L + p_R);
  real r = 0.5_fp * (r_L + r_R);
  real cs2 = gamma*p/r;
  real cs = sqrt(cs2);
  real w1 = p_R/2 - ru_R*cs/2;
  real w2 = p_L/2 + ru_L*cs/2;
  p_upw  = w1 + w2;
  ru_upw = (w2 - w1) / cs;
}

YAKL_INLINE void riemann_acoust_y( real5d const &state_limits_y , real5d const &tracers_limits_y ,
                                   int k , int j , int i , int num_tracers, real C0, real gamma,
                                   real &rv_upw , real &p_upw ) {
  int  static constexpr idR = 0;  // Density
  int  static constexpr idU = 1;  // u-momentum
  int  static constexpr idV = 2;  // v-momentum
  int  static constexpr idW = 3;  // w-momentum
  int  static constexpr idT = 4;  // Density * potential temperature
  real r_L  = state_limits_y(idR,0,k,j,i);        real r_R  = state_limits_y(idR,1,k,j,i);
  real rv_L = state_limits_y(idV,0,k,j,i);        real rv_R = state_limits_y(idV,1,k,j,i);
  real rt_L = state_limits_y(idT,0,k,j,i);        real rt_R = state_limits_y(idT,1,k,j,i);
  real p_L  = C0*pow(rt_L,gamma);                 real p_R  = C0*pow(rt_R,gamma);
  real p = 0.5_fp * (p_L + p_R);
  real r = 0.5_fp * (r_L + r_R);
  real cs2 = gamma*p/r;
  real cs = sqrt(cs2);
  real w1 = p_R/2 - rv_R*cs/2;
  real w2 = p_L/2 + rv_L*cs/2;
  p_upw  = w1 + w2;
  rv_upw = (w2 - w1) / cs;
}

YAKL_INLINE void riemann_acoust_z( real5d const &state_limits_z , real5d const &tracers_limits_z ,
                                   int k , int j , int i , int num_tracers, real C0, real gamma,
                                   real &rw_upw , real &p_upw ) {
  int  static constexpr idR = 0;  // Density
  int  static constexpr idU = 1;  // u-momentum
  int  static constexpr idV = 2;  // v-momentum
  int  static constexpr idW = 3;  // w-momentum
  int  static constexpr idT = 4;  // Density * potential temperature
  real r_L  = state_limits_z(idR,0,k,j,i);        real r_R  = state_limits_z(idR,1,k,j,i);
  real rw_L = state_limits_z(idW,0,k,j,i);        real rw_R = state_limits_z(idW,1,k,j,i);
  real rt_L = state_limits_z(idT,0,k,j,i);        real rt_R = state_limits_z(idT,1,k,j,i);
  real p_L  = C0*pow(rt_L,gamma);                 real p_R  = C0*pow(rt_R,gamma);
  real p = 0.5_fp * (p_L + p_R);
  real r = 0.5_fp * (r_L + r_R);
  real cs2 = gamma*p/r;
  real cs = sqrt(cs2);
  real w1 = p_R/2 - rw_R*cs/2;
  real w2 = p_L/2 + rw_L*cs/2;
  p_upw  = w1 + w2;
  rw_upw = (w2 - w1) / cs;
}



