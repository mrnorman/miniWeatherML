
#pragma once

#include "main_header.h"

namespace coupler {

  template <unsigned int ord>
  class VerticalInterp {
  public:
    int  static constexpr hs = (ord-1)/2;
    real static constexpr eps = 1.0e-20;

    int static constexpr BC_ZERO_GRADIENT = 0;
    int static constexpr BC_ZERO_VALUE    = 1;

    struct InternalData {
      real5d weno_recon_lo;
      real4d weno_recon_hi;
      SArray<real,1,hs+2> idl;
    };

    InternalData internal;

    VerticalInterp() {
      if        (ord == 3) {
        internal.idl(0) = 1;
        internal.idl(1) = 1;
        internal.idl(2) = 1000;
      } else if (ord == 5) {
        internal.idl(0) = 1;
        internal.idl(1) = 1;
        internal.idl(2) = 1;
        internal.idl(3) = 1000;
      } else if (ord == 7) {
        internal.idl(0) = 1;
        internal.idl(1) = 1;
        internal.idl(2) = 1;
        internal.idl(3) = 1;
        internal.idl(4) = 1000;
      } else if (ord == 9) {
        internal.idl(0) = 1;
        internal.idl(1) = 1;
        internal.idl(2) = 1;
        internal.idl(3) = 1;
        internal.idl(4) = 1;
        internal.idl(5) = 1000;
      }
      convexify( internal.idl );
    }



    inline real4d cells_to_edges( realConst4d data ,
                                  int bc_lower ,
                                  int bc_upper ) const {
      int nz   = data.dimension[0];
      int ny   = data.dimension[1];
      int nx   = data.dimension[2];
      int nens = data.dimension[3];

      YAKL_SCOPE( internal , this->internal );

      real5d limits("limits",2,nz+1,ny,nx,nens);

      // Get cell edge estimates from cell-centered reconstructions
      parallel_for( "recon to edges" , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        // Gather the stencil
        SArray<real,1,ord> stencil;
        for (int kk=0; kk < ord; kk++) {
          int kval = k-hs+kk;
          if      (kval >= 0 && kval < nz) {
            stencil(kk) = data(kval,j,i,iens);
          } else if (kval < 0) {
            if (bc_lower == BC_ZERO_GRADIENT) stencil(kk) = data(0   ,j,i,iens);
            if (bc_lower == BC_ZERO_VALUE   ) stencil(kk) = 0;
          } else if (kval >= nz) {
            if (bc_upper == BC_ZERO_GRADIENT) stencil(kk) = data(nz-1,j,i,iens);
            if (bc_upper == BC_ZERO_VALUE   ) stencil(kk) = 0;
          }
        }

        // Compute reconstruction coefficients
        SArray<real,1,ord> coefs;
        compute_weno_coefs( stencil , coefs , internal , k , iens );

        // Sample values at cell edges
        limits(1,k  ,j,i,iens) = sample_val( coefs , -0.5_fp );
        limits(0,k+1,j,i,iens) = sample_val( coefs ,  0.5_fp );

        // Manage boundaries
        if (k == 0) {
          if      (bc_lower == BC_ZERO_VALUE   ) {
            limits(0,0,j,i,iens) = 0;
            limits(1,0,j,i,iens) = 0;
          } else if (bc_lower == BC_ZERO_GRADIENT) {
            limits(0,0,j,i,iens) = limits(1,0,j,i,iens);
          }
        }

        if (k == nz-1) {
          if      (bc_upper == BC_ZERO_VALUE   ) {
            limits(0,nz,j,i,iens) = 0;
            limits(1,nz,j,i,iens) = 0;
          } else if (bc_upper == BC_ZERO_GRADIENT) {
            limits(1,nz,j,i,iens) = limits(0,nz,j,i,iens);
          }
        }
      });

      real4d edges("edges",nz+1,ny,nx,nens);

      // Reconcile multple estimates at each cell edge (simple average for now)
      parallel_for( "avg edges" , SimpleBounds<4>(nz+1,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        edges(k,j,i,iens) = 0.5_fp * (limits(0,k,j,i,iens) + limits(1,k,j,i,iens));
      });
      
      return edges;
    }



    YAKL_INLINE static real sample_val(SArray<real,1,3> const &coefs , real z ) {
      return (coefs(2)*z + coefs(1))*z + coefs(0);
    }



    YAKL_INLINE static real sample_val(SArray<real,1,5> const &coefs , real z ) {
      return (((coefs(4)*z + coefs(3))*z + coefs(2))*z + coefs(1))*z + coefs(0);
    }



    YAKL_INLINE static real sample_val(SArray<real,1,7> const &coefs , real z ) {
      return (((((coefs(6)*z + coefs(5))*z + coefs(4))*z + coefs(3))*z + coefs(2))*z + coefs(1)) + coefs(0);
    }



    YAKL_INLINE static real sample_val(SArray<real,1,9> const &coefs , real z ) {
      return (((((((coefs(8)*z + coefs(7))*z + coefs(6))*z + coefs(5))*z + coefs(4))*z + coefs(3)) + coefs(2))*z + coefs(1))*z + coefs(0);
    }



    inline void init( realConst2d zint ) {
      int nz   = zint.dimension[0]-1;
      int nens = zint.dimension[1];
      real2d zint_ghost("zint_ghost",nz+2*hs+1,nens);

      // Create ghost cells for cell interface heights
      parallel_for( "ghost" , SimpleBounds<2>(nz+2*hs+1,nens) , YAKL_LAMBDA (int k, int iens) {
        if (k < hs) {
          real dz0 = zint(1,iens) - zint(0,iens);
          zint_ghost(k,iens) = zint(0,iens) - (hs-k)*dz0;
        } else if (k >= hs && k < hs+nz+1) {
          zint_ghost(k,iens) = zint(k-hs,iens);
        } else {
          real dztop = zint(nz,iens) - zint(nz-1,iens);
          zint_ghost(k,iens) = zint(nz,iens) + dztop * (k-hs-nz);
        }
      });

      internal.weno_recon_lo = real5d("weno_recon_lo",nz,hs+1,hs+1,hs+1,nens);
      internal.weno_recon_hi = real4d("weno_recon_hi",nz,ord,ord,nens);

      YAKL_SCOPE( weno_recon_lo , this->internal.weno_recon_lo );
      YAKL_SCOPE( weno_recon_hi , this->internal.weno_recon_hi );

      parallel_for( "recon matrices" , SimpleBounds<2>(nz,nens) , YAKL_LAMBDA (int k, int iens) {
        int constexpr hs = (ord-1)/2;

        // Cell interface height locations
        SArray<double,1,ord+1> locs;
        for (int kk=0; kk < ord+1; kk++) { locs(kk) = zint_ghost(k+kk,iens); }

        // Normalize stencil locations
        double zmid = ( locs(hs+1) + locs(hs) ) / 2;
        double dzmid = locs(hs+1) - locs(hs);
        for (int kk=0; kk < ord+1; kk++) {
          locs(kk) = ( locs(kk) - zmid ) / dzmid;
        }

        // Compute WENO reocnstruction matrices
        SArray<double,3,hs+1,hs+1,hs+1> recon_lo;
        SArray<double,2,ord,ord>        recon_hi;
        sten_to_coefs_variable  (locs,recon_hi);
        weno_lower_sten_to_coefs(locs,recon_lo);

        // Store WENO reconstruction matrices
        for (int kk=0; kk < hs+1; kk++) {
          for (int jj=0; jj < hs+1; jj++) {
            for (int ii=0; ii < hs+1; ii++) {
              weno_recon_lo(k,kk,jj,ii,iens) = recon_lo(kk,jj,ii);
            }
          }
        }
        for (int jj=0; jj < ord; jj++) {
          for (int ii=0; ii < ord; ii++) {
            weno_recon_hi(k,jj,ii,iens) = recon_hi(jj,ii);
          }
        }
      });
    }



    template <unsigned int ordloc>
    YAKL_INLINE static void coefs_to_sten_variable(SArray<double,1,ordloc+1> const &locs ,
                                                   SArray<double,2,ordloc,ordloc> &rslt) {
      // Create the Vandermonde matrix
      SArray<double,1,ordloc+1> locs_pwr;
      // Initialize power of locations
      for (int i=0; i < ordloc+1; i++) {
        locs_pwr(i) = locs(i);
      }
      // Store first column of the matrix
      for (int i=0; i < ordloc; i++) {
        rslt(0,i) = 1;
      }
      for (int i=1; i < ordloc; i++) {
        for (int j=0; j < ordloc+1; j++) {
          locs_pwr(j) *= locs(j);
        }
        for (int j=0; j < ordloc; j++) {
          rslt(i,j) = 1./(i+1.) * (locs_pwr(j) - locs_pwr(j+1)) / (locs(j)-locs(j+1));
        }
      }
    }



    template <unsigned int ordloc>
    YAKL_INLINE static void sten_to_coefs_variable(SArray<double,1,ordloc+1> const &locs ,
                                                   SArray<double,2,ordloc,ordloc> &rslt) {
      using yakl::intrinsics::matinv_ge;

      // Get coefs to stencil matrix
      SArray<double,2,ordloc,ordloc> c2s;
      coefs_to_sten_variable(locs , c2s);

      // Invert to get sten_to_coefs
      rslt = matinv_ge( c2s );
    }



    YAKL_INLINE static void weno_lower_sten_to_coefs( SArray<double,1,ord+1> const &locs ,
                                                      SArray<double,3,hs+1,hs+1,hs+1> &weno_recon ) {
      SArray<double,2,hs+1,hs+1> recon_lo;
      SArray<double,1,hs+2> locs_lo;

      // Create low-order matrices
      for (int i = 0; i < hs+1; i++) {
        for (int ii=0; ii < hs+2; ii++) {
          locs_lo(ii) = locs(i+ii);
        }
        sten_to_coefs_variable( locs_lo , recon_lo );
        for (int jj=0; jj < hs+1; jj++) {
          for (int ii=0; ii < hs+1; ii++) {
            weno_recon(i,jj,ii) = recon_lo(jj,ii);
          }
        }
      }
    }



    YAKL_INLINE static void convexify( SArray<real,1,hs+2> &wts ) {
      real sum = 0._fp;
      for (int i=0; i<hs+2; i++) { sum += wts(i); }
      for (int i=0; i<hs+2; i++) { wts(i) /= (sum + eps); }
    }



    YAKL_INLINE static void
    compute_weno_coefs( SArray<real,1,ord> const &u ,
                        SArray<real,1,ord> &aw ,
                        VerticalInterp<ord>::InternalData const &internal,
                        int k, int iens) {
      SArray<real,2,hs+1,hs+1> a_lo;
      SArray<real,1,ord> a_hi;

      // Compute three quadratic polynomials (left, center, and right) and the high-order polynomial
      for(int i=0; i<hs+1; i++) {
        for (int ii=0; ii<hs+1; ii++) {
          real tmp = 0;
          for (int s=0; s<hs+1; s++) {
            tmp += internal.weno_recon_lo(k,i,s,ii,iens) * u(i+s);
          }
          a_lo(i,ii) = tmp;
        }
      }
      for (int ii=0; ii<ord; ii++) {
        real tmp = 0;
        for (int s=0; s<ord; s++) {
          tmp += internal.weno_recon_hi(k,s,ii,iens) * u(s);
        }
        a_hi(ii) = tmp;
      }

      // Compute "bridge" polynomial
      for (int i=0; i<hs+1; i++) {
        for (int ii=0; ii<hs+1; ii++) {
          a_hi(ii) -= internal.idl(i)*a_lo(i,ii);
        }
      }
      for (int ii=0; ii<ord; ii++) {
        a_hi(ii) /= internal.idl(hs+1);
      }

      SArray<real,1,hs+1> lotmp;
      SArray<real,1,hs+2> tv;

      // Compute total variation of all candidate polynomials
      for (int i=0; i<hs+1; i++) {
        for (int ii=0; ii<hs+1; ii++) {
          lotmp(ii) = a_lo(i,ii);
        }
        tv(i) = TV(lotmp);
      }
      tv(hs+1) = TV(a_hi);

      SArray<real,1,hs+2> wts;

      // WENO weights are proportional to the inverse of TV**2 and then re-confexified
      for (int i=0; i<hs+2; i++) {
        wts(i) = internal.idl(i) / ( tv(i)*tv(i) + eps );
      }
      convexify(wts);

      // WENO polynomial is the weighted sum of candidate polynomials using WENO weights instead of ideal weights
      for (int i=0; i < ord; i++) {
        aw(i) = wts(hs+1) * a_hi(i);
      }
      for (int i=0; i<hs+1; i++) {
        for (int ii=0; ii<hs+1; ii++) {
          aw(ii) += wts(i) * a_lo(i,ii);
        }
      }
    }



    YAKL_INLINE static real TV(SArray<real,1,2> &a) {
      return 1.0000000000000000000000000000000000000*(a(1)*a(1));
    }



    YAKL_INLINE static real TV(SArray<real,1,3> &a) {
      return 1.0000000000000000000000000000000000000*(a(1)*a(1))+4.3333333333333333333333333333333333333*(a(2)*a(2));
    }



    YAKL_INLINE static real TV(SArray<real,1,4> &a) {
      return 1.0000000000000000000000000000000000000*(a(1)*a(1))+4.3333333333333333333333333333333333333*(a(2)*a(2))+0.50000000000000000000000000000000000000*a(1)*a(3)+39.112500000000000000000000000000000000*(a(3)*a(3));
    }



    YAKL_INLINE static real TV(SArray<real,1,5> &a) {
      return 1.0000000000000000000000000000000000000*(a(1)*a(1))+4.3333333333333333333333333333333333333*(a(2)*a(2))+0.50000000000000000000000000000000000000*a(1)*a(3)+39.112500000000000000000000000000000000*(a(3)*a(3))+4.2000000000000000000000000000000000000*a(2)*a(4)+625.83571428571428571428571428571428571*(a(4)*a(4));
    }



    YAKL_INLINE static real TV(SArray<real,1,7> &a) {
      return 1.0000000000000000000000000000000000000*(a(1)*a(1))+4.3333333333333333333333333333333333333*(a(2)*a(2))+0.50000000000000000000000000000000000000*a(1)*a(3)+39.112500000000000000000000000000000000*(a(3)*a(3))+4.2000000000000000000000000000000000000*a(2)*a(4)+625.83571428571428571428571428571428571*(a(4)*a(4))+0.12500000000000000000000000000000000000*a(1)*a(5)+63.066964285714285714285714285714285714*a(3)*a(5)+15645.903707837301587301587301587301587*(a(5)*a(5))+1.5535714285714285714285714285714285714*a(2)*a(6)+1513.6279761904761904761904761904761905*a(4)*a(6)+563252.53667816558441558441558441558442*(a(6)*a(6));
    }



    YAKL_INLINE static real TV(SArray<real,1,9> &a) {
      return 1.0000000000000000000000000000000000000*(a(1)*a(1))+4.3333333333333333333333333333333333333*(a(2)*a(2))+0.50000000000000000000000000000000000000*a(1)*a(3)+39.112500000000000000000000000000000000*(a(3)*a(3))+4.2000000000000000000000000000000000000*a(2)*a(4)+625.83571428571428571428571428571428571*(a(4)*a(4))+0.12500000000000000000000000000000000000*a(1)*a(5)+63.066964285714285714285714285714285714*a(3)*a(5)+15645.903707837301587301587301587301587*(a(5)*a(5))+1.5535714285714285714285714285714285714*a(2)*a(6)+1513.6279761904761904761904761904761905*a(4)*a(6)+563252.53667816558441558441558441558442*(a(6)*a(6))+0.031250000000000000000000000000000000000*a(1)*a(7)+32.643229166666666666666666666666666667*a(3)*a(7)+52976.985381155303030303030303030303030*a(5)*a(7)+2.7599374298150335992132867132867132867e7*(a(7)*a(7))+0.51388888888888888888888888888888888889*a(2)*a(8)+1044.5890151515151515151515151515151515*a(4)*a(8)+2.5428953000983391608391608391608391608e6*a(6)*a(8)+1.7663599550818819201631701631701631702e9*(a(8)*a(8));
    }


  };

}







