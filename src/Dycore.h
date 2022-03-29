
#pragma once

#include "main_header.h"

class Dycore {
  public:

  int static constexpr num_state = 5;

  int static constexpr hs = 2;

  int static constexpr idR = 0;
  int static constexpr idU = 1;
  int static constexpr idV = 2;
  int static constexpr idW = 3;
  int static constexpr idT = 4;


  real1d      hyDensCells;
  real1d      hyDensThetaCells;
  real1d      hyDensEdges;
  real1d      hyDensThetaEdges;
  real        etime;
  std::string fname;

  void init(core::Coupler &coupler) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;
    using yakl::c::SimpleBounds;
    auto inFile = coupler.get_option<std::string>( "standalone_input_file" );
    YAML::Node config = YAML::LoadFile(inFile);
    auto init_data = config["init_data"].as<std::string>();
    fname          = config["out_fname"].as<std::string>();

    int  nx, ny, nz;
    real xlen, ylen, zlen, dx, dy, dz;
    bool sim2d;
    get_grid(coupler,nx,ny,nz,xlen,ylen,zlen,dx,dy,dz,sim2d);

    real R_d, R_v, cp_d, cp_v, p0, grav, kappa, gamma, C0;
    get_constants(coupler, R_d, R_v, cp_d, cp_v, p0, grav, kappa, gamma, C0);

    etime = 0;

    // Define quadrature weights and points
    const int nqpoints = 3;
    SArray<real,1,nqpoints> qpoints;
    SArray<real,1,nqpoints> qweights;

    qpoints(0) = 0.112701665379258311482073460022;
    qpoints(1) = 0.500000000000000000000000000000;
    qpoints(2) = 0.887298334620741688517926539980;

    qweights(0) = 0.277777777777777777777777777779;
    qweights(1) = 0.444444444444444444444444444444;
    qweights(2) = 0.277777777777777777777777777779;

    // Compute the state that the dycore uses
    real4d state("state",num_state,nz+2*hs,ny+2*hs,nx+2*hs);

    parallel_for( SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      for (int l=0; l < num_state; l++) {
        state(l,hs+k,hs+j,hs+i) = 0.;
      }
      //Use Gauss-Legendre quadrature
      for (int kk=0; kk<nqpoints; kk++) {
        for (int jj=0; jj<nqpoints; jj++) {
          for (int ii=0; ii<nqpoints; ii++) {
            real x = (i+0.5)*dx + (qpoints(ii)-0.5)*dx;
            real y = (j+0.5)*dy + (qpoints(jj)-0.5)*dy;   if (sim2d) y = ylen/2;
            real z = (k+0.5)*dz + (qpoints(kk)-0.5)*dz;
            real r, u, v, w, t, hr, ht;

            if (init_data == "thermal") { thermal(x,y,z,xlen,ylen,grav,C0,gamma,cp_d,p0,R_d,r,u,v,w,t,hr,ht); }
            else { endrun("ERROR: init_data not supported"); }

            if (sim2d) v = 0;

            real wt = qweights(ii)*qweights(jj)*qweights(kk);
            state(idR,hs+k,hs+j,hs+i) += r                         * wt;
            state(idU,hs+k,hs+j,hs+i) += (r+hr)*u                  * wt;
            state(idV,hs+k,hs+j,hs+i) += (r+hr)*v                  * wt;
            state(idW,hs+k,hs+j,hs+i) += (r+hr)*w                  * wt;
            state(idT,hs+k,hs+j,hs+i) += ( (r+hr)*(t+ht) - hr*ht ) * wt;
          }
        }
      }
    });

    hyDensCells      = real1d("hyDensCells"     ,nz  );
    hyDensThetaCells = real1d("hyDensThetaCells",nz  );
    hyDensEdges      = real1d("hyDensEdges"     ,nz+1);
    hyDensThetaEdges = real1d("hyDensThetaEdges",nz+1);

    parallel_for( SimpleBounds<1>(nz) , YAKL_LAMBDA (int k) {
      hyDensCells     (k) = 0.;
      hyDensThetaCells(k) = 0.;
      for (int kk=0; kk<nqpoints; kk++) {
        real z = (k+0.5)*dz + (qpoints(kk)-0.5)*dz;
        real hr, ht;

        if (init_data == "thermal") { hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht); }

        hyDensCells     (k) += hr    * qweights(kk);
        hyDensThetaCells(k) += hr*ht * qweights(kk);
      }
    });

    parallel_for( SimpleBounds<1>(nz+1) , YAKL_LAMBDA (int k) {
      real z = k*dz;
      real hr, ht;

      if (init_data == "thermal") { hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht); }

      hyDensEdges     (k) = hr   ;
      hyDensThetaEdges(k) = hr*ht;
    });

    convert_dynamics_to_coupler( coupler , state );

    output( coupler , etime );
  }


  void convert_dynamics_to_coupler( core::Coupler &coupler , realConst4d state ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;
    using yakl::c::SimpleBounds;
    int  nx, ny, nz;
    real xlen, ylen, zlen, dx, dy, dz;
    bool sim2d;
    get_grid(coupler,nx,ny,nz,xlen,ylen,zlen,dx,dy,dz,sim2d);

    real R_d, R_v, cp_d, cp_v, p0, grav, kappa, gamma, C0;
    get_constants(coupler, R_d, R_v, cp_d, cp_v, p0, grav, kappa, gamma, C0);

    YAKL_SCOPE( hyDensCells      , this->hyDensCells      );
    YAKL_SCOPE( hyDensThetaCells , this->hyDensThetaCells );

    auto rho_d = coupler.dm.get<real,3>("density_dry");
    auto uvel  = coupler.dm.get<real,3>("uvel"       );
    auto vvel  = coupler.dm.get<real,3>("vvel"       );
    auto wvel  = coupler.dm.get<real,3>("wvel"       );
    auto temp  = coupler.dm.get<real,3>("temp"       );

    parallel_for( SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      real r  = state(idR,hs+k,hs+j,hs+i) + hyDensCells(k);
      real ru = state(idU,hs+k,hs+j,hs+i);
      real rv = state(idV,hs+k,hs+j,hs+i);
      real rw = state(idW,hs+k,hs+j,hs+i);
      real rt = state(idT,hs+k,hs+j,hs+i) + hyDensThetaCells(k);
      real p  = C0 * pow( rt , gamma );

      rho_d(k,j,i) = r;
      uvel (k,j,i) = ru / r;
      vvel (k,j,i) = rv / r;
      wvel (k,j,i) = rw / r;
      temp (k,j,i) = p / rho_d(k,j,i) / R_d;
    });
  }


  void convert_coupler_to_dynamics( core::Coupler const &coupler , real4d &state ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;
    using yakl::c::SimpleBounds;
    int  nx, ny, nz;
    real xlen, ylen, zlen, dx, dy, dz;
    bool sim2d;
    get_grid(coupler,nx,ny,nz,xlen,ylen,zlen,dx,dy,dz,sim2d);

    real R_d, R_v, cp_d, cp_v, p0, grav, kappa, gamma, C0;
    get_constants(coupler, R_d, R_v, cp_d, cp_v, p0, grav, kappa, gamma, C0);

    YAKL_SCOPE( hyDensCells      , this->hyDensCells      );
    YAKL_SCOPE( hyDensThetaCells , this->hyDensThetaCells );

    auto rho_d = coupler.dm.get<real const,3>("density_dry");
    auto uvel  = coupler.dm.get<real const,3>("uvel"       );
    auto vvel  = coupler.dm.get<real const,3>("vvel"       );
    auto wvel  = coupler.dm.get<real const,3>("wvel"       );
    auto temp  = coupler.dm.get<real const,3>("temp"       );

    parallel_for( SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      real r  = rho_d(k,j,i);
      real ru = uvel(k,j,i) * r;
      real rv = vvel(k,j,i) * r;
      real rw = wvel(k,j,i) * r;
      real T  = temp(k,j,i);
      real p  = rho_d(k,j,i) * R_d * T;
      real rt = pow( p/C0 , 1._fp / gamma );

      state(idR,hs+k,hs+j,hs+i) = r - hyDensCells(k);
      state(idU,hs+k,hs+j,hs+i) = ru;
      state(idV,hs+k,hs+j,hs+i) = rv;
      state(idW,hs+k,hs+j,hs+i) = rw;
      state(idT,hs+k,hs+j,hs+i) = rt - hyDensThetaCells(k);
    });
  }


  void output( core::Coupler const &coupler , real etime ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;
    using yakl::c::SimpleBounds;

    int  nz = coupler.get_nz();   int  ny = coupler.get_ny();   int  nx = coupler.get_nx();
    real dz = coupler.get_dz();   real dy = coupler.get_dy();   real dx = coupler.get_dx();

    real4d state("state",num_state,hs+nz,hs+ny,hs+nx);
    convert_coupler_to_dynamics( coupler , state );

    yakl::SimpleNetCDF nc;
    int ulIndex = 0; // Unlimited dimension index to place this data at

    if (etime == 0) {
      nc.create(fname);

      // x-coordinate
      real1d xloc("xloc",nx);
      parallel_for( "Spatial.h output 1" , nx , YAKL_LAMBDA (int i) { xloc(i) = (i+0.5)*dx; });
      nc.write(xloc.createHostCopy(),"x",{"x"});

      // y-coordinate
      real1d yloc("yloc",ny);
      parallel_for( "Spatial.h output 2" , ny , YAKL_LAMBDA (int i) { yloc(i) = (i+0.5)*dy; });
      nc.write(yloc.createHostCopy(),"y",{"y"});

      // z-coordinate
      real1d zloc("zloc",nz);
      parallel_for( "Spatial.h output 3" , nz , YAKL_LAMBDA (int i) { zloc(i) = (i+0.5)*dz; });
      nc.write(zloc.createHostCopy(),"z",{"z"});

      nc.write(hyDensCells     .createHostCopy(),"hydrostatic_density"      ,{"z"});
      nc.write(hyDensThetaCells.createHostCopy(),"hydrostatic_density_theta",{"z"});

      // Elapsed time
      nc.write1(0._fp,"t",0,"t");
    } else {
      nc.open(fname,yakl::NETCDF_MODE_WRITE);
      ulIndex = nc.getDimSize("t");

      // Write the elapsed time
      nc.write1(etime,"t",ulIndex,"t");
    }

    auto &dm = coupler.dm;
    nc.write1(dm.get<real const,3>("density_dry"),"density_dry",{"z","y","x"},ulIndex,"t");
    nc.write1(dm.get<real const,3>("uvel"       ),"uvel"       ,{"z","y","x"},ulIndex,"t");
    nc.write1(dm.get<real const,3>("vvel"       ),"vvel"       ,{"z","y","x"},ulIndex,"t");
    nc.write1(dm.get<real const,3>("wvel"       ),"wvel"       ,{"z","y","x"},ulIndex,"t");
    nc.write1(dm.get<real const,3>("temp"       ),"temperature",{"z","y","x"},ulIndex,"t");

    real3d data("data",nz,ny,nx);
    parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      data(k,j,i) = state(idR,hs+k,hs+j,hs+i);
    });
    nc.write1(data,"density_pert",{"z","y","x"},ulIndex,"t");
    parallel_for( Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      real hy_r  = hyDensCells     (k);
      real hy_rt = hyDensThetaCells(k);
      real r     = state(idR,hs+k,hs+j,hs+i) + hy_r;
      real rt    = state(idT,hs+k,hs+j,hs+i) + hy_rt;
      data(k,j,i) = rt / r - hy_rt / hy_r;
    });
    nc.write1(data,"theta_pert",{"z","y","x"},ulIndex,"t");

    nc.close();
  }


  YAKL_INLINE static void thermal(real x, real y, real z, real xlen, real ylen, real grav, real C0, real gamma,
                                  real cp, real p0, real rd, real &r, real &u, real &v, real &w, real &t,
                                  real &hr, real &ht) {
    hydro_const_theta(z,grav,C0,cp,p0,gamma,rd,hr,ht);
    r = 0.;
    t = 0.;
    u = 0.;
    v = 0.;
    w = 0.;
    t = t + sample_ellipse_cosine(3._fp  ,  x,y,z  ,  xlen/2,ylen/2,2000.  ,  2000.,2000.,2000.);
  }


  YAKL_INLINE static void hydro_const_theta( real z, real grav, real C0, real cp, real p0, real gamma, real rd,
                                             real &r, real &t ) {
    const real theta0 = 300.;  //Background potential temperature
    const real exner0 = 1.;    //Surface-level Exner pressure
    t = theta0;                                       //Potential Temperature at z
    real exner = exner0 - grav * z / (cp * theta0);   //Exner pressure at z
    real p = p0 * pow(exner,(cp/rd));                 //Pressure at z
    real rt = pow((p / C0),(1._fp / gamma));          //rho*theta at z
    r = rt / t;                                       //Density at z
  }


  YAKL_INLINE static real sample_ellipse_cosine(real amp, real x   , real y   , real z   ,
                                                          real x0  , real y0  , real z0  ,
                                                          real xrad, real yrad, real zrad) {
    //Compute distance from bubble center
    real dist = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) +
                      ((y-y0)/yrad)*((y-y0)/yrad) +
                      ((z-z0)/zrad)*((z-z0)/zrad) ) * M_PI / 2.;
    //If the distance from bubble center is less than the radius, create a cos**2 profile
    if (dist <= M_PI / 2.) {
      return amp * pow(cos(dist),2._fp);
    } else {
      return 0.;
    }
  }


  static void get_grid( core::Coupler const &coupler, int  &nx  , int  &ny  , int  &nz  ,
                                                      real &xlen, real &ylen, real &zlen,
                                                      real &dx  , real &dy  , real &dz  , bool &sim2d) {
    xlen = coupler.get_xlen();   ylen = coupler.get_ylen();   zlen = coupler.get_zlen();
    nx   = coupler.get_nx  ();   ny   = coupler.get_ny  ();   nz   = coupler.get_nz  ();
    dx   = coupler.get_dx  ();   dy   = coupler.get_dy  ();   dz   = coupler.get_dz  ();
    sim2d = (ny == 1);
  }


  static void get_constants(core::Coupler const &coupler, real &R_d, real &R_v, real &cp_d, real &cp_v, real &p0,
                                                          real &grav, real &kappa, real &gamma, real &C0) {
    R_d  = coupler.R_d ;
    R_v  = coupler.R_v ;
    cp_d = coupler.cp_d;
    cp_v = coupler.cp_v;
    p0   = coupler.p0  ;
    grav = coupler.grav;
    kappa = R_d / cp_d;
    gamma = cp_d / (cp_d - R_d);
    C0    = pow( R_d * pow( p0 , -kappa ) , gamma );
  }

};


