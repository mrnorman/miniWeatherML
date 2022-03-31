
#pragma once

#include "main_header.h"
#include "DataManager.h"
#include "YAKL_netcdf.h"
#include "Options.h"

// The Coupler class holds everything a component or module of this model would need in order to perform its
// changes to the model state


namespace core {

  class Coupler {
    public:

    Options options;

    real R_d;    // Dry air gas constant
    real R_v;    // Water vapor gas constant
    real cp_d;   // Dry air specific heat at constant pressure
    real cp_v;   // Water vapor specific heat at constant pressure
    real grav;   // Acceleration due to gravity (m s^-2): typically 9.81
    real p0;     // Reference pressure (Pa): typically 10^5
    real xlen;   // Domain length in the x-direction in meters
    real ylen;   // Domain length in the y-direction in meters
    real zlen;   // Domain length in the z-direction in meters
    real dt_gcm; // Time step of the GCM for this MMF invocation

    DataManager     dm;
    DataManagerHost dm_host;

    struct Tracer {
      std::string name;
      std::string desc;
      bool        positive;
      bool        adds_mass;
    };
    std::vector<Tracer> tracers;

    struct DycoreFunction {
      std::string                                   name;
      std::function< void ( Coupler & , real ) > func;
    };
    std::vector< DycoreFunction > dycore_functions;

    struct MMFFunction {
      std::string                                   name;
      std::function< void ( Coupler & , real ) > func;
    };
    std::vector< MMFFunction > mmf_functions;


    Coupler() {
      this->R_d    = 287 ;
      this->R_v    = 461 ;
      this->cp_d   = 1004;
      this->cp_v   = 1859;
      this->grav   = 9.81;
      this->p0     = 1.e5;
      this->xlen   = -1;
      this->ylen   = -1;
      this->zlen   = -1;
      this->dt_gcm = -1;
    }


    Coupler(Coupler &&) = default;
    Coupler &operator=(Coupler &&) = default;
    Coupler(Coupler const &) = delete;
    Coupler &operator=(Coupler const &) = delete;


    ~Coupler() {
      dm.finalize();
      options.finalize();
      tracers = std::vector<Tracer>();
      this->R_d    = 287 ;
      this->R_v    = 461 ;
      this->cp_d   = 1004;
      this->cp_v   = 1859;
      this->grav   = 9.81;
      this->p0     = 1.e5;
      this->xlen   = -1;
      this->ylen   = -1;
      this->zlen   = -1;
      this->dt_gcm = -1;
    }


    void set_dt_gcm(real dt_gcm) { this->dt_gcm = dt_gcm; }


    real get_dt_gcm() const { return this->dt_gcm; }


    real get_xlen() const { return this->xlen; }


    real get_ylen() const { return this->ylen; }


    real get_zlen() const { return this->zlen; }


    int get_nx() const {
      if (dm.find_dimension("x") == -1) return -1;
      return dm.get_dimension_size("x");
    }


    int get_ny() const {
      if (dm.find_dimension("y") == -1) return -1;
      return dm.get_dimension_size("y");
    }


    int get_nz() const {
      if (dm.find_dimension("z") == -1) return -1;
      return dm.get_dimension_size("z");
    }


    real get_dx() const { return get_xlen() / get_nx(); }


    real get_dy() const { return get_ylen() / get_ny(); }


    real get_dz() const { return get_zlen() / get_nz(); }


    int get_num_tracers() const { return tracers.size(); }


    template <class T>
    void add_option( std::string key , T value ) {
      options.add_option<T>(key,value);
    }


    template <class T>
    void set_option( std::string key , T value ) {
      options.set_option<T>(key,value);
    }


    template <class T>
    T get_option( std::string key ) const {
      return options.get_option<T>(key);
    }


    bool option_exists( std::string key ) const {
      return options.option_exists(key);
    }


    void delete_option( std::string key ) {
      options.delete_option(key);
    }


    void set_phys_constants(real R_d, real R_v, real cp_d, real cp_v, real grav=9.81, real p0=1.e5) {
      this->R_d  = R_d ;
      this->R_v  = R_v ;
      this->cp_d = cp_d;
      this->cp_v = cp_v;
      this->grav = grav;
      this->p0   = p0  ;
    }


    void set_grid(real xlen, real ylen, real zlen) {
      this->xlen = xlen;
      this->ylen = ylen;
      this->zlen = zlen;
    }

    
    void add_tracer( std::string tracer_name , std::string tracer_desc , bool positive , bool adds_mass ) {
      int nz = get_nz();
      int ny = get_ny();
      int nx = get_nx();
      dm.register_and_allocate<real>( tracer_name , tracer_desc , {nz,ny,nx} , {"z","y","x"} );
      tracers.push_back( { tracer_name , tracer_desc , positive , adds_mass } );
    }

    
    std::vector<std::string> get_tracer_names() const {
      std::vector<std::string> ret;
      for (int i=0; i < tracers.size(); i++) { ret.push_back( tracers[i].name ); }
      return ret;
    }

    
    void get_tracer_info(std::string tracer_name , std::string &tracer_desc, bool &tracer_found ,
                         bool &positive , bool &adds_mass) const {
      std::vector<std::string> ret;
      for (int i=0; i < tracers.size(); i++) {
        if (tracer_name == tracers[i].name) {
          positive     = tracers[i].positive ;
          tracer_desc  = tracers[i].desc     ;
          adds_mass    = tracers[i].adds_mass;
          tracer_found = true;
          return;
        }
      }
      tracer_found = false;
    }

    
    bool tracer_exists( std::string tracer_name ) const {
      for (int i=0; i < tracers.size(); i++) {
        if (tracer_name == tracers[i].name) return true;
      }
      return false;
    }


    void allocate_coupler_state( int nz, int ny, int nx ) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;
      using yakl::c::SimpleBounds;
      dm.register_and_allocate<real>("density_dry","dry density"         ,{nz,ny,nx},{"z","y","x"});
      dm.register_and_allocate<real>("uvel"       ,"x-direction velocity",{nz,ny,nx},{"z","y","x"});
      dm.register_and_allocate<real>("vvel"       ,"y-direction velocity",{nz,ny,nx},{"z","y","x"});
      dm.register_and_allocate<real>("wvel"       ,"z-direction velocity",{nz,ny,nx},{"z","y","x"});
      dm.register_and_allocate<real>("temp"       ,"temperature"         ,{nz,ny,nx},{"z","y","x"});

      auto density_dry  = dm.get_collapsed<real>("density_dry");
      auto uvel         = dm.get_collapsed<real>("uvel"       );
      auto vvel         = dm.get_collapsed<real>("vvel"       );
      auto wvel         = dm.get_collapsed<real>("wvel"       );
      auto temp         = dm.get_collapsed<real>("temp"       );

      parallel_for( "coupler zero" , SimpleBounds<1>(nz*ny*nx) , YAKL_LAMBDA (int i) {
        density_dry(i) = 0;
        uvel       (i) = 0;
        vvel       (i) = 0;
        wvel       (i) = 0;
        temp       (i) = 0;
      });
    }


    real3d compute_pressure_array() const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;
      using yakl::c::SimpleBounds;
      auto dens_dry = dm.get<real const,3>("density_dry");
      auto dens_wv  = dm.get<real const,3>("water_vapor");
      auto temp     = dm.get<real const,3>("temp");

      int nz = get_nz();
      int ny = get_ny();
      int nx = get_nx();

      real3d pressure("pressure",nz,ny,nx);

      YAKL_SCOPE( R_d , this->R_d );
      YAKL_SCOPE( R_v , this->R_v );

      parallel_for( "coupler pressure" , SimpleBounds<3>(nz,ny,nx) ,
                    YAKL_LAMBDA (int k, int j, int i) {
        real rho_d = dens_dry(k,j,i);
        real rho_v = dens_wv (k,j,i);
        real T     = temp    (k,j,i);
        pressure(k,j,i) = rho_d*R_d*T + rho_v*R_v*T;
      });

      return pressure;
    }

  };

}


