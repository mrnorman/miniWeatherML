
#include "coupler.h"
#include "Dycore.h"


int main(int argc, char** argv) {
  yakl::init();
  {
    using yakl::intrinsics::abs;
    using yakl::intrinsics::maxval;
    yakl::timer_start("main");

    core::Coupler coupler;

    if (argc <= 1) { endrun("ERROR: Must pass the input YAML filename as a parameter"); }
    std::string inFile(argv[1]);
    YAML::Node config = YAML::LoadFile(inFile);
    if ( !config            ) { endrun("ERROR: Invalid YAML input file"); }
    real sim_time  = config["sim_time"].as<real>();
    int  nx        = config["nx"      ].as<int>();
    int  ny        = config["ny"      ].as<int>();
    int  nz        = config["nz"      ].as<int>();
    real xlen      = config["xlen"    ].as<real>();
    real ylen      = config["ylen"    ].as<real>();
    real zlen      = config["zlen"    ].as<real>();
    real dtphys_in = config["dt_phys" ].as<real>();

    Dycore dycore;

    // coupler.set_phys_constants( micro.R_d , micro.R_v , micro.cp_d , micro.cp_v , micro.grav , micro.p0 );

    coupler.allocate_coupler_state( nz, ny, nx );

    coupler.set_grid( xlen , ylen , zlen );

    // This is for the dycore to pull out to determine how to do idealized test cases
    coupler.set_option<std::string>( "standalone_input_file" , inFile );

    dycore.init( coupler ); // Dycore should initialize its own state here

    // Now that we have an initial state, define hydrostasis for each ensemble member
    // if (use_coupler_hydrostasis) coupler.update_hydrostasis( coupler.compute_pressure_array() );

    real etime = 0;

    real dtphys = dtphys_in;
    while (etime < sim_time) {
      if (dtphys_in <= 0.) { dtphys = dycore.compute_time_step(coupler); }
      if (etime + dtphys > sim_time) { dtphys = sim_time - etime; }

      yakl::timer_start("dycore");
      dycore.time_step( coupler , dtphys );
      yakl::timer_stop("dycore");

      etime += dtphys;
      real maxw = maxval(abs(coupler.dm.get_collapsed<real const>("wvel")));
      std::cout << "Etime , dtphys, maxw: " << etime  << " , " 
                                            << dtphys << " , "
                                            << std::setw(10) << maxw << "\n";
    }

    std::cout << "Elapsed Time: " << etime << "\n";

    // dycore.finalize( coupler );

    yakl::timer_stop("main");
  }
  yakl::finalize();
}
