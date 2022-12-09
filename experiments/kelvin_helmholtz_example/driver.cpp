
#include "coupler.h"
#include "dynamics_euler_energy.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  yakl::init();
  {
    yakl::timer_start("main");

    // This holds all of the model's variables, dimension sizes, and options
    core::Coupler coupler;

    // Read the YAML input file for variables pertinent to running the driver
    if (argc <= 1) { endrun("ERROR: Must pass the input YAML filename as a parameter"); }
    std::string inFile(argv[1]);
    YAML::Node config = YAML::LoadFile(inFile);
    if ( !config            ) { endrun("ERROR: Invalid YAML input file"); }
    real   sim_time  = config["sim_time"].as<real>();
    size_t nx_glob   = config["nx_glob" ].as<size_t>();
    size_t ny_glob   = config["ny_glob" ].as<size_t>();
    int    nz        = config["nz"      ].as<int>();
    real   xlen      = config["xlen"    ].as<real>();
    real   ylen      = config["ylen"    ].as<real>();
    real   zlen      = config["zlen"    ].as<real>();

    coupler.set_option<std::string>( "standalone_input_file" , inFile );
    coupler.set_option<std::string>( "out_fname" , config["out_fname"].as<std::string>() );
    coupler.distribute_mpi_and_allocate_coupled_state(nz, ny_glob, nx_glob);
    coupler.set_grid( xlen , ylen , zlen );

    modules::Dynamics_Euler_Energy  dycore;
    dycore.init( coupler );

    real etime = 0;   // Elapsed time

    while (etime < sim_time) {
      real dt = dycore.compute_time_step(coupler);
      if (etime + dt > sim_time) { dt = sim_time - etime; }
      dycore.time_step( coupler , dt );
      etime += dt;
    }

    yakl::timer_stop("main");
  }
  yakl::finalize();
  MPI_Finalize();
}


