
#include "coupler.h"
#include "dynamics_euler_stratified_wenofv.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  yakl::init();
  {
    using yakl::intrinsics::abs;
    using yakl::intrinsics::maxval;
    yakl::timer_start("main");

    // This holds all of the model's variables, dimension sizes, and options
    core::Coupler coupler;

    // Read the YAML input file for variables pertinent to running the driver
    if (argc <= 1) { endrun("ERROR: Must pass the input YAML filename as a parameter"); }
    std::string inFile(argv[1]);
    YAML::Node config = YAML::LoadFile(inFile);
    if ( !config            ) { endrun("ERROR: Invalid YAML input file"); }
    auto sim_time  = config["sim_time"].as<real>(5.);
    auto nens      = config["nens"    ].as<int>(1);
    auto nx_glob   = config["nx_glob" ].as<size_t>();
    auto ny_glob   = config["ny_glob" ].as<size_t>();
    auto nz        = config["nz"      ].as<int>();
    auto xlen      = config["xlen"    ].as<real>(1.);
    auto ylen      = config["ylen"    ].as<real>(1.);
    auto zlen      = config["zlen"    ].as<real>(1.);

    coupler.set_option<std::string>( "out_prefix"      , config["out_prefix"      ].as<std::string>() );
    coupler.set_option<std::string>( "init_data"       , config["init_data"       ].as<std::string>() );
    coupler.set_option<real       >( "out_freq"        , config["out_freq"        ].as<real       >() );
    coupler.set_option<bool       >( "enable_gravity"  , config["enable_gravity"  ].as<bool       >(false));
    coupler.set_option<bool       >( "file_per_process", config["file_per_process"].as<bool       >(false));
    coupler.set_option<real       >( "kh_alpha"        , config["kh_alpha"        ].as<real       >(0.25) );

    // Coupler state is: (1) dry density;  (2) u-velocity;  (3) v-velocity;  (4) w-velocity;  (5) temperature
    //                   (6+) tracer masses (*not* mixing ratios!)
    coupler.distribute_mpi_and_allocate_coupled_state(nz, ny_glob, nx_glob, nens);

    // Just tells the coupler how big the domain is in each dimensions
    coupler.set_grid( xlen , ylen , zlen );

    // This is for the dycore to pull out to determine how to do idealized test cases
    coupler.set_option<std::string>( "standalone_input_file" , inFile );

    // They dynamical core "dycore" integrates the Euler equations and performans transport of tracers
    modules::Dynamics_Euler_Stratified_WenoFV  dycore;

    coupler.add_tracer("water_vapor","water_vapor",true,true);
    coupler.get_data_manager_readwrite().get<real,4>("water_vapor") = 0;

    // Run the initialization modules
    dycore.init( coupler ); // Dycore should initialize its own state here

    real etime = 0;   // Elapsed time

    while (etime < sim_time) {
      real dtphys = dycore.compute_time_step(coupler);
      if (etime + dtphys > sim_time) { dtphys = sim_time - etime; }
      dycore.time_step( coupler , dtphys );
      etime += dtphys;
    }

    yakl::timer_stop("main");
  }
  yakl::finalize();
  MPI_Finalize();
}


