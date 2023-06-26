
#include "coupler.h"
#include "dynamics_euler_stratified_wenofv_subtract_exact.h"
#include "microphysics_kessler.h"
#include "sponge_layer.h"
#include "perturb_temperature.h"
#include "column_nudging.h"

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
    auto sim_time  = config["sim_time"].as<real>();
    auto nens      = config["nens"    ].as<int>();
    auto nx_glob   = config["nx_glob" ].as<size_t>();
    auto ny_glob   = config["ny_glob" ].as<size_t>();
    auto nz        = config["nz"      ].as<int>();
    auto xlen      = config["xlen"    ].as<real>();
    auto ylen      = config["ylen"    ].as<real>();
    auto zlen      = config["zlen"    ].as<real>();
    auto dtphys_in = config["dt_phys" ].as<real>();

    coupler.set_option<std::string>( "out_prefix" , config["out_prefix"].as<std::string>() );
    coupler.set_option<std::string>( "init_data" , config["init_data"].as<std::string>() );
    coupler.set_option<real       >( "out_freq"  , config["out_freq" ].as<real       >() );

    // Coupler state is: (1) dry density;  (2) u-velocity;  (3) v-velocity;  (4) w-velocity;  (5) temperature
    //                   (6+) tracer masses (*not* mixing ratios!)
    coupler.distribute_mpi_and_allocate_coupled_state(nz, ny_glob, nx_glob, nens);

    // Just tells the coupler how big the domain is in each dimensions
    coupler.set_grid( xlen , ylen , zlen );

    // This is for the dycore to pull out to determine how to do idealized test cases
    coupler.set_option<std::string>( "standalone_input_file" , inFile );

    // The column nudger nudges the column-average of the model state toward the initial column-averaged state
    // This is primarily for the supercell test case to keep the the instability persistently strong
    modules::ColumnNudger                     column_nudger;
    // Microphysics performs water phase changess + hydrometeor production, transport, collision, and aggregation
    modules::Microphysics_Kessler             micro;
    // They dynamical core "dycore" integrates the Euler equations and performans transport of tracers
    modules::Dynamics_Euler_Stratified_WenoFV dycore;

    // Run the initialization modules
    micro .init                 ( coupler ); // Allocate micro state and register its tracers in the coupler
    dycore.init                 ( coupler ); // Dycore should initialize its own state here
    column_nudger.set_column    ( coupler ); // Set the column before perturbing
    modules::perturb_temperature( coupler ); // Randomly perturb bottom layers of temperature to initiate convection

    real etime = 0;   // Elapsed time

    real dtphys = dtphys_in;
    while (etime < sim_time) {
      // If dtphys <= 0, then set it to the dynamical core's max stable time step
      if (dtphys_in <= 0.) { dtphys = dycore.compute_time_step(coupler); }
      // If we're about to go past the final time, then limit to time step to exactly hit the final time
      if (etime + dtphys > sim_time) { dtphys = sim_time - etime; }

      // Run the runtime modules
      dycore.time_step             ( coupler , dtphys );  // Move the flow forward according to the Euler equations
      micro .time_step             ( coupler , dtphys );  // Perform phase changes for water + precipitation / falling
      modules::sponge_layer        ( coupler , dtphys );  // Damp spurious waves to the horiz. mean at model top
      column_nudger.nudge_to_column( coupler , dtphys );  // Nudge slightly back toward unstable profile
                                                          // so that supercell persists for all time

      etime += dtphys; // Advance elapsed time
    }

    // TODO: Add finalize( coupler ) modules here

    yakl::timer_stop("main");
  }
  yakl::finalize();
  MPI_Finalize();
}


