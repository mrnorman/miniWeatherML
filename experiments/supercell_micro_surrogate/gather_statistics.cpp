
#include "coupler.h"
#include "Dycore.h"
#include "Microphysics.h"
#include "sponge_layer.h"
#include "perturb_temperature.h"
#include "column_nudging.h"
#include "gather_micro_statistics.h"

int main(int argc, char** argv) {
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
    real sim_time  = config["sim_time"].as<real>();
    int  nx        = config["nx"      ].as<int>();
    int  ny        = config["ny"      ].as<int>();
    int  nz        = config["nz"      ].as<int>();
    real xlen      = config["xlen"    ].as<real>();
    real ylen      = config["ylen"    ].as<real>();
    real zlen      = config["zlen"    ].as<real>();
    real dtphys_in = config["dt_phys" ].as<real>();

    // The column nudger nudges the column-average of the model state toward the initial column-averaged state
    // This is primarily for the supercell test case to keep the the instability persistently strong
    modules::ColumnNudger column_nudger;
    // Microphysics performs water phase changess + hydrometeor production, transport, collision, and aggregation
    Microphysics          micro;
    // They dynamical core "dycore" integrates the Euler equations and performans transport of tracers
    Dycore                dycore;

    // To gather statistics on how frequently microphysics is active
    custom_modules::StatisticsGatherer statistics_gatherer;

    coupler.set_phys_constants( micro.R_d , micro.R_v , micro.cp_d , micro.cp_v , micro.grav , micro.p0 );

    // Coupler state is: (1) dry density;  (2) u-velocity;  (3) v-velocity;  (4) w-velocity;  (5) temperature
    //                   (6+) tracer masses (*not* mixing ratios!)
    coupler.allocate_coupler_state( nz, ny, nx );

    // Just tells the coupler how big the domain is in each dimensions
    coupler.set_grid( xlen , ylen , zlen );

    // This is for the dycore to pull out to determine how to do idealized test cases
    coupler.set_option<std::string>( "standalone_input_file" , inFile );

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
      dycore.time_step             ( coupler , dtphys );

      // Get coupler state before and after microphysics
      core::Coupler input;
      coupler.clone_into(input);
      micro .time_step             ( coupler , dtphys );
      statistics_gatherer.gather_micro_statistics(input,coupler,dtphys,etime);

      modules::sponge_layer        ( coupler , dtphys );  // Damp spurious waves to the horiz. mean at model top
      column_nudger.nudge_to_column( coupler , dtphys );

      etime += dtphys; // Advance elapsed time
    }

    // TODO: Add finalize( coupler ) modules here
    statistics_gatherer.finalize( coupler );

    yakl::timer_stop("main");
  }
  yakl::finalize();
}


