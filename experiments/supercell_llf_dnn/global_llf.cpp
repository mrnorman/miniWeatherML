
#include "coupler.h"
#include "dynamics_euler_stratified_wenofv.h"
#include "microphysics_kessler.h"
#include "sponge_layer.h"
#include "perturb_temperature.h"
#include "column_nudging.h"
#include "Experiment_Manager.h"
#include "ponni_Trainer_GD_Adam_FD.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  yakl::init();
  {
    using yakl::intrinsics::abs;
    using yakl::intrinsics::maxval;
    yakl::timer_start("main");
    #ifndef MW_ORD
      int  static constexpr ord = 5;
    #else
      int  static constexpr ord = MW_ORD;
    #endif

    // This holds all of the model's variables, dimension sizes, and options
    core::Coupler coupler_lo;
    core::Coupler coupler_hi;

    // Read the YAML input file for variables pertinent to running the driver
    if (argc <= 1) { endrun("ERROR: Must pass the input YAML filename as a parameter"); }
    std::string inFile(argv[1]);
    YAML::Node config = YAML::LoadFile(inFile);
    if ( !config            ) { endrun("ERROR: Invalid YAML input file"); }
    auto sim_time      = config["sim_time"     ].as<real>();
    auto nx_glob       = config["nx_glob"      ].as<size_t>();
    auto ny_glob       = config["ny_glob"      ].as<size_t>();
    auto nz            = config["nz"           ].as<int>();
    auto xlen          = config["xlen"         ].as<real>();
    auto ylen          = config["ylen"         ].as<real>();
    auto zlen          = config["zlen"         ].as<real>();
    auto refine_factor = config["refine_factor"].as<int>(4);

    coupler_lo.set_option<std::string>( "out_prefix" , config["out_prefix"].as<std::string>()+std::string("_lo") );
    coupler_lo.set_option<std::string>( "init_data"  , config["init_data" ].as<std::string>() );
    coupler_lo.set_option<real       >( "out_freq"   , config["out_freq"  ].as<real       >() );
    coupler_lo.set_option<std::string>( "standalone_input_file" , inFile );

    coupler_hi.set_option<std::string>( "out_prefix" , config["out_prefix"].as<std::string>()+std::string("_hi") );
    coupler_hi.set_option<std::string>( "init_data"  , config["init_data" ].as<std::string>() );
    coupler_hi.set_option<real       >( "out_freq"   , config["out_freq"  ].as<real       >() );
    coupler_hi.set_option<std::string>( "standalone_input_file" , inFile );

    // Create the trainer
    int num_trainable_parameters = 2;
    real1d trainable_parameters("trainable_parameters",num_trainable_parameters);
    trainable_parameters = 0.5;
    ponni::Trainer_GD_Adam_FD<real> trainer( trainable_parameters );
    int nens_lo = trainer.get_num_ensembles();
    auto &dm_lo = coupler_lo.get_data_manager_readwrite();
    dm_lo.register_and_allocate<real>("trainable_parameters","",{num_trainable_parameters,nens_lo});

    // Create lo-res coupler state
    coupler_lo.distribute_mpi_and_allocate_coupled_state(nz, ny_glob, nx_glob, nens_lo);
    // Create hi-res coupler state to share the lo-res's exact physical domain for each MPI task
    int refine_factor_y = coupler_lo.is_sim2d() ? 1 : refine_factor;
    int nens_hi = 1;
    coupler_hi.distribute_mpi_and_allocate_coupled_state(nz     *refine_factor                        ,
                                                         ny_glob*refine_factor_y                      ,
                                                         nx_glob*refine_factor                        ,
                                                         nens_hi                                      ,
                                                         coupler_lo.get_nproc_x()                     ,
                                                         coupler_lo.get_nproc_y()                     ,
                                                          coupler_lo.get_i_beg()   *refine_factor     ,
                                                         (coupler_lo.get_i_end()+1)*refine_factor-1   ,
                                                          coupler_lo.get_j_beg()   *refine_factor_y   ,
                                                         (coupler_lo.get_j_end()+1)*refine_factor_y-1 );

    // Just tells the coupler how big the domain is in each dimensions
    coupler_lo.set_grid( xlen , ylen , zlen );
    coupler_hi.set_grid( xlen , ylen , zlen );

    // The column nudger nudges the column-average of the model state toward the initial column-averaged state
    // This is primarily for the supercell test case to keep the the instability persistently strong
    modules::ColumnNudger                           column_nudger_lo;
    modules::ColumnNudger                           column_nudger_hi;
    // Microphysics performs water phase changess + hydrometeor production, transport, collision, and aggregation
    modules::Microphysics_Kessler                   micro_lo;
    modules::Microphysics_Kessler                   micro_hi;
    // They dynamical core "dycore" integrates the Euler equations and performans transport of tracers
    modules::Dynamics_Euler_Stratified_WenoFV<ord>  dycore_lo;
    modules::Dynamics_Euler_Stratified_WenoFV<ord>  dycore_hi;

    // Run the initialization modules
    micro_lo .init              ( coupler_lo ); // Allocate micro state and register its tracers in the coupler
    dycore_lo.init              ( coupler_lo ); // Dycore should initialize its own state here
    column_nudger_lo.set_column ( coupler_lo ); // Set the column before perturbing
    modules::perturb_temperature( coupler_lo ); // Randomly perturb bottom layers of temperature to initiate convection

    micro_hi .init              ( coupler_hi ); // Allocate micro state and register its tracers in the coupler
    dycore_hi.init              ( coupler_hi ); // Dycore should initialize its own state here
    column_nudger_hi.set_column ( coupler_hi ); // Set the column before perturbing
    modules::perturb_temperature( coupler_hi ); // Randomly perturb bottom layers of temperature to initiate convection

    // Keep track of loss and parameter values
    custom_modules::Experiment_Manager  exp_manager;

    real etime = 0;   // Elapsed time

    while (etime < sim_time) {
      real dtphys = dycore_lo.compute_time_step(coupler_lo);  // Use lo-res model to determine time step
      // If we're about to go past the final time, then limit to time step to exactly hit the final time
      if (etime + dtphys > sim_time) { dtphys = sim_time - etime; }

      // Run the dycore
      exp_manager.overwrite_lo( coupler_lo , coupler_hi );
      dycore_hi.time_step( coupler_hi , dtphys );  // Flow forward according to the Euler equations
      auto ensemble = trainer.get_ensemble();
      ensemble.get_parameters().deep_copy_to(dm_lo.get<real,2>("trainable_parameters"));
      dycore_lo.time_step( coupler_lo , dtphys );  // Flow forward according to the Euler equations
      // TODO: Add ponni trainer to the argument list to penalize beta values out of physical range
      exp_manager.compute_loss_and_overwrite_lo( coupler_lo , coupler_hi , ensemble.get_loss() );
      trainer.update_from_ensemble( ensemble );
      trainer.increment_epoch();

      // Run hi-res modules
      micro_hi .time_step             ( coupler_hi , dtphys );  // Phase changes for water + precipitation / falling
      modules::sponge_layer           ( coupler_hi , dtphys );  // Damp spurious waves to the horiz. mean at model top
      column_nudger_hi.nudge_to_column( coupler_hi , dtphys );  // Nudge slowly back toward unstable profile

      // // Run lo-res modules
      // micro_lo .time_step             ( coupler_lo , dtphys );  // Phase changes for water + precipitation / falling
      // modules::sponge_layer           ( coupler_lo , dtphys );  // Damp spurious waves to the horiz. mean at model top
      // column_nudger_lo.nudge_to_column( coupler_lo , dtphys );  // Nudge slowly back toward unstable profile

      etime += dtphys; // Advance elapsed time
    }

    // TODO: Add finalize( coupler ) modules here

    yakl::timer_stop("main");
  }
  yakl::finalize();
  MPI_Finalize();
}


