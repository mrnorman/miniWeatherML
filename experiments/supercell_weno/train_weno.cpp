
#include "coupler.h"
#include "dynamics_euler_stratified_wenofv.h"
#include "microphysics_kessler.h"
#include "sponge_layer.h"
#include "perturb_temperature.h"
#include "column_nudging.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  yakl::init();
  {
    yakl::timer_start("main");

    // This holds all of the model's variables, dimension sizes, and options
    core::Coupler coupler_lo;
    core::Coupler coupler_hi;

    // Read the YAML input file for variables pertinent to running the driver
    if (argc <= 1) { endrun("ERROR: Must pass the input YAML filename as a parameter"); }
    std::string inFile(argv[1]);
    YAML::Node config = YAML::LoadFile(inFile);
    if ( !config            ) { endrun("ERROR: Invalid YAML input file"); }
    auto sim_time      = config["sim_time"     ].as<real>();
    auto nx_glob       = config["nx_glob_lo"   ].as<size_t>();
    auto ny_glob       = config["ny_glob_lo"   ].as<size_t>();
    auto nz            = config["nz_lo"        ].as<int>();
    auto xlen          = config["xlen"         ].as<real>();
    auto ylen          = config["ylen"         ].as<real>();
    auto zlen          = config["zlen"         ].as<real>();
    auto dtphys_in     = config["dt_phys"      ].as<real>();
    auto refine_factor = config["refine_factor"].as<real>();

    coupler_lo.set_option<std::string>( "out_prefix" , config["out_prefix"].as<std::string>()+std::string("_lo") );
    coupler_lo.set_option<std::string>( "init_data"  , config["init_data" ].as<std::string>() );
    coupler_lo.set_option<real       >( "out_freq"   , config["out_freq"  ].as<real       >() );
    coupler_lo.set_option<std::string>( "standalone_input_file" , inFile );

    coupler_hi.set_option<std::string>( "out_prefix" , config["out_prefix"].as<std::string>()+std::string("_hi") );
    coupler_hi.set_option<std::string>( "init_data"  , config["init_data" ].as<std::string>() );
    coupler_hi.set_option<real       >( "out_freq"   , config["out_freq"  ].as<real       >() );
    coupler_hi.set_option<std::string>( "standalone_input_file" , inFile );

    // Coupler state is: (1) dry density;  (2) u-velocity;  (3) v-velocity;  (4) w-velocity;  (5) temperature
    //                   (6+) tracer masses (*not* mixing ratios!)
    coupler_lo.distribute_mpi_and_allocate_coupled_state(nz, ny_glob, nx_glob, nens);
    auto ny_glob_hi = ny_glob == 1 ? 1 : ny_glob*refine_factor;
    coupler_hi.distribute_mpi_and_allocate_coupled_state(
                                            nz*refine_factor , ny_glob_hi , nx_glob*refine_factor, 1 ,
                                            coupler_lo.get_nproc_x() , coupler_lo.get_nproc_y() ,
                                            coupler_lo.get_px()      , coupler_lo.get_py()      ,
                                            coupler_lo.get_i_beg()   , coupler_lo.get_i_end()   ,
                                            coupler_lo.get_j_beg()   , coupler_lo.get_j_end()   );

    ////////////////////////////////////////////////////////////////////////////
    // BEGIN: PONNI MODEL AND TRAINER CREATION
    ////////////////////////////////////////////////////////////////////////////
    using ponni::create_inference_model;
    using ponni::Matvec;
    using ponni::Bias;
    using ponni::Relu;
    using ponni::Save_State;
    using ponni::Binop_Add;
    using ponni::Trainer_GD_Adam_FD;
    int  num_inputs          = MW_ORD;   // Normalize stencil of inputs
    int  num_outputs         = MW_ORD-1; // WENO parameters outputs
    int  num_neurons         = MW_ORD;   // Size of hidden layers
    int  nens                = 1;
    real relu_negative_slope = 0.3;
    auto model = create_inference_model( Matvec<real>      ( num_inputs,num_neurons,nens     ) ,
                                         Bias  <real>      ( num_neurons,nens                ) ,
                                         Relu  <real>      ( num_neurons,relu_negative_slope ) ,
                                         Save_State<0,real>( num_neurons                     ) ,
                                         Matvec<real>      ( num_neurons,num_neurons,nens    ) ,
                                         Bias  <real>      ( num_neurons,nens                ) ,
                                         Relu  <real>      ( num_neurons,relu_negative_slope ) ,
                                         Binop_Add<0,real> ( num_neurons                     ) ,
                                         Matvec<real>      ( num_neurons,num_outputs,nens    ) ,
                                         Bias  <real>      ( num_outputs,nens                ) );
    auto num_parameters = test.get_num_trainable_parameters();
    Trainer_GD_Adam_FD<real> trainer( test.get_trainable_parameters().reshape(num_parameters) );
    auto nens = test.get_num_trainable_parameters() + 1;
    model = create_inference_model( Matvec<real>      ( num_inputs,num_neurons,nens     ) ,
                                    Bias  <real>      ( num_neurons,nens                ) ,
                                    Relu  <real>      ( num_neurons,relu_negative_slope ) ,
                                    Save_State<0,real>( num_neurons                     ) ,
                                    Matvec<real>      ( num_neurons,num_neurons,nens    ) ,
                                    Bias  <real>      ( num_neurons,nens                ) ,
                                    Relu  <real>      ( num_neurons,relu_negative_slope ) ,
                                    Binop_Add<0,real> ( num_neurons                     ) ,
                                    Matvec<real>      ( num_neurons,num_outputs,nens    ) ,
                                    Bias  <real>      ( num_outputs,nens                ) );
    model.init( coupler_lo.get_nx()*coupler_lo.get_ny()*coupler_lo.get_nz() , nens );
    ////////////////////////////////////////////////////////////////////////////
    // END: PONNI MODEL AND TRAINER CREATION
    ////////////////////////////////////////////////////////////////////////////

    // Just tells the coupler how big the domain is in each dimensions
    coupler_lo.set_grid( xlen , ylen , zlen );
    coupler_hi.set_grid( xlen , ylen , zlen );

    // The column nudger nudges the column-average of the model state toward the initial column-averaged state
    // This is primarily for the supercell test case to keep the the instability persistently strong
    modules::ColumnNudger                             column_nudger_lo;
    modules::ColumnNudger                             column_nudger_hi;
    // Microphysics performs water phase changess + hydrometeor production, transport, collision, and aggregation
    modules::Microphysics_Kessler                     micro_lo;
    modules::Microphysics_Kessler                     micro_hi;
    // They dynamical core "dycore" integrates the Euler equations and performans transport of tracers
    custom_modules::Dynamics_Euler_Stratified_WenoFV  dycore_lo;
    modules::Dynamics_Euler_Stratified_WenoFV         dycore_hi;

    // Run the initialization modules
    micro_lo .init              ( coupler_lo ); // Allocate micro state and register its tracers in the coupler
    dycore_lo.init              ( coupler_lo ); // Dycore should initialize its own state here
    column_nudger_lo.set_column ( coupler_lo ); // Set the column before perturbing
    modules::perturb_temperature( coupler_lo ); // Randomly perturb bottom layers of temperature to initiate convection

    micro_hi .init              ( coupler_hi ); // Allocate micro state and register its tracers in the coupler
    dycore_hi.init              ( coupler_hi ); // Dycore should initialize its own state here
    column_nudger_hi.set_column ( coupler_hi ); // Set the column before perturbing
    modules::perturb_temperature( coupler_hi ); // Randomly perturb bottom layers of temperature to initiate convection

    real etime = 0;   // Elapsed time

    real dtphys = dtphys_in;
    while (etime < sim_time) {
      // If dtphys <= 0, then set it to the low-res dynamical core's max stable time step
      if (dtphys_in <= 0.) { dtphys = dycore_lo.compute_time_step(coupler); }
      // If we're about to go past the final time, then limit to time step to exactly hit the final time
      if (etime + dtphys > sim_time) { dtphys = sim_time - etime; }

      // Run the low resolution model
      auto ensemble = trainer.get_ensemble();
      model.set_trainable_parameters( ensemble.get_parameters() );
      dycore_lo.time_step             ( coupler_lo , dtphys , model );
      micro _lo.time_step             ( coupler_lo , dtphys );
      modules::sponge_layer           ( coupler_lo , dtphys );
      column_nudger_lo.nudge_to_column( coupler_lo , dtphys );

      // Run the high resolution model
      dycore_hi.time_step             ( coupler_hi , dtphys );
      micro _hi.time_step             ( coupler_hi , dtphys );
      modules::sponge_layer           ( coupler_hi , dtphys );
      column_nudger_hi.nudge_to_column( coupler_hi , dtphys );

      // Calculate the loss, and overwrite the low-res state with the high-res state
      auto loss2d = custom_modules::calculate_loss_and_overwrite_lo( coupler_lo , coupler_hi , model );

      etime += dtphys; // Advance elapsed time
    }

    // TODO: Add finalize( coupler ) modules here

    yakl::timer_stop("main");
  }
  yakl::finalize();
  MPI_Finalize();
}


