
#include "coupler.h"
#include "dynamics_euler_stratified_wenofv.h"
// ===================Torch================
#include "microphysics_torch.h"
#include "sponge_layer.h"
#include "perturb_temperature.h"
#include "column_nudging.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  yakl::init();
  {
    using yakl::intrinsics::abs;
    using yakl::intrinsics::maxval;
    using yakl::c::parallel_for;
    using yakl::c::Bounds;
    yakl::timer_start("main");

    // This holds all of the model's variables, dimension sizes, and options
    core::Coupler coupler;

    // Read the YAML input file for variables pertinent to running the driver
    if (argc <= 2) { endrun("ERROR: Must pass the input YAML filenames as a parameter"); }
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
    real   dtphys_in = config["dt_phys" ].as<real>();

    // Coupler state is: (1) dry density;  (2) u-velocity;  (3) v-velocity;  (4) w-velocity;  (5) temperature
    //                   (6+) tracer masses (*not* mixing ratios!)
    coupler.distribute_mpi_and_allocate_coupled_state(nz, ny_glob, nx_glob);

    // Just tells the coupler how big the domain is in each dimensions
    coupler.set_grid( xlen , ylen , zlen );

    // This is for the dycore to pull out to determine how to do idealized test cases
    coupler.set_option<std::string>( "standalone_input_file" , inFile );

    // The column nudger nudges the column-average of the model state toward the initial column-averaged state
    // This is primarily for the supercell test case to keep the the instability persistently strong
    modules::ColumnNudger                     column_nudger;
    
    // ===================Torch================
    // Microphysics surrogate for Kessler performs water phase changess + hydrometeor production, transport, collision, and aggregation
    custom_modules::Microphysics_NN             micro_nn;
    
    // They dynamical core "dycore" integrates the Euler equations and performans transport of tracers
    modules::Dynamics_Euler_Stratified_WenoFV dycore;

    coupler.set_phys_constants( micro_nn.R_d , micro_nn.R_v , micro_nn.cp_d , micro_nn.cp_v , micro_nn.grav , micro_nn.p0 );

    // Run the initialization modules
    micro_nn .init              ( coupler ); // Allocate micro state and register its tracers in the coupler
    dycore.init                 ( coupler ); // Dycore should initialize its own state here
    column_nudger.set_column    ( coupler ); // Set the column before perturbing
    modules::perturb_temperature( coupler ); // Randomly perturb bottom layers of temperature to initiate convection

    // ===================Torch================
    // Load the NN variables
    std::cout << "*** BEGIN: Loading NN Models ***\n";
    // Read the YAML input file for variables pertinent to the NN model
    std::string file_nn(argv[2]);
    YAML::Node config_nn = YAML::LoadFile(file_nn);
    
    // Load the NN model
    int mod_id = torch_add_module( config_nn["in_file_nn"].as<std::string>() );
    // Set device: CPU or GPU
    int torchDevice = -1;   // custom devicenum input from user; default -1 is CPU
    if(const char* env_p = std::getenv("TORCH_DEVICE")){
      torchDevice = *env_p - '0';
    }
    int devicenum = -1;    // default is CPU
    int devicecount = torch_get_cuda_device_count();
    if (devicecount > 0 and torchDevice >= 0){
      devicenum = torchDevice;
      std::cout << "Running PyTorch on GPU device number: " << devicenum << "\n";
    }
    else {
      std::cout << "Running PyTorch on CPU\n";
    }
    torch_move_module_to_gpu( mod_id , devicenum );
    std::cout << "*** END:   Loading NN Models ***\n";

    // Load the data scaling arrays
    std::cout << "*** BEGIN: Loading scaling arrays ***\n";
    std::ifstream file1;
    // input scaler
    real3d scl_in("scl_in" ,4,3,2);
    real scl_Lin[4][3][2];
    file1.open( config_nn["in_file_sclin"].as<std::string>() );
    for (int l = 0; l < 4; l++) {
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) { 
          file1 >> scl_Lin[l][i][j];
        }
      }
    }
    parallel_for( Bounds<3>(4,3,2) , YAKL_LAMBDA (int l, int i, int j) {
        scl_in(l,i,j) = scl_Lin[l][i][j];
        });
    file1.close();
    // output scaler
    real2d scl_out("scl_out",4,2);
    real scl_Lout[4][2];
    file1.open( config_nn["in_file_sclfi"].as<std::string>() );
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 2; j++) {
        file1 >> scl_Lout[i][j];
      }
    }
    parallel_for( Bounds<2>(4,2) , YAKL_LAMBDA (int i, int j) {
        scl_out(i,j) = scl_Lout[i][j];
        });
    file1.close();
    std::cout << "*** END:   Loading scaling arrays ***\n";

    ////////////////////////////////////////////////////////////////
    // main time iteration loop
    ////////////////////////////////////////////////////////////////
    real etime  = 0;      // Elapsed time
    real dtphys = dtphys_in;
    while (etime < sim_time) {
      // If dtphys <= 0, then set it to the dynamical core's max stable time step
      if (dtphys_in <= 0.) { dtphys = dycore.compute_time_step(coupler); }
      // If we're about to go past the final time, then limit to time step to exactly hit the final time
      if (etime + dtphys > sim_time) { dtphys = sim_time - etime; }

      // Run the runtime modules
      dycore.time_step             ( coupler , dtphys );

      // ===================Torch================
      // Run microphysics using NN surrogate
      micro_nn .time_step          ( coupler , dtphys, etime/sim_time , time_init_nn , scl_in , scl_out , devicenum , mod_id);

      modules::sponge_layer        ( coupler , dtphys );  // Damp spurious waves to the horiz. mean at model top
      column_nudger.nudge_to_column( coupler , dtphys );

      etime += dtphys; // Advance elapsed time
    }

    yakl::timer_stop("main");
  }
  yakl::finalize();
  MPI_Finalize();
}


