
#include "main_header.h"
#include "coupler.h"


int main(int argc, char** argv) {
  yakl::init();
  {
    using yakl::intrinsics::abs;
    using yakl::intrinsics::maxval;
    yakl::timer_start("main");

    // if (argc <= 1) { endrun("ERROR: Must pass the input YAML filename as a parameter"); }
    // std::string inFile(argv[1]);
    // YAML::Node config = YAML::LoadFile(inFile);
    // if ( !config            ) { endrun("ERROR: Invalid YAML input file"); }
    // real simTime        = config["simTime"    ].as<real>();
    // int  crm_nx         = config["crm_nx"     ].as<int>();
    // int  crm_ny         = config["crm_ny"     ].as<int>();
    // int  nens           = config["nens"       ].as<int>();
    // real xlen           = config["xlen"       ].as<real>();
    // real ylen           = config["ylen"       ].as<real>();
    // real dt_gcm         = config["dt_gcm"     ].as<real>();
    // real dt_crm_phys    = config["dt_crm_phys"].as<real>();
    // std::string coldata = config["column_data"].as<std::string>();
    // bool advect_tke     = config["advect_tke" ].as<bool>();

    Coupler coupler;

    // This is for the dycore to pull out to determine how to do idealized test cases
    // coupler.set_option<std::string>( "standalone_input_file" , inFile );

    // // Store vertical coordinates
    // std::string vcoords_file = config["vcoords"].as<std::string>();
    // yakl::SimpleNetCDF nc;
    // nc.open(vcoords_file);
    // int crm_nz = nc.getDimSize("num_interfaces") - 1;
    // real1d zint_in;
    // nc.read(zint_in,"vertical_interfaces");
    // nc.close();

    // // Allocates the coupler state (density_dry, uvel, vvel, wvel, temp, vert grid, hydro background) for thread 0
    // coupler.allocate_coupler_state( crm_nz , crm_ny , crm_nx , nens );

    // // NORMALLY THIS WOULD BE DONE INSIDE THE CRM, BUT WE'RE USING CONSTANTS DEFINED BY THE CRM MICRO SCHEME
    // // Create the dycore and the microphysics
    // Dycore       dycore;
    // Microphysics micro;
    // SGS          sgs;

    // // Set physical constants for coupler at thread 0 using microphysics data
    // coupler.set_phys_constants( micro.R_d , micro.R_v , micro.cp_d , micro.cp_v , micro.grav , micro.p0 );

    // // Set the vertical grid in the coupler
    // coupler.set_grid( xlen , ylen , zint_in );

    // coupler.dm.register_and_allocate<real>("gcm_density_dry","GCM column dry density"     ,{crm_nz,nens});
    // coupler.dm.register_and_allocate<real>("gcm_uvel"       ,"GCM column u-velocity"      ,{crm_nz,nens});
    // coupler.dm.register_and_allocate<real>("gcm_vvel"       ,"GCM column v-velocity"      ,{crm_nz,nens});
    // coupler.dm.register_and_allocate<real>("gcm_wvel"       ,"GCM column w-velocity"      ,{crm_nz,nens});
    // coupler.dm.register_and_allocate<real>("gcm_temp"       ,"GCM column temperature"     ,{crm_nz,nens});
    // coupler.dm.register_and_allocate<real>("gcm_water_vapor","GCM column water vapor mass",{crm_nz,nens});

    // micro .init( coupler );
    // sgs   .init( coupler );
    // dycore.init( coupler );  // dycore should set idealized conditions here

    // #ifdef PAM_STANDALONE
    //   std::cout << "Dycore: " << dycore.dycore_name() << std::endl;
    //   std::cout << "Micro : " << micro .micro_name () << std::endl;
    //   std::cout << "SGS   : " << sgs   .sgs_name   () << std::endl;
    //   std::cout << "\n";
    // #endif

    // auto gcm_rho_d = coupler.dm.get<real,2>("gcm_density_dry");
    // auto gcm_uvel  = coupler.dm.get<real,2>("gcm_uvel"       );
    // auto gcm_vvel  = coupler.dm.get<real,2>("gcm_vvel"       );
    // auto gcm_wvel  = coupler.dm.get<real,2>("gcm_wvel"       );
    // auto gcm_temp  = coupler.dm.get<real,2>("gcm_temp"       );
    // auto gcm_rho_v = coupler.dm.get<real,2>("gcm_water_vapor");

    // auto rho_d = coupler.dm.get<real const,4>("density_dry" );
    // auto uvel  = coupler.dm.get<real const,4>("uvel"        );
    // auto vvel  = coupler.dm.get<real const,4>("vvel"        );
    // auto wvel  = coupler.dm.get<real const,4>("wvel"        );
    // auto temp  = coupler.dm.get<real const,4>("temp"        );
    // auto rho_v = coupler.dm.get<real const,4>("water_vapor" );

    // // Compute a column to force the model with by averaging the columns at init
    // parallel_for( Bounds<4>(crm_nz,crm_ny,crm_nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
    //   gcm_rho_d(k,iens) = 0;
    //   gcm_uvel (k,iens) = 0;
    //   gcm_vvel (k,iens) = 0;
    //   gcm_wvel (k,iens) = 0;
    //   gcm_temp (k,iens) = 0;
    //   gcm_rho_v(k,iens) = 0;
    // });
    // real r_nx_ny = 1._fp / (crm_nx*crm_ny);  // Avoid costly divisions
    // parallel_for( Bounds<4>(crm_nz,crm_ny,crm_nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
    //   yakl::atomicAdd( gcm_rho_d(k,iens) , rho_d(k,j,i,iens) * r_nx_ny );
    //   yakl::atomicAdd( gcm_uvel (k,iens) , uvel (k,j,i,iens) * r_nx_ny );
    //   yakl::atomicAdd( gcm_vvel (k,iens) , vvel (k,j,i,iens) * r_nx_ny );
    //   yakl::atomicAdd( gcm_wvel (k,iens) , wvel (k,j,i,iens) * r_nx_ny );
    //   yakl::atomicAdd( gcm_temp (k,iens) , temp (k,j,i,iens) * r_nx_ny );
    //   yakl::atomicAdd( gcm_rho_v(k,iens) , rho_v(k,j,i,iens) * r_nx_ny );
    // });

    // perturb_temperature( coupler , 0 );

    // // Now that we have an initial state, define hydrostasis for each ensemble member
    // coupler.update_hydrostasis( coupler.compute_pressure_array() );

    // coupler.add_mmf_function( "sgs"    , [&] (PamCoupler &coupler, real dt) { sgs   .timeStep(coupler,dt); } );
    // coupler.add_mmf_function( "micro"  , [&] (PamCoupler &coupler, real dt) { micro .timeStep(coupler,dt); } );
    // coupler.add_mmf_function( "dycore" , [&] (PamCoupler &coupler, real dt) { dycore.timeStep(coupler,dt); } );

    // bool forcing_at_dycore_time_step = coupler.get_option<bool>("forcing_at_dycore_time_step");

    // if (forcing_at_dycore_time_step) {
    //   if ( coupler.get_option<std::string>("density_forcing") == "strict" ) {
    //     coupler.add_dycore_function( "gcm_density_forcing" , gcm_density_forcing );
    //   }
    //   coupler.add_dycore_function( "apply_gcm_forcing_tendencies" , apply_gcm_forcing_tendencies );
    //   coupler.add_dycore_function( "sponge_layer"                 , sponge_layer                 );
    // } else {
    //   if ( coupler.get_option<std::string>("density_forcing") == "strict" ) {
    //     coupler.add_mmf_function( "gcm_density_forcing" , gcm_density_forcing );
    //   }
    //   coupler.add_mmf_function( "apply_gcm_forcing_tendencies" , apply_gcm_forcing_tendencies );
    //   coupler.add_mmf_function( "sponge_layer"                 , sponge_layer                 );
    // }

    // // coupler.add_dycore_function( "saturation_adjustment" , saturation_adjustment );

    // std::cout << "The following functions are called at the MMF time step:\n";
    // coupler.print_mmf_functions();
    // std::cout << "\n";

    // std::cout << "The following functions are called within the dycore at the dycore time step:\n";
    // coupler.print_dycore_functions();
    // std::cout << "\n";

    // real etime_gcm = 0;

    // while (etime_gcm < simTime) {
    //   if (etime_gcm + dt_gcm > simTime) { dt_gcm = simTime - etime_gcm; }

    //   compute_gcm_forcing_tendencies( coupler , dt_gcm );

    //   real etime_crm = 0;
    //   real simTime_crm = dt_gcm;
    //   real dt_crm = dt_crm_phys;
    //   while (etime_crm < simTime_crm) {
    //     if (dt_crm == 0.) { dt_crm = dycore.compute_time_step(coupler); }
    //     if (etime_crm + dt_crm > simTime_crm) { dt_crm = simTime_crm - etime_crm; }

    //     // You can run things this way:
    //     coupler.run_mmf_function( "sgs"    , dt_crm );
    //     coupler.run_mmf_function( "micro"  , dt_crm );
    //     // Dycore runs all functions added via coupler.add_dycore_function in the order they were added after
    //     //   each dycore time step
    //     coupler.run_mmf_function( "dycore" , dt_crm );
    //     if (! forcing_at_dycore_time_step) {
    //       coupler.run_mmf_function( "apply_gcm_forcing_tendencies" , dt_crm );
    //       coupler.run_mmf_function( "sponge_layer"                 , dt_crm );
    //     }

    //     // OR you can run it this way, which does the same thing:
    //     // coupler.run_mmf_functions(dt_crm);

    //     etime_crm += dt_crm;
    //     etime_gcm += dt_crm;
    //     real maxw = maxval(abs(coupler.dm.get_collapsed<real const>("wvel")));
    //     std::cout << "Etime , dtphys, maxw: " << etime_gcm << " , " 
    //                                           << dt_crm    << " , "
    //                                           << std::setw(10) << maxw << "\n";
    //   }
    // }

    // std::cout << "Elapsed Time: " << etime_gcm << "\n";

    // dycore.finalize( coupler );

    yakl::timer_stop("main");
  }
  yakl::finalize();
}


