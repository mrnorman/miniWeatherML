
#pragma once

namespace custom_modules {


  struct Train_WENO {
    int          batch_size_per_rank;
    int          num_batches;
    int          data_size;
    int          nranks;
    MPI_Datatype mpi_dtype;

    void init( core::Coupler const &coupler , int desired_batch_size ) {
      data_size = coupler.get_nx()*coupler.get_ny()*coupler.get_nz();
      nranks = coupler.get_nranks();
      mpi_dtype = coupler.get_mpi_data_type();
      if (coupler.get_myrank() == 0) {
        if (desired_batch_size < 0) desired_batch_size = nranks;
        batch_size_per_rank = std::max( 1 , (int) std::round(static_cast<real>(desired_batch_size)/nranks) );
      }
      MPI_Bcast(&batch_size_per_rank,1,MPI_INT,0,MPI_COMM_WORLD);
      int num_batches_loc = (int) std::ceil(static_cast<real>(data_size) / batch_size_per_rank);
      MPI_Allreduce( &num_batches_loc , &num_batches , 1 , MPI_INT , MPI_MAX , MPI_COMM_WORLD );
    }

    template <class ENSEMBLE, class TRAINER>
    void train_mini_batches( TRAINER &trainer , ENSEMBLE &ensemble , real2d const &loss ) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;
      int num_ensembles = ensemble.get_ensemble_size();
      ponni::shuffle_data(loss,1); // Shuffle the batch dimension of the losses
      real2d batch_loss("batch_loss",num_ensembles,batch_size_per_rank);
      for (int ibatch = 0; ibatch < num_batches; ibatch++) {
        int ibeg =  ibatch   *batch_size_per_rank;
        int iend = (ibatch+1)*batch_size_per_rank-1;
        parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(num_ensembles,{ibeg,iend}) ,
                                          YAKL_LAMBDA (int iens, int ibatch_glob) {
          int ibatch_loc = ibatch_glob - ibeg;  // local batch index
          while (ibatch_glob >= data_size) { ibatch_glob -= data_size; }  // periodic wrapping
          batch_loss(iens,ibatch_loc) = loss(iens,ibatch_glob);
        });
        realHost1d losses_loc_host("losses_loc_host",num_ensembles);
        for (int iens = 0; iens < num_ensembles; iens++) {
          auto slice = batch_loss.slice<1>(iens,yakl::COLON);
          losses_loc_host(iens) = yakl::intrinsics::sum( slice );
        }
        realHost1d losses_glob_host("losses_glob_host",num_ensembles);
        MPI_Allreduce( losses_loc_host.data() , losses_glob_host.data() , losses_loc_host.size() ,
                       mpi_dtype , MPI_SUM , MPI_COMM_WORLD );
        auto losses_glob = losses_glob_host.createDeviceObject();
        losses_glob_host.deep_copy_to(losses_glob);
        auto ensemble_loss = ensemble.get_loss();
        parallel_for( YAKL_AUTO_LABEL() , num_ensembles , YAKL_LAMBDA (int iens) {
          ensemble_loss(iens) = losses_glob(iens) / (batch_size_per_rank * nranks);
        });
        trainer.update_from_ensemble( ensemble );
      }
    }
  };

}


