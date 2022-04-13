
#pragma once

#include <torch/script.h> // One-stop header.
#include <torch/cuda.h>


// Initializes a Torch module, and returns the ID of that module
int torch_add_module( std::string fname );


// Return the number of CUDA-capable devices
int torch_get_cuda_device_count();


// Move a module to the GPU
void torch_move_module_to_gpu(int id, int devicenum);


// Add a tensor (on the CPU)
int torch_add_tensor( real * data , std::vector<int64_t> dims );


// Move a tensor to the GPU
void torch_move_tensor_to_gpu(int id , int devicenum);


// Move a tensor to the CPU
void torch_move_tensor_to_cpu(int id);


// Apply the model, and return the tensor ID of the output
at::Tensor torch_module_forward( int module_id , std::vector<int> input_ids );


// Convert a tensor to a vector
void torch_tensor_to_vector( int tensor_id , std::vector<std::vector<float>> &vec );

