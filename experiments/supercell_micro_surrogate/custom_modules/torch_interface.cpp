
#include "torch_interface.h"

std::vector<torch::jit::script::Module> torch_modules(0);
std::vector<at::Tensor> torch_tensors(0);


int torch_add_module( std::string fname ) {
  torch_modules.push_back( torch::jit::load( fname.c_str() ) );
  return torch_modules.size()-1;
}


int torch_get_cuda_device_count() {
  return torch::cuda::device_count();
}


void torch_move_module_to_gpu(int id, int devicenum) {
    torch_modules[id].to(torch::Device(torch::kCUDA, devicenum));
}


int torch_add_tensor( real * data , std::vector<int64_t> dims ) {
  torch_tensors.push_back( torch::from_blob( data , at::ArrayRef<int64_t>(dims) ) );
  return torch_tensors.size() - 1;
}


void torch_move_tensor_to_gpu(int id , int devicenum) {
    torch_tensors[id] = torch_tensors[id].to(torch::Device(torch::kCUDA, devicenum));
}


void torch_move_tensor_to_cpu(int id) {
  torch_tensors[id] = torch_tensors[id].to(torch::Device(torch::kCPU));
}


at::Tensor torch_module_forward( int module_id , std::vector<int> input_ids ) {
  std::vector<torch::jit::IValue> inputs;
  for (int i=0; i < input_ids.size(); i++) {
    inputs.push_back( torch_tensors[ input_ids[i] ] );
  }
  torch_tensors.push_back( torch_modules[module_id].forward( inputs ).toTensor() );

  // return array
  at::Tensor tensor = torch_tensors[torch_tensors.size() - 1];
  // clear torch_tensors
  torch_tensors.clear();
  return tensor;
}
