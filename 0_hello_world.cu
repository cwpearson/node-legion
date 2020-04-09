// https://legion.stanford.edu/tutorial/hello_world.html

#include <cstdio>

#include "legion.h"

#include "node_aware_mapper.hpp"

using namespace Legion;

enum TaskID {
  HELLO_WORLD_ID,
};

__global__ void hello_world_kernel() {
  printf("Hello World! (from the GPU)\n");
}

void hello_world_gpu_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions, Context ctx,
                      Runtime *runtime) {
  hello_world_kernel<<<1,1>>>();
}

void hello_world_cpu_task(const Task *task,
  const std::vector<PhysicalRegion> &regions, Context ctx,
  Runtime *runtime) {
printf("Hello World! (from the CPU)\n");
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(HELLO_WORLD_ID);

  // seems to prefer whichever task is registered first
  {
    TaskVariantRegistrar registrar(HELLO_WORLD_ID, "hello_world GPU variant");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<hello_world_gpu_task>(registrar,
                                                        "hello_world task");
  }
  {
    TaskVariantRegistrar registrar(HELLO_WORLD_ID, "hello_world CPU variant");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<hello_world_cpu_task>(registrar,
                                                        "hello_world task");
  }


  Runtime::add_registration_callback(NodeAwareMapper::mapper_registration);

  return Runtime::start(argc, argv);
}
