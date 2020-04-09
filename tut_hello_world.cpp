// https://legion.stanford.edu/tutorial/hello_world.html

#include <cstdio>
#include "legion.h"

using namespace Legion;

enum TaskID {
  HELLO_WORLD_ID,
};

void hello_world_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime) {
  printf("Hello World!\n");
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(HELLO_WORLD_ID);

  {
    TaskVariantRegistrar registrar(HELLO_WORLD_ID, "hello_world variant");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<hello_world_task>(registrar, "hello_world task");
  }

  return Runtime::start(argc, argv);
}
