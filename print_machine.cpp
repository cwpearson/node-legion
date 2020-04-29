// https://legion.stanford.edu/tutorial/multiple.html

#include "legion.h"
#include <cstdio>
#include <cstdlib>

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
};

// lifted from runtime/legion/legion_ops.cc
const char *c_str_name(const Memory::Kind &k) {
  // clang-format off
const char *names[] = {
    "GLOBAL_MEM",
    "SYSTEM_MEM",
    "REGDMA_MEM",
    "SOCKET_MEM",
    "Z_COPY_MEM",
    "GPU_FB_MEM",
    "DISK_MEM",
    "HDF_MEM",
    "FILE_MEM",
    "LEVEL3_CACHE,",
    "LEVEL2_CACHE,",
    "LEVEL1_CACHE,",
};
//clang-format on
    return names[k];
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions, Context ctx,
                    Runtime *runtime) {

  auto machine = Machine::get_machine();

  /* Query GPUs
   */
  std::vector<Processor> gpus;
  {
    Machine::ProcessorQuery pq(machine);
    pq.only_kind(Processor::TOC_PROC);
    for (const auto &proc : pq) {
      gpus.push_back(proc);
    }
  }

  std::cerr << "Legion GPUs:\n";
  for (auto &gpu : gpus) {
    std::cerr << "  " << gpu << "\n";
  }

  /* Query Memories
   */
  std::vector<Memory> memories;
  {
    Machine::MemoryQuery mq(machine);
    for (const auto &mem : mq) {
      memories.push_back(mem);
    }
  }

  std::cerr << "Legion Memories:\n";
  for (auto &mem : memories) {
    std::cerr << "  " << mem << " " << c_str_name(mem.kind()) <<  " "<< mem.capacity() << "\n";
  }


  std::vector<Machine::ProcessorMemoryAffinity> procMemAffinities;
  machine.get_proc_mem_affinity(procMemAffinities);

  std::cerr << "GPU processor-memory affinities\n";
  for (auto &aff : procMemAffinities) {
    if (aff.p.kind() == Processor::TOC_PROC) {
      std::cerr << aff.p << " - " << aff.m << " " << aff.bandwidth << " "
                << aff.latency << "\n";
    }
  }

  std::vector<Machine::MemoryMemoryAffinity> memMemAffinities;
  machine.get_mem_mem_affinity(memMemAffinities);

  std::cerr << "memory-memory affinities\n";
  for (auto &aff : memMemAffinities) {
    std::cerr << aff.m1 << " - " << aff.m2 << " " << aff.bandwidth << " "
              << aff.latency << "\n";
  }
}

int main(int argc, char **argv) {

  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  return Runtime::start(argc, argv);
}