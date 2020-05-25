#pragma once

// based off of https://legion.stanford.edu/tutorial/custom_mappers.html

#include "legion.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "default_mapper.h"
#include "test_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

enum {
  SUBREGION_TUNABLE,
};

enum {
  PARTITIONING_MAPPER_ID = 1,
};

class NodeAwareMapper : public TestMapper {
public:
  NodeAwareMapper(Machine machine, Runtime *rt, Processor local);

public:
  virtual void select_task_options(const MapperContext ctx, const Task &task,
                                   TaskOptions &output);
  virtual void select_tasks_to_map(const MapperContext ctx,
                                   const SelectMappingInput &input,
                                   SelectMappingOutput &output);
  virtual void slice_task(const MapperContext ctx, const Task &task,
                          const SliceTaskInput &input, SliceTaskOutput &output);
  virtual void map_task(const MapperContext ctx, const Task &task,
                        const MapTaskInput &input, MapTaskOutput &output);
  virtual void report_profiling(const MapperContext ctx, const Task &task,
                                const TaskProfilingInfo &input);

  /* to be passed to Runtime::add_registration_callback
   */
  static void mapper_registration(Machine machine, Runtime *rt,
                                  const std::set<Processor> &local_procs) {
    printf("NodeAwareMapper::mapper_registration(): entry\n");

    for (std::set<Processor>::const_iterator it = local_procs.begin();
         it != local_procs.end(); it++) {
      rt->replace_default_mapper(new NodeAwareMapper(machine, rt, *it), *it);
    }
  }

private:
  Runtime *runtime_;
};

NodeAwareMapper::NodeAwareMapper(Machine m, Runtime *rt, Processor p)
    : TestMapper(rt->get_mapper_runtime(), m, p), runtime_(rt) {
  printf("NodeAwareMapper::%s() entry\n", __func__);

  std::set<Processor> all_procs;
  machine.get_all_processors(all_procs);
  if (all_procs.begin()->id + 1 == local_proc.id) {
    printf("There are %zd processors:\n", all_procs.size());
    for (std::set<Processor>::const_iterator it = all_procs.begin();
         it != all_procs.end(); it++) {
      Processor::Kind kind = it->kind();
      switch (kind) {
      // Latency-optimized cores (LOCs) are CPUs
      case Processor::LOC_PROC: {
        printf("  Processor ID " IDFMT " is CPU\n", it->id);
        break;
      }
      // Throughput-optimized cores (TOCs) are GPUs
      case Processor::TOC_PROC: {
        printf("  Processor ID " IDFMT " is GPU\n", it->id);
        break;
      }
      // Processor for doing I/O
      case Processor::IO_PROC: {
        printf("  Processor ID " IDFMT " is I/O Proc\n", it->id);
        break;
      }
      // Utility processors are helper processors for
      // running Legion runtime meta-level tasks and
      // should not be used for running application tasks
      case Processor::UTIL_PROC: {
        printf("  Processor ID " IDFMT " is utility\n", it->id);
        break;
      }
      default:
        assert(false);
      }
    }
    std::set<Memory> all_mems;
    machine.get_all_memories(all_mems);
    printf("There are %zd memories:\n", all_mems.size());
    for (std::set<Memory>::const_iterator it = all_mems.begin();
         it != all_mems.end(); it++) {
      Memory::Kind kind = it->kind();
      size_t memory_size_in_kb = it->capacity() >> 10;
      switch (kind) {
      // RDMA addressable memory when running with GASNet
      case Memory::GLOBAL_MEM: {
        printf("  GASNet Global Memory ID " IDFMT " has %zd KB\n", it->id,
               memory_size_in_kb);
        break;
      }
      // DRAM on a single node
      case Memory::SYSTEM_MEM: {
        printf("  System Memory ID " IDFMT " has %zd KB\n", it->id,
               memory_size_in_kb);
        break;
      }
      // Pinned memory on a single node
      case Memory::REGDMA_MEM: {
        printf("  Pinned Memory ID " IDFMT " has %zd KB\n", it->id,
               memory_size_in_kb);
        break;
      }
      // A memory associated with a single socket
      case Memory::SOCKET_MEM: {
        printf("  Socket Memory ID " IDFMT " has %zd KB\n", it->id,
               memory_size_in_kb);
        break;
      }
      // Zero-copy memory betweeen CPU DRAM and
      // all GPUs on a single node
      case Memory::Z_COPY_MEM: {
        printf("  Zero-Copy Memory ID " IDFMT " has %zd KB\n", it->id,
               memory_size_in_kb);
        break;
      }
      // GPU framebuffer memory for a single GPU
      case Memory::GPU_FB_MEM: {
        printf("  GPU Frame Buffer Memory ID " IDFMT " has %zd KB\n", it->id,
               memory_size_in_kb);
        break;
      }
      // Disk memory on a single node
      case Memory::DISK_MEM: {
        printf("  Disk Memory ID " IDFMT " has %zd KB\n", it->id,
               memory_size_in_kb);
        break;
      }
      // HDF framebuffer memory for a single GPU
      case Memory::HDF_MEM: {
        printf("  HDF Memory ID " IDFMT " has %zd KB\n", it->id,
               memory_size_in_kb);
        break;
      }
      // File memory on a single node
      case Memory::FILE_MEM: {
        printf("  File Memory ID " IDFMT " has %zd KB\n", it->id,
               memory_size_in_kb);
        break;
      }
      // Block of memory sized for L3 cache
      case Memory::LEVEL3_CACHE: {
        printf("  Level 3 Cache ID " IDFMT " has %zd KB\n", it->id,
               memory_size_in_kb);
        break;
      }
      // Block of memory sized for L2 cache
      case Memory::LEVEL2_CACHE: {
        printf("  Level 2 Cache ID " IDFMT " has %zd KB\n", it->id,
               memory_size_in_kb);
        break;
      }
      // Block of memory sized for L1 cache
      case Memory::LEVEL1_CACHE: {
        printf("  Level 1 Cache ID " IDFMT " has %zd KB\n", it->id,
               memory_size_in_kb);
        break;
      }
      default:
        assert(false);
      }
    }

    std::set<Memory> vis_mems;
    machine.get_visible_memories(local_proc, vis_mems);
    printf("There are %zd memories visible from processor " IDFMT "\n",
           vis_mems.size(), local_proc.id);
    for (std::set<Memory>::const_iterator it = vis_mems.begin();
         it != vis_mems.end(); it++) {
      std::vector<ProcessorMemoryAffinity> affinities;
      int results = machine.get_proc_mem_affinity(affinities, local_proc, *it);
      assert(results == 1);
      printf("  Memory " IDFMT " has bandwidth %d and latency %d\n", it->id,
             affinities[0].bandwidth, affinities[0].latency);
    }
  }
  printf("NodeAwareMapper::%s() exit\n", __func__);
}

void NodeAwareMapper::select_task_options(const MapperContext ctx,
                                          const Task &task,
                                          TaskOptions &output) {
  printf("NodeAwareMapper::%s() entry\n", __func__);
  // https://legion.stanford.edu/tutorial/custom_mappers.html

  output.inline_task = false;
  output.stealable = false;

  /*
   determines whether subsequent mapper calls (such as map_task) should be
   processed by the current mapper, or the mapper for the processor to which the
   task is to be assigned.
  */
  output.map_locally = false;

  /* Send the task to a processor to be mapped

     Choose a random processor for the initial map
  */
  Processor::Kind kind = select_random_processor_kind(ctx, task.task_id);
  output.initial_proc = select_random_processor(kind);
  printf("NodeAwareMapper::%s() exit\n", __func__);
}

void NodeAwareMapper::select_tasks_to_map(const MapperContext ctx,
                                          const SelectMappingInput &input,
                                          SelectMappingOutput &output) {
printf("NodeAwareMapper::%s() entry\n", __func__);

printf("NodeAwareMapper::%s() %lu ready\n", __func__, input.ready_tasks.size());
  /*
struct SelectMappingInput {
std::list<const Task*>                  ready_tasks;
};

struct SelectMappingOutput {
std::set<const Task*>                   map_tasks;
std::map<const Task*,Processor>         relocate_tasks;
MapperEvent                             deferral_event;
};
*/

  for (auto task : input.ready_tasks) {
    output.map_tasks.insert(task);
  }
  printf("NodeAwareMapper::%s() exit\n", __func__);
}

void NodeAwareMapper::slice_task(const MapperContext ctx, const Task &task,
                                 const SliceTaskInput &input,
                                 SliceTaskOutput &output) {

  printf("NodeAwareMapper::%s() entry\n", __func__);

  // Iterate over all the points and send them all over the world
  output.slices.resize(input.domain.get_volume());
  unsigned idx = 0;
  switch (input.domain.get_dim()) {
  case 1: {
    Rect<1> rect = input.domain;
    for (PointInRectIterator<1> pir(rect); pir(); pir++, idx++) {
      Rect<1> slice(*pir, *pir);
      output.slices[idx] =
          TaskSlice(slice, select_random_processor(task.target_proc.kind()),
                    false /*recurse*/, true /*stealable*/);
    }
    break;
  }
  case 2: {
    Rect<2> rect = input.domain;
    for (PointInRectIterator<2> pir(rect); pir(); pir++, idx++) {
      Rect<2> slice(*pir, *pir);
      output.slices[idx] =
          TaskSlice(slice, select_random_processor(task.target_proc.kind()),
                    false /*recurse*/, true /*stealable*/);
    }
    break;
  }
  case 3: {
    Rect<3> rect = input.domain;
    for (PointInRectIterator<3> pir(rect); pir(); pir++, idx++) {
      Rect<3> slice(*pir, *pir);
      output.slices[idx] =
          TaskSlice(slice, select_random_processor(task.target_proc.kind()),
                    false /*recurse*/, true /*stealable*/);
    }
    break;
  }
  default:
    assert(false);
  }
  printf("NodeAwareMapper::%s() exit\n", __func__);
}

// https://legion.stanford.edu/tutorial/custom_mappers.html
void NodeAwareMapper::map_task(const MapperContext ctx, const Task &task,
                               const MapTaskInput &input,
                               MapTaskOutput &output) {
  printf("NodeAwareMapper::%s() entry\n", __func__);

  // https://legion.stanford.edu/doxygen/class_legion_1_1_domain.html
  std::cerr << "index_domain:\n";
  Domain domain = task.index_domain;
  std::cerr << "  lo:" << domain.lo() << "-hi:" << domain.hi() << "\n";
  std::cerr << "  point:" << task.index_point << "\n";

  switch (domain.get_dim()) {
  case 0: {
    break;
  }
  case 1: {
    Rect<1> rect = domain;
    std::cerr << rect.lo << " " << rect.hi << "\n";
    for (PointInRectIterator<1> pir(rect); pir(); pir++) {
      Rect<1> slice(*pir, *pir);
    }
    break;
  }
  case 2: {
    Rect<2> rect = domain;
    std::cerr << rect.lo << " " << rect.hi << "\n";
    for (PointInRectIterator<2> pir(rect); pir(); pir++) {
    }
    break;
  }
  case 3: {
    Rect<3> rect = domain;
    std::cerr << rect.lo << " " << rect.hi << "\n";
    for (PointInRectIterator<3> pir(rect); pir(); pir++) {
    }
    break;
  }
  default:
    assert(false);
  }

  /* print some stats about all the task's regions
   */
  std::cerr << "regions:\n";
  for (unsigned idx = 0; idx < task.regions.size(); idx++) {
    // https://legion.stanford.edu/doxygen/struct_legion_1_1_region_requirement.html
    std::cerr << "  " << idx << "\n";
    RegionRequirement req = task.regions[idx];

    Domain domain =
        runtime_->get_index_space_domain(req.region.get_index_space());

    switch (domain.get_dim()) {
    case 0: {
      break;
    }
    case 1: {
      Rect<1> rect = domain;
      std::cerr << "    domain:    " << rect.lo << "-" << rect.hi << "\n";
      break;
    }
    case 2: {
      Rect<2> rect = domain;
      std::cerr << "    domain:    " << rect.lo << "-" << rect.hi << "\n";
      break;
    }
    case 3: {
      Rect<3> rect = domain;
      std::cerr << "    domain:    " << rect.lo << "-" << rect.hi << "\n";
      break;
    }
    default:
      assert(false);
    }

    std::cerr << "    region:    " << req.region << "\n";
    std::cerr << "    partition: " << req.partition << "\n";
    std::cerr << "    parent:    " << req.parent << "\n";
    LogicalRegion region = req.region;
    LogicalPartition partition = req.partition;
  }

  /* print some stats about all the task's indexes
   */
  std::cerr << "indexes:\n";
  for (unsigned idx = 0; idx < task.indexes.size(); idx++) {
    std::cerr << "  " << idx << "\n";
    IndexSpaceRequirement req = task.indexes[idx];
    std::cerr << "    handle: " << req.handle << "\n";
    std::cerr << "    parent: " << req.parent << "\n";
  }

  // chose a variant that matches the selected processor
  const std::map<VariantID, Processor::Kind> &variant_kinds =
      find_task_variants(ctx, task.task_id);
  std::vector<VariantID> variants;
  for (std::map<VariantID, Processor::Kind>::const_iterator it =
           variant_kinds.begin();
       it != variant_kinds.end(); it++) {
    if (task.target_proc.kind() == it->second)
      variants.push_back(it->first);
  }
  assert(!variants.empty());
  if (variants.size() > 1) {
    int chosen = default_generate_random_integer() % variants.size();
    output.chosen_variant = variants[chosen];
  } else
    output.chosen_variant = variants[0];

  // select the target processor for this task
  output.target_procs.push_back(task.target_proc);

  /* chose how the task's logical instances should be mapped to physical
   * instances?
   */

  // some regions may already be mapped (from previous tasks that have run?)
  std::vector<bool> premapped(task.regions.size(), false);
  for (unsigned idx = 0; idx < input.premapped_regions.size(); idx++) {
    unsigned index = input.premapped_regions[idx];
    output.chosen_instances[index] = input.valid_instances[index];
    premapped[index] = true;
  }

  /* certain task variants may require certain data layouts
   */
  const TaskLayoutConstraintSet &layout_constraints =
      runtime->find_task_layout_constraints(ctx, task.task_id,
                                            output.chosen_variant);

  for (unsigned idx = 0; idx < task.regions.size(); idx++) {
    /* if task (ed. instance?) is premapped we need not and cannot map it
     */
    if (premapped[idx]) {
      continue;
    }
    /* if instance is restricted a valid instance already exists and we can use
     * it
     */
    if (task.regions[idx].is_restricted()) {
      output.chosen_instances[idx] = input.valid_instances[idx];
      continue;
    }

    /* chose a specific layout for a task.
       If there are constraints, use them, otherwise pick a random layout
    */
    if (layout_constraints.layouts.find(idx) !=
        layout_constraints.layouts.end()) {
      std::vector<LayoutConstraintID> constraints;
      for (std::multimap<unsigned, LayoutConstraintID>::const_iterator it =
               layout_constraints.layouts.lower_bound(idx);
           it != layout_constraints.layouts.upper_bound(idx); it++)
        constraints.push_back(it->second);
      map_constrained_requirement(ctx, task.regions[idx], TASK_MAPPING,
                                  constraints, output.chosen_instances[idx],
                                  task.target_proc);
    } else
      map_random_requirement(ctx, task.regions[idx],
                             output.chosen_instances[idx], task.target_proc);
  }
  output.task_priority = default_generate_random_integer();

  {
    using namespace ProfilingMeasurements;
    output.task_prof_requests.add_measurement<OperationStatus>();
    output.task_prof_requests.add_measurement<OperationTimeline>();
    output.task_prof_requests.add_measurement<RuntimeOverhead>();
  }
  printf("NodeAwareMapper::%s() exit\n", __func__);
}

void NodeAwareMapper::report_profiling(const MapperContext ctx,
                                       const Task &task,
                                       const TaskProfilingInfo &input) {
  printf("NodeAwareMapper::%s() entry\n", __func__);

  using namespace ProfilingMeasurements;

  OperationStatus *status =
      input.profiling_responses.get_measurement<OperationStatus>();
  if (status) {
    switch (status->result) {
    case OperationStatus::COMPLETED_SUCCESSFULLY: {
      printf("Task %s COMPLETED SUCCESSFULLY\n", task.get_task_name());
      break;
    }
    case OperationStatus::COMPLETED_WITH_ERRORS: {
      printf("Task %s COMPLETED WITH ERRORS\n", task.get_task_name());
      break;
    }
    case OperationStatus::INTERRUPT_REQUESTED: {
      printf("Task %s was INTERRUPTED\n", task.get_task_name());
      break;
    }
    case OperationStatus::TERMINATED_EARLY: {
      printf("Task %s TERMINATED EARLY\n", task.get_task_name());
      break;
    }
    case OperationStatus::CANCELLED: {
      printf("Task %s was CANCELLED\n", task.get_task_name());
      break;
    }
    default:
      assert(false); // shouldn't get any of the rest currently
    }
    delete status;
  } else
    printf("No operation status for task %s\n", task.get_task_name());

  OperationTimeline *timeline =
      input.profiling_responses.get_measurement<OperationTimeline>();
  if (timeline) {
    printf("Operation timeline for task %s: ready=%lld start=%lld stop=%lld\n",
           task.get_task_name(), timeline->ready_time, timeline->start_time,
           timeline->end_time);
    delete timeline;
  } else
    printf("No operation timeline for task %s\n", task.get_task_name());

  RuntimeOverhead *overhead =
      input.profiling_responses.get_measurement<RuntimeOverhead>();
  if (overhead) {
    long long total = (overhead->application_time + overhead->runtime_time +
                       overhead->wait_time);
    if (total <= 0)
      total = 1;
    printf("Runtime overhead for task %s: runtime=%.1f%% wait=%.1f%%\n",
           task.get_task_name(), (100.0 * overhead->runtime_time / total),
           (100.0 * overhead->wait_time / total));
    delete overhead;
  } else
    printf("No runtime overhead data for task %s\n", task.get_task_name());

  printf("NodeAwareMapper::%s() exit\n", __func__);
}
