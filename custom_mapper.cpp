// taken from https://legion.stanford.edu/tutorial/custom_mappers.html

#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"

#include "test_mapper.h"
#include "default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

enum {
  SUBREGION_TUNABLE,
};

enum {
  PARTITIONING_MAPPER_ID = 1,
};

class AdversarialMapper : public TestMapper {
public:
  AdversarialMapper(Machine machine,
      Runtime *rt, Processor local);
public:
  virtual void select_task_options(const MapperContext    ctx,
				   const Task&            task,
				         TaskOptions&     output);
  virtual void slice_task(const MapperContext ctx,
                          const Task& task,
                          const SliceTaskInput& input,
                                SliceTaskOutput& output);
  virtual void map_task(const MapperContext ctx,
                        const Task& task,
                        const MapTaskInput& input,
                              MapTaskOutput& output);
  virtual void report_profiling(const MapperContext      ctx,
				const Task&              task,
				const TaskProfilingInfo& input);
};

class PartitioningMapper : public DefaultMapper {
public:
  PartitioningMapper(Machine machine,
      Runtime *rt, Processor local);
public:
  virtual void select_tunable_value(const MapperContext ctx,
                                    const Task& task,
                                    const SelectTunableInput& input,
                                          SelectTunableOutput& output);
};

void mapper_registration(Machine machine, Runtime *rt,
                          const std::set<Processor> &local_procs) {
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(
        new AdversarialMapper(machine, rt, *it), *it);
    rt->add_mapper(PARTITIONING_MAPPER_ID,
        new PartitioningMapper(machine, rt, *it), *it);
  }
}

AdversarialMapper::AdversarialMapper(Machine m,
                                     Runtime *rt, Processor p)
  : TestMapper(rt->get_mapper_runtime(), m, p)
{
  std::set<Processor> all_procs;
  machine.get_all_processors(all_procs);
  if (all_procs.begin()->id + 1 == local_proc.id) {
    printf("There are %zd processors:\n", all_procs.size());
    for (std::set<Processor>::const_iterator it = all_procs.begin();
          it != all_procs.end(); it++) {
      Processor::Kind kind = it->kind();
      switch (kind) {
        // Latency-optimized cores (LOCs) are CPUs
        case Processor::LOC_PROC:
          {
            printf("  Processor ID " IDFMT " is CPU\n", it->id);
            break;
          }
        // Throughput-optimized cores (TOCs) are GPUs
        case Processor::TOC_PROC:
          {
            printf("  Processor ID " IDFMT " is GPU\n", it->id);
            break;
          }
        // Processor for doing I/O
        case Processor::IO_PROC:
          {
            printf("  Processor ID " IDFMT " is I/O Proc\n", it->id);
            break;
          }
        // Utility processors are helper processors for
        // running Legion runtime meta-level tasks and
        // should not be used for running application tasks
        case Processor::UTIL_PROC:
          {
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
        case Memory::GLOBAL_MEM:
          {
            printf("  GASNet Global Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // DRAM on a single node
        case Memory::SYSTEM_MEM:
          {
            printf("  System Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Pinned memory on a single node
        case Memory::REGDMA_MEM:
          {
            printf("  Pinned Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // A memory associated with a single socket
        case Memory::SOCKET_MEM:
          {
            printf("  Socket Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Zero-copy memory betweeen CPU DRAM and
        // all GPUs on a single node
        case Memory::Z_COPY_MEM:
          {
            printf("  Zero-Copy Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // GPU framebuffer memory for a single GPU
        case Memory::GPU_FB_MEM:
          {
            printf("  GPU Frame Buffer Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Disk memory on a single node
        case Memory::DISK_MEM:
          {
            printf("  Disk Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // HDF framebuffer memory for a single GPU
        case Memory::HDF_MEM:
          {
            printf("  HDF Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // File memory on a single node
        case Memory::FILE_MEM:
          {
            printf("  File Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Block of memory sized for L3 cache
        case Memory::LEVEL3_CACHE:
          {
            printf("  Level 3 Cache ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Block of memory sized for L2 cache
        case Memory::LEVEL2_CACHE:
          {
            printf("  Level 2 Cache ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Block of memory sized for L1 cache
        case Memory::LEVEL1_CACHE:
          {
            printf("  Level 1 Cache ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
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
      int results =
        machine.get_proc_mem_affinity(affinities, local_proc, *it);
      assert(results == 1);
      printf("  Memory " IDFMT " has bandwidth %d and latency %d\n",
              it->id, affinities[0].bandwidth, affinities[0].latency);
    }
  }
}

void AdversarialMapper::select_task_options(const MapperContext ctx,
                                            const Task& task,
                                                  TaskOptions& output) {
  output.inline_task = false;
  output.stealable = false;
  output.map_locally = true;
  Processor::Kind kind = select_random_processor_kind(ctx, task.task_id);
  output.initial_proc = select_random_processor(kind);
}

void AdversarialMapper::slice_task(const MapperContext      ctx,
                                   const Task&              task,
                                   const SliceTaskInput&    input,
                                         SliceTaskOutput&   output) {
  // Iterate over all the points and send them all over the world
  output.slices.resize(input.domain.get_volume());
  unsigned idx = 0;
  switch (input.domain.get_dim()) {
    case 1:
      {
        Rect<1> rect = input.domain;
        for (PointInRectIterator<1> pir(rect); pir(); pir++, idx++)
        {
          Rect<1> slice(*pir, *pir);
          output.slices[idx] = TaskSlice(slice,
              select_random_processor(task.target_proc.kind()),
              false/*recurse*/, true/*stealable*/);
        }
        break;
      }
    case 2:
      {
        Rect<2> rect = input.domain;
        for (PointInRectIterator<2> pir(rect); pir(); pir++, idx++)
        {
          Rect<2> slice(*pir, *pir);
          output.slices[idx] = TaskSlice(slice,
              select_random_processor(task.target_proc.kind()),
              false/*recurse*/, true/*stealable*/);
        }
        break;
      }
    case 3:
      {
        Rect<3> rect = input.domain;
        for (PointInRectIterator<3> pir(rect); pir(); pir++, idx++)
        {
          Rect<3> slice(*pir, *pir);
          output.slices[idx] = TaskSlice(slice,
              select_random_processor(task.target_proc.kind()),
              false/*recurse*/, true/*stealable*/);
        }
        break;
      }
    default:
      assert(false);
  }
}

void AdversarialMapper::map_task(const MapperContext         ctx,
                                 const Task&                 task,
                                 const MapTaskInput&         input,
                                       MapTaskOutput&        output) {
  const std::map<VariantID,Processor::Kind> &variant_kinds =
    find_task_variants(ctx, task.task_id);
  std::vector<VariantID> variants;
  for (std::map<VariantID,Processor::Kind>::const_iterator it =
        variant_kinds.begin(); it != variant_kinds.end(); it++) {
    if (task.target_proc.kind() == it->second)
      variants.push_back(it->first);
  }
  assert(!variants.empty());
  if (variants.size() > 1) {
    int chosen = default_generate_random_integer() % variants.size();
    output.chosen_variant = variants[chosen];
  }
  else
    output.chosen_variant = variants[0];
  output.target_procs.push_back(task.target_proc);
  std::vector<bool> premapped(task.regions.size(), false);
  for (unsigned idx = 0; idx < input.premapped_regions.size(); idx++) {
    unsigned index = input.premapped_regions[idx];
    output.chosen_instances[index] = input.valid_instances[index];
    premapped[index] = true;
  }
  const TaskLayoutConstraintSet &layout_constraints =
    runtime->find_task_layout_constraints(ctx, task.task_id,
                                          output.chosen_variant);
  for (unsigned idx = 0; idx < task.regions.size(); idx++) {
    if (premapped[idx])
      continue;
    if (task.regions[idx].is_restricted()) {
      output.chosen_instances[idx] = input.valid_instances[idx];
      continue;
    }
    if (layout_constraints.layouts.find(idx) !=
          layout_constraints.layouts.end()) {
      std::vector<LayoutConstraintID> constraints;
      for (std::multimap<unsigned,LayoutConstraintID>::const_iterator it =
            layout_constraints.layouts.lower_bound(idx); it !=
            layout_constraints.layouts.upper_bound(idx); it++)
        constraints.push_back(it->second);
      map_constrained_requirement(ctx, task.regions[idx], TASK_MAPPING,
          constraints, output.chosen_instances[idx], task.target_proc);
    }
    else
      map_random_requirement(ctx, task.regions[idx],
                             output.chosen_instances[idx],
                             task.target_proc);
  }
  output.task_priority = default_generate_random_integer();

  {
    using namespace ProfilingMeasurements;
    output.task_prof_requests.add_measurement<OperationStatus>();
    output.task_prof_requests.add_measurement<OperationTimeline>();
    output.task_prof_requests.add_measurement<RuntimeOverhead>();
  }
}

void AdversarialMapper::report_profiling(const MapperContext      ctx,
					 const Task&              task,
					 const TaskProfilingInfo& input) {
  using namespace ProfilingMeasurements;

  OperationStatus *status =
    input.profiling_responses.get_measurement<OperationStatus>();
  if (status) {
    switch (status->result) {
      case OperationStatus::COMPLETED_SUCCESSFULLY:
        {
          printf("Task %s COMPLETED SUCCESSFULLY\n", task.get_task_name());
          break;
        }
      case OperationStatus::COMPLETED_WITH_ERRORS:
        {
          printf("Task %s COMPLETED WITH ERRORS\n", task.get_task_name());
          break;
        }
      case OperationStatus::INTERRUPT_REQUESTED:
        {
          printf("Task %s was INTERRUPTED\n", task.get_task_name());
          break;
        }
      case OperationStatus::TERMINATED_EARLY:
        {
          printf("Task %s TERMINATED EARLY\n", task.get_task_name());
          break;
        }
      case OperationStatus::CANCELLED:
        {
          printf("Task %s was CANCELLED\n", task.get_task_name());
          break;
        }
      default:
        assert(false); // shouldn't get any of the rest currently
    }
    delete status;
  }
  else
    printf("No operation status for task %s\n", task.get_task_name());

  OperationTimeline *timeline =
    input.profiling_responses.get_measurement<OperationTimeline>();
  if (timeline) {
    printf("Operation timeline for task %s: ready=%lld start=%lld stop=%lld\n",
	   task.get_task_name(),
	   timeline->ready_time,
	   timeline->start_time,
	   timeline->end_time);
    delete timeline;
  }
  else
    printf("No operation timeline for task %s\n", task.get_task_name());

  RuntimeOverhead *overhead =
    input.profiling_responses.get_measurement<RuntimeOverhead>();
  if (overhead) {
    long long total = (overhead->application_time +
		       overhead->runtime_time +
		       overhead->wait_time);
    if (total <= 0) total = 1;
    printf("Runtime overhead for task %s: runtime=%.1f%% wait=%.1f%%\n",
	   task.get_task_name(),
	   (100.0 * overhead->runtime_time / total),
	   (100.0 * overhead->wait_time / total));
    delete overhead;
  }
  else
    printf("No runtime overhead data for task %s\n", task.get_task_name());
}

PartitioningMapper::PartitioningMapper(Machine m,
                                       Runtime *rt,
                                       Processor p)
  : DefaultMapper(rt->get_mapper_runtime(), m, p)
{
}

void PartitioningMapper::select_tunable_value(const MapperContext ctx,
                                              const Task& task,
                                              const SelectTunableInput& input,
                                                    SelectTunableOutput& output) {
  if (input.tunable_id == SUBREGION_TUNABLE) {
    Machine::ProcessorQuery all_procs(machine);
    all_procs.only_kind(Processor::LOC_PROC);
    runtime->pack_tunable<size_t>(all_procs.count(), output);
    return;
  }
  assert(false);
}

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_FIELD_TASK_ID,
  DAXPY_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Y,
  FID_Z,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
  int num_elements = 1024;
  int num_subregions = 4;
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-b"))
        num_subregions = atoi(command_args.argv[++i]);
    }
  }
  printf("Running daxpy for %d elements...\n", num_elements);
  printf("Partitioning data into %d sub-regions...\n", num_subregions);

  Rect<1> elem_rect(0,num_elements-1);
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);
  runtime->attach_name(is, "is");
  FieldSpace input_fs = runtime->create_field_space(ctx);
  runtime->attach_name(input_fs, "input_fs");
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, input_fs);
    allocator.allocate_field(sizeof(double),FID_X);
    runtime->attach_name(input_fs, FID_X, "X");
    allocator.allocate_field(sizeof(double),FID_Y);
    runtime->attach_name(input_fs, FID_Y, "Y");
  }
  FieldSpace output_fs = runtime->create_field_space(ctx);
  runtime->attach_name(output_fs, "output_fs");
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, output_fs);
    allocator.allocate_field(sizeof(double),FID_Z);
    runtime->attach_name(output_fs, FID_Z, "Z");
  }
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);
  runtime->attach_name(input_lr, "input_lr");
  LogicalRegion output_lr = runtime->create_logical_region(ctx, is, output_fs);
  runtime->attach_name(output_lr, "output_lr");

  Rect<1> color_bounds(0,num_subregions-1);
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);

  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
  runtime->attach_name(ip, "ip");

  LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, ip);
  runtime->attach_name(input_lp, "input_lp");
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, ip);
  runtime->attach_name(output_lp, "output_lp");

  ArgumentMap arg_map;

  IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is,
                              TaskArgument(NULL, 0), arg_map);
  init_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, input_lr));
  init_launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_index_space(ctx, init_launcher);

  init_launcher.region_requirements[0].privilege_fields.clear();
  init_launcher.region_requirements[0].instance_fields.clear();
  init_launcher.region_requirements[0].add_field(FID_Y);
  runtime->execute_index_space(ctx, init_launcher);

  const double alpha = drand48();
  IndexLauncher daxpy_launcher(DAXPY_TASK_ID, color_is,
                TaskArgument(&alpha, sizeof(alpha)), arg_map);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0/*projection ID*/,
                        READ_ONLY, EXCLUSIVE, input_lr));
  daxpy_launcher.region_requirements[0].add_field(FID_X);
  daxpy_launcher.region_requirements[0].add_field(FID_Y);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(output_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, output_lr));
  daxpy_launcher.region_requirements[1].add_field(FID_Z);
  runtime->execute_index_space(ctx, daxpy_launcher);

  TaskLauncher check_launcher(CHECK_TASK_ID, TaskArgument(&alpha, sizeof(alpha)));
  check_launcher.add_region_requirement(
      RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
  check_launcher.region_requirements[0].add_field(FID_X);
  check_launcher.region_requirements[0].add_field(FID_Y);
  check_launcher.add_region_requirement(
      RegionRequirement(output_lr, READ_ONLY, EXCLUSIVE, output_lr));
  check_launcher.region_requirements[1].add_field(FID_Z);
  runtime->execute_task(ctx, check_launcher);

  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, output_lr);
  runtime->destroy_field_space(ctx, input_fs);
  runtime->destroy_field_space(ctx, output_fs);
  runtime->destroy_index_space(ctx, is);
}

void init_field_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  const int point = task->index_point.point_data[0];
  printf("Initializing field %d for block %d...\n", fid, point);

  const FieldAccessor<WRITE_DISCARD,double,1> acc(regions[0], fid);
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    acc[*pir] = drand48();
}

void daxpy_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(double));
  const double alpha = *((const double*)task->args);
  const int point = task->index_point.point_data[0];

  const FieldAccessor<READ_ONLY,double,1> acc_x(regions[0], FID_X);
  const FieldAccessor<READ_ONLY,double,1> acc_y(regions[0], FID_Y);
  const FieldAccessor<WRITE_DISCARD,double,1> acc_z(regions[1], FID_Z);
  printf("Running daxpy computation with alpha %.8g for point %d...\n",
          alpha, point);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    acc_z[*pir] = alpha * acc_x[*pir] + acc_y[*pir];
}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(double));
  const double alpha = *((const double*)task->args);

  const FieldAccessor<READ_ONLY,double,1> acc_x(regions[0], FID_X);
  const FieldAccessor<READ_ONLY,double,1> acc_y(regions[0], FID_Y);
  const FieldAccessor<READ_ONLY,double,1> acc_z(regions[1], FID_Z);

  printf("Checking results...");
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  bool all_passed = true;
  for (PointInRectIterator<1> pir(rect); pir(); pir++) {
    double expected = alpha * acc_x[*pir] + acc_y[*pir];
    double received = acc_z[*pir];
    if (expected != received)
      all_passed = false;
  }
  if (all_passed)
    printf("SUCCESS!\n");
  else
    printf("FAILURE!\n");
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INIT_FIELD_TASK_ID, "init_field");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_field_task>(registrar, "init_field");
  }

  {
    TaskVariantRegistrar registrar(DAXPY_TASK_ID, "daxpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<daxpy_task>(registrar, "daxpy");
  }

  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  // Here is where we register the callback function for
  // creating custom mappers.
  Runtime::add_registration_callback(mapper_registration);

  return Runtime::start(argc, argv);
}