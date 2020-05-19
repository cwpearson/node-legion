#pragma once

// based off of https://legion.stanford.edu/tutorial/custom_mappers.html

#include "legion.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "default_mapper.h"

#include <nvToolsExt.h>

#include "solve.hpp"

/* assignment problem utilities */
namespace ap {

/* brute-force solution to QAP, placing the final cost in costp if not null
 */
inline std::vector<size_t> solve_qap(double *costp, const solve::Mat2D<int64_t> &w,
                                     solve::Mat2D<double> &d) {

  assert(w.shape() == d.shape());
  assert(w.shape()[0] == d.shape()[1]);

  std::vector<size_t> f(w.shape()[0]);
  for (size_t i = 0; i < w.shape()[0]; ++i) {
    f[i] = i;
  }

  std::vector<size_t> bestF = f;
  double bestCost = solve::sum_cost(w, d, f);
  do {
    const double cost = solve::sum_cost(w, d, f);
    if (bestCost > cost) {
      bestF = f;
      bestCost = cost;
    }
  } while (std::next_permutation(f.begin(), f.end()));

  if (costp) {
    *costp = bestCost;
  }

  return bestF;
}

/* greedy swap2 solution to assignment problem
   Consider all 2-swaps of assignments, and choose the first better one found
   Do this until no improvements can be found

   objective: minimize total flow * distance product under assignment

   `w`: flow between tasks
   `d`: distance between agents

   return empty vector if no valid assignment was found

   load-balancing requires that the difference in assigned tasks between
   any two GPUs is 1.
 */
inline std::vector<size_t> solve_ap_swap2(double *costp, const solve::Mat2D<int64_t> &w,
                                    solve::Mat2D<double> &d) {

  // w and d are square
  assert(d.shape().is_cube());
  assert(w.shape().is_cube());

  const int64_t numAgents = d.shape()[0];
  const int64_t numTasks = w.shape()[0];

  // round-robin assign tasks to agents
  std::vector<size_t> f(numTasks, 0);
  for (size_t t = 0; t < numTasks; ++t) {
    f[t] = t % numAgents;
  }

  RollingStatistics stats;

  std::vector<size_t> bestF = f;
  double bestCost = solve::max_cost(w,d,f);
  stats.insert(bestCost);

  bool changed = true;
  while(changed) {
    changed = false;

    // check all possible swaps
    for (size_t i = 0; i < numTasks; ++i) {
      for (size_t j = i + 1; j < numTasks; ++j) {
        std::vector<size_t> swappedF = f; // swapped f
        std::swap(swappedF[i], swappedF[j]);

        double swappedCost = solve::max_cost(w,d,swappedF);
        stats.insert(swappedCost);

        if (swappedCost < bestCost) {
          bestCost = swappedCost;
          bestF = swappedF;
          changed = true;
          goto body_end; // fast exit
        }
      }
    }

    body_end:
    ;
  }

  if (costp) {
    *costp = bestCost;
  }

  std::cerr << "Considered " << stats.count()
            << " placements: min=" << stats.min() << " avg=" << stats.mean()
            << " max=" << stats.max() << "\n";

  return bestF;
}



} // namespace ap

using namespace Legion;
using namespace Legion::Mapping;

Logger log_mapper("node_aware_must_epoch_mapper");

class NodeAwareMustEpochMapper : public DefaultMapper {

private:
  /* Get the overlap between region requirements in bytes
   */
  int64_t get_region_requirement_overlap(MapperContext ctx,
                                         const RegionRequirement &rra,
                                         const RegionRequirement &rrb);

  /* Get the overlap between Logical Regions in bytes
   */
  int64_t get_logical_region_overlap(MapperContext ctx,
                                     const LogicalRegion &lra,
                                     const LogicalRegion &lrb);

public:
  NodeAwareMustEpochMapper(MapperRuntime *rt, Machine machine, Processor local,
                           const char *mapper_name);

  /*
  If the task is an index task launch the runtime calls slice_task to divide the
  index task into a set of slices that contain point tasks. One slice
  corresponds to one target processor. Each slice identifies an index space, a
  subregion of the original domain and a target processor. All of the point
  tasks for the subregion will be mapped by the mapper for the target processor.

  If slice.stealable is true the task can be stolen for load balancing. If
  slice.recurse is true the mapper for the target processor will invoke
  slice_task again with the slice as input. Here is sample code to create a
  stealable slice:

  ```
  struct SliceTaskInput {
    IndexSpace                             domain_is;
    Domain                                 domain;
  };

  struct SliceTaskOutput {
    std::vector<TaskSlice>                 slices;
    bool                                   verify_correctness; // = false
  };
  ```

  A stealable slice:
  ```
  TaskSlice slice;
  slice.domain = slice_subregion;
  slice.proc = targets[target_proc_index];
  slice.recurse = false;
  slice.stealable = true;
  slices.push_back(slice);
  ```

  Typically, the idea is that groups of nearby tasks would have locality, so we
  would group these point tasks into slices, each of which has a rect of point
  tasks and then assign those groups to procs. At the end of the day, each of
  those point tasks is still mapped individually.

  Instead, we will treat each point task independently, and create slices of 1.
  This allows us to figure out which point tasks communicate and which should be
  on nearby GPUs.
  */
  virtual void slice_task(const MapperContext ctx, const Task &task,
                          const SliceTaskInput &input, SliceTaskOutput &output);

  /*
  If a mapper has one or more tasks that are ready to execute it calls
  select_tasks_to_map. This method can copy tasks to the map_tasks list to
  indicate the task should be mapped by this mapper. The method can copy tasks
  to the relocate_tasks list to indicate the task should be mapped by a mapper
  for a different processor. If it does neither the task stays in the ready
  list.
  */
  virtual void select_tasks_to_map(const MapperContext ctx,
                                   const SelectMappingInput &input,
                                   SelectMappingOutput &output);

  virtual void map_task(const MapperContext ctx, const Task &task,
                        const MapTaskInput &input, MapTaskOutput &output);

  /* Do some node-aware must-epoch mapping

  */
  virtual void map_must_epoch(const MapperContext ctx,
                              const MapMustEpochInput &input,
                              MapMustEpochOutput &output);

  /* NodeAwareMustEpochMapper requests the runtime calls this
    when the task is finished
  */
  virtual void postmap_task(const MapperContext ctx, const Task &task,
                            const PostMapInput &input, PostMapOutput &output);

  /* Replace the default mapper with a NodeAwareMustEpochMapper

  to be passed to Runtime::add_registration_callback
   */
  static void mapper_registration(Machine machine, Runtime *rt,
                                  const std::set<Processor> &local_procs);

protected:
  /* true if there is a GPU variant of the task
   */
  bool has_gpu_variant(const MapperContext ctx, TaskID id);

  /* return GPUs pair with their closest FBs
   */
  std::vector<std::pair<Processor, Memory>> get_gpu_fbs();

  /* return the distance matrix between the processors in `procs`
   */
  solve::Mat2D<double> get_gpu_distance_matrix(
      const std::vector<std::pair<Processor, Memory>> &gpus);
};

NodeAwareMustEpochMapper::NodeAwareMustEpochMapper(MapperRuntime *rt,
                                                   Machine machine,
                                                   Processor local,
                                                   const char *mapper_name)
    : DefaultMapper(rt, machine, local, mapper_name) {}

void NodeAwareMustEpochMapper::mapper_registration(
    Machine machine, Runtime *rt, const std::set<Processor> &local_procs) {
  printf("NodeAwareMustEpochMapper::%s(): [entry]\n", __FUNCTION__);

  for (std::set<Processor>::const_iterator it = local_procs.begin();
       it != local_procs.end(); it++) {
    rt->replace_default_mapper(
        new NodeAwareMustEpochMapper(rt->get_mapper_runtime(), machine, *it,
                                     "NodeAwareMustEpochMapper"),
        *it);
  }
  printf("NodeAwareMustEpochMapper::%s(): [exit]\n", __FUNCTION__);
}

// inspired by DefaultMapper::have_proc_kind_variant
bool NodeAwareMustEpochMapper::has_gpu_variant(const MapperContext ctx,
                                               TaskID id) {
  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, id, variants);

  for (unsigned i = 0; i < variants.size(); i++) {
    const ExecutionConstraintSet exset =
        runtime->find_execution_constraints(ctx, id, variants[i]);
    if (exset.processor_constraint.can_use(Processor::TOC_PROC))
      return true;
  }
  return false;
}

/*

struct SliceTaskInput {
  IndexSpace                             domain_is;
  Domain                                 domain;
};

struct SliceTaskOutput {
  std::vector<TaskSlice>                 slices;
  bool                                   verify_correctness; // = false
};

  TaskSlice slice;
  slice.domain = slice_subregion;
  slice.proc = targets[target_proc_index];
  slice.recurse = false;
  slice.stealable = true;
  slices.push_back(slice);


  We do similar to the default mapper here:
  Use default_select_num_blocks to split along prime factors
  Instead of group

*/
void NodeAwareMustEpochMapper::slice_task(const MapperContext ctx,
                                          const Task &task,
                                          const SliceTaskInput &input,
                                          SliceTaskOutput &output) {
  log_mapper.spew("[entry] %s()", __FUNCTION__);

  log_mapper.spew() << __FUNCTION__
                    << "(): input.domain_is = " << input.domain_is;
  log_mapper.spew() << __FUNCTION__ << "(): input.domain    = " << input.domain;

  /* Build the GPU distance matrix
   */
  std::vector<std::pair<Processor, Memory>> gpus = get_gpu_fbs();
  {
    log_mapper.spew("GPU numbering:");
    int i = 0;
    for (auto p : gpus) {
      log_mapper.spew() << i << " gpu=" << p.first << " fb=" << p.second;
    }
  }


  solve::Mat2D<double> distance = get_gpu_distance_matrix(gpus);

  printf("NodeAwareMustEpochMapper::%s(): distance matrix\n", __FUNCTION__);
  for (size_t i = 0; i < distance.shape()[1]; ++i) {
    printf("NodeAwareMustEpochMapper::%s():", __FUNCTION__);
    for (size_t j = 0; j < distance.shape()[0]; ++j) {
      printf(" %.2e", distance.at(i, j));
    }
    printf("\n");
  }

  /* Compute the point task domain overlap for all pairs of point tasks.
  This is the weight matrix in our assignment problem.
  */
  solve::Mat2D<int64_t> weight(input.domain.get_volume(), input.domain.get_volume(),
                           0);

  assert(input.domain.dim == 2 && "TODO: only implemented for dim=2");

  {
    log_mapper.spew("point space numbering:");
    int i = 0;
    for (PointInDomainIterator<2> pir(input.domain); pir(); ++pir, ++i) {
      log_mapper.spew() << i << " p=" << *pir;
      for (int r = 0; r < task.regions.size(); ++r) {
        LogicalRegion li = runtime->get_logical_subregion_by_color(
            ctx, task.regions[r].partition, *pir);
        log_mapper.spew() << "  " << r << " " << runtime->get_index_space_domain(ctx, li.get_index_space());
      }
    }
  }

  int i = 0;
  for (PointInDomainIterator<2> pi(input.domain); pi(); ++pi, ++i) {
    int j = 0;
    for (PointInDomainIterator<2> pj(input.domain); pj(); ++pj, ++j) {

      int64_t bytes = 0;
      for (int ri = 0; ri < task.regions.size(); ++ri) {
        LogicalRegion li = runtime->get_logical_subregion_by_color(
            ctx, task.regions[ri].partition, *pi);

        // TODO: is this sketchy
        // create fake region requirements that move the partition to the
        // region so we can reuse the region distance code
        RegionRequirement rra = task.regions[ri];
        rra.partition = LogicalPartition::NO_PART;
        rra.region = li;
        for (int rj = 0; rj < task.regions.size(); ++rj) {
          LogicalRegion lj = runtime->get_logical_subregion_by_color(
              ctx, task.regions[rj].partition, *pj);

          // TODO this feels sketchy
          RegionRequirement rrb = task.regions[rj];
          rrb.partition = LogicalPartition::NO_PART;
          rrb.region = lj;

          int64_t newBytes = get_region_requirement_overlap(ctx, rra, rrb);
          // std::cerr << i << " " << j << " " << ri << " " << rj << " bytes="
          // << newBytes << "\n";
          bytes += newBytes;
        }
      }
      // std::cerr << "slice " << *pi << " " << *pj << " bytes=" << bytes << "\n";
      weight.at(i, j) = bytes;
    }
  }

  printf("NodeAwareMustEpochMapper::%s(): weight matrix\n", __FUNCTION__);
  for (size_t i = 0; i < weight.shape()[1]; ++i) {
    printf("NodeAwareMustEpochMapper::%s():", __FUNCTION__);
    for (size_t j = 0; j < weight.shape()[0]; ++j) {
      printf(" %6ld", weight.at(i, j));
    }
    printf("\n");
  }

  nvtxRangePush("solve_ap");
  double cost;
  std::vector<size_t> f = solve::ap_sum_brute_force(&cost, weight, distance);
  assert(f.size() == weight.shape()[0]);
  nvtxRangePop();

  log_mapper.spew() << "assignment";
  {
    std::stringstream ss;
    for (auto &e : f) {
      ss << e << " ";
    }
    log_mapper.spew() << ss.str();
  }

  /* create the slices based on the assignments
   */
  {
    size_t i = 0;
    for (PointInDomainIterator<2> pir(input.domain); pir(); ++pir, ++i) {
      TaskSlice slice;
      // slice subdomain is a single point
      slice.domain = Rect<2>(*pir, *pir);
      slice.proc = gpus[f[i]].first;
      log_mapper.spew() << "assign slice domain " << slice.domain << " to proc "
                        << slice.proc;
      slice.recurse = false;
      slice.stealable = true;
      output.slices.push_back(slice);
    }
  }
  log_mapper.spew("[exit] %s()", __FUNCTION__);
}

/*
This method can copy tasks to the map_tasks list to indicate the task should
be mapped by this mapper. The method can copy tasks to the relocate_tasks list
to indicate the task should be mapped by a mapper for a different processor.
If it does neither the task stays in the ready list.
*/
void NodeAwareMustEpochMapper::select_tasks_to_map(
    const MapperContext ctx, const SelectMappingInput &input,
    SelectMappingOutput &output) {

  log_mapper.spew("[entry] %s()", __FUNCTION__);

  for (const Task *task : input.ready_tasks) {
    log_mapper.spew("task %u", task->task_id);
  }

  // just take all tasks
  log_mapper.debug("%s(): selecting all %lu tasks", __FUNCTION__,
                   input.ready_tasks.size());
  for (const Task *task : input.ready_tasks) {
    output.map_tasks.insert(task);
  }

  log_mapper.spew("[exit] %s()", __FUNCTION__);
}

void NodeAwareMustEpochMapper::map_task(const MapperContext ctx,
                                        const Task &task,
                                        const MapTaskInput &input,
                                        MapTaskOutput &output) {
  nvtxRangePush("NodeAwareMustEpochMapper::map_task");
  log_mapper.spew("[entry] map_task()");

  log_mapper.spew("%lu task.regions:", task.regions.size());
  for (auto &rr : task.regions) {
    log_mapper.spew() << rr.region;
  }

  if (task.target_proc.kind() == Processor::TOC_PROC) {

    log_mapper.spew("task %u (parent_task=%u)", task.task_id,
                    task.parent_task->task_id);

    /* some regions may already be mapped
     */
    std::vector<bool> premapped(task.regions.size(), false);
    for (unsigned idx = 0; idx < input.premapped_regions.size(); idx++) {
      unsigned index = input.premapped_regions[idx];
      output.chosen_instances[index] = input.valid_instances[index];
      premapped[index] = true;
      printf("region %u is premapped\n", index);
    }
  }

  log_mapper.spew("map_task() defer to DefaultMapper::map_task");
  DefaultMapper::map_task(ctx, task, input, output);

  // get the runtime to call `postmap_task` when the task finishes running
  // TODO: causes a crash by itself
  output.postmap_task = false;

  log_mapper.spew("target_procs.size()=%lu", output.target_procs.size());
  for (auto &proc : output.target_procs) {
    log_mapper.spew() << proc;
  }

  log_mapper.spew("[exit] map_task()");
  nvtxRangePop();
}

void NodeAwareMustEpochMapper::map_must_epoch(const MapperContext ctx,
                                              const MapMustEpochInput &input,
                                              MapMustEpochOutput &output) {
  nvtxRangePush("NodeAwareMustEpochMapper::map_must_epoch");
  log_mapper.debug("%s(): [entry]", __FUNCTION__);

  for (const auto &task : input.tasks) {
    log_mapper.spew("task %u", task->task_id);
  }

  // ensure all tasks can run on GPU
  for (const auto &task : input.tasks) {
    bool ok = has_gpu_variant(ctx, task->task_id);
    if (!ok) {
      log_mapper.error("NodeAwareMustEpochMapper error: a task without a "
                       "TOC_PROC variant cannot be mapped.");
      assert(false);
    }
  }

  /* MappingConstraint says that certain logical regions must be in the same
  physical instance. If two tasks have a logical region in the same mapping
  constraint, they are in the same group
  */
  typedef std::set<const Task *> TaskGroup;

  auto make_intersection = [](TaskGroup &a, TaskGroup &b) {
    TaskGroup ret;
    for (auto &e : a) {
      if (b.count(e)) {
        ret.insert(e);
      }
    }
    return ret;
  };

  auto make_union = [](TaskGroup &a, TaskGroup &b) {
    TaskGroup ret(a.begin(), a.end());
    for (auto &e : b) {
      ret.insert(e);
    }
    return ret;
  };

  std::vector<TaskGroup> groups;
  // put each task in its own group in case it's not in the constraints
  for (auto task : input.tasks) {
    TaskGroup group = {task};
    groups.push_back(group);
  }
  log_mapper.debug("%s(): %lu task groups after tasks", __FUNCTION__,
                   groups.size());

  // which logical region in each task must be mapped to the same physical
  // instance
  for (auto constraint : input.constraints) {
    TaskGroup group;
    for (unsigned ti = 0; ti < constraint.constrained_tasks.size(); ++ti) {
      const Task *task = constraint.constrained_tasks[ti];
      const unsigned ri = constraint.requirement_indexes[ti];
      // If the task does not access the region, we can ignore it
      if (!task->regions[ri].is_no_access()) {
        group.insert(task);
      }
    }
    groups.push_back(group);
  }
  log_mapper.debug("%s(): %lu task groups after constraints", __FUNCTION__,
                   groups.size());

  // iteratively merge any groups that have the same task
  bool changed = true;
  while (changed) {
    std::vector<TaskGroup>::iterator srci, dsti;
    for (srci = groups.begin(); srci != groups.end(); ++srci) {
      for (dsti = groups.begin(); dsti != groups.end(); ++dsti) {
        if (srci != dsti) {
          // merge src into dst if they overlap
          if (!make_intersection(*srci, *dsti).empty()) {
            for (auto &task : *srci) {
              dsti->insert(task);
            }
            changed = true;
            goto erase_src;
          }
        }
      }
    }
    changed = false;
  erase_src:
    if (changed) {
      groups.erase(srci);
    }
  }

  log_mapper.debug("%s(): %lu task groups after merge", __FUNCTION__,
                   groups.size());
  for (size_t gi = 0; gi < groups.size(); ++gi) {
  }

  assert(groups.size() <= input.tasks.size() && "at worst, one task per group");

  /* print some info about domains
   */
  for (auto &group : groups) {
    for (auto &task : group) {

      // https://legion.stanford.edu/doxygen/class_legion_1_1_domain.html
      std::cerr << "index_domain:\n";
      Domain domain = task->index_domain;
      std::cerr << "  lo:" << domain.lo() << "-hi:" << domain.hi() << "\n";
      std::cerr << "  point:" << task->index_point << "\n";

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
        assert(false);
      }
      case 3: {
        assert(false);
      }
      default:
        assert(false);
      }

      /* print some stats about all the task's regions
       */
      std::cerr << "regions:\n";
      for (unsigned idx = 0; idx < task->regions.size(); idx++) {
        // https://legion.stanford.edu/doxygen/struct_legion_1_1_region_requirement.html
        std::cerr << "  " << idx << "\n";
        RegionRequirement req = task->regions[idx];

        Domain domain =
            runtime->get_index_space_domain(ctx, req.region.get_index_space());

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
    }
  }

  /* Build task-group communication matrix.
   overlap[gi][gi] = B, where gi is the group index in groups
   B is the size in bytes of the index space overlap between the two
   communication groups
  */
  std::map<size_t, std::map<size_t, int64_t>> overlap;

  /* return the implicit overlap of two RegionRequirements.
    some discussion on non-interference of RegionRequirements
    https://legion.stanford.edu/tutorial/privileges.html
   */
  auto region_requirement_overlap = [&](const RegionRequirement &a,
                                        const RegionRequirement &b) -> int64_t {
    // if the two regions are not rooted by the same LogicalRegion, they can't
    if (a.region.get_tree_id() != b.region.get_tree_id()) {
      return 0;
    }

    // if both are read-only they don't overlap
    if (a.privilege == READ_ONLY && b.privilege == READ_ONLY) {
      return 0;
    }

    // if a does not write then 0
    if ((a.privilege != WRITE_DISCARD) && (a.privilege != READ_WRITE)) {
      return 0;
    }

    // which fields overlap between RegionRequirements a and b
    std::set<FieldID> fieldOverlap;
    for (auto &field : b.instance_fields) {
      auto it =
          std::find(a.instance_fields.begin(), a.instance_fields.end(), field);
      if (it != a.instance_fields.end()) {
        fieldOverlap.insert(field);
      }
    }
    if (fieldOverlap.empty()) {
      return 0;
    }

    Domain aDom =
        runtime->get_index_space_domain(ctx, a.region.get_index_space());
    Domain bDom =
        runtime->get_index_space_domain(ctx, b.region.get_index_space());
    Domain intersection = aDom.intersection(bDom);

    size_t totalFieldSize = 0;
    for (auto &field : fieldOverlap) {
      totalFieldSize +=
          runtime->get_field_size(ctx, a.region.get_field_space(), field);
    }

    return intersection.get_volume() * totalFieldSize;
  };

  /* return the data size modified by Task `a` and accessed by Task `b`
   */
  auto get_task_overlap = [&](const Task *a, const Task *b) -> int64_t {
    int64_t bytes = 0;

    for (auto ra : a->regions) {
      for (auto rb : b->regions) {
        bytes += region_requirement_overlap(ra, rb);
      }
    }

    return bytes;
  };

  /* return the data size modified by TaskGroup `a` and accessed by TaskGroup
   * `b`
   */
  auto get_overlap = [&](TaskGroup &a, TaskGroup &b) -> int64_t {
    int64_t bytes = 0;
    for (auto &ta : a) {
      for (auto &tb : b) {
        bytes += get_task_overlap(ta, tb);
      }
    }
    return bytes;
  };

  nvtxRangePush("get_overlap");
  for (size_t i = 0; i < groups.size(); ++i) {
    for (size_t j = 0; j < groups.size(); ++j) {
      overlap[i][j] = get_overlap(groups[i], groups[j]);
    }
  }
  nvtxRangePop();
  printf("NodeAwareMustEpochMapper::%s(): task-group overlap matrix\n",
         __FUNCTION__);
  for (auto &src : overlap) {
    printf("NodeAwareMustEpochMapper::%s():", __FUNCTION__);
    for (auto &dst : src.second) {
      printf(" %6ld", dst.second);
    }
    printf("\n");
  }

  nvtxRangePush("closest GPU & FB");
  std::vector<std::pair<Processor, Memory>> gpus = get_gpu_fbs();
  nvtxRangePop(); // "closest GPU & FB"

  /* print our GPU number
   */
  for (size_t i = 0; i < gpus.size(); ++i) {
    log_mapper.debug() << "GPU index " << i << "proc:" << gpus[i].first
                       << "/mem:" << gpus[i].second;
  }

  /* build the distance matrix */
  nvtxRangePush("distance matrix");
  solve::Mat2D<double> distance = get_gpu_distance_matrix(gpus);
  nvtxRangePop(); // distance matrix

  printf("NodeAwareMustEpochMapper::%s(): distance matrix\n", __FUNCTION__);
  for (size_t i = 0; i < distance.shape()[1]; ++i) {
    printf("NodeAwareMustEpochMapper::%s():", __FUNCTION__);
    for (size_t j = 0; j < distance.shape()[0]; ++j) {
      printf(" %6f", distance.at(i, j));
    }
    printf("\n");
  }

  /* build the flow matrix */
  solve::Mat2D<int64_t> weight(overlap.size(), overlap.size(), 0);
  {
    size_t i = 0;
    for (auto &src : overlap) {
      size_t j = 0;
      for (auto &dst : src.second) {
        weight.at(i, j) = dst.second;
        ++j;
      }
      ++i;
    }
  }

  printf("NodeAwareMustEpochMapper::%s(): weight matrix\n", __FUNCTION__);
  for (size_t i = 0; i < weight.shape()[1]; ++i) {
    printf("NodeAwareMustEpochMapper::%s():", __FUNCTION__);
    for (size_t j = 0; j < weight.shape()[0]; ++j) {
      printf(" %6ld", weight.at(i, j));
    }
    printf("\n");
  }

  // TODO: for a must-epoch task, we should never have more tasks than agents,
  // so solve_ap only needs to work for distance >= weight
  // TODO this should be QAP
  nvtxRangePush("solve_ap");
  double cost;
  std::vector<size_t> assignment =
      solve::ap_sum_brute_force(&cost, weight, distance);
  nvtxRangePop(); // solve_ap
  if (assignment.empty()) {
    std::cerr << "couldn't find an assignment\n";
    exit(1);
  }

  printf("NodeAwareMustEpochMapper::%s(): task assignment:", __FUNCTION__);
  for (auto &e : assignment) {
    std::cerr << e << " ";
  }
  std::cerr << "\n";
  printf("NodeAwareMustEpochMapper::%s(): cost was %f\n", __FUNCTION__, cost);

  // copy the mapping to the output
  std::map<const Task *, Processor> procMap;
  for (size_t gi = 0; gi < groups.size(); ++gi) {
    for (const Task *task : groups[gi]) {
      procMap[task] = gpus[assignment[gi]].first;
    }
  }
  for (unsigned i = 0; i < input.tasks.size(); ++i) {
    output.task_processors[i] = procMap[input.tasks[i]];
  }

  // map all constraints.
  // BEGIN: lifted from default mapper
  // Now let's map the constraints, find one requirement to use for
  // mapping each of the constraints, but get the set of fields we
  // care about and the set of logical regions for all the requirements
  for (unsigned cid = 0; cid < input.constraints.size(); cid++) {

    const MappingConstraint &constraint = input.constraints[cid];
    std::vector<PhysicalInstance> &constraint_mapping =
        output.constraint_mappings[cid];
    std::set<LogicalRegion> needed_regions;
    std::set<FieldID> needed_fields;
    for (unsigned idx = 0; idx < constraint.constrained_tasks.size(); idx++) {

      const Task *task = constraint.constrained_tasks[idx];
      unsigned req_idx = constraint.requirement_indexes[idx];
      log_mapper.debug("input constraint %u: task %u region %u", cid,
                       task->task_id, req_idx);
      needed_regions.insert(task->regions[req_idx].region);
      needed_fields.insert(task->regions[req_idx].privilege_fields.begin(),
                           task->regions[req_idx].privilege_fields.end());
    }

    // Now delegate to a policy routine to decide on a memory and layout
    // constraints for this constrained instance
    std::vector<Processor> target_procs;
    for (std::vector<const Task *>::const_iterator it =
             constraint.constrained_tasks.begin();
         it != constraint.constrained_tasks.end(); ++it)
      target_procs.push_back(procMap[*it]);
    LayoutConstraintSet layout_constraints;
    layout_constraints.add_constraint(
        FieldConstraint(needed_fields, false /*!contiguous*/));
    Memory mem = default_policy_select_constrained_instance_constraints(
        ctx, constraint.constrained_tasks, constraint.requirement_indexes,
        target_procs, needed_regions, needed_fields, layout_constraints);

    LogicalRegion to_create =
        ((needed_regions.size() == 1)
             ? *(needed_regions.begin())
             : default_find_common_ancestor(ctx, needed_regions));
    PhysicalInstance inst;
    bool created;
    bool ok = runtime->find_or_create_physical_instance(
        ctx, mem, layout_constraints, std::vector<LogicalRegion>(1, to_create),
        inst, created, true /*acquire*/);
    assert(ok);
    if (!ok) {
      log_mapper.error("Default mapper error. Unable to make instance(s) "
                       "in memory " IDFMT " for index %d of constrained "
                       "task %s (ID %lld) in must epoch launch.",
                       mem.id, constraint.requirement_indexes[0],
                       constraint.constrained_tasks[0]->get_task_name(),
                       constraint.constrained_tasks[0]->get_unique_id());
      assert(false);
    }
    constraint_mapping.push_back(inst);
  }
  // END: lifted from default mapper

#if 0
  printf("NodeAwareMustEpochMapper::%s(): actually just use "
         "DefaultMapper::map_must_epoch()\n",
         __FUNCTION__);
  DefaultMapper::map_must_epoch(ctx, input, output);
#endif
  printf("NodeAwareMustEpochMapper::%s(): [exit]\n", __FUNCTION__);

  nvtxRangePop(); // NodeAwareMustEpochMapper::map_must_epoch
}

/*
struct PostMapInput {
  std::vector<std::vector<PhysicalInstance> >     mapped_regions;
  std::vector<std::vector<PhysicalInstance> >     valid_instances;
};

struct PostMapOutput {
  std::vector<std::vector<PhysicalInstance> >     chosen_instances;
};
*/
void NodeAwareMustEpochMapper::postmap_task(const MapperContext ctx,
                                            const Task &task,
                                            const PostMapInput &input,
                                            PostMapOutput &output) {

  log_mapper.debug() << "in NodeAwareMustEpochMapper::postmap_task";
}

// TODO: incomplete
int64_t NodeAwareMustEpochMapper::get_logical_region_overlap(
    MapperContext ctx, const LogicalRegion &lra, const LogicalRegion &lrb) {
  // if the two regions are not rooted by the same LogicalRegion, they can't
  // overlap
  if (lra.get_tree_id() != lrb.get_tree_id()) {
    return 0;
  }

  // if the two regions don't have the same field space, they can't overlap
  // TODO: true?
  if (lra.get_field_space() != lrb.get_field_space()) {
    return 0;
  }

  Domain aDom = runtime->get_index_space_domain(ctx, lra.get_index_space());
  Domain bDom = runtime->get_index_space_domain(ctx, lrb.get_index_space());
  Domain intersection = aDom.intersection(bDom);

  return intersection.get_volume();
}

int64_t NodeAwareMustEpochMapper::get_region_requirement_overlap(
    MapperContext ctx, const RegionRequirement &rra,
    const RegionRequirement &rrb) {

  // If the region requirements have no shared fields, the overlap is zero
  std::set<FieldID> sharedFields;
  for (auto &field : rrb.privilege_fields) {
    auto it = std::find(rra.privilege_fields.begin(),
                        rra.privilege_fields.end(), field);
    if (it != rra.privilege_fields.end()) {
      sharedFields.insert(field);
    }
  }
  if (sharedFields.empty()) {
    return 0;
  }

  // if the RegionRequirement was a logical partition, the caller
  // should have converted it to a logical region for us
  LogicalRegion lra;
  if (rra.region.exists()) {
    lra = rra.region;
  } else if (rra.partition.exists()) {
    assert(false);
  } else {
    assert(false);
  }

  LogicalRegion lrb;
  if (rrb.region.exists()) {
    lrb = rrb.region;
  } else if (rrb.partition.exists()) {
    assert(false);
  } else {
    assert(false);
  }

  int64_t sharedFieldSize = 0;
  for (auto &f : sharedFields) {
    sharedFieldSize += runtime->get_field_size(ctx, lra.get_field_space(), f);
  }

  int64_t numPoints = get_logical_region_overlap(ctx, lra, lrb);

  return numPoints * sharedFieldSize;
}

std::vector<std::pair<Processor, Memory>>
NodeAwareMustEpochMapper::get_gpu_fbs() {

  printf("NodeAwareMustEpochMapper::%s(): GPU-FB affinities\n", __FUNCTION__);
  std::vector<Machine::ProcessorMemoryAffinity> procMemAffinities;
  machine.get_proc_mem_affinity(procMemAffinities);
  for (auto &aff : procMemAffinities) {
    // find the closes FB mem to each GPU
    if (aff.p.kind() == Processor::TOC_PROC &&
        aff.m.kind() == Memory::GPU_FB_MEM) {
      std::cerr << aff.p << "-" << aff.m << " " << aff.bandwidth << " "
                << aff.latency << "\n";
    }
  }

  std::vector<std::pair<Processor, Memory>> gpus;
  {
    // find the highest-bandwidth fb each GPU has access to
    std::map<Processor, Machine::ProcessorMemoryAffinity> best;
    for (auto &aff : procMemAffinities) {
      if (aff.p.kind() == Processor::TOC_PROC &&
          aff.m.kind() == Memory::GPU_FB_MEM) {
        auto it = best.find(aff.p);
        if (it != best.end()) {
          if (aff.bandwidth > it->second.bandwidth) {
            it->second = aff;
          }
        } else {
          best[aff.p] = aff;
        }
      }
    }

    size_t i = 0;
    for (auto &kv : best) {
      assert(kv.first == kv.second.p);
      assert(kv.first.kind() == Processor::TOC_PROC);
      assert(kv.second.m.kind() == Memory::GPU_FB_MEM);
      log_mapper.spew() << "proc " << kv.first << ": closes mem=" << kv.second.m
                        << " bw=" << kv.second.bandwidth
                        << "latency=" << kv.second.latency;
      std::pair<Processor, Memory> pmp;
      pmp.first = kv.first;
      pmp.second = kv.second.m;
      gpus.push_back(pmp);
      ++i;
    }
  }

  return gpus;
}

solve::Mat2D<double> NodeAwareMustEpochMapper::get_gpu_distance_matrix(
    const std::vector<std::pair<Processor, Memory>> &gpus) {

  printf("NodeAwareMustEpochMapper::%s(): GPU memory-memory affinities\n",
         __FUNCTION__);
  std::vector<Machine::MemoryMemoryAffinity> memMemAffinities;
  machine.get_mem_mem_affinity(memMemAffinities);
  for (auto &aff : memMemAffinities) {
    log_mapper.spew() << aff.m1 << "-" << aff.m2 << " " << aff.bandwidth << " "
                      << aff.latency;
  }

  solve::Mat2D<double> ret(gpus.size(), gpus.size(), 0);
  size_t i = 0;
  for (auto &src : gpus) {
    size_t j = 0;
    for (auto &dst : gpus) {
      // TODO: self distance is 0
      if (src.first == dst.first) {
        ret.at(i, j) = 0;
        // Look for a MemoryMemoryAffinity between the GPU fbs and use that
      } else {
        bool found = false;
        for (auto &mma : memMemAffinities) {
          if (mma.m1 == src.second && mma.m2 == dst.second) {
            ret.at(i, j) = 1.0 / mma.bandwidth;
            found = true;
            break;
          }
        }
        if (!found) {
          log_mapper.error("couldn't find mem-mem affinity for GPU FBs");
          assert(false);
        }
      }
      ++j;
    }
    ++i;
  }
  return ret;
}
