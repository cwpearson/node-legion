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

#if 0
/* assignment problem utilities */
namespace ap {

/* brute-force solution to QAP, placing the final cost in costp if not null
 */
inline std::vector<size_t> solve_qap(double *costp,
                                     const solve::Mat2D<int64_t> &w,
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
inline std::vector<size_t> solve_ap_swap2(double *costp,
                                          const solve::Mat2D<int64_t> &w,
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
  double bestCost = solve::max_cost(w, d, f);
  stats.insert(bestCost);

  bool changed = true;
  while (changed) {
    changed = false;

    // check all possible swaps
    for (size_t i = 0; i < numTasks; ++i) {
      for (size_t j = i + 1; j < numTasks; ++j) {
        std::vector<size_t> swappedF = f; // swapped f
        std::swap(swappedF[i], swappedF[j]);

        double swappedCost = solve::max_cost(w, d, swappedF);
        stats.insert(swappedCost);

        if (swappedCost < bestCost) {
          bestCost = swappedCost;
          bestF = swappedF;
          changed = true;
          goto body_end; // fast exit
        }
      }
    }

  body_end:;
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
#endif

using namespace Legion;
using namespace Legion::Mapping;

typedef unsigned long long TaskHash;

class NodeAwareMustEpochMapper : public DefaultMapper {

private:
  /* pairs of each GPU and its closes FB
  */
  std::vector<std::pair<Processor, Memory>> gpuFBs;


  /* compute the hash of a task
   */
  TaskHash compute_task_hash(const Task &task);

  /* can we just cache results of mapTask?
   */
  std::map<TaskHash, MapTaskOutput> mapTaskCache;

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

  /* fill the weight matrix for an index space task
   */
  template <unsigned DIM>
  void index_task_create_weight_matrix(solve::Mat2D<int64_t> &weight,
                                       MapperContext ctx,
                                       const Domain &inputDomain,
                                       const Task &task);

public:
  NodeAwareMustEpochMapper(MapperRuntime *rt, Machine machine, Processor local,
                           const char *mapper_name);

  /*
  If the task is an index task launch the runtime calls slice_task to divide
  the index task into a set of slices that contain point tasks. One slice
  corresponds to one target processor. Each slice identifies an index space, a
  subregion of the original domain and a target processor. All of the point
  tasks for the subregion will be mapped by the mapper for the target
  processor.

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

  Typically, the idea is that groups of nearby tasks would have locality, so
  we would group these point tasks into slices, each of which has a rect of
  point tasks and then assign those groups to procs. At the end of the day,
  each of those point tasks is still mapped individually.

  Instead, we will treat each point task independently, and create slices
  of 1. This allows us to figure out which point tasks communicate and which
  should be on nearby GPUs.
  */
  virtual void slice_task(MapperContext ctx, const Task &task,
                          const SliceTaskInput &input,
                          SliceTaskOutput &output) override;

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

  /* return GPUs paired with their closest FBs
   */
  std::vector<std::pair<Processor, Memory>> get_gpu_fbs();

  /* return the distance matrix between the processors in `procs`
   */
  solve::Mat2D<double> get_gpu_distance_matrix(
      const std::vector<std::pair<Processor, Memory>> &gpus);
};
