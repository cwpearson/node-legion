#include "node_aware_mapper.hpp"

#define NAM_USE_NVTX

using namespace Legion;
using namespace Legion::Mapping;

Logger log_nam("node_aware_mapper");

NodeAwareMustEpochMapper::NodeAwareMustEpochMapper(MapperRuntime *rt,
                                                   Machine machine,
                                                   Processor local,
                                                   const char *mapper_name)
    : DefaultMapper(rt, machine, local, mapper_name) {
      gpuFBs = get_gpu_fbs();
    }

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

/* generate a weight matrix from the slice
 */
template <unsigned DIM>
void NodeAwareMustEpochMapper::index_task_create_weight_matrix(
    solve::Mat2D<int64_t> &weight, MapperContext ctx, const Domain &inputDomain,
    const Task &task) {
  assert(inputDomain.dim == DIM);

  {
    log_nam.spew("point space numbering:");
    int i = 0;
    for (PointInDomainIterator<DIM> pir(inputDomain); pir(); ++pir, ++i) {
      log_nam.spew() << i << " p=" << *pir;
      for (int r = 0; r < task.regions.size(); ++r) {
        // is this a DomainPoint or a color
        LogicalRegion li = runtime->get_logical_subregion_by_color(
            ctx, task.regions[r].partition, DomainPoint(*pir));
        log_nam.spew() << "  " << r << " "
                   << runtime->get_index_space_domain(ctx,
                                                      li.get_index_space());
      }
    }
  }

  int i = 0;
  for (PointInDomainIterator<DIM> pi(inputDomain); pi(); ++pi, ++i) {
    int j = 0;
    for (PointInDomainIterator<DIM> pj(inputDomain); pj(); ++pj, ++j) {

      int64_t bytes = 0;
      for (int ri = 0; ri < task.regions.size(); ++ri) {
        LogicalRegion li = runtime->get_logical_subregion_by_color(
            ctx, task.regions[ri].partition, DomainPoint(*pi));

        // TODO: is this sketchy
        // create fake region requirements that move the partition to the
        // region so we can reuse the region distance code
        RegionRequirement rra = task.regions[ri];
        rra.partition = LogicalPartition::NO_PART;
        rra.region = li;
        for (int rj = 0; rj < task.regions.size(); ++rj) {
          LogicalRegion lj = runtime->get_logical_subregion_by_color(
              ctx, task.regions[rj].partition, DomainPoint(*pj));

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
      // std::cerr << "slice " << *pi << " " << *pj << " bytes=" << bytes <<
      // "\n";
      weight.at(i, j) = bytes;
    }
  }
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
void NodeAwareMustEpochMapper::slice_task(MapperContext ctx, const Task &task,
                                          const SliceTaskInput &input,
                                          SliceTaskOutput &output) {
  log_nam.spew("[entry] %s()", __FUNCTION__);
#ifdef NAM_USE_NVTX
  nvtxRangePush("slice_task");
#endif

  log_nam.spew() << __FUNCTION__ << "(): input.domain_is = " << input.domain_is;
  log_nam.spew() << __FUNCTION__ << "(): input.domain    = " << input.domain;

  /* Build the GPU distance matrix
   */
  {
    log_nam.spew("GPU numbering:");
    int i = 0;
    for (auto p : gpuFBs) {
      log_nam.spew() << i << " gpu=" << p.first << " fb=" << p.second;
    }
  }
#ifdef NAM_USE_NVTX
  nvtxRangePush("get_gpu_distance_matrix");
#endif
  solve::Mat2D<double> distance = get_gpu_distance_matrix(gpuFBs);
#ifdef NAM_USE_NVTX
  nvtxRangePop();
#endif

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
  solve::Mat2D<int64_t> weight(input.domain.get_volume(),
                               input.domain.get_volume(), 0);

  // assert(input.domain.dim == 2 && "TODO: only implemented for dim=2");

#ifdef NAM_USE_NVTX
  nvtxRangePush("index_task_create_weight_matrix");
#endif
  switch (input.domain.dim) {
  case 1:
    index_task_create_weight_matrix<1>(weight, ctx, input.domain, task);
    break;
  case 2:
    index_task_create_weight_matrix<2>(weight, ctx, input.domain, task);
    break;
  case 3:
    index_task_create_weight_matrix<3>(weight, ctx, input.domain, task);
    break;
  default:
    log_nam.fatal() << "unhandled dimensionality in slice_task";
  }
#ifdef NAM_USE_NVTX
  nvtxRangePop();
#endif

  printf("NodeAwareMustEpochMapper::%s(): weight matrix\n", __FUNCTION__);
  for (size_t i = 0; i < weight.shape()[1]; ++i) {
    printf("NodeAwareMustEpochMapper::%s():", __FUNCTION__);
    for (size_t j = 0; j < weight.shape()[0]; ++j) {
      printf(" %6ld", weight.at(i, j));
    }
    printf("\n");
  }

#ifdef NAM_USE_NVTX
  nvtxRangePush("solve_ap");
#endif
  double cost;
  std::vector<size_t> f = solve::ap_sum_brute_force(&cost, weight, distance);
  assert(f.size() == weight.shape()[0]);
#ifdef NAM_USE_NVTX
  nvtxRangePop();
#endif

  {
    std::stringstream ss;
    for (auto &e : f) {
      ss << e << " ";
    }
    log_nam.info() << "assignment: " << ss.str();
    log_nam.info() << "cost:       " << cost;
  }

  /* create the slices based on the assignments
     TODO: template function?
   */
  switch (input.domain.dim) {
  case 2: {
    size_t i = 0;
    for (PointInDomainIterator<2> pir(input.domain); pir(); ++pir, ++i) {
      TaskSlice slice;
      // slice subdomain is a single point
      slice.domain = Rect<2>(*pir, *pir);
      slice.proc = gpuFBs[f[i]].first;
      log_nam.spew() << "assign slice domain " << slice.domain << " to proc "
                 << slice.proc;
      slice.recurse = false;
      slice.stealable = true;
      output.slices.push_back(slice);
    }
    break;
  }
  case 3: {
    size_t i = 0;
    for (PointInDomainIterator<3> pir(input.domain); pir(); ++pir, ++i) {
      TaskSlice slice;
      // slice subdomain is a single point
      slice.domain = Rect<3>(*pir, *pir);
      slice.proc = gpuFBs[f[i]].first;
      log_nam.spew() << "assign slice domain " << slice.domain << " to proc "
                 << slice.proc;
      slice.recurse = false;
      slice.stealable = true;
      output.slices.push_back(slice);
    }
    break;
  }
  }

#ifdef NAM_USE_NVTX
  nvtxRangePop(); // slice_task
#endif
  log_nam.spew("[exit] %s()", __FUNCTION__);
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
This method can copy tasks to the map_tasks list to indicate the task should
be mapped by this mapper. The method can copy tasks to the relocate_tasks list
to indicate the task should be mapped by a mapper for a different processor.
If it does neither the task stays in the ready list.
*/
void NodeAwareMustEpochMapper::select_tasks_to_map(
    const MapperContext ctx, const SelectMappingInput &input,
    SelectMappingOutput &output) {

  log_nam.spew("[entry] %s()", __FUNCTION__);
#ifdef NAM_USE_NVTX
  nvtxRangePush("NAM::select_tasks_to_map()");
#endif


  for (const Task *task : input.ready_tasks) {
    log_nam.spew("task %u", task->task_id);
  }

  // just take all tasks
  log_nam.debug("%s(): selecting all %lu tasks", __FUNCTION__,
            input.ready_tasks.size());
  for (const Task *task : input.ready_tasks) {
    output.map_tasks.insert(task);
  }

#ifdef NAM_USE_NVTX
  nvtxRangePop();
#endif
  log_nam.spew("[exit] %s()", __FUNCTION__);
}

void NodeAwareMustEpochMapper::map_task(const MapperContext ctx,
                                        const Task &task,
                                        const MapTaskInput &input,
                                        MapTaskOutput &output) {
  log_nam.spew("[entry] map_task()");
#ifdef NAM_USE_NVTX
  nvtxRangePush("NAM::map_task");
#endif

  log_nam.spew("%lu task.regions:", task.regions.size());
  for (auto &rr : task.regions) {
    log_nam.spew() << rr.region;
  }

  if (task.target_proc.kind() == Processor::TOC_PROC) {

    log_nam.spew("task %u (parent_task=%u)", task.task_id,
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

  log_nam.spew("map_task() defer to DefaultMapper::map_task");
  DefaultMapper::map_task(ctx, task, input, output);

  // get the runtime to call `postmap_task` when the task finishes running
  // TODO: causes a crash by itself
  output.postmap_task = false;

  log_nam.spew("target_procs.size()=%lu", output.target_procs.size());
  for (auto &proc : output.target_procs) {
    log_nam.spew() << proc;
  }

  log_nam.spew("[exit] map_task()");
#ifdef NAM_USE_NVTX
  nvtxRangePop();
#endif
}

void NodeAwareMustEpochMapper::map_must_epoch(const MapperContext ctx,
                                              const MapMustEpochInput &input,
                                              MapMustEpochOutput &output) {
#ifdef NAM_USE_NVTX
  nvtxRangePush("NodeAwareMustEpochMapper::map_must_epoch");
#endif
  log_nam.debug("%s(): [entry]", __FUNCTION__);

  for (const auto &task : input.tasks) {
    log_nam.spew("task %u", task->task_id);
  }

  // ensure all tasks can run on GPU
  for (const auto &task : input.tasks) {
    bool ok = has_gpu_variant(ctx, task->task_id);
    if (!ok) {
      log_nam.error("NodeAwareMustEpochMapper error: a task without a "
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
  log_nam.debug("%s(): %lu task groups after tasks", __FUNCTION__, groups.size());

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
  log_nam.debug("%s(): %lu task groups after constraints", __FUNCTION__,
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

  log_nam.debug("%s(): %lu task groups after merge", __FUNCTION__, groups.size());
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

#ifdef NAM_USE_NVTX
  nvtxRangePush("get_overlap");
#endif
  for (size_t i = 0; i < groups.size(); ++i) {
    for (size_t j = 0; j < groups.size(); ++j) {
      overlap[i][j] = get_overlap(groups[i], groups[j]);
    }
  }
#ifdef NAM_USE_NVTX
  nvtxRangePop();
#endif
  printf("NodeAwareMustEpochMapper::%s(): task-group overlap matrix\n",
         __FUNCTION__);
  for (auto &src : overlap) {
    printf("NodeAwareMustEpochMapper::%s():", __FUNCTION__);
    for (auto &dst : src.second) {
      printf(" %6ld", dst.second);
    }
    printf("\n");
  }


  /* print our GPU number
   */
  for (size_t i = 0; i < gpuFBs.size(); ++i) {
    log_nam.debug() << "GPU index " << i << "proc:" << gpuFBs[i].first
                << "/mem:" << gpuFBs[i].second;
  }

/* build the distance matrix */
#ifdef NAM_USE_NVTX
  nvtxRangePush("distance matrix");
#endif
  solve::Mat2D<double> distance = get_gpu_distance_matrix(gpuFBs);
#ifdef NAM_USE_NVTX
  nvtxRangePop(); // distance matrix
#endif

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
      printf(" %8ld", weight.at(i, j));
    }
    printf("\n");
  }

// TODO: for a must-epoch task, we should never have more tasks than agents,
// so solve_ap only needs to work for distance >= weight
// TODO this should be QAP
#ifdef NAM_USE_NVTX
  nvtxRangePush("solve_ap");
#endif
  double cost;
  std::vector<size_t> assignment =
      solve::ap_sum_brute_force(&cost, weight, distance);
#ifdef NAM_USE_NVTX
  nvtxRangePop(); // solve_ap
#endif
  if (assignment.empty()) {
    log_nam.fatal() << "couldn't find an assignment";
    exit(1);
  }

  {
    std::stringstream ss;
    ss << __FUNCTION__ << "(): task assignment:";
    for (auto &e : assignment) {
      ss << e << " ";
    }
    log_nam.info() << ss.str();
    log_nam.info() << __FUNCTION__ << "(): cost was " << cost;
  }

  // copy the mapping to the output
  std::map<const Task *, Processor> procMap;
  for (size_t gi = 0; gi < groups.size(); ++gi) {
    for (const Task *task : groups[gi]) {
      procMap[task] = gpuFBs[assignment[gi]].first;
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
      log_nam.debug("input constraint %u: task %u region %u", cid, task->task_id,
                req_idx);
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
      log_nam.error("Default mapper error. Unable to make instance(s) "
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
#ifdef NAM_USE_NVTX
  nvtxRangePop(); // NodeAwareMustEpochMapper::map_must_epoch
#endif
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

  log_nam.debug() << "in NodeAwareMustEpochMapper::postmap_task";
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

TaskHash NodeAwareMustEpochMapper::compute_task_hash(const Task &task) {
  return DefaultMapper::compute_task_hash(task);
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
      log_nam.spew() << "proc " << kv.first << ": closes mem=" << kv.second.m
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
    log_nam.spew() << aff.m1 << "-" << aff.m2 << " " << aff.bandwidth << " "
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
          log_nam.error("couldn't find mem-mem affinity for GPU FBs");
          assert(false);
        }
      }
      ++j;
    }
    ++i;
  }
  return ret;
}
