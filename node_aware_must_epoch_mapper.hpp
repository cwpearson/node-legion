#pragma once

// based off of https://legion.stanford.edu/tutorial/custom_mappers.html

#include "legion.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

class NodeAwareMustEpochMapper : public DefaultMapper {
public:
  NodeAwareMustEpochMapper(MapperRuntime *rt, Machine machine, Processor local,
                           const char *mapper_name);

public:
  /* Do some node-aware must-epoch mapping

  */
  virtual void map_must_epoch(const MapperContext ctx,
                              const MapMustEpochInput &input,
                              MapMustEpochOutput &output);

  /* Replace the default mapper with a NodeAwareMustEpochMapper

  to be passed to Runtime::add_registration_callback
   */
  static void mapper_registration(Machine machine, Runtime *rt,
                                  const std::set<Processor> &local_procs);

protected:
  /* true if there is a GPU variant of the task
   */
  bool has_gpu_variant(const MapperContext ctx, TaskID id);
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

void NodeAwareMustEpochMapper::map_must_epoch(const MapperContext ctx,
                                              const MapMustEpochInput &input,
                                              MapMustEpochOutput &output) {
  printf("NodeAwareMustEpochMapper::%s(): [entry]\n", __FUNCTION__);

  // ensure all tasks can run on GPU
  for (const auto &task : input.tasks) {
    assert(has_gpu_variant(ctx, task->task_id));
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
  printf("NodeAwareMustEpochMapper::%s(): %lu task groups after tasks\n",
         __FUNCTION__, groups.size());

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
  printf("NodeAwareMustEpochMapper::%s(): %lu task groups after constraints\n",
         __FUNCTION__, groups.size());

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
  printf("NodeAwareMustEpochMapper::%s(): %lu task groups after merge\n",
         __FUNCTION__, groups.size());

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
      totalFieldSize += runtime->get_field_size(ctx, a.region.get_field_space(), field);
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

  /* return the data size modified by TaskGroup `a` and accessed by TaskGroup `b`
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

  for (size_t i = 0; i < groups.size(); ++i) {
    for (size_t j = 0; j < groups.size(); ++j) {
      overlap[i][j] = get_overlap(groups[i], groups[j]);
    }
  }
  printf("NodeAwareMustEpochMapper::%s(): task-group overlap matrix\n",
         __FUNCTION__);
  for (auto &src : overlap) {
    printf("NodeAwareMustEpochMapper::%s():", __FUNCTION__);
    for (auto &dst : src.second) {
      printf(" %ld", dst.second);
    }
    printf("\n");
  }
  printf("NodeAwareMustEpochMapper::%s(): %lu task groups after merge\n",
         __FUNCTION__, groups.size());

  printf("NodeAwareMustEpochMapper::%s(): actually just use "
         "DefaultMapper::map_must_epoch()\n",
         __FUNCTION__);
  DefaultMapper::map_must_epoch(ctx, input, output);
  printf("NodeAwareMustEpochMapper::%s(): [exit]\n", __FUNCTION__);
}