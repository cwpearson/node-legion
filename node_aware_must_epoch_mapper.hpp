#pragma once

// based off of https://legion.stanford.edu/tutorial/custom_mappers.html

#include "legion.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "default_mapper.h"

/* assignment problem utilities */
namespace ap {

struct Rect {
  uint64_t x;
  uint64_t y;
  Rect(uint64_t _x, uint64_t _y) : x(_x), y(_y) {}

  uint64_t flatten() const noexcept { return x * y; }
  bool operator==(const Rect &rhs) const noexcept {
    return x == rhs.x && y == rhs.y;
  }
  bool operator!=(const Rect &rhs) const noexcept { return !((*this) == rhs); }
};

template <typename T> class Mat2D {
private:
  void swap(Mat2D &other) noexcept {
    std::swap(data_, other.data_);
    std::swap(rect_, other.rect_);
  }

public:
  std::vector<T> data_;
  Rect rect_;

  Mat2D() : rect_(0, 0) {}
  Mat2D(int64_t x, int64_t y) : data_(x * y), rect_(x, y) {}
  Mat2D(int64_t x, int64_t y, const T &v) : data_(x * y, v), rect_(x, y) {}
  Mat2D(Rect s) : Mat2D(s.x, s.y) {}
  Mat2D(Rect s, const T &val) : Mat2D(s.x, s.y, val) {}

  Mat2D(const std::initializer_list<std::initializer_list<T>> &ll) : Mat2D() {

    if (ll.size() > 0) {
      resize(ll.begin()->size(), ll.size());
    }

    auto llit = ll.begin();
    for (size_t i = 0; i < rect_.y; ++i, ++llit) {
      assert(llit->size() == rect_.x);
      auto lit = llit->begin();
      for (size_t j = 0; j < rect_.x; ++j, ++lit) {
        at(i, j) = *lit;
      }
    }
  }

  Mat2D(const Mat2D &other) = default;
  Mat2D &operator=(const Mat2D &rhs) = default;
  Mat2D(Mat2D &&other) = default;
  Mat2D &operator=(Mat2D &&rhs) = default;

  inline T &at(int64_t i, int64_t j) noexcept {
    assert(i < rect_.y);
    assert(j < rect_.x);
    return data_[i * rect_.x + j];
  }
  inline const T &at(int64_t i, int64_t j) const noexcept {
    assert(i < rect_.y);
    assert(j < rect_.x);
    return data_[i * rect_.x + j];
  }

  /* grow or shrink to [x,y], preserving top-left corner of matrix */
  void resize(int64_t x, int64_t y) {
    Mat2D mat(x, y);

    const int64_t copyRows = std::min(mat.rect_.y, rect_.y);
    const int64_t copyCols = std::min(mat.rect_.x, rect_.x);

    for (int64_t i = 0; i < copyRows; ++i) {
      std::memcpy(&mat.at(i, 0), &at(i, 0), copyCols * sizeof(T));
    }
    swap(mat);
  }

  inline const Rect &shape() const noexcept { return rect_; }

  bool operator==(const Mat2D &rhs) const noexcept {
    if (rect_ != rhs.rect_) {
      return false;
    }
    for (uint64_t i = 0; i < rect_.y; ++i) {
      for (uint64_t j = 0; j < rect_.x; ++j) {
        if (data_[i * rect_.x + j] != rhs.data_[i * rect_.x + j]) {
          return false;
        }
      }
    }
    return true;
  }

  template <typename S> Mat2D &operator/=(const S &s) {
    for (uint64_t i = 0; i < rect_.y; ++i) {
      for (uint64_t j = 0; j < rect_.x; ++j) {
        data_[i * rect_.x + j] /= s;
      }
    }
    return *this;
  }
};

namespace detail {

inline double cost_product(double we, double de) {
  if (0 == we || 0 == de) {
    return 0;
  } else {
    return we * de;
  }
}


inline double cost(const Mat2D<double> &w,      // weight
                   const Mat2D<double> &d,      // distance
                   const std::vector<size_t> &f // agent for each task
) {
  assert(w.shape().x == w.shape().y);
  assert(d.shape().x == d.shape().y);
  assert(w.shape().x == f.size()); // one weight per task

  double ret = 0;

  for (size_t a = 0; a < w.shape().y; ++a) {
    for (size_t b = 0; b < w.shape().x; ++b) {
      double p;
      size_t fa = f[a];
      size_t fb = f[b];
      assert(fa < d.shape().x && "task assigned to non-existant agent");
      assert(fb < d.shape().y && "task assigned to non-existant agent");
      p = cost_product(w.at(a, b), d.at(f[a], f[b]));
      ret += p;
    }
  }

  return ret;
}

} // namespace detail

/* brute-force solution to QAP, placing the final cost in costp if not null
 */
inline std::vector<size_t> solve_qap(double *costp, const Mat2D<double> &w,
                                     Mat2D<double> &d) {

  assert(w.shape() == d.shape());
  assert(w.shape().x == d.shape().y);

  std::vector<size_t> f(w.shape().x);
  for (size_t i = 0; i < w.shape().x; ++i) {
    f[i] = i;
  }

  std::vector<size_t> bestF = f;
  double bestCost = detail::cost(w, d, f);
  do {
    const double cost = detail::cost(w, d, f);
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

/* brute-force solution to assignment problem

   objective: minimize total flow * distance product under assignment

   `s`: cardinality, maximum amount of tasks per agent
   `w`: flow between tasks
   `d`: distance between agents

   return empty vector if no valid assignment was found
 */
inline std::vector<size_t> solve_ap(double *costp, const Mat2D<double> &w,
                                    Mat2D<double> &d, const int64_t s) {

  assert(d.shape().x == d.shape().y);
  assert(w.shape().x == w.shape().y);

  const int64_t numAgents = d.shape().x;
  const int64_t numTasks = w.shape().x;

  std::vector<size_t> f(numTasks, 0);

  auto next_f = [&]() -> bool {
    // if f == [numAgents-1, numAgents-1, ...]
    if (std::all_of(f.begin(), f.end(), [&](size_t u) {
          return u == (numAgents ? numAgents - 1 : 0);
        })) {
      return false;
    }

    bool carry;
    int64_t i = 0;
    do {
      carry = false;
      ++f[i];
      if (f[i] >= numAgents) {
        f[i] = 0;
        ++i;
        carry = true;
      }
    } while (true == carry);
    return true;
  };

  std::vector<size_t> bestF;
  double bestCost = std::numeric_limits<double>::infinity();
  do {
    // if cardinality of any assignment is too large, skip cost check
    for (int64_t a = 0; a < numAgents; ++a) {
      if (std::count(f.begin(), f.end(), a) > s) {
        continue;
      }
    }

    const double cost = detail::cost(w, d, f);
    if (bestCost > cost) {
      bestF = f;
      bestCost = cost;
    }
  } while (next_f());

  if (costp) {
    *costp = bestCost;
  }

  return bestF;
}

} // namespace ap

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
      printf(" %6ld", dst.second);
    }
    printf("\n");
  }

  // the framebuffer for each GPU
  std::map<Processor, Memory> gpuFbs;

  printf("NodeAwareMustEpochMapper::%s(): GPU-FB affinities\n",
         __FUNCTION__);
  std::vector<Machine::ProcessorMemoryAffinity> procMemAffinities;
  machine.get_proc_mem_affinity(procMemAffinities);
  for (auto &aff : procMemAffinities) {
    if (aff.p.kind() == Processor::TOC_PROC &&
        aff.m.kind() == Memory::GPU_FB_MEM) {
      gpuFbs[aff.p] = aff.m;
      std::cerr << aff.p << "-" << aff.m << " " << aff.bandwidth << " "
                << aff.latency << "\n";
    }
  }

  printf("NodeAwareMustEpochMapper::%s(): GPU memory-memory affinities\n",
         __FUNCTION__);
  std::vector<Machine::MemoryMemoryAffinity> memMemAffinities;
  machine.get_mem_mem_affinity(memMemAffinities);
  for (auto &aff : memMemAffinities) {
    std::cerr << aff.m1 << "-" << aff.m2 << " " << aff.bandwidth << " "
              << aff.latency << "\n";
  }

  /* build the distance matrix */
  assert(gpuFbs.size() > 0);
  ap::Mat2D<double> distance(gpuFbs.size(), gpuFbs.size(), 0);
  {
    size_t i = 0;
    for (auto &src : gpuFbs) {
      size_t j = 0;
      for (auto &dst : gpuFbs) {
        // TODO: self distance is 0
        if (src == dst) {
          distance.at(i, j) = 0;
        } else {
          bool found = false;
          for (auto &mma : memMemAffinities) {
            if (mma.m1 == src.second && mma.m2 == dst.second) {
              distance.at(i, j) = mma.bandwidth;
              found = true;
              break;
            }
          }
          assert(found && "couldn't find mem-mem affinity for GPU FBs");
        }
        ++j;
      }
      ++i;
    }
  }

  printf("NodeAwareMustEpochMapper::%s(): distance matrix\n", __FUNCTION__);
  for (size_t i = 0; i < distance.shape().y; ++i) {
    printf("NodeAwareMustEpochMapper::%s():", __FUNCTION__);
    for (size_t j = 0; j < distance.shape().x; ++j) {
      printf(" %6f", distance.at(i, j));
    }
    printf("\n");
  }

  /* build the flow matrix */
  ap::Mat2D<double> weight(overlap.size(), overlap.size(), 0);
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
  for (size_t i = 0; i < weight.shape().y; ++i) {
    printf("NodeAwareMustEpochMapper::%s():", __FUNCTION__);
    for (size_t j = 0; j < weight.shape().x; ++j) {
      printf(" %6f", weight.at(i, j));
    }
    printf("\n");
  }

  // Max task -> agent assignment
  int64_t cardinality = (overlap.size() + gpuFbs.size() - 1) / gpuFbs.size();
  printf("NodeAwareMustEpochMapper::%s(): max cardinality %ld\n", __FUNCTION__,
         cardinality);

  double cost;
  std::vector<size_t> assignment =
      ap::solve_ap(&cost, weight, distance, cardinality);

  printf("NodeAwareMustEpochMapper::%s(): task assignment:\n", __FUNCTION__);
  for (auto &e : assignment) {
    std::cerr << e << " ";
  }
  std::cerr << "\n";
  printf("NodeAwareMustEpochMapper::%s(): cost was %f\n", __FUNCTION__,
         cost);

  printf("NodeAwareMustEpochMapper::%s(): actually just use "
         "DefaultMapper::map_must_epoch()\n",
         __FUNCTION__);
  DefaultMapper::map_must_epoch(ctx, input, output);
  printf("NodeAwareMustEpochMapper::%s(): [exit]\n", __FUNCTION__);
}