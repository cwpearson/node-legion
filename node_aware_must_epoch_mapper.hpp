#pragma once

// based off of https://legion.stanford.edu/tutorial/custom_mappers.html

#include "legion.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "default_mapper.h"

#include <nvToolsExt.h>

/* assignment problem utilities */
namespace ap {

struct RollingStatistics {
  RollingStatistics() { reset(); }
  size_t n;
  double sum_;
  double min_;
  double max_;

  void reset() {
    n = 0;
    sum_ = 0;
    min_ = std::numeric_limits<double>::infinity();
    max_ = -1 * std::numeric_limits<double>::infinity();
  }

  void insert(double d) {
    ++n;
    sum_ += d;
    min_ = std::min(min_, d);
    max_ = std::max(max_, d);
  }

  double mean() const noexcept { return sum_ / n; }
  double min() const noexcept { return min_; }
  double max() const noexcept { return max_; }
  double count() const noexcept { return n; }
};

template <unsigned N> class Extent {
  int64_t x[N];

public:
  Extent() = default;
  Extent(int64_t _x[N]) { x = _x; }
  Extent(Extent &&other) = default;
  Extent(const Extent &other) = default;

  Extent &operator=(Extent &&other) = default;
  Extent &operator=(const Extent &other) = default;

  uint64_t flatten() const noexcept {
    int64_t p = x[0];
    for (unsigned i = 1; i < N; ++i) {
      p *= x[i];
    }
    return p;
  }
  bool operator==(const Extent &rhs) const noexcept {
    bool ret = true;
    for (unsigned i = 0; i < N; ++i) {
      ret &= (x[i] == rhs.x[i]);
    }
    return ret;
  }
  bool operator!=(const Extent &rhs) const noexcept {
    return !((*this) == rhs);
  }

  int64_t &operator[](size_t i) noexcept { return x[i]; }
  int64_t const &operator[](size_t i) const noexcept { return x[i]; }
};

template <typename T> class Mat2D {
private:
  void swap(Mat2D &other) noexcept {
    std::swap(data_, other.data_);
    std::swap(rect_, other.rect_);
  }

public:
  std::vector<T> data_;
  Extent<2> rect_;

  Mat2D() {
    rect_[0] = 0;
    rect_[1] = 0;
  }
  Mat2D(int64_t x, int64_t y) : data_(x * y) {
    rect_[0] = x;
    rect_[1] = y;
  }
  Mat2D(int64_t x, int64_t y, const T &v) : data_(x * y, v) {
    rect_[0] = x;
    rect_[1] = y;
  }
  Mat2D(Extent<2> s) : Mat2D(s[0], s[1]) {}
  Mat2D(Extent<2> s, const T &val) : Mat2D(s[0], s[1], val) {}

  Mat2D(const Mat2D &other) = default;
  Mat2D &operator=(const Mat2D &rhs) = default;
  Mat2D(Mat2D &&other) = default;
  Mat2D &operator=(Mat2D &&rhs) = default;

  inline T &at(int64_t i, int64_t j) noexcept {
    assert(i < rect_[1]);
    assert(j < rect_[0]);
    return data_[i * rect_[0] + j];
  }
  inline const T &at(int64_t i, int64_t j) const noexcept {
    assert(i < rect_[1]);
    assert(j < rect_[0]);
    return data_[i * rect_[0] + j];
  }

  /* grow or shrink to [x,y], preserving top-left corner of matrix */
  void resize(int64_t x, int64_t y) {
    Mat2D mat(x, y);

    const int64_t copyRows = std::min(mat.rect_[1], rect_[1]);
    const int64_t copyCols = std::min(mat.rect_[0], rect_[0]);

    for (int64_t i = 0; i < copyRows; ++i) {
      std::memcpy(&mat.at(i, 0), &at(i, 0), copyCols * sizeof(T));
    }
    swap(mat);
  }

  inline const Extent<2> &shape() const noexcept { return rect_; }

  bool operator==(const Mat2D &rhs) const noexcept {
    if (rect_ != rhs.rect_) {
      return false;
    }
    for (uint64_t i = 0; i < rect_[1]; ++i) {
      for (uint64_t j = 0; j < rect_[0]; ++j) {
        if (data_[i * rect_[0] + j] != rhs.data_[i * rect_[0] + j]) {
          return false;
        }
      }
    }
    return true;
  }

  template <typename S> Mat2D &operator/=(const S &s) {
    for (uint64_t i = 0; i < rect_[1]; ++i) {
      for (uint64_t j = 0; j < rect_[0]; ++j) {
        data_[i * rect_[0] + j] /= s;
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
  assert(w.shape()[0] == w.shape()[1]);
  assert(d.shape()[0] == d.shape()[1]);
  assert(w.shape()[0] == f.size()); // one weight per task

  double ret = 0;

  for (size_t a = 0; a < w.shape()[1]; ++a) {
    for (size_t b = 0; b < w.shape()[0]; ++b) {
      double p;
      size_t fa = f[a];
      size_t fb = f[b];
      assert(fa < d.shape()[0] && "task assigned to non-existant agent");
      assert(fb < d.shape()[1] && "task assigned to non-existant agent");
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
  assert(w.shape()[0] == d.shape()[1]);

  std::vector<size_t> f(w.shape()[0]);
  for (size_t i = 0; i < w.shape()[0]; ++i) {
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

  assert(d.shape()[0] == d.shape()[1]);
  assert(w.shape()[0] == w.shape()[1]);

  const int64_t numAgents = d.shape()[0];
  const int64_t numTasks = w.shape()[0];

  std::vector<size_t> f(numTasks, 0);

  RollingStatistics stats;

  auto next_f = [&]() -> bool {
    // if f == [numAgents-1, numAgents-1, ...]
    if (std::all_of(f.begin(), f.end(), [&](size_t u) {
          return u == (numAgents > 0 ? numAgents - 1 : 0);
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
    bool cardCheckFail = false;
    for (int64_t a = 0; a < numAgents; ++a) {
      if (std::count(f.begin(), f.end(), a) > s) {
        cardCheckFail = true;
        break;
      }
    }
    if (cardCheckFail) {
      continue;
    }

    const double cost = detail::cost(w, d, f);
    stats.insert(cost);
    if (bestCost > cost) {
      bestF = f;
      bestCost = cost;
    }
  } while (next_f());

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
  nvtxRangePush("NodeAwareMustEpochMapper::map_must_epoch");
      	log_mapper.debug("%s(): [entry]", __FUNCTION__);

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

  // the closest FB for each GPU
  struct ProcMemPair {
    Processor gpu;
    Memory fb;
  };
  std::vector<ProcMemPair> gpus;
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
      std::cerr << "best " << kv.first << " " << kv.second.m << "\n";
      ProcMemPair pmp;
      pmp.gpu = kv.first;
      pmp.fb = kv.second.m;
      gpus.push_back(pmp);
      ++i;
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
  ap::Mat2D<double> distance(gpus.size(), gpus.size(), 0);
  {
    size_t i = 0;
    for (auto &src : gpus) {
      size_t j = 0;
      for (auto &dst : gpus) {

        std::cerr << "looking for mma " << src.fb << " " << dst.fb << "\n";

        // TODO: self distance is 0
        if (src.gpu == dst.gpu) {
          distance.at(i, j) = 0;
        } else {
          bool found = false;
          for (auto &mma : memMemAffinities) {
            if (mma.m1 == src.fb && mma.m2 == dst.fb) {
              distance.at(i, j) = 1.0 / mma.bandwidth;
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
  for (size_t i = 0; i < distance.shape()[1]; ++i) {
    printf("NodeAwareMustEpochMapper::%s():", __FUNCTION__);
    for (size_t j = 0; j < distance.shape()[0]; ++j) {
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
  for (size_t i = 0; i < weight.shape()[1]; ++i) {
    printf("NodeAwareMustEpochMapper::%s():", __FUNCTION__);
    for (size_t j = 0; j < weight.shape()[0]; ++j) {
      printf(" %6f", weight.at(i, j));
    }
    printf("\n");
  }

  // Max task -> agent assignment
  // if more tasks than agents, distribute tasks
  // if more agents than tasks, max of one task per agent
  // TODO: for a must-epoch task, we should never have more tasks than agents
  const int64_t cardinality = (overlap.size() + gpus.size() - 1) / gpus.size();
  printf("NodeAwareMustEpochMapper::%s(): max cardinality %ld\n", __FUNCTION__,
         cardinality);

  // TODO: for a must-epoch task, we should never have more tasks than agents,
  // so solve_ap only needs to work for distance >= weight
  double cost;
  std::vector<size_t> assignment =
      ap::solve_ap(&cost, weight, distance, cardinality);
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
      procMap[task] = gpus[assignment[gi]].gpu;
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
