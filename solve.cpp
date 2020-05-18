#include "solve.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>

namespace solve {

double safe_product(double we, double de) {
  if (std::isnan(we) || std::isnan(de)) {
    assert(false && "unexpected NaN");
  }
  if (0 == we && std::isinf(de)) {
    assert(false && "0 weight and infinite distance");
  }
  if (std::isinf(we) && 0 == de) {
    assert(false && "inf weight and 0 distance");
  }
  if (0 == we || 0 == de) {
    return 0;
  } else {
    return we * de;
  }
}

struct JointCost {
  double max;
  double sum;
  bool operator<(const JointCost &rhs) const {
    if (max == rhs.max) {
      return sum < rhs.sum;
    } else {
      return max < rhs.max;
    }
  }
};

double cost(const Mat2D<int64_t> &w,     // weight
            const Mat2D<double> &d,      // distance
            const std::vector<size_t> &f // agent for each task
) {
  assert(w.shape()[0] == w.shape()[1]);
  assert(d.shape()[0] == d.shape()[1]);

  // agent communication matrix
  // needed if there are more tasks than agents to consolidate communication
  // from tasks on the same agent
  Mat2D<int64_t> c(d.shape(), 0);
  for (size_t i = 0; i < w.shape()[1]; ++i) {
    for (size_t j = 0; j < w.shape()[0]; ++j) {
      size_t fi = f[i];
      size_t fj = f[j];
      assert(fi < d.shape()[0] && "task assigned to non-existant agent");
      assert(fj < d.shape()[1] && "task assigned to non-existant agent");
      c.at(fi, fj) += w.at(i, j);
    }
  }

  double ret = 0;
  for (size_t i = 0; i < d.shape()[1]; ++i) {
    for (size_t j = 0; j < d.shape()[0]; ++j) {
      ret += safe_product(d.at(i, j), c.at(i, j));
    }
  }
  return ret;
}

JointCost joint_cost(const Mat2D<int64_t> &w,     // weight
                     const Mat2D<double> &d,      // distance
                     const std::vector<size_t> &f // agent for each task
) {
  assert(w.shape()[0] == w.shape()[1]);
  assert(d.shape()[0] == d.shape()[1]);

  // agent communication matrix
  // needed if there are more tasks than agents to consolidate communication
  // from tasks on the same agent
  Mat2D<int64_t> c(d.shape(), 0);
  for (size_t i = 0; i < w.shape()[1]; ++i) {
    for (size_t j = 0; j < w.shape()[0]; ++j) {
      size_t fi = f[i];
      size_t fj = f[j];
      assert(fi < d.shape()[0] && "task assigned to non-existant agent");
      assert(fj < d.shape()[1] && "task assigned to non-existant agent");
      c.at(fi, fj) += w.at(i, j);
    }
  }

  JointCost ret;
  ret.max = 0;
  ret.sum = 0;
  for (size_t i = 0; i < d.shape()[1]; ++i) {
    for (size_t j = 0; j < d.shape()[0]; ++j) {
      double p = safe_product(d.at(i, j), c.at(i, j));
      ret.max = std::max(ret.max, p);
      ret.sum += p;
    }
  }
  return ret;
}

template <typename C>
std::vector<size_t>
ap_brute_force(C *costp,
               std::function<C(const Mat2D<int64_t> &w, const Mat2D<double> &d,
                               const std::vector<size_t> &f)>
                   costFunc,
               const Mat2D<int64_t> &w, const Mat2D<double> &d) {
  // w and d are square
  assert(d.shape().is_cube());
  assert(w.shape().is_cube());

  const int64_t numAgents = d.shape()[0];
  const int64_t numTasks = w.shape()[0];

  std::vector<size_t> f(numTasks, 0);

  RollingStatistics stats;

  auto is_lb_okay = [&]() -> bool {
    std::vector<int64_t> hist(numAgents, 0);
    for (auto &e : f)
      ++hist[e];
    for (auto &i : hist)
      for (auto &j : hist)
        if (std::abs(i - j) > 1)
          return false;
    return true;
  };

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
  C bestCost;
  bool unset = true;
  do {
    // only compute cost if load-balancing is okay
    if (!is_lb_okay()) {
      continue;
    }

    const double c = cost(w, d, f);
    stats.insert(c);
    if (unset || (bestCost > c)) {
      bestF = f;
      bestCost = c;
      unset = false;
    }
  } while (next_f());

  if (costp) {
    *costp = bestCost;
  }

#if 0
  std::cerr << "Considered " << stats.count()
            << " placements: min=" << stats.min() << " avg=" << stats.mean()
            << " max=" << stats.max() << "\n";
#endif

  return bestF;
}

std::vector<size_t> ap_brute_force(double *costp, const Mat2D<int64_t> &w,
                                   const Mat2D<double> &d) {
  return ap_brute_force<double>(costp, solve::cost, w, d);
}

template <typename C>
std::vector<size_t>
ap_swap2(C *costp,
         std::function<C(const Mat2D<int64_t> &w, const Mat2D<double> &d,
                         const std::vector<size_t> &f)>
             costFunc,
         const Mat2D<int64_t> &w, const Mat2D<double> &d) {
  // w and d are square
  assert(d.shape().is_cube());
  assert(w.shape().is_cube());

  const int64_t numAgents = d.shape()[0];
  const int64_t numTasks = w.shape()[0];

  std::vector<size_t> f(numTasks, 0);

  RollingStatistics stats;

  auto is_lb_okay = [&]() -> bool {
    std::vector<int64_t> hist(numAgents, 0);
    for (auto &e : f)
      ++hist[e];
    for (auto &i : hist)
      for (auto &j : hist)
        if (std::abs(i - j) > 1)
          return false;
    return true;
  };

  // initial round-robin assignment
  for (size_t i = 0; i < numTasks; ++i) {
    f[i] = i % numAgents;
  }
  assert(is_lb_okay()); // initial load balance should be good
  C bestCost = costFunc(w, d, f);

  bool changed = true;
  while (changed) {
    changed = false;

    // check the cost of all possible swaps
    for (size_t i = 0; i < numTasks; ++i) {
      for (size_t j = i + 1; j < numTasks; ++j) {
        std::vector<size_t> swappedF = f; // swapped f
        std::swap(swappedF[i], swappedF[j]);

        double swappedCost = costFunc(w, d, swappedF);
        stats.insert(swappedCost);

        if (swappedCost < bestCost) {
          bestCost = swappedCost;
          f = swappedF;
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

#if 0
  std::cerr << "Considered " << stats.count()
            << " placements: min=" << stats.min() << " avg=" << stats.mean()
            << " max=" << stats.max() << "\n";
#endif

  return f;
}

std::vector<size_t> ap_swap2(double *costp, const Mat2D<int64_t> &w,
                             const Mat2D<double> &d) {
  return ap_swap2<double>(costp, solve::cost, w, d);
}

} // namespace solve