#include "solve.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>

template <typename V> std::ostream &dump_vec(std::ostream &os, const V &v) {
  for (auto &e : v) {
    os << e << " ";
  }
  return os;
}

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

double sum_cost(const Mat2D<int64_t> &w,     // weight
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

double max_cost(const Mat2D<int64_t> &w,     // weight
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

  double ret = -1 * std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < d.shape()[1]; ++i) {
    for (size_t j = 0; j < d.shape()[0]; ++j) {
      ret = std::max(ret, safe_product(d.at(i, j), c.at(i, j)));
    }
  }
  return ret;
}

/* brute-force solution to assignment problem

  lexicographically ordered optimization of cost functions: minimize
  costFuncs[0], then costFuncs[1] for the minimum costFuncs[0], etc

*/
std::vector<size_t> ap_brute_force(std::vector<double> *costs,
                                   const std::vector<CostFunction> &costFuncs,
                                   const Mat2D<int64_t> &w,
                                   const Mat2D<double> &d) {
  // w and d are square
  assert(d.shape().is_cube());
  assert(w.shape().is_cube());

  const int64_t numAgents = d.shape()[0];
  const int64_t numTasks = w.shape()[0];

  std::vector<size_t> f(numTasks, 0);
  std::vector<RollingStatistics> stats(costFuncs.size());

  auto is_load_balanced = [&]() -> bool {
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
  std::vector<double> bestCost(costFuncs.size());
  bool unset = true;
  do {
    if (!is_load_balanced()) {
      continue;
    }

    std::vector<double> c(costFuncs.size(),
                          std::numeric_limits<double>::infinity());
    for (size_t i = 0; i < costFuncs.size(); ++i) {
      c[i] = costFuncs[i](w, d, f);
      stats[i].insert(c[i]);
    }

    if (unset || (c < bestCost)) {
      bestF = f;
      bestCost = c;
      unset = false;
    }

  } while (next_f());

  if (costs) {
    *costs = bestCost;
  }

  return bestF;
}

std::vector<size_t> ap_max_brute_force(double *costp, const Mat2D<int64_t> &w,
                                       const Mat2D<double> &d) {

  std::vector<CostFunction> funcs = {max_cost};
  std::vector<double> costs;
  std::vector<size_t> f = ap_brute_force(&costs, funcs, w, d);
  assert(costs.size() == 1);
  if (costp) {
    *costp = costs[0];
  }
  return f;
}

std::vector<size_t> ap_sum_brute_force(double *costp, const Mat2D<int64_t> &w,
                                       const Mat2D<double> &d) {

  std::vector<CostFunction> funcs = {sum_cost};
  std::vector<double> costs;
  std::vector<size_t> f = ap_brute_force(&costs, funcs, w, d);
  assert(costs.size() == 1);
  if (costp) {
    *costp = costs[0];
  }
  return f;
}

std::vector<size_t> ap_brute_force(std::array<double, 2> *costs,
                                   const Mat2D<int64_t> &w,
                                   const Mat2D<double> &d) {

  std::vector<CostFunction> funcs = {max_cost, sum_cost};
  std::vector<double> costv;
  std::vector<size_t> f = ap_brute_force(&costv, funcs, w, d);
  assert(costv.size() == 2);
  if (costs) {
    (*costs)[0] = costv[0];
    (*costs)[1] = costv[1];
  }
  return f;
}

std::vector<size_t> ap_lex_swap2(std::vector<double> *costs,
                                 const std::vector<CostFunction> &costFuncs,
                                 const Mat2D<int64_t> &w,
                                 const Mat2D<double> &d) {
  // w and d are square
  assert(d.shape().is_cube());
  assert(w.shape().is_cube());

  const int64_t numAgents = d.shape()[0];
  const int64_t numTasks = w.shape()[0];

  std::vector<size_t> f(numTasks, 0);

  std::vector<RollingStatistics> stats(costFuncs.size());

  // initial round-robin assignment
  for (size_t i = 0; i < numTasks; ++i) {
    f[i] = i % numAgents;
  }

  std::vector<double> bestCost(costFuncs.size());
  for (size_t i = 0; i < bestCost.size(); ++i) {
    bestCost[i] = costFuncs[i](w, d, f);
  }

  auto minimize_cost_func = [&](const size_t fi) {
    assert(fi < costFuncs.size());
    // std::cerr << "minimze " << fi << "\n";

    bool changed = true;
    while (changed) {
      changed = false;

      double costI;

      // search for a swap that reduces costFunc[fi]
      for (size_t i = 0; i < numTasks; ++i) {
        for (size_t j = i + 1; j < numTasks; ++j) {
          std::vector<size_t> swappedF = f; // swapped f
          std::swap(swappedF[i], swappedF[j]);

          costI = costFuncs[fi](w, d, swappedF);

          // if it improves or matches this specific cost function
          if (costI <= bestCost[fi]) {

            // if it improves the overall cost
            std::vector<double> swappedCost(costFuncs.size());
            for (size_t k = 0; k < costFuncs.size(); ++k) {
              swappedCost[k] = costFuncs[k](w, d, swappedF);
            }
            if (swappedCost < bestCost) {
              // dump_vec(std::cerr << "f: ", swappedF);
              // dump_vec(std::cerr << ": ", swappedCost);
              // std::cerr << "< ";
              // dump_vec(std::cerr, bestCost);
              // std::cerr << "\n";
              bestCost = swappedCost;
              f = swappedF;
              changed = true;
              goto body_end; // fast exit
            }
          }
        }
      }
    body_end:;
    }
  };

  for (size_t fi = 0; fi < costFuncs.size(); ++fi) {
    minimize_cost_func(fi);
  }

  if (costs) {
    *costs = bestCost;
  }

#if 0
  std::cerr << "Considered " << stats.count()
            << " placements: min=" << stats.min() << " avg=" << stats.mean()
            << " max=" << stats.max() << "\n";
#endif

  return f;
}

std::vector<size_t> ap_max_swap2(double *costp, const Mat2D<int64_t> &w,
                                 const Mat2D<double> &d) {
  std::vector<double> costVec;
  std::vector<size_t> f = ap_lex_swap2(&costVec, {max_cost}, w, d);
  assert(costVec.size() == 1);
  if (costp) {
    *costp = costVec[0];
  }
  return f;
}

std::vector<size_t> ap_sum_swap2(double *costp, const Mat2D<int64_t> &w,
                                 const Mat2D<double> &d) {
  std::vector<double> costVec;
  std::vector<size_t> f = ap_lex_swap2(&costVec, {sum_cost}, w, d);
  assert(costVec.size() == 1);
  if (costp) {
    *costp = costVec[0];
  }
  return f;
}

std::vector<size_t> ap_swap2(std::array<double, 2> *costs,
                             const Mat2D<int64_t> &w, const Mat2D<double> &d) {

  std::vector<CostFunction> funcs = {max_cost, sum_cost};
  std::vector<double> costv;
  std::vector<size_t> f = ap_lex_swap2(&costv, funcs, w, d);
  assert(costv.size() == 2);
  if (costs) {
    (*costs)[0] = costv[0];
    (*costs)[1] = costv[1];
  }
  return f;
}

} // namespace solve
