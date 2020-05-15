#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// use z3 to optimize the assignment? its an integer programming thing?
#include "z3++.h"
#include "z3.h"

// https://stackoverflow.com/questions/15599030/z3-performing-matrix-operations
// https://rise4fun.com/Z3/tutorial/optimization

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<float> Duration;

/* make an n x n block-diagonal matrix with block size bs, with diagonals of 0,
 * block values of `blockVal`, and off-diagonal values of `offVal`.
 */
std::vector<double> make_block_diagonal_matrix(const size_t bs, const size_t n,
                                                const double blockVal,
                                                const double offVal) {
  std::vector<double> ret(n * n, offVal);

  // block starting positions
  for (size_t b = 0; b < n; b += bs) {
    // fill in block
    for (size_t i = b; i < b + bs && i < n; ++i) {
      for (size_t j = b; j < b + bs && j < n; ++j) {
        if (i == j) {
          ret[i * n + j] = 0;
        } else {
          ret[i * n + j] = blockVal;
        }
      }
    }
  }

  return ret;
}

/* `nx` * `ny` tasks, comm for stencil `order`
 */
std::vector<int64_t> make_stencil_weight_matrix(int64_t nx, int64_t ny,
                                                int64_t bsx, int64_t bsy,
                                                int64_t order) {
  std::vector<int64_t> ret(nx * ny * nx * ny);

  for (int i = 0; i < ny; ++i) {
    for (int j = 0; j < nx; ++j) {
      int64_t srcI = i * nx + j;
      // -y nbr
      if (i > 0) {
        int64_t dstI = (i - 1) * nx + j;
        ret[srcI * ny * nx + dstI] = bsx * order;
      }
      // +y nbr
      if (i + 1 < ny) {
        int64_t dstI = (i + 1) * nx + j;
        ret[srcI * ny * nx + dstI] = bsx * order;
      }
      // -x nbr
      if (j > 0) {
        int64_t dstI = (i)*nx + (j - 1);
        ret[srcI * ny * nx + dstI] = bsy * order;
      }
      // +x nbr
      if (j + 1 < nx) {
        int64_t dstI = (i)*nx + (j + 1);
        ret[srcI * ny * nx + dstI] = bsy * order;
      }
      // -x/-y nrb
      if (i > 0 && j > 0) {
        int64_t dstI = (i - 1) * nx + (j - 1);
        ret[srcI * ny * nx + dstI] = order * order;
      }
      // -x/+y nrb
      if (i + 1 < ny && j > 0) {
        int64_t dstI = (i + 1) * nx + (j - 1);
        ret[srcI * ny * nx + dstI] = order * order;
      }
      // +x/-y nrb
      if (i > 0 && j + 1 < nx) {
        int64_t dstI = (i - 1) * nx + (j + 1);
        ret[srcI * ny * nx + dstI] = order * order;
      }
      // +x/+y nrb
      if (i + 1 < ny && j + 1 < nx) {
        int64_t dstI = (i + 1) * nx + (j + 1);
        ret[srcI * ny * nx + dstI] = order * order;
      }
    }
  }
  return ret;
}

/* n tasks, random communication
 */
std::vector<int64_t> make_random_symmetric_matrix(size_t n) {
  std::vector<int64_t> ret(n * n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i; j < n; ++j) {
      int64_t val = rand() % n;
      ret[i * n + j] = val;
      ret[j * n + i] = val;
    }
  }
  return ret;
}

std::vector<size_t> solve(const std::vector<int64_t> &_w,
                          const std::vector<double> &_d, int timeout = 1000) {

  assert(_w.size() == _d.size());
  const size_t n = std::sqrt(_w.size());
  assert(n * n == _w.size() && "expected w and d to be square");

  auto wallStart = Clock::now();

  z3::context ctx;
  ctx.set("timeout", timeout);

  // assignment to be solved for
  z3::expr_vector f(ctx);
  for (size_t i = 0; i < n; ++i) {
    std::stringstream name;
    name << "f_" << i;
    f.push_back(ctx.int_const(name.str().c_str()));
  }

  // convert weight matrix to z3 variables
  z3::expr_vector w(ctx);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      // std::stringstream valStr;
      // valStr << _w[i * n + j];
      w.push_back(ctx.int_val(_w[i * n + j]));
    }
  }

  // convert distance matrix to z3 array, since we index into it through f
  z3::expr d = z3::const_array(ctx.int_sort(), ctx.real_val(0));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      // std::stringstream valStr;
      // valStr << _d[i * n + j];
      d = z3::store(d, i * n + j, _d[i * n + j]);
    }
  }

  z3::optimize opt(ctx);

  // All agents are numbered 0..n, so restrict space of 0 <= f[i] < n
  for (size_t i = 0; i < n; ++i) {
    opt.add(f[i] >= 0);
    opt.add(f[i] < int(n));
  }

  // each agent should appear once in f
  opt.add(distinct(f));

  // partial products for cost function
  z3::expr_vector partials(ctx);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      z3::expr wij = w[i * n + j];
      z3::expr dij = z3::select(d, f[i] * int(n) + f[j]);
      partials.push_back(wij * dij);
    }
  }

  // maximum communication cost
  z3::expr maxCost = partials[0];
  for (size_t i = 1; i < n * n; ++i) {
    maxCost = z3::max(maxCost, partials[i]);
  }

  // sum of communication costs
  z3::expr sumCost = z3::sum(partials);

  opt.minimize(maxCost);
  // opt.minimize(sumCost);

  Duration setupElapsed = Clock::now() - wallStart;


  z3::check_result res;
    auto start = Clock::now();
  try {
    {
      res = opt.check();
    }
  } catch (const z3::exception &e) {
    std::cerr << "EXCEPTION: " << e.what() << "\n";
    res = z3::unsat;
  }
  Duration dur = Clock::now() - start;

  if (z3::sat == res) {
    std::cout << "(finish)  ";
  } else {
    std::cout << "(timeout) ";
  }

  std::cout << "setup=" << setupElapsed.count() << " opt=" << dur.count();

  z3::model m = opt.get_model();

  std::vector<size_t> retF;

  if (!m.eval(maxCost).is_numeral()) {
    std::cout << " max=-- sum=--" << "\n";
    return retF;
  }

  {
    int64_t num, den;
    if (Z3_get_numeral_rational_int64(ctx, m.eval(maxCost), &num, &den)) {
      std::cout << " max=" << double(num) / den;
    } else {
      std::cout << " max=" << m.eval(maxCost);
    }
  }

  {
    int64_t num, den;
    if (Z3_get_numeral_rational_int64(ctx, m.eval(sumCost), &num, &den)) {
      std::cout << " sum=" << double(num) / den << "\n";
    } else {
      std::cout << " sum=" << m.eval(sumCost) << "\n";
    }
  }

  for (size_t i = 0; i < n; ++i) {
    int64_t fi;
    if (Z3_get_numeral_int64(ctx, m.eval(f[i]), &fi)) {
      retF.push_back(fi);
    }
  }

  return retF;
}

int main(void) {

  int numTasksX = 3;
  int numTasksY = 3;
  const int numTasks = numTasksX * numTasksY;
  const int numAgents = numTasks;
  int bsx = 20;
  int bsy = 10;
  int order = 2;

  // weight matrix
  std::vector<int64_t> w = make_random_symmetric_matrix(numTasks);
  w = make_stencil_weight_matrix(numTasksX, numTasksY, bsx, bsy, order);

  std::cout << "weight:\n";
  for (int i = 0; i < numTasks; ++i) {
    for (int j = 0; j < numTasks; ++j) {
      std::cout << std::setw(4) << w[i * numTasks + j] << " ";
    }
    std::cout << "\n";
  }

  // distance matrix
  std::vector<double> d = make_block_diagonal_matrix((numAgents + 1)/2, numAgents, 0.6, 1.0);

  // std::cout << "dist:\n";
  // for (int i = 0; i < numAgents; ++i) {
  //   for (int j = 0; j < numAgents; ++j) {
  //     std::cout << std::setw(4) << d[i * numAgents + j] << " ";
  //   }
  //   std::cout << "\n";
  // }


  for (auto timeout : {16, 32, 64, 128, 256, 500, 1000, 2000, 4096, 8192, 16384, 32768, 65536}) {
    std::vector<size_t> f = solve(w, d, timeout);
  }

  // std::cout << "assignment:\n";
  // for (int i = 0; i < numTasksY; ++i) {
  //   for (int j = 0; j < numTasksX; ++j) {
  //     std::cout << f[i * numTasksX + j] << " ";
  //   }
  //   std::cout << "\n";
  // }


  // https://stackoverflow.com/questions/23064533/statistics-in-z3, led to
  // https://stackoverflow.com/questions/18491922/interpretation-of-z3-statistics
  // (good summary)
  // https://stackoverflow.com/questions/6841193/which-statistics-indicate-an-efficient-run-of-z3
  //   std::cerr << opt.statistics();
}