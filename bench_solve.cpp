#include "solve.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <sstream>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<float> Duration;

/* make an n x n block-diagonal matrix with block size bs, with diagonals of 0,
 * block values of `blockVal`, and off-diagonal values of `offVal`.
 *
 * useful for GPU distances
 */
solve::Mat2D<double> make_block_diagonal_matrix(const size_t bs, const size_t n,
                                                const double blockVal,
                                                const double offVal) {
  solve::Mat2D<double> ret(n, n, offVal);

  // block starting positions
  for (size_t b = 0; b < n; b += bs) {
    // fill in block
    for (size_t i = b; i < b + bs && i < n; ++i) {
      for (size_t j = b; j < b + bs && j < n; ++j) {
        if (i == j) {
          ret.at(i, j) = 0;
        } else {
          ret.at(i, j) = blockVal;
        }
      }
    }
  }

  return ret;
}

/* `nx` * `ny` tasks, comm for stencil `order`
 */
solve::Mat2D<int64_t> make_stencil_weight_matrix(int64_t nx, int64_t ny,
                                                 int64_t bsx, int64_t bsy,
                                                 int64_t order) {
  solve::Mat2D<int64_t> ret(nx * ny, nx * ny);

  for (int i = 0; i < ny; ++i) {
    for (int j = 0; j < nx; ++j) {
      int64_t srcI = i * nx + j;
      // -y nbr
      if (i > 0) {
        int64_t dstI = (i - 1) * nx + j;
        ret.at(srcI, dstI) = bsx * order;
      }
      // +y nbr
      if (i + 1 < ny) {
        int64_t dstI = (i + 1) * nx + j;
        ret.at(srcI, dstI) = bsx * order;
      }
      // -x nbr
      if (j > 0) {
        int64_t dstI = (i)*nx + (j - 1);
        ret.at(srcI, dstI) = bsy * order;
      }
      // +x nbr
      if (j + 1 < nx) {
        int64_t dstI = (i)*nx + (j + 1);
        ret.at(srcI, dstI) = bsy * order;
      }
      // -x/-y nrb
      if (i > 0 && j > 0) {
        int64_t dstI = (i - 1) * nx + (j - 1);
        ret.at(srcI, dstI) = order * order;
      }
      // -x/+y nrb
      if (i + 1 < ny && j > 0) {
        int64_t dstI = (i + 1) * nx + (j - 1);
        ret.at(srcI, dstI) = order * order;
      }
      // +x/-y nrb
      if (i > 0 && j + 1 < nx) {
        int64_t dstI = (i - 1) * nx + (j + 1);
        ret.at(srcI, dstI) = order * order;
      }
      // +x/+y nrb
      if (i + 1 < ny && j + 1 < nx) {
        int64_t dstI = (i + 1) * nx + (j + 1);
        ret.at(srcI, dstI) = order * order;
      }
    }
  }
  return ret;
}

/* n tasks, random communication
 */
solve::Mat2D<int64_t> make_random_symmetric_matrix(size_t n) {
  solve::Mat2D<int64_t> ret(n, n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i; j < n; ++j) {
      int64_t val = rand() % n;
      ret.at(i, j) = val;
      ret.at(j, i) = val;
    }
  }
  return ret;
}

struct Bencher {
  int64_t reps;
  Duration timeout;

  Bencher(int64_t _reps, float timeoutSecs)
      : reps(_reps), timeout(timeoutSecs) {}

  Bencher(float timeoutSecs)
      : reps(std::numeric_limits<int64_t>::max()), timeout(timeoutSecs) {}

  Duration min, avg, max;
  int64_t count;
  std::vector<Duration> runs;
  std::string stopReason;

  double stddev() {
    if (runs.size() < 2) {
      return std::numeric_limits<double>::infinity();
    }

    double acc = 0;
    for (size_t i = 0; i < runs.size(); ++i) {
      acc += std::pow(runs[i].count() - avg.count(), 2);
    }
    return std::sqrt(acc / (runs.size() - 1));
  }

  void go(std::function<void()> f) {
    runs.clear();
    count = 0;
    min = Duration(std::numeric_limits<float>::infinity());
    max = Duration(0);
    Duration time(0);
    for (int64_t i = 0; i < reps && time < timeout; ++i) {

      auto start = Clock::now();
      f();
      Duration elapsed = Clock::now() - start;
      runs.push_back(elapsed);
      count += 1;
      time += elapsed;
      if (elapsed < min)
        min = elapsed;
      if (elapsed > max)
        max = elapsed;
    }
    avg = time / count;

    if (time > timeout) {
      stopReason = "timeout";
    } else {
      stopReason = "finish";
    }
  }
};

int main(void) {

  int64_t nAgents = 4;
  int64_t nTasks = 16;

  solve::Mat2D<double> d =
      make_block_diagonal_matrix((nAgents + 1) / 2, nAgents, 0.5, 1);
  solve::Mat2D<int64_t> w = make_random_symmetric_matrix(nTasks);

  int64_t nx = 4;
  int64_t ny = 4;
  assert(nx * ny == nTasks);
  if (nx * ny != nTasks) {
    std::cerr << "config error\n";
    exit(EXIT_FAILURE);
  }
  w = make_stencil_weight_matrix(nx, ny, 10, 10, 2);

  for (size_t i = 0; i < nTasks; ++i) {
    for (size_t j = 0; j < nTasks; ++j) {
      std::cerr << w.at(i, j) << " ";
    }
    std::cerr << "\n";
  }

  double cost;
  std::vector<size_t> f;
  Bencher b(2.0);

  {
    b.go([&]() { f = solve::ap_max_swap2(&cost, w, d); });

    std::cerr << "ap_max_swap2 cost=" << cost << " f: ";
    for (auto &e : f) {
      std::cerr << e << " ";
    }
    std::cerr << "\n";

    std::cerr << "(" << b.stopReason << ") " << b.count << " runs "
              << std::scientific << b.avg.count() << "+-" << b.stddev()
              << std::defaultfloat << "\n";
  }

  {
    b.go([&]() { f = solve::ap_sum_swap2(&cost, w, d); });

    std::cerr << "ap_sum_swap2 cost=" << cost << " f: ";
    for (auto &e : f) {
      std::cerr << e << " ";
    }
    std::cerr << "\n";

    std::cerr << "(" << b.stopReason << ") " << b.count << " runs "
              << std::scientific << b.avg.count() << "+-" << b.stddev()
              << std::defaultfloat << "\n";
  }

{
  std::array<double, 2> costs;
  b.go([&]() { f = solve::ap_swap2(&costs, w, d); });
  std::cerr << "ap_swap2 cost=" << costs[0] << " " << costs[1] << " f: ";
  for (auto &e : f) {
    std::cerr << e << " ";
  }
  std::cerr << "\n";

  std::cerr << "(" << b.stopReason << ") " << b.count << " runs "
            << std::scientific << b.avg.count() << "+-" << b.stddev()
            << std::defaultfloat << "\n";
}

{
  b.go([&]() { f = solve::ap_sum_brute_force(&cost, w, d); });
  std::cerr << "ap_sum_brute_force cost=" << cost << " f: ";
  for (auto &e : f) {
    std::cerr << e << " ";
  }
  std::cerr << "\n";

  std::cerr << "(" << b.stopReason << ") " << b.count << " runs "
            << std::scientific << b.avg.count() << "+-" << b.stddev()
            << std::defaultfloat << "\n";
}

{
  b.go([&]() { f = solve::ap_max_brute_force(&cost, w, d); });
  std::cerr << "ap_max_brute_force cost=" << cost << " f: ";
  for (auto &e : f) {
    std::cerr << e << " ";
  }
  std::cerr << "\n";

  std::cerr << "(" << b.stopReason << ") " << b.count << " runs "
            << std::scientific << b.avg.count() << "+-" << b.stddev()
            << std::defaultfloat << "\n";
}

{
  std::array<double, 2> costs;
  b.go([&]() { f = solve::ap_brute_force(&costs, w, d); });
  std::cerr << "ap_brute_force cost=" << costs[0] << " " << costs[1] << " f: ";
  for (auto &e : f) {
    std::cerr << e << " ";
  }
  std::cerr << "\n";

  std::cerr << "(" << b.stopReason << ") " << b.count << " runs "
            << std::scientific << b.avg.count() << "+-" << b.stddev()
            << std::defaultfloat << "\n";
}
}