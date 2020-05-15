#include "solve.hpp"

#include <chrono>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <cmath>
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

  Duration min, avg, max;
  int64_t count;
  std::vector<Duration> runs;

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


  }
};

int main(void) {

  int64_t nAgents = 6;
  int64_t nTasks = 9;

  std::cerr << "d\n";
  solve::Mat2D<double> d =
      make_block_diagonal_matrix((nAgents + 1) / 2, nAgents, 0.5, 1);
  std::cerr << "w\n";
  solve::Mat2D<int64_t> w = make_random_symmetric_matrix(nTasks);
  std::cerr << "solve\n";

  double cost;
  std::vector<size_t> f;
  Bencher b(5, 100);
  b.go([&]() { f = solve::ap_brute_force(&cost, w, d); });

  std::cerr << "f: ";
  for (auto &e : f) {
    std::cerr << e << " ";
  }
  std::cerr << "\n";

  std::cerr << b.count << "3 runs " << b.avg.count() << "+-" << b.stddev() << "\n";
}