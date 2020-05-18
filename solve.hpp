#pragma once

#include <vector>
#include <limits>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <iostream>

namespace solve {

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

  bool is_cube() const noexcept {
    for (unsigned i = 1; i < N; ++i) {
      if (x[i] != x[0]) {
        return false;
      }
    }
    return true;
  }
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
    if (i >= rect_[1]) {
        std::cerr << ".at(" << i << ", ...) in matrix" << rect_[0] << "x" << rect_[1] << "\n";
        assert(i < rect_[1]);
    }
    
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


/* brute-force solution to assignment problem

   an `n` x `n` matrix `w` describing inter-task communication volumes.
   a `p` x `p` matrix `d` describing inter-agent communication distances.
   objective, minimize total flow * distance between agents, where the flow is
   the sum communication for each task

   return empty vector if no valid assignment was found

   load-balancing requires that the difference in assigned tasks between
   any two GPUs is 1.
 */
std::vector<size_t> ap_brute_force(double *costp, const Mat2D<int64_t> &w,
                             const Mat2D<double> &d);



/* cost of assigning task i to agent f[i], where `w` describes the inter-task communication
and `d` the inter-agent distance
*/
double cost(const Mat2D<int64_t> &w,      // weight
                   const Mat2D<double> &d,      // distance
                   const std::vector<size_t> &f // agent for each task
);


/* greedy swap solution to assignment problem

   an `n` x `n` matrix `w` describing inter-task communication volumes.
   a `p` x `p` matrix `d` describing inter-agent communication distances.
   objective, minimize total flow * distance between agents, where the flow is
   the sum communication for each task

   return empty vector if no valid assignment was found

   load-balancing requires that the difference in assigned tasks between
   any two GPUs is 1.

   We iterate until the cost function cannot be improved
   each iteration, we check all possible assignment swaps and pick the first that improves
   the cost
 */
std::vector<size_t> ap_swap2(double *costp, const Mat2D<int64_t> &w,
                             const Mat2D<double> &d);


}



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