// https://github.com/magnatelee/PRK/blob/master/LEGION/Stencil/stencil.cc

#include "legion.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <unistd.h>

// use the Node Aware Must Epoch Mapper
#define USE_NAMEM 1

#include "node_aware_mapper.hpp"

// be noisy
// #define DEBUG_STENCIL_CALC

constexpr int64_t RADIUS = 2;

typedef double DTYPE;

using namespace Legion;

/* Need this special kind of GPU accessor for some reason?
 */
typedef FieldAccessor<READ_ONLY, double, 2, coord_t,
                      Realm::AffineAccessor<double, 2, coord_t>>
    AccessorRO;
typedef FieldAccessor<WRITE_DISCARD, double, 2, coord_t,
                      Realm::AffineAccessor<double, 2, coord_t>>
    AccessorWD;

enum {
  TOP_LEVEL_TASK_ID,
  STENCIL_TASK_ID,
  INIT_TASK_ID,
};

/* stencil iterations alternate 0->1 and 1->0
 */
enum {
  FID_IN,
  FID_OUT,
  FID_GHOST,
};

enum {
  GHOST_LEFT = 0,
  GHOST_RIGHT = 1,
  GHOST_TOP = 2,
  GHOST_BOT = 3,
  PRIVATE = 4,
};

struct SPMDArgs {
public:
  PhaseBarrier notify_ready[2];
  PhaseBarrier notify_empty[2];
  PhaseBarrier wait_ready[2];
  PhaseBarrier wait_empty[2];
  int num_elements;
  int num_subregions;
  int num_steps;
};

__global__ void init_kernel(Rect<2> rect, AccessorWD acc) {

  constexpr double ripple[4] = {0, 0.25, 0, -0.25};
  constexpr size_t period = sizeof(ripple) / sizeof(ripple[0]);

  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  for (int64_t y = rect.lo[1] + ty; y <= rect.hi[1];
       y += gridDim.y * blockDim.y) {
    for (int64_t x = rect.lo[0] + tx; x <= rect.hi[0];
         x += gridDim.x * blockDim.x) {
      double v = x + y + ripple[x % period] + ripple[y % period];
      Point<2> p(x, y);
      acc[p] = v;
    }
  }
}

#define INIT_TASK_CPU 0
void init_task(const Task *task, const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime) {

  std::cerr << "init_task: regions.size()=" << regions.size() << "\n";

  Rect<2> rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  std::cerr << "init_task: rect=" << rect << "\n";
  const FieldID fid = *(task->regions[0].privilege_fields.begin());

#if INIT_TASK_CPU
  constexpr double ripple[4] = {0, 0.25, 0, -0.25};
  constexpr size_t period = sizeof(ripple) / sizeof(ripple[0]);
  const FieldAccessor<WRITE_DISCARD, double, 2> acc(regions[0], fid);
  for (int64_t y = rect.lo[1]; y <= rect.hi[1]; ++y) {
    for (int64_t x = rect.lo[0]; x <= rect.hi[0]; ++x) {
      double v = x + y + ripple[x % period] + ripple[y % period];
      Point<2> p(x, y);
      acc[p] = v;
    }
  }
#else
  const AccessorWD acc(regions[0], fid);
  dim3 dimBlock(32, 32);
  dim3 dimGrid;
  dimGrid.x = ((rect.hi[0] - rect.lo[0] + 1) + dimBlock.x - 1) / dimBlock.x;
  dimGrid.y = ((rect.hi[1] - rect.lo[1] + 1) + dimBlock.y - 1) / dimBlock.y;
  init_kernel<<<dimGrid, dimBlock>>>(rect, acc);
#endif
}

__global__ void stencil_kernel(Rect<2> wrRect, Rect<2> rdRect, AccessorWD wrAcc,
                               AccessorRO rdAcc) {

  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  for (int64_t y = wrRect.lo[1] + ty; y <= wrRect.hi[1];
       y += gridDim.y + blockDim.y) {
    for (int64_t x = wrRect.lo[0] + tx; x <= wrRect.hi[0];
         x += gridDim.x + blockDim.x) {

      if ((x - RADIUS) >= rdRect.lo[0] && (x + RADIUS) <= rdRect.hi[0] &&
          y >= rdRect.lo[1] && y <= rdRect.hi[1]) {
        // first derivative in x
        double v = 0;
        v += 1 * rdAcc[Point<2>(x - 2, y)];
        v += -8 * rdAcc[Point<2>(x - 1, y)];
        v += -1 * rdAcc[Point<2>(x + 2, y)];
        v += 8 * rdAcc[Point<2>(x + 1, y)];
        v /= 12;
        Point<2> p(x, y);
        wrAcc[p] = v;
      }
    }
  }
}

#define STENCIL_TASK_CPU 0
void stencil_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime) {

  std::cerr << "stencil_task: regions.size()=" << regions.size() << "\n";

  FieldID readFid = *(task->regions[0].privilege_fields.begin());
  FieldID writeFid = *(task->regions[1].privilege_fields.begin());

  Rect<2> rdRect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> wrRect = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  std::cerr << "stencil_task: rdRect=" << rdRect << " wrRect=" << wrRect
            << "\n";

#if STENCIL_TASK_CPU
  const FieldAccessor<READ_ONLY, double, 2> rdAcc(regions[0], readFid);
  const FieldAccessor<WRITE_DISCARD, double, 2> wrAcc(regions[1], writeFid);

#define STENCIL_TASK_DUMP_INPUT 0
#if STENCIL_TASK_DUMP_INPUT
  for (int64_t y = rdRect.lo[1]; y <= rdRect.hi[1]; ++y) {
    for (int64_t x = rdRect.lo[0]; x <= rdRect.hi[0]; ++x) {
      Point<2> p(x, y);
      std::cerr << std::setw(6) << rdAcc[p] << " ";
    }
    std::cerr << "\n";
  }
#endif // STENCIL_TASK_DUMP_INPUT

#if 1
  for (int64_t y = wrRect.lo[1]; y <= wrRect.hi[1]; ++y) {
    for (int64_t x = wrRect.lo[0]; x <= wrRect.hi[0]; ++x) {

      if ((x - RADIUS) >= rdRect.lo[0] && (x + RADIUS) <= rdRect.hi[0] &&
          y >= rdRect.lo[1] && y <= rdRect.hi[1]) {
        // first derivative in x
        double v = 0;
        v += 1 * rdAcc[Point<2>(x - 2, y)];
        v += -8 * rdAcc[Point<2>(x - 1, y)];
        v += -1 * rdAcc[Point<2>(x + 2, y)];
        v += 8 * rdAcc[Point<2>(x + 1, y)];
        v /= 12;
        Point<2> p(x, y);
        wrAcc[p] = v;
        std::cerr << p << " " << v << "\n";
      }
    }
  }
#endif

#else
  const AccessorRO rdAcc(regions[0], readFid);
  const AccessorWD wrAcc(regions[1], writeFid);
  dim3 dimBlock(32, 32);
  dim3 dimGrid;
  dimGrid.x = ((wrRect.hi[0] - wrRect.lo[0] + 1) + dimBlock.x - 1) / dimBlock.x;
  dimGrid.y = ((wrRect.hi[1] - wrRect.lo[1] + 1) + dimBlock.y - 1) / dimBlock.y;
  stencil_kernel<<<dimGrid, dimBlock>>>(wrRect, rdRect, wrAcc, rdAcc);

#endif // STENCIL_TASK_CPU
}

LogicalPartition create_halo_partition(Context ctx, LogicalRegion lr,
                                       const Point<2> numElements,
                                       Runtime *runtime) {
  IndexSpace is = lr.get_index_space();
  Rect<2> isRect = runtime->get_index_space_domain(ctx, is);
  std::cerr << "create_halo_partition: " << isRect << "\n";

  Rect<1> colorSpace = Rect<1>(GHOST_LEFT, PRIVATE);
  DomainPointColoring coloring;

  // lo and hi corner of the private area
  Point<2> bbLo, bbHi;
  for (int dim = 0; dim < 2; ++dim) {
    if (0 == isRect.lo[dim]) {
      bbLo[dim] = 0;
    } else {
      bbLo[dim] = isRect.lo[dim] + RADIUS;
    }

    if (numElements[dim] - 1 == isRect.hi[dim]) {
      bbHi[dim] = isRect.hi[dim];
    } else {
      bbHi[dim] = isRect.hi[dim] - RADIUS;
    }
  }

  // set all partitions to be empty
  Point<2> lo(1, 1);
  Point<2> hi(0, 0);
  for (int color = GHOST_LEFT; color <= GHOST_BOT; ++color) {
    coloring[color] = Rect<2>(lo, hi);
  }

  {
    Rect<2> rect(bbLo, bbHi);
    coloring[PRIVATE] = rect;
  }
  if (bbLo[0] > 0) {
    Rect<2> rect(isRect.lo, Point<2>(bbLo[0] - 1, isRect.hi[1]));
    coloring[GHOST_LEFT] = rect;
  }
  if (bbHi[0] < numElements[0] - 1) {
    Rect<2> rect(Point<2>(bbHi[0] + 1, isRect.lo[1]), isRect.hi);
    coloring[GHOST_RIGHT] = rect;
  }
  if (bbLo[1] > 0) {
    Rect<2> rect(isRect.lo, Point<2>(isRect.hi[0], bbLo[1] - 1));
    coloring[GHOST_TOP] = rect;
  }
  if (bbHi[1] < numElements[1] - 1) {
    Rect<2> rect(Point<2>(isRect.lo[0], bbHi[1] + 1), isRect.hi);
    coloring[GHOST_BOT] = rect;
  }
  std::cerr << "  private=" << coloring[PRIVATE] << "\n";
  std::cerr << "  left=" << coloring[GHOST_LEFT] << "\n";
  std::cerr << "  right=" << coloring[GHOST_RIGHT] << "\n";
  std::cerr << "  top=" << coloring[GHOST_TOP] << "\n";
  std::cerr << "  bottom=" << coloring[GHOST_BOT] << "\n";

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  // these ghost regions overlap in the corners, so they are not disjoint
  // https://github.com/magnatelee/PRK/blob/master/LEGION/Stencil/stencil.cc#L614
  // seems to be an error
  IndexPartition ip =
      runtime->create_index_partition(ctx, is, colorSpace, coloring);
#pragma GCC diagnostic pop
  return runtime->get_logical_partition(ctx, lr, ip);
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions, Context ctx,
                    Runtime *runtime) {

  Point<2> numElements(17 /*x*/, 16 /*y*/);
  Point<2> numSubregions(3, 2);
  int numSteps = 1;
  // Check for any command line arguments
  { const InputArgs &command_args = Runtime::get_input_args(); }

  // big top-level index space
  Point<2, size_t> lo(0, 0);
  Point<2, size_t> hi(numElements[0] - 1, numElements[1] - 1);
  Rect<2> elem_rect(lo, hi);
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);
  runtime->attach_name(is, "is");

  // field space
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(DTYPE), FID_IN);
    allocator.allocate_field(sizeof(DTYPE), FID_OUT);
  }

  // top-level logical region
  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

  // create 2D index partition
  Rect<2> colorSpaceRect(Point<2>(0, 0),
                         Point<2>(numSubregions[0] - 1, numSubregions[1] - 1));
  Domain colorSpace(colorSpaceRect);
  DomainPointColoring haloColoring;

  Point<2> tileSz;
  for (int dim = 0; dim < 2; ++dim) {
    tileSz[dim] =
        (numElements[dim] + numSubregions[dim] - 1) / numSubregions[dim];
  }

  for (int y = 0; y < numSubregions[1]; ++y) {
    for (int x = 0; x < numSubregions[0]; ++x) {
      DomainPoint color = DomainPoint(Point<2>(x, y));
      Rect<2> haloRect;
      haloRect.lo[0] = std::max(tileSz[0] * x - RADIUS, 0ll);
      haloRect.lo[1] = std::max(tileSz[1] * y - RADIUS, 0ll);
      haloRect.hi[0] =
          std::min(tileSz[0] * (x + 1) + RADIUS, numElements[0]) - 1;
      haloRect.hi[1] =
          std::min(tileSz[1] * (y + 1) + RADIUS, numElements[1]) - 1;

      std::cerr << color << " halo=" << haloRect << "\n";

      haloColoring[color] = haloRect;
    }
  }

// overlapping partitions of the index space, including the halo regions
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  IndexPartition haloIp =
      runtime->create_index_partition(ctx, is, colorSpace, haloColoring);
#pragma GCC diagnostic pop
  LogicalPartition haloPartition =
      runtime->get_logical_partition(ctx, lr, haloIp);

  /* partition each tile into the private region and the ghost regions
   */
  std::map<Point<2>, LogicalPartition> logicalPartitions;
  std::map<Point<2>, LogicalRegion> tiles;
  for (int y = 0; y < numSubregions[1]; ++y) {
    for (int x = 0; x < numSubregions[0]; ++x) {
      Point<2> partPoint(x, y);
      std::cerr << "get region for " << partPoint << "\n";
      LogicalRegion sr = runtime->get_logical_subregion_by_color(
          ctx, haloPartition, partPoint);
      tiles[partPoint] = sr;
      LogicalPartition lp =
          create_halo_partition(ctx, sr, numElements, runtime);
      logicalPartitions[partPoint] = lp;
    }
  }

  // initialize data
  for (int y = 0; y < numSubregions[1]; ++y) {
    for (int x = 0; x < numSubregions[0]; ++x) {

      Point<2> taskPoint(x, y);
      TaskLauncher init_launcher(INIT_TASK_ID, TaskArgument(NULL, 0));

      // write-discard access to center region in output
      {
        LogicalRegion reg = runtime->get_logical_subregion_by_color(
            ctx, logicalPartitions[taskPoint], PRIVATE);
        // tiles[] is not the parent because that logical region was never
        // created
        RegionRequirement req(reg, WRITE_DISCARD, EXCLUSIVE, lr);
        req.add_field(FID_OUT);
        init_launcher.add_region_requirement(req);
      }

      std::cerr << "init task for " << taskPoint << "\n";
      auto fut = runtime->execute_task(ctx, init_launcher);
      fut.wait();
    }
  }

  for (int n = 0; n < numSteps; ++n) {
    std::cerr << "iteration " << n << "\n";

    std::map<Point<2>, Future> futs;

    // copy FID_OUT to FID_IN
    {
      RegionRequirement dst(lr, WRITE_DISCARD, EXCLUSIVE, lr);
      dst.add_field(FID_IN);
      RegionRequirement src(lr, READ_ONLY, EXCLUSIVE, lr);
      src.add_field(FID_OUT);
      CopyLauncher launcher;
      launcher.add_copy_requirements(src, dst);
      std::cerr << "issue copy\n";
      runtime->issue_copy_operation(ctx, launcher);
    }

    // launch stencil tasks
    for (int y = 0; y < numSubregions[1]; ++y) {
      for (int x = 0; x < numSubregions[0]; ++x) {

        Point<2> taskPoint(x, y);
        TaskLauncher stencil_launcher(STENCIL_TASK_ID, TaskArgument(NULL, 0));

        // read-only access to the whole input region
        {
          RegionRequirement req(tiles[taskPoint], READ_ONLY, SIMULTANEOUS, lr);
          req.add_field(FID_IN);
          stencil_launcher.add_region_requirement(req);
        }

        // write-discard access to center region in output
        {
          LogicalRegion reg = runtime->get_logical_subregion_by_color(
              ctx, logicalPartitions[taskPoint], PRIVATE);
          RegionRequirement req(reg, WRITE_DISCARD, EXCLUSIVE, lr);
          req.add_field(FID_OUT);
          stencil_launcher.add_region_requirement(req);
        }

        std::cerr << "launch task for " << taskPoint << "\n";
        futs[taskPoint] = runtime->execute_task(ctx, stencil_launcher);
      }
    }

    // wait for stencil tasks
    for (auto &kv : futs) {
      kv.second.wait();
    }
  }

  runtime->destroy_index_space(ctx, is);
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INIT_TASK_ID, "init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<init_task>(registrar, "init");
  }

  {
    TaskVariantRegistrar registrar(STENCIL_TASK_ID, "stencil");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<stencil_task>(registrar, "stencil");
  }

#if USE_NAMEM
  Runtime::add_registration_callback(
      NodeAwareMustEpochMapper::mapper_registration);
#endif

  return Runtime::start(argc, argv);
}
