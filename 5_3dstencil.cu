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
typedef FieldAccessor<READ_ONLY, double, 3, coord_t,
                      Realm::AffineAccessor<double, 3, coord_t>>
    AccessorRO;
typedef FieldAccessor<WRITE_DISCARD, double, 3, coord_t,
                      Realm::AffineAccessor<double, 3, coord_t>>
    AccessorWD;

enum {
  TOP_LEVEL_TASK_ID,
  INIT_TASK_ID,
  STENCIL_TASK_ID,
  CHECK_TASK_ID,
};

/* stencil iterations alternate 0->1 and 1->0
   each iteration reads from one field and writes to the other
 */
enum {
  FID_0,
  FID_1,
};

__global__ void init_kernel(Rect<3> rect, AccessorWD acc) {

  constexpr double ripple[4] = {0, 0.25, 0, -0.25};
  constexpr size_t period = sizeof(ripple) / sizeof(ripple[0]);

  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const int tz = blockIdx.z * blockDim.z + threadIdx.z;

  for (int64_t z = rect.lo[2] + tz; z <= rect.hi[2];
       z += gridDim.z * blockDim.z) {
    for (int64_t y = rect.lo[1] + ty; y <= rect.hi[1];
         y += gridDim.y * blockDim.y) {
      for (int64_t x = rect.lo[0] + tx; x <= rect.hi[0];
           x += gridDim.x * blockDim.x) {
        double v = z + y + x + ripple[x % period] + ripple[y % period];
        Point<3> p(x, y, z);
        acc[p] = v;
      }
    }
  }
}

#define INIT_TASK_CPU 0
void init_task(const Task *task, const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime) {

  std::cerr << "init_task: regions.size()=" << regions.size() << "\n";

  Rect<3> rect = runtime->get_index_space_domain(
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
  dimGrid.z = ((rect.hi[2] - rect.lo[2] + 1) + dimBlock.z - 1) / dimBlock.z;
  init_kernel<<<dimGrid, dimBlock>>>(rect, acc);
#endif
}

__global__ void stencil_kernel(Rect<3> wrRect, Rect<3> rdRect, AccessorWD wrAcc,
                               AccessorRO rdAcc) {

  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const int tz = blockIdx.z * blockDim.z + threadIdx.z;

  for (int64_t z = wrRect.lo[1] + tz; z <= wrRect.hi[1];
       z += gridDim.z + blockDim.z) {
    for (int64_t y = wrRect.lo[1] + ty; y <= wrRect.hi[1];
         y += gridDim.y + blockDim.y) {
      for (int64_t x = wrRect.lo[0] + tx; x <= wrRect.hi[0];
           x += gridDim.x + blockDim.x) {

        if ((x - RADIUS) >= rdRect.lo[0] && (x + RADIUS) <= rdRect.hi[0] &&
            y >= rdRect.lo[1] && y <= rdRect.hi[1]) {
          // first derivative in x
          double v = 0;
          v += 1 * rdAcc[Point<3>(x - 2, y, z)];
          v += -8 * rdAcc[Point<3>(x - 1, y, z)];
          v += -1 * rdAcc[Point<3>(x + 2, y, z)];
          v += 8 * rdAcc[Point<3>(x + 1, y, z)];
          v /= 12;
          Point<3> p(x, y, z);
          wrAcc[p] = v;
        }
      }
    }
  }
}

void stencil_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime) {

  std::cerr << "stencil_task: regions.size()=" << regions.size() << "\n";

  FieldID readFid = *(task->regions[0].privilege_fields.begin());
  FieldID writeFid = *(task->regions[1].privilege_fields.begin());

  Rect<3> rdRect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<3> wrRect = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  std::cerr << "stencil_task: rdRect=" << rdRect << " wrRect=" << wrRect
            << "\n";

  const AccessorRO rdAcc(regions[0], readFid);
  const AccessorWD wrAcc(regions[1], writeFid);
  dim3 dimBlock(32, 32);
  dim3 dimGrid;
  dimGrid.x = ((wrRect.hi[0] - wrRect.lo[0] + 1) + dimBlock.x - 1) / dimBlock.x;
  dimGrid.y = ((wrRect.hi[1] - wrRect.lo[1] + 1) + dimBlock.y - 1) / dimBlock.y;
  dimGrid.z = ((wrRect.hi[2] - wrRect.lo[2] + 1) + dimBlock.z - 1) / dimBlock.z;
  stencil_kernel<<<dimGrid, dimBlock>>>(wrRect, rdRect, wrAcc, rdAcc);
}

__global__ void check_kernel(int *errors, Rect<3> rect, AccessorRO acc) {

  constexpr double ripple[4] = {0, 1, 0, -1};
  constexpr size_t period = sizeof(ripple) / sizeof(ripple[0]);

  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const int tz = blockIdx.z * blockDim.z + threadIdx.z;

  for (int64_t z = rect.lo[2] + tz; z <= rect.hi[2];
       z += gridDim.z * blockDim.z) {
    for (int64_t y = rect.lo[1] + ty; y <= rect.hi[1];
         y += gridDim.y * blockDim.y) {
      for (int64_t x = rect.lo[0] + tx; x <= rect.hi[0];
           x += gridDim.x * blockDim.x) {

        double e = 1 + ripple[x % period];
        Point<3> p(x, y, z);
        double v = acc[p];

        if (std::abs(e / v) < 1.01 && std::abs(e / v) > 0.99) {
          atomicAdd(errors, 1);
        }
      }
    }
  }
}

void check_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime) {

  std::cerr << "check_task: regions.size()=" << regions.size() << "\n";

  Rect<3> rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  std::cerr << "check_task: rect=" << rect << "\n";
  const FieldID fid = *(task->regions[0].privilege_fields.begin());
  const AccessorRO acc(regions[0], fid);

  int *dErrors, hErrors = 1;
  cudaMalloc(&dErrors, sizeof(int));
  cudaMemset(dErrors, 0, sizeof(int));

  dim3 dimBlock(32, 32);
  dim3 dimGrid;
  dimGrid.x = ((rect.hi[0] - rect.lo[0] + 1) + dimBlock.x - 1) / dimBlock.x;
  dimGrid.y = ((rect.hi[1] - rect.lo[1] + 1) + dimBlock.y - 1) / dimBlock.y;
  dimGrid.z = ((rect.hi[2] - rect.lo[2] + 1) + dimBlock.z - 1) / dimBlock.z;
  check_kernel<<<dimGrid, dimBlock>>>(dErrors, rect, acc);

  cudaMemcpy(&hErrors, dErrors, sizeof(int), cudaMemcpyDeviceToHost);

  if (hErrors) {
    std::cerr << "ERROR:\n";
  }

  cudaFree(dErrors);
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions, Context ctx,
                    Runtime *runtime) {

  Point<3> numElements(6 /*x*/, 5 /*y*/, 4);
  Point<3> numSubregions(2, 2, 2);
  int numSteps = 3;
  // Check for any command line arguments

  const InputArgs &command_args = Runtime::get_input_args();
  for (int i = 1; i < command_args.argc; i++) {
    if (!strcmp(command_args.argv[i], "-nx"))
      numElements[0] = atoi(command_args.argv[++i]);
    if (!strcmp(command_args.argv[i], "-ny"))
      numElements[1] = atoi(command_args.argv[++i]);
    if (!strcmp(command_args.argv[i], "-nz"))
      numElements[2] = atoi(command_args.argv[++i]);
    if (!strcmp(command_args.argv[i], "-bx"))
      numSubregions[0] = atoi(command_args.argv[++i]);
    if (!strcmp(command_args.argv[i], "-by"))
      numSubregions[1] = atoi(command_args.argv[++i]);
    if (!strcmp(command_args.argv[i], "-bz"))
      numSubregions[2] = atoi(command_args.argv[++i]);
    if (!strcmp(command_args.argv[i], "-s"))
      numSteps = atoi(command_args.argv[++i]);
  }

  // Compute domain index space
  Point<3> lo(0, 0, 0);
  Point<3> hi(numElements[0] - 1, numElements[1] - 1, numElements[2] - 1);
  Rect<3> elem_rect(lo, hi);
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);
  std::cerr << is << "\n";
  runtime->attach_name(is, "is");

  // Compute domain field space
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(DTYPE), FID_0);
    allocator.allocate_field(sizeof(DTYPE), FID_1);
  }

  // compute domain logical region
  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);
  std::cerr << lr << "\n";

  std::cerr << lr.get_index_space() << "\n";
  Rect<3> rect = runtime->get_index_space_domain(ctx, lr.get_index_space());
  std::cerr << rect << "\n";

  // launch index space
  Rect<3> launchRect(Point<3>(0, 0, 0),
                     Point<3>(numSubregions[0] - 1, numSubregions[1] - 1,
                              numSubregions[2] - 1));
  IndexSpace launchSpace = runtime->create_index_space(ctx, launchRect);

  // empty argument map
  ArgumentMap argMap;

  // create the partition for the index space
  // tiles are overlapping and include the ghost regions
  // disjoints are non-overlapping
  Domain colorSpace(launchRect);
  DomainPointColoring tileColoring;
  DomainPointColoring disjointColoring;
  {
    Point<3> tileSz;
    for (int dim = 0; dim < 3; ++dim) {
      tileSz[dim] =
          (numElements[dim] + numSubregions[dim] - 1) / numSubregions[dim];
    }

    for (int z = 0; z < numSubregions[2]; ++z) {
      for (int y = 0; y < numSubregions[1]; ++y) {
        for (int x = 0; x < numSubregions[0]; ++x) {
          DomainPoint color = DomainPoint(Point<3>(x, y, z));

          {
            Rect<3> tileRect;
            tileRect.lo[0] = std::max(tileSz[0] * x - RADIUS, 0ll);
            tileRect.lo[1] = std::max(tileSz[1] * y - RADIUS, 0ll);
            tileRect.lo[2] = std::max(tileSz[2] * z - RADIUS, 0ll);
            tileRect.hi[0] =
                std::min(tileSz[0] * (x + 1) + RADIUS, numElements[0]) - 1;
            tileRect.hi[1] =
                std::min(tileSz[1] * (y + 1) + RADIUS, numElements[1]) - 1;
            tileRect.hi[2] =
                std::min(tileSz[2] * (z + 1) + RADIUS, numElements[2]) - 1;
            std::cerr << "color=" << color << " tile=" << tileRect << "\n";
            tileColoring[color] = tileRect;
          }

          {
            Rect<3> disjointRect;
            disjointRect.lo[0] = tileSz[0] * x;
            disjointRect.lo[1] = tileSz[1] * y;
            disjointRect.lo[2] = tileSz[2] * z;
            disjointRect.hi[0] =
                std::min(tileSz[0] * (x + 1), numElements[0]) - 1;
            disjointRect.hi[1] =
                std::min(tileSz[1] * (y + 1), numElements[1]) - 1;
            disjointRect.hi[2] =
                std::min(tileSz[2] * (z + 1), numElements[2]) - 1;
            std::cerr << color << " disjoint=" << disjointRect << "\n";
            disjointColoring[color] = disjointRect;
          }
        }
      }
    }
  }

// overlapping partitions of the index space, including the halo regions
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  IndexPartition tileIp =
      runtime->create_index_partition(ctx, is, colorSpace, tileColoring);
#pragma GCC diagnostic pop
  LogicalPartition tilePartition =
      runtime->get_logical_partition(ctx, lr, tileIp);

// disjoint partitions
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  IndexPartition disjointIp =
      runtime->create_index_partition(ctx, is, colorSpace, disjointColoring);
#pragma GCC diagnostic pop
  LogicalPartition disjointPartition =
      runtime->get_logical_partition(ctx, lr, disjointIp);

  // record input and output fields
  int inputField = FID_0;
  int outputField = FID_1;

  {
    IndexTaskLauncher init_launcher(INIT_TASK_ID, launchSpace,
                                    TaskArgument(NULL, 0), ArgumentMap());
    // write-discard access to center region in output
    {
      RegionRequirement req(disjointPartition, 0 /*identity projection?*/,
                            WRITE_DISCARD, EXCLUSIVE, lr);
      req.add_field(outputField);
      init_launcher.add_region_requirement(req);
    }

    std::cerr << "execute index space: init\n";
    FutureMap futs = runtime->execute_index_space(ctx, init_launcher);
    futs.wait_all_results();
  }

  for (int n = 0; n < numSteps; ++n) {
    std::cerr << "iteration " << n << "\n";

    std::map<Point<3>, Future> futs;

    // swap input field and output field
    std::swap(inputField, outputField);

    // launch stencil tasks
    {
      IndexTaskLauncher stencil_launcher(STENCIL_TASK_ID, launchSpace,
                                         TaskArgument(NULL, 0), ArgumentMap());

      // read-only access to the whole input region
      {
        RegionRequirement req(tilePartition, 0 /* identity projection?*/,
                              READ_ONLY, SIMULTANEOUS, lr);
        req.add_field(inputField);
        stencil_launcher.add_region_requirement(req);
      }

      // write-discard access to center region in output
      {
        RegionRequirement req(disjointPartition, 0 /*identity projection?*/,
                              WRITE_DISCARD, EXCLUSIVE, lr);
        req.add_field(outputField);
        stencil_launcher.add_region_requirement(req);
      }

      std::cerr << "execute_index_space: stencil\n";
      FutureMap futs = runtime->execute_index_space(ctx, stencil_launcher);
      futs.wait_all_results();
    }

    // check first derivative for correctness
    if (0 == n) {
      IndexTaskLauncher check_launcher(CHECK_TASK_ID, launchSpace,
                                       TaskArgument(NULL, 0), ArgumentMap());
      // write-discard access to center region in output
      {
        RegionRequirement req(disjointPartition, 0 /*identity projection?*/,
                              READ_ONLY, EXCLUSIVE, lr);
        req.add_field(outputField);
        check_launcher.add_region_requirement(req);
      }

      std::cerr << "execute index space: check\n";
      FutureMap futs = runtime->execute_index_space(ctx, check_launcher);
      futs.wait_all_results();
    }
  }

  runtime->destroy_logical_region(ctx, lr);
  runtime->destroy_index_space(ctx, is);
  runtime->destroy_field_space(ctx, fs);
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

  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

#if USE_NAMEM
  Runtime::add_registration_callback(
      NodeAwareMustEpochMapper::mapper_registration);
#endif

  return Runtime::start(argc, argv);
}
