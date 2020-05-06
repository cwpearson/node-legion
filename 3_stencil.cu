// https://github.com/magnatelee/PRK/blob/master/LEGION/Stencil/stencil.cc

#include "legion.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

// use the Node Aware Must Epoch Mapper
// #define USE_NAMEM

#include "node_aware_must_epoch_mapper.hpp"

// be noisy
// #define DEBUG_STENCIL_CALC

constexpr int64_t RADIUS = 2;

typedef double DTYPE;

using namespace Legion;

/* Need this special kind of GPU accessor for some reason?
 */
typedef FieldAccessor<READ_ONLY, double, 1, coord_t,
                      Realm::AffineAccessor<double, 1, coord_t>>
    AccessorROdouble;
typedef FieldAccessor<WRITE_DISCARD, double, 1, coord_t,
                      Realm::AffineAccessor<double, 1, coord_t>>
    AccessorWDdouble;

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

void init_task(const Task *task, const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime) {}

void stencil_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime) {

  std::cerr << "stencil_task: regions.size()=" << regions.size() << "\n";

  FieldID readFid = *(task->regions[0].privilege_fields.begin());
  FieldID writeFid = *(task->regions[1].privilege_fields.begin());

  Rect<2> readRect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> writeRect = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  std::cerr << "stencil_task: readRect=" << readRect
            << " writeRect=" << writeRect << "\n";
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
  Point<2> lo(1,1);
  Point<2> hi(0,0);
  for (int color = GHOST_LEFT; color <= GHOST_BOT; ++color) {
    coloring[color] = Rect<2>(lo, hi);
  }

  {
    Rect<2> rect(bbLo, bbHi);
    coloring[PRIVATE] = rect;
  }
  if (bbLo[0] > 0 ) {
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
  if (bbHi[1] < numElements[1] - 1){
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
  // https://github.com/magnatelee/PRK/blob/master/LEGION/Stencil/stencil.cc#L614 seems to be an error
  IndexPartition ip = runtime->create_index_partition(ctx, is, colorSpace,
                                                      coloring);
#pragma GCC diagnostic pop
  return runtime->get_logical_partition(ctx, lr, ip);
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions, Context ctx,
                    Runtime *runtime) {

  Point<2> numElements(1024 /*x*/, 1024 /*y*/);
  Point<2> numSubregions(2, 2);
  int num_steps = 10;
  // Check for any command line arguments
  { const InputArgs &command_args = Runtime::get_input_args(); }

  // big top-level index space
  Point<2, size_t> lo(0, 0);
  Point<2, size_t> hi(numElements[0] - 1, numElements[1] - 1);
  Rect<2> elem_rect(lo, hi);
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);
  runtime->attach_name(is, "is");

  // create 2D index partition
  Rect<2> colorSpaceRect(Point<2>(0, 0),
                         Point<2>(numSubregions[0] - 1, numSubregions[1] - 1));
  Domain colorSpace(colorSpaceRect);
  DomainPointColoring haloColoring;
  for (int y = 0; y < numSubregions[1]; ++y) {
    for (int x = 0; x < numSubregions[0]; ++x) {
      DomainPoint color = DomainPoint(Point<2>(x, y));
      Rect<2> haloRect;
      haloRect.lo[0] =
          std::max(numElements[0] / numSubregions[0] * x - RADIUS, 0ll);
      haloRect.lo[1] =
          std::max(numElements[1] / numSubregions[1] * y - RADIUS, 0ll);
      haloRect.hi[0] =
          std::min(numElements[0] / numSubregions[0] * (x + 1) + RADIUS,
                   numElements[0]) -
          1;
      haloRect.hi[1] =
          std::min(numElements[1] / numSubregions[1] * (y + 1) + RADIUS,
                   numElements[1]) -
          1;

      std::cerr << color << " halo=" << haloRect << "\n";

      haloColoring[color] = haloRect;
    }
  }

  // overlapping partitions of the index space, including the halo regions
  IndexPartition haloIp =
      runtime->create_index_partition(ctx, is, colorSpace, haloColoring);

  /* create regions
   */
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(DTYPE), FID_IN);
    allocator.allocate_field(sizeof(DTYPE), FID_OUT);
  }
  std::map<Point<2>, LogicalRegion> logicalRegions;
  std::map<Point<2>, LogicalPartition> logicalPartitions;
  for (int y = 0; y < numSubregions[1]; ++y) {
    for (int x = 0; x < numSubregions[0]; ++x) {
      Point<2> partPoint(x, y);
      IndexSpace subspace = runtime->get_index_subspace(ctx, haloIp, partPoint);
      std::cerr << "create region for " << partPoint << "\n";
      LogicalRegion lr = runtime->create_logical_region(ctx, subspace, fs);
      logicalRegions[partPoint] = lr;
      LogicalPartition lp =
          create_halo_partition(ctx, lr, numElements, runtime);
      logicalPartitions[partPoint] = lp;
    }
  }

  for (int n = 0; n < 3; ++n) {
    std::cerr << "iteration " << n << "\n";

    std::map<Point<2>, Future> futs;

    // initialize data
    for (int y = 0; y < numSubregions[1]; ++y) {
      for (int x = 0; x < numSubregions[0]; ++x) {

        Point<2> taskPoint(x, y);
        TaskLauncher init_launcher(INIT_TASK_ID, TaskArgument(NULL, 0));

        // write-discard access to center region in output
        {
          LogicalRegion reg = runtime->get_logical_subregion_by_color(
              ctx, logicalPartitions[taskPoint], PRIVATE);
          RegionRequirement req(reg, WRITE_DISCARD, EXCLUSIVE,
                                logicalRegions[taskPoint]);
          req.add_field(FID_OUT);
          init_launcher.add_region_requirement(req);
        }

        std::cerr << "init task for " << taskPoint << "\n";
        runtime->execute_task(ctx, init_launcher);
      }
    }

    // copy private FID_OUT the private region of FID_IN
    for (int y = 0; y < numSubregions[1]; ++y) {
      for (int x = 0; x < numSubregions[0]; ++x) {

        Point<2> taskPoint(x, y);

        LogicalRegion reg = runtime->get_logical_subregion_by_color(
            ctx, logicalPartitions[taskPoint], PRIVATE);
        RegionRequirement dst(reg, WRITE_DISCARD, EXCLUSIVE,
                              logicalRegions[taskPoint]);
        dst.add_field(FID_IN);
        RegionRequirement src(reg, READ_ONLY, EXCLUSIVE,
                              logicalRegions[taskPoint]);
        src.add_field(FID_OUT);
        CopyLauncher launcher;
        launcher.add_copy_requirements(src, dst);

        std::cerr << "issue copy for " << taskPoint << "\n";
        runtime->issue_copy_operation(ctx, launcher);
      }
    }

    // launch stencil tasks
    for (int y = 0; y < numSubregions[1]; ++y) {
      for (int x = 0; x < numSubregions[0]; ++x) {

        Point<2> taskPoint(x, y);
        TaskLauncher stencil_launcher(STENCIL_TASK_ID, TaskArgument(NULL, 0));

        // read-only access to the whole input region
        {
          RegionRequirement req(logicalRegions[taskPoint], READ_ONLY,
                                SIMULTANEOUS, logicalRegions[taskPoint]);
          req.add_field(FID_IN);
          stencil_launcher.add_region_requirement(req);
        }

        // write-discard access to center region in output
        {
          LogicalRegion reg = runtime->get_logical_subregion_by_color(
              ctx, logicalPartitions[taskPoint], PRIVATE);
          RegionRequirement req(reg, WRITE_DISCARD, EXCLUSIVE,
                                logicalRegions[taskPoint]);
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

#ifdef USE_NAMEM
  Runtime::add_registration_callback(
      NodeAwareMustEpochMapper::mapper_registration);
#endif

  return Runtime::start(argc, argv);
}
