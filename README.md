# node-legion

Node-Aware Placement in Legion

## Install Legion

This will build legion with CUDA support for SM 61 and install into `legion-install` at the repo root.

* GTX 1070 (Pascal): `-DLegion_CUDA_ARCH=61`
* V100 (Volta): `-DLegion_CUDA_ARCH=70`
* `-DLegion_OUTPUT_LEVEL=[SPEW DEBUG INFO PRINT WARNING ERROR FATAL NONE]`

First, install legion:
```
git clone git@github.com:cwpearson/legion.git
cd legion
git checkout cwpearson-nvidiaml
mkdir build
cd build
cmake .. -DLegion_USE_CUDA=ON -DLegion_CUDA_ARCH=61 -DCMAKE_INSTALL_PREFIX=`readlink -f ../../legion-install`
```

Then, point node-legion at the build of legion
```
cd node-legion
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=`readlink -f ../legion-install/` \
  -DLegion_USE_CUDA=ON -DLegion_CUDA_ARCH=61
```

For a Debug build:
```
cmake .. \
  -DLegion_OUTPUT_LEVEL=SPEW \
  -DCMAKE_BUILD_TYPE=Debug
make
make install
```

For a Release build:
```
cmake .. \
  -DLegion_OUTPUT_LEVEL=INFO \
  -DCMAKE_BUILD_TYPE=Release
make
make install
```
## Build our code

Assuming legion was installed at the repo root.

```
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=../legion-install
```

## Run our code


* `print-machine`: print some info Legion can tell about the `machine
* `2-ghost-gpu`: 1D stencil with explicit ghost and exchange, with node-aware mapping
* `5-3dstencil`: 3D stencil with index space launch
  * `-nx/-ny/-nz`: x/y/z elements
  * `-bx/-by/-bz`: x/y/z subregions
  * `-s`: iterations


* Legion Runtime Flags ([more info](https://github.com/StanfordLegion/legion#command-line-flags))
  * `-ll:gpu <int>`: GPUs to create per process
  * `-ll:csize <int>`: size of CPU DRAM memory per process (in MB)
  * `-ll:fsize <int>`: size of framebuffer memory for each GPU in MB
  * `-ll:zsize <int>`: size of zero-copy memory for each GPU in MB
  * `-lg:sched <int>`: minimum number of tasks to try to schedule for each invocation of the scheduler
  * log level `0` is everything, `5` is errors only
    * `-level gpu=2`: to see some logging about GPU distances
    * `-level node_aware_must_epoch_mapper=1` to see some debug messages from the Mapper
    * `-level node_aware_mapper=0` to see all Node-Aware Mapper output
    * `-level nvml=0,realm/linux=0,system,topology=0`: to see info about system topology



## Files

* `NodeAwareMapper.hpp`: impl of node-aware mapper
* `0_hello_world.cu`: Hello World with `NodeAwareMapper`, CPU and GPU tasks
* `2_ghost_gpu.cu`:  Ghost tutorial with GPU stencil task
  - [x] GPU task
  - [x] Data accessible from GPU
  - [x] GPU stencil
  - [ ] Data in FrameBuffer

## Resources

* [Legion Github](https://github.com/StanfordLegion/legion)
* [Legion Programming System](https://legion.stanford.edu)
* [Legion C++ API](https://legion.stanford.edu/doxygen/)
* [Explit Ghost Regions (`tut_ghost.cpp`)](https://legion.stanford.edu/tutorial/ghost.html)
* [Comments on GPU tasks in the circuit example](https://legion.stanford.edu/tutorial/circuit.html)

* Mappers
  * [Legion Mapper API](https://legion.stanford.edu/mapper/index.html)
  * [Custom Mapper Tutorial](https://legion.stanford.edu/tutorial/custom_mappers.html)
  * [Machine Query Interface](https://github.com/StanfordLegion/legion/blob/stable/runtime/mappers/mapping_utilities.cc)


* `map_must_epoch`
  * [MappingConstraint, MapMustEpochInput, MapMustEpochOutput](https://github.com/StanfordLegion/legion/blob/f3f4e7d987768598b554ffca65d730f697956dc8/runtime/legion/legion_mapping.h#L1406)
    * `MappingConstraint` which logical regions of different tasks must be mapped to the same physical instance
  * [Legion Mapper API](https://legion.stanford.edu/mapper/index.html)

## Legion Concepts

* `LogicalRegion`
  * Has an *index space* and *field space*
  * A field or index space can be in multiple logical regions
* `RegionRequirement`
  * An access to data required for a `Task`
  * `IndexSpace`
  * `FieldSpace`
  * `PrivilegeMode`

* Interference [Tutorial: Privileges](https://legion.stanford.edu/tutorial/privileges.html)
  * Region non-interference: RegionRequirements are non-interfering on regions if they access logical regions from different region trees, or disjoint logical regions from the same tree.

* Index Space Tasks
  * [Index Space Tasks (Tutorial)](https://legion.stanford.edu/tutorial/index_tasks.html)
  * [slice_task (Mapper API)](https://legion.stanford.edu/mapper/index.html)
  * [slice_task (Custom Mappers)](https://legion.stanford.edu/tutorial/custom_mappers.html)
  * Index Space tasks are supposed to be non-interfering, but we have no visibility into say read-only data that is shared between these non-interfering tasks

  IndexTaskLaunchers deliver a single Task with a RegionRequirement with a `.partition` (and not a `.region`) to `<mapper>::slice_task`.
  * First we adopt the fundamental theorem of arithmetic to split up the launch space along its longest dimension to prime number factors to match the number of GPUs
    * in the future, it would be cool to do some kind of communication-aware splitting. Maybe trial communication costs along a few axes to decide how to split?
  *  We iterate over the index space of that partition to extract the amount that each subregion overlaps with its neighboring subregions.
  * Then we use that to assign subregions to GPUs

## Machine Mode

* [Discussion in `Introduction to the Legion Mapper API`](https://legion.stanford.edu/mapper/#Machine-model)
* `ProcessorMemoryAffinity` [(github)](https://github.com/StanfordLegion/legion/blob/f3f4e7d987768598b554ffca65d730f697956dc8/runtime/realm/machine.h#L77)
* `Processor` in `realm/processor` [(github)](https://github.com/StanfordLegion/legion/blob/f3f4e7d987768598b554ffca65d730f697956dc8/runtime/realm/processor.h#L35)

## Building a better Legion::Machine

* `runtime_impl.cc`
  * for all modules, calls 
    * `Module::create_memories`
    * `Module::create_processors`
    * ...
    * `Module::create_dma_channels`
* `runtime/realm/cuda/cuda_module.cc`
  * `CudaModule::create_module`:
    * Gets the `GPUInfo`
    * checks canAccessPeer
  * `CudaModule::initialize`:
    * creates the `GPU`s
  * `CudaModule::create_memories`:
    * calls each `GPU::create_fb_memory`
      * ...which creates the `GPU::fbmem`
  * `GPU::create_processor`
    * where peer devices are discovered and their framebuffers are put into `peer_fbs`
    * where the `ProcessorMemoryAffinity`s with its own ZC and FB memories are created
    * where the `ProcessorMemoryAffinity`s with peer GPU FB memories are created.
      * this is assumed to be PCIe, with 1/2 the ZC bandwidth and 2x the latency
  * `CudaModule::create_dma_channels()`
    * calls `GPU::create_dma_channels()`
      * where `MemoryMemoryAffinity`s with all other peer GPU FBs are created.

Ideally we'd like to be able to differentiate between 5 different GPU-GPU cases
  * (1) GPU - NvLink - CPU - SMP - CPU - NvLink - GPU (Newell)
  * (2a) GPU - NvLink - GPU (DGX2, Newell)
  * (3) GPU - PCIe - CPU - SMP - CPU - PCIe - GPU
  * (4) GPU - PCIe - GPU
These could be further refined, especially in the PCIe hierarchy case.
There is one special case in the DXG-2, where it is not clear how data moves. It could be either of
  * (2b) GPU - NvLink - GPU - NvLink - GPU
  * (3)

| kind | when | latency | bandwidth |
|-|-|-|-|
|ZC (CPU/GPU)| original distance to zc memory |200 | 20
|NVLINK_CLOSE | DistanceKind::NVLINK_CLOSE | 260 | 16 |
|NVLINK_FAR | DistanceKind::NVLINK_FAR | 320 | 13 |
|pcie | all other cases (original GPU/GPU distance) | 400 |  10


* Version 1:
  * We collect pairwise GPU connection info in `CudaModule::create_module`. If src is connected to dst, it is (1). If src is not connected to dst with NvLink directly, then we assume case (2a) or (2b), which we will not distinguish.
  If src has no NvLinks, depending on the last common PCIe ancestor, we distinguish between (3) and (4). The case will be used in `GPU::create_processor` and `GPU::create_dma_channels` to set the right latency and bandwidth values for `MemoryMemoryAffinity`s and `ProcessorMemoryAffinity`s. `channel.cc` has case `XFER_GPU_PEER_FB` that probably needs some numbers?
* Verson 2:
  * We will build a true system graph, including NvSwitches. Each node will have a link with estimated bandwidth and latency properties, and the overall affinity will be determined by merging links along the correct path.



We will start with proposal 1.


## Using `LogicalRegion` overlap to model communication between must-epoch tasks

Each task comes with a set of `RegionRequirement`s, which describe the data access that a Task will make


* If `LogicalRegion`s are in different trees, overlap is 0
* If `LogicalRegion`s overlap, but fields are disjoint, overlap is 0
* If I access with `WRITE_DISCARD` privilege, no communication to me is required

* It seems the NO_ACCESS flag just says the `spmd_task` will not be making accessors for the left and right regions (they are accessed through the main logical region).
  * Therefore, we ignore it during grouping because the task will not access it directly, so it does not actually constrain which logical regions need to be in the same physical instance.
  * We do *not* ignore it during communication, because NO_ACCESS does not prevent the task from issuing an explicit copy on that region. 
* The `is_no_access()` on a task doesn't mean we should ignore it. It means that 

## DefaultMapper `map_must_epoch`

* Group tasks by processor kind
* Then, within each processor kind, group tasks by mapping constraints
  * MappingConstraint says that certain logical regions must be in the same physical instance.
  * If two tasks have a logical region in the same mapping constraint, they are in the same group
* Then spread groups of tasks evenly among address spaces'
  * [impl](https://github.com/StanfordLegion/legion/blob/f3f4e7d987768598b554ffca65d730f697956dc8/runtime/mappers/default_mapper.cc#L3207)
  * [callsite](https://github.com/StanfordLegion/legion/blob/f3f4e7d987768598b554ffca65d730f697956dc8/runtime/mappers/default_mapper.cc#L3552)

## `node_aware_must_epoch_mapper`

* Restrict to GPU tasks for now
* Group tasks by mapping constraints
* Query machine graph to compute distance between GPUs
* Query logical regions to figure out overlap
* Match tasks

Use region interference as a proxy for required data exchange between tasks

## Future Experiments

* 2D Stencil
* Neural Network training
  * interplay between combined model and data parallelism
* Mert's hierarchical methods? Or sinogram/tomogram data placement
* Block Sparse matrix operations?



## Task placement

* Do a good partition and then solve quadratic assignment problem
* or, overdecompose and then solve general assignment problem
  * Greedy approaches for a class of nonlinear Generalized Assignment Problems (Sharkey and Romeijn)
  * Maybe use dlib

