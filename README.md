# node-legion

Node-Aware Placement in Legion

## Install Legion

This will build legion with CUDA support for SM 61 and install into `legion-install` at the repo root.

* GTX 1070 (Pascal): `61`
* V100 (Volta): `70`
```
git clone https://github.com/StanfordLegion/legion.git 
cd legion
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=`readlink -f ../../legion-install/` -DLegion_USE_CUDA=ON -DLegion_CUDA_ARCH=61
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

* `-ll:gpu <int>`: "GPUs to create per process"
* `-ll:fsize <int>`: size of framebuffer memory for each GPU in MB
* `-ll:zsize <int>`: size of zero-copy memory for each GPU in MB

Seems this flag should be set to the number of GPUs per node?

```
./0-hello-world -ll:gpu 1
```

## Files

* `NodeAwareMapper.hpp`: impl of node-aware mapper
* `0_hello_world.cu`: Hello World with `NodeAwareMapper`, CPU and GPU tasks
* `2_ghost_gpu.cu`:  Ghost tutorial with GPU stencil task
  - [x] GPU task
  - [x] Data accessible from GPU
  - [x] GPU stencil
  - [ ] Data in FrameBuffer

## Resources

* [Legion Programming System](https://legion.stanford.edu)
* [Legion C++ API](https://legion.stanford.edu/doxygen/)
* [Explit Ghost Regions (`tut_ghost.cpp`)](https://legion.stanford.edu/tutorial/ghost.html)
* [Comments on GPU tasks in the circuit example](https://legion.stanford.edu/tutorial/circuit.html)

* Mappers
  * [Legion Mapper API](https://legion.stanford.edu/mapper/index.html)
  * [Machine Query Interface](https://github.com/StanfordLegion/legion/blob/stable/runtime/mappers/mapping_utilities.cc)


* `map_must_epoch`
  * [MappingConstraint, MapMustEpochInput, MapMustEpochOutput](https://github.com/StanfordLegion/legion/blob/f3f4e7d987768598b554ffca65d730f697956dc8/runtime/legion/legion_mapping.h#L1406)
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

## Using `LogicalRegion` overlap to model communication between must-epoch tasks

Each task comes with a set of `RegionRequirement`s, which describe the data access that a Task will make

* If the `RegionRequirement` `is_no_access()`, then it does not contribute any communication
* If `LogicalRegion`s are in different trees, overlap is 0
* If `LogicalRegion`s overlap, but fields are disjoint, overlap is 0
* If I access with `WRITE_DISCARD` privilege, no communication to me is required

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

