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
  * [MappingConstraint, MapMustEpochInput, MapMustEpochOutput](https://github.com/StanfordLegion/legion/blob/f3f4e7d987768598b554ffca65d730f697956dc8/runtime/legion/legion_mapping.h#L1406)
