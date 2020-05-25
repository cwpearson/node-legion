cmake_minimum_required(VERSION 3.1)
project(hello_world LANGUAGES CXX CUDA VERSION 0.1.0.0)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

add_executable(hello-world tut_hello_world.cpp)
target_link_libraries(hello-world Legion::Legion)
if(Legion_ENABLE_TESTING)
  add_test(NAME hello_world COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:hello_world> ${Legion_TEST_ARGS})
endif()

add_library(solve OBJECT solve.cpp)

add_library(mapper OBJECT node_aware_mapper.cpp)
target_include_directories(mapper PRIVATE SYSTEM /usr/local/cuda/include)
target_link_libraries(mapper Legion::Legion)
target_link_libraries(mapper solve)

add_executable(multiple tut_multiple.cpp)
target_link_libraries(multiple Legion::Legion)

add_executable(ghost tut_ghost.cpp)
target_link_libraries(ghost Legion::Legion)

add_executable(custom-mapper custom_mapper.cpp)
target_link_libraries(custom-mapper Legion::Legion)

add_executable(0-hello-world 0_hello_world.cu)
target_link_libraries(0-hello-world Legion::Legion)

add_executable(1-multiple 1_multiple.cu)
target_link_libraries(1-multiple Legion::Legion)

add_executable(2-ghost-gpu 2_ghost_gpu.cu)
target_link_libraries(2-ghost-gpu Legion::Legion)
target_link_libraries(2-ghost-gpu nvToolsExt)
target_link_libraries(2-ghost-gpu mapper)
target_link_libraries(2-ghost-gpu solve)

add_executable(print-machine print_machine.cpp)
target_link_libraries(print-machine Legion::Legion)

add_executable(3-stencil-indexspace 3_stencil_indexspace.cu)
target_link_libraries(3-stencil-indexspace Legion::Legion)
target_link_libraries(3-stencil-indexspace nvToolsExt)
target_link_libraries(3-stencil-indexspace mapper)
target_link_libraries(3-stencil-indexspace solve)

add_executable(4-stencil 4_stencil.cu)
target_link_libraries(4-stencil Legion::Legion)
target_link_libraries(4-stencil nvToolsExt)
target_link_libraries(4-stencil mapper)
target_link_libraries(4-stencil solve)

add_executable(5-3dstencil 5_3dstencil.cu)
target_link_libraries(5-3dstencil Legion::Legion)
target_link_libraries(5-3dstencil nvToolsExt)
target_link_libraries(5-3dstencil mapper)
target_link_libraries(5-3dstencil solve)

add_executable(bench-solve bench_solve.cpp)
target_link_libraries(bench-solve solve)