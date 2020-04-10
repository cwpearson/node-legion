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