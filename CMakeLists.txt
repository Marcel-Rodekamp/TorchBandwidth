cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")

add_executable(bandwidth_torch bandwidth_torch.cpp)
target_link_libraries(bandwidth_torch "${TORCH_LIBRARIES}")
set_property(TARGET bandwidth_torch PROPERTY CXX_STANDARD 14)

add_executable(bandwidth_cpp bandwidth_cpp.cpp)
set_property(TARGET bandwidth_cpp PROPERTY CXX_STANDARD 14)

add_executable(bandwidth_c bandwidth_cc.cc)
set_property(TARGET bandwidth_c PROPERTY CXX_STANDARD 14)
