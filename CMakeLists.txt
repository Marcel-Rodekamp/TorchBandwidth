cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")

add_executable(vectorAdd_torch vectorAdd_torch.cpp)
target_link_libraries(vectorAdd_torch "${TORCH_LIBRARIES}")
set_property(TARGET vectorAdd_torch PROPERTY CXX_STANDARD 14)

add_executable(vectorAdd_cpp vectorAdd_cpp.cpp)
set_property(TARGET vectorAdd_cpp PROPERTY CXX_STANDARD 14)

add_executable(vectorAdd_cc vectorAdd_cc.cc)
set_property(TARGET vectorAdd_cc PROPERTY CXX_STANDARD 14)
