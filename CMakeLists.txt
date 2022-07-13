cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

find_package(OpenMP)
# =======================================================================
# Vec Add
# =======================================================================

add_executable(vectorAdd_torch vectorAdd_torch.cpp)
target_link_libraries(vectorAdd_torch "${TORCH_LIBRARIES}" OpenMP::OpenMP_CXX)
set_property(TARGET vectorAdd_torch PROPERTY CXX_STANDARD 14)

add_executable(vectorAdd_cpp vectorAdd_cpp.cpp)
target_link_libraries(vectorAdd_cpp OpenMP::OpenMP_CXX)
set_property(TARGET vectorAdd_cpp PROPERTY CXX_STANDARD 14)

add_executable(vectorAdd_cc vectorAdd_cc.cc)
target_link_libraries(vectorAdd_cc OpenMP::OpenMP_CXX)
set_property(TARGET vectorAdd_cc PROPERTY CXX_STANDARD 14)

add_executable(vectorAdd_cu vectorAdd_cu.cu)
target_link_libraries(vectorAdd_cu OpenMP::OpenMP_CXX)

add_executable(vectorAdd_torch_cu vectorAdd_torch_cu.cpp)
target_link_libraries(vectorAdd_torch_cu "${TORCH_LIBRARIES}" OpenMP::OpenMP_CXX)
set_property(TARGET vectorAdd_torch_cu PROPERTY CXX_STANDARD 14)



# =======================================================================
# Mat Vec
# =======================================================================

add_executable(matVec_cpp matVec_cpp.cpp)
target_link_libraries(matVec_cpp OpenMP::OpenMP_CXX)
set_property(TARGET matVec_cpp PROPERTY CXX_STANDARD 14)

add_executable(matVec_torch matVec_torch.cpp)
target_link_libraries(matVec_torch "${TORCH_LIBRARIES}" OpenMP::OpenMP_CXX)
set_property(TARGET matVec_torch PROPERTY CXX_STANDARD 14)
