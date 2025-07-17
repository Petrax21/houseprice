cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(aiprojectcpp)

set(CMAKE_CUDA_ARCHITECTURES 86)
enable_language(CUDA)

add_library(CUDA::nvToolsExt INTERFACE IMPORTED GLOBAL)

set(CMAKE_PREFIX_PATH "/home/mustafa/libtorch/libtorch")
find_package(Torch REQUIRED)

set(MATPLOT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MATPLOT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(matplotplusplus)

add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}" matplot)
set_property(TARGET main PROPERTY CXX_STANDARD 17)

target_include_directories(main PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        /opt/cuda/include
)
