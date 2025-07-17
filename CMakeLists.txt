cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(aiprojectcpp)

# CUDA ve NVTX ayarları
find_package(CUDA REQUIRED)
find_library(NVTOOLSEXT_LIBRARY nvToolsExt PATHS /opt/cuda/lib64 /usr/lib64 /usr/lib)

# NVTX desteğini tamamen devre dışı bırak
set(TORCH_NVTX OFF)

# LibTorch ayarları
set(CMAKE_PREFIX_PATH "~/libtorch/libtorch")

# LibTorch'u bul
find_package(Torch REQUIRED)

# NVTX target'ını manuel olarak tanımla
if(NVTOOLSEXT_LIBRARY)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    set_target_properties(CUDA::nvToolsExt PROPERTIES
            INTERFACE_LINK_LIBRARIES "${NVTOOLSEXT_LIBRARY}"
    )
endif()

# Matplot++ ayarları
set(MATPLOT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MATPLOT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(matplotplusplus)

# cuDNN ayarları
find_library(CUDNN_LIBRARY cudnn PATHS /opt/cuda/lib64 /usr/lib64 /usr/lib)
if(CUDNN_LIBRARY)
    message(STATUS "cuDNN found: ${CUDNN_LIBRARY}")
else()
    message(WARNING "cuDNN not found, using CPU only")
endif()

# Executable
add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}" matplot)
set_property(TARGET main PROPERTY CXX_STANDARD 17)

# Include directories
target_include_directories(main PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        /opt/cuda/include
)
