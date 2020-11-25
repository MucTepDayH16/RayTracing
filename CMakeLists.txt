cmake_minimum_required(VERSION 3.17)
project(RayTracing VERSION 0.1 LANGUAGES CXX CUDA)

set(RT_INC)
set(RT_CXX_SRC
        "${PROJECT_SOURCE_DIR}/Source/main.cpp"
        # "${PROJECT_SOURCE_DIR}/Source/fileIO.cpp"
        "${PROJECT_SOURCE_DIR}/Source/cuda_rays.cu"
        )
set(RT_LIB)


# CUDA configuration
set(RT_CUDA_SRC
        "${PROJECT_SOURCE_DIR}/Source/cuda_kernels.cu"
        )
set(CMAKE_CUDA_FLAGS "-arch=sm_${CUDA_ARCH} --use_fast_math")

find_package(CUDA)
list(APPEND RT_INC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
list(APPEND RT_LIB ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib)

# OpenGL configuration
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

list(APPEND RT_LIB ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})

# SDL2 configuration
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/SDL2Extra)
find_package(SDL2 REQUIRED)

list(APPEND RT_INC ${SDL2_INCLUDE_DIRS})
list(APPEND RT_LIB ${SDL2_LIBRARIES})

# Project configuration
add_library(CudaPTX OBJECT ${RT_CUDA_SRC})
target_include_directories(CudaPTX PUBLIC ${RT_INC})
set_target_properties(CudaPTX PROPERTIES CUDA_PTX_COMPILATION ON)

add_executable(RayTracing ${RT_CXX_SRC})
target_include_directories(RayTracing PUBLIC ${RT_INC})
target_link_libraries(RayTracing PUBLIC ${RT_LIB})
target_compile_features(RayTracing PUBLIC cxx_std_20)
# set_target_properties(RayTracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(APPLE)
  set_target_properties(RayTracing PROPERTIES BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
