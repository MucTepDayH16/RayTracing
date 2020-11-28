cmake_minimum_required(VERSION 3.17)
project(RayTracing VERSION 1.0 LANGUAGES CXX CUDA)

set(RT_INC)
set(RT_CXX_SRC
        "${PROJECT_SOURCE_DIR}/Source/main.cpp"
        "${PROJECT_SOURCE_DIR}/Source/fileIO.cpp"
        "${PROJECT_SOURCE_DIR}/Source/cuda_rays.cu"
        )
set(RT_LIB)


# CUDA configuration
find_package(CUDA)
set(CMAKE_CUDA_FLAGS "-arch=sm_${CUDA_ARCH} --use_fast_math")
set(CMAKE_CUDA_LIBRARY_DIRECTORIES "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")

list(APPEND RT_INC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
list(APPEND RT_LIB ${CMAKE_CUDA_LIBRARY_DIRECTORIES}/cuda.lib ${CMAKE_CUDA_LIBRARY_DIRECTORIES}/nvrtc.lib)

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
configure_file("${PROJECT_SOURCE_DIR}/Include/config" "${PROJECT_SOURCE_DIR}/Include/config.h")

add_executable(RayTracing ${RT_CXX_SRC})
target_include_directories(RayTracing PUBLIC ${RT_INC})
target_link_libraries(RayTracing PUBLIC ${RT_LIB})
target_compile_features(RayTracing PUBLIC cxx_std_20)

if(APPLE)
  set_target_properties(RayTracing PROPERTIES BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
