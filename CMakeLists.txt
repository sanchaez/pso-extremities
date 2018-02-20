cmake_minimum_required(VERSION 3.0.2)

project(pso-extremities)

# project options
option(USE_OMP "Use OpenMP." OFF)
option(USE_SSE "Use the SSE2 optimizations." OFF)
option(USE_AVX "Use AVX optimizations." ON)

# require C++14
add_definitions(-std=c++14)

include_directories(${CMAKE_SOURCE_DIR}/include)

# different flags for builds
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG  "-Od -g")


# find OpenMP
if(USE_OMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  endif()
endif()

if(NOT (USE_OMP AND OPENMP_FOUND))
  #ignore unknown omp pragmas
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
endif()

if(USE_SSE)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -msse2")
endif()

if(USE_AVX)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx")
endif()

# test executable
add_executable(interactive_test ${CMAKE_SOURCE_DIR}/test/test.cpp)

