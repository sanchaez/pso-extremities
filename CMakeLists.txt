cmake_minimum_required(VERSION 3.0.2)

project(pso-extremities)

# require C++14
add_definitions(-std=c++14)

# find OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

# test executable
add_executable(interactive_test ${CMAKE_SOURCE_DIR}/test/test.cpp)
