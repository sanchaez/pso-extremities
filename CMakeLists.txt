cmake_minimum_required(VERSION 3.0.2)

project(pso-extremities)

# project options
option(USE_OMP "Use OpenMP." OFF)

# require C++14
set (CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_SOURCE_DIR}/include)

# different flags for builds
if (CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall /EHsc")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /Zi /arch:AVX /fp:precise")
    set(CMAKE_CXX_FLAGS_RELEASE "/Ox /favor:blend /arch:AVX /fp:precise")
    set(CMAKE_CXX_FLAGS_DEBUG  "/Od /Zi ")        
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG  "-Od -g")
endif()

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
	if (CMAKE_CXX_COMPILER_ID MATCHES MSVC)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} /wd4068")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    endif()
endif()

# test executable
add_executable(interactive_test ${CMAKE_SOURCE_DIR}/test/test.cpp)
