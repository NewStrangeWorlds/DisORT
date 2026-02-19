cmake_minimum_required(VERSION 3.15)
project(disortpp VERSION 1.0 LANGUAGES CXX)

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings via pybind11" OFF)
option(DISORTPP_MARCH_NATIVE "Use -march=native optimization (disable for portable binaries)" ON)

# C++17 Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(MSVC)
    add_compile_definitions(_USE_MATH_DEFINES)
    add_compile_options(/bigobj)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /DEIGEN_NO_DEBUG")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DEIGEN_NO_DEBUG")
    if(DISORTPP_MARCH_NATIVE)
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -march=native")
    endif()
endif()

# Dependencies: Eigen3 via FetchContent (downloaded into _deps/)
include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
# Eigen's CMakeLists builds tests/docs by default — disable them
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
# Use manual populate + EXCLUDE_FROM_ALL to prevent Eigen's install targets
# from polluting the Python wheel with thousands of header files
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
    add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Dependencies: pybind11 via FetchContent (optional, for Python bindings)
if(BUILD_PYTHON_BINDINGS)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Source files
set(DISORTPP_SOURCES
    src/DisortConfig.cpp
    src/DisortFluxConfig.cpp
    src/PhaseFunction.cpp
    src/DisortResult.cpp
    src/Legendre.cpp
    src/Planck.cpp
    src/LinearAlgebra.cpp
    src/DisortSolver.cpp
    src/FluxResult.cpp
    src/FluxSolver.cpp
)

# Library target
add_library(disortpp ${DISORTPP_SOURCES})

target_include_directories(disortpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:src>
)

target_link_libraries(disortpp PUBLIC Eigen3::Eigen)

# OpenMP (optional, for parallel spectral loops in Python bindings)
find_package(OpenMP)

# Python bindings (optional)
if(BUILD_PYTHON_BINDINGS)
    set_target_properties(disortpp PROPERTIES POSITION_INDEPENDENT_CODE ON)
    pybind11_add_module(disortpp_python src/python_bindings.cpp)
    target_link_libraries(disortpp_python PRIVATE disortpp)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(disortpp_python PRIVATE OpenMP::OpenMP_CXX)
    endif()
    set_target_properties(disortpp_python PROPERTIES OUTPUT_NAME "disortpp")
    install(TARGETS disortpp_python LIBRARY DESTINATION .)
endif()

# Tests
enable_testing()
add_subdirectory(tests)

# Benchmarks (uncomment when benchmark sources are available)
# add_executable(bench_flux_solver bench_flux_solver.cpp)
# target_link_libraries(bench_flux_solver disortpp)

# add_executable(bench_intensity bench_intensity.cpp)
# target_link_libraries(bench_intensity disortpp)

# Installation
# install(TARGETS disortpp DESTINATION lib)
# install(DIRECTORY include/disortpp DESTINATION include)

# Print configuration
message(STATUS "")
message(STATUS "========== disort++ C++ Configuration ==========")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Eigen3: fetched via FetchContent (3.4.0)")
message(STATUS "Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "==============================================")
message(STATUS "")
