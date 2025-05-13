# Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required(VERSION 3.5...4.0.1 FATAL_ERROR)

# Consider removing this in the future This should appear before the project
# command, because it does not use FORCE
set(CMAKE_INSTALL_PREFIX
    "/opt/rocm"
    CACHE PATH "Install path prefix, prepended onto install directories")

# This has to be initialized before the project() command appears Set the
# default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.
# MSVC_IDE does not use CMAKE_BUILD_TYPE
if(NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE
        STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
  )
endif()

# This project may compile dependencies for clients
project(rocgraph-clients LANGUAGES CXX)

# Determine if CXX Compiler is hip-clang
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(STATUS "Using hip-clang to build for amdgpu backend")
else()
  message(FATAL_ERROR "'hipcc' compiler required to compile for ROCm platform.")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# This option only works for make/nmake and the ninja generators, but no reason
# it shouldn't be on all the time This tells cmake to create a
# compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT TARGET rocgraph)
  find_package(rocgraph REQUIRED CONFIG PATHS /opt/rocm/rocgraph)
  option(BUILD_CLIENTS_TESTS "Build tests (requires googletest)" OFF)
endif()

# HIP headers required of all clients; clients use hip to allocate device memory
list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
find_package(HIP REQUIRED CONFIG PATHS ${CMAKE_PREFIX_PATH})

# Build flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# cmake-format: off
# # If OpenMP is available, we can use it to speed up some tests
# find_package(OpenMP QUIET)
#
# if(OPENMP_FOUND)
#   if(NOT TARGET OpenMP::OpenMP_CXX)
#     # OpenMP cmake fix for cmake <= 3.9
#     add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
#     set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS
#                                                     ${OpenMP_CXX_FLAGS})
#     set_property(
#       TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_LIBRARIES
#                                          ${OpenMP_CXX_FLAGS} Threads::Threads)
#   endif()
# endif()
# cmake-format: on

if(BUILD_CLIENTS_TESTS)
  # For now, we require Python3 and its yaml module.
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import yaml"
    RESULT_VARIABLE EXIT_CODE
    ERROR_QUIET OUTPUT_QUIET)
  if(NOT ${EXIT_CODE} EQUAL 0)
    message(FATAL_ERROR "The rocGRAPH tests require the Python3 'yaml' module.")
  endif()

  enable_testing()
  add_subdirectory(tests)

  set(ROCGRAPH_COMMON "${PROJECT_BINARY_DIR}/staging/rocgraph_common.yaml")
  add_custom_command(
    OUTPUT "${ROCGRAPH_COMMON}"
    COMMAND ${CMAKE_COMMAND} -E copy include/rocgraph_common.yaml
            "${ROCGRAPH_COMMON}"
    DEPENDS include/rocgraph_common.yaml
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  set(ROCGRAPH_TEMPLATE "${PROJECT_BINARY_DIR}/staging/rocgraph_template.yaml")
  add_custom_command(
    OUTPUT "${ROCGRAPH_TEMPLATE}"
    COMMAND ${CMAKE_COMMAND} -E copy include/rocgraph_template.yaml
            "${ROCGRAPH_TEMPLATE}"
    DEPENDS include/rocgraph_template.yaml
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  set(ROCGRAPH_SMOKE "${PROJECT_BINARY_DIR}/staging/rocgraph_smoke.yaml")
  set(ROCGRAPH_REGRESSION
      "${PROJECT_BINARY_DIR}/staging/rocgraph_regression.yaml")
  set(ROCGRAPH_EXTENDED "${PROJECT_BINARY_DIR}/staging/rocgraph_extended.yaml")
  add_custom_command(
    OUTPUT "${ROCGRAPH_SMOKE}" "${ROCGRAPH_REGRESSION}" "${ROCGRAPH_EXTENDED}"
    COMMAND ${CMAKE_COMMAND} -E copy include/rocgraph_smoke.yaml
            "${ROCGRAPH_SMOKE}"
    COMMAND ${CMAKE_COMMAND} -E copy include/rocgraph_regression.yaml
            "${ROCGRAPH_REGRESSION}"
    COMMAND ${CMAKE_COMMAND} -E copy include/rocgraph_extended.yaml
            "${ROCGRAPH_EXTENDED}"
    DEPENDS include/rocgraph_smoke.yaml include/rocgraph_regression.yaml
            include/rocgraph_extended.yaml
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  set(ROCGRAPH_GENTEST "${PROJECT_BINARY_DIR}/staging/rocgraph_gentest.py")
  add_custom_command(
    OUTPUT "${ROCGRAPH_GENTEST}"
    COMMAND ${CMAKE_COMMAND} -E copy common/rocgraph_gentest.py
            "${ROCGRAPH_GENTEST}"
    DEPENDS common/rocgraph_gentest.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  add_custom_target(
    rocgraph-common
    DEPENDS "${ROCGRAPH_COMMON}" "${ROCGRAPH_TEMPLATE}" "${ROCGRAPH_SMOKE}"
            "${ROCGRAPH_GENTEST}" "${ROCGRAPH_CONVERT}")
  rocm_install(
    FILES
    "${ROCGRAPH_COMMON}"
    "${ROCGRAPH_TEMPLATE}"
    "${ROCGRAPH_SMOKE}"
    COMPONENT
    clients-common
    DESTINATION
    "${CMAKE_INSTALL_DATADIR}/rocgraph/test")
  #
  rocm_install(PROGRAMS "${ROCGRAPH_GENTEST}" COMPONENT clients-common
               DESTINATION "${CMAKE_INSTALL_DATADIR}/rocgraph/test")
  #
  rocm_install(PROGRAMS "${ROCGRAPH_CONVERT}" COMPONENT clients-common
               DESTINATION ${CMAKE_INSTALL_BINDIR})
  #
  rocm_install(FILES "${ROCGRAPH_CLIENTMATRICES}" COMPONENT clients-common
               DESTINATION "${CMAKE_INSTALL_DATADIR}/rocgraph/test")

endif()
