# Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# The ROCm platform requires at least Ubuntu 16.04 or Fedora 24, which have
# cmake 3.5
cmake_minimum_required(VERSION 3.5...4.0.1 FATAL_ERROR)

# Consider removing this in the future This should appear before the project
# command, because it does not use FORCE
set(CMAKE_INSTALL_PREFIX
    "/opt/rocm"
    CACHE PATH "Install path prefix, prepended onto install directories")

# Adding CMAKE_PREFIX_PATH, needed for static builds
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/llvm /opt/rocm)

# CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake
     ${ROCM_PATH}/lib/cmake/hip /opt/rocm/lib/cmake/hip /opt/rocm/hip/cmake)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()

# Honor per-config flags in try_compile() source-file signature. cmake v3.7 and
# up
if(POLICY CMP0066)
  cmake_policy(SET CMP0066 NEW)
endif()

# rocGRAPH project
project(
  rocgraph
  LANGUAGES CXX C
  VERSION 0.99.0)

# Determine if CXX Compiler is hip-clang
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(STATUS "Using hip-clang to build for amdgpu backend")
else()
  message(FATAL_ERROR "'hipcc' compiler required to compile for ROCm platform.")
endif()

# Build flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_SHARED_LIBS "Build rocGRAPH as a shared library" ON)
option(BUILD_CLIENTS_TESTS "Build tests (requires googletest)" OFF)
option(BUILD_VERBOSE "Output additional build information" OFF)
option(BUILD_CODE_COVERAGE "Build rocGRAPH with code coverage enabled" OFF)
option(BUILD_ADDRESS_SANITIZER "Build rocGRAPH with address sanitizer enabled"
       OFF)
option(BUILD_MEMSTAT "Build rocGRAPH with memory statistics enabled" OFF)
option(BUILD_ROCGRAPH_ILP64 "Build rocGRAPH with rocgraph_int equal to int64_t"
       OFF)
option(BUILD_COMPRESSED_DBG "Enable compressed debug symbols" ON)
option(DOWNLOAD_CLIENTS_TEST_MATRICES "Download client test matrices" OFF)
option(BUILD_WITH_OFFLOAD_COMPRESS
       "Enable offload compression during compilation" ON)
# Clients utilize rocgraph fortran API and a fortran compiler
if(NOT BUILD_FORTRAN_CLIENTS)
  option(BUILD_FORTRAN_CLIENTS
         "Build rocGRAPH clients requiring Fortran capabilities" ON)
endif()
option(BUILD_DOCS "Build documentation" OFF)

# Check if the compiler option exists
if(BUILD_WITH_OFFLOAD_COMPRESS)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("--offload-compress"
                          CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
  if(NOT CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
    message(
      FATAL_ERROR
        "The option '--offload-compress' is not supported from the CXX compiler, turn off the option BUILD_WITH_OFFLOAD_COMPRESS."
    )
  endif()
endif()

#
if(BUILD_CODE_COVERAGE)
  add_compile_options(-fprofile-arcs -ftest-coverage)
  add_link_options(--coverage -lgcov)
endif()

if(BUILD_COMPRESSED_DBG)
  add_compile_options($<$<CONFIG:Debug>:-gz> $<$<CONFIG:RelWithDebInfo>:-gz>)
endif()

if(BUILD_ADDRESS_SANITIZER)

  # CXX
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
  set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -fuse-ld=lld")

  # C
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
  set(CMAKE_C_LINK_EXECUTABLE "${CMAKE_C_LINK_EXECUTABLE} -fuse-ld=lld")

  # Fortran not supported, add_link_options below invalid for fortran linking
  set(BUILD_FORTRAN_CLIENTS OFF)
endif()

# Dependencies
include(cmake/Dependencies.cmake)

# FOR HANDLING ENABLE/DISABLE OPTIONAL BACKWARD COMPATIBILITY for FILE/FOLDER
# REORG
option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY
       "Build with file/folder reorg with backward compatibility enabled" OFF)
if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_wrap_header_dir(
    ${CMAKE_SOURCE_DIR}/library/include
    PATTERNS
    "*.h"
    GUARDS
    SYMLINK
    WRAPPER
    WRAPPER_LOCATIONS
    ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# Detect compiler support for target ID
if(CMAKE_CXX_COMPILER MATCHES ".*/hipcc$")
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} "--help"
    OUTPUT_VARIABLE CXX_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCH ".mcode\-object\-version" TARGET_ID_SUPPORT ${CXX_OUTPUT})
endif()

# Detect compiler support for target ID This section is deprecated. Please use
# rocm_check_target_ids for future use.
if(CMAKE_CXX_COMPILER MATCHES ".*/hipcc$")
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} "--help"
    OUTPUT_VARIABLE CXX_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCH ".mcode\-object\-version" TARGET_ID_SUPPORT ${CXX_OUTPUT})
endif()

# Set the GPU_TARGETS with backward compatibility
if(COMMAND rocm_check_target_ids)
  if(BUILD_ADDRESS_SANITIZER)
    # Disabling ;gfx1030;gfx1100;gfx1101;gfx1102 because of wavefront_size=32
    # see
    # hipco/include/hip_extensions/hip_warp_primitives/amd_warp_primitives.cuh
    rocm_check_target_ids(
      DEFAULT_GPU_TARGETS
      TARGETS
      "gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx942"
    )
  else()
    rocm_check_target_ids(
      DEFAULT_GPU_TARGETS
      TARGETS
      "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx942"
    )
  endif()
else()
  # Use target ID syntax if supported for GPU_TARGETS
  if(TARGET_ID_SUPPORT)
    set(DEFAULT_GPU_TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-")
  else()
    set(DEFAULT_GPU_TARGETS "gfx803;gfx900;gfx906;gfx908")
  endif()
endif()

# if GPU_TARGETS is not set, or set to an empty string or set to "all", than set
# to default
if(NOT GPU_TARGETS OR GPU_TARGETS STREQUAL "all")
  set(GPU_TARGETS "${DEFAULT_GPU_TARGETS}")
endif()

message(STATUS "GPU targets picked: ${GPU_TARGETS}")

# make a sub list of targets that use wavefront size 64
set(WF64_GPU_TARGETS ${GPU_TARGETS})
list(FILTER WF64_GPU_TARGETS INCLUDE REGEX
     "^gfx(942|90a|908|906|900|803)([^0-9])*")

list(LENGTH WF64_GPU_TARGETS WF64_AMDGPU_TARGETS_len)
list(LENGTH GPU_TARGETS AMDGPU_TARGETS_len)

if(WF64_AMDGPU_TARGETS_len GREATER 0)
  # Some are WF64
  if(AMDGPU_TARGETS_len GREATER WF64_AMDGPU_TARGETS_len)
    # Mixed, this is a problem
    message(
      FATAL_ERROR
        "This preview release does not support building WaveFront heterogeneous multi-architecture binaries, where different architectures have dissimilar wavefront sizes."
    )
  else()
    # All are WF64, build without ROCGRAPH_USE_WARPSIZE_32
    if(NOT (ROCGRAPH_USE_WARPSIZE_32 EQUAL 0))
      message(STATUS "Overriding to build without ROCGRAPH_USE_WARPSIZE_32")
    endif()
    set(ROCGRAPH_USE_WARPSIZE_32 0)
    message(STATUS "Building without ROCGRAPH_USE_WARPSIZE_32")
  endif()
else()
  # None are WF64, build with ROCGRAPH_USE_WARPSIZE_32
  if(ROCGRAPH_USE_WARPSIZE_32 EQUAL 0)
    message(STATUS "Overriding to build with ROCGRAPH_USE_WARPSIZE_32")
  endif()
  set(ROCGRAPH_USE_WARPSIZE_32 1)
  message(STATUS "Building with ROCGRAPH_USE_WARPSIZE_32")
endif()

if(GPU_TARGETS)
  set(TMP_GPU_TARGETS "${GPU_TARGETS}")
  if(TMP_GPU_TARGETS STREQUAL "all")
    set(GPU_TARGETS
        "${DEFAULT_GPU_TARGETS}"
        CACHE STRING "List of specific machine types for library to target"
              FORCE)
  else()
    set(GPU_TARGETS
        "${TMP_GPU_TARGETS}"
        CACHE STRING "AMD GPU targets to compile for" FORCE)
  endif()
else()
  set(GPU_TARGETS
      "${DEFAULT_GPU_TARGETS}"
      CACHE STRING "List of specific machine types for library to target")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  find_package(
    hip
    REQUIRED
    CONFIG
    PATHS
    ${HIP_DIR}
    ${ROCM_PATH}
    /opt/rocm)
endif()

# Setup version
rocm_setup_version(VERSION 0.99.0)
set(rocgraph_SOVERSION 0.99.0)

if(BUILD_CLIENTS_TESTS)
  set(BUILD_CLIENTS ON)
  if(NOT CLIENTS_OS)
    rocm_set_os_id(CLIENTS_OS)
    string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
    rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
  endif()

  rocm_package_setup_component(clients)
  rocm_package_setup_client_component(clients-common)
  if(BUILD_CLIENTS_TESTS)
    rocm_package_setup_client_component(tests DEPENDS COMPONENT clients-common)
  endif()
endif()

# rocGRAPH library
add_subdirectory(library)

# Trigger client builds if selected
if(BUILD_CLIENTS)
  add_subdirectory(clients)
endif()

# Build docs
if(BUILD_DOCS)
  add_subdirectory(docs)
endif()

# Package specific CPACK vars
if(BUILD_ADDRESS_SANITIZER)
  set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan")
else()
  set(DEPENDS_HIP_RUNTIME "hip-runtime-amd")
endif()
rocm_package_add_dependencies(DEPENDS "${DEPENDS_HIP_RUNTIME} >= 4.5.0")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
    "\${CPACK_PACKAGING_INSTALL_PREFIX}")

# Package name
set(package_name rocgraph)

set(ROCGRAPH_CONFIG_DIR
    "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
    CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
  NAME
  ${package_name}
  DESCRIPTION
  "ROCm GRAPH library"
  MAINTAINER
  "rocGRAPH Maintainer <rocgraph-maintainer@amd.com>"
  LDCONFIG
  LDCONFIG_DIR
  ${ROCGRAPH_CONFIG_DIR})

#
# ADDITIONAL TARGETS FOR CODE COVERAGE
#
if(BUILD_CODE_COVERAGE)
  #
  # > make coverage_cleanup (clean coverage related files.) > make coverage
  # GTEST_FILTER=<> will run: > make coverage_analysis GTEST_FILTER=<> (analyze
  # tests) > make coverage_output (generate html documentation)
  #

  #
  # Run coverage analysis
  #
  add_custom_target(
    coverage_analysis
    COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
    COMMAND ./clients/staging/rocgraph-test --gtest_filter=\"\${GTEST_FILTER}\"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_dependencies(coverage_analysis rocgraph)

  #
  # Prepare coverage output This little script is generated because the option
  # '--gcov-tool <program name>' of lcov cannot take arguments.
  #
  add_custom_target(
    coverage_output
    DEPENDS coverage_analysis
    COMMAND mkdir -p lcoverage
    COMMAND echo "\\#!/bin/bash" > llvm-gcov.sh
    COMMAND echo "\\# THIS FILE HAS BEEN GENERATED" >> llvm-gcov.sh
    COMMAND printf "exec /opt/rocm/llvm/bin/llvm-cov gcov $$\\@" >> llvm-gcov.sh
    COMMAND chmod +x llvm-gcov.sh)

  #
  # Generate coverage output.
  #
  add_custom_command(
    TARGET coverage_output
    COMMAND
      lcov --directory . --base-directory . --gcov-tool
      ${CMAKE_BINARY_DIR}/llvm-gcov.sh --capture -o
      lcoverage/raw_main_coverage.info
    COMMAND lcov --remove lcoverage/raw_main_coverage.info "'/opt/*'" "'/usr/*'"
            -o lcoverage/main_coverage.info
    COMMAND genhtml lcoverage/main_coverage.info --output-directory lcoverage)

  add_custom_target(coverage DEPENDS coverage_output)

  #
  # Coverage cleanup
  #
  add_custom_target(
    coverage_cleanup
    COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()
