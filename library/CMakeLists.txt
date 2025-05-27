# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# ##############################################################################
# Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################

# This option only works for make/nmake and the ninja generators, but no reason
# it shouldn't be on all the time This tells cmake to create a
# compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Print verbose compiler flags
if(BUILD_VERBOSE)
  include(../cmake/Verbose.cmake)
endif()

# MEMSTAT ?
if(BUILD_MEMSTAT)
  add_compile_options(-DROCGRAPH_WITH_MEMSTAT)
endif()

if(BUILD_ROCGRAPH_ILP64)
  add_compile_options(-Drocgraph_ILP64)
endif()

# Configure a header file to pass the rocGRAPH version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocgraph-version.h.in"
               "${PROJECT_BINARY_DIR}/include/rocgraph/rocgraph-version.h")

# Public rocGRAPH headers
set(rocgraph_headers_public
    include/rocgraph-auxiliary.h include/rocgraph-functions.h
    include/rocgraph-types.h include/rocgraph.h
    ${PROJECT_BINARY_DIR}/include/rocgraph/rocgraph-version.h)

source_group("Header Files\\Public" FILES ${rocgraph_headers_public})

# Include sources
include(src/CMakeLists.txt)

# Create rocGRAPH library
add_library(rocgraph ${rocgraph_source} ${rocgraph_headers_public})
add_library(roc::rocgraph ALIAS rocgraph)

# Target compile options
target_compile_options(rocgraph PRIVATE -Wno-unused-command-line-argument -Wall)
target_compile_options(
  rocgraph PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${ROCGRAPH_CXX_FLAGS}>")

# Offload compression ?
if(BUILD_WITH_OFFLOAD_COMPRESS)
  set_target_properties(rocgraph PROPERTIES COMPILE_FLAGS "--offload-compress")
endif()

# Target include directories
target_include_directories(
  rocgraph
  PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include/cpp>
  PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/rocgraph>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/rocgraph/cpp>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(rocgraph PRIVATE hip::device hiprand)

# Target properties
rocm_set_soversion(rocgraph ${rocgraph_SOVERSION})
set_target_properties(rocgraph PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                          VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(rocgraph PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                          "${PROJECT_BINARY_DIR}/staging")
set_target_properties(rocgraph PROPERTIES DEBUG_POSTFIX "-d")

include(FetchContent)

if(BUILD_VERBOSE)
  # This will make FetchContent verbose
  set(FETCHCONTENT_QUIET False)
endif()

set(old_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)

# TODO: The following two are *not* private for the C++ interface and are also
# exposed by dependencies like hipMM. This will be fixed in a future version by
# "vendoring" or another method.

# Import fmtlib
FetchContent_Declare(
  fmt URL https://github.com/fmtlib/fmt/archive/refs/tags/11.1.3.tar.gz)
FetchContent_MakeAvailable(fmt)

target_link_libraries(rocgraph PRIVATE fmt::fmt-header-only)

# Import spdlog
FetchContent_Declare(
  spdlog URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.1.tar.gz)
FetchContent_MakeAvailable(spdlog)

target_link_libraries(rocgraph PRIVATE spdlog::spdlog_header_only)
target_compile_definitions(rocgraph PRIVATE SPDLOG_FMT_EXTERNAL_HO)

set(BUILD_SHARED_LIBS ${old_BUILD_SHARED_LIBS})

set(DEPENDS "")

# Import ROCm libhipcxx
find_package(libhipcxx QUIET)
if(NOT libhipcxx_FOUND)
  set(ROCGRAPH_LIBHIPCXX_BRANCH release/2.2.x)
  message(STATUS "Fetching libhipcxx @ ${ROCGRAPH_LIBHIPCXX_BRANCH}")
  FetchContent_Declare(
    libhipcxx
    GIT_REPOSITORY https://github.com/ROCm/libhipcxx.git
    GIT_TAG ${ROCGRAPH_LIBHIPCXX_BRANCH}
    UPDATE_COMMAND git stash --all
    PATCH_COMMAND rm CMakeLists.txt)
  FetchContent_MakeAvailable(libhipcxx)
  add_library(libhipcxx INTERFACE)
  target_link_libraries(rocgraph PRIVATE libhipcxx)
  target_include_directories(rocgraph PRIVATE "${libhipcxx_SOURCE_DIR}/include")
else()
  target_link_libraries(rocgraph PUBLIC libhipcxx::libhipcxx)
  list(APPEND DEPENDS PACKAGE libhipcxx)
endif()

# Import ROCm hipMM
find_package(rmm QUIET)
if(NOT rmm_FOUND)
  set(ROCGRAPH_HIPMM_BRANCH release/2.0.x)
  message(STATUS "Fetching hipMM @ ${ROCGRAPH_HIPMM_BRANCH}")
  FetchContent_Declare(
    rmm
    GIT_REPOSITORY https://github.com/ROCm-DS/hipMM
    GIT_TAG ${ROCGRAPH_HIPMM_BRANCH}
    UPDATE_COMMAND git stash --all
    PATCH_COMMAND rm CMakeLists.txt)
  FetchContent_MakeAvailable(rmm)
  add_library(rmm INTERFACE)
  target_link_libraries(rocgraph PRIVATE rmm)
  target_include_directories(rocgraph PRIVATE "${rmm_SOURCE_DIR}/include")
else()
  target_link_libraries(rocgraph PUBLIC rmm::rmm)
  list(APPEND DEPENDS PACKAGE rmm)
endif()

include(CMakePrintHelpers)

# Import ROCm hipCollections
find_package(cuco QUIET)
if(NOT cuco_FOUND)
  # set(ROCGRAPH_HIPCOLLECTIONS_BRANCH amd-integration/branch-24.06)
  set(ROCGRAPH_HIPCOLLECTIONS_BRANCH release/0.2.x)
  message(STATUS "Fetching hipCollections @ ${ROCGRAPH_HIPCOLLECTIONS_BRANCH}")
  FetchContent_Declare(
    cuco
    GIT_REPOSITORY https://github.com/ROCm/hipCollections
    GIT_TAG ${ROCGRAPH_HIPCOLLECTIONS_BRANCH}
    UPDATE_COMMAND git stash --all
    PATCH_COMMAND rm CMakeLists.txt)
  FetchContent_MakeAvailable(cuco)
  add_library(cuco INTERFACE)
  target_include_directories(rocgraph PRIVATE "${cuco_SOURCE_DIR}/include")
  target_link_libraries(rocgraph PRIVATE cuco)
else()
  target_link_libraries(rocgraph PUBLIC cuco::cuco)
  list(APPEND DEPENDS PACKAGE cuco)
endif()

# Import AMD-AI/raft.
set(OVERRIDE_RAFT_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/.early-access-unreleased/raft"
    CACHE PATH "Force usage of an existing RAFT checkout.")
mark_as_advanced(OVERRIDE_RAFT_SOURCE_DIR)
if(OVERRIDE_RAFT_SOURCE_DIR)
  message(STATUS "Using raft source tree in ${OVERRIDE_RAFT_SOURCE_DIR}")
  add_library(raft INTERFACE)
  target_include_directories(rocgraph
                             PRIVATE "${OVERRIDE_RAFT_SOURCE_DIR}/cpp/include")
else()
  find_package(raft QUIET)
  if(NOT raft_FOUND)
    set(ROCGRAPH_RAFT_BRANCH dev_aiss/raft_preview_release)
    include(CheckIncludeFileCXX)
    list(APPEND CMAKE_REQUIRED_INCLUDES "${ROCM_PATH}/include")
    check_include_file_cxx(raft.hpp raft_FOUND)
    find_path(raft_includes raft.hpp PATHS "${ROCM_PATH}/include")
    if(NOT raft_includes)
      message(STATUS "Fetching raft @ ${ROCGRAPH_RAFT_BRANCH}")
      FetchContent_Declare(
        raft
        GIT_REPOSITORY https://github.com/ROCm-DS/raft
        GIT_TAG ${ROCGRAPH_RAFT_BRANCH})
      FetchContent_MakeAvailable(raft)
      add_library(raft INTERFACE)
      target_include_directories(rocgraph
                                 PRIVATE "${raft_SOURCE_DIR}/cpp/include")
    else()
      add_library(raft INTERFACE)
      target_include_directories(rocgraph PRIVATE "${raft_includes}")
    endif()
  else()
    target_link_libraries(rocgraph PRIVATE raft)
    list(APPEND DEPENDS PACKAGE raft)
  endif()
endif()

# Generate export header
include(GenerateExportHeader)
generate_export_header(rocgraph EXPORT_FILE_NAME
                       ${PROJECT_BINARY_DIR}/include/rocgraph/rocgraph-export.h)

if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_wrap_header_file(
    rocgraph-version.h
    rocgraph-export.h
    GUARDS
    SYMLINK
    WRAPPER
    WRAPPER_LOCATIONS
    ${CMAKE_INSTALL_INCLUDEDIR}
    rocgraph/${CMAKE_INSTALL_INCLUDEDIR})
endif()

if(BUILD_CODE_COVERAGE)
  target_compile_options( rocgraph PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping)
  target_link_options( rocgraph PUBLIC -fprofile-instr-generate )
endif()

execute_process(
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/library/include
    ${PROJECT_BINARY_DIR}/include/rocgraph)
# Install targets
rocm_install_targets(TARGETS rocgraph INCLUDE ${CMAKE_BINARY_DIR}/include)

# Export targets
rocm_export_targets(
  TARGETS
  roc::rocgraph
  DEPENDS
  ${DEPENDS}
  STATIC_DEPENDS
  ${static_depends}
  NAMESPACE
  roc::)
