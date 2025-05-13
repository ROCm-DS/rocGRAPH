# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

mdspan_add_openmp_benchmark(matvec_openmp)
if(OpenMP_CXX_FOUND)
  target_include_directories(
    matvec_openmp
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/benchmarks/matvec>)
endif()
