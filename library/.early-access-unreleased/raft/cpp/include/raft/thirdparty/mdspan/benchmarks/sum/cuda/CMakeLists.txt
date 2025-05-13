# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

mdspan_add_cuda_benchmark(sum_3d_cuda)
target_include_directories(
  sum_3d_cuda PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/benchmarks/sum>)
