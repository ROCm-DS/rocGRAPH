# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

mdspan_add_benchmark(tiny_matrix_add)

if(MDSPAN_ENABLE_OPENMP)
  add_subdirectory(openmp)
endif()
