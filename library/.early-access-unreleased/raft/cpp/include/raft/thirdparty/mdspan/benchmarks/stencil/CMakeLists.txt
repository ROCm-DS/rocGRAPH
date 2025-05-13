# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

mdspan_add_benchmark(stencil_3d)

if(MDSPAN_ENABLE_CUDA)
  add_subdirectory(cuda)
endif()

if(MDSPAN_ENABLE_OPENMP)
  add_subdirectory(openmp)
endif()
