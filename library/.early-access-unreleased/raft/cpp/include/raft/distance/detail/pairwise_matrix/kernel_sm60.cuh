// Copyright (c) 2023-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/*
 * Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#pragma once

#include <raft/core/operators.hpp> // raft::void_op
#include <raft/distance/detail/pairwise_distance_base.cuh> // PairwiseDistances
#include <raft/distance/detail/pairwise_matrix/params.cuh> // pairwise_matrix_params
#include <raft/util/arch.cuh> // raft::util::arch::SM_compute_arch

#include <cassert> // assert

namespace raft::distance::detail
{

    template <typename Policy,
              bool row_major,
              typename SM_compat_t,
              typename OpT,
              typename IdxT,
              typename DataT,
              typename OutT,
              typename FinOpT>
    __launch_bounds__(Policy::Nthreads, 2) RAFT_KERNEL
        pairwise_matrix_kernel(OpT                                               distance_op,
                               pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> params)
    {
#ifdef __HIP_PLATFORM_AMD__
// TODO(HIP/AMD): Add early exit call
#else
        // Early exit to minimize the size of the kernel when it is not supposed to be compiled.
        constexpr SM_compat_t sm_compat_range{};
        if constexpr(!sm_compat_range.contains(raft::util::arch::SM_compute_arch()))
        {
            assert(false);
            return;
        }
#endif

        extern __shared__ char smem[];

        // The epilog is already provided by distance_op. Do not provide additional
        // epilogs.
        auto epilog_op = raft::void_op();
        // No support for row_epilog_op.
        auto row_epilog_op = raft::void_op();

        // Always write output
        constexpr bool write_out = true;
        constexpr bool use_norms = distance_op.use_norms;
        PairwiseDistances<DataT,
                          OutT,
                          IdxT,
                          Policy,
                          decltype(distance_op),
                          decltype(epilog_op),
                          decltype(params.fin_op),
                          decltype(row_epilog_op),
                          row_major,
                          write_out>
            obj(params.x,
                params.y,
                params.m,
                params.n,
                params.k,
                params.ldx,
                params.ldy,
                params.ld_out,
                params.x_norm,
                params.y_norm,
                params.out,
                smem,
                distance_op,
                epilog_op,
                params.fin_op,
                row_epilog_op);
        obj.run();
    }

    // The type of a pointer to the pairwise matrix kernel. The following template
    // arguments are type-erased:
    //
    // - The kernel policy
    // - row_major
    // - SM_compat_t
    template <typename OpT, typename IdxT, typename DataT, typename OutT, typename FinOpT>
    using pairwise_matrix_kernel_t
        = void (*)(OpT, pairwise_matrix_params<IdxT, DataT, OutT, FinOpT>);

    // A wrapper for the pairwise matrix kernel launch. Includes kernel launch
    // parameters.
    template <typename OpT, typename IdxT, typename DataT, typename OutT, typename FinOpT>
    struct pairwise_matrix_sm60_wrapper
    {
        dim3                                                     grid;
        dim3                                                     block;
        int                                                      smem_size;
        pairwise_matrix_kernel_t<OpT, IdxT, DataT, OutT, FinOpT> kernel_ptr;

        void launch(OpT                                               distance_op,
                    pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> params,
                    cudaStream_t                                      stream)
        {
            kernel_ptr<<<grid, block, smem_size, stream>>>(distance_op, params);
            RAFT_CUDA_TRY(cudaGetLastError());
        }
    };

    /** @brief: Create kernel launch wrapper for pairwise matrix kernel
 *
 * This can be used to type-erase the kernel execution policy, row_major, and SM
 * compatibility range.
 *
 * @tparam Policy: Kernel execution policy
 * @tparam row_major: Indicates whether input matrices are row major
 * @tparam OpT: Type of distance operation
 * @tparam IdxT: Index type
 * @tparam DataT: Data type
 * @tparam OutT: Output data type
 * @tparam FinOpT: Final operation type
 * @tparam SM_compat_t: Type of the SM architecture compatibility
 *
 * @param distance_op: Distance operation
 * @param params: Parameters
 * @param sm_compat_range: Which SM architectures to compile for.
 */
    template <typename Policy,
              bool row_major,
              typename OpT,
              typename IdxT,
              typename DataT,
              typename OutT,
              typename FinOpT,
              typename SM_compat_t>
    pairwise_matrix_sm60_wrapper<OpT, IdxT, DataT, OutT, FinOpT>
        make_pairwise_matrix_sm60_wrapper(OpT distance_op,
                                          pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> params,
                                          SM_compat_t sm_compat_range)
    {
        dim3 block(Policy::Nthreads);
        // Use ::template to disambiguate (See:
        // https://en.cppreference.com/w/cpp/language/dependent_name)
        int smem_size = OpT::template shared_mem_size<Policy>();
        // Obtain function pointer to kernel
        auto kernel = pairwise_matrix_kernel<Policy,
                                             row_major,
                                             SM_compat_t,
                                             OpT,
                                             IdxT,
                                             DataT,
                                             OutT,
                                             FinOpT>;
        dim3 grid   = launchConfigGenerator<Policy>(params.m, params.n, smem_size, kernel);

        return pairwise_matrix_sm60_wrapper<OpT, IdxT, DataT, OutT, FinOpT>{
            grid, block, smem_size, kernel};
    }

}; // namespace raft::distance::detail
