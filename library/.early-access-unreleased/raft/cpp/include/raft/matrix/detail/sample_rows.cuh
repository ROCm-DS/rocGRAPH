// Copyright (c) 2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/gather.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft::matrix::detail
{

    /** Select rows randomly from input and copy to output. */
    template <typename T, typename IdxT = int64_t>
    void sample_rows(raft::resources const&            res,
                     random::RngState                  random_state,
                     const T*                          input,
                     IdxT                              n_rows_input,
                     raft::device_matrix_view<T, IdxT> output)
    {
        IdxT n_dim     = output.extent(1);
        IdxT n_samples = output.extent(0);

        raft::device_vector<IdxT, IdxT> train_indices
            = raft::random::excess_subsample<IdxT, int64_t>(
                res, random_state, n_rows_input, n_samples);

        cudaPointerAttributes attr;
        RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, input));
        T* ptr = reinterpret_cast<T*>(attr.devicePointer);
        if(ptr != nullptr)
        {
            raft::matrix::gather(
                res,
                raft::make_device_matrix_view<const T, IdxT>(ptr, n_rows_input, n_dim),
                raft::make_const_mdspan(train_indices.view()),
                output);
        }
        else
        {
            auto dataset = raft::make_host_matrix_view<const T, IdxT>(input, n_rows_input, n_dim);
            raft::matrix::detail::gather(
                res, dataset, make_const_mdspan(train_indices.view()), output);
        }
    }
} // namespace raft::matrix::detail
