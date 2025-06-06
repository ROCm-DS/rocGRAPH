// Copyright (c) 2019-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/**
 * @file kl_divergence.cuh
 * @brief The KL divergence tells us how well the probability distribution Q AKA candidatePDF
 * approximates the probability distribution P AKA modelPDF.
 */

#pragma once

#include <raft/linalg/map_then_reduce.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>

#include <math.h>

namespace raft
{
    namespace stats
    {
        namespace detail
        {

            /**
 * @brief the KL Diverence mapping function
 *
 * @tparam Type: Data type of the input
 * @param modelPDF: the model probability density function of type DataT
 * @param candidatePDF: the candidate probability density function of type DataT
 */
            template <typename Type>
            struct KLDOp
            {
                HDI Type operator()(Type modelPDF, Type candidatePDF)
                {
                    if(modelPDF == 0.0)
                        return 0;

                    else
                        return modelPDF * (log(modelPDF) - log(candidatePDF));
                }
            };

            /**
 * @brief Function to calculate KL Divergence
 * <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">more info on KL
 * Divergence</a>
 *
 * @tparam DataT: Data type of the input array
 * @param modelPDF: the model array of probability density functions of type DataT
 * @param candidatePDF: the candidate array of probability density functions of type DataT
 * @param size: the size of the data points of type int
 * @param stream: the cudaStream object
 */
            template <typename DataT>
            DataT kl_divergence(const DataT* modelPDF,
                                const DataT* candidatePDF,
                                int          size,
                                cudaStream_t stream)
            {
                rmm::device_scalar<DataT> d_KLDVal(stream);
                RAFT_CUDA_TRY(cudaMemsetAsync(d_KLDVal.data(), 0, sizeof(DataT), stream));

                raft::linalg::mapThenSumReduce<DataT, KLDOp<DataT>, size_t, 256, const DataT*>(
                    d_KLDVal.data(), (size_t)size, KLDOp<DataT>(), stream, modelPDF, candidatePDF);

                DataT h_KLDVal;

                raft::update_host(&h_KLDVal, d_KLDVal.data(), 1, stream);

                raft::interruptible::synchronize(stream);

                return h_KLDVal;
            }

        }; // end namespace detail
    }; // end namespace stats
}; // end namespace raft
