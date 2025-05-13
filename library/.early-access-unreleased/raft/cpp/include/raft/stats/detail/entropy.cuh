// Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
/**
 * @file entropy.cuh
 * @brief Calculates the entropy for a labeling in nats.(ie, uses natural logarithm for the
 * calculations)
 */

#pragma once
#include <raft/linalg/divide.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif

#include <math.h>

namespace raft
{
    namespace stats
    {
        namespace detail
        {

            /**
 * @brief Lambda to calculate the entropy of a sample given its probability value
 *
 * @param p: the input to the functional mapping
 * @param q: dummy param
 */
            struct entropyOp
            {
                HDI double operator()(double p, double q)
                {
                    if(p)
                        return -1 * (p) * (log(p));
                    else
                        return 0.0;
                }
            };

            /**
 * @brief function to calculate the bincounts of number of samples in every label
 *
 * @tparam LabelT: type of the labels
 * @param labels: the pointer to the array containing labels for every data sample
 * @param binCountArray: pointer to the 1D array that contains the count of samples per cluster
 * @param nRows: number of data samples
 * @param lowerLabelRange
 * @param upperLabelRange
 * @param workspace: device buffer containing workspace memory
 * @param stream: the cuda stream where to launch this kernel
 */
            template <typename LabelT>
            void countLabels(const LabelT*              labels,
                             double*                    binCountArray,
                             int                        nRows,
                             LabelT                     lowerLabelRange,
                             LabelT                     upperLabelRange,
                             rmm::device_uvector<char>& workspace,
                             cudaStream_t               stream)
            {
                int    num_levels         = upperLabelRange - lowerLabelRange + 2;
                LabelT lower_level        = lowerLabelRange;
                LabelT upper_level        = upperLabelRange + 1;
                size_t temp_storage_bytes = 0;

                RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(nullptr,
                                                                  temp_storage_bytes,
                                                                  labels,
                                                                  binCountArray,
                                                                  num_levels,
                                                                  lower_level,
                                                                  upper_level,
                                                                  nRows,
                                                                  stream));

                workspace.resize(temp_storage_bytes, stream);

                RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(workspace.data(),
                                                                  temp_storage_bytes,
                                                                  labels,
                                                                  binCountArray,
                                                                  num_levels,
                                                                  lower_level,
                                                                  upper_level,
                                                                  nRows,
                                                                  stream));
            }

            /**
 * @brief Function to calculate entropy
 * <a href="https://en.wikipedia.org/wiki/Entropy_(information_theory)">more info on entropy</a>
 *
 * @param clusterArray: the array of classes of type T
 * @param size: the size of the data points of type int
 * @param lowerLabelRange: the lower bound of the range of labels
 * @param upperLabelRange: the upper bound of the range of labels
 * @param stream: the cudaStream object
 * @return the entropy score
 */
            template <typename T>
            double entropy(const T*     clusterArray,
                           const int    size,
                           const T      lowerLabelRange,
                           const T      upperLabelRange,
                           cudaStream_t stream)
            {
                if(!size)
                    return 1.0;

                T numUniqueClasses = upperLabelRange - lowerLabelRange + 1;

                // declaring, allocating and initializing memory for bincount array and entropy values
                rmm::device_uvector<double> prob(numUniqueClasses, stream);
                RAFT_CUDA_TRY(
                    cudaMemsetAsync(prob.data(), 0, numUniqueClasses * sizeof(double), stream));
                rmm::device_scalar<double> d_entropy(stream);
                RAFT_CUDA_TRY(cudaMemsetAsync(d_entropy.data(), 0, sizeof(double), stream));

                // workspace allocation
                rmm::device_uvector<char> workspace(1, stream);

                // calculating the bincounts and populating the prob array
                countLabels(clusterArray,
                            prob.data(),
                            size,
                            lowerLabelRange,
                            upperLabelRange,
                            workspace,
                            stream);

                // scalar dividing by size
                raft::linalg::divideScalar<double>(
                    prob.data(), prob.data(), (double)size, numUniqueClasses, stream);

                // calculating the aggregate entropy
                raft::linalg::mapThenSumReduce<double, entropyOp>(d_entropy.data(),
                                                                  numUniqueClasses,
                                                                  entropyOp(),
                                                                  stream,
                                                                  prob.data(),
                                                                  prob.data());

                // updating in the host memory
                double h_entropy;
                raft::update_host(&h_entropy, d_entropy.data(), 1, stream);

                raft::interruptible::synchronize(stream);

                return h_entropy;
            }

        }; // end namespace detail
    }; // end namespace stats
}; // end namespace raft
