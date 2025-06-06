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

#pragma once

#include <raft/core/interruptible.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif

#include <memory>

namespace raft
{
    namespace stats
    {
        namespace detail
        {

            ///@todo: ColsPerBlk has been tested only for 32!
            template <typename DataT, typename IdxT, int TPB, int ColsPerBlk = 32>
            RAFT_KERNEL
                weightedMeanKernel(DataT* mu, const DataT* data, const IdxT* counts, IdxT D, IdxT N)
            {
                constexpr int    RowsPerBlkPerIter = TPB / ColsPerBlk;
                IdxT             thisColId         = threadIdx.x % ColsPerBlk;
                IdxT             thisRowId         = threadIdx.x / ColsPerBlk;
                IdxT             colId             = thisColId + ((IdxT)blockIdx.y * ColsPerBlk);
                IdxT             rowId       = thisRowId + ((IdxT)blockIdx.x * RowsPerBlkPerIter);
                DataT            thread_data = DataT(0);
                const IdxT       stride      = RowsPerBlkPerIter * gridDim.x;
                __shared__ DataT smu[ColsPerBlk];
                if(threadIdx.x < ColsPerBlk)
                    smu[threadIdx.x] = DataT(0);
                for(IdxT i = rowId; i < N; i += stride)
                {
                    thread_data += (colId < D) ? data[i * D + colId] * (DataT)counts[i] : DataT(0);
                }
                __syncthreads();
                raft::myAtomicAdd(smu + thisColId, thread_data);
                __syncthreads();
                if(threadIdx.x < ColsPerBlk && colId < D)
                    raft::myAtomicAdd(mu + colId, smu[thisColId]);
            }

            template <typename DataT, typename IdxT, int TPB>
            RAFT_KERNEL dispersionKernel(DataT*       result,
                                         const DataT* clusters,
                                         const IdxT*  clusterSizes,
                                         const DataT* mu,
                                         IdxT         dim,
                                         IdxT         nClusters)
            {
                IdxT  tid    = threadIdx.x + blockIdx.x * blockDim.x;
                IdxT  len    = dim * nClusters;
                IdxT  stride = blockDim.x * gridDim.x;
                DataT sum    = DataT(0);
                for(; tid < len; tid += stride)
                {
                    IdxT  col  = tid % dim;
                    IdxT  row  = tid / dim;
                    DataT diff = clusters[tid] - mu[col];
                    sum += diff * diff * DataT(clusterSizes[row]);
                }
                typedef cub::BlockReduce<DataT, TPB>         BlockReduce;
                __shared__ typename BlockReduce::TempStorage temp_storage;
                __syncthreads();
                auto acc = BlockReduce(temp_storage).Sum(sum);
                __syncthreads();
                if(threadIdx.x == 0)
                    raft::myAtomicAdd(result, acc);
            }

            /**
 * @brief Compute cluster dispersion metric. This is very useful for
 * automatically finding the 'k' (in kmeans) that improves this metric.
 * @tparam DataT data type
 * @tparam IdxT index type
 * @tparam TPB threads block for kernels launched
 * @param centroids the cluster centroids. This is assumed to be row-major
 *   and of dimension (nClusters x dim)
 * @param clusterSizes number of points in the dataset which belong to each
 *   cluster. This is of length nClusters
 * @param globalCentroid compute the global weighted centroid of all cluster
 *   centroids. This is of length dim. Pass a nullptr if this is not needed
 * @param nClusters number of clusters
 * @param nPoints number of points in the dataset
 * @param dim dataset dimensionality
 * @param stream cuda stream
 * @return the cluster dispersion value
 */
            template <typename DataT, typename IdxT = int, int TPB = 256>
            DataT dispersion(const DataT* centroids,
                             const IdxT*  clusterSizes,
                             DataT*       globalCentroid,
                             IdxT         nClusters,
                             IdxT         nPoints,
                             IdxT         dim,
                             cudaStream_t stream)
            {
                static const int           RowsPerThread = 4;
                static const int           ColsPerBlk    = 32;
                static const int           RowsPerBlk    = (TPB / ColsPerBlk) * RowsPerThread;
                dim3                       grid(raft::ceildiv(nPoints, (IdxT)RowsPerBlk),
                          raft::ceildiv(dim, (IdxT)ColsPerBlk));
                rmm::device_uvector<DataT> mean(0, stream);
                rmm::device_uvector<DataT> result(1, stream);
                DataT*                     mu = globalCentroid;
                if(globalCentroid == nullptr)
                {
                    mean.resize(dim, stream);
                    mu = mean.data();
                }
                RAFT_CUDA_TRY(cudaMemsetAsync(mu, 0, sizeof(DataT) * dim, stream));
                RAFT_CUDA_TRY(cudaMemsetAsync(result.data(), 0, sizeof(DataT), stream));
                weightedMeanKernel<DataT, IdxT, TPB, ColsPerBlk>
                    <<<grid, TPB, 0, stream>>>(mu, centroids, clusterSizes, dim, nClusters);
                RAFT_CUDA_TRY(cudaGetLastError());
                DataT ratio = DataT(1) / DataT(nPoints);
                raft::linalg::scalarMultiply(mu, mu, ratio, dim, stream);
                // finally, compute the dispersion
                constexpr int ItemsPerThread = 4;
                int           nblks = raft::ceildiv<int>(dim * nClusters, TPB * ItemsPerThread);
                dispersionKernel<DataT, IdxT, TPB><<<nblks, TPB, 0, stream>>>(
                    result.data(), centroids, clusterSizes, mu, dim, nClusters);
                RAFT_CUDA_TRY(cudaGetLastError());
                DataT h_result;
                raft::update_host(&h_result, result.data(), 1, stream);
                raft::interruptible::synchronize(stream);
                return sqrt(h_result);
            }

        } // end namespace detail
    } // end namespace stats
} // end namespace raft
