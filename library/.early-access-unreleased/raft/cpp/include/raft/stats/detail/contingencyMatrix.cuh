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

#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif

#include <raft/thrust_execution_policy.h>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

#include <math.h>

namespace raft
{
    namespace stats
    {
        namespace detail
        {

            typedef enum
            {
                IMPL_NONE,
                SMEM_ATOMICS,
                GLOBAL_ATOMICS,
                SORT_AND_GATOMICS
            } ContingencyMatrixImplType;

            template <typename T, typename OutT = int>
            RAFT_KERNEL devConstructContingencyMatrix(const T* groundTruth,
                                                      const T* predicted,
                                                      int      nSamples,
                                                      OutT*    outMat,
                                                      int      outIdxOffset,
                                                      int      outMatWidth)
            {
                int elementId = threadIdx.x + blockDim.x * blockIdx.x;
                if(elementId < nSamples)
                {
                    T    gt        = groundTruth[elementId];
                    T    pd        = predicted[elementId];
                    auto outputIdx = (gt - outIdxOffset) * outMatWidth + pd - outIdxOffset;
                    raft::myAtomicAdd(outMat + outputIdx, OutT(1));
                }
            }

            template <typename T, typename OutT = int>
            void computeCMatWAtomics(const T*     groundTruth,
                                     const T*     predictedLabel,
                                     int          nSamples,
                                     OutT*        outMat,
                                     int          outIdxOffset,
                                     int          outDimN,
                                     cudaStream_t stream)
            {
                RAFT_CUDA_TRY(cudaFuncSetCacheConfig(
                    reinterpret_cast<const void*>(devConstructContingencyMatrix<T, OutT>),
                    cudaFuncCachePreferL1));
                static const int block = 128;
                auto             grid  = raft::ceildiv(nSamples, block);
                devConstructContingencyMatrix<T, OutT><<<grid, block, 0, stream>>>(
                    groundTruth, predictedLabel, nSamples, outMat, outIdxOffset, outDimN);
                RAFT_CUDA_TRY(cudaGetLastError());
            }

            template <typename T, typename OutT = int>
            RAFT_KERNEL devConstructContingencyMatrixSmem(const T* groundTruth,
                                                          const T* predicted,
                                                          int      nSamples,
                                                          OutT*    outMat,
                                                          int      outIdxOffset,
                                                          int      outMatWidth)
            {
                extern __shared__ char smem[];
                auto*                  sMemMatrix = reinterpret_cast<OutT*>(smem);
                for(int smemIdx = threadIdx.x; smemIdx < outMatWidth * outMatWidth;
                    smemIdx += blockDim.x)
                {
                    sMemMatrix[smemIdx] = 0;
                }
                __syncthreads();
                int elementId = threadIdx.x + blockDim.x * blockIdx.x;
                if(elementId < nSamples)
                {
                    T    gt        = groundTruth[elementId];
                    T    pd        = predicted[elementId];
                    auto outputIdx = (gt - outIdxOffset) * outMatWidth + pd - outIdxOffset;
                    raft::myAtomicAdd(sMemMatrix + outputIdx, OutT(1));
                }
                __syncthreads();
                for(int smemIdx = threadIdx.x; smemIdx < outMatWidth * outMatWidth;
                    smemIdx += blockDim.x)
                {
                    raft::myAtomicAdd(outMat + smemIdx, sMemMatrix[smemIdx]);
                }
            }

            template <typename T, typename OutT = int>
            void computeCMatWSmemAtomics(const T*     groundTruth,
                                         const T*     predictedLabel,
                                         int          nSamples,
                                         OutT*        outMat,
                                         int          outIdxOffset,
                                         int          outDimN,
                                         cudaStream_t stream)
            {
                static const int block            = 128;
                auto             grid             = raft::ceildiv(nSamples, block);
                size_t           smemSizePerBlock = outDimN * outDimN * sizeof(OutT);
                devConstructContingencyMatrixSmem<T, OutT>
                    <<<grid, block, smemSizePerBlock, stream>>>(
                        groundTruth, predictedLabel, nSamples, outMat, outIdxOffset, outDimN);
                RAFT_CUDA_TRY(cudaGetLastError());
            }

            template <typename T, typename OutT = int>
            void contingencyMatrixWSort(const T*     groundTruth,
                                        const T*     predictedLabel,
                                        int          nSamples,
                                        OutT*        outMat,
                                        T            minLabel,
                                        T            maxLabel,
                                        void*        workspace,
                                        size_t       workspaceSize,
                                        cudaStream_t stream)
            {
                T*    outKeys         = reinterpret_cast<T*>(workspace);
                auto  alignedBufferSz = raft::alignTo<size_t>(nSamples * sizeof(T), 256);
                T*    outValue        = reinterpret_cast<T*>((size_t)workspace + alignedBufferSz);
                void* pWorkspaceCub
                    = reinterpret_cast<void*>((size_t)workspace + 2 * alignedBufferSz);
                auto bitsToSort = log2<int>(maxLabel);
                if(!raft::isPo2(maxLabel))
                    ++bitsToSort;
                // we dont really need perfect sorting, should get by with some sort of
                // binning-reordering operation
                ///@todo: future work - explore "efficient" custom binning kernels vs cub sort
                RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(pWorkspaceCub,
                                                              workspaceSize,
                                                              groundTruth,
                                                              outKeys,
                                                              predictedLabel,
                                                              outValue,
                                                              nSamples,
                                                              0,
                                                              bitsToSort,
                                                              stream));
                auto outDimM_N = int(maxLabel - minLabel + 1);
                computeCMatWAtomics<T, OutT>(
                    outKeys, outValue, nSamples, outMat, minLabel, outDimM_N, stream);
            }

            template <typename OutT = int>
            ContingencyMatrixImplType getImplVersion(OutT outDimN)
            {
                int currDevice  = 0;
                int l2CacheSize = 0;
                // no way to query this from CUDA APIs, value for CC 7.0, 3.0
                int maxBlocksResidentPerSM = 16;
                RAFT_CUDA_TRY(cudaGetDevice(&currDevice));
                RAFT_CUDA_TRY(
                    cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, currDevice));
                auto                      maxSmemPerBlock = raft::getSharedMemPerBlock();
                ContingencyMatrixImplType implVersion     = IMPL_NONE;
                // keeping 8 block per SM to get good utilization
                // can go higher but reduced L1 size degrades perf
                OutT upperLimitSmemAtomics = std::floor(
                    std::sqrt(maxSmemPerBlock / (sizeof(OutT) * (maxBlocksResidentPerSM / 2))));
                OutT upperLimitL2Atomics = std::floor(std::sqrt(l2CacheSize / sizeof(OutT)));
                if(outDimN <= upperLimitSmemAtomics)
                    implVersion = SMEM_ATOMICS;
                else if(outDimN <= upperLimitL2Atomics)
                    implVersion = GLOBAL_ATOMICS;
                else
                    implVersion = SORT_AND_GATOMICS;
                return implVersion;
            }

            /**
 * @brief use this to allocate output matrix size
 * size of matrix = (maxLabel - minLabel + 1)^2 * sizeof(int)
 * @param groundTruth: device 1-d array for ground truth (num of rows)
 * @param nSamples: number of elements in input array
 * @param stream: cuda stream for execution
 * @param minLabel: [out] calculated min value in input array
 * @param maxLabel: [out] calculated max value in input array
 */
            template <typename T>
            void getInputClassCardinality(const T*     groundTruth,
                                          const int    nSamples,
                                          cudaStream_t stream,
                                          T&           minLabel,
                                          T&           maxLabel)
            {
                thrust::device_ptr<const T> dTrueLabel = thrust::device_pointer_cast(groundTruth);
                auto                        min_max    = thrust::minmax_element(
                    THRUST_EXECUTION_POLICY.on(stream), dTrueLabel, dTrueLabel + nSamples);
                minLabel = *min_max.first;
                maxLabel = *min_max.second;
            }

            /**
 * @brief Calculate workspace size for running contingency matrix calculations
 * @tparam T label type
 * @tparam OutT output matrix type
 * @param nSamples: number of elements in input array
 * @param groundTruth: device 1-d array for ground truth (num of rows)
 * @param stream: cuda stream for execution
 * @param minLabel: Optional, min value in input array
 * @param maxLabel: Optional, max value in input array
 */
            template <typename T, typename OutT = int>
            size_t getContingencyMatrixWorkspaceSize(int          nSamples,
                                                     const T*     groundTruth,
                                                     cudaStream_t stream,
                                                     T minLabel = std::numeric_limits<T>::max(),
                                                     T maxLabel = std::numeric_limits<T>::max())
            {
                size_t workspaceSize = 0;
                // below is a redundant computation - can be avoided
                if(minLabel == std::numeric_limits<T>::max()
                   || maxLabel == std::numeric_limits<T>::max())
                {
                    getInputClassCardinality<T>(groundTruth, nSamples, stream, minLabel, maxLabel);
                }
                auto                      outDimN     = OutT(maxLabel - minLabel + 1);
                ContingencyMatrixImplType implVersion = getImplVersion<OutT>(outDimN);
                if(implVersion == SORT_AND_GATOMICS)
                {
                    void*  pWorkspaceCub{};
                    size_t tmpStorageBytes = 0;
                    // no-op pointers to get workspace size
                    T* pTmpUnused{};
                    RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(pWorkspaceCub,
                                                                  tmpStorageBytes,
                                                                  pTmpUnused,
                                                                  pTmpUnused,
                                                                  pTmpUnused,
                                                                  pTmpUnused,
                                                                  nSamples));
                    auto tmpStagingMemorySize = raft::alignTo<size_t>(nSamples * sizeof(T), 256);
                    tmpStagingMemorySize *= 2;
                    workspaceSize = tmpStagingMemorySize + tmpStorageBytes;
                }
                return workspaceSize;
            }

            /**
 * @brief construct contingency matrix given input ground truth and prediction
 *        labels. Users should call function getInputClassCardinality to find
 *        and allocate memory for output. Similarly workspace requirements
 *        should be checked using function getContingencyMatrixWorkspaceSize
 * @tparam T label type
 * @tparam OutT output matrix type
 * @param groundTruth: device 1-d array for ground truth (num of rows)
 * @param predictedLabel: device 1-d array for prediction (num of columns)
 * @param nSamples: number of elements in input array
 * @param outMat: output buffer for contingecy matrix
 * @param stream: cuda stream for execution
 * @param workspace: Optional, workspace memory allocation
 * @param workspaceSize: Optional, size of workspace memory
 * @param minLabel: Optional, min value in input ground truth array
 * @param maxLabel: Optional, max value in input ground truth array
 */
            template <typename T, typename OutT = int>
            void contingencyMatrix(const T*     groundTruth,
                                   const T*     predictedLabel,
                                   int          nSamples,
                                   OutT*        outMat,
                                   cudaStream_t stream,
                                   void*        workspace     = nullptr,
                                   size_t       workspaceSize = 0,
                                   T            minLabel      = std::numeric_limits<T>::max(),
                                   T            maxLabel      = std::numeric_limits<T>::max())
            {
                // assumptions:
                // output is not at par with scikit learn - output will be square matrix
                // always with numRows = numColumns = numOfClassesInTrueLabel
                // it is also assumed that true labels are monotically increasing
                // if for some reason groundTruth completely skips some labels
                // eg: {0,1,2,5} instead of {0,1,2,3}.
                // Output matrix will still have empty rows for label value {3,4}
                // Users can use "make_monotonic" to convert their discontinuous input label
                // range to a monotonically increasing one  //
                // this also serves as way to measure co-occurrence/joint counts for NLP tasks which
                // can be used to then compute pointwise mutual information and mutual information
                if(minLabel == std::numeric_limits<T>::max()
                   || maxLabel == std::numeric_limits<T>::max())
                {
                    getInputClassCardinality<T>(groundTruth, nSamples, stream, minLabel, maxLabel);
                }
                auto outDimM_N = OutT(maxLabel - minLabel + 1);
                RAFT_CUDA_TRY(
                    cudaMemsetAsync(outMat, 0, sizeof(OutT) * outDimM_N * outDimM_N, stream));
                ContingencyMatrixImplType implVersion = getImplVersion<OutT>(outDimM_N);
                switch(implVersion)
                {
                case SMEM_ATOMICS:
                    // smem atomics and then single global mem atomics only works
                    // when all label count can fit in smem for a block
                    // helps when GLOBAL_ATOMICS performance blocked by atomic update
                    // serialization -when very less labels ~10 labels
                    computeCMatWSmemAtomics<T, OutT>(
                        groundTruth, predictedLabel, nSamples, outMat, minLabel, outDimM_N, stream);
                    break;
                case GLOBAL_ATOMICS:
                    // launch kernel - global atomic ops per (groundTruth,predictedValue) pair
                    computeCMatWAtomics<T, OutT>(
                        groundTruth, predictedLabel, nSamples, outMat, minLabel, outDimM_N, stream);
                    break;
                    // more L2 thrashing if atomic OPs land in completely different mem
                    // segment - when more labels
                case SORT_AND_GATOMICS:
                    contingencyMatrixWSort<T, OutT>(groundTruth,
                                                    predictedLabel,
                                                    nSamples,
                                                    outMat,
                                                    minLabel,
                                                    maxLabel,
                                                    workspace,
                                                    workspaceSize,
                                                    stream);
                    break;
                case IMPL_NONE:
                    break;
                }
            }

        }; // namespace detail
    }; // namespace stats
}; // namespace raft
