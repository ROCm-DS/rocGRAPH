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
 * @file rand_index.cuh
 * @todo TODO(Ganesh Venkataramana):
 * <pre>
 * The below rand_index calculation implementation is a Brute force one that uses
 (nElements*nElements) threads (2 dimensional grids and blocks)
 * For small datasets, this will suffice; but for larger ones, work done by the threads increase
 dramatically.
 * A more mathematically intensive implementation that uses half the above threads can be done,
 which will prove to be more efficient for larger datasets
 * the idea is as follows:
  * instead of 2D block and grid configuration with a total of (nElements*nElements) threads (where
 each (i,j) through these threads represent an ordered pair selection of 2 data points), a 1D block
 and grid configuration with a total of (nElements*(nElements))/2 threads (each thread index
 represents an element part of the set of unordered pairwise selections from the dataset (nChoose2))
  * In this setup, one has to generate a one-to-one mapping between this 1D thread index (for each
 kernel) and the unordered pair of chosen datapoints.
  * More specifically, thread0-> {dataPoint1, dataPoint0}, thread1-> {dataPoint2, dataPoint0},
 thread2-> {dataPoint2, dataPoint1} ... thread((nElements*(nElements))/2 - 1)->
 {dataPoint(nElements-1),dataPoint(nElements-2)}
  * say ,
     * threadNum: thread index | threadNum = threadIdx.x + BlockIdx.x*BlockDim.x,
     * i : index of dataPoint i
     * j : index of dataPoint j
  * then the mapping is as follows:
     * i = ceil((-1 + sqrt(1 + 8*(1 + threadNum)))/2) = floor((1 + sqrt(1 + 8*threadNum))/2)
     * j = threadNum - i(i-1)/2
  * after obtaining the the pair of datapoints, calculation of rand index is the same as done in
 this implementation
 * Caveat: since the kernel implementation involves use of emulated sqrt() operations:
  * the number of instructions executed per kernel is ~40-50 times
  * as the O(nElements*nElements) increase beyond the floating point limit, floating point
 inaccuracies occur, and hence the above floor(...) !=  ceil(...)
 * </pre>
 */

#pragma once

#include <raft/core/interruptible.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

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
 * @brief kernel to calculate the values of a and b
 * @param firstClusterArray: the array of classes of type T
 * @param secondClusterArray: the array of classes of type T
 * @param size: the size of the data points
 * @param a: number of pairs of points that both the clusters have classified the same
 * @param b: number of pairs of points that both the clusters have classified differently
 */
            template <typename T, int BLOCK_DIM_X, int BLOCK_DIM_Y>
            RAFT_KERNEL computeTheNumerator(const T*  firstClusterArray,
                                            const T*  secondClusterArray,
                                            uint64_t  size,
                                            uint64_t* a,
                                            uint64_t* b)
            {
                // calculating the indices of pairs of datapoints compared by the current thread
                uint64_t j = threadIdx.x + blockIdx.x * blockDim.x;
                uint64_t i = threadIdx.y + blockIdx.y * blockDim.y;

                // thread-local variables to count a and b
                uint64_t myA = 0, myB = 0;

                if(i < size && j < size && j < i)
                {
                    // checking if the pair have been classified the same by both the clusters
                    if(firstClusterArray[i] == firstClusterArray[j]
                       && secondClusterArray[i] == secondClusterArray[j])
                    {
                        ++myA;
                    }

                    // checking if the pair have been classified differently by both the clusters
                    else if(firstClusterArray[i] != firstClusterArray[j]
                            && secondClusterArray[i] != secondClusterArray[j])
                    {
                        ++myB;
                    }
                }

                // specialize blockReduce for a 2D block of 1024 threads of type uint64_t
                typedef cub::BlockReduce<uint64_t,
                                         BLOCK_DIM_X,
                                         cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                                         BLOCK_DIM_Y>
                    BlockReduce;

                // Allocate shared memory for blockReduce
                __shared__ typename BlockReduce::TempStorage temp_storage;

                // summing up thread-local counts specific to a block
                myA = BlockReduce(temp_storage).Sum(myA);
                __syncthreads();
                myB = BlockReduce(temp_storage).Sum(myB);
                __syncthreads();

                // executed once per block
                if(threadIdx.x == 0 && threadIdx.y == 0)
                {
                    raft::myAtomicAdd<unsigned long long int>((unsigned long long int*)a, myA);
                    raft::myAtomicAdd<unsigned long long int>((unsigned long long int*)b, myB);
                }
            }

            /**
 * @brief Function to calculate RandIndex
 * <a href="https://en.wikipedia.org/wiki/Rand_index">more info on rand index</a>
 * @param firstClusterArray: the array of classes of type T
 * @param secondClusterArray: the array of classes of type T
 * @param size: the size of the data points of type uint64_t
 * @param stream: the cudaStream object
 */
            template <typename T>
            double compute_rand_index(const T*     firstClusterArray,
                                      const T*     secondClusterArray,
                                      uint64_t     size,
                                      cudaStream_t stream)
            {
                // rand index for size less than 2 is not defined
                ASSERT(size >= 2, "Rand Index for size less than 2 not defined!");

                // allocating and initializing memory for a and b in the GPU
                rmm::device_uvector<uint64_t> arr_buf(2, stream);
                RAFT_CUDA_TRY(cudaMemsetAsync(arr_buf.data(), 0, 2 * sizeof(uint64_t), stream));

                // kernel configuration
                static const int BLOCK_DIM_Y = 16, BLOCK_DIM_X = 16;
                dim3             numThreadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
                dim3             numBlocks(raft::ceildiv<int>(size, numThreadsPerBlock.x),
                               raft::ceildiv<int>(size, numThreadsPerBlock.y));

                // calling the kernel
                computeTheNumerator<T, BLOCK_DIM_X, BLOCK_DIM_Y>
                    <<<numBlocks, numThreadsPerBlock, 0, stream>>>(firstClusterArray,
                                                                   secondClusterArray,
                                                                   size,
                                                                   arr_buf.data(),
                                                                   arr_buf.data() + 1);

                // synchronizing and updating the calculated values of a and b from device to host
                uint64_t ab_host[2] = {0};
                raft::update_host(ab_host, arr_buf.data(), 2, stream);
                raft::interruptible::synchronize(stream);

                // error handling
                RAFT_CUDA_TRY(cudaGetLastError());

                // denominator
                uint64_t nChooseTwo = size * (size - 1) / 2;

                // calculating the rand_index
                return (double)(((double)(ab_host[0] + ab_host[1])) / (double)nChooseTwo);
            }

        }; // end namespace detail
    }; // end namespace stats
}; // end namespace raft
