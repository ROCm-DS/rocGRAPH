// Copyright (c) 2019-2023, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <limits>

namespace raft
{
    namespace stats
    {
        namespace detail
        {

            // TODO: replace with `std::bitcast` once we adopt C++20 or libcu++ adds it
            template <class To, class From>
            constexpr To bit_cast(const From& from) noexcept
            {
                To to{};
                static_assert(sizeof(To) == sizeof(From));
                memcpy(&to, &from, sizeof(To));
                return to;
            }

            template <typename T>
            struct encode_traits
            {
            };

            template <>
            struct encode_traits<float>
            {
                using E = int;
            };

            template <>
            struct encode_traits<double>
            {
                using E = long long;
            };

            HDI int encode(float val)
            {
                int i = detail::bit_cast<int>(val);
                return i >= 0 ? i : (1 << 31) | ~i;
            }

            HDI long long encode(double val)
            {
                std::int64_t i = detail::bit_cast<std::int64_t>(val);
                return i >= 0 ? i : (1ULL << 63) | ~i;
            }

            HDI float decode(int val)
            {
                if(val < 0)
                    val = (1 << 31) | ~val;
                return detail::bit_cast<float>(val);
            }

            HDI double decode(long long val)
            {
                if(val < 0)
                    val = (1ULL << 63) | ~val;
                return detail::bit_cast<double>(val);
            }

            template <typename T, typename E>
            DI T atomicMaxBits(T* address, T val)
            {
                E old = atomicMax((E*)address, encode(val));
                return decode(old);
            }

            template <typename T, typename E>
            DI T atomicMinBits(T* address, T val)
            {
                E old = atomicMin((E*)address, encode(val));
                return decode(old);
            }

            template <typename T, typename E>
            RAFT_KERNEL decodeKernel(T* globalmin, T* globalmax, int ncols)
            {
                int tid = threadIdx.x + blockIdx.x * blockDim.x;
                if(tid < ncols)
                {
                    globalmin[tid] = decode(*(E*)&globalmin[tid]);
                    globalmax[tid] = decode(*(E*)&globalmax[tid]);
                }
            }

            ///@todo: implement a proper "fill" kernel
            template <typename T, typename E>
            RAFT_KERNEL minmaxInitKernel(int ncols, T* globalmin, T* globalmax, T init_val)
            {
                int tid = threadIdx.x + blockIdx.x * blockDim.x;
                if(tid >= ncols)
                    return;
                *(E*)&globalmin[tid] = encode(init_val);
                *(E*)&globalmax[tid] = encode(-init_val);
            }

            template <typename T, typename E>
            RAFT_KERNEL minmaxKernel(const T*            data,
                                     const unsigned int* rowids,
                                     const unsigned int* colids,
                                     int                 nrows,
                                     int                 ncols,
                                     int                 row_stride,
                                     T*                  g_min,
                                     T*                  g_max,
                                     T*                  sampledcols,
                                     T                   init_min_val,
                                     int                 batch_ncols,
                                     int                 num_batches)
            {
                int                    tid = threadIdx.x + blockIdx.x * blockDim.x;
                extern __shared__ char shmem[];
                T*                     s_min = (T*)shmem;
                T*                     s_max = (T*)(shmem + sizeof(T) * batch_ncols);

                int last_batch_ncols = ncols % batch_ncols;
                if(last_batch_ncols == 0)
                {
                    last_batch_ncols = batch_ncols;
                }
                int orig_batch_ncols = batch_ncols;

                for(int batch_id = 0; batch_id < num_batches; batch_id++)
                {
                    if(batch_id == num_batches - 1)
                    {
                        batch_ncols = last_batch_ncols;
                    }

                    for(int i = threadIdx.x; i < batch_ncols; i += blockDim.x)
                    {
                        *(E*)&s_min[i] = encode(init_min_val);
                        *(E*)&s_max[i] = encode(-init_min_val);
                    }
                    __syncthreads();

                    for(int i = tid; i < nrows * batch_ncols; i += blockDim.x * gridDim.x)
                    {
                        int col = (batch_id * orig_batch_ncols) + (i / nrows);
                        int row = i % nrows;
                        if(colids != nullptr)
                        {
                            col = colids[col];
                        }
                        if(rowids != nullptr)
                        {
                            row = rowids[row];
                        }
                        int index   = row + col * row_stride;
                        T   coldata = data[index];
                        if(!isnan(coldata))
                        {
                            // Min max values are saved in shared memory and global memory as per the shuffled colids.
                            atomicMinBits<T, E>(&s_min[(int)(i / nrows)], coldata);
                            atomicMaxBits<T, E>(&s_max[(int)(i / nrows)], coldata);
                        }
                        if(sampledcols != nullptr)
                        {
                            sampledcols[batch_id * orig_batch_ncols + i] = coldata;
                        }
                    }
                    __syncthreads();

                    // finally, perform global mem atomics
                    for(int j = threadIdx.x; j < batch_ncols; j += blockDim.x)
                    {
                        atomicMinBits<T, E>(&g_min[batch_id * orig_batch_ncols + j],
                                            decode(*(E*)&s_min[j]));
                        atomicMaxBits<T, E>(&g_max[batch_id * orig_batch_ncols + j],
                                            decode(*(E*)&s_max[j]));
                    }
                    __syncthreads();
                }
            }

            /**
 * @brief Computes min/max across every column of the input matrix, as well as
 * optionally allow to subsample based on the given row/col ID mapping vectors
 *
 * @tparam T the data type
 * @tparam TPB number of threads per block
 * @param data input data
 * @param rowids actual row ID mappings. It is of length nrows. If you want to
 * skip this index lookup entirely, pass nullptr
 * @param colids actual col ID mappings. It is of length ncols. If you want to
 * skip this index lookup entirely, pass nullptr
 * @param nrows number of rows of data to be worked upon. The actual rows of the
 * input "data" can be bigger than this!
 * @param ncols number of cols of data to be worked upon. The actual cols of the
 * input "data" can be bigger than this!
 * @param row_stride stride (in number of elements) between 2 adjacent columns
 * @param globalmin final col-wise global minimum (size = ncols)
 * @param globalmax final col-wise global maximum (size = ncols)
 * @param sampledcols output sampled data. Pass nullptr if you don't need this
 * @param stream cuda stream
 * @note This method makes the following assumptions:
 * 1. input and output matrices are assumed to be col-major
 * 2. ncols is small enough to fit the whole of min/max values across all cols
 *    in shared memory
 */
            template <typename T, int TPB = 512>
            void minmax(const T*        data,
                        const unsigned* rowids,
                        const unsigned* colids,
                        int             nrows,
                        int             ncols,
                        int             row_stride,
                        T*              globalmin,
                        T*              globalmax,
                        T*              sampledcols,
                        cudaStream_t    stream)
            {
                using E      = typename encode_traits<T>::E;
                int nblks    = raft::ceildiv(ncols, TPB);
                T   init_val = std::numeric_limits<T>::max();
                minmaxInitKernel<T, E>
                    <<<nblks, TPB, 0, stream>>>(ncols, globalmin, globalmax, init_val);
                RAFT_CUDA_TRY(cudaPeekAtLastError());
                nblks           = raft::ceildiv(nrows * ncols, TPB);
                nblks           = min(nblks, 65536);
                size_t smemSize = sizeof(T) * 2 * ncols;

                // Compute the batch_ncols, in [1, ncols] range, that meet the available
                // shared memory constraints.
                auto smemPerBlk  = raft::getSharedMemPerBlock();
                int  batch_ncols = min(ncols, (int)(smemPerBlk / (sizeof(T) * 2)));
                int  num_batches = raft::ceildiv(ncols, batch_ncols);
                smemSize         = sizeof(T) * 2 * batch_ncols;

                minmaxKernel<T, E><<<nblks, TPB, smemSize, stream>>>(data,
                                                                     rowids,
                                                                     colids,
                                                                     nrows,
                                                                     ncols,
                                                                     row_stride,
                                                                     globalmin,
                                                                     globalmax,
                                                                     sampledcols,
                                                                     init_val,
                                                                     batch_ncols,
                                                                     num_batches);
                RAFT_CUDA_TRY(cudaPeekAtLastError());
                decodeKernel<T, E><<<nblks, TPB, 0, stream>>>(globalmin, globalmax, ncols);
                RAFT_CUDA_TRY(cudaPeekAtLastError());
            }

        }; // end namespace detail
    }; // end namespace stats
}; // end namespace raft
