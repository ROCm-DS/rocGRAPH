/*
 * SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef AMDGPU_WARP_PRIMITIVES_H
#define AMDGPU_WARP_PRIMITIVES_H

#include <raft/util/bitwise_operations.hpp>

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

namespace hip_warp_primitives
{

    inline constexpr __device__ auto WAVEFRONT_SIZE = rocprim::device_warp_size();
    static_assert(WAVEFRONT_SIZE == 32 || WAVEFRONT_SIZE == 64, "Unsupported WAVEFRONT_SIZE");

    __device__ inline lane_mask __activemask(lane_mask mask)
    {
        return __ballot(1) & mask;
    }

    __device__ inline lane_mask __branchmask()
    {
        return __ballot(1);
    }

    __device__ inline bool __is_thread_in_mask(lane_mask mask)
    {
        return mask & (1LLU << __lane_id()) ? 1 : 0;
    }

    __device__ inline bool __is_thread_in_mask(lane_mask mask, unsigned int i)
    {
        return mask & (1LLU << i) ? 1 : 0;
    }

    __device__ inline int __thread_rank(lane_mask mask)
    {
        /* calling thread must be set in the mask */
#ifndef WARP_NO_MASK_CHECK
        assert(__is_thread_in_mask(mask));
#else
        /* to make compiler happy */
        (void)mask;
#endif

        return cooperative_groups::internal::coalesced_group::masked_bit_count(mask, 0);
    }

    __device__ inline unsigned int __mask_size(lane_mask mask)
    {
        return raft::__POPC(mask);
    }

    __device__ inline int __thread_rank_to_lane_id(lane_mask mask, int i)
    {
        int size = __mask_size(mask);

        if(i < 0 || i >= size)
            return -1;

        return (size == WAVEFRONT_SIZE) ? i
               : (WAVEFRONT_SIZE == 64) ? __fns64(mask, 0, (i + 1))
                                        : __fns32(mask, 0, (i + 1));
    }

    /* sync active threads inside a warp / wavefront */
    __device__ inline void __sync_active_threads()
    {
        /* sync/barrier all threads in a warp or a branch */
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }

    __device__ inline void __syncwarp()
    {
        /* sync/barrier all threads in a warp */
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }

    __device__ inline int __all_sync(lane_mask mask, int predicate)
    {
        /* calling thread must be set in the mask */
#ifndef WARP_NO_MASK_CHECK
        assert(__is_thread_in_mask(mask));
#else
        /* to make compiler happy */
        (void)mask;
#endif

        return ((__branchmask() & mask) == __ballot(predicate)) ? 1 : 0;
    }

    __device__ inline int __any_sync(lane_mask mask, int predicate)
    {
        /* calling thread must be set in the mask */
#ifndef WARP_NO_MASK_CHECK
        assert(__is_thread_in_mask(mask));
#else
        /* to make compiler happy */
        (void)mask;
#endif

        return (__ballot(predicate) & mask) ? 1 : 0;
    }

    __device__ inline lane_mask __ballot_sync(lane_mask mask, int predicate)
    {
        /* calling thread must be set in the mask */
#ifndef WARP_NO_MASK_CHECK
        assert(__is_thread_in_mask(mask));
#else
        /* to make compiler happy */
        (void)mask;
#endif

        return __ballot(predicate) & mask;
    }

    template <class T>
    __device__ inline T __shfl_sync(lane_mask mask, T var, int src, int width = WAVEFRONT_SIZE)
    {
        /* calling thread must be set in the mask */
#ifndef WARP_NO_MASK_CHECK
        assert(__is_thread_in_mask(mask));
#else
        /* to make compiler happy */
        (void)mask;
#endif

        return __shfl(var, src, width);
    }

    template <class T>
    __device__ inline T
        __shfl_down_sync(lane_mask mask, T var, unsigned int lane_delta, int width = WAVEFRONT_SIZE)
    {
        /* calling thread must be set in the mask */
#ifndef WARP_NO_MASK_CHECK
        assert(__is_thread_in_mask(mask));
#else
        /* to make compiler happy */
        (void)mask;
#endif

        return __shfl_down(var, lane_delta, width);
    }

    template <class T>
    __device__ inline T
        __shfl_up_sync(lane_mask mask, T var, unsigned int lane_delta, int width = WAVEFRONT_SIZE)
    {
        /* calling thread must be set in the mask */
#ifndef WARP_NO_MASK_CHECK
        assert(__is_thread_in_mask(mask));
#else
        /* to make compiler happy */
        (void)mask;
#endif

        return __shfl_up(var, lane_delta, width);
    }

    template <class T>
    __device__ inline T
        __shfl_xor_sync(lane_mask mask, T var, int lane_mask, int width = WAVEFRONT_SIZE)
    {
        /* calling thread must be set in the mask */
#ifndef WARP_NO_MASK_CHECK
        assert(__is_thread_in_mask(mask));
#else
        /* to make compiler happy */
        (void)mask;
#endif

        return __shfl_xor(var, lane_mask, width);
    }

    template <class T>
    __device__ inline T
        __shfl_local_sync(lane_mask mask, T var, int src, int width = WAVEFRONT_SIZE)
    {
        /* calling thread must be set in the mask */
#ifndef WARP_NO_MASK_CHECK
        assert(__is_thread_in_mask(mask));
#else
        /* to make compiler happy */
        (void)mask;
#endif

        unsigned int size = __mask_size(mask);

        /* check src lane */
        src = src % size;

        int lane = (size == WAVEFRONT_SIZE) ? src
                   : (WAVEFRONT_SIZE == 64) ? __fns64(mask, 0, (src + 1))
                                            : __fns32(mask, 0, (src + 1));

        return __shfl(var, lane, width);
    }

    template <class T>
    __device__ inline T __shfl_down_local_sync(lane_mask    mask,
                                               T            var,
                                               unsigned int lane_delta,
                                               int          width = WAVEFRONT_SIZE)
    {
        /* calling thread must be set in the mask */
#ifndef WARP_NO_MASK_CHECK
        assert(__is_thread_in_mask(mask));
#else
        /* to make compiler happy */
        (void)mask;
#endif

        unsigned int size = __mask_size(mask);

        /* if mask uses all lanes */
        if(size == WAVEFRONT_SIZE)
        {
            return __shfl_down(var, lane_delta, width);
        }

        int lane;

        if constexpr(WAVEFRONT_SIZE == 64)
        {
            lane = __fns64(mask, __lane_id(), lane_delta + 1);
        }
        else
        {
            lane = __fns32(mask, __lane_id(), lane_delta + 1);
        }

        if(lane == -1)
        {
            lane = __lane_id();
        }

        return __shfl(var, lane, width);
    }

    template <class T>
    __device__ inline T __shfl_up_local_sync(lane_mask    mask,
                                             T            var,
                                             unsigned int lane_delta,
                                             int          width = WAVEFRONT_SIZE)
    {
        /* calling thread must be set in the mask */
#ifndef WARP_NO_MASK_CHECK
        assert(__is_thread_in_mask(mask));
#else
        /* to make compiler happy */
        (void)mask;
#endif

        unsigned int size = __mask_size(mask);

        /* if mask uses all lanes */
        if(size == WAVEFRONT_SIZE)
        {
            return __shfl_up(var, lane_delta, width);
        }

        int lane;

        if constexpr(WAVEFRONT_SIZE == 64)
        {
            lane = __fns64(mask, __lane_id(), -((int)lane_delta + 1));
        }
        else
        {
            lane = __fns32(mask, __lane_id(), -((int)lane_delta + 1));
        }

        if(lane == -1)
        {
            lane = __lane_id();
        }

        return __shfl(var, lane, width);
    }

    template <class T>
    __device__ inline lane_mask __match_any_sync(lane_mask mask, T value)
    {
#if 1
        lane_mask smask = 0, bmask;

        /* each calling lane/thread must be in mask */
#ifndef WARP_NO_MASK_CHECK
        assert(__is_thread_in_mask(mask));
#else
        /* to make compiler happy */
        (void)mask;
#endif

        /* all threads */
        bmask = __branchmask();

        while(1)
        {
            int const i = raft::__FFS(bmask);

            if(i < 0)
                break;

            T rvar = __shfl(value, i);

            lane_mask ballot = __ballot_sync(bmask, value == rvar);

            if(value == rvar)
            {
                smask = ballot & mask;
                break;
            }

            bmask = bmask & (~ballot);
        }
#else
        lane_mask smask = 0, tmask;

        /* each calling lane/thread must be in mask */
#ifndef WARP_NO_MASK_CHECK
        assert(__is_thread_in_mask(mask));
#else
        /* to make compiler happy */
        (void)mask;
#endif

        while(1)
        {
            int const i = raft::__FFS(bmask)

                if(i < 0) break;

            T rvar = __shfl(value, i);

            lane_mask ballot = __ballot_sync(bmask, value == rvar);

            if(value == rvar)
            {
                smask = ballot & mask;
            }

            bmask = bmask & (~ballot);
        }
#endif

        return smask;
    }

    template <class T>
    __device__ inline lane_mask __match_all_sync(lane_mask mask, T value, int* pred)
    {
        /* non exited threads */
        mask = mask & __branchmask();

        lane_mask smask = __match_any_sync(mask, value);

        if((mask & smask) == mask)
        {
            *pred = 1;
            return mask;
        }
        else
        {
            *pred = 0;
            return 0;
        }
    }

    /* binary OPs */
    template <typename T>
    struct binop_multiply
    {
        __device__ T operator()(const T& lhs, const T& rhs)
        {
            return lhs * rhs;
        }
    };

    template <typename T>
    struct binop_add
    {
        __device__ T operator()(const T& lhs, const T& rhs)
        {
            return lhs + rhs;
        }
    };

    template <typename T>
    struct binop_min
    {
        __device__ T operator()(const T& lhs, const T& rhs)
        {
            return lhs < rhs ? lhs : rhs;
        }
    };

    template <typename T>
    struct binop_max
    {
        __device__ T operator()(const T& lhs, const T& rhs)
        {
            return lhs > rhs ? lhs : rhs;
        }
    };

    template <typename T>
    struct binop_and
    {
        __device__ T operator()(const T& lhs, const T& rhs)
        {
            return lhs & rhs;
        }
    };

    template <typename T>
    struct binop_or
    {
        __device__ T operator()(const T& lhs, const T& rhs)
        {
            return lhs | rhs;
        }
    };

    template <typename T>
    struct binop_xor
    {
        __device__ T operator()(const T& lhs, const T& rhs)
        {
            return lhs ^ rhs;
        }
    };

    template <class T, class BinaryOP>
    __device__ inline T __reduce_impl_sync(lane_mask mask, T var, BinaryOP op)
    {
        /* calling thread must be set in the mask */
#ifndef WARP_NO_MASK_CHECK
        assert(__is_thread_in_mask(mask));
#else
        /* to make compiler happy */
        (void)mask;
#endif

        int src;
        int size = __mask_size(mask);
        int lane;
        int tid = __thread_rank(mask);

        if(size == 1)
            return var;

        /* binary tree alg */
        if(size == WAVEFRONT_SIZE)
        {
            for(int mask = size / 2; mask > 0; mask /= 2)
                var = op(var, __shfl_xor(var, mask));
            return var;
        }
        else
        {
            while(size > 1 && tid < size)
            {
                /* check src lane */
                src = tid + size / 2;

                lane = (size == WAVEFRONT_SIZE) ? src
                       : (WAVEFRONT_SIZE == 64) ? __fns64(mask, 0, (src + 1))
                                                : __fns32(mask, 0, (src + 1));

                T tp = __shfl(var, lane);

                if(size & 1)
                {
                    if(tid > 0)
                        var = op(var, tp);
                }
                else
                {
                    var = op(var, tp);
                }

                size = (size + 1) / 2;
            }

            lane = (size == WAVEFRONT_SIZE) ? 0
                   : (WAVEFRONT_SIZE == 64) ? __fns64(mask, 0, 1)
                                            : __fns32(mask, 0, 1);

            return __shfl(var, lane);
        }
    }

    /* add min and max support int, unsigned, long, unsigned long,
 * long long, unsigned long long, float and double */
    template <typename T>
    __device__ T __reduce_mul_sync(lane_mask mask, T var)
    {
        binop_multiply<T> op;

        return __reduce_impl_sync<T, binop_multiply<T>>(mask, var, op);
    }

    template <typename T>
    __device__ T __reduce_add_sync(lane_mask mask, T var)
    {
        binop_add<T> op;

        return __reduce_impl_sync<T, binop_add<T>>(mask, var, op);
    }

    template <typename T>
    __device__ T __reduce_min_sync(lane_mask mask, T var)
    {
        binop_min<T> op;

        return __reduce_impl_sync<T, binop_min<T>>(mask, var, op);
    }

    template <typename T>
    __device__ T __reduce_max_sync(lane_mask mask, T var)
    {
        binop_max<T> op;

        return __reduce_impl_sync<T, binop_max<T>>(mask, var, op);
    }

    template <typename T>
    __device__ T __reduce_and_sync(lane_mask mask, T var)
    {
        binop_and<T> op;

        return __reduce_impl_sync<T, binop_and<T>>(mask, var, op);
    }

    template <typename T>
    __device__ T __reduce_or_sync(lane_mask mask, T var)
    {
        binop_or<T> op;

        return __reduce_impl_sync<T, binop_or<T>>(mask, var, op);
    }

    template <typename T>
    __device__ T __reduce_xor_sync(lane_mask mask, T var)
    {
        binop_xor<T> op;

        return __reduce_impl_sync<T, binop_xor<T>>(mask, var, op);
    }

} // namespace hip_warp_primitives

#endif
