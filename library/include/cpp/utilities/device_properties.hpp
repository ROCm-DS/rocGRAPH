// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef DEVICE_PROPERTIES_H
#define DEVICE_PROPERTIES_H

#include <hip/hip_runtime.h>
#include <limits>

namespace rocgraph
{

    /* The following are meant to be device replacements for the equivalent host/device functions in raft that have hard coded values */
    /* NB that these should *not* be run on the host since warpSize is not defined except in  a GPU kernel. This is only decorated with __host__
because of prior issues passing __device__ lambdas to functions required us to specify __host__ as well */

    __host__ __device__ int inline constexpr warp_size()
    {
        return warpSize;
    }

    // The following function is used to send a mask to warp intrinsics such as __shfl_sync, etc. These functions take an unsigned long long mask
    // so that's what is returned here rather than, say, a uint64_t.
    __host__ __device__ unsigned long long constexpr warp_full_mask()
    {

        // return a 64 lane bit mask by default
        return 0xFFFFFFFFFFFFFFFFULL;
    }

    __host__ __device__ unsigned long long constexpr warp_empty_mask()
    {

        return 0x0;
    }

}

#endif //end include fence
