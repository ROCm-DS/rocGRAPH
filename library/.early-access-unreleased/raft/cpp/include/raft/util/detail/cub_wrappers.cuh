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

#include <rmm/device_uvector.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif

namespace raft
{

    /**
 * @brief Convenience wrapper over cub's SortPairs method
 * @tparam KeyT key type
 * @tparam ValueT value type
 * @param workspace workspace buffer which will get resized if not enough space
 * @param inKeys input keys array
 * @param outKeys output keys array
 * @param inVals input values array
 * @param outVals output values array
 * @param len array length
 * @param stream cuda stream
 */
    template <typename KeyT, typename ValueT>
    void sortPairs(rmm::device_uvector<char>& workspace,
                   const KeyT*                inKeys,
                   KeyT*                      outKeys,
                   const ValueT*              inVals,
                   ValueT*                    outVals,
                   int                        len,
                   cudaStream_t               stream)
    {
        size_t worksize = 0; //  Fix 'worksize' may be used uninitialized in this function.
        cub::DeviceRadixSort::SortPairs(
            nullptr, worksize, inKeys, outKeys, inVals, outVals, len, 0, sizeof(KeyT) * 8, stream);
        workspace.resize(worksize, stream);
        cub::DeviceRadixSort::SortPairs(workspace.data(),
                                        worksize,
                                        inKeys,
                                        outKeys,
                                        inVals,
                                        outVals,
                                        len,
                                        0,
                                        sizeof(KeyT) * 8,
                                        stream);
    }

} // namespace raft
