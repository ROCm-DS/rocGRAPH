// Copyright (c) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/util/cuda_utils.cuh>

#include <stdint.h>

namespace raft::util
{

    /**
 * @brief Perform fast integer division and modulo using a known divisor
 * From Hacker's Delight, Second Edition, Chapter 10
 *
 * @note This currently only supports 32b signed integers
 * @todo Extend support for signed divisors
 */
    struct FastIntDiv
    {
        /**
   * @defgroup HostMethods Ctor's that are accessible only from host
   * @{
   * @brief Host-only ctor's
   * @param _d the divisor
   */
        FastIntDiv(int _d)
            : d(_d)
        {
            computeScalars();
        }
        FastIntDiv& operator=(int _d)
        {
            d = _d;
            computeScalars();
            return *this;
        }
        /** @} */

        /**
   * @defgroup DeviceMethods Ctor's which even the device-side can access
   * @{
   * @brief host and device ctor's
   * @param other source object to be copied from
   */
        HDI FastIntDiv(const FastIntDiv& other)
            : d(other.d)
            , m(other.m)
            , p(other.p)
        {
        }
        HDI FastIntDiv& operator=(const FastIntDiv& other)
        {
            d = other.d;
            m = other.m;
            p = other.p;
            return *this;
        }
        /** @} */

        /** divisor */
        int d;
        /** the term 'm' as found in the reference chapter */
        unsigned m;
        /** the term 'p' as found in the reference chapter */
        int p;

    private:
        void computeScalars()
        {
            if(d == 1)
            {
                m = 0;
                p = 1;
                return;
            }
            else if(d < 0)
            {
                ASSERT(false, "FastIntDiv: division by negative numbers not supported!");
            }
            else if(d == 0)
            {
                ASSERT(false, "FastIntDiv: got division by zero!");
            }
            int64_t nc = ((1LL << 31) / d) * d - 1;
            p          = 31;
            int64_t twoP, rhs;
            do
            {
                ++p;
                twoP = 1LL << p;
                rhs  = nc * (d - twoP % d);
            } while(twoP <= rhs);
            m = (twoP + d - twoP % d) / d;
        }
    }; // struct FastIntDiv

    /**
 * @brief Division overload, so that FastIntDiv can be transparently switched
 *        to even on device
 * @param n numerator
 * @param divisor the denominator
 * @return the quotient
 */
    HDI int operator/(int n, const FastIntDiv& divisor)
    {
        if(divisor.d == 1)
            return n;
        int ret = (int64_t(divisor.m) * int64_t(n)) >> divisor.p;
        if(n < 0)
            ++ret;
        return ret;
    }

    /**
 * @brief Modulo overload, so that FastIntDiv can be transparently switched
 *        to even on device
 * @param n numerator
 * @param divisor the denominator
 * @return the remainder
 */
    HDI int operator%(int n, const FastIntDiv& divisor)
    {
        int quotient  = n / divisor;
        int remainder = n - quotient * divisor.d;
        return remainder;
    }

}; // namespace raft::util
