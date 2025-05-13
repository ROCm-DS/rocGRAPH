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
#pragma once

#include <hiprand/hiprand.h>

// types
#define curandGenerator_t hiprandGenerator_t
#define curandStatus_t hiprandStatus_t

// macros, constants, enums
#define CURAND_RNG_PSEUDO_DEFAULT HIPRAND_RNG_PSEUDO_DEFAULT
#define CURAND_STATUS_SUCCESS HIPRAND_STATUS_SUCCESS
#define CURAND_RNG_PSEUDO_PHILOX4_32_10 HIPRAND_RNG_PSEUDO_PHILOX4_32_10

// functions
#define curandCreateGenerator hiprandCreateGenerator
#define curandDestroyGenerator hiprandDestroyGenerator
#define curandGenerateNormal hiprandGenerateNormal
#define curandGenerateNormalDouble hiprandGenerateNormalDouble
#define curandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define curandGenerateUniform hiprandGenerateUniform
#define curandGenerateUniformDouble hiprandGenerateUniformDouble
