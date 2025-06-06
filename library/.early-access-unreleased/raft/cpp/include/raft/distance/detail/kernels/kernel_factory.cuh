// Copyright (c) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "gram_matrix.cuh"
#include "kernel_matrices.cuh"

#include <raft/distance/distance_types.hpp>
#include <raft/util/cudart_utils.hpp>

namespace raft::distance::kernels::detail
{

    template <typename math_t>
    class KernelFactory
    {
    public:
        static GramMatrixBase<math_t>* create(KernelParams params)
        {
            GramMatrixBase<math_t>* res;
            // KernelParams is not templated, we convert the parameters to math_t here:
            math_t coef0 = params.coef0;
            math_t gamma = params.gamma;
            switch(params.kernel)
            {
            case LINEAR:
                res = new GramMatrixBase<math_t>();
                break;
            case POLYNOMIAL:
                res = new PolynomialKernel<math_t, int>(params.degree, gamma, coef0);
                break;
            case TANH:
                res = new TanhKernel<math_t>(gamma, coef0);
                break;
            case RBF:
                res = new RBFKernel<math_t>(gamma);
                break;
            default:
                throw raft::exception("Kernel not implemented");
            }
            return res;
        }

        [[deprecated]] static GramMatrixBase<math_t>* create(KernelParams   params,
                                                             cublasHandle_t handle)
        {
            GramMatrixBase<math_t>* res;
            // KernelParams is not templated, we convert the parameters to math_t here:
            math_t coef0 = params.coef0;
            math_t gamma = params.gamma;
            switch(params.kernel)
            {
            case LINEAR:
                res = new GramMatrixBase<math_t>(handle);
                break;
            case POLYNOMIAL:
                res = new PolynomialKernel<math_t, int>(params.degree, gamma, coef0, handle);
                break;
            case TANH:
                res = new TanhKernel<math_t>(gamma, coef0, handle);
                break;
            case RBF:
                res = new RBFKernel<math_t>(gamma, handle);
                break;
            default:
                throw raft::exception("Kernel not implemented");
            }
            return res;
        }
    };

}; // end namespace raft::distance::kernels::detail
