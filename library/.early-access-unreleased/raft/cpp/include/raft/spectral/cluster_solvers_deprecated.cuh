// Copyright (c) 2019-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/**
 * Note: This file is deprecated and will be removed in a future release
 * Please use include/raft/cluster/kmeans.cuh instead
 */

#ifndef __CLUSTER_SOLVERS_deprecated_H
#define __CLUSTER_SOLVERS_deprecated_H

#pragma once

#include <raft/cluster/kmeans_deprecated.cuh>

#include <utility> // for std::pair

namespace raft
{
    namespace spectral
    {

        using namespace matrix;

        // aggregate of control params for Eigen Solver:
        //
        template <typename index_type_t, typename value_type_t, typename size_type_t = index_type_t>
        struct cluster_solver_config_deprecated_t
        {
            size_type_t n_clusters;
            size_type_t maxIter;

            value_type_t tol;

            unsigned long long seed{123456};
        };

        template <typename index_type_t, typename value_type_t, typename size_type_t = index_type_t>
        struct kmeans_solver_deprecated_t
        {
            explicit kmeans_solver_deprecated_t(
                cluster_solver_config_deprecated_t<index_type_t, value_type_t, size_type_t> const&
                    config)
                : config_(config)
            {
            }

            std::pair<value_type_t, index_type_t> solve(raft::resources const& handle,
                                                        size_type_t            n_obs_vecs,
                                                        size_type_t            dim,
                                                        value_type_t const* __restrict__ obs,
                                                        index_type_t* __restrict__ codes) const
            {
                RAFT_EXPECTS(obs != nullptr, "Null obs buffer.");
                RAFT_EXPECTS(codes != nullptr, "Null codes buffer.");
                value_type_t residual{};
                index_type_t iters{};

                raft::cluster::kmeans(handle,
                                      n_obs_vecs,
                                      dim,
                                      config_.n_clusters,
                                      config_.tol,
                                      config_.maxIter,
                                      obs,
                                      codes,
                                      residual,
                                      iters,
                                      config_.seed);
                return std::make_pair(residual, iters);
            }

            auto const& get_config(void) const
            {
                return config_;
            }

        private:
            cluster_solver_config_deprecated_t<index_type_t, value_type_t, size_type_t> config_;
        };

    } // namespace spectral
} // namespace raft

#endif
