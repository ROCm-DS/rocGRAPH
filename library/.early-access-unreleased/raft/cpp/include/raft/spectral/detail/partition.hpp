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

#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/spectral/cluster_solvers.cuh>
#include <raft/spectral/detail/spectral_util.cuh>
#include <raft/spectral/eigen_solvers.cuh>
#include <raft/spectral/matrix_wrappers.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <raft/cuda_runtime.h>
#else
#include <cuda.h>
#endif

#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <math.h>
#include <stdio.h>

#include <tuple>

namespace raft
{
    namespace spectral
    {
        namespace detail
        {

            // =========================================================
            // Spectral partitioner
            // =========================================================

            /// Compute spectral graph partition
            /** Compute partition for a weighted undirected graph. This
 *  partition attempts to minimize the cost function:
 *    Cost = \sum_i (Edges cut by ith partition)/(Vertices in ith partition)
 *
 *  @param G Weighted graph in CSR format
 *  @param nClusters Number of partitions.
 *  @param nEigVecs Number of eigenvectors to compute.
 *  @param maxIter_lanczos Maximum number of Lanczos iterations.
 *  @param restartIter_lanczos Maximum size of Lanczos system before
 *    implicit restart.
 *  @param tol_lanczos Convergence tolerance for Lanczos method.
 *  @param maxIter_kmeans Maximum number of k-means iterations.
 *  @param tol_kmeans Convergence tolerance for k-means algorithm.
 *  @param clusters (Output, device memory, n entries) Partition
 *    assignments.
 *  @param iters_lanczos On exit, number of Lanczos iterations
 *    performed.
 *  @param iters_kmeans On exit, number of k-means iterations
 *    performed.
 *  @return statistics: number of eigensolver iterations, .
 */
            template <typename vertex_t,
                      typename weight_t,
                      typename EigenSolver,
                      typename ClusterSolver>
            std::tuple<vertex_t, weight_t, vertex_t>
                partition(raft::resources const&                                       handle,
                          spectral::matrix::sparse_matrix_t<vertex_t, weight_t> const& csr_m,
                          EigenSolver const&                                           eigen_solver,
                          ClusterSolver const& cluster_solver,
                          vertex_t* __restrict__ clusters,
                          weight_t* eigVals,
                          weight_t* eigVecs)
            {
                RAFT_EXPECTS(clusters != nullptr, "Null clusters buffer.");
                RAFT_EXPECTS(eigVals != nullptr, "Null eigVals buffer.");
                RAFT_EXPECTS(eigVecs != nullptr, "Null eigVecs buffer.");

                auto stream   = resource::get_cuda_stream(handle);
                auto cublas_h = resource::get_cublas_handle(handle);

                std::tuple<vertex_t, weight_t, vertex_t>
                    stats; //{iters_eig_solver,residual_cluster,iters_cluster_solver} // # iters eigen solver,
                // cluster solver residual, # iters cluster solver

                vertex_t n = csr_m.nrows_;

                // -------------------------------------------------------
                // Spectral partitioner
                // -------------------------------------------------------

                // Compute eigenvectors of Laplacian

                // Initialize Laplacian
                /// sparse_matrix_t<vertex_t, weight_t> A{handle, graph};
                spectral::matrix::laplacian_matrix_t<vertex_t, weight_t> L{handle, csr_m};

                auto eigen_config = eigen_solver.get_config();
                auto nEigVecs     = eigen_config.n_eigVecs;

                // Compute smallest eigenvalues and eigenvectors
                std::get<0>(stats)
                    = eigen_solver.solve_smallest_eigenvectors(handle, L, eigVals, eigVecs);

                // Whiten eigenvector matrix
                transform_eigen_matrix(handle, n, nEigVecs, eigVecs);

                // Find partition clustering
                auto pair_cluster = cluster_solver.solve(handle, n, nEigVecs, eigVecs, clusters);

                std::get<1>(stats) = pair_cluster.first;
                std::get<2>(stats) = pair_cluster.second;

                return stats;
            }

            // =========================================================
            // Analysis of graph partition
            // =========================================================

            /// Compute cost function for partition
            /** This function determines the edges cut by a partition and a cost
 *  function:
 *    Cost = \sum_i (Edges cut by ith partition)/(Vertices in ith partition)
 *  Graph is assumed to be weighted and undirected.
 *
 *  @param G Weighted graph in CSR format
 *  @param nClusters Number of partitions.
 *  @param clusters (Input, device memory, n entries) Partition
 *    assignments.
 *  @param edgeCut On exit, weight of edges cut by partition.
 *  @param cost On exit, partition cost function.
 *  @return error flag.
 */
            template <typename vertex_t, typename weight_t>
            void
                analyzePartition(raft::resources const& handle,
                                 spectral::matrix::sparse_matrix_t<vertex_t, weight_t> const& csr_m,
                                 vertex_t nClusters,
                                 const vertex_t* __restrict__ clusters,
                                 weight_t& edgeCut,
                                 weight_t& cost)
            {
                RAFT_EXPECTS(clusters != nullptr, "Null clusters buffer.");

                vertex_t i;
                vertex_t n = csr_m.nrows_;

                auto stream   = resource::get_cuda_stream(handle);
                auto cublas_h = resource::get_cublas_handle(handle);

                weight_t partEdgesCut, clustersize;

                // Device memory
                spectral::matrix::vector_t<weight_t> part_i(handle, n);
                spectral::matrix::vector_t<weight_t> Lx(handle, n);

                // Initialize cuBLAS
                RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
                    cublas_h, CUBLAS_POINTER_MODE_HOST, stream));

                // Initialize Laplacian
                /// sparse_matrix_t<vertex_t, weight_t> A{handle, graph};
                spectral::matrix::laplacian_matrix_t<vertex_t, weight_t> L{handle, csr_m};

                // Initialize output
                cost    = 0;
                edgeCut = 0;

                // Iterate through partitions
                for(i = 0; i < nClusters; ++i)
                {
                    // Construct indicator vector for ith partition
                    if(!construct_indicator(
                           handle, i, n, clustersize, partEdgesCut, clusters, part_i, Lx, L))
                    {
                        WARNING("empty partition");
                        continue;
                    }

                    // Record results
                    cost += partEdgesCut / clustersize;
                    edgeCut += partEdgesCut / 2;
                }
            }

        } // namespace detail
    } // namespace spectral
} // namespace raft
