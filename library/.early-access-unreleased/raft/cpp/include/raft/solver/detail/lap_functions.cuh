// Copyright (c) 2020-2024, NVIDIA CORPORATION.
// Copyright 2020 KETAN DATE & RAKESH NAGI
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/solver/detail/lap_kernels.cuh>
#include <raft/solver/linear_assignment_types.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <cstddef>

namespace raft::solver::detail
{

    const int BLOCKDIMX{64};
    const int BLOCKDIMY{1};

    // Function for calculating grid and block dimensions from the given input size.
    inline void calculateLinearDims(dim3& blocks_per_grid,
                                    dim3& threads_per_block,
                                    int&  total_blocks,
                                    int   size)
    {
        threads_per_block.x = BLOCKDIMX * BLOCKDIMY;

        int value = size / threads_per_block.x;
        if(size % threads_per_block.x > 0)
            value++;

        total_blocks      = value;
        blocks_per_grid.x = value;
    }

    // Function for calculating grid and block dimensions from the given input size for square grid.
    inline void calculateSquareDims(dim3& blocks_per_grid,
                                    dim3& threads_per_block,
                                    int&  total_blocks,
                                    int   size)
    {
        threads_per_block.x = BLOCKDIMX;
        threads_per_block.y = BLOCKDIMY;

        int sq_size = (int)ceil(sqrt(size));

        int valuex = (int)ceil((float)(sq_size) / BLOCKDIMX);
        int valuey = (int)ceil((float)(sq_size) / BLOCKDIMY);

        total_blocks      = valuex * valuey;
        blocks_per_grid.x = valuex;
        blocks_per_grid.y = valuey;
    }

    // Function for calculating grid and block dimensions from the given input size for rectangular
    // grid.
    inline void calculateRectangularDims(
        dim3& blocks_per_grid, dim3& threads_per_block, int& total_blocks, int xsize, int ysize)
    {
        threads_per_block.x = BLOCKDIMX;
        threads_per_block.y = BLOCKDIMY;

        int valuex = xsize / threads_per_block.x;
        if(xsize % threads_per_block.x > 0)
            valuex++;

        int valuey = ysize / threads_per_block.y;
        if(ysize % threads_per_block.y > 0)
            valuey++;

        total_blocks      = valuex * valuey;
        blocks_per_grid.x = valuex;
        blocks_per_grid.y = valuey;
    }

    template <typename vertex_t, typename weight_t>
    inline void initialReduction(raft::resources const&        handle,
                                 weight_t const*               d_costs,
                                 Vertices<vertex_t, weight_t>& d_vertices_dev,
                                 int                           SP,
                                 vertex_t                      N)
    {
        dim3 blocks_per_grid;
        dim3 threads_per_block;
        int  total_blocks = 0;

        detail::calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, SP);

        kernel_rowReduction<<<blocks_per_grid,
                              threads_per_block,
                              0,
                              resource::get_cuda_stream(handle)>>>(
            d_costs, d_vertices_dev.row_duals, SP, N, std::numeric_limits<weight_t>::max());

        RAFT_CHECK_CUDA(resource::get_cuda_stream(handle));
        kernel_columnReduction<<<blocks_per_grid,
                                 threads_per_block,
                                 0,
                                 resource::get_cuda_stream(handle)>>>(
            d_costs,
            d_vertices_dev.row_duals,
            d_vertices_dev.col_duals,
            SP,
            N,
            std::numeric_limits<weight_t>::max());
        RAFT_CHECK_CUDA(resource::get_cuda_stream(handle));
    }

    template <typename vertex_t, typename weight_t>
    inline void computeInitialAssignments(raft::resources const&        handle,
                                          weight_t const*               d_costs,
                                          Vertices<vertex_t, weight_t>& d_vertices,
                                          int                           SP,
                                          vertex_t                      N,
                                          weight_t                      epsilon)
    {
        dim3 blocks_per_grid;
        dim3 threads_per_block;
        int  total_blocks = 0;

        std::size_t size = SP * N;

        rmm::device_uvector<int> row_lock_v(size, resource::get_cuda_stream(handle));
        rmm::device_uvector<int> col_lock_v(size, resource::get_cuda_stream(handle));

        thrust::fill_n(thrust::device, d_vertices.row_assignments, size, -1);
        thrust::fill_n(thrust::device, d_vertices.col_assignments, size, -1);
        thrust::fill_n(thrust::device, row_lock_v.data(), size, 0);
        thrust::fill_n(thrust::device, col_lock_v.data(), size, 0);

        detail::calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, SP);

        kernel_computeInitialAssignments<<<blocks_per_grid,
                                           threads_per_block,
                                           0,
                                           resource::get_cuda_stream(handle)>>>(
            d_costs,
            d_vertices.row_duals,
            d_vertices.col_duals,
            d_vertices.row_assignments,
            d_vertices.col_assignments,
            row_lock_v.data(),
            col_lock_v.data(),
            SP,
            N,
            epsilon);
        RAFT_CHECK_CUDA(resource::get_cuda_stream(handle));
    }

    // Function for finding row cover on individual devices.
    template <typename vertex_t, typename weight_t>
    inline int computeRowCovers(raft::resources const&        handle,
                                Vertices<vertex_t, weight_t>& d_vertices,
                                VertexData<vertex_t>&         d_row_data,
                                VertexData<vertex_t>&         d_col_data,
                                int                           SP,
                                vertex_t                      N)
    {
        dim3 blocks_per_grid;
        dim3 threads_per_block;
        int  total_blocks = 0;

        std::size_t size = SP * N;

        thrust::fill_n(thrust::device, d_vertices.row_covers, size, int{0});
        thrust::fill_n(thrust::device, d_vertices.col_covers, size, int{0});
        thrust::fill_n(
            thrust::device, d_vertices.col_slacks, size, std::numeric_limits<weight_t>::max());
        thrust::fill_n(thrust::device, d_row_data.is_visited, size, DORMANT);
        thrust::fill_n(thrust::device, d_col_data.is_visited, size, DORMANT);
        thrust::fill_n(thrust::device, d_row_data.parents, size, vertex_t{-1});
        thrust::fill_n(thrust::device, d_row_data.children, size, vertex_t{-1});
        thrust::fill_n(thrust::device, d_col_data.parents, size, vertex_t{-1});
        thrust::fill_n(thrust::device, d_col_data.children, size, vertex_t{-1});

        detail::calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, SP);
        kernel_computeRowCovers<<<blocks_per_grid,
                                  threads_per_block,
                                  0,
                                  resource::get_cuda_stream(handle)>>>(
            d_vertices.row_assignments, d_vertices.row_covers, d_row_data.is_visited, SP, N);

        RAFT_CHECK_CUDA(resource::get_cuda_stream(handle));

        return thrust::reduce(thrust::device, d_vertices.row_covers, d_vertices.row_covers + size);
    }

    // Function for covering the zeros in uncovered rows and expanding the frontier.
    template <typename vertex_t, typename weight_t>
    inline void coverZeroAndExpand(raft::resources const&        handle,
                                   weight_t const*               d_costs_dev,
                                   vertex_t const*               d_rows_csr_neighbors,
                                   vertex_t const*               d_rows_csr_ptrs,
                                   Vertices<vertex_t, weight_t>& d_vertices_dev,
                                   VertexData<vertex_t>&         d_row_data_dev,
                                   VertexData<vertex_t>&         d_col_data_dev,
                                   bool*                         d_flag,
                                   int                           SP,
                                   vertex_t                      N,
                                   weight_t                      epsilon)
    {
        int  total_blocks = 0;
        dim3 blocks_per_grid;
        dim3 threads_per_block;

        detail::calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, SP);

        kernel_coverAndExpand<<<blocks_per_grid,
                                threads_per_block,
                                0,
                                resource::get_cuda_stream(handle)>>>(d_flag,
                                                                     d_rows_csr_ptrs,
                                                                     d_rows_csr_neighbors,
                                                                     d_costs_dev,
                                                                     d_vertices_dev,
                                                                     d_row_data_dev,
                                                                     d_col_data_dev,
                                                                     SP,
                                                                     N,
                                                                     epsilon);
    }

    template <typename vertex_t, typename weight_t>
    inline vertex_t zeroCoverIteration(raft::resources const&        handle,
                                       weight_t const*               d_costs_dev,
                                       Vertices<vertex_t, weight_t>& d_vertices_dev,
                                       VertexData<vertex_t>&         d_row_data_dev,
                                       VertexData<vertex_t>&         d_col_data_dev,
                                       bool*                         d_flag,
                                       int                           SP,
                                       vertex_t                      N,
                                       weight_t                      epsilon)
    {
        vertex_t M;

        rmm::device_uvector<vertex_t> csr_ptrs_v(0, resource::get_cuda_stream(handle));
        rmm::device_uvector<vertex_t> csr_neighbors_v(0, resource::get_cuda_stream(handle));

        {
            dim3 blocks_per_grid;
            dim3 threads_per_block;
            int  total_blocks = 0;

            rmm::device_uvector<bool>     predicates_v(SP * N, resource::get_cuda_stream(handle));
            rmm::device_uvector<vertex_t> addresses_v(SP * N, resource::get_cuda_stream(handle));

            thrust::fill_n(thrust::device, predicates_v.data(), SP * N, false);
            thrust::fill_n(thrust::device, addresses_v.data(), SP * N, vertex_t{0});

            csr_ptrs_v.resize(SP + 1, resource::get_cuda_stream(handle));

            thrust::fill_n(thrust::device, csr_ptrs_v.data(), (SP + 1), vertex_t{-1});

            detail::calculateRectangularDims(
                blocks_per_grid, threads_per_block, total_blocks, N, SP);

            // construct predicate matrix for edges.
            kernel_rowPredicateConstructionCSR<<<blocks_per_grid,
                                                 threads_per_block,
                                                 0,
                                                 resource::get_cuda_stream(handle)>>>(
                predicates_v.data(), addresses_v.data(), d_row_data_dev.is_visited, SP, N);
            RAFT_CHECK_CUDA(resource::get_cuda_stream(handle));

            M = thrust::reduce(thrust::device, addresses_v.begin(), addresses_v.end());
            thrust::exclusive_scan(
                thrust::device, addresses_v.begin(), addresses_v.end(), addresses_v.begin());

            if(M > 0)
            {
                csr_neighbors_v.resize(M, resource::get_cuda_stream(handle));

                kernel_rowScatterCSR<<<blocks_per_grid,
                                       threads_per_block,
                                       0,
                                       resource::get_cuda_stream(handle)>>>(predicates_v.data(),
                                                                            addresses_v.data(),
                                                                            csr_neighbors_v.data(),
                                                                            csr_ptrs_v.data(),
                                                                            M,
                                                                            SP,
                                                                            N);

                RAFT_CHECK_CUDA(resource::get_cuda_stream(handle));
            }
        }

        if(M > 0)
        {
            coverZeroAndExpand(handle,
                               d_costs_dev,
                               csr_neighbors_v.data(),
                               csr_ptrs_v.data(),
                               d_vertices_dev,
                               d_row_data_dev,
                               d_col_data_dev,
                               d_flag,
                               SP,
                               N,
                               epsilon);
        }

        return M;
    }

    // Function for executing recursive zero cover. Returns the next step (Step 4 or Step 5) depending
    // on the presence of uncovered zeros.
    template <typename vertex_t, typename weight_t>
    inline void executeZeroCover(raft::resources const&        handle,
                                 weight_t const*               d_costs_dev,
                                 Vertices<vertex_t, weight_t>& d_vertices_dev,
                                 VertexData<vertex_t>&         d_row_data_dev,
                                 VertexData<vertex_t>&         d_col_data_dev,
                                 bool*                         d_flag,
                                 int                           SP,
                                 vertex_t                      N,
                                 weight_t                      epsilon)
    {
        vertex_t M = 1;
        while(M > 0)
        {
            M = zeroCoverIteration(handle,
                                   d_costs_dev,
                                   d_vertices_dev,
                                   d_row_data_dev,
                                   d_col_data_dev,
                                   d_flag,
                                   SP,
                                   N,
                                   epsilon);
        }
    }

    // Function for executing reverse pass of the maximum matching.
    template <typename vertex_t>
    inline void reversePass(raft::resources const& handle,
                            VertexData<vertex_t>&  d_row_data_dev,
                            VertexData<vertex_t>&  d_col_data_dev,
                            int                    SP,
                            int                    N)
    {
        int  total_blocks = 0;
        dim3 blocks_per_grid;
        dim3 threads_per_block;

        std::size_t size = SP * N;

        detail::calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, size);

        rmm::device_uvector<bool>     predicates_v(size, resource::get_cuda_stream(handle));
        rmm::device_uvector<vertex_t> addresses_v(size, resource::get_cuda_stream(handle));

        thrust::fill_n(thrust::device, predicates_v.data(), size, false);
        thrust::fill_n(thrust::device, addresses_v.data(), size, vertex_t{0});

        // compact the reverse pass row vertices.
        kernel_augmentPredicateConstruction<<<blocks_per_grid,
                                              threads_per_block,
                                              0,
                                              resource::get_cuda_stream(handle)>>>(
            predicates_v.data(), addresses_v.data(), d_col_data_dev.is_visited, size);

        RAFT_CHECK_CUDA(resource::get_cuda_stream(handle));

        // calculate total number of vertices.
        std::size_t csr_size
            = thrust::reduce(thrust::device, addresses_v.begin(), addresses_v.end());
        // exclusive scan for calculating the scatter addresses.
        thrust::exclusive_scan(
            thrust::device, addresses_v.begin(), addresses_v.end(), addresses_v.begin());

        if(csr_size > 0)
        {
            int  total_blocks_1 = 0;
            dim3 blocks_per_grid_1;
            dim3 threads_per_block_1;
            detail::calculateLinearDims(
                blocks_per_grid_1, threads_per_block_1, total_blocks_1, csr_size);

            rmm::device_uvector<vertex_t> elements_v(csr_size, resource::get_cuda_stream(handle));

            kernel_augmentScatter<<<blocks_per_grid,
                                    threads_per_block,
                                    0,
                                    resource::get_cuda_stream(handle)>>>(
                elements_v.data(), predicates_v.data(), addresses_v.data(), size);

            RAFT_CHECK_CUDA(resource::get_cuda_stream(handle));

            kernel_reverseTraversal<<<blocks_per_grid_1,
                                      threads_per_block_1,
                                      0,
                                      resource::get_cuda_stream(handle)>>>(
                elements_v.data(), d_row_data_dev, d_col_data_dev, csr_size);
            RAFT_CHECK_CUDA(resource::get_cuda_stream(handle));
        }
    }

    // Function for executing augmentation pass of the maximum matching.
    template <typename vertex_t, typename weight_t>
    inline void augmentationPass(raft::resources const&        handle,
                                 Vertices<vertex_t, weight_t>& d_vertices_dev,
                                 VertexData<vertex_t>&         d_row_data_dev,
                                 VertexData<vertex_t>&         d_col_data_dev,
                                 int                           SP,
                                 int                           N)
    {
        int  total_blocks = 0;
        dim3 blocks_per_grid;
        dim3 threads_per_block;
        detail::calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, SP * N);

        rmm::device_uvector<bool>     predicates_v(SP * N, resource::get_cuda_stream(handle));
        rmm::device_uvector<vertex_t> addresses_v(SP * N, resource::get_cuda_stream(handle));

        thrust::fill_n(thrust::device, predicates_v.data(), SP * N, false);
        thrust::fill_n(thrust::device, addresses_v.data(), SP * N, vertex_t{0});

        // compact the reverse pass row vertices.
        kernel_augmentPredicateConstruction<<<blocks_per_grid,
                                              threads_per_block,
                                              0,
                                              resource::get_cuda_stream(handle)>>>(
            predicates_v.data(), addresses_v.data(), d_row_data_dev.is_visited, SP * N);

        RAFT_CHECK_CUDA(resource::get_cuda_stream(handle));

        // calculate total number of vertices.
        // TODO: should be vertex_t
        vertex_t row_ids_csr_size
            = thrust::reduce(thrust::device, addresses_v.begin(), addresses_v.end());
        // exclusive scan for calculating the scatter addresses.
        thrust::exclusive_scan(
            thrust::device, addresses_v.begin(), addresses_v.end(), addresses_v.begin());

        if(row_ids_csr_size > 0)
        {
            int  total_blocks_1 = 0;
            dim3 blocks_per_grid_1;
            dim3 threads_per_block_1;
            detail::calculateLinearDims(
                blocks_per_grid_1, threads_per_block_1, total_blocks_1, row_ids_csr_size);

            rmm::device_uvector<vertex_t> elements_v(row_ids_csr_size,
                                                     resource::get_cuda_stream(handle));

            kernel_augmentScatter<<<blocks_per_grid,
                                    threads_per_block,
                                    0,
                                    resource::get_cuda_stream(handle)>>>(
                elements_v.data(), predicates_v.data(), addresses_v.data(), vertex_t{SP * N});

            RAFT_CHECK_CUDA(resource::get_cuda_stream(handle));

            kernel_augmentation<<<blocks_per_grid_1,
                                  threads_per_block_1,
                                  0,
                                  resource::get_cuda_stream(handle)>>>(
                d_vertices_dev.row_assignments,
                d_vertices_dev.col_assignments,
                elements_v.data(),
                d_row_data_dev,
                d_col_data_dev,
                vertex_t{N},
                row_ids_csr_size);

            RAFT_CHECK_CUDA(resource::get_cuda_stream(handle));
        }
    }

    template <typename vertex_t, typename weight_t>
    inline void dualUpdate(raft::resources const&        handle,
                           Vertices<vertex_t, weight_t>& d_vertices_dev,
                           VertexData<vertex_t>&         d_row_data_dev,
                           VertexData<vertex_t>&         d_col_data_dev,
                           int                           SP,
                           vertex_t                      N,
                           weight_t                      epsilon)
    {
        dim3 blocks_per_grid;
        dim3 threads_per_block;
        int  total_blocks;

        rmm::device_uvector<weight_t> sp_min_v(SP, resource::get_cuda_stream(handle));

        detail::calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, SP);
        kernel_dualUpdate_1<<<blocks_per_grid,
                              threads_per_block,
                              0,
                              resource::get_cuda_stream(handle)>>>(
            sp_min_v.data(),
            d_vertices_dev.col_slacks,
            d_vertices_dev.col_covers,
            SP,
            N,
            std::numeric_limits<weight_t>::max());

        RAFT_CHECK_CUDA(resource::get_cuda_stream(handle));

        detail::calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, SP);
        kernel_dualUpdate_2<<<blocks_per_grid,
                              threads_per_block,
                              0,
                              resource::get_cuda_stream(handle)>>>(
            sp_min_v.data(),
            d_vertices_dev.row_duals,
            d_vertices_dev.col_duals,
            d_vertices_dev.col_slacks,
            d_vertices_dev.row_covers,
            d_vertices_dev.col_covers,
            d_row_data_dev.is_visited,
            d_col_data_dev.parents,
            SP,
            N,
            std::numeric_limits<weight_t>::max(),
            epsilon);

        RAFT_CHECK_CUDA(resource::get_cuda_stream(handle));
    }

    // Function for calculating optimal objective function value using dual variables.
    template <typename vertex_t, typename weight_t>
    inline void calcObjValDual(raft::resources const&        handle,
                               weight_t*                     d_obj_val,
                               Vertices<vertex_t, weight_t>& d_vertices_dev,
                               int                           SP,
                               int                           N)
    {
        dim3 blocks_per_grid;
        dim3 threads_per_block;
        int  total_blocks = 0;

        detail::calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, SP);

        kernel_calcObjValDual<<<blocks_per_grid,
                                threads_per_block,
                                0,
                                resource::get_cuda_stream(handle)>>>(
            d_obj_val, d_vertices_dev.row_duals, d_vertices_dev.col_duals, SP, N);

        RAFT_CHECK_CUDA(resource::get_cuda_stream(handle));
    }

    // Function for calculating optimal objective function value using dual variables.
    template <typename vertex_t, typename weight_t>
    inline void calcObjValPrimal(raft::resources const& handle,
                                 weight_t*              d_obj_val,
                                 weight_t const*        d_costs,
                                 vertex_t const*        d_row_assignments,
                                 int                    SP,
                                 vertex_t               N)
    {
        dim3 blocks_per_grid;
        dim3 threads_per_block;
        int  total_blocks = 0;

        detail::calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, SP);

        kernel_calcObjValPrimal<<<blocks_per_grid,
                                  threads_per_block,
                                  0,
                                  resource::get_cuda_stream(handle)>>>(
            d_obj_val, d_costs, d_row_assignments, SP, N);

        RAFT_CHECK_CUDA(resource::get_cuda_stream(handle));
    }

} // namespace raft::solver::detail
