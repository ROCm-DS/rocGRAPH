// Copyright (C) 2019-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "scc_matrix.cuh"
#include "utilities/graph_utils.cuh"
#include "weak_cc.cuh"

#include "algorithms.hpp"
#include "legacy/graph.hpp"
#include "utilities/error.hpp"

#include <thrust/sequence.h>

#include <cstdint>
#include <iostream>
#include <type_traits>

namespace rocgraph
{
    namespace detail
    {

        /**
 * @brief Compute connected components.
 * The weak version has been eliminated in lieu of the primitive based implementation
 *
 * The strong version (for directed or undirected graphs) is based on:
 * [2] Gilbert, J. et al, 2011. "Graph Algorithms in the Language of Linear Algebra"
 *
 * C = I | A | A^2 |...| A^k
 * where matrix multiplication is via semi-ring:
 * (combine, reduce) == (&, |) (bitwise ops)
 * Then: X = C & transpose(C); and finally, apply get_labels(X);
 *
 *
 * @tparam IndexT the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @param graph input graph; assumed undirected for weakly CC [in]
 * @param connectivity_type Ignored [in]
 * @param stream the cuda stream [in]
 */
        template <typename VT, typename ET, typename WT, int TPB_X = 32>
        std::enable_if_t<std::is_signed<VT>::value>
            connected_components_impl(legacy::GraphCSRView<VT, ET, WT> const& graph,
                                      rocgraph_cc_t                           connectivity_type,
                                      VT*                                     labels,
                                      hipStream_t                             stream)
        {
            using ByteT = unsigned char; // minimum addressable unit

            ROCGRAPH_EXPECTS(graph.offsets != nullptr,
                             "Invalid input argument: graph.offsets is nullptr");
            ROCGRAPH_EXPECTS(graph.indices != nullptr,
                             "Invalid input argument: graph.indices is nullptr");

            VT nrows = graph.number_of_vertices;

            SCC_Data<ByteT, VT> sccd(nrows, graph.offsets, graph.indices);
            auto                num_iters = sccd.run_scc(labels);
        }
    } // namespace detail

    template <typename VT, typename ET, typename WT>
    void connected_components(legacy::GraphCSRView<VT, ET, WT> const& graph,
                              rocgraph_cc_t                           connectivity_type,
                              VT*                                     labels)
    {
        hipStream_t stream{nullptr};

        ROCGRAPH_EXPECTS(labels != nullptr, "Invalid input argument: labels parameter is NULL");

        return detail::connected_components_impl<VT, ET, WT>(
            graph, connectivity_type, labels, stream);
    }

    template void connected_components<int32_t, int32_t, float>(
        legacy::GraphCSRView<int32_t, int32_t, float> const&, rocgraph_cc_t, int32_t*);
    template void connected_components<int64_t, int64_t, float>(
        legacy::GraphCSRView<int64_t, int64_t, float> const&, rocgraph_cc_t, int64_t*);

} // namespace rocgraph
