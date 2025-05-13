// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include "internal/rocgraph_community_algorithms.h"
#include "c_api/rocgraph_abstract_functor.hpp"
#include "c_api/rocgraph_capi_helper.hpp"
#include "c_api/rocgraph_graph.hpp"
#include "c_api/rocgraph_handle.hpp"
#include "c_api/rocgraph_utils.hpp"

#include "algorithms.hpp"
#include "detail/utility_wrappers.hpp"
#include "internal/aux/rocgraph_clustering_result_aux.h"
#include "legacy/graph.hpp"
#include "rocgraph_control.hpp"

namespace rocgraph
{
    namespace c_api
    {

        struct rocgraph_clustering_result_t
        {
            rocgraph_type_erased_device_array_t* vertices_{nullptr};
            rocgraph_type_erased_device_array_t* clusters_{nullptr};
        };

    } // namespace c_api
} // namespace rocgraph

namespace
{

    struct balanced_cut_clustering_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                          handle_;
        rocgraph::c_api::rocgraph_graph_t*             graph_;
        size_t                                         n_clusters_;
        size_t                                         n_eigenvectors_;
        double                                         evs_tolerance_;
        int                                            evs_max_iterations_;
        double                                         k_means_tolerance_;
        int                                            k_means_max_iterations_;
        bool                                           do_expensive_check_;
        rocgraph::c_api::rocgraph_clustering_result_t* result_{};

        balanced_cut_clustering_functor(::rocgraph_handle_t const* handle,
                                        ::rocgraph_graph_t*        graph,
                                        size_t                     n_clusters,
                                        size_t                     n_eigenvectors,
                                        double                     evs_tolerance,
                                        int                        evs_max_iterations,
                                        double                     k_means_tolerance,
                                        int                        k_means_max_iterations,
                                        bool                       do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , n_clusters_(n_clusters)
            , n_eigenvectors_(n_eigenvectors)
            , evs_tolerance_(evs_tolerance)
            , evs_max_iterations_(evs_max_iterations)
            , k_means_tolerance_(k_means_tolerance)
            , k_means_max_iterations_(k_means_max_iterations)
            , do_expensive_check_(do_expensive_check)
        {
        }

        template <typename vertex_t,
                  typename edge_t,
                  typename weight_t,
                  typename edge_type_type_t,
                  bool store_transposed,
                  bool multi_gpu>
        void operator()()
        {
            if constexpr(!rocgraph::is_candidate<vertex_t, edge_t, weight_t>::value)
            {
                unsupported();
            }
            else if constexpr(multi_gpu)
            {
                unsupported();
            }
            else if constexpr(!std::is_same_v<edge_t, int32_t>)
            {
                unsupported();
            }
            else
            {
                // balanced_cut expects store_transposed == false
                if constexpr(store_transposed)
                {
                    status_ = rocgraph::c_api::
                        transpose_storage<vertex_t, edge_t, weight_t, store_transposed, false>(
                            handle_, graph_, error_.get());
                    if(status_ != rocgraph_status_success)
                        return;
                }

                auto graph = reinterpret_cast<rocgraph::graph_t<vertex_t, edge_t, false, false>*>(
                    graph_->graph_);

                auto edge_weights = reinterpret_cast<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, false, false>,
                    weight_t>*>(graph_->edge_weights_);

                auto number_map
                    = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

                auto graph_view          = graph->view();
                auto edge_partition_view = graph_view.local_edge_partition_view();

                rmm::device_uvector<weight_t> tmp_weights(0, handle_.get_stream());
                if(edge_weights == nullptr)
                {
                    tmp_weights.resize(edge_partition_view.indices().size(), handle_.get_stream());
                    rocgraph::detail::scalar_fill(
                        handle_, tmp_weights.data(), tmp_weights.size(), weight_t{1});
                }

                rocgraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> legacy_graph_view(
                    const_cast<edge_t*>(edge_partition_view.offsets().data()),
                    const_cast<vertex_t*>(edge_partition_view.indices().data()),
                    (edge_weights == nullptr)
                        ? tmp_weights.data()
                        : const_cast<weight_t*>(edge_weights->view().value_firsts().front()),
                    edge_partition_view.offsets().size() - 1,
                    edge_partition_view.indices().size());

                rmm::device_uvector<vertex_t> clusters(
                    graph_view.local_vertex_partition_range_size(), handle_.get_stream());

                rocgraph::ext_raft::balancedCutClustering(legacy_graph_view,
                                                          static_cast<vertex_t>(n_clusters_),
                                                          static_cast<vertex_t>(n_eigenvectors_),
                                                          static_cast<weight_t>(evs_tolerance_),
                                                          evs_max_iterations_,
                                                          static_cast<weight_t>(k_means_tolerance_),
                                                          k_means_max_iterations_,
                                                          clusters.data());

                rmm::device_uvector<vertex_t> vertices(
                    graph_view.local_vertex_partition_range_size(), handle_.get_stream());
                raft::copy(
                    vertices.data(), number_map->data(), vertices.size(), handle_.get_stream());

                result_ = new rocgraph::c_api::rocgraph_clustering_result_t{
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(vertices,
                                                                             graph_->vertex_type_),
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(clusters,
                                                                             graph_->vertex_type_)};
            }
        }
    };

    struct spectral_clustering_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                          handle_;
        rocgraph::c_api::rocgraph_graph_t*             graph_;
        size_t                                         n_clusters_;
        size_t                                         n_eigenvectors_;
        double                                         evs_tolerance_;
        int                                            evs_max_iterations_;
        double                                         k_means_tolerance_;
        int                                            k_means_max_iterations_;
        bool                                           do_expensive_check_;
        rocgraph::c_api::rocgraph_clustering_result_t* result_{};

        spectral_clustering_functor(::rocgraph_handle_t const* handle,
                                    ::rocgraph_graph_t*        graph,
                                    size_t                     n_clusters,
                                    size_t                     n_eigenvectors,
                                    double                     evs_tolerance,
                                    int                        evs_max_iterations,
                                    double                     k_means_tolerance,
                                    int                        k_means_max_iterations,
                                    bool                       do_expensive_check)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , n_clusters_(n_clusters)
            , n_eigenvectors_(n_eigenvectors)
            , evs_tolerance_(evs_tolerance)
            , evs_max_iterations_(evs_max_iterations)
            , k_means_tolerance_(k_means_tolerance)
            , k_means_max_iterations_(k_means_max_iterations)
            , do_expensive_check_(do_expensive_check)
        {
        }

        template <typename vertex_t,
                  typename edge_t,
                  typename weight_t,
                  typename edge_type_type_t,
                  bool store_transposed,
                  bool multi_gpu>
        void operator()()
        {
            if constexpr(!rocgraph::is_candidate<vertex_t, edge_t, weight_t>::value)
            {
                unsupported();
            }
            else if constexpr(multi_gpu)
            {
                unsupported();
            }
            else if constexpr(!std::is_same_v<edge_t, int32_t>)
            {
                unsupported();
            }
            else
            {
                // spectral clustering expects store_transposed == false
                if constexpr(store_transposed)
                {
                    status_ = rocgraph::c_api::
                        transpose_storage<vertex_t, edge_t, weight_t, store_transposed, false>(
                            handle_, graph_, error_.get());
                    if(status_ != rocgraph_status_success)
                        return;
                }

                auto graph = reinterpret_cast<rocgraph::graph_t<vertex_t, edge_t, false, false>*>(
                    graph_->graph_);

                auto edge_weights = reinterpret_cast<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, false, false>,
                    weight_t>*>(graph_->edge_weights_);

                auto number_map
                    = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

                auto graph_view          = graph->view();
                auto edge_partition_view = graph_view.local_edge_partition_view();

                rmm::device_uvector<weight_t> tmp_weights(0, handle_.get_stream());
                if(edge_weights == nullptr)
                {
                    tmp_weights.resize(edge_partition_view.indices().size(), handle_.get_stream());
                    rocgraph::detail::scalar_fill(
                        handle_, tmp_weights.data(), tmp_weights.size(), weight_t{1});
                }

                rocgraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> legacy_graph_view(
                    const_cast<edge_t*>(edge_partition_view.offsets().data()),
                    const_cast<vertex_t*>(edge_partition_view.indices().data()),
                    (edge_weights == nullptr)
                        ? tmp_weights.data()
                        : const_cast<weight_t*>(edge_weights->view().value_firsts().front()),
                    edge_partition_view.offsets().size() - 1,
                    edge_partition_view.indices().size());

                rmm::device_uvector<vertex_t> clusters(
                    graph_view.local_vertex_partition_range_size(), handle_.get_stream());

                rocgraph::ext_raft::spectralModularityMaximization(
                    legacy_graph_view,
                    static_cast<vertex_t>(n_clusters_),
                    static_cast<vertex_t>(n_eigenvectors_),
                    static_cast<weight_t>(evs_tolerance_),
                    evs_max_iterations_,
                    static_cast<weight_t>(k_means_tolerance_),
                    k_means_max_iterations_,
                    clusters.data());

                rmm::device_uvector<vertex_t> vertices(
                    graph_view.local_vertex_partition_range_size(), handle_.get_stream());
                raft::copy(
                    vertices.data(), number_map->data(), vertices.size(), handle_.get_stream());

                result_ = new rocgraph::c_api::rocgraph_clustering_result_t{
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(vertices,
                                                                             graph_->vertex_type_),
                    new rocgraph::c_api::rocgraph_type_erased_device_array_t(clusters,
                                                                             graph_->vertex_type_)};
            }
        }
    };

    struct analyze_clustering_ratio_cut_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_;
        rocgraph::c_api::rocgraph_graph_t*                               graph_{nullptr};
        size_t                                                           n_clusters_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* vertices_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* clusters_{};
        double                                                           result_{};

        analyze_clustering_ratio_cut_functor(
            ::rocgraph_handle_t const*                        handle,
            ::rocgraph_graph_t*                               graph,
            size_t                                            n_clusters,
            ::rocgraph_type_erased_device_array_view_t const* vertices,
            ::rocgraph_type_erased_device_array_view_t const* clusters)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , n_clusters_(n_clusters)
            , vertices_(reinterpret_cast<
                        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(vertices))
            , clusters_(reinterpret_cast<
                        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(clusters))
        {
        }

        template <typename vertex_t,
                  typename edge_t,
                  typename weight_t,
                  typename edge_type_type_t,
                  bool store_transposed,
                  bool multi_gpu>
        void operator()()
        {
            if constexpr(!rocgraph::is_candidate<vertex_t, edge_t, weight_t>::value)
            {
                unsupported();
            }
            else if constexpr(multi_gpu)
            {
                unsupported();
            }
            else if constexpr(!std::is_same_v<edge_t, int32_t>)
            {
                unsupported();
            }
            else
            {
                // analyze clustering expects store_transposed == false
                if constexpr(store_transposed)
                {
                    status_ = rocgraph::c_api::
                        transpose_storage<vertex_t, edge_t, weight_t, store_transposed, false>(
                            handle_, graph_, error_.get());
                    if(status_ != rocgraph_status_success)
                        return;
                }

                auto graph = reinterpret_cast<rocgraph::graph_t<vertex_t, edge_t, false, false>*>(
                    graph_->graph_);

                auto edge_weights = reinterpret_cast<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, false, false>,
                    weight_t>*>(graph_->edge_weights_);

                auto number_map
                    = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

                auto graph_view          = graph->view();
                auto edge_partition_view = graph_view.local_edge_partition_view();

                rmm::device_uvector<weight_t> tmp_weights(0, handle_.get_stream());
                if(edge_weights == nullptr)
                {
                    tmp_weights.resize(edge_partition_view.indices().size(), handle_.get_stream());
                    rocgraph::detail::scalar_fill(
                        handle_, tmp_weights.data(), tmp_weights.size(), weight_t{1});
                }

                rocgraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> legacy_graph_view(
                    const_cast<edge_t*>(edge_partition_view.offsets().data()),
                    const_cast<vertex_t*>(edge_partition_view.indices().data()),
                    (edge_weights == nullptr)
                        ? tmp_weights.data()
                        : const_cast<weight_t*>(edge_weights->view().value_firsts().front()),
                    edge_partition_view.offsets().size() - 1,
                    edge_partition_view.indices().size());

                weight_t score;

                if(rocgraph::detail::is_equal(
                       handle_,
                       raft::device_span<vertex_t const>{vertices_->as_type<vertex_t const>(),
                                                         vertices_->size_},
                       raft::device_span<vertex_t const>{number_map->data(), number_map->size()}))
                {
                    rocgraph::ext_raft::analyzeClustering_ratio_cut(
                        legacy_graph_view, n_clusters_, clusters_->as_type<vertex_t>(), &score);
                }
                else
                {
                    rmm::device_uvector<vertex_t> tmp_v(vertices_->size_, handle_.get_stream());
                    rmm::device_uvector<vertex_t> tmp_c(clusters_->size_, handle_.get_stream());

                    raft::copy(tmp_v.data(),
                               vertices_->as_type<vertex_t>(),
                               vertices_->size_,
                               handle_.get_stream());
                    raft::copy(tmp_c.data(),
                               clusters_->as_type<vertex_t>(),
                               clusters_->size_,
                               handle_.get_stream());

                    rocgraph::renumber_ext_vertices<vertex_t, false>(
                        handle_,
                        tmp_v.data(),
                        tmp_v.size(),
                        number_map->data(),
                        graph_view.local_vertex_partition_range_first(),
                        graph_view.local_vertex_partition_range_last(),
                        false);

                    rocgraph::c_api::detail::sort_by_key(
                        handle_,
                        raft::device_span<vertex_t>{tmp_v.data(), tmp_v.size()},
                        raft::device_span<vertex_t>{tmp_c.data(), tmp_c.size()});

                    rocgraph::ext_raft::analyzeClustering_ratio_cut(
                        legacy_graph_view, n_clusters_, tmp_c.data(), &score);
                }

                result_ = static_cast<double>(score);
            }
        }
    };

    struct analyze_clustering_edge_cut_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_;
        rocgraph::c_api::rocgraph_graph_t*                               graph_{nullptr};
        size_t                                                           n_clusters_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* vertices_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* clusters_{};
        double                                                           result_{};

        analyze_clustering_edge_cut_functor(
            ::rocgraph_handle_t const*                        handle,
            ::rocgraph_graph_t*                               graph,
            size_t                                            n_clusters,
            ::rocgraph_type_erased_device_array_view_t const* vertices,
            ::rocgraph_type_erased_device_array_view_t const* clusters)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , n_clusters_(n_clusters)
            , vertices_(reinterpret_cast<
                        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(vertices))
            , clusters_(reinterpret_cast<
                        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(clusters))
        {
        }

        template <typename vertex_t,
                  typename edge_t,
                  typename weight_t,
                  typename edge_type_type_t,
                  bool store_transposed,
                  bool multi_gpu>
        void operator()()
        {
            if constexpr(!rocgraph::is_candidate<vertex_t, edge_t, weight_t>::value)
            {
                unsupported();
            }
            else if constexpr(multi_gpu)
            {
                unsupported();
            }
            else if constexpr(!std::is_same_v<edge_t, int32_t>)
            {
                unsupported();
            }
            else
            {
                // analyze clustering expects store_transposed == false
                if constexpr(store_transposed)
                {
                    status_ = rocgraph::c_api::
                        transpose_storage<vertex_t, edge_t, weight_t, store_transposed, false>(
                            handle_, graph_, error_.get());
                    if(status_ != rocgraph_status_success)
                        return;
                }

                auto graph = reinterpret_cast<rocgraph::graph_t<vertex_t, edge_t, false, false>*>(
                    graph_->graph_);

                auto edge_weights = reinterpret_cast<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, false, false>,
                    weight_t>*>(graph_->edge_weights_);

                auto number_map
                    = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

                auto graph_view          = graph->view();
                auto edge_partition_view = graph_view.local_edge_partition_view();

                rmm::device_uvector<weight_t> tmp_weights(0, handle_.get_stream());
                if(edge_weights == nullptr)
                {
                    tmp_weights.resize(edge_partition_view.indices().size(), handle_.get_stream());
                    rocgraph::detail::scalar_fill(
                        handle_, tmp_weights.data(), tmp_weights.size(), weight_t{1});
                }

                rocgraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> legacy_graph_view(
                    const_cast<edge_t*>(edge_partition_view.offsets().data()),
                    const_cast<vertex_t*>(edge_partition_view.indices().data()),
                    (edge_weights == nullptr)
                        ? tmp_weights.data()
                        : const_cast<weight_t*>(edge_weights->view().value_firsts().front()),
                    edge_partition_view.offsets().size() - 1,
                    edge_partition_view.indices().size());

                weight_t score;

                if(rocgraph::detail::is_equal(
                       handle_,
                       raft::device_span<vertex_t const>{vertices_->as_type<vertex_t const>(),
                                                         vertices_->size_},
                       raft::device_span<vertex_t const>{number_map->data(), number_map->size()}))
                {
                    rocgraph::ext_raft::analyzeClustering_edge_cut(
                        legacy_graph_view, n_clusters_, clusters_->as_type<vertex_t>(), &score);
                }
                else
                {
                    rmm::device_uvector<vertex_t> tmp_v(vertices_->size_, handle_.get_stream());
                    rmm::device_uvector<vertex_t> tmp_c(clusters_->size_, handle_.get_stream());

                    raft::copy(tmp_v.data(),
                               vertices_->as_type<vertex_t>(),
                               vertices_->size_,
                               handle_.get_stream());
                    raft::copy(tmp_c.data(),
                               clusters_->as_type<vertex_t>(),
                               clusters_->size_,
                               handle_.get_stream());

                    rocgraph::renumber_ext_vertices<vertex_t, false>(
                        handle_,
                        tmp_v.data(),
                        tmp_v.size(),
                        number_map->data(),
                        graph_view.local_vertex_partition_range_first(),
                        graph_view.local_vertex_partition_range_last(),
                        false);

                    rocgraph::c_api::detail::sort_by_key(
                        handle_,
                        raft::device_span<vertex_t>{tmp_v.data(), tmp_v.size()},
                        raft::device_span<vertex_t>{tmp_c.data(), tmp_c.size()});

                    rocgraph::ext_raft::analyzeClustering_edge_cut(
                        legacy_graph_view, n_clusters_, tmp_c.data(), &score);
                }

                result_ = static_cast<double>(score);
            }
        }
    };

    struct analyze_clustering_modularity_functor : public rocgraph::c_api::abstract_functor
    {
        raft::handle_t const&                                            handle_;
        rocgraph::c_api::rocgraph_graph_t*                               graph_{nullptr};
        size_t                                                           n_clusters_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* vertices_{};
        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const* clusters_{};
        double                                                           result_{};

        analyze_clustering_modularity_functor(
            ::rocgraph_handle_t const*                        handle,
            ::rocgraph_graph_t*                               graph,
            size_t                                            n_clusters,
            ::rocgraph_type_erased_device_array_view_t const* vertices,
            ::rocgraph_type_erased_device_array_view_t const* clusters)
            : abstract_functor()
            , handle_(*handle->get_raft_handle())
            , graph_(reinterpret_cast<rocgraph::c_api::rocgraph_graph_t*>(graph))
            , n_clusters_(n_clusters)
            , vertices_(reinterpret_cast<
                        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(vertices))
            , clusters_(reinterpret_cast<
                        rocgraph::c_api::rocgraph_type_erased_device_array_view_t const*>(clusters))
        {
        }

        template <typename vertex_t,
                  typename edge_t,
                  typename weight_t,
                  typename edge_type_type_t,
                  bool store_transposed,
                  bool multi_gpu>
        void operator()()
        {
            if constexpr(!rocgraph::is_candidate<vertex_t, edge_t, weight_t>::value)
            {
                unsupported();
            }
            else if constexpr(multi_gpu)
            {
                unsupported();
            }
            else if constexpr(!std::is_same_v<edge_t, int32_t>)
            {
                unsupported();
            }
            else
            {
                // analyze clustering expects store_transposed == false
                if constexpr(store_transposed)
                {
                    status_ = rocgraph::c_api::
                        transpose_storage<vertex_t, edge_t, weight_t, store_transposed, false>(
                            handle_, graph_, error_.get());
                    if(status_ != rocgraph_status_success)
                        return;
                }

                auto graph = reinterpret_cast<rocgraph::graph_t<vertex_t, edge_t, false, false>*>(
                    graph_->graph_);

                auto edge_weights = reinterpret_cast<rocgraph::edge_property_t<
                    rocgraph::graph_view_t<vertex_t, edge_t, false, false>,
                    weight_t>*>(graph_->edge_weights_);

                auto number_map
                    = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

                auto graph_view          = graph->view();
                auto edge_partition_view = graph_view.local_edge_partition_view();

                rmm::device_uvector<weight_t> tmp_weights(0, handle_.get_stream());
                if(edge_weights == nullptr)
                {
                    tmp_weights.resize(edge_partition_view.indices().size(), handle_.get_stream());
                    rocgraph::detail::scalar_fill(
                        handle_, tmp_weights.data(), tmp_weights.size(), weight_t{1});
                }

                rocgraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> legacy_graph_view(
                    const_cast<edge_t*>(edge_partition_view.offsets().data()),
                    const_cast<vertex_t*>(edge_partition_view.indices().data()),
                    (edge_weights == nullptr)
                        ? tmp_weights.data()
                        : const_cast<weight_t*>(edge_weights->view().value_firsts().front()),
                    edge_partition_view.offsets().size() - 1,
                    edge_partition_view.indices().size());

                weight_t score;

                if(rocgraph::detail::is_equal(
                       handle_,
                       raft::device_span<vertex_t const>{vertices_->as_type<vertex_t const>(),
                                                         vertices_->size_},
                       raft::device_span<vertex_t const>{number_map->data(), number_map->size()}))
                {
                    rocgraph::ext_raft::analyzeClustering_modularity(
                        legacy_graph_view, n_clusters_, clusters_->as_type<vertex_t>(), &score);
                }
                else
                {
                    rmm::device_uvector<vertex_t> tmp_v(vertices_->size_, handle_.get_stream());
                    rmm::device_uvector<vertex_t> tmp_c(clusters_->size_, handle_.get_stream());

                    raft::copy(tmp_v.data(),
                               vertices_->as_type<vertex_t>(),
                               vertices_->size_,
                               handle_.get_stream());
                    raft::copy(tmp_c.data(),
                               clusters_->as_type<vertex_t>(),
                               clusters_->size_,
                               handle_.get_stream());

                    rocgraph::renumber_ext_vertices<vertex_t, false>(
                        handle_,
                        tmp_v.data(),
                        tmp_v.size(),
                        number_map->data(),
                        graph_view.local_vertex_partition_range_first(),
                        graph_view.local_vertex_partition_range_last(),
                        false);

                    rocgraph::c_api::detail::sort_by_key(
                        handle_,
                        raft::device_span<vertex_t>{tmp_v.data(), tmp_v.size()},
                        raft::device_span<vertex_t>{tmp_c.data(), tmp_c.size()});

                    rocgraph::ext_raft::analyzeClustering_modularity(
                        legacy_graph_view, n_clusters_, tmp_c.data(), &score);
                }

                result_ = static_cast<double>(score);
            }
        }
    };

} // namespace

extern "C" {

rocgraph_type_erased_device_array_view_t*
    rocgraph_clustering_result_get_vertices(rocgraph_clustering_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_clustering_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->vertices_->view());
}

rocgraph_type_erased_device_array_view_t*
    rocgraph_clustering_result_get_clusters(rocgraph_clustering_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_clustering_result_t*>(result);
    return reinterpret_cast<rocgraph_type_erased_device_array_view_t*>(
        internal_pointer->clusters_->view());
}

void rocgraph_clustering_result_free(rocgraph_clustering_result_t* result)
{
    auto internal_pointer
        = reinterpret_cast<rocgraph::c_api::rocgraph_clustering_result_t*>(result);
    delete internal_pointer->vertices_;
    delete internal_pointer->clusters_;
    delete internal_pointer;
}

rocgraph_status rocgraph_balanced_cut_clustering(const rocgraph_handle_t* handle,
                                                 rocgraph_graph_t*        graph,
                                                 size_t                   n_clusters,
                                                 size_t                   n_eigenvectors,
                                                 double                   evs_tolerance,
                                                 int                      evs_max_iterations,
                                                 double                   k_means_tolerance,
                                                 int                      k_means_max_iterations,
                                                 rocgraph_bool            do_expensive_check,
                                                 rocgraph_clustering_result_t** result,
                                                 rocgraph_error_t**             error)
{
    balanced_cut_clustering_functor functor(handle,
                                            graph,
                                            n_clusters,
                                            n_eigenvectors,
                                            evs_tolerance,
                                            evs_max_iterations,
                                            k_means_tolerance,
                                            k_means_max_iterations,
                                            do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}

rocgraph_status rocgraph_spectral_modularity_maximization(const rocgraph_handle_t* handle,
                                                          rocgraph_graph_t*        graph,
                                                          size_t                   n_clusters,
                                                          size_t                   n_eigenvectors,
                                                          double                   evs_tolerance,
                                                          int           evs_max_iterations,
                                                          double        k_means_tolerance,
                                                          int           k_means_max_iterations,
                                                          rocgraph_bool do_expensive_check,
                                                          rocgraph_clustering_result_t** result,
                                                          rocgraph_error_t**             error)
{
    spectral_clustering_functor functor(handle,
                                        graph,
                                        n_clusters,
                                        n_eigenvectors,
                                        evs_tolerance,
                                        evs_max_iterations,
                                        k_means_tolerance,
                                        k_means_max_iterations,
                                        do_expensive_check);

    return rocgraph::c_api::run_algorithm(graph, functor, result, error);
}

rocgraph_status
    rocgraph_analyze_clustering_modularity(const rocgraph_handle_t* handle,
                                           rocgraph_graph_t*        graph,
                                           size_t                   n_clusters,
                                           const rocgraph_type_erased_device_array_view_t* vertices,
                                           const rocgraph_type_erased_device_array_view_t* clusters,
                                           double*                                         score,
                                           rocgraph_error_t**                              error)
{
    analyze_clustering_modularity_functor functor(handle, graph, n_clusters, vertices, clusters);

    return rocgraph::c_api::run_algorithm(graph, functor, score, error);
}

rocgraph_status
    rocgraph_analyze_clustering_edge_cut(const rocgraph_handle_t*                        handle,
                                         rocgraph_graph_t*                               graph,
                                         size_t                                          n_clusters,
                                         const rocgraph_type_erased_device_array_view_t* vertices,
                                         const rocgraph_type_erased_device_array_view_t* clusters,
                                         double*                                         score,
                                         rocgraph_error_t**                              error)
{
    analyze_clustering_edge_cut_functor functor(handle, graph, n_clusters, vertices, clusters);

    return rocgraph::c_api::run_algorithm(graph, functor, score, error);
}

rocgraph_status
    rocgraph_analyze_clustering_ratio_cut(const rocgraph_handle_t* handle,
                                          rocgraph_graph_t*        graph,
                                          size_t                   n_clusters,
                                          const rocgraph_type_erased_device_array_view_t* vertices,
                                          const rocgraph_type_erased_device_array_view_t* clusters,
                                          double*                                         score,
                                          rocgraph_error_t**                              error)
{
    analyze_clustering_ratio_cut_functor functor(handle, graph, n_clusters, vertices, clusters);

    return rocgraph::c_api::run_algorithm(graph, functor, score, error);
}

} // extern C
