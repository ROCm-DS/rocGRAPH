// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

// Andrei Schaffer, aschaffer@nvidia.com
//
#include "random_walks.cuh"

namespace rocgraph
{
    // template explicit instantiation directives (EIDir's):
    //
    // SG FP32{
    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<float>,
                        rmm::device_uvector<int32_t>>
        random_walks(raft::handle_t const&                                      handle,
                     graph_view_t<int32_t, int32_t, false, false> const&        gview,
                     std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
                     int32_t const*                                             ptr_d_start,
                     int32_t                                                    num_paths,
                     int32_t                                                    max_depth,
                     bool                                                       use_padding,
                     std::unique_ptr<sampling_params_t>                         sampling_strategy);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<float>,
                        rmm::device_uvector<int64_t>>
        random_walks(raft::handle_t const&                                      handle,
                     graph_view_t<int32_t, int64_t, false, false> const&        gview,
                     std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
                     int32_t const*                                             ptr_d_start,
                     int64_t                                                    num_paths,
                     int64_t                                                    max_depth,
                     bool                                                       use_padding,
                     std::unique_ptr<sampling_params_t>                         sampling_strategy);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<float>,
                        rmm::device_uvector<int64_t>>
        random_walks(raft::handle_t const&                                      handle,
                     graph_view_t<int64_t, int64_t, false, false> const&        gview,
                     std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
                     int64_t const*                                             ptr_d_start,
                     int64_t                                                    num_paths,
                     int64_t                                                    max_depth,
                     bool                                                       use_padding,
                     std::unique_ptr<sampling_params_t>                         sampling_strategy);
    //}
    //
    // SG FP64{
    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<double>,
                        rmm::device_uvector<int32_t>>
        random_walks(raft::handle_t const&                                       handle,
                     graph_view_t<int32_t, int32_t, false, false> const&         gview,
                     std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
                     int32_t const*                                              ptr_d_start,
                     int32_t                                                     num_paths,
                     int32_t                                                     max_depth,
                     bool                                                        use_padding,
                     std::unique_ptr<sampling_params_t>                          sampling_strategy);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<double>,
                        rmm::device_uvector<int64_t>>
        random_walks(raft::handle_t const&                                       handle,
                     graph_view_t<int32_t, int64_t, false, false> const&         gview,
                     std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
                     int32_t const*                                              ptr_d_start,
                     int64_t                                                     num_paths,
                     int64_t                                                     max_depth,
                     bool                                                        use_padding,
                     std::unique_ptr<sampling_params_t>                          sampling_strategy);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<double>,
                        rmm::device_uvector<int64_t>>
        random_walks(raft::handle_t const&                                       handle,
                     graph_view_t<int64_t, int64_t, false, false> const&         gview,
                     std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
                     int64_t const*                                              ptr_d_start,
                     int64_t                                                     num_paths,
                     int64_t                                                     max_depth,
                     bool                                                        use_padding,
                     std::unique_ptr<sampling_params_t>                          sampling_strategy);
    //}

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>>
        convert_paths_to_coo(raft::handle_t const& handle,
                             int32_t               coalesced_sz_v,
                             int32_t               num_paths,
                             rmm::device_buffer&&  d_coalesced_v,
                             rmm::device_buffer&&  d_sizes);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int64_t>>
        convert_paths_to_coo(raft::handle_t const& handle,
                             int64_t               coalesced_sz_v,
                             int64_t               num_paths,
                             rmm::device_buffer&&  d_coalesced_v,
                             rmm::device_buffer&&  d_sizes);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>>
        convert_paths_to_coo(raft::handle_t const& handle,
                             int64_t               coalesced_sz_v,
                             int64_t               num_paths,
                             rmm::device_buffer&&  d_coalesced_v,
                             rmm::device_buffer&&  d_sizes);

    template std::tuple<rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>,
                        rmm::device_uvector<int32_t>>
        query_rw_sizes_offsets(raft::handle_t const& handle,
                               int32_t               num_paths,
                               int32_t const*        ptr_d_sizes);

    template std::tuple<rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>,
                        rmm::device_uvector<int64_t>>
        query_rw_sizes_offsets(raft::handle_t const& handle,
                               int64_t               num_paths,
                               int64_t const*        ptr_d_sizes);

} // namespace rocgraph
