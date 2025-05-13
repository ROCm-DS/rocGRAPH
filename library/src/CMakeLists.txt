# Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# rocGRAPH source

find_package(HIP REQUIRED CONFIG PATHS ${HIP_PATH})
find_package(hipcub REQUIRED CONFIG PATHS ${ROCM_PATH}/hipcub)
find_package(hiprand REQUIRED CONFIG PATHS ${ROCM_PATH}/hiprand)

#
# NB check the flags here
#
# NB: technical debt, remove when no longer required -Wno-constant-conversion is
# to suppress warnings from raft about bitmask length down conversion from 64
# bits in rocm-raft to 32 bits in rocGraph (since cuGraph was designed and
# hardcoded in many places to a warp size of 32) -Wno-unknown-pragmas is to
# suppress warnings from libhipcxx and rocm-raft (mostly the former) reporting
# on pragmas that are not valid in# in clang, particularly some GCC and NVCC
# pragmas. -Wno-unknown-warning-option, suppress warnings about a particular GCC
# pragma in rocm-rmm, i.e. #pragma GCC diagnostic ignored
# "-Wnon-template-friend" -Wno-sign-conversion, suppress warnings about an
# implicit conversion from a long to an unsiged long in type_axis.cxx:290
# observed in build/*/hipbenc-src/nvbench -Wno-unused-but-set-variable suppress
# warnings about a variable 'i' set but not used in _deps/raft-src
# (mst_solver_inl.cuh:290)
set(ROCGRAPH_CXX_FLAGS
    -DNO_ROCGRAPH_OPS
    -DRAFT_HIDE_DEPRECATION_WARNINGS
    -DHIP_ENABLE_WARP_SYNC_BUILTINS=1
    -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
    -DRAFT_SYSTEM_LITTLE_ENDIAN=1
    -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP
    -DTHRUST_DISABLE_ABI_NAMESPACE
    -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP
    -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR
    -Drocgraph_EXPORTS
    -Wno-deprecated-pragma
    -Wno-reorder-ctor
    -Wno-unused-variable
    -Wno-unused-lambda-capture
    -Wno-unused-local-typedef
    -Wno-constant-conversion
    -Wno-unknown-pragmas
    -Wno-unknown-warning-option
    -Wno-sign-conversion
    -Wno-unused-but-set-variable)

set(ROCGRAPH_HIP_FLAGS "")

if(ROCGRAPH_USE_WARPSIZE_32)
  list(APPEND ROCGRAPH_CXX_FLAGS -DROCGRAPH_USE_WARPSIZE_32)
endif()

if(ROCGRAPH_SPECTRAL_IMPLEMENTED)
  list(APPEND ROCGRAPH_CXX_FLAGS -DROCGRAPH_SPECTRAL_IMPLEMENTED)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # list(APPEND ROCGRAPH_CXX_FLAGS -Werror -Wno-error=deprecated-declarations)
  list(APPEND ROCGRAPH_CXX_FLAGS -Wno-error=deprecated-declarations)
endif()

# _LIBCUDACXX_ALLOW_UNSUPPORTED_ARCHITECTURE is to allow libhipcxx to build for
# architectures we don't know the real time clock rate for. See
# libhipcxx/include/cuda/std/detail/libcxx/include/__config
list(APPEND ROCGRAPH_CXX_FLAGS -D_LIBCUDACXX_ALLOW_UNSUPPORTED_ARCHITECTURE)

list(APPEND ROCGRAPH_CXX_FLAGS -DUSE_HIP)
list(APPEND ROCGRAPH_CXX_FLAGS -Wno-c++11-narrowing)
list(APPEND ROCGRAPH_CXX_FLAGS -Wno-unused-result)
list(APPEND ROCGRAPH_CXX_FLAGS -I${HIP_INCLUDE_DIRS})
list(APPEND ROCGRAPH_CXX_FLAGS -I${HIP_INCLUDE_DIRS}/hipcub)
list(APPEND ROCGRAPH_CXX_FLAGS -I${HIP_INCLUDE_DIRS}/hiprand)
list(APPEND ROCGRAPH_CXX_FLAGS -cxx-isystem
     ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/src/)
list(APPEND ROCGRAPH_CXX_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/src/include)
list(APPEND ROCGRAPH_CXX_FLAGS
     -I${CMAKE_CURRENT_SOURCE_DIR}/src/sparse_utilities/include)

# Option to enable line info in HIP device compilation to allow introspection
# when profiling / memchecking
if(CMAKE_HIP_LINEINFO)
  list(APPEND ROCGRAPH_HIP_FLAGS -lineinfo)
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
  message(STATUS "Building with debugging flags")
  list(APPEND ROCGRAPH_HIP_FLAGS -G -Xcompiler=-rdynamic)
endif()

set(rocgraph_source
    src/sparse_utilities/handle.cpp
    src/sparse_utilities/enum_utils.cpp
    src/sparse_utilities/rocgraph_auxiliary.cpp
    src/sparse_utilities/rocgraph_envariables.cpp
    src/sparse_utilities/rocgraph_memstat.cpp
    src/sparse_utilities/rocgraph_debug.cpp
    src/sparse_utilities/rocgraph_argdescr.cpp
    src/sparse_utilities/rocgraph_message.cpp
    src/sparse_utilities/to_string.cpp)

list(APPEND rocgraph_source ${ROCGRAPH_CPP_SOURCE})

option(USE_HIP "..." ON)

set(HIPROCGRAPH_SOURCES_C
    src/c_api/rocgraph_allgather.cpp
    src/c_api/rocgraph_array.cpp
    src/c_api/rocgraph_betweenness_centrality.cpp
    src/c_api/rocgraph_bfs.cpp
    src/c_api/rocgraph_capi_helper.cpp
    src/c_api/rocgraph_centrality_result.cpp
    src/c_api/rocgraph_core_number.cpp
    src/c_api/rocgraph_core_result.cpp
    src/c_api/rocgraph_degrees.cpp
    src/c_api/rocgraph_degrees_result.cpp
    src/c_api/rocgraph_ecg.cpp
    src/c_api/rocgraph_eigenvector_centrality.cpp
    src/c_api/rocgraph_error.cpp
    src/c_api/rocgraph_extract_ego.cpp
    src/c_api/rocgraph_extract_paths.cpp
    src/c_api/rocgraph_graph_functions.cpp
    src/c_api/rocgraph_graph_generators.cpp
    src/c_api/rocgraph_graph_helper_sg.cpp
    src/c_api/rocgraph_graph_sg.cpp
    src/c_api/rocgraph_hierarchical_clustering_result.cpp
    src/c_api/rocgraph_hits.cpp
    src/c_api/rocgraph_induced_subgraph.cpp
    src/c_api/rocgraph_induced_subgraph_result.cpp
    src/c_api/rocgraph_katz.cpp
    src/c_api/rocgraph_k_core.cpp
    src/c_api/rocgraph_labeling_result.cpp
    src/c_api/rocgraph_leiden.cpp
    src/c_api/rocgraph_louvain.cpp
    src/c_api/rocgraph_pagerank.cpp
    src/c_api/rocgraph_random.cpp
    src/c_api/rocgraph_random_walks.cpp
    src/c_api/rocgraph_handle.cpp
    src/c_api/rocgraph_similarity.cpp
    src/c_api/rocgraph_sssp.cpp
    src/c_api/rocgraph_strongly_connected_components.cpp
    src/c_api/rocgraph_triangle_count.cpp
    src/c_api/rocgraph_uniform_neighbor_sampling.cpp
    src/c_api/rocgraph_weakly_connected_components.cpp)

set(HIPROCGRAPH_MULTI_SOURCES_C src/c_api/rocgraph_graph_helper_mg.cpp
                                src/c_api/rocgraph_graph_mg.cpp)

set(HIPROCGRAPH_FAIL_SOURCES_C src/c_api/rocgraph_legacy_ecg.cpp)

set(HIPROCGRAPH_SOURCES_CPP
    src/c_api/rocgraph_community_algorithms.cpp
    src/c_api/rocgraph_k_truss.cpp
    src/centrality/betweenness_centrality_sg.cpp
    src/centrality/eigenvector_centrality_sg.cpp
    src/centrality/katz_centrality_sg.cpp
    src/community/detail/common_methods_sg.cpp
    src/community/detail/maximal_independent_moves_sg.cpp
    src/community/detail/refine_sg.cpp
    src/community/ecg_sg.cpp
    src/community/edge_triangle_count_sg.cpp
    src/community/egonet_sg.cpp
    src/community/k_truss_sg.cpp
    src/community/leiden_sg.cpp
    src/community/louvain_sg.cpp
    src/community/triangle_count_sg.cpp
    src/components/legacy/connectivity.cpp
    src/components/mis_sg.cpp
    src/components/vertex_coloring_sg.cpp
    src/components/weakly_connected_components_sg.cpp
    src/converters/legacy/COOtoCSR.cpp
    src/cores/core_number_sg.cpp
    src/cores/k_core_sg.cpp
    src/detail/collect_comm_wrapper.cpp
    src/detail/collect_local_vertex_values.cpp
    src/detail/groupby_and_count.cpp
    src/detail/permute_range.cpp
    src/detail/utility_wrappers.cpp
    src/generators/erdos_renyi_generator.cpp
    src/generators/generate_bipartite_rmat_edgelist.cpp
    src/generators/generate_rmat_edgelist.cpp
    src/generators/generator_tools.cpp
    src/generators/simple_generators.cpp
    src/layout/legacy/force_atlas2.cpp
    src/linear_assignment/legacy/hungarian.cpp
    src/link_analysis/hits_sg.cpp
    src/link_analysis/pagerank_sg.cpp
    src/link_prediction/jaccard_sg.cpp
    src/link_prediction/overlap_sg.cpp
    src/link_prediction/sorensen_sg.cpp
    src/sampling/detail/gather_one_hop_edgelist_sg.cpp
    src/sampling/detail/prepare_next_frontier_sg.cpp
    src/sampling/detail/remove_visited_vertices_from_frontier.cpp
    src/sampling/detail/sample_edges_sg.cpp
    src/sampling/random_walks.cpp
    src/sampling/random_walks_sg.cpp
    src/sampling/renumber_sampled_edgelist_sg.cpp
    src/sampling/sampling_post_processing_sg.cpp
    src/sampling/uniform_neighbor_sampling_sg.cpp
    src/structure/coarsen_graph_sg.cpp
    src/structure/create_graph_from_edgelist_sg.cpp
    src/structure/decompress_to_edgelist_sg.cpp
    src/structure/graph_sg.cpp
    src/structure/graph_view_sg.cpp
    src/structure/graph_weight_utils_sg.cpp
    src/structure/induced_subgraph_sg.cpp
    src/structure/legacy/graph.cpp
    src/structure/relabel_sg.cpp
    src/structure/remove_multi_edges.cpp
    src/structure/remove_self_loops.cpp
    src/structure/renumber_edgelist_sg.cpp
    src/structure/renumber_utils_sg.cpp
    src/structure/select_random_vertices_sg.cpp
    src/structure/symmetrize_edgelist_sg.cpp
    src/structure/symmetrize_graph_sg.cpp
    src/structure/transpose_graph_sg.cpp
    src/structure/transpose_graph_storage_sg.cpp
    src/traversal/bfs_sg.cpp
    src/traversal/extract_bfs_paths_sg.cpp
    src/traversal/k_hop_nbrs_sg.cpp
    src/traversal/sssp_sg.cpp
    src/tree/legacy/mst.cpp
    src/utilities/path_retrieval.cpp
    src/utilities/shuffle_vertex_pairs.cpp
    src/utilities/shuffle_vertices.cpp)

set(HIPROCGRAPH_MULTI_SOURCES_CPP
    src/centrality/betweenness_centrality_mg.cpp
    src/centrality/eigenvector_centrality_mg.cpp
    src/centrality/katz_centrality_mg.cpp
    src/community/detail/common_methods_mg.cpp
    src/community/detail/maximal_independent_moves_mg.cpp
    src/community/detail/refine_mg.cpp
    src/community/ecg_mg.cpp
    src/community/egonet_mg.cpp
    src/community/leiden_mg.cpp
    src/community/louvain_mg.cpp
    src/community/triangle_count_mg.cpp
    src/components/mis_mg.cpp
    src/components/vertex_coloring_mg.cpp
    src/components/weakly_connected_components_mg.cpp
    src/cores/core_number_mg.cpp
    src/cores/k_core_mg.cpp
    src/link_analysis/hits_mg.cpp
    src/link_analysis/pagerank_mg.cpp
    src/link_prediction/jaccard_mg.cpp
    src/link_prediction/overlap_mg.cpp
    src/link_prediction/sorensen_mg.cpp
    src/mtmg/vertex_result.cpp
    src/sampling/detail/gather_one_hop_edgelist_mg.cpp
    src/sampling/detail/prepare_next_frontier_mg.cpp
    src/sampling/detail/sample_edges_mg.cpp
    src/sampling/detail/shuffle_and_organize_output_mg.cpp
    src/sampling/random_walks_mg.cpp
    src/sampling/uniform_neighbor_sampling_mg.cpp
    src/structure/coarsen_graph_mg.cpp
    src/structure/create_graph_from_edgelist_mg.cpp
    src/structure/decompress_to_edgelist_mg.cpp
    src/structure/graph_mg.cpp
    src/structure/graph_view_mg.cpp
    src/structure/graph_weight_utils_mg.cpp
    src/structure/induced_subgraph_mg.cpp
    src/structure/relabel_mg.cpp
    src/structure/renumber_edgelist_mg.cpp
    src/structure/renumber_utils_mg.cpp
    src/structure/select_random_vertices_mg.cpp
    src/structure/symmetrize_edgelist_mg.cpp
    src/structure/symmetrize_graph_mg.cpp
    src/structure/transpose_graph_mg.cpp
    src/structure/transpose_graph_storage_mg.cpp
    src/traversal/bfs_mg.cpp
    src/traversal/extract_bfs_paths_mg.cpp
    src/traversal/k_hop_nbrs_mg.cpp
    src/traversal/sssp_mg.cpp)

set(HIPROCGRAPH_SOURCES_CPP_SPECTRAL
    src/community/legacy/spectral_clustering.cpp)

set(HIPROCGRAPH_FAIL_SOURCES_CPP src/traversal/od_shortest_distances_sg.cpp)

list(APPEND rocgraph_source ${HIPROCGRAPH_SOURCES_C})

# Uncomment this to include spectral_clustering.cpp in the build
list(APPEND rocgraph_source ${HIPROCGRAPH_SOURCES_CPP_SPECTRAL})

list(APPEND rocgraph_source ${HIPROCGRAPH_SOURCES_CPP})

# list(APPEND rocgraph_source ${HIPROCGRAPH_MULTI_SOURCES_CPP})

# list(APPEND rocgraph_source ${HIPROCGRAPH_MULTI_SOURCES_C})
