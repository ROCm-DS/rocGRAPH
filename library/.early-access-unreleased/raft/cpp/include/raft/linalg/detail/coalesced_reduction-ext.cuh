// Copyright (c) 2022-2023, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/core/operators.hpp>

// The explicit instantiation of raft::linalg::detail::coalescedReduction is not
// forced because there would be too many instances. Instead, we cover the most
// common instantiations with extern template instantiations below.

#define instantiate_raft_linalg_detail_coalescedReduction(                                 \
    InType, OutType, IdxType, MainLambda, ReduceLambda, FinalLambda)                       \
    extern template void raft::linalg::detail::coalescedReduction(OutType*      dots,      \
                                                                  const InType* data,      \
                                                                  IdxType       D,         \
                                                                  IdxType       N,         \
                                                                  OutType       init,      \
                                                                  cudaStream_t  stream,    \
                                                                  bool          inplace,   \
                                                                  MainLambda    main_op,   \
                                                                  ReduceLambda  reduce_op, \
                                                                  FinalLambda   final_op)

instantiate_raft_linalg_detail_coalescedReduction(
    double, double, int, raft::identity_op, raft::min_op, raft::identity_op);
instantiate_raft_linalg_detail_coalescedReduction(
    double, double, int, raft::sq_op, raft::add_op, raft::identity_op);
instantiate_raft_linalg_detail_coalescedReduction(
    double, double, int, raft::sq_op, raft::add_op, raft::sqrt_op);
instantiate_raft_linalg_detail_coalescedReduction(
    double, double, int, raft::abs_op, raft::add_op, raft::identity_op);
instantiate_raft_linalg_detail_coalescedReduction(
    double, double, int, raft::abs_op, raft::max_op, raft::identity_op);
instantiate_raft_linalg_detail_coalescedReduction(
    float, float, size_t, raft::abs_op, raft::add_op, raft::sqrt_op);
instantiate_raft_linalg_detail_coalescedReduction(
    float, float, int, raft::abs_op, raft::add_op, raft::identity_op);
instantiate_raft_linalg_detail_coalescedReduction(
    float, float, int, raft::identity_op, raft::add_op, raft::identity_op);
instantiate_raft_linalg_detail_coalescedReduction(
    float, float, int, raft::identity_op, raft::min_op, raft::identity_op);
instantiate_raft_linalg_detail_coalescedReduction(
    float, float, int, raft::sq_op, raft::add_op, raft::identity_op);
instantiate_raft_linalg_detail_coalescedReduction(
    float, float, int, raft::sq_op, raft::add_op, raft::sqrt_op);
instantiate_raft_linalg_detail_coalescedReduction(
    float, float, long, raft::sq_op, raft::add_op, raft::identity_op);
instantiate_raft_linalg_detail_coalescedReduction(
    float, float, size_t, raft::identity_op, raft::add_op, raft::identity_op);
instantiate_raft_linalg_detail_coalescedReduction(
    float, float, size_t, raft::sq_op, raft::add_op, raft::identity_op);
instantiate_raft_linalg_detail_coalescedReduction(
    float, float, size_t, raft::abs_op, raft::max_op, raft::sqrt_op);
instantiate_raft_linalg_detail_coalescedReduction(
    float, float, size_t, raft::sq_op, raft::add_op, raft::sqrt_op);
instantiate_raft_linalg_detail_coalescedReduction(
    float, float, unsigned int, raft::sq_op, raft::add_op, raft::identity_op);

#undef instantiate_raft_linalg_detail_coalescedReduction
