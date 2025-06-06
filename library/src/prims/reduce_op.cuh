// Copyright (C) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "prims/property_op_utils.cuh"

#include "edge_partition_endpoint_property_device_view_device.hpp"
#include "utilities/atomic_ops_device.hpp"
#include "utilities/thrust_tuple_utils.hpp"

#include <raft/core/comms.hpp>

#include <thrust/functional.h>

#include <utility>

namespace rocgraph
{
    namespace reduce_op
    {

        namespace detail
        {

            template <typename T, std::size_t... Is>
            __host__
                __device__ std::enable_if_t<rocgraph::is_thrust_tuple_of_arithmetic<T>::value, T>
                           elementwise_thrust_min(T lhs, T rhs, std::index_sequence<Is...>)
            {
                return thrust::make_tuple((thrust::get<Is>(lhs) < thrust::get<Is>(rhs)
                                               ? thrust::get<Is>(lhs)
                                               : thrust::get<Is>(rhs))...);
            }

            template <typename T, std::size_t... Is>
            __host__
                __device__ std::enable_if_t<rocgraph::is_thrust_tuple_of_arithmetic<T>::value, T>
                           elementwise_thrust_max(T lhs, T rhs, std::index_sequence<Is...>)
            {
                return thrust::make_tuple((thrust::get<Is>(lhs) < thrust::get<Is>(rhs)
                                               ? thrust::get<Is>(rhs)
                                               : thrust::get<Is>(lhs))...);
            }

        } // namespace detail

        // Guidance on writing a custom reduction operator.
        // 1. It is required to add an "using value_type = type_of_the_reduced_values" statement.
        // 2. A custom reduction operator MUST be side-effect free. We use thrust::reduce internally to
        // implement reductions in multiple primitives. The current (version 1.16)  implementation of thrust
        // reduce rounds up the number of invocations based on the CUDA block size and discards the values
        // outside the valid range.
        // 3. If the return value of the reduction operator is solely determined by input argument values,
        // define the pure function static member variable (i.e. "static constexpr pure_function = true;").
        // This may enable better performance in multi-GPU as this flag indicates that the reduction
        // operator can be executed in any GPU (this sometimes enable hierarchical reduction reducing
        // communication volume & peak memory usage).
        // 4. For simple reduction operations with a matching raft::comms::op_t value, specify the
        // compatible_raft_comms_op static member variable (e.g. "static constexpr raft::comms::op_t
        // compatible_raft_comms_op = raft::comms::op_t::MIN"). This often enables direct use of highly
        // optimized the NCCL reduce functions instead of relying on a less efficient gather based reduction
        // mechanism (we may implement a basic tree-based reduction mechanism in the future to improve the
        // efficiency but this is still expected to be slower than the NCCL reduction).
        // 5. Defining the identity_element static member variable (e.g. "inline static T const
        // identity_element = T{}") potentially improves performance as well by avoiding special treatments
        // for tricky corner cases.
        // 6. See the pre-defined reduction operators below as examples.

        // in case there is no payload to reduce
        struct null
        {
            using value_type = void;
        };

        // Binary reduction operator selecting any of the two input arguments, T should be an arithmetic
        // type or a thrust tuple of arithmetic types.
        template <typename T>
        struct any
        {
            using value_type                    = T;
            static constexpr bool pure_function = true; // this can be called in any process

            __host__ __device__ T operator()(T const& lhs, T const& rhs) const
            {
                return lhs;
            }
        };

        template <typename T, typename Enable = void>
        struct minimum;

        // Binary reduction operator selecting the minimum element of the two input arguments (using
        // operator <), a compatible raft comms op exists if T is an arithmetic type.
        template <typename T>
        struct minimum<T, std::enable_if_t<std::is_arithmetic_v<T>>>
        {
            using value_type                    = T;
            static constexpr bool pure_function = true; // this can be called in any process
            static constexpr raft::comms::op_t compatible_raft_comms_op = raft::comms::op_t::MIN;
            inline static T const              identity_element         = max_identity_element<T>();

            __host__ __device__ T operator()(T const& lhs, T const& rhs) const
            {
                return lhs < rhs ? lhs : rhs;
            }
        };

        // Binary reduction operator selecting the minimum element of the two input arguments (using
        // operator <), a compatible raft comms op does not exist when T is a thrust::tuple type.
        template <typename T>
        struct minimum<T, std::enable_if_t<rocgraph::is_thrust_tuple_of_arithmetic<T>::value>>
        {
            using value_type                       = T;
            static constexpr bool pure_function    = true; // this can be called in any process
            inline static T const identity_element = max_identity_element<T>();

            __host__ __device__ T operator()(T const& lhs, T const& rhs) const
            {
                return lhs < rhs ? lhs : rhs;
            }
        };

        // Binary reduction operator selecting the minimum element of the two input arguments elementwise
        // (using operator < for each element), T should be an arithmetic type (this is identical to
        // reduce_op::minimum if T is an arithmetic type) or a thrust tuple of arithmetic types.
        template <typename T>
        struct elementwise_minimum
        {
            static_assert(rocgraph::is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

            using value_type                    = T;
            static constexpr bool pure_function = true; // this can be called in any process
            static constexpr raft::comms::op_t compatible_raft_comms_op = raft::comms::op_t::MIN;
            inline static T const              identity_element         = max_identity_element<T>();

            template <typename U = T>
            __host__ __device__ std::enable_if_t<std::is_arithmetic_v<U>, T>
                                operator()(T const& lhs, T const& rhs) const
            {
                return lhs < rhs ? lhs : rhs;
            }

            template <typename U = T>
            __host__
                __device__ std::enable_if_t<rocgraph::is_thrust_tuple_of_arithmetic<U>::value, T>
                           operator()(T const& lhs, T const& rhs) const
            {
                return detail::elementwise_thrust_min(
                    lhs, rhs, std::make_index_sequence<thrust::tuple_size<T>::value>());
            }
        };

        template <typename T, typename Enable = void>
        struct maximum;

        // Binary reduction operator selecting the maximum element of the two input arguments (using
        // operator <), a compatible raft comms op exists if T is an arithmetic type.
        template <typename T>
        struct maximum<T, std::enable_if_t<std::is_arithmetic_v<T>>>
        {
            using value_type                    = T;
            static constexpr bool pure_function = true; // this can be called in any process
            static constexpr raft::comms::op_t compatible_raft_comms_op = raft::comms::op_t::MAX;
            inline static T const              identity_element         = min_identity_element<T>();

            __host__ __device__ T operator()(T const& lhs, T const& rhs) const
            {
                return lhs < rhs ? rhs : lhs;
            }
        };

        // Binary reduction operator selecting the maximum element of the two input arguments (using
        // operator <), a compatible raft comms op does not exist when T is a thrust::tuple type.
        template <typename T>
        struct maximum<T, std::enable_if_t<rocgraph::is_thrust_tuple_of_arithmetic<T>::value>>
        {
            using value_type                       = T;
            static constexpr bool pure_function    = true; // this can be called in any process
            inline static T const identity_element = min_identity_element<T>();

            __host__ __device__ T operator()(T const& lhs, T const& rhs) const
            {
                return lhs < rhs ? rhs : lhs;
            }
        };

        // Binary reduction operator selecting the maximum element of the two input arguments elementwise
        // (using operator < for each element), T should be an arithmetic type (this is identical to
        // reduce_op::maximum if T is an arithmetic type) or a thrust tuple of arithmetic types.
        template <typename T>
        struct elementwise_maximum
        {
            static_assert(rocgraph::is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

            using value_type                    = T;
            static constexpr bool pure_function = true; // this can be called in any process
            static constexpr raft::comms::op_t compatible_raft_comms_op = raft::comms::op_t::MAX;
            inline static T const              identity_element         = min_identity_element<T>();

            template <typename U = T>
            __host__ __device__ std::enable_if_t<std::is_arithmetic_v<U>, T>
                                operator()(T const& lhs, T const& rhs) const
            {
                return lhs < rhs ? rhs : lhs;
            }

            template <typename U = T>
            __host__
                __device__ std::enable_if_t<rocgraph::is_thrust_tuple_of_arithmetic<U>::value, T>
                           operator()(T const& lhs, T const& rhs) const
            {
                return detail::elementwise_thrust_max(
                    lhs, rhs, std::make_index_sequence<thrust::tuple_size<T>::value>());
            }
        };

        // Binary reduction operator summing the two input arguments, T should be an arithmetic type or a
        // thrust tuple of arithmetic types.
        template <typename T>
        struct plus
        {
            static_assert(rocgraph::is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

            using value_type                    = T;
            static constexpr bool pure_function = true; // this can be called in any process
            static constexpr raft::comms::op_t compatible_raft_comms_op = raft::comms::op_t::SUM;
            inline static T const              identity_element         = T{};
            property_op<T, thrust::plus>       op{};

            __host__ __device__ T operator()(T const& lhs, T const& rhs) const
            {
                return op(lhs, rhs);
            }
        };

        template <typename ReduceOp, typename = raft::comms::op_t>
        struct has_compatible_raft_comms_op : std::false_type
        {
        };

        template <typename ReduceOp>
        struct has_compatible_raft_comms_op<
            ReduceOp,
            std::remove_cv_t<decltype(ReduceOp::compatible_raft_comms_op)>> : std::true_type
        {
        };

        template <typename ReduceOp>
        inline constexpr bool has_compatible_raft_comms_op_v
            = has_compatible_raft_comms_op<ReduceOp>::value;

        template <typename ReduceOp, typename = typename ReduceOp::value_type>
        struct has_identity_element : std::false_type
        {
        };

        template <typename ReduceOp>
        struct has_identity_element<ReduceOp,
                                    std::remove_cv_t<decltype(ReduceOp::identity_element)>>
            : std::true_type
        {
        };

        template <typename ReduceOp>
        inline constexpr bool has_identity_element_v = has_identity_element<ReduceOp>::value;

        template <typename ReduceOp, typename Iterator>
        __device__ std::enable_if_t<has_compatible_raft_comms_op_v<ReduceOp>, void>
                   atomic_reduce(Iterator                                               iter,
                                 typename thrust::iterator_traits<Iterator>::value_type value)
        {
            static_assert(std::is_same_v<typename ReduceOp::value_type,
                                         typename thrust::iterator_traits<Iterator>::value_type>);
            static_assert(
                (ReduceOp::compatible_raft_comms_op == raft::comms::op_t::SUM)
                || (ReduceOp::compatible_raft_comms_op == raft::comms::op_t::MIN)
                || (ReduceOp::compatible_raft_comms_op
                    == raft::comms::op_t::
                        MAX)); // currently, only (element-wise) sum, min, and max are supported.

            if constexpr(ReduceOp::compatible_raft_comms_op == raft::comms::op_t::SUM)
            {
                atomic_add(iter, value);
            }
            else if constexpr(ReduceOp::compatible_raft_comms_op == raft::comms::op_t::MIN)
            {
                elementwise_atomic_min(iter, value);
            }
            else
            {
                elementwise_atomic_max(iter, value);
            }
        }

        template <typename ReduceOp, typename EdgePartitionEndpointPropertyValueWrapper>
        __device__ std::enable_if_t<has_compatible_raft_comms_op_v<ReduceOp>, void> atomic_reduce(
            EdgePartitionEndpointPropertyValueWrapper edge_partition_endpoint_property_value,
            typename EdgePartitionEndpointPropertyValueWrapper::vertex_type offset,
            typename EdgePartitionEndpointPropertyValueWrapper::value_type  value)
        {
            static_assert(
                std::is_same_v<typename ReduceOp::value_type,
                               typename EdgePartitionEndpointPropertyValueWrapper::value_type>);
            static_assert(
                (ReduceOp::compatible_raft_comms_op == raft::comms::op_t::SUM)
                || (ReduceOp::compatible_raft_comms_op == raft::comms::op_t::MIN)
                || (ReduceOp::compatible_raft_comms_op
                    == raft::comms::op_t::
                        MAX)); // currently, only (element-wise) sum, min, and max are supported.

            if constexpr(ReduceOp::compatible_raft_comms_op == raft::comms::op_t::SUM)
            {
                edge_partition_endpoint_property_value.atomic_add(offset, value);
            }
            else if constexpr(ReduceOp::compatible_raft_comms_op == raft::comms::op_t::MIN)
            {
                edge_partition_endpoint_property_value.elementwise_atomic_min(offset, value);
            }
            else
            {
                edge_partition_endpoint_property_value.elementwise_atomic_max(offset, value);
            }
        }

    } // namespace reduce_op
} // namespace rocgraph
