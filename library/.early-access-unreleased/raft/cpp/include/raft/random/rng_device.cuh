// Copyright (c) 2022, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#ifndef __RNG_DEVICE_H
#define __RNG_DEVICE_H

#pragma once

#include "detail/rng_device.cuh"
#include "rng_state.hpp"

namespace raft
{
    namespace random
    {

        using detail::DeviceState;

        using detail::PCGenerator;
        using detail::PhiloxGenerator;

        using detail::BernoulliDistParams;
        using detail::ExponentialDistParams;
        using detail::GumbelDistParams;
        using detail::InvariantDistParams;
        using detail::LaplaceDistParams;
        using detail::LogisticDistParams;
        using detail::LogNormalDistParams;
        using detail::NormalDistParams;
        using detail::NormalIntDistParams;
        using detail::NormalTableDistParams;
        using detail::RayleighDistParams;
        using detail::SamplingParams;
        using detail::ScaledBernoulliDistParams;
        using detail::UniformDistParams;
        using detail::UniformIntDistParams;

        // Not strictly needed due to C++ ADL rules
        using detail::custom_next;
        // this is necessary again since all arguments are primitive types
        using detail::box_muller_transform;

    }; // end namespace random
}; // end namespace raft

#endif
