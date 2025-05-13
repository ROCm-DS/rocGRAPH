// Copyright (C) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "c_api/rocgraph_error.hpp"

#include "utilities/graph_traits.hpp"

#include <memory>

namespace rocgraph
{
    namespace c_api
    {

        struct abstract_functor
        {
            // Move to abstract functor... make operator a void, add rocgraph_graph_t * result to functor
            // try that with instantiation questions
            std::unique_ptr<rocgraph_error_t> error_ = {std::make_unique<rocgraph_error_t>("")};
            rocgraph_status                   status_{rocgraph_status_success};

            void unsupported()
            {
                mark_error(rocgraph_status_unsupported_type_combination,
                           "Type Dispatcher executing unsupported combination of types");
            }

            void mark_error(rocgraph_status status, std::string const& error_message)
            {
                status_                = status;
                error_->error_message_ = error_message;
            }
        };

    } // namespace c_api
} // namespace rocgraph
