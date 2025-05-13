// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Copyright (C) 2020-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "rocgraph/rocgraph-export.h"

namespace rocgraph
{
    namespace legacy
    {
        extern template class ROCGRAPH_EXPORT GraphViewBase<int32_t, int32_t, float>;
        extern template class ROCGRAPH_EXPORT GraphViewBase<int32_t, int32_t, double>;
        extern template class ROCGRAPH_EXPORT GraphViewBase<int32_t, int64_t, float>;
        extern template class ROCGRAPH_EXPORT GraphViewBase<int32_t, int64_t, double>;
        extern template class ROCGRAPH_EXPORT GraphViewBase<int64_t, int32_t, float>;
        extern template class ROCGRAPH_EXPORT GraphViewBase<int64_t, int32_t, double>;
        extern template class ROCGRAPH_EXPORT GraphViewBase<int64_t, int64_t, float>;
        extern template class ROCGRAPH_EXPORT GraphViewBase<int64_t, int64_t, double>;
        extern template class ROCGRAPH_EXPORT
            GraphCompressedSparseBaseView<int32_t, int32_t, float>;
        extern template class ROCGRAPH_EXPORT
            GraphCompressedSparseBaseView<int32_t, int32_t, double>;
        extern template class ROCGRAPH_EXPORT
            GraphCompressedSparseBaseView<int32_t, int64_t, float>;
        extern template class ROCGRAPH_EXPORT
            GraphCompressedSparseBaseView<int32_t, int64_t, double>;
        extern template class ROCGRAPH_EXPORT
            GraphCompressedSparseBaseView<int64_t, int32_t, float>;
        extern template class ROCGRAPH_EXPORT
            GraphCompressedSparseBaseView<int64_t, int32_t, double>;
        extern template class ROCGRAPH_EXPORT
            GraphCompressedSparseBaseView<int64_t, int64_t, float>;
        extern template class ROCGRAPH_EXPORT
            GraphCompressedSparseBaseView<int64_t, int64_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCompressedSparseBase<int32_t, int32_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCompressedSparseBase<int32_t, int32_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCompressedSparseBase<int32_t, int64_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCompressedSparseBase<int32_t, int64_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCompressedSparseBase<int64_t, int32_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCompressedSparseBase<int64_t, int32_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCompressedSparseBase<int64_t, int64_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCompressedSparseBase<int64_t, int64_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCOOView<int32_t, int32_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCOOView<int32_t, int32_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCOOView<int32_t, int64_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCOOView<int32_t, int64_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCOOView<int64_t, int32_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCOOView<int64_t, int32_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCOOView<int64_t, int64_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCOOView<int64_t, int64_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCSRView<int32_t, int32_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCSRView<int32_t, int32_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCSRView<int32_t, int64_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCSRView<int32_t, int64_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCSRView<int64_t, int32_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCSRView<int64_t, int32_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCSRView<int64_t, int64_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCSRView<int64_t, int64_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCOO<int32_t, int32_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCOO<int32_t, int32_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCOO<int32_t, int64_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCOO<int32_t, int64_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCOO<int64_t, int32_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCOO<int64_t, int32_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCOO<int64_t, int64_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCOO<int64_t, int64_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCSR<int32_t, int32_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCSR<int32_t, int32_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCSR<int32_t, int64_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCSR<int32_t, int64_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCSR<int64_t, int32_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCSR<int64_t, int32_t, double>;
        extern template class ROCGRAPH_EXPORT GraphCSR<int64_t, int64_t, float>;
        extern template class ROCGRAPH_EXPORT GraphCSR<int64_t, int64_t, double>;
    } // namespace legacy
} // namespace rocgraph
