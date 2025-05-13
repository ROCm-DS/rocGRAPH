// Copyright (C) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <rocgraph/rocgraph-export.h>

namespace rocgraph
{
    namespace legacy
    {
        template class ROCGRAPH_EXPORT GraphViewBase<int32_t, int32_t, float>;
        template class ROCGRAPH_EXPORT GraphViewBase<int32_t, int32_t, double>;
        template class ROCGRAPH_EXPORT GraphViewBase<int32_t, int64_t, float>;
        template class ROCGRAPH_EXPORT GraphViewBase<int32_t, int64_t, double>;
        template class ROCGRAPH_EXPORT GraphViewBase<int64_t, int64_t, float>;
        template class ROCGRAPH_EXPORT GraphViewBase<int64_t, int64_t, double>;
        template class ROCGRAPH_EXPORT GraphCompressedSparseBaseView<int32_t, int32_t, float>;
        template class ROCGRAPH_EXPORT GraphCompressedSparseBaseView<int32_t, int32_t, double>;
        template class ROCGRAPH_EXPORT GraphCompressedSparseBaseView<int32_t, int64_t, float>;
        template class ROCGRAPH_EXPORT GraphCompressedSparseBaseView<int32_t, int64_t, double>;
        template class ROCGRAPH_EXPORT GraphCompressedSparseBaseView<int64_t, int64_t, float>;
        template class ROCGRAPH_EXPORT GraphCompressedSparseBaseView<int64_t, int64_t, double>;
        template class ROCGRAPH_EXPORT GraphCompressedSparseBase<int32_t, int32_t, float>;
        template class ROCGRAPH_EXPORT GraphCompressedSparseBase<int32_t, int32_t, double>;
        template class ROCGRAPH_EXPORT GraphCompressedSparseBase<int32_t, int64_t, float>;
        template class ROCGRAPH_EXPORT GraphCompressedSparseBase<int32_t, int64_t, double>;
        template class ROCGRAPH_EXPORT GraphCompressedSparseBase<int64_t, int64_t, float>;
        template class ROCGRAPH_EXPORT GraphCompressedSparseBase<int64_t, int64_t, double>;
        template class ROCGRAPH_EXPORT GraphCOOView<int32_t, int32_t, float>;
        template class ROCGRAPH_EXPORT GraphCOOView<int32_t, int32_t, double>;
        template class ROCGRAPH_EXPORT GraphCOOView<int32_t, int64_t, float>;
        template class ROCGRAPH_EXPORT GraphCOOView<int32_t, int64_t, double>;
        template class ROCGRAPH_EXPORT GraphCOOView<int64_t, int64_t, float>;
        template class ROCGRAPH_EXPORT GraphCOOView<int64_t, int64_t, double>;
        template class ROCGRAPH_EXPORT GraphCSRView<int32_t, int32_t, float>;
        template class ROCGRAPH_EXPORT GraphCSRView<int32_t, int32_t, double>;
        template class ROCGRAPH_EXPORT GraphCSRView<int32_t, int64_t, float>;
        template class ROCGRAPH_EXPORT GraphCSRView<int32_t, int64_t, double>;
        template class ROCGRAPH_EXPORT GraphCSRView<int64_t, int64_t, float>;
        template class ROCGRAPH_EXPORT GraphCSRView<int64_t, int64_t, double>;
        template class ROCGRAPH_EXPORT GraphCOO<int32_t, int32_t, float>;
        template class ROCGRAPH_EXPORT GraphCOO<int32_t, int32_t, double>;
        template class ROCGRAPH_EXPORT GraphCOO<int32_t, int64_t, float>;
        template class ROCGRAPH_EXPORT GraphCOO<int32_t, int64_t, double>;
        template class ROCGRAPH_EXPORT GraphCOO<int64_t, int64_t, float>;
        template class ROCGRAPH_EXPORT GraphCOO<int64_t, int64_t, double>;
        template class ROCGRAPH_EXPORT GraphCSR<int32_t, int32_t, float>;
        template class ROCGRAPH_EXPORT GraphCSR<int32_t, int32_t, double>;
        template class ROCGRAPH_EXPORT GraphCSR<int32_t, int64_t, float>;
        template class ROCGRAPH_EXPORT GraphCSR<int32_t, int64_t, double>;
        template class ROCGRAPH_EXPORT GraphCSR<int64_t, int64_t, float>;
        template class ROCGRAPH_EXPORT GraphCSR<int64_t, int64_t, double>;
    } // namespace legacy
} // namespace rocgraph
