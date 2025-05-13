// Copyright (c) 2020-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once
#include <raft/core/error.hpp>
#include <raft/cusolver.h>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/detail/cusolver_wrappers.hpp>

#include <cblas.h>

namespace raft
{

#define lapackCheckError(status)                                                           \
    {                                                                                      \
        if(status < 0)                                                                     \
        {                                                                                  \
            std::stringstream ss;                                                          \
            ss << "Lapack error: argument number " << -status << " had an illegal value."; \
            throw exception(ss.str());                                                     \
        }                                                                                  \
        else if(status > 0)                                                                \
            RAFT_FAIL("Lapack error: internal error.");                                    \
    }

    extern "C" void
        sgeqrf_(int* m, int* n, float* a, int* lda, float* tau, float* work, int* lwork, int* info);
    extern "C" void dgeqrf_(
        int* m, int* n, double* a, int* lda, double* tau, double* work, int* lwork, int* info);

    extern "C" void sormqr_(char*        side,
                            char*        trans,
                            int*         m,
                            int*         n,
                            int*         k,
                            float*       a,
                            int*         lda,
                            const float* tau,
                            float*       c,
                            int*         ldc,
                            float*       work,
                            int*         lwork,
                            int*         info);
    extern "C" void dormqr_(char*         side,
                            char*         trans,
                            int*          m,
                            int*          n,
                            int*          k,
                            double*       a,
                            int*          lda,
                            const double* tau,
                            double*       c,
                            int*          ldc,
                            double*       work,
                            int*          lwork,
                            int*          info);
    extern "C" int  dgeev_(char*   jobvl,
                           char*   jobvr,
                           int*    n,
                           double* a,
                           int*    lda,
                           double* wr,
                           double* wi,
                           double* vl,
                           int*    ldvl,
                           double* vr,
                           int*    ldvr,
                           double* work,
                           int*    lwork,
                           int*    info);

    extern "C" int sgeev_(char*  jobvl,
                          char*  jobvr,
                          int*   n,
                          float* a,
                          int*   lda,
                          float* wr,
                          float* wi,
                          float* vl,
                          int*   ldvl,
                          float* vr,
                          int*   ldvr,
                          float* work,
                          int*   lwork,
                          int*   info);

    // TODO(HIP/AMD): Replace function call with hipsolver call when available: Please see internal
    // issue 22
    extern "C" {
    void ssterf_(int* n, float* d, float* e, int* info);
    }

    // TODO(HIP/AMD): Replace function call with hipsolver call when available: Please see internal
    // issue 22
    extern "C" {
    void dsterf_(int* n, double* d, double* e, int* info);
    }

    // TODO(HIP/AMD): Replace function call with hipsolver call when available: Please see internal
    // issue 22
    extern "C" {
    void ssteqr_(
        const char* compz, int* n, float* d, float* e, float* z, int* ldz, float* work, int* info);
    }

    // TODO(HIP/AMD): Replace function call with hipsolver call when available: Please see internal
    // issue 22
    extern "C" {
    void dsteqr_(const char* compz,
                 int*        n,
                 double*     d,
                 double*     e,
                 double*     z,
                 int*        ldz,
                 double*     work,
                 int*        info);
    }

    template <typename T>
    class Lapack
    {
    private:
        Lapack();
        ~Lapack();

    public:
        static void check_lapack_enabled();

        static void gemm(bool     transa,
                         bool     transb,
                         int      m,
                         int      n,
                         int      k,
                         T        alpha,
                         const T* A,
                         int      lda,
                         const T* B,
                         int      ldb,
                         T        beta,
                         T*       C,
                         int      ldc);

        // special QR for lanczos
        static void sterf(int n, T* d, T* e);
        static void steqr(char compz, int n, T* d, T* e, T* z, int ldz, T* work);

        // QR
        // computes the QR factorization of a general matrix
        static void geqrf(int m, int n, T* a, int lda, T* tau, T* work, int* lwork);
        // Generates the real orthogonal matrix Q of the QR factorization formed by geqrf.

        // multiply C by implicit Q
        static void ormqr(bool right_side,
                          bool transq,
                          int  m,
                          int  n,
                          int  k,
                          T*   a,
                          int  lda,
                          T*   tau,
                          T*   c,
                          int  ldc,
                          T*   work,
                          int* lwork);

        static void geev(T* A, T* eigenvalues, int dim, int lda);
        static void geev(T* A, T* eigenvalues, T* eigenvectors, int dim, int lda, int ldvr);
        static void geev(T*  A,
                         T*  eigenvalues_r,
                         T*  eigenvalues_i,
                         T*  eigenvectors_r,
                         T*  eigenvectors_i,
                         int dim,
                         int lda,
                         int ldvr);

    private:
        static void lapack_gemm(const char   transa,
                                const char   transb,
                                int          m,
                                int          n,
                                int          k,
                                float        alpha,
                                const float* a,
                                int          lda,
                                const float* b,
                                int          ldb,
                                float        beta,
                                float*       c,
                                int          ldc)
        {
            CBLAS_TRANSPOSE cblas_transa = (transa == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans;
            CBLAS_TRANSPOSE cblas_transb = (transb == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans;

            // TODO(HIP/AMD): Replace function call with hipsolver call when available: Please see internal
            // issue 22
            cblas_sgemm(CblasColMajor,
                        cblas_transa,
                        cblas_transb,
                        m,
                        n,
                        k,
                        alpha,
                        (float*)a,
                        lda,
                        (float*)b,
                        ldb,
                        beta,
                        c,
                        ldc);
        }

        static void lapack_gemm(const signed char transa,
                                const signed char transb,
                                int               m,
                                int               n,
                                int               k,
                                double            alpha,
                                const double*     a,
                                int               lda,
                                const double*     b,
                                int               ldb,
                                double            beta,
                                double*           c,
                                int               ldc)
        {
            CBLAS_TRANSPOSE cblas_transa = (transa == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans;
            CBLAS_TRANSPOSE cblas_transb = (transb == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans;

            // TODO(HIP/AMD): Replace function call with hipsolver call when available: Please see internal
            // issue 22
            cblas_dgemm(CblasColMajor,
                        cblas_transa,
                        cblas_transa,
                        m,
                        n,
                        k,
                        alpha,
                        (double*)a,
                        lda,
                        (double*)b,
                        ldb,
                        beta,
                        c,
                        ldc);
        }

        static void lapack_sterf(int n, float* d, float* e, int* info)
        {
            ssterf_(&n, d, e, info);
        }

        static void lapack_sterf(int n, double* d, double* e, int* info)
        {
            dsterf_(&n, d, e, info);
        }

        static void lapack_steqr(
            const char compz, int n, float* d, float* e, float* z, int ldz, float* work, int* info)
        {
            ssteqr_(&compz, &n, d, e, z, &ldz, work, info);
        }

        static void lapack_steqr(const char compz,
                                 int        n,
                                 double*    d,
                                 double*    e,
                                 double*    z,
                                 int        ldz,
                                 double*    work,
                                 int*       info)
        {
            dsteqr_(&compz, &n, d, e, z, &ldz, work, info);
        }

        static void lapack_geqrf(
            int m, int n, float* a, int lda, float* tau, float* work, int* lwork, int* info)
        {
            sgeqrf_(&m, &n, a, &lda, tau, work, lwork, info);
        }

        static void lapack_geqrf(
            int m, int n, double* a, int lda, double* tau, double* work, int* lwork, int* info)
        {
            dgeqrf_(&m, &n, a, &lda, tau, work, lwork, info);
        }

        static void lapack_ormqr(char   side,
                                 char   trans,
                                 int    m,
                                 int    n,
                                 int    k,
                                 float* a,
                                 int    lda,
                                 float* tau,
                                 float* c,
                                 int    ldc,
                                 float* work,
                                 int*   lwork,
                                 int*   info)
        {
            sormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, lwork, info);
        }

        static void lapack_ormqr(char    side,
                                 char    trans,
                                 int     m,
                                 int     n,
                                 int     k,
                                 double* a,
                                 int     lda,
                                 double* tau,
                                 double* c,
                                 int     ldc,
                                 double* work,
                                 int*    lwork,
                                 int*    info)
        {
            dormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, lwork, info);
        }

        static int lapack_geev_dispatch(char*   jobvl,
                                        char*   jobvr,
                                        int*    n,
                                        double* a,
                                        int*    lda,
                                        double* wr,
                                        double* wi,
                                        double* vl,
                                        int*    ldvl,
                                        double* vr,
                                        int*    ldvr,
                                        double* work,
                                        int*    lwork,
                                        int*    info)
        {
            return dgeev_(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
        }

        static int lapack_geev_dispatch(char*  jobvl,
                                        char*  jobvr,
                                        int*   n,
                                        float* a,
                                        int*   lda,
                                        float* wr,
                                        float* wi,
                                        float* vl,
                                        int*   ldvl,
                                        float* vr,
                                        int*   ldvr,
                                        float* work,
                                        int*   lwork,
                                        int*   info)
        {
            return sgeev_(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
        }

        // real eigenvalues
        static void lapack_geev(T* A, T* eigenvalues, int dim, int lda)
        {
            char           job = 'N';
            std::vector<T> WI(dim);
            int            ldv       = 1;
            T*             vl        = 0;
            int            work_size = 6 * dim;
            std::vector<T> work(work_size);
            int            info;
            lapack_geev_dispatch(&job,
                                 &job,
                                 &dim,
                                 A,
                                 &lda,
                                 eigenvalues,
                                 WI.data(),
                                 vl,
                                 &ldv,
                                 vl,
                                 &ldv,
                                 work.data(),
                                 &work_size,
                                 &info);
            lapackCheckError(info);
        }

        // real eigenpairs
        static void lapack_geev(T* A, T* eigenvalues, T* eigenvectors, int dim, int lda, int ldvr)
        {
            char           jobvl = 'N';
            char           jobvr = 'V';
            std::vector<T> WI(dim);
            int            work_size = 6 * dim;
            T*             vl        = 0;
            int            ldvl      = 1;
            std::vector<T> work(work_size);
            int            info;
            lapack_geev_dispatch(&jobvl,
                                 &jobvr,
                                 &dim,
                                 A,
                                 &lda,
                                 eigenvalues,
                                 WI.data(),
                                 vl,
                                 &ldvl,
                                 eigenvectors,
                                 &ldvr,
                                 work.data(),
                                 &work_size,
                                 &info);
            lapackCheckError(info);
        }

        // complex eigenpairs
        static void lapack_geev(T*  A,
                                T*  eigenvalues_r,
                                T*  eigenvalues_i,
                                T*  eigenvectors_r,
                                T*  eigenvectors_i,
                                int dim,
                                int lda,
                                int ldvr)
        {
            char           jobvl     = 'N';
            char           jobvr     = 'V';
            int            work_size = 8 * dim;
            int            ldvl      = 1;
            std::vector<T> work(work_size);
            int            info;
            lapack_geev_dispatch(&jobvl,
                                 &jobvr,
                                 &dim,
                                 A,
                                 &lda,
                                 eigenvalues_r,
                                 eigenvalues_i,
                                 0,
                                 &ldvl,
                                 eigenvectors_r,
                                 &ldvr,
                                 work.data(),
                                 &work_size,
                                 &info);
            lapackCheckError(info);
        }
    };

    template <typename T>
    void Lapack<T>::check_lapack_enabled()
    {
#ifndef USE_LAPACK
        RAFT_FAIL("Error: LAPACK not enabled.");
#endif
    }

    template <typename T>
    void Lapack<T>::gemm(bool     transa,
                         bool     transb,
                         int      m,
                         int      n,
                         int      k,
                         T        alpha,
                         const T* A,
                         int      lda,
                         const T* B,
                         int      ldb,
                         T        beta,
                         T*       C,
                         int      ldc)
    {
        // check_lapack_enabled();
        // #ifdef NVGRAPH_USE_LAPACK
        const char transA_char = transa ? 'T' : 'N';
        const char transB_char = transb ? 'T' : 'N';
        lapack_gemm(transA_char, transB_char, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        // #endif
    }

    template <typename T>
    void Lapack<T>::sterf(int n, T* d, T* e)
    {
        //    check_lapack_enabled();
        // #ifdef NVGRAPH_USE_LAPACK
        int info;
        lapack_sterf(n, d, e, &info);
        lapackCheckError(info);
        // #endif
    }

    template <typename T>
    void Lapack<T>::steqr(char compz, int n, T* d, T* e, T* z, int ldz, T* work)
    {
        //    check_lapack_enabled();
        // #ifdef NVGRAPH_USE_LAPACK
        int info;
        lapack_steqr(compz, n, d, e, z, ldz, work, &info);
        lapackCheckError(info);
        // #endif
    }

    template <typename T>
    void Lapack<T>::geqrf(int m, int n, T* a, int lda, T* tau, T* work, int* lwork)
    {
        check_lapack_enabled();
#ifdef USE_LAPACK
        int info;
        lapack_geqrf(m, n, a, lda, tau, work, lwork, &info);
        lapackCheckError(info);
#endif
    }
    template <typename T>
    void Lapack<T>::ormqr(bool right_side,
                          bool transq,
                          int  m,
                          int  n,
                          int  k,
                          T*   a,
                          int  lda,
                          T*   tau,
                          T*   c,
                          int  ldc,
                          T*   work,
                          int* lwork)
    {
        check_lapack_enabled();
#ifdef USE_LAPACK
        char side  = right_side ? 'R' : 'L';
        char trans = transq ? 'T' : 'N';
        int  info;
        lapack_ormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, &info);
        lapackCheckError(info);
#endif
    }

    // real eigenvalues
    template <typename T>
    void Lapack<T>::geev(T* A, T* eigenvalues, int dim, int lda)
    {
        check_lapack_enabled();
#ifdef USE_LAPACK
        lapack_geev(A, eigenvalues, dim, lda);
#endif
    }
    // real eigenpairs
    template <typename T>
    void Lapack<T>::geev(T* A, T* eigenvalues, T* eigenvectors, int dim, int lda, int ldvr)
    {
        check_lapack_enabled();
#ifdef USE_LAPACK
        lapack_geev(A, eigenvalues, eigenvectors, dim, lda, ldvr);
#endif
    }
    // complex eigenpairs
    template <typename T>
    void Lapack<T>::geev(T*  A,
                         T*  eigenvalues_r,
                         T*  eigenvalues_i,
                         T*  eigenvectors_r,
                         T*  eigenvectors_i,
                         int dim,
                         int lda,
                         int ldvr)
    {
        check_lapack_enabled();
#ifdef USE_LAPACK
        lapack_geev(
            A, eigenvalues_r, eigenvalues_i, eigenvectors_r, eigenvectors_i, dim, lda, ldvr);
#endif
    }

} // namespace raft
