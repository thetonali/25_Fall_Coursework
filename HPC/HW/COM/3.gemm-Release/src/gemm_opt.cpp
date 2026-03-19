#include <cstdlib>
#include <iostream>
#include <cstring>
#include <omp.h>
#include "gemm_opt.h"
#include <immintrin.h>

// 针对小块的内核函数，使用AVX2向量化和寄存器分块
inline void gemm_kernel_8x8(const float* A, const float* B, float* C, 
                             int M, int N, int K, 
                             int i, int j, int k_start, int k_end) {
    // 8x8的寄存器分块
    __m256 c00 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    
    for (int k = k_start; k < k_end; k++) {
        // 加载A的4个元素并广播
        __m256 a0 = _mm256_set1_ps(A[i * K + k]);
        __m256 a1 = _mm256_set1_ps(A[(i + 1) * K + k]);
        __m256 a2 = _mm256_set1_ps(A[(i + 2) * K + k]);
        __m256 a3 = _mm256_set1_ps(A[(i + 3) * K + k]);
        
        // 加载B的两个向量(8个元素)
        __m256 b0 = _mm256_loadu_ps(&B[k * N + j]);
        __m256 b1 = _mm256_loadu_ps(&B[k * N + j + 8]);
        
        // FMA操作
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
    }
    
    // 存储结果
    _mm256_storeu_ps(&C[i * N + j], _mm256_add_ps(_mm256_loadu_ps(&C[i * N + j]), c00));
    _mm256_storeu_ps(&C[i * N + j + 8], _mm256_add_ps(_mm256_loadu_ps(&C[i * N + j + 8]), c01));
    _mm256_storeu_ps(&C[(i + 1) * N + j], _mm256_add_ps(_mm256_loadu_ps(&C[(i + 1) * N + j]), c10));
    _mm256_storeu_ps(&C[(i + 1) * N + j + 8], _mm256_add_ps(_mm256_loadu_ps(&C[(i + 1) * N + j + 8]), c11));
    _mm256_storeu_ps(&C[(i + 2) * N + j], _mm256_add_ps(_mm256_loadu_ps(&C[(i + 2) * N + j]), c20));
    _mm256_storeu_ps(&C[(i + 2) * N + j + 8], _mm256_add_ps(_mm256_loadu_ps(&C[(i + 2) * N + j + 8]), c21));
    _mm256_storeu_ps(&C[(i + 3) * N + j], _mm256_add_ps(_mm256_loadu_ps(&C[(i + 3) * N + j]), c30));
    _mm256_storeu_ps(&C[(i + 3) * N + j + 8], _mm256_add_ps(_mm256_loadu_ps(&C[(i + 3) * N + j + 8]), c31));
}

// 针对边界情况的标量处理
inline void gemm_scalar(const float* A, const float* B, float* C, 
                        int M, int N, int K,
                        int i_start, int i_end, 
                        int j_start, int j_end,
                        int k_start, int k_end) {
    for (int i = i_start; i < i_end; i++) {
        for (int k = k_start; k < k_end; k++) {
            float a_val = A[i * K + k];
            for (int j = j_start; j < j_end; j++) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
}

void gemm_opt(const float* A, const float* B, float* C, int M, int N, int K) {
    // 分块大小 - 针对L1/L2缓存优化
    const int MC = 256;  // M方向的分块大小
    const int NC = 4096; // N方向的分块大小
    const int KC = 512;  // K方向的分块大小
    
    // OpenMP并行化外层循环
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int ii = 0; ii < M; ii += MC) {
        for (int jj = 0; jj < N; jj += NC) {
            for (int kk = 0; kk < K; kk += KC) {
                int i_end = (ii + MC < M) ? ii + MC : M;
                int j_end = (jj + NC < N) ? jj + NC : N;
                int k_end = (kk + KC < K) ? kk + KC : K;
                
                // 微块处理 - 4x16 的块
                int i = ii;
                for (; i + 3 < i_end; i += 4) {
                    int j = jj;
                    for (; j + 15 < j_end; j += 16) {
                        gemm_kernel_8x8(A, B, C, M, N, K, i, j, kk, k_end);
                    }
                    
                    // 处理j方向剩余
                    if (j < j_end) {
                        gemm_scalar(A, B, C, M, N, K, i, i + 4, j, j_end, kk, k_end);
                    }
                }
                
                // 处理i方向剩余
                if (i < i_end) {
                    gemm_scalar(A, B, C, M, N, K, i, i_end, jj, j_end, kk, k_end);
                }
            }
        }
    }
}