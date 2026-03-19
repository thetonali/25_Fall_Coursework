#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// 矩阵维度
#define N 1024

// CPU版本的GEMM：C = A * B
void gemm_cpu(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// CUDA Kernel：简单版本
__global__ void gemm_kernel_simple(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// CUDA Kernel：使用共享内存优化
#define TILE_SIZE 16
__global__ void gemm_kernel_shared(float *A, float *B, float *C, int n) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // 分块计算
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // 加载A的子块到共享内存
        if (row < n && (tile * TILE_SIZE + tx) < n)
            As[ty][tx] = A[row * n + tile * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
        
        // 加载B的子块到共享内存
        if (col < n && (tile * TILE_SIZE + ty) < n)
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * n + col];
        else
            Bs[ty][tx] = 0.0f;
        
        __syncthreads();
        
        // 计算部分和
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// 初始化矩阵
void init_matrix(float *mat, int n) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = (float)(rand() % 100) / 10.0f;
    }
}

// 验证结果
int verify_result(float *C_cpu, float *C_gpu, int n) {
    float epsilon = 1e-3;
    for (int i = 0; i < n * n; i++) {
        if (fabs(C_cpu[i] - C_gpu[i]) > epsilon) {
            printf("验证失败在索引 %d: CPU=%f, GPU=%f\n", i, C_cpu[i], C_gpu[i]);
            return 0;
        }
    }
    return 1;
}

int main() {
    int n = N;
    size_t bytes = n * n * sizeof(float);
    
    // 分配主机内存
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);
    
    // 初始化矩阵
    srand(time(NULL));
    init_matrix(h_A, n);
    init_matrix(h_B, n);
    
    printf("矩阵维度: %d x %d\n\n", n, n);
    
    // ========== CPU版本 ==========
    printf("运行 CPU 版本...\n");
    clock_t start_cpu = clock();
    gemm_cpu(h_A, h_B, h_C_cpu, n);
    clock_t end_cpu = clock();
    double time_cpu = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    printf("CPU 执行时间: %.4f 秒\n\n", time_cpu);
    
    // ========== CUDA版本（简单） ==========
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    printf("运行 CUDA 版本（简单）...\n");
    cudaEvent_t start_simple, stop_simple;
    cudaEventCreate(&start_simple);
    cudaEventCreate(&stop_simple);
    
    cudaEventRecord(start_simple);
    gemm_kernel_simple<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop_simple);
    cudaEventSynchronize(stop_simple);
    
    float time_cuda_simple = 0;
    cudaEventElapsedTime(&time_cuda_simple, start_simple, stop_simple);
    printf("CUDA（简单）执行时间: %.4f 秒\n", time_cuda_simple / 1000.0f);
    
    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
    if (verify_result(h_C_cpu, h_C_gpu, n)) {
        printf("✓ 结果验证通过\n\n");
    }
    
    // ========== CUDA版本（共享内存优化） ==========
    printf("运行 CUDA 版本（共享内存优化）...\n");
    cudaEvent_t start_shared, stop_shared;
    cudaEventCreate(&start_shared);
    cudaEventCreate(&stop_shared);
    
    cudaEventRecord(start_shared);
    gemm_kernel_shared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop_shared);
    cudaEventSynchronize(stop_shared);
    
    float time_cuda_shared = 0;
    cudaEventElapsedTime(&time_cuda_shared, start_shared, stop_shared);
    printf("CUDA（共享内存）执行时间: %.4f 秒\n", time_cuda_shared / 1000.0f);
    
    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
    if (verify_result(h_C_cpu, h_C_gpu, n)) {
        printf("✓ 结果验证通过\n\n");
    }
    
    // ========== 性能对比 ==========
    printf("========== 性能对比 ==========\n");
    printf("CPU 时间:              %.4f 秒\n", time_cpu);
    printf("CUDA（简单）时间:      %.4f 秒 (加速 %.2fx)\n", 
           time_cuda_simple / 1000.0f, time_cpu / (time_cuda_simple / 1000.0f));
    printf("CUDA（共享内存）时间:  %.4f 秒 (加速 %.2fx)\n", 
           time_cuda_shared / 1000.0f, time_cpu / (time_cuda_shared / 1000.0f));
    printf("共享内存优化提升:      %.2fx\n", time_cuda_simple / time_cuda_shared);
    
    // 清理
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start_simple);
    cudaEventDestroy(stop_simple);
    cudaEventDestroy(start_shared);
    cudaEventDestroy(stop_shared);
    
    return 0;
}