#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA 错误检查宏
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* func, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d: %s failed with error %s\n",
                file, line, func, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// GPU 核函数: C = A * B
// M, N, K 是矩阵的维度
__global__ void matrix_mult_cuda(const float *A, const float *B, float *C, int M, int N, int K) {
    // 计算 C 矩阵中当前线程要处理的行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // A[row][k] * B[k][col] (行主序)
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 初始化矩阵（在主机端）
void initialize_matrix_host(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / (float)RAND_MAX;
    }
}

// 主函数
int main_cuda(int M, int N, int K) {
    // 1. 定义 Host 和 Device 矩阵指针
    float *h_A, *h_B, *h_C; // Host 矩阵
    float *d_A, *d_B, *d_C; // Device 矩阵

    // 矩阵总大小
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // 2. Host 端分配和初始化内存
    h_A = (float *)malloc(size_A);
    h_B = (float *)malloc(size_B);
    h_C = (float *)malloc(size_C);
    
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Host Memory allocation failed.\n");
        return 1;
    }

    srand(time(NULL));
    initialize_matrix_host(h_A, M, K);
    initialize_matrix_host(h_B, K, N);

    // 3. Device 端分配内存
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size_A));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, size_B));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, size_C));

    // 4. 将数据从 Host 复制到 Device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // 5. 配置 Kernel 启动参数
    const int TILE_SIZE = 16; // 线程块大小 (16x16 线程)
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);

    // 6. 使用 CUDA Event 计时
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0)); // 记录开始时间

    // 启动 Kernel
    matrix_mult_cuda<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    
    CHECK_CUDA_ERROR(cudaPeekAtLastError()); // 检查核函数启动错误

    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0)); // 记录结束时间
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop)); // 等待 GPU 完成计算

    float elapsed_time_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_ms, start, stop)); // 计算毫秒 (ms)

    // 7. 将结果从 Device 复制回 Host (用于验证或查看结果)
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    // 8. 打印结果
    printf("CUDA GEMM (Matrix Size: %d x %d x %d):\n", M, N, K);
    printf("Execution Time (Kernel Only): %.3f ms\n", elapsed_time_ms);

    // 9. 释放内存和事件
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    
    return 0;
}
