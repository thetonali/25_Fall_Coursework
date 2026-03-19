#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 矩阵乘法 C = A * B
// A 是 M x K 矩阵, B 是 K x N 矩阵, C 是 M x N 矩阵
void matrix_mult_cpu(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                // A[i][k] * B[k][j]
                // 假设矩阵是行主序 (Row-major) 存储
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 初始化矩阵
void initialize_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / (float)RAND_MAX; // 随机浮点数
    }
}

// 主函数
int main_cpu(int M, int N, int K) {
    // 1. 分配内存
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C = (float *)malloc(M * N * sizeof(float));

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    // 2. 初始化数据
    srand(time(NULL));
    initialize_matrix(A, M, K);
    initialize_matrix(B, K, N);

    // 3. 计时并执行 CPU GEMM
    clock_t start = clock();
    
    matrix_mult_cpu(A, B, C, M, N, K);
    
    clock_t end = clock();

    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0; // 转换为毫秒 (ms)

    // 4. 打印结果
    printf("CPU GEMM (Matrix Size: %d x %d x %d):\n", M, N, K);
    printf("Execution Time: %.3f ms\n", cpu_time_used);

    // 5. 释放内存
    free(A);
    free(B);
    free(C);

    return 0;
}


// 添加这个标准的main函数
int main() {
    int M = 1024;  // 或从命令行参数读取
    int N = 1024;
    int K = 1024;
    return main_cpu(M, N, K);
}