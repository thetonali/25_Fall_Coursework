// reduce.cu
#include "reduce.h"
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// 用于检查 CUDA API 调用错误的辅助宏 (保留)
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while (0)

// 1. Warp级别的归约（使用shuffle指令）
__inline__ __device__ float warpReduce(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 2. 统一的归约 Kernel (使用 Grid-Stride Loop 加载数据)
// 适用于第一阶段（处理大规模输入）和第二阶段（处理中间结果）
__global__ void generalReduceKernel(const float* __restrict__ input, float* output, int size) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    // 计算全局索引，使用 gridDim.x * blockDim.x 作为步长
    int globalIdx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x; 
    
    float sum = 0.0f;
    
    // Grid-Stride Loop: 确保所有元素都被累加
    // 这种方式非常健壮，能处理任何 size 和 block/grid 配置
    for (int i = globalIdx; i < size; i += stride) {
        sum += input[i];
    }
    
    // 写入共享内存
    sdata[tid] = sum;
    __syncthreads();
    
    // Block 内归约（从 BLOCK_SIZE / 2 阶段到 WARP_SIZE 阶段）
    for (int s = blockDim.x / 2; s > WARP_SIZE; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 最后一个 warp 使用 Shuffle 指令
    if (tid < WARP_SIZE) {
        float val = sdata[tid];
        val = warpReduce(val);
        // 只有线程 0 将最终结果写入全局内存
        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// 3. GPU Reduce包装函数
float gpuReduce(const float* d_data, const int size) {
    if (size <= 0) return 0.0f;
    
    // 策略：使用最大化 GPU 占用的 Blocks 数量。
    // A100 GPU 通常有 108 个 SM，我们设置一个较大的 Grid size，例如 4 * 108 = 432 Blocks
    // 这样可以确保 Grid-Stride Loop 被充分利用。
    int maxSM = 108; // 接近 A100 的 SM 数量
    int blocksFactor = 4; // 每个 SM 启动 4 个 Block，以隐藏延迟
    int numBlocks1 = maxSM * blocksFactor; 
    
    // 如果数据量太小，则减少 Block 数量
    if (size / BLOCK_SIZE < numBlocks1) {
        numBlocks1 = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }
    
    // 分配中间结果内存
    float* d_intermediate;
    CUDA_CHECK(cudaMalloc(&d_intermediate, numBlocks1 * sizeof(float)));
    
    // 1. 第一阶段归约：使用 Grid-Stride Loop
    // 启动 numBlocks1 个 Block
    generalReduceKernel<<<numBlocks1, BLOCK_SIZE>>>(d_data, d_intermediate, size);
    CUDA_CHECK(cudaGetLastError());
    
    // 分配最终结果内存
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    
    if (numBlocks1 > 1) {
        // 2. 第二阶段归约：处理 numBlocks1 个中间结果
        int numBlocks2 = 1; // 只需要 1 个 Block 完成最终归约
        generalReduceKernel<<<numBlocks2, BLOCK_SIZE>>>(d_intermediate, d_output, numBlocks1);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // 如果只有一个 Block，结果已经写入 d_intermediate[0]
        CUDA_CHECK(cudaMemcpy(d_output, d_intermediate, sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    // 复制结果到主机
    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    // 释放内存
    CUDA_CHECK(cudaFree(d_intermediate));
    CUDA_CHECK(cudaFree(d_output));
    
    return result;
}