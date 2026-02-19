#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

// CUDA kernel declaration
__global__ void vectorMultiplyKernel(int *Va, int *Vb, int *partialSums, int vectorSize);

// CPU dot product function
long long cpuDotProduct(int *Va, int *Vb, int vectorSize) {
    long long sum = 0;
    for(int i = 0; i < vectorSize; i++) {
        sum += (long long)Va[i] * Vb[i];
    }
    return sum;
}

int main(void){ 
    int vectorSize = 1000000000; // 100 million elements
    int *Va = (int*)malloc(vectorSize * sizeof(int));
    int *Vb = (int*)malloc(vectorSize * sizeof(int));
    for(int i = 0; i < vectorSize; i++) {
        Va[i] = rand() % 100;
        Vb[i] = rand() % 100;
    }

    // --- CPU Dot Product ---
    clock_t startCPU = clock();
    long long dotCPU = cpuDotProduct(Va, Vb, vectorSize);
    clock_t endCPU = clock();
    printf("CPU dot product result: %lld\n", dotCPU);
    printf("CPU dot product took %f seconds\n", ((double)(endCPU - startCPU)) / CLOCKS_PER_SEC);

    // --- GPU Dot Product ---
    int threadsPerBlock = 256;
    int blocksPerGrid = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;

    int *partialSums = (int*)malloc(blocksPerGrid * sizeof(int));
    int *d_a, *d_b, *d_partialSums;

    cudaMalloc(&d_a, vectorSize * sizeof(int));
    cudaMalloc(&d_b, vectorSize * sizeof(int));
    cudaMalloc(&d_partialSums, blocksPerGrid * sizeof(int));

    cudaMemcpy(d_a, Va, vectorSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, Vb, vectorSize * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent);

    // Launch kernel with shared memory size = threadsPerBlock * sizeof(int)
    vectorMultiplyKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_a, d_b, d_partialSums, vectorSize);

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    printf("GPU kernel only took %f ms\n", milliseconds);

    cudaMemcpy(partialSums, d_partialSums, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    long long dotGPU = 0;
    for(int i = 0; i < blocksPerGrid; i++) {
        dotGPU += partialSums[i];
    }

    printf("GPU dot product result: %lld\n", dotGPU);

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partialSums);

    free(Va);
    free(Vb);
    free(partialSums);

    return 0;
}

// CUDA kernel for dot product
__global__ void vectorMultiplyKernel(int *Va, int *Vb, int *partialSums, int vectorSize) {
    extern __shared__ int blockSum[];
    int id = blockIdx.x * blockDim.x + threadIdx.x; // Global thread ID
    int tid = threadIdx.x; // local thread ID within the block

    int temp = 0;
    if(id < vectorSize) temp = Va[id] * Vb[id];
    blockSum[tid] = temp;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2) { //this weird code makes it so that the threads in the block can sum up their results together in parralel
        if(tid < stride) blockSum[tid] += blockSum[tid + stride];
        __syncthreads();
    }

    if(tid == 0) partialSums[blockIdx.x] = blockSum[0];
}
