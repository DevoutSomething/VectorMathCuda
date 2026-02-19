#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

// CUDA kernel declaration
__global__ void vectorLengthKernel(int *Va, int *partialSums, int vectorSize);

// CPU dot product function
long long cpuLength(int *Va, int vectorSize) {
    long long sum = 0;
    for(int i = 0; i < vectorSize; i++) {
        sum += (long long)Va[i] * Va[i];
    }
    return sqrt(sum);
}

int main(void){ 
    int vectorSize = 1000000000; // 100 million elements
    int *Va = (int*)malloc(vectorSize * sizeof(int));
    for(int i = 0; i < vectorSize; i++) {
        Va[i] = rand() % 100;
    }

    clock_t startCPU = clock();
    long long lengthCPU = cpuLength(Va, vectorSize);
    clock_t endCPU = clock();
    printf("CPU vector length result: %lld\n", lengthCPU);
    printf("CPU vector length took %f seconds\n", ((double)(endCPU - startCPU)) / CLOCKS_PER_SEC);

    // --- GPU Vector Length ---
    int threadsPerBlock = 256;
    int blocksPerGrid = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;

    int *partialSums = (int*)malloc(blocksPerGrid * sizeof(int));
    int *d_a, *d_partialSums;

    cudaMalloc(&d_a, vectorSize * sizeof(int));
    cudaMalloc(&d_partialSums, blocksPerGrid * sizeof(int));

    cudaMemcpy(d_a, Va, vectorSize * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent);

    // Launch kernel with shared memory size = threadsPerBlock * sizeof(int)
    vectorLengthKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_a, d_partialSums, vectorSize);
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
    dotGPU = sqrt(dotGPU);

    printf("GPU vector length result: %lld\n", dotGPU);

    // Free memory
    cudaFree(d_a);
    cudaFree(d_partialSums);

    free(Va);
    free(partialSums);

    return 0;
}

// CUDA kernel for vector length
__global__ void vectorLengthKernel(int *Va, int *partialSums, int vectorSize) {
    extern __shared__ int blockSum[];
    int id = blockIdx.x * blockDim.x + threadIdx.x; // Global thread ID
    int tid = threadIdx.x; // local thread ID within the block

    int temp = 0;
    if(id < vectorSize) temp = Va[id] * Va[id];
    blockSum[tid] = temp;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2) { //this weird code makes it so that the threads in the block can sum up their results together in parralel
        if(tid < stride) blockSum[tid] += blockSum[tid + stride];
        __syncthreads();
    }

    if(tid == 0) partialSums[blockIdx.x] = blockSum[0];
}
