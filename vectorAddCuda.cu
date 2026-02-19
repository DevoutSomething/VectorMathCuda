#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

int* vectorAdd(int *Va, int *Vb, int vectorSize);
__global__ void vectorAddKernel(int *Va, int *Vb, int *Vres, int vectorSize);

int main(void){ 
    int vectorSize = 100000000; // 100 million elements
    int *Va = (int*)malloc(vectorSize * sizeof(int));
    int *Vb = (int*)malloc(vectorSize * sizeof(int));
    for(int i = 0; i < vectorSize; i++) {
        Va[i] = rand() % 100;
        Vb[i] = rand() % 100;
    }

    clock_t start = clock();
    int *Vres = vectorAdd(Va, Vb, vectorSize);
    clock_t end = clock();
    printf("CPU vector add took %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    int *VresCuda = (int*)malloc(vectorSize * sizeof(int));

    int *d_a, *d_b, *d_res;

    // Allocate GPU memory
    cudaMalloc(&d_a, vectorSize * sizeof(int));
    cudaMalloc(&d_b, vectorSize * sizeof(int));
    cudaMalloc(&d_res, vectorSize * sizeof(int));

    // Copy input vectors to GPU
    cudaMemcpy(d_a, Va, vectorSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, Vb, vectorSize * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Record start event
    cudaEventRecord(startEvent);

    // Launch kernel
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_res, vectorSize);

    // Record stop event
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    printf("GPU kernel only took %f ms\n", milliseconds);

    // Copy result back to CPU
    cudaMemcpy(VresCuda, d_res, vectorSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    // Verify results
    for(int i = 0; i < vectorSize; i++) {
        if (Vres[i] != VresCuda[i]) {
            printf("Error at index %d: %d != %d\n", i, Vres[i], VresCuda[i]);
            break;
        }
    }

    free(Va);
    free(Vb);
    free(Vres);
    free(VresCuda);

    return 0;
}

// CPU version of vector add
int* vectorAdd(int *Va, int *Vb, int vectorSize) { 
    int *Vres = (int*)malloc(vectorSize * sizeof(int));
    for(int i = 0; i < vectorSize; i++) {
        Vres[i] = Va[i] + Vb[i];
    }
    return Vres;
}

// CUDA kernel for vector add
__global__ void vectorAddKernel(int *Va, int *Vb, int *Vres, int vectorSize) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < vectorSize) {
        Vres[id] = Va[id] + Vb[id];
    }
}
