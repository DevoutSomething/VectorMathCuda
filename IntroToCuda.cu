#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello() {
int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("gridDim is %d\n", gridDim.x);
    printf("Hello from GPU! Thread ID: %d, Block ID: %d\n",
           id, blockIdx.x   );
}

int main() {
    //Grid has 2 blocks with 8 threads each, gridDim.x = 2, blockDim.x = 8
    hello<<<2, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}
