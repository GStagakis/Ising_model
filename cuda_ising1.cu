#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int blocksize = 16;    //max blocksize is 32 for our GPU

void print2D(int *G, int m, int n){
    int i,j;
    for(i = 0;i < m;i++){
        for(j = 0;j < n;j++){
            printf("%2d ", G[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void cuda_ising1(int *G, int *G1, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < n && j < n){
        //fit indexes in [0,n-1] properly using modulo
        //jleft(j-1)    jright(j+1)     iup(i-1)    idown(i+1)
        int jleft = (j - 1 + n) % n;    int jright = (j + 1) % n;
        int iup = (i - 1 + n) % n;      int idown = (i + 1) % n;
        //calculate majority of sign simply by using sum
        int sum = G[iup*n + j] + G[i*n + jleft] + G[i*n + j] + 
                    G[idown*n +j] + G[i*n +jright];
        G1[i*n + j] = (sum > 0) - (sum < 0);    //implemets sign function
    }
}  

int main(int argc, char **argv){
    int N = atoi(argv[1]);
    int K = atoi(argv[2]);
    int i;
    //create uniform random initial state
    srand(time(NULL));
    int *G = (int *)malloc((N * N) * sizeof(int));
    for(i = 0;i < N*N;i++) G[i] = (rand() % 2) == 0 ? -1 : 1;
    //printf("********Uniform random initial state********\n");
    //print2D(G, N, N);

    //second array used to store the results of first iteration
    int *G1 = (int *)malloc((N * N) * sizeof(int));

    //malloc memory int the gpu
    int *d_G, *d_G1;
    cudaMalloc(&d_G, (N * N) * sizeof(int));
    cudaMalloc(&d_G1, (N * N) * sizeof(int));

    //memcpy from host to device
    if(cudaMemcpy(d_G, G, (N * N) * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess){
        printf("Couldn't copy from host to device!\n");
    }

    //events for time measuring
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //call function
    dim3 dimBlock( blocksize, blocksize );
    dim3 dimGrid( N/dimBlock.x, N/dimBlock.y );
    cudaEventRecord(start);
    for(int itr = 0;itr < K; itr++){
        if(!(itr % 2))  cuda_ising1 <<<dimGrid, dimBlock>>> (d_G, d_G1, N);
        else cuda_ising1 <<<dimGrid, dimBlock>>> (d_G1, d_G, N);
    }
    cudaEventRecord(stop);  

    //memcpy from device to host
    if(cudaMemcpy(G, d_G, (N * N) * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess){
        printf("Couldn't copy from device to host!\n");
    }
    if(cudaMemcpy(G1, d_G1, (N * N) * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess){
        printf("Couldn't copy from device to host!\n");
    }

    //!Because we couldn't use call by reference in kernel we need to check if
    //G and G1 point at the right array (odd # of iterations -> swap)
    if(K % 2){
        int *temp = G;
        G = G1;
        G1 = temp;
    }

    //print results
    //printf("********After %d Iterations********\n", K);
    //print2D(G, N, N);

    //print time
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f\n", milliseconds/1000.0);

    //free data
    free(G);
    free(G1);
    cudaFree(d_G);
    cudaFree(d_G1);

    return 0;
}