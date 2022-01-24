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

__global__ void cuda_ising3(int *G, int *G1, int n){
    int momblocksize = n/(blocksize * gridDim.x);                //dimension of moment block(work per thread)
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * momblocksize;
    int j = (blockIdx.y * blockDim.y + threadIdx.y) * momblocksize;
    int smsize = blocksize * momblocksize;      //shared memory size = threadsperblock * momentsperthread
        
    if(i < n && j < n){
        extern __shared__ int SG[];

        for(int i1 = i;i1 < i + momblocksize;i1++){
            for(int j1 = j;j1 < j + momblocksize;j1++){
                int i2 = i1 - blockIdx.x * blockDim.x * momblocksize;
                int j2 = j1 - blockIdx.y * blockDim.y * momblocksize;
                SG[i2*smsize + j2] = G[i1*n + j1];   //each thread brings its moments to the shared memory
            }
        }
        __syncthreads();
        //nested for loop to give each thread more work
        //check if wanted value is in shared memory of block or read from global
        for(int i1 = i;i1 < i + momblocksize;i1++){
            for(int j1 = j;j1 < j + momblocksize;j1++){
                //shared mem indexes
                int i2 = i1 - blockIdx.x * blockDim.x * momblocksize;
                int j2 = j1 - blockIdx.y * blockDim.y * momblocksize;
                //fit indexes in [0,n-1] properly using modulo
                //jleft(j-1)    jright(j+1)     iup(i-1)    idown(i+1)
                int jleft = (j1 - 1 + n) % n;    int jright = (j1 + 1) % n;
                int iup = (i1 - 1 + n) % n;      int idown = (i1 + 1) % n;
                //shared mem indexes
                int jleft2 = jleft - blockIdx.y * blockDim.y * momblocksize;     
                int jright2 = jright - blockIdx.y * blockDim.y * momblocksize;
                int iup2 = iup - blockIdx.x * blockDim.x * momblocksize;       
                int idown2 = idown - blockIdx.x * blockDim.x * momblocksize;
                //TO DO:
                int sum;
                if(i1 % blockDim.x == 0 && j1 % blockDim.y == 0){
                    //calculate majority of sign simply by using sum
                    sum = G[iup*n + j1] + G[i1*n + jleft] + SG[i2*smsize + j2] + 
                                SG[idown2*smsize +j2] + SG[i2*smsize +jright2];
                }
                else if(i1 % blockDim.x == 0 && j1 % blockDim.y == blockDim.y - 1){
                    //calculate majority of sign simply by using sum
                    sum = G[iup*n + j1] + SG[i2*smsize + jleft2] + SG[i2*smsize + j2] + 
                                SG[idown2*smsize +j2] + G[i1*n +jright];
                }
                else if(i1 % blockDim.x == blockDim.x - 1 && j1 % blockDim.y == 0){
                    //calculate majority of sign simply by using sum
                    sum = SG[iup2*smsize + j2] + G[i1*n + jleft] + SG[i2*smsize + j2] + 
                                G[idown*n +j1] + SG[i2*smsize +jright2];
                }
                else if(i1 % blockDim.x == blockDim.x - 1 && j1 % blockDim.y == blockDim.y - 1){
                    //calculate majority of sign simply by using sum
                    sum = SG[iup2*smsize + j2] + SG[i2*smsize + jleft2] + SG[i2*smsize + j2] + 
                                G[idown*n +j1] + G[i1*n +jright];
                }
                else if(i1 % blockDim.x == 0){
                    //calculate majority of sign simply by using sum
                    sum = G[iup*n + j1] + SG[i2*smsize + jleft2] + SG[i2*smsize + j2] + 
                                SG[idown2*smsize +j2] + SG[i2*smsize +jright2];
                }
                else if(i1 % blockDim.x == blockDim.x - 1){
                    //calculate majority of sign simply by using sum
                    sum = SG[iup2*smsize + j2] + SG[i2*smsize + jleft2] + SG[i2*smsize + j2] + 
                                G[idown*n +j1] + SG[i2*smsize +jright2];
                }
                else if(j1 % blockDim.y == 0){
                    //calculate majority of sign simply by using sum
                    sum = SG[iup2*smsize + j2] + G[i1*n + jleft] + SG[i2*smsize + j2] + 
                                SG[idown2*smsize +j2] + SG[i2*smsize +jright2];
                }
                else if(j1 % blockDim.y == blockDim.y - 1){
                    //calculate majority of sign simply by using sum
                    sum = SG[iup2*smsize + j2] + SG[i2*smsize + jleft2] + SG[i2*smsize + j2] + 
                                SG[idown2*smsize +j2] + G[i1*n +jright];
                }
                else{
                    //calculate majority of sign simply by using sum
                    sum = SG[iup2*smsize + j2] + SG[i2*smsize + jleft2] + SG[i2*smsize + j2] + 
                                SG[idown2*smsize +j2] + SG[i2*smsize +jright2];
                }
                G1[i1*n + j1] = (sum > 0) - (sum < 0);    //implements sign function
            }
        }
    }
}  

int main(int argc, char **argv){
    int N = atoi(argv[1]);
    int gridsize = atoi(argv[2]);
    int K = atoi(argv[3]);
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
    if(cudaMalloc(&d_G, (N * N) * sizeof(int)) != cudaSuccess){
        printf("Couldn't allocate G!\n");
        free(G);
        free(G1);
        exit(1);
    }
    if(cudaMalloc(&d_G1, (N * N) * sizeof(int))){
        printf("Couldn't allocate G1!\n");
        free(G);
        free(G1);
        cudaFree(d_G);
        exit(1);
    }

    //memcpy from host to device
    if(cudaMemcpy(d_G, G, (N * N) * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess){
        printf("Couldn't copy from host to device!\n");
        free(G);
        free(G1);
        cudaFree(d_G);
        cudaFree(d_G1);
        exit(1);
    }
    
    //events for time measuring
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //call function
    dim3 dimBlock( blocksize, blocksize );
    dim3 dimGrid( gridsize, gridsize );
    int smsize = N/gridsize; //shared mem size
    cudaEventRecord(start);
    for(int itr = 0;itr < K; itr++){
        if(!(itr % 2)) cuda_ising3 <<<dimGrid, dimBlock, (smsize * smsize) * sizeof(int)>>> (d_G, d_G1, N);
        else cuda_ising3 <<<dimGrid, dimBlock, (smsize * smsize) * sizeof(int)>>> (d_G1, d_G, N);
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if(error!=cudaSuccess){
            fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
            free(G);
            free(G1);
            exit(1);
        }
    }
    cudaEventRecord(stop);

    //memcpy from device to host
    if(cudaMemcpy(G, d_G, (N * N) * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess){
        printf("Couldn't copy from device to host!\n");
        free(G);
        free(G1);
        cudaFree(d_G);
        cudaFree(d_G1);
        exit(1);
    }
    if(cudaMemcpy(G1, d_G1, (N * N) * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess){
        printf("Couldn't copy from device to host!\n");
        free(G);
        free(G1);
        cudaFree(d_G);
        cudaFree(d_G1);
        exit(1);
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