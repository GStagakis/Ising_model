#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

void seq_ising(int *G, int *G1, int n, int k){
    //G -> uniform random initial state
    //n -> matrix dimension
    //k -> iterations
    int itr, i, j;
    //temporary pointer used to swap the pointers at the end of each iteration
    int *temp;

    for(itr = 0;itr < k;itr++){
        for(i = 0; i < n; i++){
            for(j = 0; j < n; j++){
                //fit indexes in [0,n-1] properly using modulo
                //jleft(j-1)    jright(j+1)     iup(i-1)    idown(i+1)
                int jleft = (j - 1 + n) % n;    int jright = (j + 1) % n;
                int iup = (i - 1 + n) % n;      int idown = (i + 1) % n;
                //calculate majority of sign simply by using sum
                int sum = G[iup*n + j] + G[i*n + jleft] + G[i*n + j] + G[idown*n + j] 
                                    + G[i*n + jright];
                G1[i*n + j] = (sum > 0) - (sum < 0);    //implemets sign function
            }    
        }
        //swap the pointers at the end of the iteration
        temp = G;
        G = G1;
        G1 = temp;
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
    
    struct timespec start, finish;
    //call function
    clock_gettime(CLOCK_MONOTONIC, &start);
    seq_ising(G, G1, N, K);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    
    //Because we didn't use call by reference in function call we need to check 
    //if G and G1 point at the right array (odd # of iterations -> swap)
    if(K % 2){
        int *temp = G;
        G = G1;
        G1 = temp;
    }

    //print results
    //printf("********After %d Iterations********\n", K);
    //print2D(G, N, N);

    //print time
    double elapsed;
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("%f\n", elapsed);

    //free data
    free(G);
    free(G1);
    
    return 0;
}