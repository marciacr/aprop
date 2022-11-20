#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>

/**
 * Matrix multiplication 
 **/
#define N 1024
#define BS 32
#define MIN_RAND 0
#define MAX_RAND 100
#define N_THREADS 12

FILE *myFile;

int M[N][N];
int original_M[N][N];
int correct_M[N][N];

void fill(int* matrix, int height,int width);
void print(int* matrix,int height,int width);
void copy_to_M();
void setup_correct_M();
void assert(int M[N][N],int expected[N][N]);

/**
 * The sequential version
*/
void seq()
{
    int i, j;
    for(i = 1; i < N - 1; i++)
        for(j = 1; j < N - 1; j++)
           M[i][j] = (M[i][j-1] + M[i-1][j] + M[i][j+1] + M[i+1][j])/4.0;
}

/**
 * The parallel version
*/
void par()
{
    int step =N/BS; //ratio 
    int task_i, task_j,i, j;

    for (int task_i = 0; task_i < step; task_i++){ //task_i e task_j representam a posićão linha e coluna do primeiro elemento de cada bloco (bli, blj)
        for (int task_j = 0; task_j < step; task_j++){

            int init_i = 1 + (task_i*BS); //task_i should init at task_i * block size
            int end_i = init_i + BS; //task_i ends when the next task_i begin
            int init_j = 1 + (task_j*BS); //task_j should init at task_j * block size
            int end_j = init_j + BS; ////task_j ends when the next task_j begin

            if (end_i > N-1) //Keep the matrix size correct
                end_i = N-1;
            if (end_j > N-1)
                end_j = N-1;    

            #pragma omp task depend(in: M[init_i][init_j-BS], M[init_i-BS][init_j]) depend(out: M[init_i][init_j])
            for(i = init_i; i < end_i; i++)
                for(j = init_j; j < end_j; j++)
                //Parallel code for letter a) parallelize task per task
                // #pragma omp task depend(in: M[i][j-1], M[i-1][j]) depend(out: M[i][j]) //firstprivate(i,j)
                M[i][j] = (M[i][j-1] + M[i-1][j] + M[i][j+1] + M[i+1][j])/4.0;
        }
    }    
}

int main(int argc, char *argv[])
{
      
    myFile = fopen("ex5.csv", "a");

    for(int i=0; i<5;i++){
        if(N % BS != 0){
            fprintf(stderr,"[ERROR] N is not multiple of BS: %d %% %d = %d\n", N,BS,N%BS);
            exit(1);
        }
        srand(time(NULL));

        //Fill A and B with random ints
        fill((int *)original_M,N,N);
        copy_to_M();

        //Run sequential version to compare time 
        //and also to have the correct result
        double begin = omp_get_wtime();
        seq();
        double end = omp_get_wtime();
        double sequential_time = end - begin;

        //save correct result in global variable 'correct_M'
        setup_correct_M();
        // reset matrix
        copy_to_M();

        //parallel code
        omp_set_num_threads(N_THREADS);

        begin = omp_get_wtime();
        #pragma omp parallel
        #pragma omp single
        {
            par();
        }
        end = omp_get_wtime();
        //print((int*)M,N,N);
        double parallel_time = end - begin;

        //compare your result invoking the following code (just uncomment the code):
        assert(M,correct_M);

        printf("\n- ==== Performance ==== -\n");
        printf("Sequential time: %fs\n",sequential_time);
        printf("Parallel   time: %fs\n",parallel_time);
        
        fprintf(myFile, "%dx%d, %dx%d, %d, %f, %f \n", N, N, BS, BS, N_THREADS, sequential_time, parallel_time);
    }
    fclose(myFile);
    
}

void copy_to_M(){
    for (int l = 0; l < N; l++)
    {
        for (int n = 0; n < N; n++)
        {
            M[l][n] = original_M[l][n] ;
        }
    }
}

void fill(int* matrix, int height,int width){
    for (int l = 0; l < height; l++)
    {
        for (int n = 0; n < width; n++)
        {
            *((matrix+l*width) + n) = MIN_RAND + rand()%(MAX_RAND-MIN_RAND+1);
        }
    }
}

void print(int* matrix,int height,int width){
    
    for (int l = 0; l < height; l++)
    {
        printf("[");
        for (int n = 0; n < width; n++)
        {
            printf(" %5d",*((matrix+l*width) + n));
        }
        printf(" ]\n");
    }
}

void assert(int C[N][N],int expected[N][N]){
    for (int l = 0; l < N; l++)
    {
        for (int n = 0; n < N; n++)
        {
            if(C[l][n] != expected[l][n]){
                printf("Wrong value at position [%d,%d], expected %d, but got %d instead\n",l,n,expected[l][n],C[l][n]);
                exit(-1);
            }
        }
        
    }
}

void setup_correct_M(){
    
    for (int l = 0; l < N; l++)
    {
        for (int n = 0; n < N; n++)
        {
            correct_M[l][n] = M[l][n];
        }
    }
}
