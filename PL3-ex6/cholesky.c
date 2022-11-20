#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include "omp.h"
#include "cholesky.h"



void print_matrix(int n, double * const matrix){
   printf("\t{\n\t");
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			printf("%.1f, ",matrix[i*n + j]);
		}
		printf("\n\t");
	}
	printf("};\n");
}


//Parallel For
void cholesky_blocked_par_for(const int ts, const int nt, double* Ah[nt][nt])
{
   
   for (int k = 0; k < nt; k++) {

      // Diagonal Block factorization
      potrf (Ah[k][k], ts, ts);
      // Triangular systems
      #pragma omp parallel for
      for (int i = k + 1; i < nt; i++) {
         trsm (Ah[k][k], Ah[k][i], ts, ts);
      }

      // Update trailing matrix
      for (int i = k + 1; i < nt; i++) {
         #pragma omp parallel for
         for (int j = k + 1; j < i; j++) {
            gemm (Ah[k][i], Ah[k][j], Ah[j][i], ts, ts);
         }
         syrk (Ah[k][i], Ah[i][i], ts, ts);
      }

   }
}

//Sequential
void cholesky_blocked(const int ts, const int nt, double* Ah[nt][nt])
{

   for (int k = 0; k < nt; k++) {
      // Diagonal Block factorization
      potrf (Ah[k][k], ts, ts);

      // Triangular systems
      for (int i = k + 1; i < nt; i++) {
         trsm (Ah[k][k], Ah[k][i], ts, ts);
      }

      // Update trailing matrix
      for (int i = k + 1; i < nt; i++) {
         for (int j = k + 1; j < i; j++) {
            gemm (Ah[k][i], Ah[k][j], Ah[j][i], ts, ts);
         }
         syrk (Ah[k][i], Ah[i][i], ts, ts);
      }
   }
}

//task wait
void cholesky_blocked_task_wait(const int ts, const int nt, double* Ah[nt][nt])
{

   for (int k = 0; k < nt; k++) {
      // Diagonal Block factorization
      #pragma omp task
      potrf(Ah[k][k], ts, ts);
      //Decomposition

      // Triangular systems
      for (int i = k + 1; i < nt; i++) {
         #pragma omp task
         trsm (Ah[k][k], Ah[k][i], ts, ts);
      }

      #pragma omp taskwait

      // Update trailing matrix
      for (int i = k + 1; i < nt; i++) {
         for (int j = k + 1; j < i; j++) {
            #pragma omp task
            gemm (Ah[k][i], Ah[k][j], Ah[j][i], ts, ts);
            //matrix multiplication
         }
         #pragma omp task
         syrk (Ah[k][i], Ah[i][i], ts, ts);
         //symmetric rank update
      }
      #pragma omp taskwait

   }
}

//Task dependencies
void cholesky_blocked_task(const int ts, const int nt, double* Ah[nt][nt])
{

   for (int k = 0; k < nt; k++) {
      // Diagonal Block factorization
      #pragma omp task depend(inout: Ah[k][k])
      potrf (Ah[k][k], ts, ts);
      //Decomposition

      // Triangular systems
      for (int i = k + 1; i < nt; i++) {
         #pragma omp task depend(in: Ah[k][k]) depend(inout: Ah[k][i])
         trsm (Ah[k][k], Ah[k][i], ts, ts);
      }

      // Update trailing matrix
      for (int i = k + 1; i < nt; i++) {
         for (int j = k + 1; j < i; j++) {
            #pragma omp task depend(inout: Ah[j][i]) depend(in: Ah[k][i], Ah[k][j])
            gemm (Ah[k][i], Ah[k][j], Ah[j][i], ts, ts);
            //matrix multiplication
         }
         #pragma omp task depend(inout: Ah[i][i]) depend(in: Ah[k][i])
         syrk (Ah[k][i], Ah[i][i], ts, ts);
         //symmetric rank update
      }
   }
}

char file_name[16];
int main(int argc, char* argv[])
{

   if ( argc < 5 || strlen(argv[4])>16) {
      printf( "cholesky matrix_size block_size num_threads filename (max 15 characters)\n" );
      exit( -1 );
   }
   const int  n = atoi(argv[1]); // matrix size
   const int ts = atoi(argv[2]); // tile size
   int num_threads = atoi(argv[3]); // number of threads to use
   memcpy(file_name,argv[4],strlen(argv[4]));


   omp_set_num_threads(num_threads);
   // Allocate matrix
   double * const matrix = (double *) malloc(n * n * sizeof(double));
   assert(matrix != NULL);

   // Init matrix
   initialize_matrix(n, ts, matrix);

   // Allocate matrix
   double * const original_matrix = (double *) malloc(n * n * sizeof(double));
   assert(original_matrix != NULL);

   // Allocate matrix
   double * const expected_matrix = (double *) malloc(n * n * sizeof(double));
   assert(expected_matrix != NULL);

   const int nt = n / ts;

   // Allocate blocked matrix
   double *Ah[nt][nt];

   for (int i = 0; i < nt; i++) {
      for (int j = 0; j < nt; j++) {
         Ah[i][j] = malloc(ts * ts * sizeof(double));
         assert(Ah[i][j] != NULL);
      }
   }

   for (int i = 0; i < n * n; i++ ) {
      original_matrix[i] = matrix[i];
   }
   
   // Sequential
   convert_to_blocks(ts, nt, n, (double(*)[n]) matrix, Ah);
   // warming up libraries
   cholesky_blocked(ts, nt, (double* (*)[nt]) Ah);
   convert_to_blocks(ts, nt, n, (double(*)[n]) matrix, Ah);
   // done warming up
   float t1 = get_time();
   //run sequential version
   cholesky_blocked(ts, nt, (double* (*)[nt]) Ah);
   float t2 = get_time() - t1;
   //calculate timing metrics
   const float seq_time = t2;
   float seq_gflops = (((1.0 / 3.0) * n * n * n) / ((seq_time) * 1.0e+9));

   //saving matrix to the expected result matrix
   convert_to_linear(ts, nt, n, Ah, (double (*)[n]) matrix);
   for (int i = 0; i < n * n; i++ ) {
      expected_matrix[i] = matrix[i];
   }
   // End Sequential

   /*************************************************************************************************************
    * NOTE FOR STUDENTS: 
    * COPY the following code (between multiline comments, up to "End Parallel For") to invoke your versio
    * AND make the following changes:
    *  1. change "cholesky_blocked_par_for" to the name of your method
    *  2. change "par_for_time" and "par_for_gflops" for names that reflect your implementation (e.g. par_task_time)
    *     2.1. Don't forget to change the "par_for_time" variable that is used to calculate the gflops
    *  3. add two lines that print the time for your code (below in "Print Result" section)
    *************************************************************************************************************/
    
   /*****************************************************************************************************
    * Parallel For
    *****************************************************************************************************/
   //resetting matrix
   for (int i = 0; i < n * n; i++ ) {
      matrix[i] = original_matrix[i];
   }
   //require to work with blocks
   convert_to_blocks(ts, nt, n, (double(*)[n]) matrix, Ah);
   t1 = get_time();
   //run parallel version using parallel fors
   cholesky_blocked_par_for(ts, nt, (double* (*)[nt]) Ah);
   t2 = get_time() - t1;
   //calculate timing metrics
   float par_for_time = t2;
   float par_for_gflops = (((1.0 / 3.0) * n * n * n) / ((par_for_time) * 1.0e+9));

   //asserting result, comparing the output to the expect matrix
   convert_to_linear(ts, nt, n, Ah, (double (*)[n]) matrix);
   assert_matrix(n,matrix,expected_matrix);

   /*****************************************************************************************************
    * End Parallel For
    *****************************************************************************************************/

   /*****************************************************************************************************
    * Parallel task wait
    *****************************************************************************************************/
   //resetting matrix
   for (int i = 0; i < n * n; i++ ) {
      matrix[i] = original_matrix[i];
   }
   //require to work with blocks
   convert_to_blocks(ts, nt, n, (double(*)[n]) matrix, Ah);
   t1 = get_time();
   //run parallel version using parallel fors
   cholesky_blocked_task_wait(ts, nt, (double* (*)[nt]) Ah);
   t2 = get_time() - t1;
   //calculate timing metrics
   float par_task_wait = t2;
   float par_for_gflops_task_wait = (((1.0 / 3.0) * n * n * n) / ((par_task_wait) * 1.0e+9));

   //asserting result, comparing the output to the expect matrix
   convert_to_linear(ts, nt, n, Ah, (double (*)[n]) matrix);
   assert_matrix(n,matrix,expected_matrix);

   /*****************************************************************************************************
    * End Parallel task wait
    *****************************************************************************************************/


   /*****************************************************************************************************
    * Parallel task dependencies
    *****************************************************************************************************/
   //resetting matrix
   for (int i = 0; i < n * n; i++ ) {
      matrix[i] = original_matrix[i];
   }

   //require to work with blocks
   convert_to_blocks(ts, nt, n, (double(*)[n]) matrix, Ah);
   t1 = get_time();
   //run parallel version using parallel fors
   cholesky_blocked_task(ts, nt, (double* (*)[nt]) Ah);
   t2 = get_time() - t1;
   //calculate timing metrics
   float par_task_time = t2;
   float par_for_gflops_task = (((1.0 / 3.0) * n * n * n) / ((par_task_time) * 1.0e+9));

   //asserting result, comparing the output to the expect matrix
   convert_to_linear(ts, nt, n, Ah, (double (*)[n]) matrix);
   assert_matrix(n,matrix,expected_matrix);

   /*****************************************************************************************************
    * End Parallel task dependencies
    *****************************************************************************************************/


   // Print result
   printf( "============ CHOLESKY RESULTS ============\n" );
   printf( "  matrix size:                  %dx%d\n", n, n);
   printf( "  block size:                   %dx%d\n", ts, ts);
   printf( "  number of threads:            %d\n", num_threads);
   printf( "  seq_time (s):                 %f\n", seq_time);
   printf( "  seq_performance (gflops):     %f\n", seq_gflops);
   printf( "  par_for_time (s):             %f\n", par_for_time);
   printf( "  par_for_performance(gflops):  %f\n", par_for_gflops);
   printf( "  par_task_wait (s):            %f\n", par_task_wait);
   printf( "  par_task_wait_perfor(gflops): %f\n", par_for_gflops_task_wait);
   printf( "  par_task_time (s):            %f\n", par_task_time);
   printf( "  par_task_time_perfor(gflops): %f\n", par_for_gflops_task);
   printf( "==========================================\n" );


   //file
   free(original_matrix);
   free(expected_matrix);
   // Free blocked matrix
   for (int i = 0; i < nt; i++) {
      for (int j = 0; j < nt; j++) {
         assert(Ah[i][j] != NULL);
         free(Ah[i][j]);
      }
   }
   // Free matrix
   free(matrix);

   FILE *file;
   if (fopen(file_name ,"r")==NULL){

      file = fopen(file_name, "a");
      fprintf(file,"Matrix size,Block size,num_threads,seq_time,seq_performance,par_for_time,par_for_performance,par_task_wait,par_task_wait_performance,par_task_deps_time,par_task_deps_performance\n");
      fclose(file);

   }

   file = fopen(file_name , "a");

   //fprintf(file,"n,ts,num_threads,seq_time,seq_performance\n", );
   fprintf(file,"%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f\n", n,ts, num_threads, seq_time*1000, seq_gflops, par_for_time*1000, par_for_gflops, par_task_wait*1000, par_for_gflops_task_wait, par_task_time*1000, par_for_gflops_task);

   fclose(file);


   return 0;
}

