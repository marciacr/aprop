/*
**  PROGRAM: Mandelbrot area (solution)
**
**  PURPOSE: Program to compute the area of a  Mandelbrot set.
**           The correct answer should be around 1.510659.
**
**  USAGE:   Program runs without input ... just run the executable
**
*/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
# define NPOINTS 1000
# define MAXITER 1000

struct d_complex{
   double r;
   double i;
};

void testpoint(struct d_complex);
struct d_complex c;
int numoutside = 0;

void calc_seq(){

    int i, j;
    double area, error, eps  = 1.0e-5;

    for (i=0; i<NPOINTS; i++) {
        for (j=0; j<NPOINTS; j++) {
        c.r = -2.0+2.5*(double)(i)/(double)(NPOINTS)+eps;
        c.i = 1.125*(double)(j)/(double)(NPOINTS)+eps;
        testpoint(c);
        }
    }
    // Calculate area of set and error estimate and output the results
    area=2.0*2.5*1.125*(double)(NPOINTS*NPOINTS-numoutside)/(double)(NPOINTS*NPOINTS);
    error=area/(double)NPOINTS;

    printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n",area,error);
}

void calc_parallel(struct d_complex c){

    double area, error, eps  = 1.0e-5;
    int i, j;
    #pragma omp parallel num_threads(8) private(c,i,j) 
    {
        #pragma omp for collapse(2) //schedule(dynamic, 100)
        for (i=0; i<NPOINTS; i++) {
            for (j=0; j<NPOINTS; j++) {
                c.r = -2.0+2.5*(double)(i)/(double)(NPOINTS)+eps;
                c.i = 1.125*(double)(j)/(double)(NPOINTS)+eps;
                testpoint(c);
            }
        }
    }
    
    // Calculate area of set and error estimate and output the results
    area=2.0*2.5*1.125*(double)(NPOINTS*NPOINTS-numoutside)/(double)(NPOINTS*NPOINTS);
    error=area/(double)NPOINTS;

    printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n",area,error);
}

char file_name[] = "dados_openmp.csv";

int main(){
    FILE *file;

    if (fopen(file_name,"r")==NULL){

        file = fopen(file_name, "a");
        fprintf(file,"i, seq_time, seq_par\n");

    }


    for (int i = 1; i<25; i++){


        numoutside =0;
        
        double begin = omp_get_wtime();
        calc_seq(c);
        double end = omp_get_wtime();
        double sequential_time = end - begin;

        numoutside =0;

        begin = omp_get_wtime();
        calc_parallel(c);
        end = omp_get_wtime();
        double parallel_time = end - begin;

        printf("\n- ==== Performance ==== -\n");
        printf("Sequential time: %fs\n",sequential_time);
        printf("Parallel time: %fs\n",parallel_time);

        fprintf(file,"%d,%f,%f\n", i,sequential_time, parallel_time);

    }
    fclose(file);



}

void testpoint(struct d_complex c){

// Does the iteration z=z*z+c, until |z| > 2 when point is known to be outside set
// If loop count reaches MAXITER, point is considered to be inside the set

       struct d_complex z;
       int iter;
       double temp;

       z=c;
       for (iter=0; iter<MAXITER; iter++){
         temp = (z.r*z.r)-(z.i*z.i)+c.r;
         z.i = z.r*z.i*2+c.i;
         z.r = temp;
         if ((z.r*z.r+z.i*z.i)>4.0) {
           #pragma omp atomic 
           numoutside++;
           break;
         }
       }

}