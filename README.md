# Fast Fourier Transform
## Overall Description
Contains Serial (recursive) and Parallel Version (openMPI and CUDA) of the Fast Fourier Transform algorithm. Used to converts a signal from its original domain to a representation in the frequency domain and vice versa. A performance analysis was performed to show display execution time, speed-up, and efficiency ([pdf]())

## Compilation Instructions
```
/************************************************************/
SERIAL IMPLEMENTATION CODE FILE: fft_recursive.cpp

Description--> The above application is the serialized 
	       RADIX-2 decimation in time (DIT) 
	       implementation of the Cooley-Tukey(recursive) 
     	       Algorithm for Fast Fourier Transform

Compilation/Run Instructions:
Compile using: "g++ -o fft_recursive fft_recursive.cpp -lm"
Run using    : "./fft"
/************************************************************/
```

```
/************************************************************/
OPENMP IMPLEMENTATION CODE FILE: fft_openmp.c

Description--> The above application is the parallelized 
	       implementation of the Workspace version of 
	       Self Sorting Fast Fourier Transform using OpenMP

Compilation/Run Instructions:

Compile using: "gcc -o fft_openmp -fopenmp fft_openmp.c -lm"
Run using    : "./fft_openmp"

/************************************************************/
```

```
/************************************************************/
CUDA IMPLEMENTATION CODE FILE  : fft_cuda.cu

Description--> The above application is the parallelized 
	       implementation of the RADIX-2 decimation in time 
	       (DIT) implementation of the Cooley-Tukey(recursive)
               Algorithm for Fast Fourier Transform

Compilation/Run Instructions:

NOTE: 1) This parallelized version based on CUDA uses 'Dynamic Parallelism'
         which can be compiled only with 'compute_35' or greater architecture
      2) Running with cuda-memcheck is to help in debugging memory access
         The compiled code can simply be run as "./fft_cuda" without debugging
	 verbose.

Compile using: "nvcc -arch=sm_35 -rdc=true -o fft_cuda fft_cuda.cu -lcudadevrt"
Run using    : "cuda-memcheck ./fft-cuda"

/************************************************************/
```

## Performance Analysis Breakdown
### Execution Time
Execution time decreases as the number of parallel threads increases. It plateaus around 6 threads for an input size of $$2^{20}$$ showing a slight increase when the number of threads increases to 8
![Comparison of time take to compute the FFT with OpenMP with number of threads](https://i.imgur.com/pwfAWvE.png)