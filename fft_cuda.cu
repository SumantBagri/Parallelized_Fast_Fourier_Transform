#include <iostream>
#include <cstdio>
#include <complex>
//#include <thrust/complex.h>
#include <cuComplex.h>
using namespace std;
#include <cuda.h>
#include <math_constants.h>
#include <math.h>
#include <cuda_runtime.h>

#define M_PI 3.14159265358979323846 // Pi constant with double precision

__host__ __device__ static __inline__ cuDoubleComplex cuCexp(cuDoubleComplex x)
{
	double factor = exp(cuCreal(x));
	return make_cuDoubleComplex(factor * cos(cuCimag(x)), factor * sin(cuCimag(x)));
}

/*
Separate even/odd elements to lower/upper halves of array respectively.
Defined as __device__ function as it will be acessed by the fft kernel
*/

__device__ void separate (cuDoubleComplex* a, int n, cuDoubleComplex* b)
{
   
  //cudaMalloc((void **)&b,(n*sizeof(cuDoubleComplex))/2);  // get temp heap storage
  for(int i=0; i<n/2; i++)    // copy all odd elements to heap storage
      b[i] = a[i*2 + 1];
  for(int i=0; i<n/2; i++)    // copy all even elements to lower-half of a[]
      a[i] = a[i*2];
  for(int i=0; i<n/2; i++)    // copy all odd (from heap) to upper-half of a[]
      a[i+n/2] = b[i];
                  // delete heap storage
}

__global__ void fft (cuDoubleComplex* X_d, int local_n, int numBlock, int block_size, cuDoubleComplex* b)
{
    
  if(local_n < 2) {
      // bottom of recursion.
      // Do nothing here, because already X[0] = x[0]
  } else {
      separate(X_d,local_n,b);      // all evens to lower half, all odds to upper half
      __syncthreads();
      fft <<< 1, 1 >>> (X_d, local_n/2, 1, 1, b);   // recurse even items
      __syncthreads();
      fft <<< 1, 1 >>> (X_d+local_n/2, local_n/2, 1, 1, b);   // recurse odd  items
      __syncthreads();
      // combine results of two half recursions
      for(int k=0; k<local_n/2; k++) {
          cuDoubleComplex e = X_d[k];   // even
          cuDoubleComplex o = X_d[k + local_n/2];   // odd
                       // w is the "twiddle-factor"
          __syncthreads();
          cuDoubleComplex w = cuCexp( make_cuDoubleComplex(0,-2.*M_PI*(double)(k)/((double)(local_n))) );
          X_d[k] = cuCadd(e, cuCmul(w, o));
          __syncthreads();
          X_d[k + local_n/2] = cuCsub(e, cuCmul(w, o));
          __syncthreads();
      }
  }
  //cudaFre
}

// kernel for generating large sized sampling data using the GPU
__global__ void datagen (cuDoubleComplex* x_d, int n_local, int nSamples)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  const int nFreqs = 5;
  double freq[nFreqs] = { 2, 5, 11, 17, 29 }; // known freqs for testing

  // generate samples for testing
  for(int i=0; i<n_local; i++) {
      x_d[idx+i] = make_cuDoubleComplex(0,0);
      // sum several known sinusoids into x[]
      for(int j=0; j<nFreqs; j++)
          x_d[i+idx] = cuCadd(x_d[i+idx], make_cuDoubleComplex(sin( 2*M_PI*(float)(freq[j])*(float)(i+idx)/((float)nSamples) ), 0));
  }
}

int main() {

  /*
  NOTE:
  Keep nSamples, block_size and n_local a power of 2
  otherwise bad things will happen
  */

  const int nSamples = 512;        // total number of sampling points
  int n_local = 1;                   // local number of samples in each thread
                                     // constraint to keep it a power of 2
  double nSeconds = 1.0;                         // total time for sampling
  double sampleRate = nSamples / nSeconds;       // n Hz = n / second
  double freqResolution = sampleRate / nSamples; // freq step in FFT result

  cudaError_t errorcode = cudaSuccess;          // for cuda error mgmt

  complex<double> x_h[nSamples];                // storage for sample data(host copy)
  complex<double> X_h[nSamples];                // storage for FFT answer(host copy)

  cuDoubleComplex* x_d;                          // storage for sample data(device copy)
  // Assign memory for device copy of sample data storage with error handling
  if (( errorcode = cudaMalloc((void **)&x_d,nSamples*sizeof(cuDoubleComplex)))!= cudaSuccess)
  {
    cout << "cudaMalloc(): " << cudaGetErrorString(errorcode) << endl;
    exit(1);
  }

  cuDoubleComplex* X_d;                          // storage for FFT answer(device copy)
  // Assign memory for device copy of FFT answer storage with error handling
  if (( errorcode = cudaMalloc((void **)&X_d,nSamples*sizeof(cuDoubleComplex)))!= cudaSuccess)
  {
    cout << "cudaMalloc(): " << cudaGetErrorString(errorcode) << endl;
    exit(1);
  }

  int block_size = 256;
  int numBlock = nSamples/(n_local*block_size);  // Essential that everything is a power of 2

/*****************************************************************************************/

  datagen <<< numBlock, block_size >>> (x_d, n_local, nSamples); // Kernel call for data generation

/*****************************************************************************************/

  // Copy the device copy of generated data to host copy with error handling
  if((errorcode = cudaMemcpy(x_h, x_d, nSamples*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost))!=cudaSuccess)
  {
    cout << "3cudaMemcpy(): " << cudaGetErrorString(errorcode) << endl;
    exit(1);
  }

  for(int i=0; i<nSamples; i++)  X_h[i] = x_h[i];        // copy into X[] for FFT work & result

  cudaFree(x_d);          // Needed as fresh GPU computations will begin post this point
  


  cuDoubleComplex* b;
  cudaMalloc((void **)&b,(1024*sizeof(cuDoubleComplex))/2);
  // thrust::complex<double>* ex;
  // ex = thrust::complex<double>(1.0,1.0);
  for(int i=0; i<nSamples; i++) cout<< (x_h[i])<<"\n";	
  // Copy the host copy of original FFT answer storage(X_h) to device copy(X_d)
  if((errorcode = cudaMemcpy(X_d, X_h, nSamples*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice))!=cudaSuccess)
  {
    cout << "2cudaMemcpy(): " << cudaGetErrorString(errorcode) << endl;
    exit(1);
  }

/****************************************************************************************/

  fft <<< 1, 1 >>> (X_d, nSamples, numBlock, block_size, b); // Kernel call for fft computation

/****************************************************************************************/

  if((errorcode = cudaMemcpy(X_h, X_d, nSamples*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost))!=cudaSuccess)
  {
    cout << "1cudaMemcpy(): " << cudaGetErrorString(errorcode) << endl;
    exit(1);
  }

  
  ::freopen("FFT_out.txt", "w", stdout);

  cout<<"  n\tx[]\tX[]\tf\n";       // header line
  // loop to print values
  for(int i=0; i<nSamples; i++)
  {
      cout<<i<<"\t";
      cout<< ((x_h[i]).real()) << "\t";
      cout<< abs(X_h[i]) << "\t" ;
      cout<<i*freqResolution<<"\n";
  }

  ::fclose(stdout);
 
  cudaFree(X_d); 
  cudaFree(b);  
return 0;
}
