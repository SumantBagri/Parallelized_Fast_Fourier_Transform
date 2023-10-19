#include <complex>
#include <stdio.h>
#include <math.h>
#include <omp.h>


#define M_PI 3.14159265358979323846 // Pi constant with double precision

using namespace std;


// separate even/odd elements to lower/upper halves of array respectively.
// Due to Butterfly combinations, this turns out to be the simplest way 
// to get the job done without clobbering the wrong elements.
void separate (double * a, long int n) {
    double * b = new double[n];  // get temp heap storage
    //#pragma omp parallel for default(none) shared(a,b,n)
    for(int i=0; i<n; i+=2)
    {    // copy all odd elements to heap storage
        b[i]   = a[2*i+2];
        b[i+1] = a[2*i+3];
    }
    //#pragma omp parallel for default(none) shared(a,b,n)
    for(int i=0; i<n; i+=2)    // copy all even elements to lower-half of a[]
    {
        a[i]   = a[i*2];
        a[i+1] = a[2*i + 1];
    }
    //#pragma omp parallel for default(none) shared(a,b,n)
    for(int i=0; i<n; i+=2)    // copy all odd (from heap) to upper-half of a[]
    {
        a[i+n] = b[i];
        a[i+n+1] = b[i+1];
    }
    delete[] b;                 // delete heap storage
}

// N must be a power-of-2, or bad things will happen.
// Currently no check for this condition.
//
// N input samples in X[] are FFT'd and results left in X[].
// Because of Nyquist theorem, N samples means 
// only first N/2 FFT results in X[] are the answer.
// (upper half of X[] is a reflection with no new information).
void fft2 (double * X, long int N) {
    if(N < 2) {
        // bottom of recursion.
        // Do nothing here, because already X[0] = x[0]
    } 

    else {
        separate(X,N);      // all evens to lower half, all odds to upper half
        fft2(X,     N/2);   // recurse even items
        fft2(X+N, N/2);   // recurse odd  items 

        // combine results of two half recursions
        //#pragma omp parallel for shared(X,N)
        for(int k=0; k<N; k+=2) {  
            double e[2] = {X[k],X[k+1]}; // even,
            double o[2] = {X[k+N],X[k+N+1]};   // odd
                         // w is the "twiddle-factor"
            double w[2] = {cos(-1.*M_PI*k/N),sin(-1.*M_PI*k/N)};
            X[k]   = e[0] + w[0]*o[0] - w[1]*o[1];
            X[k+1] = e[1] + w[0]*o[1] + w[1]*o[0];
            X[k+N] = e[0] - w[0]*o[0] + w[1]*o[1];
            X[k+N+1] = e[1] - w[0]*o[1] - w[1]*o[0];
        }
    }

}

// simple test program
int main () {

    //omp_set_num_threads(4);
    const long int nSamples = pow(2,25);
    //printf("%lf\n", pow(2,20));
    double nSeconds = 1.0;                      // total time for sampling
    double sampleRate = nSamples / nSeconds;    // n Hz = n / second 
    double freqResolution = sampleRate / nSamples; // freq step in FFT result

    double * x = new double[2*nSamples];
    double * X = new double[2*nSamples];

    const int nFreqs = 2;
    double freq[nFreqs] = { 2, 5}; // known freqs for testing
    
    // generate samples for testing
    for(int i=0; i<nSamples; i++) {
        //x[i] = complex<double>(0.,0.);
        // sum several known sinusoids into x[]
        for(int j=0; j<nFreqs; j++)
            x[2*i] += sin( 2*M_PI*freq[j]*i/nSamples );
        X[2*i] = x[2*i];        // copy into X[] for FFT work & result
    }
    // compute fft for this data
    fft2(X,nSamples);
    
    printf("  n\tx[]\tX[]\tf\n");       // header line
    // loop to print values
    // for(int i=0; i<nSamples; i++) {
    //     // printf("% 3d\t%+.3f\t%+.3f\t%g\n",
    //     //     i, x[i].real(), abs(X[i]), i*freqResolution );
    //     printf("%d\t%lf\t%lf\n",i,x[2*i],sqrt(X[2*i]*X[2*i] + X[2*i+1]*X[2*i+1]) );
    // }
}