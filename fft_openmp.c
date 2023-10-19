# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>
# include <string.h>

int main ( void );
void ccopy ( int n, double x[], double y[] );
void cfft2 ( int n, double x[], double y[], double w[]);
void cffti ( int n, double w[] );
void step ( int n, int mj, double a[], double b[], double c[], double d[], 
  double w[]);

/******************************************************************************/

int main ( void )
{
  omp_set_num_threads(8);
  int i,j;
  int it;
  int n = pow(2,25);
  int thread_num;
  double *w = malloc (     n * sizeof ( double ) );
  double *y = malloc ( 2 * n * sizeof ( double ) );

  double *z = malloc ( 2 * n * sizeof ( double ) );
  memset(z, 0, 2*n*sizeof(double));

  double * x = malloc (2* n * sizeof ( double ) );
  memset(x, 0, 2*n*sizeof( double ));

  const int nFreqs = 2;
  double freq[2] = {2,5}; // known freqs for testing

  for (i = 0; i < n; i++ )
  {
    for(j=0; j<nFreqs; j++)
            z[2*i] += sin( 2*M_PI*freq[j]*i/n );

    x[2*i] = z[2*i];
    // printf("%d\t%lf\t%lf\n",i,x[2*i], x[2*i+1]);
  }
    
  cffti ( n, w );
  cfft2 ( n, x, y, w );
  printf("\n\n");
  // for (i = 0; i < n; i++ )
  // {
  //   printf("%d\t%lf\t%lf\n",i,x[2*i], x[2*i+1]);
  // }

  free ( w );
  free ( x );
  free ( y );
  free ( z );
  
  printf ( "\n" );
  printf ( "FFT_OPENMP:\n" );
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );

  return 0;
}
/******************************************************************************/

void ccopy ( int n, double x[], double y[] )

{
  int i;

  for ( i = 0; i < n; i++ )
  {
    y[i*2+0] = x[i*2+0];
    y[i*2+1] = x[i*2+1];
   }
  return;
}
/******************************************************************************/

void cfft2 ( int n, double x[], double y[], double w[])

{
  int j;
  int m;
  int mj;
  int tgle;

   m = ( int ) ( log ( ( double ) n ) / log ( 1.99 ) );
   mj   = 1;
/*
  Toggling switch for work array.
*/
  tgle = 1;

  step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w);

  if ( n == 2 )
  {
    return;
  }

  for ( j = 0; j < m - 2; j++ )
  {
    mj = mj * 2;
    if ( tgle )
    {
      step ( n, mj, &y[0*2+0], &y[(n/2)*2+0], &x[0*2+0], &x[mj*2+0], w);
      tgle = 0;
    }
    else
    {
      step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w);
      tgle = 1;
    }
  }
/* 
  Last pass through data: move Y to X if needed.
*/
  if ( tgle ) 
  {
    ccopy ( n, y, x );
  }

  // mj = n / 2;
  //step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w);

  return;
}
/******************************************************************************/

void cffti ( int n, double w[] )
{
  double arg;
  double aw;
  int i;
  int n2;
  const double pi = 3.141592653589793;

  n2 = n / 2;
  aw = 2.0 * pi / ( ( double ) n );

  # pragma omp parallel \
    shared ( aw, n, w ) \
    private ( arg, i )

  # pragma omp for 

  for ( i = 0; i < n2 - 1; i++ )
  {
    arg = aw * ( ( double ) i );
    w[i*2] = cos ( arg );
    w[i*2+1] = sin ( arg );
  }
  return;
}
/******************************************************************************/

void step ( int n, int mj, double a[], double b[], double c[],
  double d[], double w[] )
{
  double ambr;
  double ambu;
  int j;
  int ja;
  int jb;
  int jc;
  int jd;
  int jw;
  int k;
  int lj;
  int mj2;
  double wjw[2];

  mj2 = 2 * mj;
  lj  = n / mj2;

# pragma omp parallel \
    shared ( a, b, c, d, lj, mj, mj2, w ) \
    private ( ambr, ambu, j, ja, jb, jc, jd, jw, k, wjw )

# pragma omp for

  for ( j = 0; j < lj; j++ )
  {
    jw = j * mj;
    ja  = jw;
    jb  = ja;
    jc  = j * mj2;
    jd  = jc;

    wjw[0] = w[jw*2+0]; 
    wjw[1] = w[jw*2+1];

    for ( k = 0; k < mj; k++ )
    {
      c[(jc+k)*2+0] = a[(ja+k)*2+0] + b[(jb+k)*2+0];
      c[(jc+k)*2+1] = a[(ja+k)*2+1] + b[(jb+k)*2+1];

      ambr = a[(ja+k)*2+0] - b[(jb+k)*2+0];
      ambu = a[(ja+k)*2+1] - b[(jb+k)*2+1];

      d[(jd+k)*2+0] = wjw[0] * ambr - wjw[1] * ambu;
      d[(jd+k)*2+1] = wjw[1] * ambr + wjw[0] * ambu;
    }
  }
  return;
}
