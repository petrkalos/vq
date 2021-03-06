#include <stdio.h>
#include <math.h>
#include <string.h>
#include <memory.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <time.h>
#include <omp.h>
//#define VERBOSE
//#define PROFILE
#define MY_DOUBLE float
#define MY_SHORT short
#define KKZ
#define FAST_NN
#define FAST_NN_THRESHOLD 3
#define SSE2
#ifdef SSE2
#define ALIGNMENT 16
#define ALIGN(x,y) (((x)+(y)-1)/(y))*(y)
#define my_malloc(size,d) _aligned_malloc(size,d)
#define my_free(ptr) _aligned_free(ptr)
#define calc_hyperplane(v1,v2,h,dim) calc_hyperplane_sse2(v1,v2,h,dim)
#define signed_distance(v,h,c0,dim) signed_distance_sse2(v,h,c0,dim)
//#define signed_distance(v,h,c0,dim) signed_distance_sse2_16(v,h,c0,dim)
#define distance2(v1,v2,dim) distance2_sse2(v1,v2,dim)
//#define distance2(v1,v2,dim) distance2_sse2_16(v1,v2,dim)
#define partial_distance2(v1,v2,dim,min_dist) partial_distance2_sse2(v1,v2,dim,min_dist)
#define my_distance2(v1,v2,dim,min_dist) distance2(v1,v2,dim)
#define accumulate_vector(v1,v2,dim) accumulate_vector_sse2(v1,v2,dim)
#define expand_vector(v1,v2,dim) expand_vector_sse2(v1,v2,dim)
#else
#define ALIGNMENT 1
#define ALIGN(x,y) x
#define calc_hyperplane(v1,v2,h,dim) calc_hyperplane_c(v1,v2,h,dim)
#define signed_distance(v,h,c0,dim) signed_distance_c(v,h,c0,dim)
#define my_malloc(size,d) malloc(size)
#define my_free(ptr) free(ptr)
#define distance2(v1,v2,dim) distance2_c(v1,v2,dim)
#define partial_distance2(v1,v2,dim,min_dist) partial_distance2_c(v1,v2,dim,min_dist)
#define my_distance2(v1,v2,dim,min_dist) partial_distance2(v1,v2,dim,min_dist)
#define accumulate_vector(v1,v2,dim) accumulate_vector_c(v1,v2,dim)
#define expand_vector(v1,v2,dim) expand_vector_c(v1,v2,dim)
#endif

#ifdef PROFILE
__int64 count_distance = 0;
__int64 count_product = 0;
__int64 count_sqrt = 0;
double max_ratio = 0.0;
#endif

// Returns pseudo-random number, Gaussian-distributed with given mean and standard deviation
MY_DOUBLE gaussian(MY_DOUBLE mean, MY_DOUBLE std_deviation)
{
	int i;
	int temp;
	int sum;
	MY_DOUBLE dtemp;

	sum = 0;
	for(i=0;i<12;i++)
	{
		temp = rand(); // uniform distribution, [0..32767] => mean = 16383.5, variance = 32767^2/12
		sum += temp;
	}
	// sum: gaussian, mean = 196602, variance = 32767^2
	dtemp = (MY_DOUBLE) ((sum-196602.0)/32767.0);

	return dtemp*std_deviation+mean;
}

// Calculates hyperplane separating 2 vectors. Returns c0 (constant) coefficient of hyperplane
MY_DOUBLE calc_hyperplane_sse2(MY_DOUBLE *vector1, MY_DOUBLE *vector2, MY_DOUBLE *hyperplane, int dim)
{
	int i;
	MY_DOUBLE val;
	MY_DOUBLE *ptr;
	__m128 xmm0,xmm1,xmm2,xmm3,xmm4;

	ptr = hyperplane;
	xmm0 = _mm_setzero_ps();
	xmm1 = _mm_setzero_ps();
	for(i=dim;i>0;i-=4)
	{
		xmm3 = _mm_load_ps ((const float *) (vector2));
		xmm2 = _mm_load_ps ((const float *) (vector1));
		xmm4 = _mm_sub_ps (xmm3, xmm2);
		xmm3 = _mm_mul_ps (xmm3, xmm3);
		_mm_store_ps ((float *) ptr,xmm4);
		xmm2 = _mm_mul_ps (xmm2, xmm2);
		xmm4 = _mm_mul_ps (xmm4, xmm4);
		xmm1 = _mm_add_ps (xmm1, xmm2);
		xmm0 = _mm_add_ps (xmm0, xmm4);
		xmm1 = _mm_sub_ps (xmm1, xmm3);
		vector2 += 4;
		vector1 += 4;
		ptr += 4;
	}
	xmm2 = _mm_shuffle_ps (xmm0, xmm0, 0x4E); // 1 0 3 2
	xmm0 = _mm_add_ps (xmm0, xmm2); // (3+1) (2+0) (3+1) (2+0)
	xmm2 = _mm_shuffle_ps (xmm0, xmm0, 0xB1); // 2 3 0 1
	xmm0 = _mm_add_ps (xmm0, xmm2);
	xmm0 = _mm_rsqrt_ps (xmm0);
	for(i=dim;i>0;i-=4)
	{
		xmm2 = _mm_load_ps ((const float *) (hyperplane));
		xmm2 = _mm_mul_ps (xmm2, xmm0);
		_mm_store_ps ((float *) hyperplane,xmm2);
		hyperplane += 4;
	}
	xmm2 = _mm_shuffle_ps (xmm1, xmm1, 0xEE);
	xmm1 = _mm_add_ps (xmm1, xmm2);
	xmm2 = _mm_shuffle_ps (xmm1, xmm1, 0x55);
	xmm1 = _mm_add_ps (xmm1, xmm2);
	xmm0 = _mm_mul_ps (xmm0, xmm1);
	_mm_store_ss (&val, xmm0);
	return val*(MY_DOUBLE) 0.5;
}

// Returns signed distance between vector and hyperplane
MY_DOUBLE signed_distance_sse2_16(MY_DOUBLE *vector, MY_DOUBLE *hyperplane, MY_DOUBLE c0, int dim)
{
	int i;
	MY_DOUBLE sum;
	__m128 xmm0,xmm1,xmm2,xmm3,xmm4;

	xmm0 =  _mm_load_ss((float *) &c0);
	for(i=dim;i>0;i-=16)
	{
		xmm1 = _mm_load_ps ((const float *) (vector+0));
		xmm2 = _mm_load_ps ((const float *) (hyperplane+0));
		xmm3 = _mm_load_ps ((const float *) (vector+4));
		xmm4 = _mm_load_ps ((const float *) (hyperplane+4));
		xmm1 = _mm_mul_ps (xmm1, xmm2);
		xmm0 = _mm_add_ps (xmm0, xmm1);
		xmm3 = _mm_mul_ps (xmm3, xmm4);
		xmm0 = _mm_add_ps (xmm0, xmm3);
		xmm1 = _mm_load_ps ((const float *) (vector+8));
		xmm2 = _mm_load_ps ((const float *) (hyperplane+8));
		xmm3 = _mm_load_ps ((const float *) (vector+12));
		xmm4 = _mm_load_ps ((const float *) (hyperplane+12));
		xmm1 = _mm_mul_ps (xmm1, xmm2);
		xmm0 = _mm_add_ps (xmm0, xmm1);
		xmm3 = _mm_mul_ps (xmm3, xmm4);
		xmm0 = _mm_add_ps (xmm0, xmm3);
		vector += 16;
		hyperplane += 16;
	}
	xmm1 = _mm_shuffle_ps (xmm0, xmm0, 0xEE);
	xmm0 = _mm_add_ps (xmm0, xmm1);
	xmm1 = _mm_shuffle_ps (xmm0, xmm0, 0x55);
	xmm0 = _mm_add_ps (xmm0, xmm1);
	_mm_store_ss (&sum, xmm0);
	return sum;
}

// Returns signed distance between vector and hyperplane
MY_DOUBLE signed_distance_sse2(MY_DOUBLE *vector, MY_DOUBLE *hyperplane, MY_DOUBLE c0, int dim)
{
	int i;
	MY_DOUBLE sum;
	__m128 xmm0,xmm1,xmm2;

	xmm0 =  _mm_load_ss((float *) &c0);
	for(i=dim;i>0;i-=4)
	{
		xmm1 = _mm_load_ps ((const float *) (vector));
		xmm2 = _mm_load_ps ((const float *) (hyperplane));
		xmm1 = _mm_mul_ps (xmm1, xmm2);
		xmm0 = _mm_add_ps (xmm0, xmm1);
		vector += 4;
		hyperplane += 4;
	}
	xmm1 = _mm_shuffle_ps (xmm0, xmm0, 0xEE);
	xmm0 = _mm_add_ps (xmm0, xmm1);
	xmm1 = _mm_shuffle_ps (xmm0, xmm0, 0x55);
	xmm0 = _mm_add_ps (xmm0, xmm1);
	_mm_store_ss (&sum, xmm0);
	return sum;
}

// Returns L2-distance between two vectors (MY_DOUBLE[])
MY_DOUBLE distance2_sse2_16(MY_DOUBLE *vector1, MY_DOUBLE *vector2, int dim)
{
	int i;
	MY_DOUBLE sum;
	__m128 xmm0,xmm1,xmm2,xmm3,xmm4;

	xmm0 = _mm_setzero_ps();
	for(i=dim;i>0;i-=16)
	{
		xmm1 = _mm_load_ps ((const float *) (vector1+0));
		xmm2 = _mm_load_ps ((const float *) (vector2+0));
		xmm3 = _mm_load_ps ((const float *) (vector1+4));
		xmm4 = _mm_load_ps ((const float *) (vector2+4));
		xmm1 = _mm_sub_ps (xmm1, xmm2);
		xmm3 = _mm_sub_ps (xmm3, xmm4);
		xmm1 = _mm_mul_ps (xmm1, xmm1);
		xmm3 = _mm_mul_ps (xmm3, xmm3);
		xmm0 = _mm_add_ps (xmm0, xmm1);
		xmm0 = _mm_add_ps (xmm0, xmm3);
		xmm1 = _mm_load_ps ((const float *) (vector1+8));
		xmm2 = _mm_load_ps ((const float *) (vector2+8));
		xmm3 = _mm_load_ps ((const float *) (vector1+12));
		xmm4 = _mm_load_ps ((const float *) (vector2+12));
		xmm1 = _mm_sub_ps (xmm1, xmm2);
		xmm3 = _mm_sub_ps (xmm3, xmm4);
		xmm1 = _mm_mul_ps (xmm1, xmm1);
		xmm3 = _mm_mul_ps (xmm3, xmm3);
		xmm0 = _mm_add_ps (xmm0, xmm1);
		xmm0 = _mm_add_ps (xmm0, xmm3);
		vector1 += 16;
		vector2 += 16;
	}
	xmm1 = _mm_shuffle_ps (xmm0, xmm0, 0xEE);
	xmm0 = _mm_add_ps (xmm0, xmm1);
	xmm1 = _mm_shuffle_ps (xmm0, xmm0, 0x55);
	xmm0 = _mm_add_ps (xmm0, xmm1);
	_mm_store_ss (&sum, xmm0);
	return sum;
}

// Returns L2-distance between two vectors (MY_DOUBLE[])
MY_DOUBLE distance2_sse2(MY_DOUBLE *vector1, MY_DOUBLE *vector2, int dim)
{
	int i;
	MY_DOUBLE sum;
	__m128 xmm0,xmm1,xmm2;

	xmm0 = _mm_setzero_ps();
	for(i=dim;i>0;i-=4)
	{
		xmm1 = _mm_load_ps ((const float *) (vector1+0));
		xmm2 = _mm_load_ps ((const float *) (vector2+0));
		xmm1 = _mm_sub_ps (xmm1, xmm2);
		xmm1 = _mm_mul_ps (xmm1, xmm1);
		xmm0 = _mm_add_ps (xmm0, xmm1);
		vector1 += 4;
		vector2 += 4;
	}
	xmm1 = _mm_shuffle_ps (xmm0, xmm0, 0xEE);
	xmm0 = _mm_add_ps (xmm0, xmm1);
	xmm1 = _mm_shuffle_ps (xmm0, xmm0, 0x55);
	xmm0 = _mm_add_ps (xmm0, xmm1);
	_mm_store_ss (&sum, xmm0);
	return sum;
}

// Returns partial L2-distance between two vectors (MY_DOUBLE[]).
// When return value is > min_dist, it is not guaranteed to be the full distance
// since there could be early termination
MY_DOUBLE partial_distance2_sse2(MY_DOUBLE *vector1, MY_DOUBLE *vector2, int dim, MY_DOUBLE min_dist)
{
	int i;
	MY_DOUBLE sum;
	MY_DOUBLE temp;
	__m128 xmm1,xmm2;

	sum = min_dist;
	for(i=dim;i>0 && sum>0.0;i-=4)
	{
		xmm1 = _mm_load_ps ((const float *) (vector1+0));
		xmm2 = _mm_load_ps ((const float *) (vector2+0));
		xmm1 = _mm_sub_ps (xmm1, xmm2);
		xmm1 = _mm_mul_ps (xmm1, xmm1);
		xmm2 = _mm_shuffle_ps (xmm1, xmm1, 0xEE);
		xmm1 = _mm_add_ps (xmm1, xmm2);
		xmm2 = _mm_shuffle_ps (xmm1, xmm1, 0x55);
		xmm1 = _mm_add_ps (xmm1, xmm2);
		_mm_store_ss (&temp, xmm1);
		sum -= temp;
		vector1 += 4;
		vector2 += 4;
	}
	return min_dist-sum;
}

// Accumulates vectors over double precision sum
void accumulate_vector_sse2(double *vector1, MY_DOUBLE *vector2, int dim)
{
	int i;
	__m128d xmm2, xmm3, xmm4, xmm5;
	__m128 xmm1;

	for(i=dim;i>0;i-=4)
	{
		xmm1 = _mm_load_ps ((const float *) (vector2+0));
		xmm2 = _mm_load_pd ((const double *) (vector1+0));
		xmm3 = _mm_load_pd ((const double *) (vector1+2));
		xmm4 = _mm_cvtps_pd (xmm1);
		xmm1 = _mm_shuffle_ps (xmm1, xmm1, 0xEE);
		xmm5 = _mm_cvtps_pd (xmm1);
		xmm2 = _mm_add_pd (xmm2, xmm4);
		xmm3 = _mm_add_pd (xmm3, xmm5);
		_mm_store_pd ((double *) (vector1+0), xmm2);
		_mm_store_pd ((double *) (vector1+2), xmm3);
		vector1 += 4;
		vector2 += 4;
	}
}

// Returns signed distance between vector and hyperplane
MY_DOUBLE signed_distance_short_float_sse2_16(MY_SHORT *vector, MY_DOUBLE *hyperplane, MY_DOUBLE c0, int dim)
{
	int i;
	MY_DOUBLE sum;
	__m128i xmm1,xmm3,xmm5,xmm7;
	__m128 xmm0,xmm2,xmm4,xmm6;

	xmm0 =  _mm_load_ss((float *) &c0);
	xmm1 = _mm_setzero_si128();
	for(i=dim;i>0;i-=16)
	{
		xmm3 = _mm_load_si128 ((__m128i *) (vector+0));
		xmm2 = _mm_load_ps ((const float *) (hyperplane+0));
		xmm4 = _mm_load_ps ((const float *) (hyperplane+4));
		xmm5 = _mm_cmplt_epi16 (xmm3, xmm1);
		xmm7 = _mm_unpacklo_epi16 (xmm3, xmm5);
		xmm5 = _mm_unpackhi_epi16 (xmm3, xmm5);
		xmm6 = _mm_cvtepi32_ps (xmm7);
		xmm2 = _mm_mul_ps (xmm2, xmm6);
		xmm0 = _mm_add_ps (xmm0, xmm2);
		xmm6 = _mm_cvtepi32_ps (xmm5);
		xmm2 = _mm_mul_ps (xmm2, xmm6);
		xmm0 = _mm_add_ps (xmm0, xmm2);
		xmm3 = _mm_load_si128 ((__m128i *) (vector+8));
		xmm2 = _mm_load_ps ((const float *) (hyperplane+8));
		xmm4 = _mm_load_ps ((const float *) (hyperplane+12));
		xmm5 = _mm_cmplt_epi16 (xmm3, xmm1);
		xmm7 = _mm_unpacklo_epi16 (xmm3, xmm5);
		xmm5 = _mm_unpackhi_epi16 (xmm3, xmm5);
		xmm6 = _mm_cvtepi32_ps (xmm7);
		xmm2 = _mm_mul_ps (xmm2, xmm6);
		xmm0 = _mm_add_ps (xmm0, xmm2);
		xmm6 = _mm_cvtepi32_ps (xmm5);
		xmm2 = _mm_mul_ps (xmm2, xmm6);
		xmm0 = _mm_add_ps (xmm0, xmm2);
		vector += 16;
		hyperplane += 16;
	}
	xmm2 = _mm_shuffle_ps (xmm0, xmm0, 0xEE);
	xmm0 = _mm_add_ps (xmm0, xmm2);
	xmm2 = _mm_shuffle_ps (xmm0, xmm0, 0x55);
	xmm0 = _mm_add_ps (xmm0, xmm2);
	_mm_store_ss (&sum, xmm0);
	return sum;
}

// Returns signed distance between vector and hyperplane
MY_DOUBLE signed_distance_short_float_sse2(MY_SHORT *vector, MY_DOUBLE *hyperplane, MY_DOUBLE c0, int dim)
{
	int i;
	MY_DOUBLE sum;
	__m128i xmm1,xmm3,xmm5,xmm7;
	__m128 xmm0,xmm2,xmm4,xmm6;

	xmm0 =  _mm_load_ss((float *) &c0);
	xmm1 = _mm_setzero_si128();
	for(i=dim;i>0;i-=8)
	{
		xmm3 = _mm_load_si128 ((__m128i *) (vector+0));
		xmm2 = _mm_load_ps ((const float *) (hyperplane+0));
		xmm4 = _mm_load_ps ((const float *) (hyperplane+4));
		xmm5 = _mm_cmplt_epi16 (xmm3, xmm1);
		xmm7 = _mm_unpacklo_epi16 (xmm3, xmm5);
		xmm5 = _mm_unpackhi_epi16 (xmm3, xmm5);
		xmm6 = _mm_cvtepi32_ps (xmm7);
		xmm2 = _mm_mul_ps (xmm2, xmm6);
		xmm0 = _mm_add_ps (xmm0, xmm2);
		xmm6 = _mm_cvtepi32_ps (xmm5);
		xmm2 = _mm_mul_ps (xmm2, xmm6);
		xmm0 = _mm_add_ps (xmm0, xmm2);
		vector += 4;
		hyperplane += 4;
	}
	xmm2 = _mm_shuffle_ps (xmm0, xmm0, 0xEE);
	xmm0 = _mm_add_ps (xmm0, xmm2);
	xmm2 = _mm_shuffle_ps (xmm0, xmm0, 0x55);
	xmm0 = _mm_add_ps (xmm0, xmm2);
	_mm_store_ss (&sum, xmm0);
	return sum;
}

// Returns L2-distance between two vectors (short[])
MY_DOUBLE distance2_float_short_sse2_16(MY_DOUBLE *vector1, MY_SHORT *vector2, int dim)
{
	int i;
	MY_DOUBLE sum;
	__m128i xmm1,xmm3,xmm5,xmm7;
	__m128 xmm0,xmm2,xmm4,xmm6;

	xmm0 = _mm_setzero_ps();
	xmm1 = _mm_setzero_si128();
	for(i=dim;i>0;i-=16)
	{
		xmm3 = _mm_load_si128 ((__m128i *) (vector2+0));
		xmm2 = _mm_load_ps ((const float *) (vector1+0));
		xmm4 = _mm_load_ps ((const float *) (vector1+4));
		xmm5 = _mm_cmplt_epi16 (xmm3, xmm1);
		xmm7 = _mm_unpacklo_epi16 (xmm3, xmm5);
		xmm5 = _mm_unpackhi_epi16 (xmm3, xmm5);
		xmm6 = _mm_cvtepi32_ps (xmm7);
		xmm2 = _mm_sub_ps (xmm2, xmm6);
		xmm6 = _mm_cvtepi32_ps (xmm5);
		xmm4 = _mm_sub_ps (xmm4, xmm6);
		xmm2 = _mm_mul_ps (xmm2, xmm2);
		xmm4 = _mm_mul_ps (xmm4, xmm4);
		xmm0 = _mm_add_ps (xmm0, xmm2);
		xmm0 = _mm_add_ps (xmm0, xmm4);
		xmm3 = _mm_load_si128 ((__m128i *) (vector2+8));
		xmm2 = _mm_load_ps ((const float *) (vector1+8));
		xmm4 = _mm_load_ps ((const float *) (vector1+12));
		xmm5 = _mm_cmplt_epi16 (xmm3, xmm1);
		xmm7 = _mm_unpacklo_epi16 (xmm3, xmm5);
		xmm5 = _mm_unpackhi_epi16 (xmm3, xmm5);
		xmm6 = _mm_cvtepi32_ps (xmm7);
		xmm2 = _mm_sub_ps (xmm2, xmm6);
		xmm6 = _mm_cvtepi32_ps (xmm5);
		xmm4 = _mm_sub_ps (xmm4, xmm6);
		xmm2 = _mm_mul_ps (xmm2, xmm2);
		xmm4 = _mm_mul_ps (xmm4, xmm4);
		xmm0 = _mm_add_ps (xmm0, xmm2);
		xmm0 = _mm_add_ps (xmm0, xmm4);
		vector1 += 16;
		vector2 += 16;
	}
	xmm2 = _mm_shuffle_ps (xmm0, xmm0, 0xEE);
	xmm0 = _mm_add_ps (xmm0, xmm2);
	xmm2 = _mm_shuffle_ps (xmm0, xmm0, 0x55);
	xmm0 = _mm_add_ps (xmm0, xmm2);
	_mm_store_ss (&sum, xmm0);
	return sum;
}

// Returns L2-distance between two vectors (short[])
MY_DOUBLE distance2_float_short_sse2(MY_DOUBLE *vector1, MY_SHORT *vector2, int dim)
{
	int i;
	MY_DOUBLE sum;
	__m128i xmm1,xmm3,xmm5,xmm7;
	__m128 xmm0,xmm2,xmm4,xmm6;

	xmm0 = _mm_setzero_ps();
	xmm1 = _mm_setzero_si128();
	for(i=dim;i>0;i-=8)
	{
		xmm3 = _mm_load_si128 ((__m128i *) (vector2+0));
		xmm2 = _mm_load_ps ((const float *) (vector1+0));
		xmm4 = _mm_load_ps ((const float *) (vector1+4));
		xmm5 = _mm_cmplt_epi16 (xmm3, xmm1);
		xmm7 = _mm_unpacklo_epi16 (xmm3, xmm5);
		xmm5 = _mm_unpackhi_epi16 (xmm3, xmm5);
		xmm6 = _mm_cvtepi32_ps (xmm7);
		xmm2 = _mm_sub_ps (xmm2, xmm6);
		xmm6 = _mm_cvtepi32_ps (xmm5);
		xmm4 = _mm_sub_ps (xmm4, xmm6);
		xmm2 = _mm_mul_ps (xmm2, xmm2);
		xmm4 = _mm_mul_ps (xmm4, xmm4);
		xmm0 = _mm_add_ps (xmm0, xmm2);
		xmm0 = _mm_add_ps (xmm0, xmm4);
		vector1 += 8;
		vector2 += 8;
	}
	xmm2 = _mm_shuffle_ps (xmm0, xmm0, 0xEE);
	xmm0 = _mm_add_ps (xmm0, xmm2);
	xmm2 = _mm_shuffle_ps (xmm0, xmm0, 0x55);
	xmm0 = _mm_add_ps (xmm0, xmm2);
	_mm_store_ss (&sum, xmm0);
	return sum;
}

// Returns partial L2-distance between two vectors (MY_DOUBLE[]).
// When return value is > min_dist, it is not guaranteed to be the full distance
// since there could be early termination
MY_DOUBLE partial_distance2_float_short_sse2(MY_DOUBLE *vector1, MY_SHORT *vector2, int dim, MY_DOUBLE min_dist)
{
	int i;
	MY_DOUBLE sum;
	MY_DOUBLE temp;
	__m128i xmm1,xmm3,xmm5,xmm7;
	__m128 xmm0,xmm2,xmm4,xmm6;

	xmm1 = _mm_setzero_si128();
	sum = min_dist;
	for(i=dim;i>0 && sum>0.0;i-=8)
	{
		xmm3 = _mm_load_si128 ((__m128i *) (vector2+0));
		xmm2 = _mm_load_ps ((const float *) (vector1+0));
		xmm4 = _mm_load_ps ((const float *) (vector1+4));
		xmm5 = _mm_cmplt_epi16 (xmm3, xmm1);
		xmm7 = _mm_unpacklo_epi16 (xmm3, xmm5);
		xmm5 = _mm_unpackhi_epi16 (xmm3, xmm5);
		xmm6 = _mm_cvtepi32_ps (xmm7);
		xmm2 = _mm_sub_ps (xmm2, xmm6);
		xmm6 = _mm_cvtepi32_ps (xmm5);
		xmm4 = _mm_sub_ps (xmm4, xmm6);
		xmm2 = _mm_mul_ps (xmm2, xmm2);
		xmm4 = _mm_mul_ps (xmm4, xmm4);
		xmm2 = _mm_add_ps (xmm2, xmm4);
		xmm0 = _mm_shuffle_ps (xmm2, xmm2, 0xEE);
		xmm2 = _mm_add_ps (xmm2, xmm0);
		xmm0 = _mm_shuffle_ps (xmm2, xmm2, 0x55);
		xmm2 = _mm_add_ps (xmm2, xmm0);
		_mm_store_ss (&temp, xmm2);
		vector1 += 8;
		vector2 += 8;
		sum -= temp;
	}
	return min_dist-sum;
}

// 
void accumulate_vector_int64_short_sse2(__int64 *vector1, MY_SHORT *vector2, int dim)
{
	int i;
	__m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

	for(i=dim;i>0;i-=4)
	{
		xmm0 = _mm_setzero_si128 ();
		xmm1 = _mm_load_si128 ((__m128i *) (vector2+0));
		xmm2 = _mm_load_si128 ((__m128i *) (vector1+0));
		xmm3 = _mm_load_si128 ((__m128i *) (vector1+2));
		xmm0 = _mm_cmplt_epi16 (xmm1, xmm0);
		xmm6 = xmm1;
		xmm6 = _mm_unpacklo_epi16 (xmm0, xmm6);
		xmm1 = _mm_unpackhi_epi16 (xmm0, xmm1);
		xmm7 = xmm0;
		xmm7 = _mm_unpacklo_epi16 (xmm7, xmm7);
		xmm0 = _mm_unpackhi_epi16 (xmm0, xmm0);
		xmm4 = xmm6;
		xmm4 = _mm_unpacklo_epi32 (xmm7, xmm4);
		xmm6 = _mm_unpackhi_epi32 (xmm7, xmm6);
		xmm2 = _mm_add_epi64 (xmm2, xmm4);
		xmm3 = _mm_add_epi64 (xmm3, xmm6);
		xmm4 = _mm_load_si128 ((__m128i *) (vector1+4));
		xmm5 = _mm_load_si128 ((__m128i *) (vector1+6));
		xmm6 = xmm1;
		xmm6 = _mm_unpacklo_epi32 (xmm0, xmm6);
		xmm1 = _mm_unpackhi_epi32 (xmm0, xmm1);
		xmm4 = _mm_add_epi64 (xmm4, xmm6);
		xmm5 = _mm_add_epi64 (xmm5, xmm1);
		_mm_store_si128 ((__m128i *) (vector1+0), xmm2);
		_mm_store_si128 ((__m128i *) (vector1+2), xmm3);
		_mm_store_si128 ((__m128i *) (vector1+4), xmm4);
		_mm_store_si128 ((__m128i *) (vector1+6), xmm5);
		vector1 += 4;
		vector2 += 4;
	}
}

void expand_vector_sse2(MY_SHORT *input, MY_DOUBLE *output, int dim)
{
	int i;
	__m128i xmm1,xmm3,xmm5,xmm7;
	__m128 xmm0,xmm2;

	xmm1 = _mm_setzero_si128();
	for(i=dim;i>0;i-=8)
	{
		xmm3 = _mm_load_si128 ((__m128i *) (input+0));
		xmm5 = _mm_cmplt_epi16 (xmm3, xmm1);
		xmm7 = _mm_unpacklo_epi16 (xmm3, xmm5);
		xmm5 = _mm_unpackhi_epi16 (xmm3, xmm5);
		xmm0 = _mm_cvtepi32_ps (xmm7);
		xmm2 = _mm_cvtepi32_ps (xmm5);
		_mm_store_ps ((float *) (output+0), xmm0);
		_mm_store_ps ((float *) (output+4), xmm2);
		input += 8;
		output += 8;
	}
}

// Calculates hyperplane separating 2 vectors. Returns c0 (constant) coefficient of hyperplane
MY_DOUBLE calc_hyperplane_c(MY_DOUBLE *vector1, MY_DOUBLE *vector2, MY_DOUBLE *hyperplane, int dim)
{
	int i;
	MY_DOUBLE diff;
	double sum;
	double sum2;
	MY_DOUBLE val;

	sum = 0.0;
	sum2 = 0.0;
	for(i=0;i<dim;i++)
	{
		diff = vector2[i]-vector1[i];
		hyperplane[i] = (MY_DOUBLE) diff;
		sum += diff*diff;
		sum2 += (vector1[i])*(vector1[i]) - (vector2[i])*(vector2[i]);
	}
	if(sum!=0.0)
	{
		double d;
		d = 1.0/sqrt((double) sum);
		for(i=0;i<dim;i++)
		{
			hyperplane[i] = (MY_DOUBLE) (hyperplane[i]*d);
		}
		val = (MY_DOUBLE) (0.5*d*((double) sum2));
	}
	else
	{
		val = 0.0;
	}
	return val;
}

// Returns signed distance between vector and hyperplane
MY_DOUBLE signed_distance_c(MY_DOUBLE *vector, MY_DOUBLE *hyperplane, MY_DOUBLE c0, int dim)
{
	int i;
	double sum;

	sum = c0;
	for(i=0;i<dim;i++)
	{
		sum += vector[i]*hyperplane[i];
	}
	return (MY_DOUBLE) sum;
}

// Returns L2-distance between two vectors (MY_DOUBLE[])
MY_DOUBLE distance2_c(MY_DOUBLE *vector1, MY_DOUBLE *vector2, int dim)
{
	int i;
	double sum;
	MY_DOUBLE diff;

	sum = 0.0;
	for(i=0;i<dim;i++)
	{
		diff = vector1[i] - vector2[i];
		sum += diff*diff;
	}
	return (MY_DOUBLE) sum;
}

// Returns partial L2-distance between two vectors (MY_DOUBLE[]).
// When return value is > min_dist, it is not guaranteed to be the full distance
// since there could be early termination
MY_DOUBLE partial_distance2_c(MY_DOUBLE *vector1, MY_DOUBLE *vector2, int dim, MY_DOUBLE min_dist)
{
	int i;
	double sum;
	MY_DOUBLE diff;

	sum = (double) min_dist;
	for(i=0;i<dim && sum>0;i++)
	{
		diff = vector1[i] - vector2[i];
		sum -= diff*diff;
	}
	return (MY_DOUBLE) (min_dist-sum);
}

// Accumulates vectors over double precision sum
void accumulate_vector_c(double *vector1, MY_DOUBLE *vector2, int dim)
{
	int i;

	for(i=0;i<dim;i++)
	{
		vector1[i] += vector2[i];
	}
}

// Returns signed distance between vector and hyperplane
MY_DOUBLE signed_distance_short_float_c(MY_SHORT *vector, MY_DOUBLE *hyperplane, MY_DOUBLE c0, int dim)
{
	int i;
	double sum;

	sum = c0;
	for(i=0;i<dim;i++)
	{
		sum += vector[i]*hyperplane[i];
	}
	return (MY_DOUBLE) sum;
}

// Returns L2-distance between two vectors (MY_DOUBLE[])
MY_DOUBLE distance2_float_short_c(MY_DOUBLE *vector1, MY_SHORT *vector2, int dim)
{
	int i;
	double sum;
	MY_DOUBLE diff;

	sum = 0.0;
	for(i=0;i<dim;i++)
	{
		diff = vector1[i] - vector2[i];
		sum += diff*diff;
	}
	return (MY_DOUBLE) sum;
}

// Returns partial L2-distance between two vectors (MY_DOUBLE[]).
// When return value is > min_dist, it is not guaranteed to be the full distance
// since there could be early termination
MY_DOUBLE partial_distance2_float_short_c(MY_DOUBLE *vector1, short *vector2, int dim, MY_DOUBLE min_dist)
{
	int i;
	double sum;
	MY_DOUBLE diff;

	sum = (double) min_dist;
	for(i=0;i<dim && sum>0;i++)
	{
		diff = vector1[i] - vector2[i];
		sum -= diff*diff;
	}
	return (MY_DOUBLE) (min_dist-sum);
}

// 
void accumulate_vector_int64_short_c(__int64 *vector1, MY_SHORT *vector2, int dim)
{
	int i;

	for(i=0;i<dim;i++)
	{
		vector1[i] += vector2[i];
	}
}

void expand_vector_c(MY_SHORT *input, MY_DOUBLE *output, int dim)
{
	int i;

	for(i=0;i<dim;i++)
	{
		output[i] = (MY_DOUBLE) input[i];
	}
}

double entropy(int *count,int size)
{
	int i;
	int sum;
	double h;
	double val;

	sum = 0;
	for(i=0;i<size;i++)
	{
		if(count[i]>0)
		{
			sum += count[i];
		}
	}
	h = 0.0;
	if(sum>0)
	{
		for(i=0;i<size;i++)
		{
			if(count[i]>0)
			{
				val = ((double) count[i])/((double) sum);
				h -= val*log(val);
			}
		}
		h /= log(2.0);
	}
	return h;
}

double kmeans_initialize(MY_DOUBLE *clusters, MY_SHORT *training, int *indices, MY_DOUBLE *distances, int *cluster_count, double *new_clusters, 
						 int dim, int num_of_clusters, int num_of_vectors, int *diff_count)
{
	int i, j;
	MY_SHORT *p1;
	MY_DOUBLE *p2;
	MY_DOUBLE dist;
	MY_DOUBLE min_dist;
	MY_DOUBLE max_dist;
	int max_ind;
	double total_sum;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));
	MY_DOUBLE *local_vector;

	*diff_count = num_of_vectors;
	if(num_of_vectors==0 || num_of_vectors<num_of_clusters)
	{
		return -1.0;
	}
	local_vector = (MY_DOUBLE *) my_malloc(dim2*sizeof(MY_DOUBLE),ALIGNMENT);
	if(local_vector==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			dim2*sizeof(MY_DOUBLE), "local_vector");
		exit(-1);
	}
	memset(new_clusters,0,dim2*sizeof(double));
	memset(cluster_count,0,num_of_clusters*sizeof(int));
	memset(indices,0,num_of_vectors*sizeof(int));
	for(i=0;i<num_of_vectors;i++)
	{
		p1 = &training[i*dim2];
		expand_vector(p1,local_vector,dim);
		accumulate_vector(new_clusters,local_vector,dim);
	}
	cluster_count[0] = num_of_vectors;
	for(i=0;i<dim;i++)
	{
		clusters[i] = (MY_DOUBLE) new_clusters[i]/(MY_DOUBLE) num_of_vectors;
	}
	for(;i<dim2;i++)
	{
		clusters[i] = (MY_DOUBLE) 0;
	}
	j = -1;
first_centroid:
	total_sum = 0.0;
	max_dist = -1.0;
	for(i=0;i<num_of_vectors;i++)
	{
		p1 = &training[i*dim2];
		expand_vector(p1,local_vector,dim);
		min_dist = distance2(local_vector,clusters,dim);
		total_sum += min_dist;
		distances[i] = min_dist;
		if(min_dist>max_dist)
		{
			max_dist = min_dist;
			max_ind = i;
		}
	}
	if(num_of_clusters==1)
	{
		my_free(local_vector);
		return ((double) total_sum)/((double) num_of_vectors);
	}
	j++;
	if(j==0)
	{
		p1 = &training[max_ind*dim2];
		for(i=0;i<dim;i++)
		{
			clusters[i] = (MY_DOUBLE) p1[i];
		}
		goto first_centroid;
	}
	for(;j<num_of_clusters;j++)
	{
		p2 = &clusters[j*dim2];
		p1 = &training[max_ind*dim2];
		for(i=0;i<dim;i++)
		{
			p2[i] = (MY_DOUBLE) p1[i];
		}
		for(;i<dim2;i++)
		{
			p2[i] = (MY_DOUBLE) 0;
		}
		max_dist = -1.0;
		total_sum = 0.0;
		for(i=0;i<num_of_vectors;i++)
		{
			p1 = &training[i*dim2];
			expand_vector(p1,local_vector,dim);
			min_dist = distances[i];
			dist = my_distance2(local_vector,p2,dim,min_dist);
			if(dist<min_dist)
			{
				min_dist = dist;
				distances[i] = dist;
				indices[i] = j;
			}
			total_sum += min_dist;
			if(min_dist>max_dist)
			{
				max_dist = min_dist;
				max_ind = i;
			}
		}
	}
	my_free(local_vector);
	return ((double) total_sum)/((double) num_of_vectors);
}

void kmeans_cluster(MY_DOUBLE *clusters, MY_SHORT *training, int *indices, MY_DOUBLE *distances, int *cluster_count, double *new_clusters, 
					int dim, int num_of_clusters, int num_of_vectors, int *diff_count)
{
	int i, j;
	MY_DOUBLE *p1;
	int min_ind;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));
	MY_DOUBLE *local_vector;

	if(num_of_vectors==0 || num_of_vectors<num_of_clusters)
	{
		return;
	}
	local_vector = (MY_DOUBLE *) my_malloc(dim2*sizeof(MY_DOUBLE),ALIGNMENT);
	if(local_vector==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			dim2*sizeof(MY_DOUBLE), "local_vector");
		exit(-1);
	}
	memset(new_clusters,0,num_of_clusters*dim2*sizeof(double));
	memset(cluster_count,0,num_of_clusters*sizeof(int));
	for(i=0;i<num_of_vectors;i++)
	{
		min_ind = indices[i];
		expand_vector(&training[i*dim2],local_vector,dim);
		accumulate_vector(&new_clusters[min_ind*dim2],local_vector,dim);
		cluster_count[min_ind]++;
	}
	for(j=0;j<num_of_clusters;j++)
	{
		if((min_ind=cluster_count[j])>0)
		{
			double *p2;
			p1 = &clusters[j*dim2];
			p2 = &new_clusters[j*dim2];
			for(i=0;i<dim;i++)
			{
				p1[i] = (MY_DOUBLE) p2[i]/(MY_DOUBLE) min_ind;
			}
		}
	}
	my_free(local_vector);
}

double kmeans_iterate(MY_DOUBLE *clusters, MY_SHORT *training, int *indices, MY_DOUBLE *distances, int *cluster_count, double *new_clusters, 
					  int dim, int num_of_clusters, int num_of_vectors, int *diff_count)
{
	int i, j;
	MY_SHORT *p1;
	MY_DOUBLE dist;
	MY_DOUBLE min_dist;
	int min_ind;
	double total_sum;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));
	MY_DOUBLE *local_vector;

	if(num_of_vectors==0 || num_of_vectors<num_of_clusters)
	{
		*diff_count = -1;
		return -1.0;
	}
	local_vector = (MY_DOUBLE *) my_malloc(dim2*sizeof(MY_DOUBLE),ALIGNMENT);
	if(local_vector==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			dim2*sizeof(MY_DOUBLE), "local_vector");
		exit(-1);
	}
	memset(new_clusters,0,num_of_clusters*dim2*sizeof(double));
	memset(cluster_count,0,num_of_clusters*sizeof(int));
	*diff_count = 0;
	total_sum = 0.0;
#ifdef VERBOSE
	printf("        ");
#endif
	for(i=0;i<num_of_vectors;i++)
	{
#ifdef VERBOSE
		if((i%10000)==0)
		{
			printf("\b\b\b\b\b\b\b\b%8d",i);
		}
#endif
		p1 = &training[i*dim2];
		expand_vector(p1,local_vector,dim);
		min_ind = indices[i];
		min_dist = distance2(local_vector,&clusters[min_ind*dim2],dim);
		for(j=0;j<num_of_clusters;j++)
		{
			dist = my_distance2(local_vector,&clusters[j*dim2],dim,min_dist);
			if(dist<min_dist)
			{
				min_dist = dist;
				min_ind = j;
			}
		}
		if(indices[i] != min_ind)
		{
			(*diff_count)++;
			indices[i] = min_ind;
		}
		accumulate_vector(&new_clusters[min_ind*dim2],local_vector,dim);
		cluster_count[min_ind]++;
		distances[i] = min_dist;
		total_sum += min_dist;
	}
	for(j=0;j<num_of_clusters;j++)
	{
		if((min_ind=cluster_count[j])>0)
		{
			double *p2;
			MY_DOUBLE *p3;
			p3 = &clusters[j*dim2];
			p2 = &new_clusters[j*dim2];
			for(i=0;i<dim;i++)
			{
				p3[i] = (MY_DOUBLE) p2[i]/(MY_DOUBLE) min_ind;
			}
		}
	}
	my_free(local_vector);
	return ((double) total_sum)/((double) (num_of_vectors*dim));
}

double kmeans_iterate2(MY_DOUBLE *clusters, MY_SHORT *training, int *indices, MY_DOUBLE *distances, int *cluster_count, double *new_clusters, 
					   int dim, int num_of_vectors, int *diff_count)
{
	int i, j;
	MY_SHORT *p1;
	MY_DOUBLE dist;
	MY_DOUBLE min_dist;
	int min_ind;
	double total_sum;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));
	MY_DOUBLE *local_vector;

	if(num_of_vectors<2)
	{
		*diff_count = -1;
		return -1.0;
	}
	local_vector = (MY_DOUBLE *) my_malloc(dim2*sizeof(MY_DOUBLE),ALIGNMENT);
	if(local_vector==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			dim2*sizeof(MY_DOUBLE), "local_vector");
		exit(-1);
	}
	memset(new_clusters,0,2*dim2*sizeof(double));
	memset(cluster_count,0,2*sizeof(int));
	*diff_count = 0;
	total_sum = 0.0;
	for(i=0;i<num_of_vectors;i++)
	{
		p1 = &training[i*dim2];
		expand_vector(p1,local_vector,dim);
		min_ind = indices[i];
		min_dist = distance2(local_vector,&clusters[min_ind*dim2],dim);
		dist = my_distance2(local_vector,&clusters[(1-min_ind)*dim2],dim,min_dist);
		if(dist<min_dist)
		{
			min_dist = dist;
			min_ind = 1-min_ind;
			(*diff_count)++;
			indices[i] = min_ind;
		}
		accumulate_vector(&new_clusters[min_ind*dim2],local_vector,dim);
		cluster_count[min_ind]++;
		distances[i] = min_dist;
		total_sum += min_dist;
	}
	for(j=0;j<2;j++)
	{
		if((min_ind=cluster_count[j])>0)
		{
			double *p2;
			MY_DOUBLE *p3;
			p3 = &clusters[j*dim2];
			p2 = &new_clusters[j*dim2];
			for(i=0;i<dim;i++)
			{
				p3[i] = (MY_DOUBLE) p2[i]/(MY_DOUBLE) min_ind;
			}
		}
	}
	my_free(local_vector);
	return ((double) total_sum)/((double) (num_of_vectors*dim));
}

double kmeans_iterate2_h(MY_DOUBLE *clusters, MY_SHORT *training, int *indices, MY_DOUBLE *distances, int *cluster_count, double *new_clusters, 
						 int dim, MY_DOUBLE *hyperplane, int num_of_vectors, int *diff_count)
{
	int i, j;
	MY_SHORT *p1;
	MY_DOUBLE min_dist;
	int min_ind;
	double total_sum;
	MY_DOUBLE c0;
	MY_DOUBLE val;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));
	MY_DOUBLE *local_vector;

	if(num_of_vectors<2)
	{
		*diff_count = -1;
		return -1.0;
	}
	local_vector = (MY_DOUBLE *) my_malloc(dim2*sizeof(MY_DOUBLE),ALIGNMENT);
	if(local_vector==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			dim2*sizeof(MY_DOUBLE), "local_vector");
		exit(-1);
	}
	memset(new_clusters,0,2*dim2*sizeof(double));
	memset(cluster_count,0,2*sizeof(int));
	*diff_count = 0;
	total_sum = 0.0;
	c0 = calc_hyperplane(&clusters[0*dim2],&clusters[1*dim2],hyperplane,dim);
	for(i=0;i<num_of_vectors;i++)
	{
		p1 = &training[i*dim2];
		expand_vector(p1,local_vector,dim);
		val = signed_distance(local_vector,hyperplane,c0,dim);
		if(val<0.0)
		{
			min_dist = distance2(local_vector,&clusters[0*dim2],dim);
			min_ind = 0;
		}
		else
		{
			min_dist = distance2(local_vector,&clusters[1*dim2],dim);
			min_ind = 1;
		}
		if(indices[i] != min_ind)
		{
			(*diff_count)++;
			indices[i] = min_ind;
		}
		accumulate_vector(&new_clusters[min_ind*dim2],local_vector,dim);
		cluster_count[min_ind]++;
		distances[i] = min_dist;
		total_sum += min_dist;
	}
	for(j=0;j<2;j++)
	{
		if((min_ind=cluster_count[j])>0)
		{
			double *p2;
			MY_DOUBLE *p3;
			p3 = &clusters[j*dim2];
			p2 = &new_clusters[j*dim2];
			for(i=0;i<dim;i++)
			{
				p3[i] = (MY_DOUBLE) p2[i]/(MY_DOUBLE) min_ind;
			}
		}
	}
	my_free(local_vector);
	return ((double) total_sum)/((double) (num_of_vectors*dim));
}

void randomize(MY_SHORT *vectors, int dim, int num_of_vectors)
{
	int i;
	int j;
	MY_SHORT *p1;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));

	for(i=0;i<num_of_vectors;i++)
	{
		p1 = &vectors[i*dim2];
		for(j=0;j<dim;j++)
		{
			p1[j] = (MY_SHORT) ((rand()&0x1FF)-256);
			//p1[j] = gaussian(0.0,25.0);
		}
		for(;j<dim2;j++)
		{
			p1[j] = (MY_SHORT) 0;
		}
	}
}

void initialize(MY_DOUBLE *clusters, MY_SHORT *training, int dim, int num_of_clusters, int num_of_vectors)
{
	int i,j;
	MY_SHORT *p1;
	MY_DOUBLE *p2;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));

	for(j=0;j<num_of_clusters;j++)
	{
		p1 = &training[j*dim2];
		p2 = &clusters[j*dim2];
		for(i=0;i<dim;i++)
		{
			p2[i] = (MY_DOUBLE) p1[i];
		}
	}
}

#ifdef FAST_NN
struct paired {
	int index;
	MY_DOUBLE signed_distance;
};

struct node
{
	struct node *left;
	struct node *right;
	MY_DOUBLE c0;
	MY_DOUBLE *hyperplane;
	int count;
	struct paired *pairs;
};

struct context {
	MY_DOUBLE *clusters2;
	int *indices2;
	int *cluster_count2;
	MY_DOUBLE *distances2;
	double *new_clusters2;
	int diff_count2;
};

struct node *allocate_node(int num_of_clusters,int dim)
{
	struct node *root = NULL;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));

	if(dim>0 && num_of_clusters>=1)
	{
		root = (struct node *) my_malloc(1*sizeof(struct node),ALIGNMENT);
		if(root==NULL)
		{
			printf("Not enough memory (%d bytes) for %s - exiting\n",
				1*sizeof(struct node), "root");
			exit(-1);
		}
		root->pairs = (struct paired *) my_malloc(num_of_clusters*sizeof(struct paired),ALIGNMENT);
		if(root->pairs==NULL)
		{
			printf("Not enough memory (%d bytes) for %s - exiting\n",
				num_of_clusters*sizeof(struct paired), "root->pairs");
			exit(-1);
		}
		if(num_of_clusters>FAST_NN_THRESHOLD)
		{
			root->hyperplane = (MY_DOUBLE *) my_malloc(dim2*sizeof(MY_DOUBLE),ALIGNMENT);
			if(root->hyperplane==NULL)
			{
				printf("Not enough memory (%d bytes) for %s - exiting\n",
					dim2*sizeof(MY_DOUBLE), "root->hyperplane");
				exit(-1);
			}
		}
		else
			root->hyperplane = NULL;
		root->left = root->right = NULL;
	}
	return root;
}

void free_node(struct node *root)
{
	if(root!=NULL)
	{
		if(root->pairs!=NULL)
			my_free(root->pairs);
		if(root->hyperplane!=NULL)
			my_free(root->hyperplane);
		my_free(root);
	}
}

void free_tree(struct node *root)
{
	if(root!=NULL)
	{
		free_tree(root->left);
		free_tree(root->right);
		free_node(root);
	}
}

double kmeans_initialize2_map(struct context *storage, MY_DOUBLE *training, int dim, int num_of_vectors, struct paired *map)
{
	MY_DOUBLE *clusters = storage->clusters2;
	int *indices = storage->indices2;
	MY_DOUBLE *distances = storage->distances2;
	int *cluster_count = storage->cluster_count2;
	double *new_clusters = storage->new_clusters2;
	int *diff_count = &storage->diff_count2;
	int i;
	MY_DOUBLE *p1;
	MY_DOUBLE *p2;
	MY_DOUBLE dist;
	MY_DOUBLE min_dist;
	MY_DOUBLE max_dist;
	int max_ind;
	double total_sum;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));

	*diff_count = num_of_vectors;
	if(num_of_vectors<2)
	{
		return -1.0;
	}
	memset(new_clusters,0,dim2*sizeof(double));
	memset(cluster_count,0,2*sizeof(int));
	memset(indices,0,num_of_vectors*sizeof(int));
	for(i=0;i<num_of_vectors;i++)
	{
		p1 = &training[map[i].index*dim2];
		accumulate_vector(new_clusters,p1,dim);
	}
	cluster_count[0] = num_of_vectors;
	#pragma loop(ivdep)
	for(i=0;i<dim;i++)
	{
		clusters[i] = (MY_DOUBLE) new_clusters[i]/(MY_DOUBLE) num_of_vectors;
	}
	for(;i<dim2;i++)
	{
		clusters[i] = (MY_DOUBLE) 0;
	}
	// clusters[0] is the overall centroid
	max_dist = -1.0;
	for(i=0;i<num_of_vectors;i++)
	{
		min_dist = distance2(&training[map[i].index*dim2],clusters,dim);
		if(min_dist>max_dist)
		{
			max_dist = min_dist;
			max_ind = i;
		}
	}
	p1 = &training[map[max_ind].index*dim2];
	for(i=0;i<dim;i++)
	{
		clusters[i] = p1[i];
	}
	// clusters[0] is the first centroid
	max_dist = -1.0;
	for(i=0;i<num_of_vectors;i++)
	{
		min_dist = distance2(&training[map[i].index*dim2],clusters,dim);
		distances[i] = min_dist;
		if(min_dist>max_dist)
		{
			max_dist = min_dist;
			max_ind = i;
		}
	}
	p2 = &clusters[dim2];
	p1 = &training[map[max_ind].index*dim2];
	for(i=0;i<dim;i++)
	{
		p2[i] = p1[i];
	}
	for(;i<dim2;i++)
	{
		p2[i] = (MY_DOUBLE) 0;
	}
	// p2=clusters[1] is the second centroid
	total_sum = 0.0;
	for(i=0;i<num_of_vectors;i++)
	{
		p1 = &training[map[i].index*dim2];
		min_dist = distances[i];
		dist = my_distance2(p1,p2,dim,min_dist);
		if(dist<min_dist)
		{
			min_dist = dist;
			distances[i] = dist;
			indices[i] = 1;
		}
		total_sum += min_dist;
	}
	return ((double) total_sum)/((double) num_of_vectors);
}

void kmeans_cluster2_map(struct context *storage, MY_DOUBLE *training, int dim, int num_of_vectors, struct paired *map)
{
	MY_DOUBLE *clusters = storage->clusters2;
	int *indices = storage->indices2;
	MY_DOUBLE *distances = storage->distances2;
	int *cluster_count = storage->cluster_count2;
	double *new_clusters = storage->new_clusters2;
	int *diff_count = &storage->diff_count2;
	int i, j;
	MY_DOUBLE *p1;
	int min_ind;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));

	if(num_of_vectors<2)
	{
		return;
	}
	memset(new_clusters,0,2*dim2*sizeof(double));
	memset(cluster_count,0,2*sizeof(int));
	for(i=0;i<num_of_vectors;i++)
	{
		min_ind = indices[i];
		accumulate_vector(&new_clusters[min_ind*dim2],&training[map[i].index*dim2],dim);
		cluster_count[min_ind]++;
	}
	for(j=0;j<2;j++)
	{
		if((min_ind=cluster_count[j])>0)
		{
			double *p2;
			p1 = &clusters[j*dim2];
			p2 = &new_clusters[j*dim2];
			#pragma loop(ivdep)
			for(i=0;i<dim;i++)
			{
				p1[i] = (MY_DOUBLE) p2[i]/(MY_DOUBLE) min_ind;
			}
		}
	}
}

double kmeans_iterate2_map(struct context *storage, MY_DOUBLE *training, int dim, int num_of_vectors, struct paired *map)
{
	MY_DOUBLE *clusters = storage->clusters2;
	int *indices = storage->indices2;
	MY_DOUBLE *distances = storage->distances2;
	int *cluster_count = storage->cluster_count2;
	double *new_clusters = storage->new_clusters2;
	int *diff_count = &storage->diff_count2;
	int i, j;
	MY_DOUBLE *p1;
	MY_DOUBLE dist;
	MY_DOUBLE min_dist;
	int min_ind;
	double total_sum;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));

	if(num_of_vectors<2)
	{
		*diff_count = -1;
		return -1.0;
	}
	memset(new_clusters,0,2*dim2*sizeof(double));
	memset(cluster_count,0,2*sizeof(int));
	*diff_count = 0;
	total_sum = 0.0;
	for(i=0;i<num_of_vectors;i++)
	{
		p1 = &training[map[i].index*dim2];
		min_ind = indices[i];
		min_dist = distance2(p1,&clusters[min_ind*dim2],dim);
		dist = my_distance2(p1,&clusters[(1-min_ind)*dim2],dim,min_dist);
		if(dist<min_dist)
		{
			min_dist = dist;
			min_ind = 1-min_ind;
			(*diff_count)++;
			indices[i] = min_ind;
		}
		accumulate_vector(&new_clusters[min_ind*dim2],p1,dim);
		cluster_count[min_ind]++;
		distances[i] = min_dist;
		total_sum += min_dist;
	}
	for(j=0;j<2;j++)
	{
		if((min_ind=cluster_count[j])>0)
		{
			double *p2;
			MY_DOUBLE *p3;
			p3 = &clusters[j*dim2];
			p2 = &new_clusters[j*dim2];

			#pragma loop(ivdep)
			for(i=0;i<dim;i++)
			{
				p3[i] = (MY_DOUBLE) p2[i]/(MY_DOUBLE) min_ind;
			}
		}
	}
	return ((double) total_sum)/((double) (num_of_vectors*dim));
}

int increasing(const void *iptr1, const void *iptr2)
{
	struct paired *ptr1 = (struct paired *) iptr1;
	struct paired *ptr2 = (struct paired *) iptr2;

	if(ptr1->signed_distance<ptr2->signed_distance)
		return -1;
	else
		return 1;
}

int binary_search(struct paired *pairs, int count, MY_DOUBLE mid_point)
{
	int min_i, max_i, mid_i;
	min_i = 0;
	max_i = count-1;
	for(;max_i>min_i+1;)
	{
		mid_i = (max_i+min_i+1)/2;
		if(pairs[mid_i].signed_distance<=mid_point)
			min_i = mid_i;
		else
			max_i = mid_i;
	}
	return max_i;
}

void tree_structure(struct node *root, MY_DOUBLE *clusters,int dim,struct context *storage)
{
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));

	if(root->count>FAST_NN_THRESHOLD)
	{
		int i;
		int iter = 0;
		double ret_val;
		int mid_i;
		ret_val = kmeans_initialize2_map(storage, clusters, dim, root->count, root->pairs);
		kmeans_cluster2_map(storage, clusters, dim, root->count, root->pairs);
		//printf("TREE: iter=%3d, ret_val= %lf, diff=%7d, H=%lf\n",iter,ret_val,diff_count2,entropy(cluster_count2,2)/((double) dim));
		iter++;
		for(iter=1;iter<100 && storage->diff_count2>0;iter++)
		{
			ret_val = kmeans_iterate2_map(storage, clusters, dim,root->count, root->pairs);
			//printf("TREE: iter=%3d, ret_val= %lf, diff=%7d, H=%lf\n",iter,ret_val,diff_count2,entropy(cluster_count2,2)/((double) dim));
		}
		root->c0 = calc_hyperplane(&storage->clusters2[0*dim2],&storage->clusters2[1*dim2],root->hyperplane,dim);
		for(i=0;i<root->count;i++)
		{
			root->pairs[i].signed_distance = signed_distance(&clusters[root->pairs[i].index*dim2],root->hyperplane,root->c0,dim);
		}
		qsort((void *) root->pairs,root->count,sizeof(struct paired),increasing);
		mid_i = binary_search(root->pairs,root->count,0.0);
		root->left = allocate_node(mid_i,dim);
		root->left->count = mid_i;
		for(i=0;i<mid_i;i++)
		{
			root->left->pairs[i].index = root->pairs[i].index;
		}
		root->right = allocate_node(root->count-mid_i,dim);
		root->right->count = root->count-mid_i;
		for(i=mid_i;i<root->count;i++)
		{
			root->right->pairs[i-mid_i].index = root->pairs[i].index;
		}
		tree_structure(root->left,clusters,dim,storage);
		tree_structure(root->right,clusters,dim,storage);
	}
}

#if 01
int fast_NN(MY_DOUBLE *vector, struct node *root, MY_DOUBLE *clusters, int dim, MY_DOUBLE *min_dist2)
{
	int i;
	int min_ind;
	MY_DOUBLE dist;
	MY_DOUBLE test_dist;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));

	if(root->count>FAST_NN_THRESHOLD)
	{
		double limit;
		struct paired *ptr;
		dist = signed_distance(vector,root->hyperplane,root->c0,dim);
#ifdef PROFILE
		count_product++;
#endif
		if(dist<=0.0)
		{
			min_ind = fast_NN(vector,root->left,clusters,dim,min_dist2);
			limit = min_dist2[1]+dist;
			i = root->left->count;
			for(ptr = &root->pairs[i];i<root->count && ptr->signed_distance<limit;i++,ptr++)
			{
				test_dist = my_distance2(vector,&clusters[ptr->index*dim2],dim,min_dist2[0]);
#ifdef PROFILE
				count_distance++;
#endif
				if(test_dist<min_dist2[0])
				{
#ifdef PROFILE
					if((ptr->signed_distance-dist)/min_dist2[1]>max_ratio)
						max_ratio = (ptr->signed_distance-dist)/min_dist2[1];
					count_sqrt++;
#endif
					min_dist2[0] = test_dist;
					min_dist2[1] = (MY_DOUBLE) sqrt(test_dist);
					limit = min_dist2[1]+dist;
					min_ind = ptr->index;
				}
			}
		}
		else
		{
			min_ind = fast_NN(vector,root->right,clusters,dim,min_dist2);
			limit = dist-min_dist2[1];
			i = root->left->count-1;
			for(ptr = &root->pairs[i];i>=0 && ptr->signed_distance>limit;i--,ptr--)
			{
				test_dist = my_distance2(vector,&clusters[ptr->index*dim2],dim,min_dist2[0]);
#ifdef PROFILE
				count_distance++;
#endif
				if(test_dist<min_dist2[0])
				{
#ifdef PROFILE
					if((dist-ptr->signed_distance)/dist2_sqrt>max_ratio)
						max_ratio = (dist-ptr->signed_distance)/min_dist2[1];
					count_sqrt++;
#endif
					min_dist2[0] = test_dist;
					min_dist2[1] = (MY_DOUBLE) sqrt(test_dist);
					limit = dist-min_dist2[1];
					min_ind = ptr->index;
				}
			}
		}
	}
	else
	{
		min_dist2[0] = distance2(vector,&clusters[root->pairs[0].index*dim2],dim);
#ifdef PROFILE
		count_distance++;
#endif
		min_ind = root->pairs[0].index;
#if FAST_NN_THRESHOLD==2
		if(root->count>1)
		{
			test_dist = my_distance2(vector,&clusters[root->pairs[1].index*dim2],dim,min_dist2[0]);
#ifdef PROFILE
			count_distance++;
#endif
			if(test_dist<min_dist2[0])
			{
				min_ind = root->pairs[1].index;
				min_dist2[0] = test_dist;
			}
		}
#elif FAST_NN_THRESHOLD>1
		for(i=1;i<root->count;i++)
		{
			test_dist = my_distance2(vector,&clusters[root->pairs[i].index*dim2],dim,min_dist2[0]);
#ifdef PROFILE
			count_distance++;
#endif
			if(test_dist<min_dist2[0])
			{
				min_ind = root->pairs[i].index;
				min_dist2[0] = test_dist;
			}
		}
#endif
		min_dist2[1] = (MY_DOUBLE) sqrt(min_dist2[0]);
#ifdef PROFILE
		count_sqrt++;
#endif
	}
	return min_ind;
}
#else
int fast_NN(MY_DOUBLE *vector, struct node *root, MY_DOUBLE *clusters, int dim, MY_DOUBLE *min_dist2)
{
	int i;
	int min_ind;
	MY_DOUBLE dist;
	MY_DOUBLE test_dist;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));

	if(root->count>FAST_NN_THRESHOLD)
	{
		struct paired *ptr;
		dist = signed_distance(vector,root->hyperplane,root->c0,dim);
		if(dist<=0.0)
		{
			min_ind = fast_NN(vector,root->left,clusters,dim,min_dist2);
			i = root->left->count;
			for(ptr = &root->pairs[i];i<root->count;i++,ptr++)
			{
				MY_DOUBLE test_dist = ptr->signed_distance - dist;
				if(test_dist*test_dist>=min_dist2[0])
					return min_ind;
				test_dist = my_distance2(vector,&clusters[ptr->index*dim2],dim,min_dist2[0]);
				if(test_dist<min_dist2[0])
				{
					min_dist2[0] = test_dist;
					min_ind = ptr->index;
				}
			}
		}
		else
		{
			min_ind = fast_NN(vector,root->right,clusters,dim,min_dist2);
			i = root->left->count-1;
			for(ptr = &root->pairs[i];i>=0;i--,ptr--)
			{
				MY_DOUBLE test_dist = dist - ptr->signed_distance;
				if(test_dist*test_dist>=min_dist2[0])
					return dist2;
				test_dist = my_distance2(vector,&clusters[ptr->index*dim2],dim,min_dist2[0]);
				if(test_dist<min_dist2[0])
				{
					min_dist2[0] = test_dist;
					min_ind = ptr->index;
				}
			}
		}
	}
	else
	{
		min_dist2[0] = distance2(vector,&clusters[root->pairs[0].index*dim2],dim);
		min_ind = root->pairs[0].index;
#if FAST_NN_THRESHOLD==2
		if(root->count>1)
		{
			test_dist = my_distance2(vector,&clusters[root->pairs[1].index*dim2],dim,min_dist2[0]);
			if(test_dist<min_dist2[0])
			{
				min_ind = root->pairs[1].index;
				min_dist2[0] = test_dist;
			}
		}
#elif FAST_NN_THRESHOLD>1
		for(i=1;i<root-count;i++)
		{
			test_dist = my_distance2(vector,&clusters[root->pairs[i].index*dim2],dim,min_dist2[0]);
			if(test_dist<min_dist2[0])
			{
				min_ind = root->pairs[i].index;
				min_dist2[0] = test_dist;
			}
		}
#endif
	}
	return min_ind;
}
#endif

double kmeans_iterate_fast(MY_DOUBLE *clusters, MY_SHORT *training, int *indices, MY_DOUBLE *distances, int *cluster_count, double *new_clusters, 
						   int dim, int num_of_clusters, int num_of_vectors, int *diff_count, struct node *root, struct context *storage)
{
	int i, j;
	MY_SHORT *p1;
	//MY_DOUBLE dist;
	MY_DOUBLE min_dist[2];
	int min_ind;
	double total_sum;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));
	__declspec(align(16)) MY_DOUBLE local_vector[8];

	if(num_of_vectors==0 || num_of_vectors<num_of_clusters)
	{
		*diff_count = -1;
		return -1.0;
	}
#if 0
	local_vector = (MY_DOUBLE *) my_malloc(dim2*sizeof(MY_DOUBLE),ALIGNMENT);
	if(local_vector==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			dim2*sizeof(MY_DOUBLE), "local_vector");
		exit(-1);
	}
#endif
	tree_structure(root,clusters,dim,storage);
	memset(new_clusters,0,num_of_clusters*dim2*sizeof(double));
	memset(cluster_count,0,num_of_clusters*sizeof(int));
	*diff_count = 0;
	total_sum = 0.0;
#ifdef VERBOSE
#ifdef PROFILE
	//printf("                                                                 ");
#else
	printf("        ");
#endif
#endif
	#pragma omp parallel for private(min_ind,p1,min_dist,local_vector)
	for(i=0;i<num_of_vectors;i++)
	{
#ifdef VERBOSE
		if((i%10000)==0)
		{
#ifdef PROFILE
			//printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b%8d %12I64d %12I64d %12I64d %8.2lf",
			//	i,count_product,count_distance,count_sqrt);
			printf("%8d %12I64d %12I64d %12I64d %8.2lf %8.4lf\n",
				i,count_product,count_distance,count_sqrt, (double) count_distance/(double) 10000,max_ratio);
			count_product = count_distance = count_sqrt = 0;
			max_ratio = 0.0;
#else
			printf("\b\b\b\b\b\b\b\b%8d",
				i);
#endif
		}
#endif
		p1 = &training[i*dim2];
		expand_vector(p1,local_vector,dim);
		//min_ind = indices[i];
		//min_dist[0] = distance2(p1,&clusters[min_ind*dim2],dim);
		min_ind = fast_NN(local_vector,root,clusters,dim,min_dist);
		#pragma omp critical
		{
			if(indices[i] != min_ind)
			{
				(*diff_count)++;
				indices[i] = min_ind;
			}
			accumulate_vector(&new_clusters[min_ind*dim2],local_vector,dim);
			cluster_count[min_ind]++;
			distances[i] = min_dist[0];
			total_sum += min_dist[0];
		}
	}
	for(j=0;j<num_of_clusters;j++)
	{
		if((min_ind=cluster_count[j])>0)
		{
			double *p2;
			MY_DOUBLE *p3;
			p3 = &clusters[j*dim2];
			p2 = &new_clusters[j*dim2];
			for(i=0;i<dim;i++)
			{
				p3[i] = (MY_DOUBLE) p2[i]/(MY_DOUBLE) min_ind;
			}
		}
	}
	free_tree(root->left);
	free_tree(root->right);
	//my_free(local_vector);
	return ((double) total_sum)/((double) (num_of_vectors*dim));
}
#endif

int main(int argc, char *argv[])
{
	double ret_val;
	MY_DOUBLE *clusters;
	MY_SHORT *training;
	int *indices;
	MY_DOUBLE *distances;
	int *cluster_count;
	double *new_clusters;
	MY_DOUBLE *hyperplane;
	int dim = 9;
	int num_of_clusters = 32;
	int num_of_vectors = 4*1024;
	int diff_count;
	int iter;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));
	clock_t start,finish;
	double duration;
	struct node *root;
	struct context *storage;

	clusters = (MY_DOUBLE *) my_malloc(num_of_clusters*dim2*sizeof(MY_DOUBLE),ALIGNMENT);
	if(clusters==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			num_of_clusters*dim2*sizeof(MY_DOUBLE), "clusters");
		exit(-1);
	}
	training = (MY_SHORT *) my_malloc(num_of_vectors*dim2*sizeof(MY_SHORT),ALIGNMENT);
	if(training==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			num_of_vectors*dim2*sizeof(MY_SHORT), "training");
		exit(-1);
	}
	indices = (int *) my_malloc(num_of_vectors*sizeof(int),ALIGNMENT);
	if(indices==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			num_of_vectors*sizeof(int), "indices");
		exit(-1);
	}
	distances = (MY_DOUBLE *) my_malloc(num_of_vectors*sizeof(MY_DOUBLE),ALIGNMENT);
	if(distances==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			num_of_vectors*sizeof(MY_DOUBLE), "distances");
		exit(-1);
	}
	cluster_count = (int *) my_malloc(num_of_clusters*sizeof(int),ALIGNMENT);
	if(cluster_count==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			num_of_clusters*sizeof(int), "cluster_count");
		exit(-1);
	}
	new_clusters = (double *) my_malloc(num_of_clusters*dim2*sizeof(double),ALIGNMENT);
	if(new_clusters==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			num_of_clusters*dim2*sizeof(double), "new_clusters");
		exit(-1);
	}
	hyperplane = (MY_DOUBLE *) my_malloc(dim2*sizeof(MY_DOUBLE),ALIGNMENT);
	if(hyperplane==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			dim2*sizeof(MY_DOUBLE), "hyperplane");
		exit(-1);
	}
#ifdef FAST_NN
	storage = (struct context *) my_malloc(1*sizeof(struct context),ALIGNMENT);
	if(storage==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			1*sizeof(struct context), "storage");
		exit(-1);
	}
	storage->clusters2 = (MY_DOUBLE *) my_malloc(2*dim2*sizeof(MY_DOUBLE),ALIGNMENT);
	if(storage->clusters2==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			2*dim2*sizeof(MY_DOUBLE), "storage->clusters2");
		exit(-1);
	}
	storage->indices2 = (int *) my_malloc(num_of_clusters*sizeof(int),ALIGNMENT);
	if(storage->indices2==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			num_of_clusters*sizeof(MY_DOUBLE), "storage->indices2");
		exit(-1);
	}
	storage->cluster_count2 = (int *) my_malloc(2*sizeof(int),ALIGNMENT);
	if(storage->cluster_count2==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			2*sizeof(int), "storage->cluster_count2");
		exit(-1);
	}
	storage->distances2 = (MY_DOUBLE *) my_malloc(num_of_clusters*sizeof(MY_DOUBLE),ALIGNMENT);
	if(storage->distances2==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			num_of_clusters*sizeof(MY_DOUBLE), "storage->distances2");
		exit(-1);
	}
	storage->new_clusters2 = (double *) my_malloc(2*dim2*sizeof(double),ALIGNMENT);
	if(storage->new_clusters2==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			2*dim2*sizeof(double), "new_clusters2");
		exit(-1);
	}
	{
		int i;
		root = allocate_node(num_of_clusters,dim);
		root->count = num_of_clusters;
		for(i=0;i<num_of_clusters;i++)
		{
			root->pairs[i].index = i;
		}
	}
#endif

	printf("Memory allocation completed\n");
	printf("VQ-dimension: %d\n",dim);
	printf("Training vector size: %d\n",num_of_vectors);
	printf("Codebook size: %d\n",num_of_clusters);
	randomize(training,dim,num_of_vectors);
	iter = 0;
	diff_count = 1;
#ifdef KKZ
	printf("KKZ initialization\n");
	start = clock();
	ret_val = kmeans_initialize(clusters, training, indices, distances, cluster_count, new_clusters, dim, num_of_clusters, num_of_vectors, &diff_count);
	kmeans_cluster(clusters, training, indices, distances, cluster_count, new_clusters, dim, num_of_clusters, num_of_vectors, &diff_count);
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("iter=%3d, ret_val= %lf, diff=%7d, H=%lf, Time=%8.3lf\n",iter,ret_val,diff_count,entropy(cluster_count,num_of_clusters)/((double) dim),duration);
	iter++;
#else
	printf("Random initialization\n");
	initialize(clusters,training,dim,num_of_clusters,num_of_vectors);
	memset(indices,0,num_of_vectors*sizeof(int));
#endif
	for(;iter<1000 && diff_count>0;iter++)
	{
		start = clock();
#ifdef FAST_NN
		ret_val = kmeans_iterate_fast(clusters, training, indices, distances, cluster_count, new_clusters, dim, num_of_clusters, num_of_vectors, &diff_count,root,storage);
#else
		ret_val = kmeans_iterate(clusters, training, indices, distances, cluster_count, new_clusters, dim, num_of_clusters, num_of_vectors, &diff_count);
		//ret_val = kmeans_iterate2(clusters, training, indices, distances, cluster_count, new_clusters, dim, num_of_vectors, &diff_count);
		//ret_val = kmeans_iterate2_h(clusters, training, indices, distances, cluster_count, new_clusters, dim, hyperplane, num_of_vectors, &diff_count);
#endif
		finish = clock();
		duration = (double)(finish - start) / CLOCKS_PER_SEC;

		printf("iter=%3d, ret_val= %lf, diff=%7d, H=%lf, Time=%8.3lf\n",iter,ret_val,diff_count,entropy(cluster_count,num_of_clusters)/((double) dim),duration);
	}

#ifdef FAST_NN
	free_node(root);
	my_free(storage->new_clusters2);
	my_free(storage->distances2);
	my_free(storage->cluster_count2);
	my_free(storage->indices2);
	my_free(storage->clusters2);
	my_free(storage);
#endif
	my_free(hyperplane);
	my_free(new_clusters);
	my_free(cluster_count);
	my_free(distances);
	my_free(indices);
	my_free(training);
	my_free(clusters);
	return 0;
}
