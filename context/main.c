#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <emmintrin.h>
#include <omp.h>
#include <time.h>

#define C1(x) x>=0 && x<=19
#define C2(x) x>19 && x<=310
#define C3(x) x>310 && x<=5179
#define C4(x) x>5179 && x<=706957

const int XSIZE = 720;
const int YSIZE = 480;

void malloc4D(short *****resi,int num_of_frames,int dims){
	int frame,m,n;

	*resi = (short ****) malloc(sizeof(short ***)*num_of_frames);
	
	for(frame=0;frame<num_of_frames;frame++){
		(*resi)[frame] = (short ***) malloc(sizeof(short **)*YSIZE/dims);
		for(m=0;m<YSIZE/dims;m++){
			(*resi)[frame][m] = (short **) malloc(sizeof(short *)*XSIZE/dims);
			for(n=0;n<XSIZE/dims;n++){
				(*resi)[frame][m][n] = (short *) _aligned_malloc(sizeof(short)*dims*dims,16);
			}
		}
	}
}

void malloc2D(short ***cb,int num_of_vectors,int dim){
	int i;
	*cb = (short **) malloc(sizeof(short *)*num_of_vectors);
	
	for(i=0;i<num_of_vectors;i++)
		(*cb)[i]= (short *)_aligned_malloc(sizeof(short)*dim,16);
}

int getCategory(short *block,int dim){
	int i;
	int energy=0;


	for(i=0;i<dim;i++){
		energy += block[i]*block[i];
	}

	if(C1(energy)) return 0;
	else if(C2(energy)) return 1;
	else if(C3(energy)) return 2;
	else if(C4(energy)) return 3;
	else return -1;
}

void readResiduals(short ****resi,int num_of_frames,int dims){
	FILE *fp;
	int frame,j,i,m,n,count;
	int dim = dims*dims;
	short **buff;
	double mode = 1.0;

	fopen_s(&fp,"res_720x480_16bit.yuv","rb");

	buff = (short **)malloc(sizeof(short *)*YSIZE*mode);
	for(i=0;i<YSIZE*mode;i++){
		buff[i] = (short *)malloc(sizeof(short)*XSIZE*mode);
	}

	for(frame=0;frame<num_of_frames;frame++){
		fseek(fp,frame*XSIZE*YSIZE*3/2,SEEK_SET); //read Y

		for(i=0;i<YSIZE*mode;i++){
				fread(buff[i],sizeof(short),XSIZE*mode,fp);	
		}

		n=0;
		for(j=0;j<XSIZE*mode;j+=dims){
			m=0;
			count = 0;
			for(i=0;i<YSIZE*mode;i++){
				memcpy(&resi[frame][m][n][count],&buff[i][j],sizeof(short)*dims);
				count+=dims;
				if(count%dim==0){
					count=0;
					m++;
				}
			}
			n++;
		}
	}

	fclose(fp);
}

void writeResiduals(short ****resi,int num_of_frames,int dims){
//#define short unsigned char
	FILE *fp;
	int frame,j,i,m,n,count,t;
	int dim = dims*dims;
	short **buff;
	short temp;

	double mode = 1.0;

	fopen_s(&fp,"res_quant_720x480_16bit.yuv","wb");

	buff = (short **)malloc(sizeof(short *)*YSIZE*mode);
	for(i=0;i<YSIZE*mode;i++){
		buff[i] = (short *)malloc(sizeof(short)*XSIZE*mode);
	}
	
	for(frame=0;frame<num_of_frames;frame++){
		n=0;
		for(j=0;j<XSIZE*mode;j+=dims){
			m=0;
			count = 0;
			for(i=0;i<YSIZE*mode;i++){
				memcpy(&buff[i][j],&resi[frame][m][n][count],sizeof(short)*dims);
				/*temp = getContext(resi[frame][m][n],dim)*16383;
				for(t=0;t<dims;t++){
					buff[i][j+t] = temp;
				}*/

				count+=dims;
				if(count%dim==0){
					count=0;
					m++;
				}
			}
			n++;
		}

		for(i=0;i<YSIZE;i++)
			fwrite(buff[i],sizeof(short),XSIZE,fp);
		
		for(i=0;i<YSIZE;i++)
			fwrite(buff[i],sizeof(short),XSIZE/2,fp);
	}
	fclose(fp);
	#undef short
}

void readCodebook(short **cb,int num_of_vectors,int dim){
	int i,j;
	FILE *fp;
	float *temp = (float *)malloc(sizeof(float)*dim*num_of_vectors);
	
	fopen_s(&fp,"codebook.bin","rb");
	fread(temp,sizeof(float),dim*num_of_vectors,fp);
	fclose(fp);

	for(i=0;i<num_of_vectors;i++)
		for(j=0;j<dim;j++)
			cb[i][j] = (short)(temp[i*dim+j]+0.5);
		
	free(temp);
}

int distance2_float_short_c(short *vector1, short *vector2, int dim,int min_dist)
{
	int i;
	int sum;
	int diff;

	sum = 0;
	for(i=0;i<dim;i++)
	{
		diff = vector1[i] - vector2[i];
		sum += diff*diff;

		if(sum>min_dist) break;
	}
	
	return sum;
}

float distance2_float_short_sse2_16(float *vector1, short *vector2, int dim)
{
	int i;
	float sum;
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

void expand_vector_sse2(short *input, float *output, int dim)
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

double quantizeBlock(short **codebook,int num_of_vectors,short *block,int dim){
	int i;
	int min_dist,dist;
	int min_ind;

	min_ind = 0;
	min_dist = INT_MAX;
	for(i=0;i<num_of_vectors;i++){
		dist = distance2_float_short_c(codebook[i],block,dim,min_dist);

		if(dist<min_dist){
			min_dist = dist;
			min_ind = i;
		}
	}

	memcpy(block,codebook[min_ind],sizeof(short)*dim);
	
	return sqrt((double)min_dist);

}

int getContext(short ***resi,int currIdy,int currIdx,int dim){
	int cat[4];

	cat[0] = getCategory(resi[currIdy-1][currIdx-1],dim);
	cat[1] = getCategory(resi[currIdy-1][currIdx],dim);
	cat[2] = getCategory(resi[currIdy-1][currIdx+1],dim);
	cat[3] = getCategory(resi[currIdy][currIdx-1],dim);

	return (cat[0]*64+cat[1]*16+cat[2]*4+cat[3]);
}

int main(int argc, char *argv[]){
	int num_of_frames=100;
	int num_of_vectors=32768;
	int dim = 16;
	int dims = (int) sqrt(dim+0.0);
	int i,j,cont;
	double err = 0;
	clock_t start,finish;
	int frame;
	short ****resi;
	short **cb;
	FILE *fp;
	unsigned int context[255];
	double dur;

	memset(context,0,sizeof(unsigned int)*255);

	malloc4D(&resi,num_of_frames,dims);
	readResiduals(resi,num_of_frames,dims);
	
	malloc2D(&cb,num_of_vectors,dim);
	readCodebook(cb,num_of_vectors,dim);

	#pragma omp parallel for private(err,i,j,dur,start,finish)
	for(frame=0;frame<num_of_frames;frame++){
		start = clock();
		err = 0;
		for(i=0;i<YSIZE/dims;i++){
			for(j=0;j<XSIZE/dims;j++){
				err += quantizeBlock(cb,num_of_vectors,resi[frame][i][j],dim);
			}
		}
		finish = clock();
		dur = (double)(finish - start) / CLOCKS_PER_SEC;
		printf("Frame %d, Total error %lf,Duration %.2lf\n",frame,err/(XSIZE*YSIZE),dur);

	}
	
	for(frame=0;frame<num_of_frames;frame++){
		for(i=1;i<YSIZE/dims-1;i++){
			for(j=1;j<XSIZE/dims-1;j++){
				context[getContext(resi[frame],i,j,dim)]++;
			}
		}
	}
	
	fopen_s(&fp,"context.bin","wb");
	if(fp!=NULL){
		fwrite(context,sizeof(unsigned int),255,fp);
		fclose(fp);
	}
	
}