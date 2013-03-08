#include <stdio.h>
#include <math.h>
#include <string.h>
#include <memory.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <time.h>
#include <omp.h>
#include <windows.h>

typedef unsigned __int64 uint64;
typedef unsigned __int64 int64;

#define C1(x) x>=0 && x<=19
#define C2(x) x>19 && x<=310
#define C3(x) x>310 && x<=5179
#define C4(x) x>5179

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

void malloc2D_64(uint64 ***cb,int num_of_vectors,int dim){
	int i;
	*cb = (uint64 **) malloc(sizeof(uint64 *)*num_of_vectors);
	
	for(i=0;i<num_of_vectors;i++)
		(*cb)[i]= (uint64 *)_aligned_malloc(sizeof(uint64)*dim,16);
}

int getCategory(short *block,int dim){
	int i;
	unsigned int energy=0;

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
	int64 frame,j,i,m,n,count;
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
	int frame,j,i,m,n,count;
	int dim = dims*dims;
	short **buff;

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

int quantizeBlock(short **codebook,int num_of_vectors,short *block,int dim){
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
	
	return min_ind;

}

int getContext(short ***resi,int currIdy,int currIdx,int dim){
	int cat[4];

	
	cat[0] = getCategory(resi[currIdy-1][currIdx-1],dim);
	cat[1] = getCategory(resi[currIdy-1][currIdx],dim);
	cat[2] = getCategory(resi[currIdy-1][currIdx+1],dim);
	cat[3] = getCategory(resi[currIdy][currIdx-1],dim);

	return (cat[0]*64+cat[1]*16+cat[2]*4+cat[3]);
}

int setContext(short ***resi,int currIdy,int currIdx,int dim,uint64 **cnt,short **codebook,int num_of_vectors){
	uint64 cat[4],ind[4];
	int i;
	uint64 context;

	
	ind[0] = quantizeBlock(codebook,num_of_vectors,resi[currIdy-1][currIdx-1],dim);
	cat[0] = getCategory(resi[currIdy-1][currIdx-1],dim);

	ind[1] = quantizeBlock(codebook,num_of_vectors,resi[currIdy-1][currIdx],dim);
	cat[1] = getCategory(resi[currIdy-1][currIdx],dim);
	
	ind[2] = quantizeBlock(codebook,num_of_vectors,resi[currIdy-1][currIdx+1],dim);
	cat[2] = getCategory(resi[currIdy-1][currIdx+1],dim);

	ind[3] = quantizeBlock(codebook,num_of_vectors,resi[currIdy][currIdx-1],dim);
	cat[3] = getCategory(resi[currIdy][currIdx-1],dim);

	context = cat[0]*64+cat[1]*16+cat[2]*4+cat[3];

	for(i=0;i<4;i++)
		cnt[context][ind[i]]++;

	return context;
}

int main(int argc, char *argv[]){
	int num_of_frames;
	const int num_of_vectors = 32768;
	const int num_of_contexts = 256;
	int dim = 16;
	int dims = (int) sqrt(dim+0.0);
	int i,j;
	double err = 0;
	clock_t start,finish;
	int frame;
	short ****resi;
	short **cb;
	FILE *fp;
	//uint64 context[num_of_contexts];
	uint64 **cnt;
	double dur;

	num_of_frames = atoi(argv[1]);

	malloc4D(&resi,num_of_frames,dims);
	readResiduals(resi,num_of_frames,dims);
	
	malloc2D(&cb,num_of_vectors,dim);
	readCodebook(cb,num_of_vectors,dim);

	malloc2D_64(&cnt,num_of_contexts,num_of_vectors);

	//omp_set_num_threads(atoi(argv[2]));

	//#pragma omp parallel for private(err,i,j,dur,start,finish)
	/*for(frame=0;frame<num_of_frames;frame++){
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
	}*/

	for(i=0;i<num_of_contexts;i++)
		memset(cnt[i],0,sizeof(uint64)*num_of_vectors);

	for(frame=0;frame<num_of_frames;frame++){
		start = clock();
		for(i=1;i<YSIZE/dims-1;i++){
			for(j=1;j<XSIZE/dims-1;j++){
				setContext(resi[frame],i,j,dim,cnt,cb,num_of_vectors);
			}
		}
		finish = clock();
		dur = (double)(finish - start) / CLOCKS_PER_SEC;
		printf("Frame %d,Duration %.2lf\n",frame,dur);
	}
	
	/*fopen_s(&fp,"context.bin","wb");
	if(fp!=NULL){
		fwrite(context,sizeof(uint64),256,fp);
		fclose(fp);
	}*/
	return 1;
}