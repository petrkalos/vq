#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>


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
				(*resi)[frame][m][n] = (short *) malloc(sizeof(short)*dims*dims);
			}
		}
	}
}

void malloc2D(float ***cb,int num_of_vectors,int dim){
	int i;
	*cb = (float **) malloc(sizeof(float *)*num_of_vectors);
	
	for(i=0;i<num_of_vectors;i++)
		(*cb)[i]= (float *)malloc(sizeof(float)*dim);
}

void readResiduals(short ****resi,int num_of_frames,int dims){
	FILE *fp;
	int frame,j,i,m,n,count;
	int dim = dims*dims;
	short **buff;
	double mode = 1.0;

	fopen_s(&fp,"res_720x480.yuv","rb");

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

void readCodebook(float **cb,int num_of_vectors,int dim){
	int i;
	FILE *fp;
	fopen_s(&fp,"codebook.bin","rb");
	
	for(i=0;i<num_of_vectors;i++){
		fread(cb[i],sizeof(float),dim,fp);
	}
}

double distance2_float_short_c(float *vector1, short *vector2, int dim)
{
	int i;
	double sum;
	float diff;

	sum = 0.0;
	for(i=0;i<dim;i++)
	{
		diff = vector1[i] - vector2[i];
		sum += diff*diff;
	}
	return sum;
}

double quantizeBlock(float **codebook,int num_of_vectors,short *block,int dim){
	int i;
	double min_dist,dist;
	int min_ind;

	min_ind = 0;
	min_dist = FLT_MAX;
	for(i=0;i<num_of_vectors;i++){
		dist = distance2_float_short_c(&codebook[i],block,dim);

		if(dist<min_dist){
			min_dist = dist;
			min_ind = i;
		}
	}

	for(i=0;i<dim;i++)
		block[i] = (short)(codebook[min_ind][i]+0.5);

	return min_dist;

}

int getContext(short *block,int dim){
	int i;
	double energy=0;
	for(i=0;i<dim;i++){
		energy += block[i]*block[i];
	}

	if(C1(energy)) return 1;
	else if(C2(energy)) return 2;
	else if(C3(energy)) return 3;
	else if(C4(energy)) return 4;
	else return -1;
}



int main(int argc, char *argv[]){
	int num_of_frames=1;
	int num_of_vectors=32768;
	int dim = 16;
	int dims = (int) sqrt(dim+0.0);
	int i,j;
	double error = 0.0;
	short ****resi;
	float **cb;

	malloc4D(&resi,num_of_frames,dims);
	readResiduals(resi,num_of_frames,dims);
	
	malloc2D(&cb,num_of_vectors,dim);
	readCodebook(cb,num_of_vectors,dim);


	for(i=0;i<YSIZE/dims;i++){
		for(j=0;j<XSIZE/dims;j++){
			error += quantizeBlock(cb,num_of_vectors,resi[0][i][j],dim);
		}
	}
	printf("Total error %lf\n",error/(XSIZE*YSIZE));
}