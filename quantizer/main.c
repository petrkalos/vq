#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Windows.h>
#include <float.h>

#define MY_DOUBLE float
#define MY_SHORT short

int XSIZE=720;
int YSIZE=480;

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


void quantize_luma(MY_SHORT *y,MY_DOUBLE *clusters,int dims){
	int i,j,r,min_ind;
	int dim = dims*dims;
	double min_dist,dist;
	MY_SHORT *block = (MY_SHORT *) malloc(sizeof(MY_SHORT)*dim);

	for(r=0;r<YSIZE;r+=dims){
		for(i=0;i<XSIZE;i+=dims){
			for(j=0;j<dims;j++){
				memcpy(&block[j*dims],&y[i+(r+j)*XSIZE],dims*sizeof(MY_SHORT));
			}

			min_dist = FLT_MAX;
			min_ind = 0;
			for(j=0;j<dim;j++){
				dist = distance2_float_short_c(&clusters[j],block,dim);

				if(dist<min_dist){
					min_ind = j;
					min_dist = dist;
				}
			}

			for(j=0;j<dim;j++){
				block[j] = (MY_SHORT)(clusters[min_ind+j]+0.5);
			}

			for(j=0;j<dims;j++){
				memcpy(&y[i+(r+j)*XSIZE],&block[j*dims],dims*sizeof(MY_SHORT));
			}

		}
	}

	free(block);
}

void quantize_uv(MY_SHORT *y,MY_DOUBLE *clusters,int dim){
}

int main(int argc, char *argv[]){

	int dims = 16;
	int num_of_clusters = 32768;
	int num_of_frames = 2600;
	int dim = dims*dims;
	int frame,i;
	char file[100];
	FILE *fp=NULL,*fpw;
	MY_DOUBLE *clusters;
	MY_SHORT *y,*u,*v;
	
	sprintf(file,"../codebook_256_%d.bin",num_of_clusters);

	fopen_s(&fp,"../codebook_256_128.bin","rb");
	if(fp==NULL){
		printf("Open codebook file error\n");
		exit(1);
	}
	clusters = (MY_DOUBLE *)malloc(sizeof(MY_DOUBLE)*num_of_clusters*dim);
	fread(clusters,sizeof(MY_DOUBLE),num_of_clusters*dim,fp);
	fclose(fp);

	fopen_s(&fp,"../src13_16b.yuv","rb");
	if(fp==NULL){
		printf("Open video file error\n");
		exit(1);
	}

	fopen_s(&fpw,"src13_quantized.yuv","wb");

	y = (MY_SHORT *)malloc(sizeof(MY_SHORT)*XSIZE*YSIZE);
	u = (MY_SHORT *)malloc(sizeof(MY_SHORT)*XSIZE*YSIZE/4);
	v = (MY_SHORT *)malloc(sizeof(MY_SHORT)*XSIZE*YSIZE/4);
	
	for(frame=0;frame<num_of_frames;frame++){
		fread(y,sizeof(MY_SHORT),XSIZE*YSIZE,fp);
		quantize_luma(y,clusters,dims);
		fwrite(y,sizeof(MY_SHORT),XSIZE*YSIZE,fpw);

		fread(u,sizeof(MY_SHORT),XSIZE*YSIZE/4,fp);
		quantize_uv(u,clusters,dims);
		fwrite(u,sizeof(MY_SHORT),XSIZE*YSIZE/4,fpw);

		fread(v,sizeof(MY_SHORT),XSIZE*YSIZE/4,fp);
		quantize_uv(v,clusters,dims);
		fwrite(v,sizeof(MY_SHORT),XSIZE*YSIZE/4,fpw);
	}

	fclose(fpw);
	fclose(fp);
}