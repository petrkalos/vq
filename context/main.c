#include <stdio.h>
#include <math.h>
#include <string.h>
#include <memory.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <time.h>
#include <omp.h>
#include <windows.h>
#include <WinError.h>

#include "fastnn.h"

#define round(r) (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5)

#define FASTNN

#ifdef FASTNN
	struct node *root;
	struct context *stor;
	float min_dist[2];
#endif

typedef unsigned __int64 uint64;
typedef __int64 int64;

#define NOC 16

//const int64 cat[1][NOC+1] = {{-1,28,245,1970,644402},{-1,30,579,6498,572890},{-1,30,579,6498,572890}};

int64 cat[NOC+1];

const int XSIZE = 720;
const int YSIZE = 480;

static uint64 total_vectors = 0;

typedef struct{
	float *resi;
	char category;
	int index;
}block;

float *energy;

void check_memory(const char *msg,void *ptr){
	if(ptr==NULL){
		printf("%s\n",msg);
		exit(ERROR_NOT_ENOUGH_MEMORY);
	}
}

void read_splits(){
	FILE *fp;
	fp = fopen("../MATLAB/splits.bin","rb");
	fread(cat,sizeof(int64),NOC+1,fp);
	fclose(fp);

}

void init_blocks(block *****bt,int num_of_frames,int dims,int type){
	int frame,m,n;
	int dim = dims*dims;
	double mode;

	if(type==0)
		mode = 1;
	else
		mode = 0.5;

	*bt = (block ****) malloc(sizeof(block ***)*num_of_frames);
	check_memory("Block frames memory",(*bt));

	for(frame=0;frame<num_of_frames;frame++){
		(*bt)[frame] = (block ***) malloc(sizeof(block **)*(mode*YSIZE)/dims);
		check_memory("Block y-dim memory",(*bt)[frame]);
		for(m=0;m<(mode*YSIZE)/dims;m++){
			(*bt)[frame][m] = (block **) malloc(sizeof(block *)*(mode*XSIZE)/dims);
			check_memory("Block x-dim memory",(*bt)[frame][m]);
			for(n=0;n<(mode*XSIZE)/dims;n++){
				(*bt)[frame][m][n] = (block *) malloc(sizeof(block));
				check_memory("Block struct memory",(*bt)[frame][m][n]);
				(*bt)[frame][m][n]->resi = (float *)_aligned_malloc(sizeof(float)*dim,16);
				check_memory("Block residual memory",(*bt)[frame][m]);
				(*bt)[frame][m][n]->category = -1;
				(*bt)[frame][m][n]->index = -1;
			}
		}
	}
}

void free_blocks(block ****resi,int num_of_frames,int dims,int type){
	int frame,m,n;
	double mode;

	if(type==0)
		mode = 1;
	else
		mode = 0.5;

	for(frame=0;frame<num_of_frames;frame++){
		for(m=0;m<(mode*YSIZE)/dims;m++){
			for(n=0;n<(mode*XSIZE)/dims;n++){
				free(resi[frame][m][n]->resi);
				free(resi[frame][m][n]);
			}
			free(resi[frame][m]);
		}
		free(resi[frame]);
	}
	free(resi);
}

void init_counters(uint64 ***cnt,int num_of_contexts,int num_of_clusters){
	int i;
	*cnt = (uint64 **) malloc(sizeof(uint64 *)*num_of_contexts);
	check_memory("Counters memory",(*cnt));
	for(i=0;i<num_of_contexts;i++){
		(*cnt)[i]= (uint64 *)calloc(num_of_clusters,sizeof(uint64));
		check_memory("Counters row memory",(*cnt)[i]);
	}
}

void free_counters(uint64 **cnt,int num_of_contexts){
	int i;
	
	for(i=0;i<num_of_contexts;i++)
		free(cnt[i]);

	free(cnt);
}

void readResiduals(char *filename,block ****bt,int fromFrame,int toFrame,int dims,int type){
	static short buff[480][720];
	FILE *fp;
	int64 frame,j,i,m,n,count,k;
	int dim = dims*dims;
	double mode;
	int step = (toFrame-fromFrame);

	fopen_s(&fp,filename,"rb");
	if(fp==NULL){
		printf("Cannot open residuals file\n");
		exit(ERROR_FILE_NOT_FOUND);
	}

	if(type==0)
		mode=1.0;
	else
		mode=0.5;
	
	for(frame=fromFrame;frame<toFrame;frame++){
		if(type==0){
			fseek(fp,frame*XSIZE*YSIZE*3/2,SEEK_SET); //read Y
		}else if(type==1){
			fseek(fp,frame*XSIZE*YSIZE*3/2+XSIZE*YSIZE,SEEK_SET); //read U
		}else if(type==2){
			fseek(fp,frame*XSIZE*YSIZE*3/2+XSIZE*YSIZE+(XSIZE*YSIZE)/4,SEEK_SET); //read V
		}

		for(i=0;i<YSIZE*mode;i++){
				fread(buff[i],sizeof(short),XSIZE*mode,fp);	
		}

		n=0;
		for(j=0;j<XSIZE*mode;j+=dims){
			m=0;
			count = 0;
			for(i=0;i<YSIZE*mode;i++){
				for(k=0;k<dims;k++){
					bt[frame%step][m][n]->resi[count+k] = (float)buff[i][j];
				}
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
	//#undef short
}

void calcEnergy(float *codebook,int num_of_clusters,int dim){
	int i,j;
	float en;

	for(i=0;i<num_of_clusters;i++){
		en = 0.0;
		for(j=0;j<dim;j++){
			en += codebook[i*dim+j]*codebook[i*dim+j];
		}
		energy[i] = en;
	}
}
 
void readCodebook(char *filename,float *cb,int num_of_clusters,int dim){
	FILE *fp;
	int i;

	printf("codebook filename %s\n",filename);
	fopen_s(&fp,filename,"rb");
	if(fp==NULL){
		printf("Cannot open codebook file\n");
		exit(ERROR_FILE_NOT_FOUND);
	}

	fread(cb,sizeof(float),dim*num_of_clusters,fp);

	for(i=0;i<dim*num_of_clusters;i++){
		cb[i] = round(cb[i]);
	}

	fclose(fp);

	initNN(&root,&stor,dim,num_of_clusters,cb);

	calcEnergy(cb,num_of_clusters,dim);

}

int getCategory(int i){
	int j;

	for(j=0;j<NOC;j++){
		if((energy[i])>cat[j] && energy[i]<=cat[j+1])
			return j;
	}

	return 3;
}

int distance(short *vector1, short *vector2, int dim,int min_dist)
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

int quantizeBlock(float *cb,int num_of_clusters,block *bt,int dim,int type){
	int min_ind;

	min_ind = fastNN(bt->resi,root,cb,dim,min_dist);

	bt->index = min_ind;
	bt->category = getCategory(min_ind);

	return min_dist[0];
}

void getContext(block ***bt,uint64 **cnt,int currIdy,int currIdx,int x_size){
	char cat[NOC];
	int ind;
	int context;
	ind = bt[currIdy][currIdx]->index;
	if(ind<0){
		printf("index error\n");
		return;
	}

	if(currIdy==0 && currIdx==0){	//diagonial left
		cnt[256][ind]++;
	}else if(currIdy==0){			//first row
		cnt[257][ind]++;
	}else if(currIdx==0 || currIdx+1==x_size){	//first and last column
		cnt[258][ind]++;
	}else{
		cat[0] = bt[currIdy-1][currIdx-1]->category;
		cat[1] = bt[currIdy-1][currIdx]->category;
		cat[2] = bt[currIdy-1][currIdx+1]->category;
		cat[3] = bt[currIdy][currIdx-1]->category;

		context = cat[0]*64+cat[1]*16+cat[2]*4+cat[3];

		cnt[context][ind]++;		
	}
	total_vectors++;
}

int main(int argc, char *argv[]){
	int num_of_frames;
	char filename[100];
	const int num_of_clusters = 65536;
	const int num_of_contexts = 256+3;
	int dim = 16;
	int dims = (int) sqrt(dim+0.0);
	int i,j,dom,dom_step,type;
	double err = 0;
	clock_t start,finish;
	int frame;
	block ****bt;
	float *cb;
	FILE *fp;
	uint64 **cnt;
	double dur,mode;
	
	if(argc!=6){
		printf("Insert [type] [num_of_frames] [num_of_threads] [codebook] [residuals]\n");
		exit(1);
	}

	type = atoi(argv[1]);
	num_of_frames = atoi(argv[2]);
	dom_step = num_of_frames/(atoi(argv[3]));
	
	if(num_of_frames % dom_step !=0){
		printf("num_of_frames %% step == 0\n");
		exit(1);
	}
	
	read_splits();

	if(dom_step>num_of_frames) dom_step = num_of_frames;

	energy = (float *) malloc(sizeof(float)*num_of_clusters);
	check_memory("Energy memory",energy);
	cb = (float *)_aligned_malloc(sizeof(float)*dim*num_of_clusters,16);
	check_memory("Codebook memory",energy);
	readCodebook(argv[4],cb,num_of_clusters,dim);

	init_counters(&cnt,num_of_contexts,num_of_clusters);

	omp_set_num_threads(atoi(argv[3]));

	init_blocks(&bt,dom_step,dims,type);

	

	
	if(type==0)
		mode = 1;
	else
		mode = 0.5;
	
	start = clock();
	l1:

	printf("\nType: %d, num_of_frames: %d, num_of_clusters: %d\n",type,num_of_frames,num_of_clusters);
	for(dom=0;dom<num_of_frames;dom+=dom_step){
		printf("\rDom: %d",dom);	
		readResiduals(argv[5],bt,dom,dom+dom_step,dims,type);
		#pragma omp parallel for private(err,i,j) 
		for(frame=0;frame<dom_step;frame++){
			err = 0;
			for(i=0;i<(mode*YSIZE)/dims;i++){
				for(j=0;j<(mode*XSIZE)/dims;j++){
					err += quantizeBlock(cb,num_of_clusters,bt[frame][i][j],dim,type);
				}
			}
			//printf("Frame = %d, Distortion = %.2lf\n",dom+frame,err/(XSIZE*YSIZE));
		}
	
		for(frame=0;frame<dom_step;frame++){
			for(i=0;i<(mode*YSIZE)/dims;i++){
				for(j=0;j<(mode*XSIZE)/dims;j++){
					getContext(bt[frame],cnt,i,j,(mode*XSIZE)/dims);
				}
			}
		}
	}

	if(type==1){
		type=2;
		goto l1;
	}

	finish = clock();
	dur = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("\nTotal duration = %.2lf seconds\n",dur);

	sprintf(filename,"context%d.bin",type);
	fopen_s(&fp,filename,"wb");
	if(fp!=NULL){

		int i;
		for(i=0;i<num_of_contexts;i++)
			fwrite(cnt[i],sizeof(uint64),num_of_clusters,fp);
		fclose(fp);
	}else{
		printf("Error opening write file\n");
		exit(ERROR_FILE_NOT_FOUND);
	}

	printf("Total vectors quantized %u\n",total_vectors);
	return 0;
}