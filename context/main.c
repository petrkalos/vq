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

#define C1(x) x>=0 && x<=22
#define C2(x) x>22 && x<=350
#define C3(x) x>350 && x<=5148
#define C4(x) x>5148

#define NOC 4

const int XSIZE = 720;
const int YSIZE = 480;


typedef struct{
	short *resi;
	char category;
	int index;
}block;

uint64 *energy;

void init_blocks(block *****resi,int num_of_frames,int dims,int type){
	int frame,m,n;
	int dim = dims*dims;
	double mode;

	if(type==0)
		mode = 1;
	else
		mode = 0.5;

	*resi = (block ****) malloc(sizeof(short ***)*num_of_frames);
	
	for(frame=0;frame<num_of_frames;frame++){
		(*resi)[frame] = (block ***) malloc(sizeof(block **)*(mode*YSIZE)/dims);
		for(m=0;m<(mode*YSIZE)/dims;m++){
			(*resi)[frame][m] = (block **) malloc(sizeof(block *)*(mode*XSIZE)/dims);
			for(n=0;n<(mode*XSIZE)/dims;n++){
				(*resi)[frame][m][n] = (block *) _aligned_malloc(sizeof(block),16);
				(*resi)[frame][m][n]->resi = (short *) malloc(sizeof(short)*dim);
				(*resi)[frame][m][n]->category = -1;
				(*resi)[frame][m][n]->index = -1;
			}
		}
	}
}

void init_codebook(short ***cb,int num_of_clusters,int dim){
	int i;
	*cb = (short **) malloc(sizeof(short *)*num_of_clusters);
	
	for(i=0;i<num_of_clusters;i++)
		(*cb)[i]= (short *)_aligned_malloc(sizeof(short)*dim,16);

	energy = (uint64 *) malloc(sizeof(uint64)*num_of_clusters);
}

void init_counters(uint64 ***cnt,int num_of_contexts,int num_of_clusters){
	int i;
	*cnt = (uint64 **) malloc(sizeof(uint64 *)*num_of_contexts);
	
	for(i=0;i<num_of_contexts;i++)
		(*cnt)[i]= (uint64 *)calloc(num_of_clusters,sizeof(uint64));
}

void readResiduals(char *filename,block ****bt,int fromFrame,int num_of_frames,int dims,int type){
	FILE *fp;
	int64 frame,j,i,m,n,count;
	int dim = dims*dims;
	short **buff;
	double mode;

	fopen_s(&fp,filename,"rb");
	if(fp==NULL){
		printf("Cannot open residuals file\n");
		exit(1);
	}


	if(type==0)
		mode=1.0;
	else
		mode=0.5;

	buff = (short **)malloc(sizeof(short *)*YSIZE*mode);
	for(i=0;i<YSIZE*mode;i++){
		buff[i] = (short *)malloc(sizeof(short)*XSIZE*mode);
	}

	bt = &bt[-fromFrame];
	
	label1:
	for(frame=fromFrame;frame<num_of_frames;frame++){
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
				memcpy(&bt[frame][m][n]->resi[count],&buff[i][j],sizeof(short)*dims);
				count+=dims;
				if(count%dim==0){
					count=0;
					m++;
				}
			}
			n++;
		}
	}
	//if(type==1){
		//if you read u go and read v;
		//type=2;
		//goto label1;
	//}

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

void readCodebook(char *filename,short **cb,int num_of_clusters,int dim){
	int i,j;
	FILE *fp;
	float *temp = (float *)malloc(sizeof(float)*dim*num_of_clusters);
	
	fopen_s(&fp,filename,"rb");
	if(fp==NULL){
		printf("Cannot open codebook file\n");
		exit(1);
	}


	fread(temp,sizeof(float),dim*num_of_clusters,fp);
	fclose(fp);

	for(i=0;i<num_of_clusters;i++)
		for(j=0;j<dim;j++)
			cb[i][j] = (short)(temp[i*dim+j]+0.5);
		
	free(temp);
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

int getCategory2(int i){
	if(C1(energy[i])) return 0;
	else if(C2(energy[i])) return 1;
	else if(C3(energy[i])) return 2;
	else if(C4(energy[i])) return 3;
	else return -1;
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

int quantizeBlock(short **codebook,int num_of_clusters,block *bt,int dim){
	int i;
	int min_dist,dist;
	int min_ind;

	min_ind = 0;
	min_dist = INT_MAX;
	for(i=0;i<num_of_clusters;i++){
		dist = distance(codebook[i],bt->resi,dim,min_dist);

		if(dist<min_dist){
			min_dist = dist;
			min_ind = i;
		}
	}

	memcpy(bt->resi,codebook[min_ind],sizeof(short)*dim);
	bt->index = min_ind;
	bt->category = getCategory2(min_ind);

	return min_dist;
}

void getContext(block ***bt,uint64 **cnt,int currIdy,int currIdx,int dim){
	char cat[NOC];
	int ind[NOC];
	int context,i;
	
	cat[0] = bt[currIdy-1][currIdx-1]->category;
	cat[1] = bt[currIdy-1][currIdx]->category;
	cat[2] = bt[currIdy-1][currIdx+1]->category;
	cat[3] = bt[currIdy][currIdx-1]->category;

	ind[0] = bt[currIdy-1][currIdx-1]->index;
	ind[1] = bt[currIdy-1][currIdx]->index;
	ind[2] = bt[currIdy-1][currIdx+1]->index;
	ind[3] = bt[currIdy][currIdx-1]->index;

	context = cat[0]*64+cat[1]*16+cat[2]*4+cat[3];

	for(i=0;i<NOC;i++){
		if(ind[i]>=0){
			cnt[context][ind[i]]++;
		}
	}

}

void calcEnergy(short **codebook,int num_of_clusters,int dim){
	int i,j;
	uint64 en;

	for(i=0;i<num_of_clusters;i++){
		en = 0;
		for(j=0;j<dim;j++){
			en += codebook[i][j]*codebook[i][j];
		}
		energy[i] = en;
	}
}

int main(int argc, char *argv[]){
	int num_of_frames;
	const int num_of_clusters = 65536;
	const int num_of_contexts = 256;
	int dim = 16;
	int dims = (int) sqrt(dim+0.0);
	int i,j,dom,dom_step,type;
	double err = 0;
	clock_t start,finish;
	int frame;
	block ****bt;
	short **cb;
	FILE *fp;
	uint64 **cnt;
	double dur,mode;
	
	
	if(argc!=6){
		printf("Insert [type] [num_of_frames] [num_of_threads] [codebook] [residuals]\n");
		exit(1);
	}

	type = atoi(argv[1]);
	num_of_frames = atoi(argv[2]);
	dom_step = atoi(argv[3]);

	if(type==0)
		mode = 1;
	else
		mode = 0.5;

	if(num_of_frames % dom_step !=0){
		printf("num_of_frames % step ==0)\n");
		exit(1);
	}
	if(dom_step>num_of_frames) dom_step = num_of_frames;

	init_codebook(&cb,num_of_clusters,dim);
	readCodebook(argv[4],cb,num_of_clusters,dim);

	init_counters(&cnt,num_of_contexts,num_of_clusters);

	omp_set_num_threads(dom_step);

	init_blocks(&bt,dom_step,dims,type);

	calcEnergy(cb,num_of_clusters,dim);

	printf("Type: %d, num_of_frames: %d, num_of_clusters: %d\n",type,num_of_frames,num_of_clusters);

	for(dom=0;dom<num_of_frames;dom+=dom_step){
		start = clock();
		readResiduals(argv[5],bt,dom,dom+dom_step,dims,type);
		#pragma omp parallel for private(err,i,j) 
		for(frame=0;frame<dom_step;frame++){
			err = 0;
			for(i=0;i<(mode*YSIZE)/dims;i++){
				for(j=0;j<(mode*XSIZE)/dims;j++){
					err += quantizeBlock(cb,num_of_clusters,bt[frame][i][j],dim);
				}
			}
			printf("Frame = %d, Distortion = %.2lf\n",dom+frame,err/(XSIZE*YSIZE));
		}
	
		for(frame=0;frame<dom_step;frame++){
			for(i=1;i<(mode*YSIZE)/dims-1;i++){
				for(j=1;j<(mode*XSIZE)/dims-1;j++){
					getContext(bt[frame],cnt,i,j,dim);
				}
			}
		}

		finish = clock();
		dur = (double)(finish - start) / CLOCKS_PER_SEC;
		printf("Seconds/Frame = %.2lf\n",dur/dom_step);
	}

	fopen_s(&fp,"context.bin","wb");
	if(fp!=NULL){
		int i;
		for(i=0;i<num_of_contexts;i++)
			fwrite(cnt[i],sizeof(uint64),num_of_clusters,fp);
		fclose(fp);
	}
	return 1;
}