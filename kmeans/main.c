#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

extern int start_kmeans(int dim,int num_of_clusters,int num_of_frames,int type,char *filename);

int main(int argc,char *argv[]){
	int dim,num_of_clusters,num_of_frames,type;
	char *filename;


	if(argc!=6){
		printf("Insert [dim] [codebook size] [number of frames] [inputfile] [number of threads]\n");
		exit(1);
	}

	dim = atoi(argv[1]);
	num_of_clusters = atoi(argv[2]);
	num_of_frames = atoi(argv[3]);
	filename = argv[4];
	omp_set_num_threads(atoi(argv[5]));
	
	for(type=0;type<2;type++){
		start_kmeans(dim,num_of_clusters,num_of_frames,type,filename);
	}
}