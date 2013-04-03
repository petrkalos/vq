#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <string.h>

extern int start_kmeans(int dim,int num_of_clusters,int num_of_frames,int type,char *i_filename,char *o_filename);

__int64 getFileSize(const char *filename){
	struct _stat64 buf;
	int fh, result;
	char buffer[] = "A line to output";

	if( (fh = _open( filename, _O_RDONLY | _O_BINARY )) ==  -1 ){
		return -1;
	}
		
	/* Get data associated with "fh": */
	result = _fstati64( fh, &buf );
	
	return buf.st_size;
}

int main(int argc,char *argv[]){
	int dim,num_of_clusters,num_of_frames,type;
	__int64 size;

	if(argc!=7){
		printf("Insert [dim] [codebook size] [number of threads] [inputfile] [outputfile]\n");
		exit(1);
	}

	dim = atoi(argv[1]);
	num_of_clusters = atoi(argv[2]);
	num_of_frames = atoi(argv[3]);
	omp_set_num_threads(atoi(argv[4]));
	
	size = getFileSize(argv[5]);
	if(size==-1){
		printf("Error opening input file\n");
		return 1;
	}

	if(num_of_frames == 0 ){
		num_of_frames = size/((720*480*3/2)*sizeof(short));
	}

	for(type=0;type<2;type++){
		start_kmeans(dim,num_of_clusters,num_of_frames,type,argv[5],argv[6]);
	}
}