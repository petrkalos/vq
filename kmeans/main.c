#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <string.h>

extern int start_kmeans(int dims,int num_of_clusters,int num_of_vectors,char *i_filename,char *o_filename);

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
	__int64 dims,num_of_clusters,num_of_vectors;
	__int64 size;

	if(argc!=7){
		printf("Insert [dims] [codebook size] [number of threads] [num_of_vectors] [inputfile] [outputfile]\n");
		exit(1);
	}

	dims = atoi(argv[1]);
	num_of_clusters = atoi(argv[2]);
	omp_set_num_threads(atoi(argv[3]));
	num_of_vectors = atoi(argv[4]);
	size = getFileSize(argv[5]);

	if(size==-1){
		printf("Error opening input file\n");
		return 1;
	}

	if(num_of_vectors==0)
		num_of_vectors = size/((dims*dims)*sizeof(short));

	
	start_kmeans(dims,num_of_clusters,num_of_vectors,argv[5],argv[6]);
	
}