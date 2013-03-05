#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define C1(x) x>=0 && x<=19
#define C2(x) x>19 && x<=310
#define C3(x) x>310 && x<=5179
#define C4(x) x>5179 && x<=706957

const int dim = 16;

int getContext(short *block){
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
	
	printf("%d\n",C1(520));
}