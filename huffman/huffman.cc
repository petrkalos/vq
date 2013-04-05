/* Copyright (c) 2013 the authors listed at the following URL, and/or
the authors of referenced articles or incorporated external code:
http://en.literateprograms.org/Huffman_coding_(C_Plus_Plus)?action=history&offset=20090129100015

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Retrieved from: http://en.literateprograms.org/Huffman_coding_(C_Plus_Plus)?oldid=16057
*/

#include "huffman.h"
#include <map>
#include <iostream>
#include <ostream>
#include <algorithm>
#include <iterator>
#include <string>

std::ostream& operator<<(std::ostream& os, std::vector<bool> vec)
{
	std::copy(vec.begin(), vec.end(), std::ostream_iterator<bool>(os, ""));
	return os;
}

int main(int argc,char *argv[])
{
	std::map<unsigned int, double> p;
	long long *counters;
	unsigned __int64 sum;
	double len,entr;
	int num_of_clusters = 65536;
	int dim = 16;
	int i;
	FILE *fp;
	
	counters = (long long *)malloc(sizeof(long long)*num_of_clusters);
	
	if(argc<2){
		printf("Enter [counters filename]\n");
		exit(1);
	}

	fopen_s(&fp,argv[1],"rb");

	if(fp==NULL){
		printf("Cannot open file\n");
		exit(0);
	}

	fread(counters,sizeof(long long),num_of_clusters,fp);
	fclose(fp);

	sum = 0;
	for(i=0;i<num_of_clusters;i++){
		sum+=counters[i];
	}

	for(i=0;i<num_of_clusters;i++){
		p[i] = (double)counters[i]/sum;
	}

	len = 0.0;
	for(i=0;i<num_of_clusters;i++){
		len+=p[i];
	}

	Hufftree<unsigned int, double> hufftree(p.begin(), p.end());
	
	len = entr = 0.0;
	for (i=0; i<p.size(); i++)
	{
		len += p[i]*hufftree.encode(i).size();
		entr += -p[i]*log(p[i]);
	}
	entr /= log(2.0);
	
	std::cout << "Huffman average length per pixel: " << len/dim << "\n\n\n";
	std::cout << "Entropy per pixel: " << entr/dim << "\n\n\n";

}
