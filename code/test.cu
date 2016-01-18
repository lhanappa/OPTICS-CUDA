
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
using namespace std;

#include "OPTICS.cuh"


int main(int argc, char* argv[])
{
	
	//float epsi = atof(argv[1]);
	//int MinPt = atoi(argv[2]);
	//OPTICS test(epsi, MinPt);
	//char matname[5] = "mat0";
	//test.set_max_stream_num(atoi(argv[3]));
	//matname[3] = *argv[4];

	float epsi = 0.4;
	int MinPt = 3;
	OPTICS test(epsi, MinPt);
	char matname[5] = "m";

	test.run("matlab.mat", matname);
	//test.output_result(atoi(argv[4]));
    return 0;
}

