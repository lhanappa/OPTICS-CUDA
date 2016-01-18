#include <cuda.h>
#include <cutil.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void Euler_distance(float* input, float* output, int index, int size);
__global__ void is_in_epsi(float* input, int* output, int index, float epsi, int size);
__global__ void is_in_epsi(float* input, int* output1, int* output2, int index, float epsi, int size);
__global__ void copy_to_euler_neighbour(int* input1, float* input2, float* output, int size, int* length);
__global__ void set_core_distance(float* input, float* output, int index, float epsi, int size);

//选择换环中每点可用的最小边;
__global__ void break_cycle_step1(int* cyc_list, int* vertex, int* mst_edge, int* edge_perv, int* group_id, float* weight, int* edge, float* cyc_delta);
//计算增量最小，删边加边;
__global__ void break_cycle_step2(int* cyc_list, int* edge, int* mst_edge, float* cyc_delta, int size);
//d_points_Euler_dis, d_points_Euler_neighbour