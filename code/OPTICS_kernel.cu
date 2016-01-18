//#define num_cuda_stream (5)

__global__ void Euler_distance(float* input, float* output, int index, int size)
{
	int thread = blockIdx.y*gridDim.x*(blockDim.x*blockDim.y) + blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int total = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
	float dx, dy;
	while (thread<size)
	{
		dx = input[thread*2]-input[index*2];
		dy = input[thread*2+1]-input[index*2+1];
		output[thread] = (dx)* (dx) + (dy)* (dy);
		output[thread] = sqrtf(output[thread]);
		thread += total;
	}
	return;
}

__global__ void is_in_epsi(float* input, int* output, int index, float epsi, int size)
{
	int thread = blockIdx.y*gridDim.x*(blockDim.x*blockDim.y) + blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int total = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
	float dx, dy;
	while (thread<size)
	{
		if (input[thread]<epsi&&thread!=index)
		{ 
			atomicAdd(output, 1);		
		}
		thread += total;
	}

	return;
}

__global__ void is_in_epsi(float* input, int* output1, int* output2, int index, float epsi, int size)
{
	int thread = blockIdx.y*gridDim.x*(blockDim.x*blockDim.y) + blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int total = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
	float dx, dy;
	while (thread<size)
	{
		output2[thread] = 0;
		if (input[thread]<epsi&&thread!=index)
		{ 
			atomicAdd(output1, 1);
			output2[thread] = 1;
		}
		thread += total;
	}

	return;
}

__global__ void copy_to_euler_neighbour(int* input1, float* input2, float* output, int size, int* length)
{
	int thread = blockIdx.y*gridDim.x*(blockDim.x*blockDim.y) + blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int total = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
	while (thread<size)
	{
		if (!thread&&input1[thread])
		{
			output[(input1[thread]-1)] = input2[thread];
			output[length[0]+(input1[thread])-1] = thread;
		}
		if(thread&&input1[thread]!=input1[thread-1])
		{
			output[(input1[thread]-1)] = input2[thread];
			output[length[0]+(input1[thread])-1] = thread;

		}
		thread += total;
	}

	return;
}

__global__ void set_core_distance(float* input, float* output, int index, float epsi, int size)
{
	//int thread = blockIdx.y*gridDim.x*(blockDim.x*blockDim.y) + blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int thread = blockIdx.x;
	int j = 0;
	if (thread+index >=size)
	{
		return;
	}
	float* d_input = input +thread* size;
	float* d_output = output + 2*thread*size;
	if (threadIdx.x==0)
	{
		for (int i=0;i<size;i++)
		{
			if (d_input[i]<epsi&&d_input[i])
			{
				d_output[j*2] = i;
				d_output[j*2+1] = d_input[i];
				j++;
			}
		}
		for (int i=0;i<j;i++)
		{
			for (int k=i;k<j;k++)
			{
				if(d_output[i*2+1]>d_output[k*2+1])
				{
					float a1,a2;
					a1 = d_output[i*2];
					a2 = d_output[i*2+1];
					d_output[i*2] = d_output[k*2];
					d_output[i*2+1] = d_output[k*2+1];
					d_output[k*2] = a1;
					d_output[k*2+1] = a2;
				}
			}
		}
	}
	return;
}

__global__ void break_cycle_step1(int* cyc_list, int* vertex, int* mst_edge, int* edge_perv, int* group_id, float* weight, int* edge,float* cyc_delta)
{
	int tid = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*(blockDim.x*blockDim.y) + blockIdx.y*gridDim.x*(blockDim.x*blockDim.y);
	int cyc_id = cyc_list[tid];
	int start, end, present, temp;
	present = start = vertex[cyc_id];
	end = vertex[cyc_id] + edge_perv[cyc_id];
	if(mst_edge[cyc_id!=-1])
	{
		while (mst_edge[cyc_id]!=edge[present]){present++;}
		
		int i=present+1;

		for (;i<end;i++)
		{
			int par_id = edge[i];

			if (group_id[par_id]!=group_id[cyc_id])
			{
				cyc_delta[tid*2] = weight[i] - weight[present];
				cyc_delta[tid*2 + 1] = i;//¼ÇÂ¼edgeÉÏµÄÆ«ÒÆ
				break;
			}

		}

		if (i==end)
		{
			cyc_delta[tid*2] = -weight[present];
			cyc_delta[tid*2 + 1] = -1;
		}
	}
	return ;
}


__global__ void break_cycle_step2(int* cyc_list, int* edge, int* mst_edge, float* cyc_delta, int size)
{
	int tid = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*(blockDim.x*blockDim.y) + blockIdx.y*gridDim.x*(blockDim.x*blockDim.y);
	int n0min=0;
	int n0max = 0;
	int i=0;
	while (cyc_delta[i*2]<0&&i<size){i++;}
	if (i==size)
	{
		n0min = -1;
	}else
	{
		n0min = i;
	}
	for(i=0;i<size;i++)
	{
		if (cyc_delta[i*2]>0&&cyc_delta[n0min*2]>cyc_delta[i*2])
		{
			n0min = i;
		}
		if (cyc_delta[i*2]<0&&cyc_delta[n0max*2]>cyc_delta[i*2])
		{
			n0max = i;
		}
	}

	if (n0min==-1)//É¾±ß;
	{
		int delete_par_id = cyc_list[n0max];
		mst_edge[delete_par_id] = -1;
	}
	if(n0min!=-1)//É¾±ß¼Ó±ß;
	{
		int replace_par_id = cyc_delta[2*n0min+1];
		int delete_par_id = cyc_list[n0min];
		mst_edge[delete_par_id] = edge[replace_par_id];
	}
	return;
}