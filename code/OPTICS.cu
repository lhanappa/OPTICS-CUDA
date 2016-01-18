#include "OPTICS.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <mat.h>
#include <mex.h>
#include "OPTICS_kernel.cuh"
#include <string>
#include <iostream>
#include <fstream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

using namespace std;
#pragma comment(lib,"libeng.lib")
#pragma comment(lib,"libmex.lib")
#pragma comment(lib,"libmx.lib")
#pragma comment(lib,"libmat.lib")
#pragma comment(lib,"libmx.lib")


#ifdef _CHAR16T
#define CHAR16_T
#endif

#define DEBUG 0
#define DEBUGT 0
#define TIME_RECORD 1
#define MAX(x,y) ((x)>=(y)?(x):(y))
#define MIN(x,y) ((x)<=(y)?(x):(y))

OPTICS::OPTICS(float E, int Min)
{
	Epsi = E;
	MinPt = Min;
	MAX_stream_num = 1;
	return ;
}

OPTICS::~OPTICS()
{
	CUDA_SAFE_CALL(cudaFree(d_vertex));
	CUDA_SAFE_CALL(cudaFree(d_edge_perv));
	CUDA_SAFE_CALL(cudaFree(d_mst_edge));
	CUDA_SAFE_CALL(cudaFree(d_group_id));
	CUDA_SAFE_CALL(cudaFree(d_edge));
	CUDA_SAFE_CALL(cudaFree(d_weight));
	return ;
}

void OPTICS::run(char* filename, char* matname)
{
	read_input_file(filename, matname);

	initial_parameter();
	//cout << "Euler_core_distance:\t";
	Euler_core_distance();

	//cout << "reachablity_distance:\t";
	reachability_distance();

	//cout << "transform:\t";
	transform_host_device();

	initial_mst_graph();

	//cout << "scan_graph:\t";
	scan_graph();

	//output_result();

	do 
	{
		//cout << "break_cycle:\t";
		break_cycle();

		//cout << "update:\t";
		update();

		//cout << "scan_graph:\t";
		scan_graph();

	} while (!stop_condition());

	output_result();

	return;
}

void OPTICS::read_input_file(char* filename, char* matname)
{
	MATFile *pmatFile = NULL;  
	mxArray *pMxArray = NULL;  

	// 读取.mat文件（例：mat文件名为"initUrban.mat"，其中包含"initA"）   
	double *initA;  

	pmatFile = matOpen(filename,"r");  
	pMxArray = matGetVariable(pmatFile, matname);  
	initA = (double*) mxGetData(pMxArray);  
	size_t M = mxGetM(pMxArray);  
	size_t N = mxGetN(pMxArray); 
	double *A = new double [(M)*N];
	for (int i=0; i<M; i++)
	{
		for (int j=0; j<N; j++)   
		{
			A[(i)*N+j] = initA[M*j+i];  
		}
	}

	matClose(pmatFile);  
	mxFree(initA);  
	//// 生成.mat文件   
	//double *outA = new double[M*N];  
	//for (int i=0; i<M; i++) 
	//{
	//	for (int j=0; j<N; j++)
	//	{
	//		outA[M*j+i] = A[(i)*N+j]; 
	//	}
	//}
	//pmatFile = matOpen("A.mat","w");  
	//mxSetData(pMxArray, outA);  
	//matPutVariable(pmatFile, "A", pMxArray);  
	//matClose(pmatFile);  
	point_num = M;
	points_input_list = new based_point[M];
	for (int i=0;i<M;i++)
	{
		this->points_input_list[i].x = (float)A[i*N];
		this->points_input_list[i].y = (float)A[i*N+ 1];
		this->points_input_list[i].Epsi_num = 0;
		this->points_input_list[i].euler_list = NULL;
		this->points_input_list[i].flag = false;
	}
	graph_nodes = new ReachabilityDistance_points[M];
	for (int i=0;i<M;i++)
	{
		this->graph_nodes[i].Epsi_num = 0;
		this->graph_nodes[i].core_num = 0;
		this->graph_nodes[i].list_RD.list = NULL;
		this->graph_nodes[i].group_id = -1;
		this->graph_nodes[i].vertex_par.vertex_id = -1;
		this->graph_nodes[i].vertex_par.weight = -1;
		this->graph_nodes[i].core_distance = -1;
	}
	return;
}

void OPTICS::initial_parameter()
{
	num_cuda_stream = MIN(MAX_stream_num,point_num);
	cudastream = new cudaStream_t[num_cuda_stream];
	h_points_Euler_xy = new float[point_num*2];
	h_points_num_neighbour = new int[point_num];
	h_points_Euler_neighbour = new float[num_cuda_stream*point_num*2];
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_points_Euler_xy, sizeof(float)*point_num*2));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_points_Euler_dis, sizeof(float)*point_num*num_cuda_stream));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_points_num_neighbour, sizeof(int)*num_cuda_stream));
	CUDA_SAFE_CALL(cudaMemset(d_points_num_neighbour, 0, sizeof(int)*num_cuda_stream));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_points_Euler_neighbour, sizeof(float)*num_cuda_stream*point_num*2));
	return;
}

void OPTICS::Euler_core_distance()
{
#ifdef TIME_RECORD
	cudaEvent_t start_t,stop_t, stop_t1, stop_t2, stop_t3, stop_t4;
	float costtime;
	CUDA_SAFE_CALL(cudaEventCreate(&start_t));
	CUDA_SAFE_CALL(cudaEventCreate(&stop_t));
	CUDA_SAFE_CALL(cudaEventCreate(&stop_t1));
	CUDA_SAFE_CALL(cudaEventCreate(&stop_t2));
	CUDA_SAFE_CALL(cudaEventCreate(&stop_t3));
	CUDA_SAFE_CALL(cudaEventCreate(&stop_t4));
	CUDA_SAFE_CALL(cudaEventRecord(start_t,0));
#endif
	dim3 gridsize(1,1,1);
	dim3 blocksize(1024,1,1);
	int* d_temp_euler_epsi;
	int* h_temp_euler_epsi;
	h_temp_euler_epsi = new int[num_cuda_stream*point_num];
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_temp_euler_epsi, sizeof(int)*num_cuda_stream*point_num));
	for (int i=0;i<point_num;i++)
	{
		h_points_Euler_xy[i*2] = points_input_list[i].x;
		h_points_Euler_xy[i*2+1] = points_input_list[i].y;
	}
	CUDA_SAFE_CALL(cudaMemcpy((void*)d_points_Euler_xy, (void*)h_points_Euler_xy, sizeof(float)*point_num*2, cudaMemcpyHostToDevice));

	int* offset1 = new int[num_cuda_stream];
	int* offset2 = new int[num_cuda_stream];
	int* offset3 = new int[num_cuda_stream];
	for (int i=0;i<num_cuda_stream;i++)
	{
		CUDA_SAFE_CALL(cudaStreamCreate(cudastream+i));
		offset1[i] = point_num*i;
		offset2[i] = i;
		offset3[i] = point_num*i*2;
	}


	for (int j=0;j<point_num;j+=num_cuda_stream)
	{
#if DEBUGT
		CUDA_SAFE_CALL(cudaEventRecord(stop_t1,0));
#endif
		for (int i=0;(i<num_cuda_stream)&&(j+i<point_num);i++)
		{
			Euler_distance<<<gridsize, blocksize, 0, cudastream[i]>>>(d_points_Euler_xy, d_points_Euler_dis+offset1[i], j+i, point_num);
		}
#if DEBUG
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		float* temp_Euler_dis = new float[point_num*num_cuda_stream];
		CUDA_SAFE_CALL(cudaMemcpy(temp_Euler_dis, d_points_Euler_dis, sizeof(float)*point_num*num_cuda_stream, cudaMemcpyDeviceToHost));
		{
			ofstream fout("Euler_dis.txt");
			for (int x=0;x<num_cuda_stream;x++)
			{
				for (int xx=0;xx<point_num;xx++)
				{
					fout << temp_Euler_dis[x*point_num+xx] << '\t';
				}
				fout << endl;
			}
			fout.close();
		}
		delete[] temp_Euler_dis;
#endif

#if DEBUGT
		CUDA_SAFE_CALL(cudaEventRecord(stop_t2,0));
		CUDA_SAFE_CALL(cudaEventSynchronize(stop_t2));
		CUDA_SAFE_CALL(cudaEventElapsedTime(&costtime,stop_t1,stop_t2));
		{cout << costtime << "\n";}

#endif

		for (int i=0;(i<num_cuda_stream)&&(j+i<point_num);i++)
		{
			//blocksize.x = 10;
			//is_in_epsi<<<gridsize, blocksize, 0, cudastream[i]>>>(d_points_Euler_dis+offset1[i], d_points_num_neighbour+i, j+i, Epsi, point_num);
			is_in_epsi<<<gridsize, blocksize, 0, cudastream[i]>>>(d_points_Euler_dis+offset1[i], d_points_num_neighbour+i, d_temp_euler_epsi+offset1[i], j+i, Epsi, point_num);
			CUDA_SAFE_CALL(cudaMemcpyAsync(	(void*)(h_temp_euler_epsi+offset1[i]), 
											(void*)(d_temp_euler_epsi+offset1[i]), 
											sizeof(int)*point_num, 
											cudaMemcpyDeviceToHost, 
											cudastream[i]));
		}
#if DEBUG
		{
			ofstream fout("h_temp_euler_epsi0.txt");
			for (int i=0;i<num_cuda_stream;i++)
			{
				for (int j=0;j<point_num;j++)
				{
					fout << h_temp_euler_epsi[i*point_num+j] << "\t";
				}
				fout << endl;
			}
			fout.close();
		}

#endif		
		for(int i=0;(i<num_cuda_stream)&&(j+i<point_num);i++)
		{
			thrust::inclusive_scan(h_temp_euler_epsi+offset1[i], h_temp_euler_epsi+offset1[i]+point_num, h_temp_euler_epsi+offset1[i]);
			CUDA_SAFE_CALL(cudaMemcpyAsync(	(void*)(d_temp_euler_epsi+offset1[i]), 
											(void*)(h_temp_euler_epsi+offset1[i]), 
											sizeof(int)*point_num, 
											cudaMemcpyHostToDevice, 
											cudastream[i]));
		}
#if DEBUG
		{
			ofstream fout("h_temp_euler_epsi.txt");
			for (int i=0;i<num_cuda_stream;i++)
			{
				for (int j=0;j<point_num;j++)
				{
					fout << h_temp_euler_epsi[i*point_num+j] << "\t";
				}
				fout << endl;
			}
			fout.close();
		}

#endif
		for(int i=0;(i<num_cuda_stream)&&(j+i<point_num);i++)
		{
			copy_to_euler_neighbour<<<gridsize, blocksize>>>(d_temp_euler_epsi+offset1[i], d_points_Euler_dis+offset1[i], d_points_Euler_neighbour+offset3[i], point_num, d_points_num_neighbour+i);
		}
#if DEBUG
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		int* temp_Neighbour_num = new int[num_cuda_stream];
		CUDA_SAFE_CALL(cudaMemcpy(temp_Neighbour_num, d_points_num_neighbour, sizeof(int)*num_cuda_stream, cudaMemcpyDeviceToHost));
		{
			ofstream fout("Neighbour_num.txt");
			for (int x=0;x<num_cuda_stream;x++)
			{	
				fout << temp_Neighbour_num[x] << '\t';
			}
			fout.close();
		}
		delete[] temp_Neighbour_num;
#endif

#if DEBUGT
		CUDA_SAFE_CALL(cudaEventRecord(stop_t3,0));
		CUDA_SAFE_CALL(cudaEventSynchronize(stop_t3));
		CUDA_SAFE_CALL(cudaEventElapsedTime(&costtime,stop_t2,stop_t3));
		{cout << costtime << "\n";}

#endif

		for (int i=0;(i<num_cuda_stream)&&(j+i<point_num);i++)
		{
			gridsize.x = 1;
			//set_core_distance<<<gridsize, blocksize, 0, cudastream[i]>>>(d_points_Euler_dis+offset1, d_points_Euler_neighbour+offset2, j+i, Epsi, point_num);
			int offset1 = i+j;
			int offset2 = i*point_num*2;
			CUDA_SAFE_CALL(	cudaMemcpyAsync((void*)(h_points_num_neighbour+offset1), 
				(void*)(d_points_num_neighbour+i), 
				sizeof(float), 
				cudaMemcpyDeviceToHost, 
				cudastream[i]));
			CUDA_SAFE_CALL(	cudaMemcpyAsync((void*)(h_points_Euler_neighbour+offset3[i]), 
											(void*)(d_points_Euler_neighbour+offset3[i]), 
											sizeof(float)*point_num*2, 
											cudaMemcpyDeviceToHost, 
											cudastream[i]));				
		}

		//CUDA_SAFE_CALL(cudaThreadSynchronize());
		//gridsize.x = num_cuda_stream;
		//set_core_distance<<<gridsize, blocksize>>>(d_points_Euler_dis, d_points_Euler_neighbour, j, Epsi, point_num);
		//CUDA_SAFE_CALL(cudaThreadSynchronize());
#if DEBUGT
		CUDA_SAFE_CALL(cudaEventRecord(stop_t4,0));
		CUDA_SAFE_CALL(cudaEventSynchronize(stop_t4));
		CUDA_SAFE_CALL(cudaEventElapsedTime(&costtime,stop_t3,stop_t4));
		{cout << costtime << "\n";}

#endif
#if DEBUG
		{
			ofstream fout("h_points_Euler_neighbour0.txt");
			for (int i=0;i<num_cuda_stream;i++)
			{
				for (int j=0;j<point_num*2;j++)
				{
					fout << h_points_Euler_neighbour[i*point_num*2+j] << "\t";
				}
				fout << endl;
			}
			fout.close();
		}
		

#endif
		for (int i=0;(i<num_cuda_stream)&&(j+i<point_num);i++)
		{
			thrust::sort_by_key(h_points_Euler_neighbour+offset3[i], h_points_Euler_neighbour+offset3[i] + h_points_num_neighbour[j+i], h_points_Euler_neighbour+offset3[i]+h_points_num_neighbour[j+i]);
		}

		CUDA_SAFE_CALL(cudaMemset(d_points_num_neighbour, 0, sizeof(int)*num_cuda_stream));
		
#if DEBUG
		{
			ofstream fout("h_points_Euler_neighbour.txt");
			for (int i=0;i<num_cuda_stream;i++)
			{
				for (int j=0;j<point_num*2;j++)
				{
					fout << h_points_Euler_neighbour[i*point_num*2+j] << "\t";
				}
				fout << endl;
			}
			fout.close();
		}


#endif

		for (int i=0;(i<num_cuda_stream)&&(j+i<point_num);i++)
		{
			//CUDA_SAFE_CALL(cudaThreadSynchronize());
			points_input_list[j+i].Epsi_num = h_points_num_neighbour[i+j];
			points_input_list[j+i].euler_list = new neighbour_point [points_input_list[j+i].Epsi_num];
			for (int k=0;k<points_input_list[j+i].Epsi_num;k++)
			{
				points_input_list[j+i].euler_list[k].weight = h_points_Euler_neighbour[i*point_num*2 + k];
				points_input_list[j+i].euler_list[k].vertex_id = h_points_Euler_neighbour[i*point_num*2 + points_input_list[j+i].Epsi_num + k];
			}
		}
	}
	CUDA_SAFE_CALL(cudaThreadSynchronize());

#if DEBUG
	{
		ofstream fout("points_input_list.txt");
		for (int i=0;i<point_num;i++)
		{

			fout << points_input_list[i].Epsi_num << '\t' << h_points_num_neighbour[i]<< endl << '\t';
			for (int j=0;j<points_input_list[i].Epsi_num;j++)
			{
				fout << points_input_list[i].euler_list[j].vertex_id << '\t'<< points_input_list[i].euler_list[j].weight << '\t';
			}
			fout << endl;
		}
		fout.close();
	}	
#endif
	for (int i=0;i<num_cuda_stream;i++)
	{
		CUDA_SAFE_CALL(cudaStreamDestroy(cudastream[i]));
	}
	if (cudastream!=NULL)
	{
		delete [] cudastream;
		cudastream = NULL;
	}
	delete[] h_temp_euler_epsi;
	CUDA_SAFE_CALL(cudaFree(d_temp_euler_epsi));
#ifdef TIME_RECORD
	CUDA_SAFE_CALL(cudaEventRecord(stop_t,0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop_t));
	CUDA_SAFE_CALL(cudaEventElapsedTime(&costtime,start_t,stop_t));
	{cout << costtime << "\n";}

#endif
	return;
}

void OPTICS::reachability_distance()
{
#ifdef TIME_RECORD
	cudaEvent_t start_t,stop_t;
	float costtime;
	CUDA_SAFE_CALL(cudaEventCreate(&start_t));
	CUDA_SAFE_CALL(cudaEventCreate(&stop_t));

	CUDA_SAFE_CALL(cudaEventRecord(start_t,0));
#endif
	int list_RD_num = 0;
	int* list_RD_id;
	for (int i=0;i<point_num;i++)
	{
		graph_nodes[i].Epsi_num = points_input_list[i].Epsi_num;

		if (graph_nodes[i].Epsi_num>=MinPt)
		{					
			graph_nodes[i].core_distance = (points_input_list[i].euler_list[MinPt-1].weight);
		}

		list_RD_id = new int[graph_nodes[i].Epsi_num];

		for (int j=0;j<graph_nodes[i].Epsi_num;j++)
		{
			int id = points_input_list[i].euler_list[j].vertex_id;
			if (points_input_list[id].Epsi_num>=MinPt)
			{
				list_RD_id[list_RD_num++] = j;
			}
		}
		graph_nodes[i].list_RD.list = new neighbour_point[list_RD_num];
		graph_nodes[i].core_num = list_RD_num;
		for (int j=0;j<list_RD_num;j++)
		{
			int idx = points_input_list[i].euler_list[list_RD_id[j]].vertex_id;
			graph_nodes[i].list_RD.list[j].vertex_id = idx;
			graph_nodes[i].list_RD.list[j].weight = MAX(points_input_list[i].euler_list[list_RD_id[j]].weight, points_input_list[idx].euler_list[MinPt-1].weight);
		}

		for (int j=0;j<list_RD_num;j++)
		{
			for (int k=j+1;k<list_RD_num;k++)
			{
				neighbour_point temp;
				if (graph_nodes[i].list_RD.list[j].weight>graph_nodes[i].list_RD.list[k].weight)
				{
					temp.vertex_id = graph_nodes[i].list_RD.list[k].vertex_id;
					temp.weight = graph_nodes[i].list_RD.list[k].weight;

					graph_nodes[i].list_RD.list[k].vertex_id = graph_nodes[i].list_RD.list[j].vertex_id;
					graph_nodes[i].list_RD.list[k].weight = graph_nodes[i].list_RD.list[j].weight;

					graph_nodes[i].list_RD.list[j].vertex_id = temp.vertex_id;
					graph_nodes[i].list_RD.list[j].weight = temp.weight;
				}
			}
		}

		if (graph_nodes[i].core_num)
		{
			graph_nodes[i].vertex_par.vertex_id = graph_nodes[i].list_RD.list[0].vertex_id;
			graph_nodes[i].vertex_par.weight = graph_nodes[i].list_RD.list[0].weight;
		} 
		
		delete[] list_RD_id;
		list_RD_num = 0;
	}

#if DEBUG
	{
		ofstream fout("points_reachability_list.txt");
		for (int i=0;i<point_num;i++)
		{

			fout << graph_nodes[i].Epsi_num << '\t' << graph_nodes[i].core_num << endl << '\t';
			for (int j=0;j<graph_nodes[i].core_num;j++)
			{
				fout << graph_nodes[i].list_RD.list[j].vertex_id << '\t'<< graph_nodes[i].list_RD.list[j].weight << '\t';
			}
			fout << endl;
		}
		fout.close();
	}	
#endif

#ifdef TIME_RECORD
	CUDA_SAFE_CALL(cudaEventRecord(stop_t,0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop_t));
	CUDA_SAFE_CALL(cudaEventElapsedTime(&costtime,start_t,stop_t));
	//cout << "P:\t" << costtime << endl;
	//if(!(deId/m_nthread)){cout << setprecision(6) << costtime << "\t";}
	{cout << costtime << "\n";}//if(!(deId/m_nthread)){cout << setprecision(6) << costtime << "\t";}

#endif
	return;
}

void OPTICS::transform_host_device()
{
#ifdef TIME_RECORD
	cudaEvent_t start_t,stop_t;
	float costtime;
	CUDA_SAFE_CALL(cudaEventCreate(&start_t));
	CUDA_SAFE_CALL(cudaEventCreate(&stop_t));

	CUDA_SAFE_CALL(cudaEventRecord(start_t,0));
#endif
	h_vertex = new int [point_num];
	h_edge_perv = new int [point_num];
	//h_flag = new int [point_num];
	h_mst_edge = new int [point_num];
	h_group_id = new int [point_num];
	cudaMemset(h_group_id, -1, sizeof(int));
	int num_edges=0;
	for (int i=0;i<point_num;i++)
	{
		if (graph_nodes[i].core_num)
		{
			h_vertex[i] = num_edges;
		}else
		{
			h_vertex[i] = -1;
		}
		h_edge_perv[i] = graph_nodes[i].core_num;
		num_edges += graph_nodes[i].core_num;

		//if (graph_nodes[i].Epsi_num>=MinPt)
		//{
			//h_flag[i] = 1;
			h_mst_edge[i] = graph_nodes[i].vertex_par.vertex_id;
		//} 
		//else
		//{
		//	//h_flag[i] = 0;
		//	h_mst_edge[i] = -1;
		//}
	}
	h_edge = new int [num_edges];
	h_weight = new float [num_edges];
	int k=0;
	for (int i=0;i<point_num;i++)
	{
		for (int j=0;j<graph_nodes[i].core_num;j++)
		{
			h_edge[k] = graph_nodes[i].list_RD.list[j].vertex_id;
			h_weight[k] = graph_nodes[i].list_RD.list[j].weight;
			k++;
		}
	}

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_vertex, sizeof(int)*point_num));
	CUDA_SAFE_CALL(cudaMemcpy(d_vertex, h_vertex, sizeof(int)*point_num, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_edge_perv, sizeof(int)*point_num));
	CUDA_SAFE_CALL(cudaMemcpy(d_edge_perv, h_edge_perv, sizeof(int)*point_num, cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMalloc((void**)&d_flag, sizeof(int)*point_num));
	//CUDA_SAFE_CALL(cudaMemcpy(d_flag, h_flag, sizeof(int)*point_num, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_mst_edge, sizeof(int)*point_num));
	CUDA_SAFE_CALL(cudaMemcpy(d_mst_edge, h_mst_edge, sizeof(int)*point_num, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_group_id, sizeof(int)*point_num));
	CUDA_SAFE_CALL(cudaMemcpy(d_group_id, h_group_id, sizeof(int)*point_num, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_edge, sizeof(int)*num_edges));
	CUDA_SAFE_CALL(cudaMemcpy(d_edge, h_edge, sizeof(int)*num_edges, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_weight, sizeof(float)*num_edges));
	CUDA_SAFE_CALL(cudaMemcpy(d_weight, h_weight, sizeof(float)*num_edges, cudaMemcpyHostToDevice));

	
#ifdef TIME_RECORD
		CUDA_SAFE_CALL(cudaEventRecord(stop_t,0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop_t));
	CUDA_SAFE_CALL(cudaEventElapsedTime(&costtime,start_t,stop_t));
	//cout << "P:\t" << costtime << endl;
	//if(!(deId/m_nthread)){cout << setprecision(6) << costtime << "\t";}
	{cout << costtime << "\n";}//if(!(deId/m_nthread)){cout << setprecision(6) << costtime << "\t";}

#endif	
		return ;
}

void OPTICS::initial_mst_graph()
{
	this->cyc_num = 0;
	this->cyc_list_head = NULL;
	this->d_cyc_list = NULL;
	return;
}

void OPTICS::scan_graph()
{
#ifdef TIME_RECORD
	cudaEvent_t start_t,stop_t;
	float costtime;
	CUDA_SAFE_CALL(cudaEventCreate(&start_t));
	CUDA_SAFE_CALL(cudaEventCreate(&stop_t));

	CUDA_SAFE_CALL(cudaEventRecord(start_t,0));
#endif
	int group_id = 0; //记录分组的索引;
	int point_count_num = 0;//遍历点的数量;
	int onegroup_i = 0;//一组中某点的id或其父节点id;
	int onegroup_num = 0;//一组中点的数量;
	int cyc_num_temp = 0;//记录圆的数目;
	int** temp_cyc_list = new int* [point_num];
	int* onegroup_cyc_num = new int [point_num];
	int* group_list = new int[point_num];
	memset(group_list, -1, sizeof(int)*point_num);
	for (int i=0;i<point_num;i++)
	{
		temp_cyc_list[i] = new int [point_num];
	}
	while(point_count_num<point_num)
	{
		//output_result();
		if (graph_nodes[onegroup_i].group_id==-1)//发现新点
		{		
			group_list[onegroup_num++] = onegroup_i;			
			graph_nodes[onegroup_i].group_id = group_id;
			point_count_num++;
			if (graph_nodes[onegroup_i].vertex_par.vertex_id!=-1)
			{
				onegroup_i = graph_nodes[onegroup_i].vertex_par.vertex_id;
			}else
			{
				onegroup_i = ++group_id;
				memset(group_list, -1, sizeof(int)*point_num);
				onegroup_num = 0;
				while (graph_nodes[onegroup_i].group_id!=-1&&onegroup_i<point_num)
				{
					onegroup_i++;
				}
			}
		} 
		else if (graph_nodes[onegroup_i].group_id == group_id)//发现环;
		{
			record_cycle(group_list, onegroup_num, onegroup_i, temp_cyc_list[cyc_num_temp], &(onegroup_cyc_num[cyc_num_temp]));
			cyc_num_temp++;
			group_id++;
			onegroup_i=group_id;
			while (graph_nodes[onegroup_i].group_id!=-1&&onegroup_i<point_num)
			{
				onegroup_i++;
			}
			memset(group_list, -1, sizeof(int)*point_num);
			onegroup_num = 0;
		}
		else if (graph_nodes[onegroup_i].group_id < group_id)//发现原有支路需要合并
		{
			merge_chain(group_list, onegroup_num, graph_nodes[onegroup_i].group_id);
			onegroup_i = group_id;
			while (graph_nodes[onegroup_i].group_id!=-1&&onegroup_i<point_num)
			{
				onegroup_i++;
			}
			memset(group_list, -1, sizeof(int)*point_num);
			onegroup_num = 0;
		}
	}
	cyc_num = cyc_num_temp;

	d_cyc_list = new int*[cyc_num];
	cycle* temp = cyc_list_head;
	for (int i=0;i<cyc_num;i++)
	{
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_cyc_list[i], sizeof(int)*(temp->num_node)));
		CUDA_SAFE_CALL(cudaMemcpy(d_cyc_list[i], temp_cyc_list[i], sizeof(int)*onegroup_cyc_num[i], cudaMemcpyHostToDevice));
		temp = temp->next;
	}
	for (int i=0;i<point_num;i++)
	{
		delete [] temp_cyc_list[i];
	}
	delete [] temp_cyc_list;
	delete [] group_list;
	delete [] onegroup_cyc_num;

	for (int i=0;i<point_num;i++)
	{
		h_group_id[i] = graph_nodes[i].group_id;
	}
	CUDA_SAFE_CALL(cudaMemcpy(d_group_id, h_group_id, sizeof(int)*point_num, cudaMemcpyHostToDevice));
#ifdef TIME_RECORD
	CUDA_SAFE_CALL(cudaEventRecord(stop_t,0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop_t));
	CUDA_SAFE_CALL(cudaEventElapsedTime(&costtime,start_t,stop_t));
	//cout << "P:\t" << costtime << endl;
	//if(!(deId/m_nthread)){cout << setprecision(6) << costtime << "\t";}
	{cout << costtime << "\n";}//if(!(deId/m_nthread)){cout << setprecision(6) << costtime << "\t";}

#endif
	return;
}

void OPTICS::record_cycle(int* list, int size, int cycle_index, int* temp_cyc_list, int* onegroup_cyc_num)
{
	int i=0;
	for (;i<size;i++)
	{
		if(list[i]==cycle_index)
			break;
	}
	memcpy(temp_cyc_list, list+i, sizeof(int)*(size-i));
	int num = size-i;
	*onegroup_cyc_num = num;
	cycle* temp;
	
	if (cyc_list_head == NULL)
	{
		cyc_list_head = new cycle;
		temp = cyc_list_head;
	}else
	{
		temp = cyc_list_head;
		while (temp->next!=NULL)
		{
			temp = temp->next;
		}
		temp->next = new cycle;
		temp = temp->next;
	}
	
	
	temp->next = NULL;
	temp->num_node = num;
	cycle_list* t_list = temp->headnode;
	if (num)
	{
		temp->headnode = new cycle_list;
		t_list = temp->headnode;
		t_list->next = NULL;
		t_list->node_id = temp_cyc_list[0];
	}
	for (int i=1;i<num;i++)
	{
		t_list->next = new cycle_list;
		t_list = t_list->next;
		t_list->node_id = temp_cyc_list[i];
		t_list->next = NULL;
	}
	return;
}

void OPTICS::merge_chain(int* list, int size, int chain_index)
{
	for (int i=0;i<size;i++)
	{
		graph_nodes[list[i]].group_id = chain_index;
	}
	return;
}

void OPTICS::break_cycle()
{
#if DEBUG
	{
		ofstream fout("mst_update0.txt");
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		for (int i=0;i<point_num;i++)
		{
			fout << h_mst_edge[i] << endl;
		}
		fout.close();		
	}
#endif

#ifdef TIME_RECORD
	cudaEvent_t start_t,stop_t;
	float costtime;
	CUDA_SAFE_CALL(cudaEventCreate(&start_t));
	CUDA_SAFE_CALL(cudaEventCreate(&stop_t));

	CUDA_SAFE_CALL(cudaEventRecord(start_t,0));
#endif
	if (cudastream!=NULL)
	{
		delete [] cudastream;
		cudastream = NULL;
	}
	num_cuda_stream = cyc_num;
	cudastream = new cudaStream_t[num_cuda_stream];
	dim3 gridsize(1,1,1);
	dim3 blocksize(1,1,1);
	for (int i=0;i<num_cuda_stream;i++)
	{
		CUDA_SAFE_CALL(cudaStreamCreate(cudastream+i));
	}

	float** d_cyc_delta = new float*[cyc_num];
	cycle* temp = cyc_list_head;

	for (int i=0;i<num_cuda_stream;i++)
	{
		CUDA_SAFE_CALL(cudaMalloc((void**)&(d_cyc_delta[i]), sizeof(float)*temp->num_node*2));	
		gridsize.x = temp->num_node;
		break_cycle_step1<<<gridsize, blocksize, 0, cudastream[i]>>>(d_cyc_list[i], d_vertex, d_mst_edge, d_edge_perv, d_group_id, d_weight, d_edge, d_cyc_delta[i]);
		temp = temp->next;
	}

#if DEBUG
	{
		ofstream fout("cyc_delta.txt");
		temp = cyc_list_head;
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		float* temp_delta;
		for (int i=0;i<num_cuda_stream;i++)
		{
			temp_delta = new float[temp->num_node*2];
			CUDA_SAFE_CALL(cudaMemcpy(temp_delta, d_cyc_delta[i], temp->num_node*2*sizeof(float), cudaMemcpyDeviceToHost));
			for (int j=0;j<temp->num_node*2;j++)
			{
				fout << temp_delta[j] << '\t';
			}
			fout << endl;
			delete[] temp_delta;
			temp = temp->next;
		}
		fout.close();		
	}
#endif
	temp = cyc_list_head;
	for (int i=0;i<num_cuda_stream;i++)
	{
		gridsize.x = temp->num_node;
		break_cycle_step2<<<gridsize, blocksize, 0, cudastream[i]>>>(d_cyc_list[i], d_edge, d_mst_edge, d_cyc_delta[i], temp->num_node);
		temp = temp->next;
	}

	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy(h_mst_edge, d_mst_edge, sizeof(int)*point_num, cudaMemcpyDeviceToHost));

#if DEBUG
	{
		ofstream fout("mst_update.txt");
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		for (int i=0;i<point_num;i++)
		{
			fout << h_mst_edge[i] << endl;
		}
		fout.close();		
	}
#endif

	for (int i=0;i<num_cuda_stream;i++)
	{
		CUDA_SAFE_CALL(cudaStreamDestroy(cudastream[i]));
	}
#ifdef TIME_RECORD
	CUDA_SAFE_CALL(cudaEventRecord(stop_t,0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop_t));
	CUDA_SAFE_CALL(cudaEventElapsedTime(&costtime,start_t,stop_t));
	//cout << "P:\t" << costtime << endl;
	//if(!(deId/m_nthread)){cout << setprecision(6) << costtime << "\t";}
	{cout << costtime << "\n";}//if(!(deId/m_nthread)){cout << setprecision(6) << costtime << "\t";}

#endif

	return;
}

void OPTICS::update()
{
#ifdef TIME_RECORD
	cudaEvent_t start_t,stop_t;
	float costtime;
	CUDA_SAFE_CALL(cudaEventCreate(&start_t));
	CUDA_SAFE_CALL(cudaEventCreate(&stop_t));

	CUDA_SAFE_CALL(cudaEventRecord(start_t,0));
#endif
	for (int i=0;i<point_num;i++)
	{
		graph_nodes[i].vertex_par.vertex_id = h_mst_edge[i];
		if (h_mst_edge[i]==-1)
		{
			graph_nodes[i].vertex_par.weight = -1;
		}else
		{
			int id;
			for(id=h_vertex[i];id<h_vertex[i]+h_edge_perv[i];id++)
			{
				if (h_edge[id]==h_mst_edge[i])
				{
					break;
				}
			}
			graph_nodes[i].vertex_par.weight = h_weight[id];
		}	
		graph_nodes[i].group_id = -1;
	}

	for (int i=0;i<cyc_num;i++)
	{
		CUDA_SAFE_CALL(cudaFree(d_cyc_list[i]));
	}

	freelist(cyc_list_head);
	cyc_list_head = NULL;
	
#ifdef TIME_RECORD
	CUDA_SAFE_CALL(cudaEventRecord(stop_t,0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop_t));
	CUDA_SAFE_CALL(cudaEventElapsedTime(&costtime,start_t,stop_t));
	//cout << "P:\t" << costtime << endl;
	//if(!(deId/m_nthread)){cout << setprecision(6) << costtime << "\t";}
	{cout << costtime << "\n";}//if(!(deId/m_nthread)){cout << setprecision(6) << costtime << "\t";}

#endif

	return;
}

bool OPTICS::stop_condition()
{
	if (cyc_num==0)
	{
		return true;
	}
	return false;
}

void OPTICS::freelist(cycle* a)
{
	if (a->next==NULL)
	{
		freelist(a->headnode);
		delete a;
		return;
	}else
	{
		freelist(a->next);
	}
	a->next = NULL;
	freelist(a->headnode);
	delete a;
	return;
}

void OPTICS::freelist(cycle_list* a)
{
	if (a->next==NULL)
	{
		delete a;
		return;
	}else
	{
		freelist(a->next);
	}
	a->next = NULL;
	delete a;
	return;
}

void OPTICS::output_result()
{
	ofstream fout("output.txt");
	for (int i=0;i<point_num;i++)
	{
		fout << i << '\t' << graph_nodes[i].vertex_par.vertex_id << '\t' << graph_nodes[i].vertex_par.weight << '\t' << graph_nodes[i].group_id << endl;
		//cout << i << '\t' << graph_nodes[i].vertex_par.vertex_id << '\t' << graph_nodes[i].vertex_par.weight << '\t' << graph_nodes[i].group_id << endl;
	}
	fout.close();
	return ; 
}

void OPTICS::output_result(int o)
{
	char filename[] = "output__.txt";
	filename[6] = o/10+48;
	filename[7] = o%10+48;
	ofstream fout(filename);
	for (int i=0;i<point_num;i++)
	{
		fout << i << '\t' << graph_nodes[i].vertex_par.vertex_id << '\t' << graph_nodes[i].vertex_par.weight << '\t' << graph_nodes[i].group_id << endl;
		//cout << i << '\t' << graph_nodes[i].vertex_par.vertex_id << '\t' << graph_nodes[i].vertex_par.weight << '\t' << graph_nodes[i].group_id << endl;
	}
	fout.close();
	return ; 
}
void OPTICS::set_max_stream_num(int num)
{
	this->MAX_stream_num = num;
	return;
}