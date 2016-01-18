#ifndef _H_OPTICS_
#define _H_OPTICS_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct cycle_list
{
	int node_id;
	cycle_list* next;
};
typedef struct cycle
{
	int num_node;
	cycle_list* headnode;
	cycle* next;
};

typedef struct neighbour_point
{
	int vertex_id;
	float weight;
};
class ReachabilityDistance
{
public:
	neighbour_point* list;
};


typedef struct based_point
{
	float x;
	float y;
	int Epsi_num;
	neighbour_point* euler_list;
	bool flag;//是否是有效点; 
};

typedef struct ReachabilityDistance_points
{
	int Epsi_num;
	int core_num;
	ReachabilityDistance list_RD;
	neighbour_point vertex_par;
	int group_id;
	float core_distance;
};

class OPTICS
{
public:
	OPTICS(float E, int Min);
	~OPTICS();
private:
	float Epsi;
	int MinPt;
	int point_num;
	//cuda Stream
	cudaStream_t* cudastream;
	int MAX_stream_num;
	int num_cuda_stream;
	//cudastream = new cudaStream_t[nstream];
	//CUDA_SAFE_CALL(cudaStreamCreate(cudastream+i));
	//CUDA_SAFE_CALL(cudaStreamDestroy(cudastream[i]));

	//input data; 
	based_point* points_input_list;
	ReachabilityDistance_points* graph_nodes;
	//Eluer distance;
	//core point;
	//neighbour list;
	float* h_points_Euler_xy;
	float* d_points_Euler_xy;
	float* d_points_Euler_dis;
	int* h_points_num_neighbour;
	int* d_points_num_neighbour;
	float* h_points_Euler_neighbour;
	float* d_points_Euler_neighbour;
	//
	//reachability_distance;
	//graph;
	int* h_vertex;
	int* d_vertex;	
	int* h_edge_perv;//每个点周围可以连接的边数，即周围核心点的数量;
	int* d_edge_perv;
	//int* h_flag;// 本身是否是core point?;
	//int* d_flag;
	int* h_mst_edge;
	int* d_mst_edge;
	int* h_group_id;
	int* d_group_id;

	int* h_edge;
	int* d_edge;
	float* h_weight;
	float* d_weight;
	//LO
	//cycle
	int cyc_num;
	cycle* cyc_list_head;
	int** d_cyc_list;
	//
private:
	void read_input_file(char* filename, char* matname);
	void initial_parameter();
	void Euler_core_distance();
	void reachability_distance();
	void transform_host_device();//将点和边的信息转换成数组,参考《GPU Computing Gems Jade Edition》;
	void initial_mst_graph();
	void scan_graph();
	//functions in scan_graph;
	void record_cycle(int* list, int size, int cycle_index, int* temp_cyc_list, int* onegroup_cyc_num);
	void merge_chain(int* list, int size, int chain_index);
	//
	void break_cycle();
	void update();//数组信息更新host端的;
	bool stop_condition();

	//free list
	void freelist(cycle* a);
	void freelist(cycle_list* a);

public:
	void run(char* filename, char* matname);
	void set_max_stream_num(int num);
	void output_result();
	void output_result(int o);
};
#endif