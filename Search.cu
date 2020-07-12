#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "omp.h"
#include <unistd.h>
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h> 
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>   

#define VERBOSE false
#define DIM_ORDERING true 

const int BLOCK_SIZE = 1024; 
const int KERNEL_BLOCKS = 32;
const int BRUTE = false;
const int CENTERED_RP = false;
const int RANDOM = false;
const int MID_RP = false;

//const int results_buffer = 5;
const int point_assumption = 15000;


typedef struct address_struct {
    int range_pair[2];
    int point_number;
    int *address; // points to a an address of rps points
} address_struct;

typedef struct index_return{
	int array_counter;
	int *point_array;
	int *address_array;
	int *range_array;
	int *point_address_array;
} index_return;


//need to pass in the neighbortable thats an array of the dataset size
//carry around a pointer to the array that has the points within epsilon 

//Need to use different members if unicomp is enabled or disabled
typedef struct neighborTable
{
	int cntNDataArrays;
	std::vector<int>vectindexmin;
	std::vector<int>vectindexmax;
	std::vector<int *>vectdataPtr;
	omp_lock_t pointLock; //one lock per point

}neighborTable;

__device__ int pow_int(int x, int y){
	int answ = 1;
	for(int i = 0; i < y; i++){
		answ *= x;
	}
	return answ;
}

//function prototypes
int brute_force( int num_points,
				int dim,
				int epsilon,
				double *A);

index_return *search_indexing(double *A, //the data from input
							double *RP, //our array of reference point coordinates
							int num_points,  //total umber of points in data set
							int rps, //number of reference points
							double epsilon, //the distance threshold
							int dim, //the number of dimensions
							int *point_array, //an array for the sorted points to go into
							int *range_array, //the range paris for each address to go into
							int * kaddress_array); //points to the device address thrust vector
__global__
void search_kernel( const int  batch_num, // this starts at 0 and increases by 1 every batch
               			const int batch_size, // this is a fixed value
	               		double * A, // this is the imported data
						const int num_points, // total number of points
						int * point_a, // a thrust vector which will store the first point in a pair
						int * point_b, // a thrust vector that will store a second point in a pair
						int * address_array, // the array of all generated addresses
						unsigned int * key_value_index, //a simple counter to keep track of how many results in a batch
						int * point_array,//the ordered points
						int * range_array,//the set of ranges
						const int array_counter, //the number of arrays
						const int rps, //the number of reference points
						const int dim, //the number of dimensions
						const double epsilon2, //the distance threshold
						const int tpp, // the number of threads per an address
						int *point_address_array);
__device__
int binary_search_add(int *array,
						int *array_size,
						int *search,
						int *rps);
__host__ __device__
int binary_search_basic(int *array,
						const int array_size,
						int *search,
						const int rps,
						int tid);

void bubble_sort_by_key(int *keys, int *address_array, int num_points, int rps);

int * stddev( double * A, int dim, int num_points) {
	double mean, devmean;
	double *deviation = (double*)malloc(sizeof(double) * dim);
	int *dimension = (int*)malloc(sizeof(int) * dim);
	for(int i = 0; i < dim; i++) {
		dimension[i] = i;
	}
	for(int i = 0; i < dim; i++){
		mean = 0.0;
		for(int j = 0; j < num_points; j++){
			mean += A[i*j];
		}
		mean /= num_points;
		devmean = 0.0;
		for(int j = 0; j < num_points; j++){
			devmean += pow(A[i*j] - mean,2);
		}
		devmean /= num_points;
		deviation[i] = sqrt(devmean);
	}
	thrust::sort_by_key(deviation, &deviation[dim-1], dimension);
	double *deviationret = (double*)malloc(sizeof(double) * dim);
	int *dimensionret = (int*)malloc(sizeof(int) * dim);
	for(int i = 0; i < dim; i++){
		deviationret[i] = deviation[dim-1-i];
		dimensionret[i] = dimension[dim-1-i];
	}
	free(deviationret);
	free(deviation);
	free(dimension);
	return dimensionret;
}

//compares 2 addresses and returns true if 1 is less than 2
__host__ __device__
bool add_comparison_ls(int *add1, int *add2, const int rps)
{
  for(char i = 0; i < rps; i++){
    if(*(add1+i+1) < *(add2+i+1))
    {
      return true;
    } else if (*(add1+i+1) > *(add2+i+1)){
			return false;
		}
  }
  return false;
}

//compares 2 addresses and returns true if 1 is equal or less than 2
__host__ __device__
bool add_comparison_eq_ls(int *add1, int *add2, const int rps)
{
  for(char i = 0; i < rps; i++){
    if(*(add1+i+1) > *(add2+i+1))
    {
      return false;
    }
  }
  return true;
}

//compares 2 addresses and returns true if  they are both equal
__host__ __device__
bool add_comparison_eq(int *add1, int *add2, const int rps)
{
  for(char i = 0; i < rps; i++){
    if(*(add1+i+1) != *(add2+i+1))
    {
      return false;
    }
  }
  return true;
}

__device__ __host__
	int binary_search_basic(int *array, //points to an array
													const int array_size, //points to an int
													int *search, //points to an array
													const int rps, //points to an int
													int tid)
{
	int first = 0;
	int last = array_size;
	int middle = (first+last)/2;
	int strider = (rps+1);

	while (first <= last){

		if(add_comparison_ls(array+middle*strider, search, rps)){
			first = middle + 1;
			middle = first+(last-first)/2;
		} else if (add_comparison_eq(array+middle*strider, search, rps)){
			return middle;
		}else{
			last = middle - 1;
			middle = first+(last-first)/2;
		}
	}
	return -1;
}

//Fixing the issue with unicomp requiring multiple updates and overwriting the data
void constructNeighborTable(int * pointInDistValue, 
							int * pointersToNeighbors, 
							unsigned int * cnt, 
							int * uniqueKeys, 
							int * uniqueKeyPosition, 
							unsigned int numUniqueKeys,
							struct neighborTable * tables)
{

	#pragma omp parallel for
	for (unsigned int i=0; i < (*cnt); i++)
	{
		pointersToNeighbors[i] = pointInDistValue[i];
	}

	//////////////////////////////
	//NEW when Using unique on GPU
	//When neighbortable is initalized (memory reserved for vectors), we can write directly in the vector

	//if using unicomp we need to update different parts of the struct
	#pragma omp parallel for
	for (unsigned int i = 0; i < numUniqueKeys; i++) {

		unsigned int keyElem = uniqueKeys[i];
		//Update counter to write position in critical section
		omp_set_lock(&tables[keyElem].pointLock);
		
		unsigned int nextIdx = tables[keyElem].cntNDataArrays;
		tables[keyElem].cntNDataArrays++;
		omp_unset_lock(&tables[keyElem].pointLock);

		tables[keyElem].vectindexmin[nextIdx] = uniqueKeyPosition[i];
		tables[keyElem].vectdataPtr[nextIdx] = pointersToNeighbors;	

		//final value will be missing
		if (i == (numUniqueKeys - 1))
		{
			tables[keyElem].vectindexmax[nextIdx] = (*cnt)-1;
		}
		else
		{
			tables[keyElem].vectindexmax[nextIdx] = (uniqueKeyPosition[i+1]) - 1;
		}
	}
	
	return;


} 

//unique key array on the GPU
__global__ 
void kernelUniqueKeys(int * pointIDKey, unsigned int * N, int * uniqueKey, int * uniqueKeyPosition, unsigned int * cnt)
{
	unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

	if (tid >= *N){
		return;
	}	

	if (tid == 0)
	{
		unsigned int idx = atomicAdd(cnt,(unsigned int)1);
		uniqueKey[idx] = pointIDKey[0];
		uniqueKeyPosition[idx] = 0;
		return;
	
	}
	
	//All other threads, compare to previous value to the array and add
	
	if (pointIDKey[tid-1] != pointIDKey[tid])
	{
		unsigned int idx = atomicAdd(cnt,(unsigned int)1);
		uniqueKey[idx] = pointIDKey[tid];
		uniqueKeyPosition[idx] = tid;
	}
	
}




//shared memory for storing address permutations for each block
extern __shared__ int address_shared[];

__global__ 
//__launch_bounds__( BLOCK_SIZE ) //can add second parameter of min blocks
void search_kernel( const int batch_num, // this starts at 0 and increases by 1 every batch
					const int batch_size, // this is a fixed value
					double * A, // this is the imported data
					const int num_points, // total number of points
					int * point_a, // an array which will store the first point in a pair
					int * point_b, // an array vector that will store a second point in a pair
					int * address_array, // the array of all generated addresses
					unsigned int * key_value_index, //a simple counter to keep track of how many results in a batch
					int * point_array,//the ordered points
					int * range_array,//the set of ranges
					const int array_counter, //the number of arrays
					const int rps, //the number of reference points
					const int dim, //the number of dimensions
					const double epsilon2, //the distance threshold
					const int tpp, // the number of threads per an address
					int *point_address_array)

{
  //the thread id is the id in the block plus the max id of the last batch
   unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x + (batch_size)*(batch_num);

	char stride = rps+1;

	// an exit clause if the number of threads wanted does not line up with block sizes
	if ( blockIdx.x*blockDim.x+threadIdx.x >= batch_size || tid >= tpp*num_points)
	{
		return;
	}

	//find the point number and the address number
	int point_location = tid/(tpp);
	int point_num = point_array[tid/(tpp)];
    int address_num = point_address_array[tid/(tpp)];
	
	/*
	if(tid%(tpp)==0){
		unsigned int index = atomicAdd(key_value_index,(unsigned int)1); // atomic add to count results
		point_b[index] = point_num; //stores the first point Number
		point_a[index] = point_num; // this store the cooresponding point number to form a pair
	}
	*/


    // just search own address
  
	//getting the ranges of the points
	//int start = range_array[2*address_array[address_num*stride]];//inclusive
	int end = range_array[2*address_array[address_num*stride]+1];//exclusive

	for(int j = point_location+1+(tid % (tpp)); j < end; j+=(tpp))
	{
		double distance = 0; // a double to hold intermediate values
		//we calculate the distance in every dimension
        for(int k = 0; k < (dim - dim % 2); k += 2)
        {
           //distance += (A[point_location*(dim) + k]-A[j*(dim) + k])*(A[point_location*(dim) + k]-A[j*(dim) + k]);
		   distance += pow(A[point_location*(dim) + k]-A[j*(dim) + k],2);
		   //distance += (A[point_location*(dim) + k+1]-A[j*(dim) + k+1])*(A[point_location*(dim) + k+1]-A[j*(dim) + k+1]);
		   distance += pow(A[point_location*(dim) + k+1]-A[j*(dim) + k+1],2);
          if(distance > (epsilon2)){break;} //this checks to see if we can short circuit

         //   distance += (A[point_location*(dim) + k+2]-A[j*(dim) + k+2])*(A[point_location*(dim) + k+2]-A[j*(dim) + k+2]);

         //   distance += (A[point_location*(dim) + k+3]-A[j*(dim) + k+3])*(A[point_location*(dim) + k+3]-A[j*(dim) + k+3]);
            
          //  distance += (A[point_location*(dim) + k+4]-A[j*(dim) + k+4])*(A[point_location*(dim) + k+4]-A[j*(dim) + k+4]);

        }

        for ( int k = (dim - dim % 2); k < dim; k ++){
           // distance += (A[point_location*(dim) + k]-A[j*(dim) + k])*(A[point_location*(dim) + k]-A[j*(dim) + k]);
		   distance += pow(A[point_location*(dim) + k]-A[j*(dim) + k],2);	
		   if(distance > (epsilon2)){break;} //this checks to see if we can short circuit                        

        }

		if(distance <= (epsilon2)) //if sqrt of the distance is <= epsilon
		{
			unsigned int index = atomicAdd(key_value_index,(unsigned int)1); // atomic add to count results
			point_a[index] = point_num; //stores the first point Number
			point_b[index] = point_array[j]; // this store the cooresponding point number to form a pair
			index = atomicAdd(key_value_index,(unsigned int)1); // atomic add to count results
			point_b[index] = point_num; //stores the first point Number
			point_a[index] = point_array[j]; // this store the cooresponding point number to form a pair
		}
	}
	
    // return if all even
    for (char i = 0; i < rps; i++){
        if (address_array[address_num*stride + i + 1] % 2 == 1){
            break;
        }
        if (i == rps-1) { return;}
    }
 


    //number possible combos is rps - the first odd address
	for (char i = 0; i < (rps); i++) // this itterates through every possible address combo
    {	
		if(address_array[(address_num)*stride+i+1] %2 != 1){continue;}
		for( char c = 0; c < 2; c++)
		{	
			for (int b = 0; b < pow_int(3, i); b++)
			{
				for (char j = 0; j < i; j++)
				{
					char temp = (b / pow_int(3, j) ) % 3;
					if (temp == 2){
						temp = -1;
					}
					*(address_shared+threadIdx.x*stride+j+1) =  temp;

				}
				for (char j = i; j < rps; j++){
					*(address_shared+threadIdx.x*stride+j+1) =  0;
				}



				// copy the modified amount back over for the binary search
				char update_counter = 0;
				for (char j = 0; j < (rps); j++)
					{
						if (update_counter < i){
							*(address_shared+threadIdx.x*stride+j+1) = *(address_shared+threadIdx.x*stride+update_counter+1) + address_array[(address_num)*stride+j+1];
							update_counter++;
						}else if(j == i){
							if(c == 0){
								*(address_shared+threadIdx.x*stride+j+1) = address_array[(address_num)*stride+j+1] + 1;
							} else {
								*(address_shared+threadIdx.x*stride+j+1) = address_array[(address_num)*stride+j+1] - 1;
							}
						}else{
							*(address_shared+threadIdx.x*stride+j+1) = address_array[(address_num)*stride+j+1];
						}

					}


				int address_location = -1;
				address_location = binary_search_basic(address_array, array_counter, &address_shared[threadIdx.x*stride], rps, tid);

				if( address_location == -1)
				{
					continue;
				}


				//getting the ranges of the points
				int start = range_array[2*address_array[address_location*stride]];//inclusive
				int end = range_array[2*address_array[address_location*stride]+1];//exclusive

				for(int j = start+(tid % (tpp)); j < end; j+=(tpp))
				{
					double distance = 0; // a double to hold intermediate values
					//we calculate the distance in every dimension
                    for(int k = 0; k < (dim - dim % 2); k += 2)
                    {         
                        //distance += (A[point_location*(dim) + k]-A[j*(dim) + k])*(A[point_location*(dim) + k]-A[j*(dim) + k]);
						distance += pow(A[point_location*(dim) + k]-A[j*(dim) + k],2);
                        //distance += (A[point_location*(dim) + k+1]-A[j*(dim) + k+1])*(A[point_location*(dim) + k+1]-A[j*(dim) + k+1]);
						distance += pow(A[point_location*(dim) + k+1]-A[j*(dim) + k+1],2);
						if(distance > (epsilon2)){break;} //this checks to see if we can short circuit

                       // distance += (A[point_location*(dim) + k+2]-A[j*(dim) + k+2])*(A[point_location*(dim) + k+2]-A[j*(dim) + k+2]);

                       // distance += (A[point_location*(dim) + k+3]-A[j*(dim) + k+3])*(A[point_location*(dim) + k+3]-A[j*(dim) + k+3]);
                  
                       // distance += (A[point_location*(dim) + k+4]-A[j*(dim) + k+4])*(A[point_location*(dim) + k+4]-A[j*(dim) + k+4]);

                    }
            
                    for ( int k = (dim - dim % 2); k < dim; k ++){
                       // distance += (A[point_location*(dim) + k]-A[j*(dim) + k])*(A[point_location*(dim) + k]-A[j*(dim) + k]);         
					   distance += pow(A[point_location*(dim) + k]-A[j*(dim) + k],2);
					   if(distance > (epsilon2)){break;} //this checks to see if we can short circuit

                    }

					if(distance <= (epsilon2)) //if sqrt of the distance is <= epsilon
					{
						unsigned int index = atomicAdd(key_value_index,(unsigned int)1); // atomic add to count results
						point_a[index] = point_num; //stores the first point Number
						point_b[index] = point_array[j]; // this store the cooresponding point number to form a pair

						index = atomicAdd(key_value_index,(unsigned int)1); // atomic add to count results
						point_a[index] = point_array[j]; //stores the first point Number
						point_b[index] = point_num; // this store the coresponding point number to form a pair
				
					}
				}
			}
		}
	}
}



//function to sort into bins and start kernel returns number of adds
index_return *search_indexing(double *A, //the data from input
							double *RP, //our array of reference point coordinates
							int num_points,  //total umber of points in data set
							const int rps, //number of reference points
							double epsilon, //the distance threshold
							int dim) //the number of dimensions
 {	
	//int *address_array = (int*)calloc(sizeof(int),(num_points)*(rps+1));
    thrust::host_vector<int*> address_array(num_points);
	for(int i = 0; i < num_points; i++)
	{
		address_array[i] = new int[rps+1];
	}


	int array_counter = 0;

  	thrust::host_vector<int> rp_bin(num_points);//thrust vector for holding bin distance to reference point

	//making the start of each address struct the point number
	for(int i = 0; i < num_points; i++)
	{
		address_array[i][0] = i;
	}


  //working from the least significant address bin

  for(int i = rps - 1; i >= 0; i--)
  {  
	#pragma omp parallel for
    for(int j = 0; j < num_points; j++) //for every address we loop through each point
    {
      double distance = 0;
      for(int k = 0; k < dim; k++) // we calculate the distnace from every dim
      {
        double a1 = A[address_array[j][0]*dim+k];
        double a2 = RP[i*dim+k];
        distance += (a2-a1)*(a2-a1);
      }
      int bin = floor(sqrt(distance) / (1.0*epsilon)); //the bin number will be the higher multiple of E distance from the rp
      address_array[j][i+1] = bin;
      rp_bin[j] = bin;
    }

    // this sorts the addresses based on the most recent bin. By sorting the from the least
    // significant to the most, the final order of the address should be correct
    // this is a parrallel sort on the cpu.
    thrust::stable_sort_by_key(thrust::omp::par,rp_bin.begin(), rp_bin.end(), address_array.begin());

  }


  //now going through, keeping count of how many are in each unique address
  //generating a new address struct with the first item pointing to a range pair
  // thats in an array, which will then point to points in the point array

  //int *point_array; //this will have the point numbers that the range will reference
  int *point_array = (int*)malloc(num_points*sizeof(int));

  int *tmp_range_array; //a tempory holder for the range pairs, until we know how many there will be
  tmp_range_array = (int*)malloc(num_points*sizeof(int)*2);

	//an array to track  what address a point is in
	int *point_address_array = (int*)malloc(sizeof(int)*num_points);

  int range_counter = 0;//this is the current range value
  for(int i = 0; i < num_points; i++)
  {

		point_address_array[i] = array_counter;
   		 point_array[i] = address_array[i][0]; //populating the range array
   		 tmp_range_array[(array_counter)*2] = range_counter;//where our range starts
		range_counter++;//increasing the range since it is non inclusive

    while(i+1<num_points && add_comparison_eq(address_array[i], address_array[i+1], rps)) //while the addresses are the same
    {
		range_counter++;
		i++;//the next address is the same so we can skip it
		point_array[i] = address_array[i][0];//storing that addresses point
		point_address_array[i] = array_counter;

    }
    tmp_range_array[array_counter*2+1] = range_counter; //the end of our range
		array_counter++;//moving on to the next range pair
  }


    int *kaddress_array = (int*)malloc(sizeof(int)*(array_counter)*(rps+1));

    int *range_array = (int*)malloc((array_counter)*sizeof(int)*2);

  //going through and filling the new arrays
  #pragma omp parallel for
  for (int i = 0; i < (array_counter); i++)
  {
    range_array[2*i] = tmp_range_array[2*i];
		range_array[2*i+1] = tmp_range_array[2*i+1];
		for(int j = 1; j < rps+1; j++)
		{
			kaddress_array[i*(rps+1)+j] = address_array[range_array[2*i]][j];
		}

		kaddress_array[i*(rps+1)] = i; // the first value will point to a range pair

  }

  free(tmp_range_array); //no longer need this because of range_array


	struct index_return *results;
	results = (index_return*)malloc(sizeof(index_return));
	results->point_array = point_array;
	results->array_counter = array_counter;
	results->address_array = kaddress_array;
	results->range_array = range_array;
	results->point_address_array = point_address_array;

	return results;
}


int brute_force( int num_points,
				int dim,
				double epsilon,
				double *A)
{
 printf("\n\n*******************************************\n");
 //brute force check
 int brute_count = 0;
 omp_lock_t brute;
 omp_init_lock(&brute);

 #pragma omp parallel for
 for(int i = 0; i < num_points; i++)
 {
   for (int j = 0; j < num_points; j++)
   {
     double distance = 0;
     for (int k = 0; k < dim; k++)
     {
       if(distance > epsilon*epsilon)
       {
         break;
       } else {
         double a1 = A[i*dim + k];
         double a2 = A[j*dim + k];
         distance += (a1-a2)*(a1-a2);
       }
     }
     if(distance <= epsilon*epsilon)
     {
	   omp_set_lock(&brute);
       brute_count++;
			 if(VERBOSE)
			 {
		 		  printf("(%d,%d),", i, j);
			 }
		omp_unset_lock(&brute);
     }
   }
 }
 printf("\nBrute force has %d pairs.\n", brute_count);
 return brute_count;
}

int main(int argc, char*argv[])
{
	

	//reading in command line arguments
	char *filename = argv[1];
	int dim = atoi(argv[2]);
	int tpp = atoi(argv[3]);
	int rps = atoi(argv[4]);
	int concurent_streams = atoi(argv[5]);
	double epsilon;
	sscanf(argv[6], "%lf", &epsilon);

	double time0 = omp_get_wtime();

	std::ifstream file(	filename, std::ios::in | std::ios::binary);
	file.seekg(0, std::ios::end); 
	size_t size = file.tellg();  
	file.seekg(0, std::ios::beg); 
	char * read_buffer = new char[size];
	file.read(read_buffer, size*sizeof(double));
	file.close();

	double time00 = omp_get_wtime();
	//printf("\nTime to read in file: %f\n", time00-time0);

	double* A = (double*)read_buffer;//reinterpret as doubles

	int num_points = size/sizeof(double)/dim;

	//printf("\nNumber points: %d ", num_points);
	//printf("\nNumber Dimensions: %d ", dim);
	//printf("\nNumber reference points: %d ", rps);
	//printf("\nNumber Threads Per Point: %d ", tpp);
	//printf("\nNumber Concurent Streams: %d", concurent_streams);
	//printf("\nDistance Threshold: %f \n*********************************\n\n", epsilon);




	int *dimension_order = (int*)malloc(sizeof(int)*dim);
	double * dim_ordered_data = (double*)malloc(sizeof(double)*num_points*dim);

	if(DIM_ORDERING){
		dimension_order = stddev(A, dim, num_points);
		#pragma omp parallel for
		for(int i = 0; i < num_points; i++){
			for(int j = 0; j < dim; j++){
				dim_ordered_data[i*dim + j] = A[i*dim + dimension_order[j]];
			}
		}
		//printf("Data reordered\n");
		A = dim_ordered_data;;
	}

	double *RP;
    RP = (double *)calloc(dim*rps, sizeof(double));

	if (CENTERED_RP) {
		//get the average for rp placement
		for(int i = 0; i < num_points; i++)
		{
			for(int j = 0; j < dim; j++)
			{
				RP[j] += A[i*dim+j];
			}
		}

		for(int i = 0; i < dim; i++)
		{
			RP[i] = RP[i] / (1.0*num_points);
		//	printf("%f ",RP[i]);
		}
	//	printf("\n");
		for(int i = 1; i < rps; i++) // the first rp is centered
		{
			for(int j = 0; j < dim; j++)
			{
				if (i % 2 == 1)
				{
					RP[i*dim+j] = RP[j]+i*epsilon;
			//		printf("%f ", RP[i*dim+j]);
				} else {
					RP[i*dim+j] = RP[j]+i*0.33*epsilon;
			//		printf("%f ", RP[i*dim+j]);
				}
			}
		//	printf("\n");
		}
	} else { //box the data
		//get the max
		for(int j = 0; j < dim; j++)
		{
			RP[j] = 0;
		}
		for(int i = 0; i < num_points; i++)
		{
			for(int j = 0; j < dim; j++)
			{
				if(RP[j] < A[i*dim+j])
				{
					RP[j] = A[i*dim+j];
				}
			}
		}
		for(int i = 1; i < rps; i++) // the first rp is set
		{
			int step = dim/rps;
			for(int j = 0; j < step; j++)
			{
					RP[i*dim+i*step + j] = RP[j+i*step];
			}
		}
		if(MID_RP){
			for(int i = 0; i < num_points; i++)
			{
				for(int j = 0; j < dim; j++)
				{
					RP[j] += A[i*dim+j];
				}
			}

			for(int i = 0; i < dim; i++)
			{
				RP[i] = RP[i] / (1.0*num_points);
			//	printf("%f ",RP[i]);
			}
		}
	}

	double time1 = omp_get_wtime();

	const int s_rps = rps;
	index_return *results = search_indexing(A, RP, num_points, s_rps, epsilon, dim);

	int *point_array = results->point_array;
	int *range_array = results->range_array;
	int *address_array = results->address_array;
	int array_counter = results->array_counter;
	int *point_address_array = results->point_address_array;

	//printf("\n\nNumber of Unique addresses: %d\n",array_counter);
	
	if(VERBOSE)
	{


		printf("\n\nThe Range Array:");
		for(int i = 0; i < array_counter*2; i+=2)
			{
				printf("(%d,%d),", range_array[i], range_array[i+1]);
			}

		if(rps == 4)
		{
		printf("\n\nThe Address Array:");
		for(int i = 0; i < array_counter*(rps+1); i+=4)
			{
				printf("(%d,%d,%d,%d,%d),", address_array[i], address_array[i+1],address_array[i+2],address_array[i+3],address_array[i+4]);
			}
		}
		printf("\nPoints  array: ");
		for(int k = 0; k<num_points; k++)
			{
			printf("%d,", point_array[k]);
			}
		printf("\nPoints address  array: ");
		for(int k = 0; k<num_points; k++)
			{
			printf("%d,", point_address_array[k]);
			}
	}

    double * point_ordered_data = (double *)malloc(sizeof(double)*num_points*dim);
    #pragma omp parallel for
	for(int i = 0; i < num_points; i++){
		for(int j = 0; j < dim; j++){
			point_ordered_data[i*dim+j] = A[point_array[i]*dim+j];
			//printf("%f ", point_ordered_data[i*dim+j]);
		}
		//printf("\n");
	}
	//printf("Points Reordered");
	A = point_ordered_data;

	int largest_index_index = 0;
	int max_index = 0;
	#pragma omp parallel for
	for(int i = 0; i < array_counter; i++){
		int index_size = range_array[i*2 + 1] - range_array[i*2];
		if(index_size > max_index){
			max_index = index_size;
			largest_index_index = i;
		}
	}

	int first_point_largest_index = range_array[largest_index_index*2];

	//printf("\nFirst Point in the largest Index: %d", first_point_largest_index);
	
	//printf("\nLargest Index Size: %d",  max_index);
	//printf("\nLargest Index Index: %d\n",  largest_index_index);

	double time2 = omp_get_wtime();

	//printf("\nTime to index: %f\n", time2-time1);

	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;
	cudaError_t cudaStat5 = cudaSuccess;
	cudaError_t cudaStat6 = cudaSuccess;
	cudaError_t cudaStat7 = cudaSuccess;
	cudaError_t cudaStat8 = cudaSuccess;
	cudaError_t cudaStat9 = cudaSuccess;
	cudaError_t cudaStat10 = cudaSuccess;
	cudaError_t cudaStat11 = cudaSuccess;
	cudaError_t cudaStat12 = cudaSuccess;
	cudaError_t cudaStat13 = cudaSuccess;
	cudaError_t cudaStat14 = cudaSuccess;
	cudaError_t cudaStat15 = cudaSuccess;
	cudaError_t cudaStat16 = cudaSuccess;
	cudaError_t cudaStat17 = cudaSuccess;
	cudaError_t cudaStat18 = cudaSuccess;
	cudaError_t cudaStat19 = cudaSuccess;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	//printf("global Mem(MB): %lu\n", prop.totalGlobalMem/1000/1000);
	//printf("data Mem(MB): %ld\n", sizeof(double)*num_points*dim/1000/1000);
	long int data_mem = sizeof(double)*num_points*dim;
	long int available_mem = prop.totalGlobalMem - data_mem*2 - sizeof(int)*num_points*3;
	//printf("Avaliable Mem(MB): %ld\n", available_mem/1000/1000);
	long int num_stored_pairs = available_mem / sizeof(int) / 2;
	//printf("number paired stored: %ld\n", num_stored_pairs);
	int batch_size = BLOCK_SIZE*tpp*KERNEL_BLOCKS;
	int batchs = ceil(num_points * tpp*1.0 / batch_size);
	

	//printf("\nBatchs: %d\n", batchs);

	int *batch_num;
	cudaStat1 = cudaMallocHost((void**)&batch_num, batchs*sizeof(int)); // this is the current batch being proccsessed aka starting batch
	assert(cudaSuccess == cudaStat1);
	for (int i = 0; i < batchs; i++) {
		batch_num[i] = batchs-i-1;
	} 
	if(RANDOM){
		std::random_shuffle(batch_num, batch_num+batchs);
	}
	//the batch size will be the number of points times the number of threads per
	// point divided by how many TOTAL batchs there will be rounded up
	
	int num_point_per_batch = BLOCK_SIZE*KERNEL_BLOCKS;//approx
	//printf("Number Points per Batch: %d\n", num_point_per_batch);

	//printf("Attempting to store up to %d pairs\n",num_point_per_batch*mem_pairs*5*(concurent_streams));

	unsigned int *key_value_index;
	cudaStat1 = cudaMallocHost((void**)&key_value_index, batchs*sizeof(unsigned int)); // this is the current batch being proccsessed aka starting batch
	assert(cudaSuccess == cudaStat1);
	for (int i = 0; i < batchs; i++) {
		key_value_index[i] = 0;
	}


	unsigned int *unique_cnt;
	cudaStat1 = cudaMallocHost((void**)&unique_cnt, batchs*sizeof(unsigned int)); // this is the current batch being proccsessed aka starting batch
	assert(cudaSuccess == cudaStat1);
	for (int i = 0; i < batchs; i++) {
		unique_cnt[i] = 0;
	}



//	int max_results = largest_index_size*6*ceil(num_point_per_batch*1.0 / largest_index_size);
//	printf("%d" , max_results);

	//int * d_batch_num; // this starts at 0 and increases by 1 every batch
	//int * d_batch_size; // this is a fixed value
	double * d_A;// this is the imported data 
	//int * d_num_points; // total number of points
	int * d_address_array; // the array of all generated addresses
	unsigned int * d_key_value_index; //a simple counter to keep track of how many results in a batch
	int * d_point_array;//the ordered points
	int * d_range_array;//the set of ranges
	//int * d_array_counter; //the number of arrays
	//int *d_rps; //the number of reference points
	//int *d_dim; //the number of dimensions
	//double *d_epsilon; //the distance threshold
	//int *d_tpp; // the number of threads per an address
	int *d_point_address_array; //pre allocated permutation
	int * d_point_a;
	int * d_point_b;
	unsigned int * d_unique_cnt;
	int * d_unique_key_position;
	int * d_unique_keys;

	//cudaStat1 = cudaMalloc((void **)&d_batch_num , sizeof(int)*batchs);
	//cudaStat2 = cudaMalloc((void **)&d_batch_size , sizeof( int));
	//cudaStat3 = cudaMalloc((void **)&d_num_points , sizeof( int));
	cudaStat4 = cudaMalloc((void **)&d_address_array , sizeof(int)*array_counter*(rps+1));
	cudaStat5 = cudaMalloc((void **)&d_key_value_index , sizeof(unsigned int)*batchs);
	cudaStat6 = cudaMalloc((void **)&d_point_array , sizeof( int)*num_points);
	cudaStat7 = cudaMalloc((void **)&d_range_array , sizeof(int)*array_counter*2);
	//cudaStat8 = cudaMalloc((void **)&d_array_counter , sizeof(int));
	//cudaStat9 = cudaMalloc((void **)&d_rps , sizeof(int));
	//cudaStat10 = cudaMalloc((void **)&d_dim , sizeof(int));
	//cudaStat11 = cudaMalloc((void **)&d_epsilon , sizeof(double));
	//cudaStat12 = cudaMalloc((void **)&d_tpp , sizeof(int));
	cudaStat13 = cudaMalloc((void **)&d_point_address_array , sizeof(int)*num_points);
	cudaStat14 = cudaMalloc((void **)&d_point_a , sizeof(int)*num_point_per_batch*point_assumption*concurent_streams);
	cudaStat15 = cudaMalloc((void **)&d_point_b , sizeof(int)*num_point_per_batch*point_assumption*concurent_streams);
	cudaStat16 = cudaMalloc((void **)&d_A , sizeof(double)*num_points*dim);
	cudaStat17 = cudaMalloc((void **)&d_unique_key_position, sizeof(int)*concurent_streams*num_points);
	cudaStat18 = cudaMalloc((void **)&d_unique_keys, sizeof(int)*concurent_streams*num_points);
	cudaStat19 = cudaMalloc((void **)&d_unique_cnt , sizeof(unsigned int)*batchs);

	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);
	assert(cudaSuccess == cudaStat5);
	assert(cudaSuccess == cudaStat6);
	assert(cudaSuccess == cudaStat7);
	assert(cudaSuccess == cudaStat8);
	assert(cudaSuccess == cudaStat9);
	assert(cudaSuccess == cudaStat10);
	assert(cudaSuccess == cudaStat11);
	assert(cudaSuccess == cudaStat12);
	assert(cudaSuccess == cudaStat13);
	assert(cudaSuccess == cudaStat14);
	assert(cudaSuccess == cudaStat15);
	assert(cudaSuccess == cudaStat16);
	assert(cudaSuccess == cudaStat17);
	assert(cudaSuccess == cudaStat18);
	assert(cudaSuccess == cudaStat19);

	//double epsilon2 = epsilon * epsilon;

	//cudaStat1 = cudaMemcpy(d_batch_num, batch_num, sizeof(int)*batchs, cudaMemcpyHostToDevice);
	//cudaStat2 = cudaMemcpy(d_batch_size, &batch_size, sizeof(int), cudaMemcpyHostToDevice);
	cudaStat3 = cudaMemcpy(d_A, A, sizeof(double)*num_points*dim, cudaMemcpyHostToDevice);
	//cudaStat4 = cudaMemcpy(d_num_points, &num_points, sizeof(int), cudaMemcpyHostToDevice);
	cudaStat5 = cudaMemcpy(d_address_array, address_array, sizeof(int)*array_counter*(rps+1), cudaMemcpyHostToDevice);
	cudaStat6 = cudaMemcpy(d_point_array, point_array, sizeof(int)*num_points, cudaMemcpyHostToDevice);
	cudaStat7 = cudaMemcpy(d_range_array, range_array, sizeof(int)*array_counter*2, cudaMemcpyHostToDevice);
	//cudaStat8 = cudaMemcpy(d_array_counter, &array_counter, sizeof(int), cudaMemcpyHostToDevice);
	//cudaStat9 = cudaMemcpy(d_rps, &rps, sizeof(int), cudaMemcpyHostToDevice);
	cudaStat10 = cudaMemcpy(d_key_value_index, key_value_index, sizeof( unsigned int)*batchs, cudaMemcpyHostToDevice);
	//cudaStat11 = cudaMemcpy(d_dim, &dim, sizeof(int), cudaMemcpyHostToDevice);
	//cudaStat12 = cudaMemcpy(d_epsilon, &epsilon2, sizeof(double), cudaMemcpyHostToDevice);
	//cudaStat13 = cudaMemcpy(d_tpp, &tpp, sizeof(int), cudaMemcpyHostToDevice);
	cudaStat14 = cudaMemcpy(d_point_address_array, point_address_array, sizeof(int)*num_points, cudaMemcpyHostToDevice);
	cudaStat15 = cudaMemcpy(d_unique_cnt, unique_cnt, sizeof(unsigned int)*batchs, cudaMemcpyHostToDevice);

	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);
	assert(cudaSuccess == cudaStat5);
	assert(cudaSuccess == cudaStat6);
	assert(cudaSuccess == cudaStat7);
	assert(cudaSuccess == cudaStat8);
	assert(cudaSuccess == cudaStat9);
	assert(cudaSuccess == cudaStat10);
	assert(cudaSuccess == cudaStat11);
	assert(cudaSuccess == cudaStat12);
	assert(cudaSuccess == cudaStat13);
	assert(cudaSuccess == cudaStat14);
	assert(cudaSuccess == cudaStat15);

	unsigned int totalBlocks = ceil(batch_size*1.0/BLOCK_SIZE); // number of blocks needed

	cudaStream_t stream[concurent_streams];
	for (int i = 0; i < concurent_streams; i++){
		cudaError_t stream_check = cudaStreamCreate(stream+i);
		assert(cudaSuccess == stream_check);
	}

    //printf("\nAllocating neighbor table.....");
	struct neighborTable * tables = (struct neighborTable*)malloc(sizeof(struct neighborTable)*num_points);
    //printf("Complete\nConstructing neighbor table....");
    //tables = new neighborTable[num_points];
  //  #pragma omp parallel for
	for (int i = 0; i < num_points; i++)
	{	
		struct neighborTable temp;
		tables[i] = temp;
		//tables[i] = (struct neighborTable)malloc(sizeof(struct neighborTable));

		tables[i].cntNDataArrays = 1; 
		tables[i].vectindexmin.resize(batchs+1);
		tables[i].vectindexmin[0] = i;
		tables[i].vectindexmax.resize(batchs+1);
		tables[i].vectindexmax[0] = i;
		tables[i].vectdataPtr.resize(batchs+1);
		tables[i].vectdataPtr[0] = point_array;
		omp_init_lock(&tables[i].pointLock);
	}

    //printf("Complete\n");
	cudaDeviceSynchronize(); 

	double time3 = omp_get_wtime();


	//printf("\nTime to transfer to Device/Allocate Pinned memory: %f\n", time3-time2);


	unsigned long total_pairs = 0;
	omp_lock_t data_memlock;
	omp_lock_t point_memlock;
	omp_lock_t index_memlock;
	omp_lock_t result_memlock;
	omp_lock_t max_memlock;
	omp_lock_t vector_memlock;
	omp_init_lock(&data_memlock);
	omp_init_lock(&point_memlock);
	omp_init_lock(&index_memlock);
	omp_init_lock(&result_memlock);
	omp_init_lock(&max_memlock);
	omp_init_lock(&vector_memlock);

	int * data_array[batchs]; 
	int * point_b[concurent_streams];
	int buffer_sizes[concurent_streams];
	for (int i = 0; i < concurent_streams; i++){
		buffer_sizes[i] = num_point_per_batch;
		assert(cudaSuccess == cudaMallocHost((void**) &point_b[i], sizeof(int)*num_point_per_batch));
	}

	//omp_set_num_threads(concurent_streams);
	#pragma omp parallel for num_threads(concurent_streams)  schedule(dynamic)
	for(int i = 0; i < batchs; i++) //itterate through each batch
	{
		int tid = omp_get_thread_num();

		int offset = (tid) * num_point_per_batch*point_assumption;
		const int d_batch_num = batch_num[i];
		const int d_batch_size = batch_size;
		const int d_num_points = num_points;
		const int d_array_counter = array_counter;
		const int d_rps = rps;
		const int d_dim = dim;
		const int d_tpp = tpp;
		const double d_epsilon2 = epsilon*epsilon;

		//shared mem size BLOCK_SIZE*(rps+1)*sizeof(int)
		search_kernel<<<totalBlocks,BLOCK_SIZE, (BLOCK_SIZE)*(rps+1)*sizeof(int), stream[tid]>>>(
			d_batch_num,
			d_batch_size,
			d_A,
			d_num_points,
			d_point_a+offset,
			d_point_b+offset,
			d_address_array,
			&d_key_value_index[i],
			d_point_array,
			d_range_array,
			d_array_counter,
			d_rps,
			d_dim,
			d_epsilon2,
			d_tpp,
			d_point_address_array);
 
		cudaStreamSynchronize(stream[tid]);
		cudaError_t err = cudaGetLastError(); 
		if (err != cudaSuccess) 
				printf("Error: %s\n", cudaGetErrorString(err));

       // printf("tid: %d kernel done\n", tid);
		assert(cudaSuccess == cudaMemcpyAsync(&key_value_index[i], &d_key_value_index[i], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid]));
        cudaStreamSynchronize(stream[tid]);
       // printf("tid: %d key value index grabbed\n", tid);

        if(key_value_index[i] > buffer_sizes[tid]){
		  //  printf("tid: %d first run\n", tid);
			  cudaFreeHost(point_b[tid]);
			//printf("tid: %d freed memory\n", tid);
            assert(cudaSuccess == cudaMallocHost((void**) &point_b[tid], sizeof(int)*(key_value_index[i])));
            //printf("tid: %d pinned memory\n", tid);
			 buffer_sizes[tid] = key_value_index[i];
        }

		cudaStreamSynchronize(stream[tid]);
		thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_point_a+offset, d_point_a+offset + key_value_index[i], d_point_b+offset);
	
		err = cudaGetLastError();
		if (err != cudaSuccess) 
				printf("Error: %s\n", cudaGetErrorString(err));

		assert(cudaSuccess == cudaMemcpyAsync(point_b[tid], d_point_b+offset, sizeof(int)*key_value_index[i], cudaMemcpyDeviceToHost, stream[tid]));

       // printf("tid: %d coppied back values\n", tid);
		
		
		//thrust::sort_by_key(thrust::device, d_point_a+offset, d_point_a + offset + key_value_index[i], d_point_b + offset);
        cudaStreamSynchronize(stream[tid]);
      //  printf("tid: %d sorted results\n", tid);

		err = cudaGetLastError();
		if (err != cudaSuccess) 
				printf("Error: %s\n", cudaGetErrorString(err));
	
		//omp_set_lock(&result_memlock);
		
		int totalBlocks2 = ceil((1.0*key_value_index[i])/(1.0*BLOCK_SIZE));	
		kernelUniqueKeys<<<totalBlocks2, BLOCK_SIZE,0,stream[tid]>>>(d_point_a+offset,
																	&d_key_value_index[i], 
																	d_unique_keys+tid*num_points, 
																	d_unique_key_position+tid*num_points, 
																	&d_unique_cnt[i]);
		
		cudaStreamSynchronize(stream[tid]);

		err = cudaGetLastError();
		if (err != cudaSuccess) 
				printf("Error: %s\n", cudaGetErrorString(err));

      //  printf("tid: %d key kernel done\n", tid);

		assert(cudaSuccess == cudaMemcpyAsync(&unique_cnt[i], &d_unique_cnt[i], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid]));
		cudaStreamSynchronize(stream[tid]);

		thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_unique_keys+tid*num_points, d_unique_keys+tid*num_points+unique_cnt[i], d_unique_key_position+tid*num_points);


		//printf("CPU tid: %d kernel: %d/%d batch: %d pairs: %u uniq cnt: %u \n", tid,i,batchs, batch_num[i], key_value_index[i], unique_cnt[i]);

		int * unique_keys = (int*)malloc(sizeof(int)*unique_cnt[i]);
		assert(cudaSuccess == cudaMemcpyAsync(unique_keys, d_unique_keys+tid*num_points, sizeof(int)*unique_cnt[i], cudaMemcpyDeviceToHost, stream[tid]));

		int * unique_key_position = (int*)malloc(sizeof(int)*unique_cnt[i]);
		assert(cudaSuccess == cudaMemcpyAsync(unique_key_position, d_unique_key_position+tid*num_points, sizeof(int)*unique_cnt[i], cudaMemcpyDeviceToHost, stream[tid]));
		
		//sort the uniques
		//thrust::sort_by_key(thrust::host, unique_keys, unique_keys+unique_cnt[i], unique_key_position);
		
		cudaStreamSynchronize(stream[tid]);
		data_array[i] = (int*)malloc(sizeof(int)*key_value_index[i]);
		
		
		//omp_set_lock(&result_memlock);
		//printf("tid: %d start table construction\n", tid);
		constructNeighborTable(point_b[tid], data_array[i], &key_value_index[i], unique_keys,unique_key_position, unique_cnt[i], tables);
		//printf("tid: %d end table construction\n", tid);

		//omp_unset_lock(&result_memlock);
			
		free(unique_keys);
		free(unique_key_position);

	}

	cudaDeviceSynchronize();

	double time4 = omp_get_wtime();
	unsigned int total_uniques = 0;
	//int total_pairs;
	total_pairs += num_points;
	for (int i = 0; i < batchs; i++){
		total_pairs += key_value_index[i];
		total_uniques += unique_cnt[i];
	}
	/*
	unsigned int check_cnt = 0;
    
	for (int i = 0 ; i < num_points; i++){
		//printf("\npoint id: %d, neighbors: ",i);
		// printf("\npoint id: %d, cntNDataArrays: %d: ",i, neighborTable[i].cntNDataArrays);
		//used for sorting the neighbors to compare neighbortables for validation
		for (int j = 0; j < tables[i].cntNDataArrays; j++)
		{
			for (int k = tables[i].vectindexmin[j]; k <= tables[i].vectindexmax[j]; k++)
			{
				check_cnt++;
			}
		}
    }
    */

	//printf("\nTime to Search: %f\n", time4-time3);
	//printf("Total Time after reading in file: %f\n\n", time4-time1);
	//printf("total unique cnt: %d\n", total_uniques);
	//printf("checkcnt: %u", check_cnt);
	//printf("Selectivity: Neighbors %f, percentage %f\n\n", (total_pairs-num_points)*1.0/num_points, (total_pairs-num_points)*1.0/num_points/num_points);
	printf("%d %d %d %f %f %lu\n", tpp, rps, concurent_streams, epsilon, time4-time1, total_pairs);

	if(VERBOSE){
		for (int i = 0 ; i < num_points; i++){
			printf("\npoint id: %d, neighbors: ",i);
			// printf("\npoint id: %d, cntNDataArrays: %d: ",i, neighborTable[i].cntNDataArrays);
			//used for sorting the neighbors to compare neighbortables for validation
			std::vector<int>tmp;
			for (int j = 0; j < tables[i].cntNDataArrays; j++)
			{
				for (int k = tables[i].vectindexmin[j]; k <= tables[i].vectindexmax[j]; k++)
				{
					tmp.push_back(tables[i].vectdataPtr[j][k]);
				}
			}

			//print sorted vector
			std::sort(tmp.begin(), tmp.end());
			for (int l = 0; l < tmp.size(); l++)
			{
				printf("%d,",tmp[l]);
			}	
		}
		printf("\n");
	}

	if(BRUTE){
		brute_force(num_points,dim, epsilon, A);
	}

	cudaFree(d_A);
	cudaFree(d_address_array);
	cudaFree(d_key_value_index);
	cudaFree(d_range_array);
	cudaFree(d_point_address_array);
	cudaFree(d_point_a);
	cudaFree(d_point_b);

	free(RP);
	free(A);

	free(tables);

	for(int i = 0; i < batchs; i++){
		free(data_array[i]);
	}
	//free(data_array);

	free(dimension_order);
	cudaFreeHost(key_value_index);
	cudaFreeHost(batch_num);
	cudaFreeHost(unique_cnt);
}

