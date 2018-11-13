#include <stdio.h>
#include <iostream>
#include <map>
#include <string>
#include <fstream>
#include <vector>

using namespace std;

#define ll long long

const int BLOCK_SIZE = 2;
const int GRID_SIZE = 1;

// Use naive method
__device__ bool isPrime(ll n)
{
    printf("IS PRIME FUNCTION %lld \n", n);
    if(n<2)
        return false;
        
    for(ll i=2;i*i<=n;i++)
        if(n%i==0)
            return false; 

    printf("%lld IS PRIME \n", n);
    return true;
}

// Read numbers from file and add to vector
std::vector<ll> reaadFile(char* arg){
    vector<ll> numbersFromFile;
    std::ifstream infile(arg);
    ll number;

    while (infile >> number) {
        numbersFromFile.push_back(number);
    }

    return numbersFromFile;
}

__global__ void calculate(ll *Arr, bool *results, int sizeOfArray){

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;

    printf("X: %d \n", x);
    printf("BLOCK_ID: %d\n", blockIdx.x);
    printf("BLOCK_DIM: %d \n", blockDim.x);
    printf("ThreaD_ID: %d\n", threadIdx.x);

	if (x < sizeOfArray) 
	{
        results[x] += isPrime(Arr[x]);
	}

}

bool isPrimeMain(ll n){
    if(n<2)
        return false;
        
    for(ll i=2;i*i<=n;i++)
        if(n%i==0)
            return false; 

    return true;
}
    
int main(int argc, char** argv )
{
    float time;

    if ( argc != 2 )
    {
        printf("Pass file path\n");
        return -1;
    }

    vector<ll> numbersFromFile = reaadFile(argv[1]);

    int sizeOfArray = numbersFromFile.size();
    int sizeToAllocateLongLong = sizeOfArray * sizeof(ll);
    int sizeToAllocateBool = sizeOfArray * sizeof(bool);

	
    ll numbersFromFileArr[sizeOfArray];
    std::copy(numbersFromFile.begin(), numbersFromFile.end(), numbersFromFileArr);

    cout << "BEFORE ALLOCATE RESULTS" << endl;

    unsigned int i;
    bool* results = (bool *) malloc (sizeToAllocateBool);

    ll* c_arr;
    bool* c_results;

    cout << "CUDA MALLOC" << endl;

    cudaMalloc((void**) &c_arr, sizeToAllocateLongLong);
    cudaMalloc((void**) &c_results, sizeToAllocateBool);

    cout << "CUDA MEMCPY" << endl;

    cudaMemcpy((void *)c_arr, (void *)numbersFromFileArr, sizeToAllocateLongLong, cudaMemcpyHostToDevice);

    cout << "BEFORE DIM3" << endl;

    dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grids(GRID_SIZE, GRID_SIZE);

    //Start timer
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

    cout << "BEFORE CALL KERNEL: " << endl;
    calculate<<<sizeOfArray, GRID_SIZE>>>(c_arr, c_results, sizeOfArray);

    //End timer and put result into time variable
    cudaDeviceSynchronize();			 
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

    printf("Czas: %.4fms\n", time);

    if (cudaMemcpy((void *)results, (void *)c_results , sizeToAllocateBool, cudaMemcpyDeviceToHost) != cudaSuccess) {
		cout<<"GPU to CPU copy error\n";
	}

    cudaFree(c_arr);
    cudaFree(c_results);



    for (int i = 0; i < sizeOfArray; i++){
        if (results[i]){
            cout << numbersFromFileArr[i] << " prime";
        } else {
            cout << numbersFromFileArr[i] << " composite";
        }

        if (isPrimeMain(numbersFromFileArr[i]) == results[i])
            cout << " - GOOD" << endl;
    }

    free(results);
    return 0;
}
