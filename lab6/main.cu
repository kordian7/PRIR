#include <stdio.h>
#include <iostream>
#include <map>
#include <string>
#include <fstream>
#include <vector>

using namespace std;

#define ll long long

const int BLOCK_SIZE = 2;
const int GRID_SIZE = 32;

// Use naive method
__device__ bool isPrime(ll n)
{
    if(n<2)
        return false;
        
    for(ll i=2;i*i<=n;i++)
        if(n%i==0)
            return false; 

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

    long x = blockIdx.x * blockDim.x + threadIdx.x;
	long y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < sizeOfArray && y < sizeOfArray) 
	{
		for (long k = 0; k < sizeOfArray; k++) {
			results[x + sizeOfArray * y] += isPrime(Arr[x + sizeOfArray * y]);
		}
	}

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
	
    ll numbersFromFileArr[sizeOfArray];
    std::copy(numbersFromFile.begin(), numbersFromFile.end(), numbersFromFileArr);

    unsigned int i;
    bool results[sizeOfArray];

    ll* c_arr;
    bool* c_results;

    cudaMemcpy(c_arr, numbersFromFileArr, sizeOfArray, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &c_results, sizeOfArray);

    dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grids(GRID_SIZE, GRID_SIZE);

    //Start timer
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

    //results and numberFromFileArr to send into kernel


    calculate<<<blocks, grids>>>(c_arr, c_results, sizeOfArray);

    //End timer and put result into time variable
    cudaDeviceSynchronize();			 
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

    
    printf("Czas: %.4fms\n", time);

    if (cudaMemcpy(results, c_results , sizeOfArray, cudaMemcpyDeviceToHost) != cudaSuccess) {
		cout<<"GPU to CPU copy error\n";
	}

    cudaFree(c_arr);
    cudaFree(c_results);

    for (int i = 0; i < sizeOfArray; i++){
        if (results[i]){
            cout << numbersFromFileArr[i] << " prime" << endl;
        } else {
            cout << numbersFromFileArr[i] << " composite" << endl;
        }
    }

    return 0;
}
