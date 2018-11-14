#include <stdio.h>
#include <iostream>
#include <map>
#include <string>
#include <fstream>
#include <vector>

using namespace std;

#define ll long long

const int GRID_SIZE = 1;

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
std::vector<ll> readFile(char* arg){
    vector<ll> numbersFromFile;
    std::ifstream infile(arg);
    ll number;

     if(!infile.is_open()) {
        throw std::invalid_argument("Problem with file");
    }

    while (infile >> number) {
        numbersFromFile.push_back(number);
    }

    return numbersFromFile;
}

__global__ void calculate(ll *Arr, bool *results, int sizeOfArray, int amountOfBlocks){

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (amountOfBlocks >= sizeOfArray){
        results[x] += isPrime(Arr[x]);
    } else{
        int sizeOfPart = sizeOfArray / amountOfBlocks;
        int restOfDivide = sizeOfArray%amountOfBlocks;

        int startPart = sizeOfPart * x;
        int endPart = sizeOfPart * (x + 1);

        if (endPart <= sizeOfArray) 
        {
            int restStart = sizeOfPart * amountOfBlocks;

            for (int i = startPart; i < endPart; i++){
                results[i] += isPrime(Arr[i]);
            }

            if (x < restOfDivide){
                results[restStart + x] += isPrime(Arr[restStart + x]);
            }
        }
    }
}
 
int main(int argc, char** argv )
{
    float time;

    if ( argc < 2 )
    {
        printf("Pass file path\n");
        return -1;
    }

    vector<ll> numbersFromFile;

    try {
        numbersFromFile = readFile(argv[1]);
    } catch ( const std::invalid_argument& ex ) {
        cout << ex.what() << endl;
        return -1;
    }

    int sizeOfArray = numbersFromFile.size();
    int sizeToAllocateLongLong = sizeOfArray * sizeof(ll);
    int sizeToAllocateBool = sizeOfArray * sizeof(bool);
	
    ll numbersFromFileArr[sizeOfArray];
    std::copy(numbersFromFile.begin(), numbersFromFile.end(), numbersFromFileArr);

    bool* results = (bool *) malloc (sizeToAllocateBool);

    ll* c_arr;
    bool* c_results;

    cudaMalloc((void**) &c_arr, sizeToAllocateLongLong);
    cudaMalloc((void**) &c_results, sizeToAllocateBool);

    cudaMemcpy((void *)c_arr, (void *)numbersFromFileArr, sizeToAllocateLongLong, cudaMemcpyHostToDevice);

    //Start timer
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

    int amountOfBlocks = sizeOfArray;

    calculate<<<amountOfBlocks, GRID_SIZE>>>(c_arr, c_results, sizeOfArray, amountOfBlocks);

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

    for(int j = 0; j < sizeOfArray ; j++)
    {
        if (results[j]){
            cout << numbersFromFileArr[j] << " prime" << endl;
        } else {
            cout << numbersFromFileArr[j] << " composite" << endl;
        }       
    }

    free(results);
    return 0;
}
