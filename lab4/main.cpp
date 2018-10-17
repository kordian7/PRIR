#include <stdio.h>
#include <iostream>
#include <map>
#include <string>
#include <fstream>
#include <vector>
#include "omp.h"

#define ll long long

using namespace std;

// Use naive method
bool isPrime(ll n)
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

int main(int argc, char** argv )
{
    if ( argc != 3 )
    {
        printf("Pass number of threads and file path\n");
        return -1;
    }
	int numberOfThreads;
	numberOfThreads= strtol(argv[1], NULL, 10);

    vector<ll> numbersFromFile = reaadFile(argv[2]);
	
    // Copy to array because std::vector is not thread safe
    ll numbersFromFileArr[numbersFromFile.size()];
    std::copy(numbersFromFile.begin(), numbersFromFile.end(), numbersFromFileArr);

    unsigned int i;
    bool results[sizeof(numbersFromFileArr)/sizeof(*numbersFromFileArr)] ;
    auto startTime = omp_get_wtime();

    #pragma omp parallel for default(none) shared(results, numbersFromFileArr) private(i) schedule(dynamic) num_threads(numberOfThreads)
    for (i = 0 ; i < sizeof(numbersFromFileArr)/sizeof(*numbersFromFileArr) ; i++) {
        if (isPrime(numbersFromFileArr[i])){
            results[i] = true;
        } else {
            results[i] = false;
        }
    }

	auto finishTime = omp_get_wtime();
	cout << "Time: " << ((finishTime - startTime) * 1000) << "ms\n"	;

    // Print results
    for(int j = 0; j < sizeof(numbersFromFileArr)/sizeof(*numbersFromFileArr) ; j++)
    {
        if (results[j]){
            cout << numbersFromFileArr[j] << " prime" << endl;
        } else {
             cout << numbersFromFileArr[j] << " composite" << endl;
        }
        
    }

    return 0;
}
