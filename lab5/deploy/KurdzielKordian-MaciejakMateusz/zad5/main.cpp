#include <stdio.h>
#include <iostream>
#include <map>
#include <string>
#include <fstream>
#include <vector>
#include "mpi.h"

#define ll long long

using namespace std;

const int MASTER_PROCESS = 0;
const int NUMBER_ARRAY_TAG = 0;
const int START_INDEX_TAG = 1;
const int END_INDEX_TAG = 2;
const int SIZE_OF_NUMBER_ARRAY_TAG = 3;
const int RESULT_ARRAY_TAG = 4;

const int MULTIPLIER_FOR_TAG_IDENTITY = 99;

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

bool isMainProcess(int rank){
    return rank == 0;
}

void send(void * data, int size, MPI_Datatype type = MPI_LONG_LONG, int target = 0, int tag = 0){
    MPI_Send(data, size, type, target, tag, MPI_COMM_WORLD);
}

void receive(void * data, int size, MPI_Datatype type = MPI_LONG_LONG, int from = 0, int tag = 0){
    MPI_Recv(data, size, type, from, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

int main(int argc, char** argv )
{
    int rank, numberOfProcesses, sizeOfArray;
    int startIndex, endIndex;
    double startTime, finishTime;
    ll primeNumber;
    bool isPrimeNumber;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);

    if (isMainProcess(rank)){
        vector<ll> numbersFromFile;

        if (numberOfProcesses < 1) {
            printf("Incorret number of threads\n");
            MPI_Finalize();
            return -1;
        }
            
        if ( argc != 2 ) {
            printf("Pass path to file with primes\n");
            MPI_Finalize();
            return -1;
        }
        try {
            numbersFromFile = readFile(argv[1]);
        } catch ( const std::invalid_argument& ex ) {
            cout << ex.what() << endl;
            MPI_Finalize();
            return -1;
        }
        ll numbersFromFileArr[numbersFromFile.size()];
        std::copy(numbersFromFile.begin(), numbersFromFile.end(), numbersFromFileArr);

        bool results[sizeof(numbersFromFileArr)/sizeof(*numbersFromFileArr)] ;

        int sizeOfNumberArray = sizeof(numbersFromFileArr)/sizeof(*numbersFromFileArr);
        int i, j, processId;

        startTime = MPI_Wtime();
        if (numberOfProcesses == 1){
            for (i = 0; i < sizeOfNumberArray; ++i){
                results[i] = isPrime(numbersFromFileArr[i]);
            }

        } else {
            int amountOfNumbers = sizeOfNumberArray / (numberOfProcesses - 1);
            int lastPartOfNumbers = amountOfNumbers + sizeOfNumberArray % (numberOfProcesses - 1);

            for (i = 1; i < numberOfProcesses; ++i){
            startIndex = (i - 1) * amountOfNumbers;
            endIndex = startIndex + amountOfNumbers;

            if (i == numberOfProcesses - 1)
                endIndex = startIndex + lastPartOfNumbers;
        
            processId = i;

            send(&startIndex, 1, MPI_INT, processId, START_INDEX_TAG);
		    send(&endIndex, 1, MPI_INT, processId, END_INDEX_TAG);
            send(&sizeOfNumberArray, 1, MPI_INT, processId, SIZE_OF_NUMBER_ARRAY_TAG);
            send(numbersFromFileArr, sizeOfNumberArray, MPI_LONG_LONG, processId, NUMBER_ARRAY_TAG);
        }

        for (i = 1; i < numberOfProcesses; ++i){
            processId = i;
            receive(&startIndex, 1, MPI_INT, processId, START_INDEX_TAG + processId * MULTIPLIER_FOR_TAG_IDENTITY);
		    receive(&endIndex, 1, MPI_INT, processId, END_INDEX_TAG + processId * MULTIPLIER_FOR_TAG_IDENTITY);
            bool receivedResultArr[endIndex - startIndex];

            receive(receivedResultArr, endIndex - startIndex, MPI_C_BOOL, processId, RESULT_ARRAY_TAG + processId * MULTIPLIER_FOR_TAG_IDENTITY);

            int temp = 0;
            for (int k = startIndex; k < endIndex; ++k){
                results[k] = receivedResultArr[temp++];
            }
        }
        }

        // Finish, print time and results

        finishTime = MPI_Wtime();
        cout << "Czas: " << ((finishTime - startTime) * 1000) << "ms\n"	;

        for(int j = 0; j < sizeof(numbersFromFileArr)/sizeof(*numbersFromFileArr) ; j++)
        {
            if (results[j]){
               cout << numbersFromFileArr[j] << " prime" << endl;
            } else {
               cout << numbersFromFileArr[j] << " composite" << endl;
            }
            
        }

    } else { // processes receive numbera do a verdict and send back result

        receive(&startIndex, 1, MPI_INT, MASTER_PROCESS, START_INDEX_TAG);
		receive(&endIndex, 1, MPI_INT, MASTER_PROCESS, END_INDEX_TAG);
        receive(&sizeOfArray, 1, MPI_INT, MASTER_PROCESS, SIZE_OF_NUMBER_ARRAY_TAG);

        ll arrayWithNumbers[sizeOfArray];

        receive(arrayWithNumbers, sizeOfArray, MPI_LONG_LONG, MASTER_PROCESS, NUMBER_ARRAY_TAG);
        bool resultArr[endIndex-startIndex];
        
        int temp = 0;
        for (int i = startIndex; i < endIndex; ++i){
            resultArr[temp++] = isPrime(arrayWithNumbers[i]);
        }

        send(&startIndex, 1, MPI_INT, MASTER_PROCESS, START_INDEX_TAG + rank * MULTIPLIER_FOR_TAG_IDENTITY);
		send(&endIndex, 1, MPI_INT, MASTER_PROCESS, END_INDEX_TAG + rank * MULTIPLIER_FOR_TAG_IDENTITY);
        send(resultArr, endIndex - startIndex, MPI_C_BOOL, MASTER_PROCESS, RESULT_ARRAY_TAG + rank * MULTIPLIER_FOR_TAG_IDENTITY);
    }

	MPI_Finalize();



    return 0;
}
