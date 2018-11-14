#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <cuda.h>

using namespace std;
using namespace cv;

const int BLOCK_SIZE = 1024;
const int GRID_SIZE = 256;

const int maskSize = 5;
const int mask[maskSize][maskSize] =    {
										{1, 1, 2, 1, 1},
                                        {1, 2, 4, 2, 1},
                                        {1, 4, 8, 4, 2},
                                        {1, 2, 4, 2, 1},
                                        {1, 1, 2, 1, 1}};
int maskWeight;
int margin;

__constant__ int globalMask[maskSize][maskSize];
__constant__ int deviceMaskWeight;

void initMask() {
	margin = maskSize / 2;
	maskWeight = 0;
    for (int i = 0; i < maskSize; ++i) {
        for (int j = 0; j < maskSize; ++j) {
            maskWeight += mask[i][j];
        }
    }
}

__global__ void gaussianBlurCuda(unsigned char* imageSrc, unsigned char* imageOut, int rows, int rowSizeInBytes, int channelsSize) {
    int startByteVal = blockIdx.x * blockDim.x + threadIdx.x;
    int totalImageSize = rows * rowSizeInBytes;
    int jump = blockDim.x * gridDim.x;

    for (int byteCounter = startByteVal; byteCounter < totalImageSize; byteCounter += jump)
    {   

        int col = (byteCounter % rowSizeInBytes) / channelsSize;

        // cut rows
        if (byteCounter < 2 * rowSizeInBytes) continue;
        if (byteCounter > totalImageSize - 2 * rowSizeInBytes) continue;

        // cut columns
        if (col < 2) continue;
        if (col >= (rowSizeInBytes / channelsSize) - 2) continue;

        int sum = 0;
    
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                sum += globalMask[i][j] * imageSrc[byteCounter + (i - 2) * rowSizeInBytes + (j - 2) * channelsSize];
            }
        }           
        imageOut[byteCounter] = (int) (sum / deviceMaskWeight);
    }
}


    
int main(int argc, char** argv )
{
    if ( argc != 3 )
    {
        printf("Bad args number\n");
        return -1;
    }

	string srcImagePath = argv[1];
	string dstImagePath = argv[2];

    Mat inputImg, outputImg;
    inputImg = imread( srcImagePath, CV_LOAD_IMAGE_COLOR);
    if ( !inputImg.data )
    {
        printf("Problem with image \n");
        return -1;
    }
    outputImg = inputImg.clone();
	initMask();

    cudaMemcpyToSymbol(globalMask, mask, 25 * sizeof(int));
    cudaMemcpyToSymbol(deviceMaskWeight, &maskWeight, sizeof(maskWeight));

    cudaEvent_t startEvent, endEvent;
    float elapsedTime;
    int rowLengthInBytes = inputImg.step;
    // TODO - to check * sizeof byte???
    int totalImageSize = rowLengthInBytes * inputImg.rows * sizeof (unsigned char);

    unsigned char *cudaSrcImage, *cudaOutImage;

    cudaMalloc<unsigned char> (&cudaSrcImage, totalImageSize);
    cudaMalloc<unsigned char> (&cudaOutImage, totalImageSize);

    // cudaMemcpyHostToDevice - enum
    cudaMemcpy(cudaSrcImage, inputImg.ptr(), totalImageSize, cudaMemcpyHostToDevice);

    dim3 dimGrid = dim3(GRID_SIZE);
    dim3 dimBlock = dim3(BLOCK_SIZE);

    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);

    cudaEventRecord(startEvent, NULL);
    gaussianBlurCuda<<<dimGrid, dimBlock>>>(cudaSrcImage, cudaOutImage, inputImg.rows, rowLengthInBytes, inputImg.channels());
    cudaEventRecord(endEvent, NULL);
    cudaEventSynchronize(endEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, endEvent);

    cudaMemcpy(outputImg.ptr(), cudaOutImage, totalImageSize, cudaMemcpyDeviceToHost);

    cudaFree(cudaSrcImage);
    cudaFree(cudaOutImage);

    imwrite(dstImagePath, outputImg);

	cout << "Czas: " << elapsedTime << "ms\n";
    waitKey(0);
    return 0;
}
