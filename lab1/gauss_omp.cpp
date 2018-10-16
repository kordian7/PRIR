#include <stdio.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "omp.h"

using namespace cv;
using namespace std;

const int maskSize = 5;
const int mask[maskSize][maskSize] =    {
										{1, 1, 2, 1, 1},
                                        {1, 2, 4, 2, 1},
                                        {1, 4, 8, 4, 2},
                                        {1, 2, 4, 2, 1},
                                        {1, 1, 2, 1, 1}};
int maskWeight;
int margin;

int calculateNewPixelChannelValue(Mat channel, int row, int col) {
    int sum = 0;
    for (int i = 0; i < maskSize; ++i) {
        for (int j = 0; j < maskSize; ++j) {
            sum+= mask[i][j] * ((int) channel.at<uchar>(row + i - 2, col + j - 2));
        }
    }
    return (int) (sum / maskWeight);
}

void initMask() {
	margin = maskSize / 2;
	maskWeight = 0;
    for (int i = 0; i < maskSize; ++i) {
        for (int j = 0; j < maskSize; ++j) {
            maskWeight += mask[i][j];
        }
    }
}

int main(int argc, char** argv )
{
    if ( argc != 4 )
    {
        printf("Pass threads number, image input, image output as args\n");
        return -1;
    }
	int threadsNumber;
	double startTime, finishTime;
	threadsNumber= strtol(argv[1], NULL, 10);
	string srcImagePath = argv[2];
	string dstImagePath = argv[3];

    Mat inputImg, outputImg;
    inputImg = imread( srcImagePath, CV_LOAD_IMAGE_COLOR);
    if ( !inputImg.data )
    {
        printf("Problem with image \n");
        return -1;
    }
    vector<Mat> rgbInputChannels;
    vector<Mat> rgbOutputChannels;
    split(inputImg, rgbInputChannels);
    split(inputImg, rgbOutputChannels);

	initMask();

    startTime = omp_get_wtime();
	int i,j;
	#pragma omp parallel for default(shared) private(i,j) schedule(runtime) num_threads(threadsNumber)
    for (i = margin; i < inputImg.rows - margin; i++) {
        for (j = margin; j < inputImg.cols - margin; j++) {
            rgbOutputChannels[0].at<uchar>(i,j) = calculateNewPixelChannelValue(rgbInputChannels[0], i, j);
            rgbOutputChannels[1].at<uchar>(i,j) = calculateNewPixelChannelValue(rgbInputChannels[1], i, j);
            rgbOutputChannels[2].at<uchar>(i,j) = calculateNewPixelChannelValue(rgbInputChannels[2], i, j);
        }
    }
    finishTime = omp_get_wtime();
	cout << "Czas: " << ((finishTime - startTime) * 1000) << "ms\n"	;
    merge(rgbOutputChannels, outputImg);
    imwrite(dstImagePath, outputImg);
    waitKey(0);
    return 0;
}
