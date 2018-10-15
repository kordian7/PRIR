#include <stdio.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "omp.h"

using namespace cv;
using namespace std;

const int maskSize = 5;
const int mask[maskSize][maskSize] =    {{1, 1, 1, 1, 1},
                                        {1, 1, 1, 1, 1},
                                        {1, 1, 1, 1, 1},
                                        {1, 1, 1, 1, 1},
                                        {1, 1, 1, 1, 1}};

int calculateNewPixelChannelValue(Mat channel, int row, int col) {
    int sum = 0;
    int maskWeight = 0;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            sum+= mask[i][j] * ((int) channel.at<uchar>(row + i - 2, col + j - 2));
            maskWeight += mask[i][j];
        }
    }
    return (int) (sum / maskWeight);
}

int main(int argc, char** argv )
{
    if ( argc != 4 )
    {
        printf("Pass threads number, image input, image output as args\n");
        return -1;
    }
	int threadsNumber;
	threadsNumber= strtol(argv[1], NULL, 10);
	omp_set_num_threads(threadsNumber);
    Mat img;
    Mat outputImg;
    img = imread( argv[2], 1 );
    if ( !img.data )
    {
        printf("Problem with image \n");
        return -1;
    }
    vector<Mat> rgbInputChannels;
    vector<Mat> rgbOutputChannels;
    split(img, rgbInputChannels);
    split(img, rgbOutputChannels);

	int margin = maskSize / 2;

	auto startTime = std::chrono::high_resolution_clock::now();
	int i,j;
	#pragma omp parallel for private(i,j) schedule(dynamic)
    for (i = margin; i < img.rows - margin; i++) {
        for (j = margin; j < img.cols - margin; j++) {
            rgbOutputChannels[0].at<uchar>(i,j) = calculateNewPixelChannelValue(rgbInputChannels[0], i, j);
            rgbOutputChannels[1].at<uchar>(i,j) = calculateNewPixelChannelValue(rgbInputChannels[1], i, j);
            rgbOutputChannels[2].at<uchar>(i,j) = calculateNewPixelChannelValue(rgbInputChannels[2], i, j);
        }
    }
	auto finishTime = std::chrono::high_resolution_clock::now();
	auto msDuration = std::chrono::duration_cast<std::chrono::milliseconds>(finishTime - startTime);
	cout << "Czas: " << msDuration.count() << "ms\n"	;
    merge(rgbOutputChannels, outputImg);
    imwrite(argv[3], outputImg);
    waitKey(0);
    return 0;
}
