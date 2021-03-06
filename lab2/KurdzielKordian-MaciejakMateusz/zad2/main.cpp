#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "mpi.h"

using namespace cv;
using namespace std;

const int MASTER_PROCESS = 0;
const int START_VALUE_TAG = 0;
const int END_VALUE_TAG = 1;
const int DATA_TAG = 2;
const int MULTIPLY_ID = 100;

MPI_Comm commmunicate = MPI_COMM_WORLD;

const int maskSize = 5;
const int mask[maskSize][maskSize] =    {
										{1, 1, 2, 1, 1},
                                        {1, 2, 4, 2, 1},
                                        {1, 4, 8, 4, 2},
                                        {1, 2, 4, 2, 1},
                                        {1, 1, 2, 1, 1}};

int maskWeight;
int margin = maskSize / 2;

void initMask() {
	maskWeight = 0;
    for (int i = 0; i < maskSize; ++i) {
        for (int j = 0; j < maskSize; ++j) {
            maskWeight += mask[i][j];
        }
    }
}

int calculateNewPixelChannelValue(Mat channel, int row, int col) {
    int sum = 0;
    for (int i = 0; i < maskSize; ++i) {
        for (int j = 0; j < maskSize; ++j) {
            sum+= mask[i][j] * ((int) channel.at<uchar>(row + i - 2, col + j - 2));
        }
    }
    return (int) (sum / maskWeight);
}

bool isMainProcess(int rank){
    return rank == 0;
}

void send(void * data, int size, MPI_Datatype type = MPI_LONG, int target = 0, int tag = 0){
    MPI_Send(data, size, type, target, tag, commmunicate);
}

void receive(void * data, int size, MPI_Datatype type = MPI_LONG, int target = 0, int tag = 0){
    MPI_Recv(data, size, type, target, tag, commmunicate, MPI_STATUS_IGNORE);
}

Mat cutPicture(Mat picture, int x, int y, int width, int height){
	cv::Rect myROI(x, y, width, height);
	cv::Mat croppedRef(picture, myROI);
	cv::Mat cropped;
	croppedRef.copyTo(cropped);
    return cropped;
}

Mat addBorders(Mat picture) {
	Mat copy;
	picture.copyTo(copy);
	copyMakeBorder(picture, copy, margin, margin, margin, margin, BORDER_REPLICATE);
	return copy;
}

Mat removeBorders(Mat picture) {
	return cutPicture(picture, margin, margin, picture.cols - 2 * margin, picture.rows - 2 * margin);
}

Mat gauss(Mat img) {
    Mat output = img.clone();
    vector<Mat> rgbInput;   
    vector<Mat> rgbOutput;
    
    split(img, rgbInput);
    split(img, rgbOutput);

    for (int i = margin; i < img.rows - margin; ++i) {
        for (int j = margin; j < img.cols - margin; ++j) {
            rgbOutput[0].at<uchar>(i,j) = calculateNewPixelChannelValue(rgbInput[0], i, j);
            rgbOutput[1].at<uchar>(i,j) = calculateNewPixelChannelValue(rgbInput[1], i, j);
            rgbOutput[2].at<uchar>(i,j) = calculateNewPixelChannelValue(rgbInput[2], i, j);
        }
    }
    merge(rgbOutput, output);

    return output;
}
                                        
int main(int argc, char** argv )
{
    string srcImagePath, dstImagePath;
    Mat inputImg, outputImg, partOfImg;
    int rank, numberOfProcesses;
    double startTime, finishTime;
    int x, y, imgSplitSize;
	initMask();

    MPI_Init(&argc, &argv);
    MPI_Comm_rank (commmunicate, &rank);
    MPI_Comm_size(commmunicate, &numberOfProcesses);

    if (isMainProcess(rank)){

		if (numberOfProcesses < 2) {
            printf("Incorret number of threads\n");
			MPI_Finalize();
            return -1;
		}
        
        if ( argc != 3 )
        {
            printf("Pass image input, image output as args\n");
            MPI_Finalize();
            return -1;
        }

        srcImagePath = argv[1];
        dstImagePath = argv[2];

        inputImg = imread( srcImagePath, CV_LOAD_IMAGE_COLOR);

        if ( !inputImg.data )
        {
            printf("Problem with image \n");
            MPI_Finalize();
            return -1;
        }

        initMask();

        imgSplitSize = inputImg.rows / (numberOfProcesses - 1);
		int lastImgSplitSize = imgSplitSize + inputImg.rows % (numberOfProcesses - 1);
        startTime = MPI_Wtime();

        // Split picture and send to other processes
        for (int i = 1; i < numberOfProcesses; ++i){
			if (i != numberOfProcesses - 1) {
            	partOfImg = cutPicture(inputImg, 0, (i -1) * imgSplitSize, inputImg.cols, imgSplitSize);
			} else {
				partOfImg = cutPicture(inputImg, 0, (i -1) * imgSplitSize, inputImg.cols, lastImgSplitSize);
			}
			partOfImg = addBorders(partOfImg);
            x = partOfImg.rows;
            y = partOfImg.cols;
            
            send(&x, 1, MPI_INT, i, START_VALUE_TAG);
            send(&y, 1, MPI_INT, i, END_VALUE_TAG);
            send(partOfImg.data, x * y * 3, MPI_BYTE, i, DATA_TAG);
        }

        outputImg = Mat(0, 0, CV_8UC3);

        // Get filtered parts and merge to image
        for(int i = 1; i < numberOfProcesses; i++) {
            receive(&x, 1, MPI_INT, i, MULTIPLY_ID*i+START_VALUE_TAG);
            receive(&y, 1, MPI_INT, i, MULTIPLY_ID*i+END_VALUE_TAG);
			
            partOfImg = Mat(x, y, CV_8UC3);
            receive(partOfImg.data, x*y*3, MPI_BYTE, i, MULTIPLY_ID*i+DATA_TAG);
			partOfImg = removeBorders(partOfImg);

            outputImg.push_back(partOfImg);
	    }

    } else {

        receive(&x, 1, MPI_INT, MASTER_PROCESS, START_VALUE_TAG);
        receive(&y, 1, MPI_INT, MASTER_PROCESS, END_VALUE_TAG);

        partOfImg = Mat(x, y, CV_8UC3);
        receive(partOfImg.data, x * y * 3, MPI_BYTE, MASTER_PROCESS, DATA_TAG);

        partOfImg = gauss(partOfImg);
    
        send(&x, 1, MPI_INT, MASTER_PROCESS, MULTIPLY_ID * rank + START_VALUE_TAG);
        send(&y, 1, MPI_INT, MASTER_PROCESS, MULTIPLY_ID * rank + END_VALUE_TAG);
        send(partOfImg.data, x*y*3, MPI_BYTE, MASTER_PROCESS, MULTIPLY_ID * rank + DATA_TAG);
    }
    
    if (isMainProcess(rank)) {
        finishTime = MPI_Wtime();
        cout << "Czas: " << ((finishTime - startTime) * 1000) << "ms\n"	;

        imwrite(dstImagePath, outputImg);
        waitKey(0);
    }

    MPI_Finalize();
    
    return 0;
}
