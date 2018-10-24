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
    cout << "CUT X: " << x << endl;
    cout << "CUT y: " << y << endl;
    cout << "CUT w: " << width << endl;
    cout << "CUT h: " << height << endl;

	cv::Rect myROI(x, y, width, height);

	cv::Mat croppedRef(picture, myROI);
	cv::Mat cropped;
	croppedRef.copyTo(cropped);
	
    cout << "cropped img rows: " << cropped.rows << endl;
    cout << "cropped img cols: " << cropped.cols << endl; 
    return cropped;
}

Mat gauss(Mat img) {
    Mat output = img.clone();
    vector<Mat> rgbInput;   
    vector<Mat> rgbOutput;
    
    split(img, rgbInput);
    split(img, rgbOutput);

    cout << "img1 rows: " << img.rows << endl;
    cout << "img1 cols: " << img.cols << endl; 

    for (int i = margin; i < img.rows - margin; ++i) {
        for (int j = margin; j < img.cols - margin; ++j) {
            rgbOutput[0].at<uchar>(i,j) = calculateNewPixelChannelValue(rgbInput[0], i, j);
            rgbOutput[1].at<uchar>(i,j) = calculateNewPixelChannelValue(rgbInput[1], i, j);
            rgbOutput[2].at<uchar>(i,j) = calculateNewPixelChannelValue(rgbInput[2], i, j);
        }
    }
    merge(rgbOutput, output);

	cout << "img2 rows: " << output.rows << endl;
    cout << "img2 cols: " << output.cols << endl; 
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
        
        if ( argc != 3 )
        {
            printf("Pass image input, image output as args\n");
            MPI_Finalize();
            return -1;
        }

        srcImagePath = argv[1];
        dstImagePath = argv[2];

        inputImg = imread( srcImagePath, CV_LOAD_IMAGE_COLOR);

        cout << "ORIGINAL IMAGE ROWS: " << inputImg.rows << endl;
        cout << "ORIGINAL IMAGE COLS: " << inputImg.cols << endl;



        if ( !inputImg.data )
        {
            printf("Problem with image \n");
            MPI_Finalize();
            return -1;
        }

        initMask();

        imgSplitSize = inputImg.rows / (numberOfProcesses - 1);
        startTime = MPI_Wtime();

        cout << "BEFORE FOR LOOP MAIN PROCESS " << rank << endl;
        cout << "NUMBER OF PROCESSES:  " << numberOfProcesses << endl;

        // Split picture and send to other processes
        for (int i = 1; i < numberOfProcesses; ++i){

            partOfImg = cutPicture(inputImg, 0, (i -1) * imgSplitSize, inputImg.cols, imgSplitSize);

            x = partOfImg.rows;
            y = partOfImg.cols;

            cout << "X " << x << endl;
            cout << "Y " << y << endl;
            cout << "DATA: " << partOfImg.data << endl;
            
            send(&x, 1, MPI_INT, i, START_VALUE_TAG);
            send(&y, 1, MPI_INT, i, END_VALUE_TAG);
            send(partOfImg.data, x * y * 3, MPI_BYTE, i, DATA_TAG);
        }

        outputImg = Mat(0, 0, CV_8UC3);

        // Get filtered parts and merge to image
        for(int i = 1; i < numberOfProcesses; i++) {
            //receive(&x, 1, MPI_INT, i, MULTIPLY_ID*i+START_VALUE_TAG);
            //receive(&y, 1, MPI_INT, i, MULTIPLY_ID*i+END_VALUE_TAG);
            partOfImg = Mat(x, y, CV_8UC3);

            receive(partOfImg.data, x*y*3, MPI_BYTE, i, MULTIPLY_ID*i+DATA_TAG);
            outputImg.push_back(partOfImg);
	    }

    } else {
        cout << "SLAVE PROCESSESS: " << rank << endl;

        receive(&x, 1, MPI_INT, MASTER_PROCESS, START_VALUE_TAG);
        receive(&y, 1, MPI_INT, MASTER_PROCESS, END_VALUE_TAG);

        cout << "RECEIVE START: " << x << endl;
        cout << "RECEIVE END: " << y << endl;

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
