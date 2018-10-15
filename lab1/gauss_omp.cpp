#include <stdio.h>
#include <opencv2/opencv.hpp>

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
    if ( argc != 3 )
    {
        printf("Pass image input and output as args");
        return -1;
    }
    Mat img;
    Mat outputImg;
    img = imread( argv[1], 1 );
    if ( !img.data )
    {
        printf("Problem with image \n");
        return -1;
    }
    vector<Mat> rgbInputChannels;
    vector<Mat> rgbOutputChannels;
    split(img, rgbInputChannels);
    split(img, rgbOutputChannels);

    for (int i = 2; i < img.rows - 2; i++) {
        for (int j = 2; j < img.cols - 2; j++) {
            rgbOutputChannels[0].at<uchar>(i,j) = calculateNewPixelChannelValue(rgbInputChannels[0], i, j);
            rgbOutputChannels[1].at<uchar>(i,j) = calculateNewPixelChannelValue(rgbInputChannels[1], i, j);
            rgbOutputChannels[2].at<uchar>(i,j) = calculateNewPixelChannelValue(rgbInputChannels[2], i, j);
        }
    }
    merge(rgbOutputChannels, outputImg);
    imwrite(argv[2], outputImg);
    waitKey(0);
    return 0;
}
