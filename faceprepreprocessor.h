#include <stdio.h>
#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class FacePreprocessor
{
private:
    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade1;
    CascadeClassifier eyeCascade2;
    void detectBothEyes(const Mat &face, Point &leftEye, Point &rightEye);
    void equalizeLeftAndRightHalves(Mat &faceImg);
public:
    void initCascadeClassifiers();
    Mat getPreprocessedFace(Mat &srcImg, Rect *storeFaceRect = NULL, Point *storeLeftEye = NULL, Point *storeRightEye = NULL);
};