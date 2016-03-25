#include <stdio.h>
#include <iostream>
#include <list>
#include <vector>

#include "opencv2/opencv.hpp"

class FacePreprocessor
{
private:
    cv::CascadeClassifier faceCascade;
    cv::CascadeClassifier eyeCascade1;
    cv::CascadeClassifier eyeCascade2;
    void detectBothEyes(const cv::Mat &face, cv::Point &leftEye, cv::Point &rightEye);
    void equalizeLeftAndRightHalves(cv::Mat &faceImg);
public:
    void initCascadeClassifiers();
    cv::Mat getPreprocessedFace(cv::Mat srcImg);
};