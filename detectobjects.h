#include <stdio.h>
#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"

void findLargestObject(const cv::Mat &img, cv::CascadeClassifier &cascade, cv::Rect &largestObject, int scaledWidth = 320);