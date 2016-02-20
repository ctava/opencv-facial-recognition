#include <stdio.h>
#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

void findLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth = 320);