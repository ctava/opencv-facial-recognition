#include "detectobjects.h"

void detectObjects(const cv::Mat &img, cv::CascadeClassifier &cascade, std::vector<cv::Rect> &objects, int scaledWidth, int flags, cv::Size minFeatureSize, float searchScaleFactor, int minNeighbors)
{
    cv::Mat gray;
    if (img.channels() == 3) {
        cvtColor(img, gray, CV_BGR2GRAY);
    }
    else if (img.channels() == 4) {
        cvtColor(img, gray, CV_BGRA2GRAY);
    }
    else {
        gray = img;
    }

    cv::Mat inputImg;
    float scale = img.cols / (float)scaledWidth;
    if (img.cols > scaledWidth) {
        int scaledHeight = cvRound(img.rows / scale);
        cv::resize(gray, inputImg, cv::Size(scaledWidth, scaledHeight));
        inputImg = gray;
    }
    else {
        inputImg = gray;
    }

    cv::Mat equalizedImg;
    equalizeHist(inputImg, equalizedImg);

    cascade.detectMultiScale(equalizedImg, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);

    if (img.cols > scaledWidth) {
        for (int i = 0; i < (int)objects.size(); i++ ) {
            objects[i].x = cvRound(objects[i].x * scale);
            objects[i].y = cvRound(objects[i].y * scale);
            objects[i].width = cvRound(objects[i].width * scale);
            objects[i].height = cvRound(objects[i].height * scale);
        }
    }

    for (int i = 0; i < (int)objects.size(); i++ ) {
        if (objects[i].x < 0)
            objects[i].x = 0;
        if (objects[i].y < 0)
            objects[i].y = 0;
        if (objects[i].x + objects[i].width > img.cols)
            objects[i].x = img.cols - objects[i].width;
        if (objects[i].y + objects[i].height > img.rows)
            objects[i].y = img.rows - objects[i].height;
    }
}

void findLargestObject(const cv::Mat &img, cv::CascadeClassifier &cascade, cv::Rect &largestObject, int scaledWidth)
{
    int flags = cv::CASCADE_FIND_BIGGEST_OBJECT;
    cv::Size minFeatureSize = cv::Size(20, 20);
    float searchScaleFactor = 1.1f;
    int minNeighbors = 3;
    std::vector<cv::Rect> objects;
    detectObjects(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
    if (objects.size() > 0) {
        largestObject = (cv::Rect)objects.at(0);
    }
    else {
        largestObject = cv::Rect(-1,-1,-1,-1);
    }
}