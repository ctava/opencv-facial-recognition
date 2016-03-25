#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>

#include "opencv2/opencv.hpp"

#include "facepreprocessor.h"
#include "facerecognizer.h"

using namespace cv;
using namespace std;

Mat loadAndPreprocessFace(FacePreprocessor facePreprocessor,  String imageURL)
{
    Mat img = imread(imageURL, CV_LOAD_IMAGE_UNCHANGED);
    const int DETECTION_WIDTH = 320;
    // Possibly shrink the image, to run much faster.
    Mat smallImg;
    float scale = img.cols / (float) DETECTION_WIDTH;
    if (img.cols > DETECTION_WIDTH) {
        // Shrink the image while keeping the same aspect ratio. int scaledHeight = cvRound(img.rows / scale);
        int scaledHeight = cvRound(img.rows / scale);
        resize(img, smallImg, Size(DETECTION_WIDTH, scaledHeight));
    }
    else {
        // Access the input directly since it is already small.
        smallImg = img;
    }
    return facePreprocessor.getPreprocessedFace(smallImg);
}

void preprocessFacesAndSaveModel(FacePreprocessor facePreprocessor)
{
    cout << "preprocessFacesAndSaveModel " << endl;
    
    vector<Mat> preprocessedFaces;
    vector<int> faceLabels;
    cv::Mat preprocessedFace;
    
    
    preprocessedFace = loadAndPreprocessFace(facePreprocessor,"./4-1.jpg");
    if (preprocessedFace.total() != 0) {
        preprocessedFaces.push_back(preprocessedFace);
        faceLabels.push_back(4);
    }
    
    preprocessedFace = loadAndPreprocessFace(facePreprocessor,"./1.jpg");
    if (preprocessedFace.total() != 0) {
        preprocessedFaces.push_back(preprocessedFace);
        faceLabels.push_back(1);
    }
    
//    preprocessedFace = loadAndPreprocessFace(facePreprocessor,"./1.jpg");
//    if (preprocessedFace.total() != 0) {
//        preprocessedFaces.push_back(preprocessedFace);
//        faceLabels.push_back(1);
//    }

    
    preprocessedFace = loadAndPreprocessFace(facePreprocessor,"./2.jpg");
    if (preprocessedFace.total() != 0) {
        preprocessedFaces.push_back(preprocessedFace);
        faceLabels.push_back(2);
    }
    
    preprocessedFace = loadAndPreprocessFace(facePreprocessor,"./3.jpg");
    if (preprocessedFace.total() != 0) {
        preprocessedFaces.push_back(preprocessedFace);
        faceLabels.push_back(3);
    }
    
//    
//    preprocessedFace = loadAndPreprocessFace(facePreprocessor,"./1-15.png");
//    if (preprocessedFace.total() != 0) {
//        preprocessedFaces.push_back(preprocessedFace);
//        faceLabels.push_back(1);
//    }
    
    preprocessedFace = loadAndPreprocessFace(facePreprocessor,"./98.jpg");
    if (preprocessedFace.total() != 0) {
        preprocessedFaces.push_back(preprocessedFace);
        faceLabels.push_back(98);
    }
    
    preprocessedFace = loadAndPreprocessFace(facePreprocessor,"./99.jpg");
    if (preprocessedFace.total() != 0) {
        preprocessedFaces.push_back(preprocessedFace);
        faceLabels.push_back(99);
    }

    FaceRecognizer facerecognizer;
    facerecognizer.learnCollectedFaces(preprocessedFaces, faceLabels);
    FileStorage fs( "./facerecmodel.yml", FileStorage::WRITE );
    facerecognizer.save(fs);
}

void loadModelAndRecognizeFace(FacePreprocessor facePreprocessor)
{
    cout << "" << endl;
    cout << "loadModelAndRecognizeFace " << endl;
    
    FileStorage fs( "./facerecmodel.yml", FileStorage::READ );
    FaceRecognizer facerecognizer;
    facerecognizer.load(fs);

    Mat mat = loadAndPreprocessFace(facePreprocessor,"./98.jpg");
    int prediction = facerecognizer.predict(mat);
    
    cout << "Predicted identity = " << prediction << "." << endl << endl;
}

int main(int argc, char *argv[])
{
    cout << "Compiled with OpenCV version " << CV_VERSION << endl << endl;
        
    FacePreprocessor facePreprocessor;
    facePreprocessor.initCascadeClassifiers();
    
    preprocessFacesAndSaveModel(facePreprocessor);
    
    loadModelAndRecognizeFace(facePreprocessor);
    
    return 0;
}