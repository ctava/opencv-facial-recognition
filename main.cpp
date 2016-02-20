#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>

#include "opencv2/opencv.hpp"

#include "faceprepreprocessor.h"
#include "facerecognizer.h"

using namespace cv;
using namespace std;

Mat loadAndPreprocessFace(FacePreprocessor facePreprocessor,  String imageURL)
{
    Mat imgMat = imread(imageURL, CV_LOAD_IMAGE_UNCHANGED);
    Rect faceRect;
    Point leftEye, rightEye;
    return facePreprocessor.getPreprocessedFace(imgMat, &faceRect, &leftEye, &rightEye);
}

void preprocessFacesAndSaveModel(FacePreprocessor facePreprocessor)
{
    cout << "preprocessFacesAndSaveModel " << endl;
    
    vector<Mat> preprocessedFaces;
    vector<int> faceLabels;
    faceLabels.push_back(1);
    faceLabels.push_back(1);
    faceLabels.push_back(99);
    faceLabels.push_back(99);
    preprocessedFaces.push_back(loadAndPreprocessFace(facePreprocessor,"./1.jpg"));
    preprocessedFaces.push_back(loadAndPreprocessFace(facePreprocessor,"./1.jpg"));
    preprocessedFaces.push_back(loadAndPreprocessFace(facePreprocessor,"./99.jpg"));
    preprocessedFaces.push_back(loadAndPreprocessFace(facePreprocessor,"./99.jpg"));

    FaceRecognizer facerecognizer;
    facerecognizer.learnCollectedFaces(preprocessedFaces, faceLabels);
    FileStorage fs( "./1-trainedModel.yml", FileStorage::WRITE );
    facerecognizer.save(fs);
}

void loadModelAndRecognizeFace(FacePreprocessor facePreprocessor)
{
    cout << "" << endl;
    cout << "loadModelAndRecognizeFace " << endl;
    
    FileStorage fs( "./1-trainedModel.yml", FileStorage::READ );
    FaceRecognizer facerecognizer;
    facerecognizer.load(fs);
    
    Mat mat = loadAndPreprocessFace(facePreprocessor,"./1.jpg");
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