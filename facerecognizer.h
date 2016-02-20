#include <stdio.h>
#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class FaceRecognizer : public Algorithm
{
private:
    int _num_components;
    double _threshold;
    Mat _eigenvectors;
    Mat _eigenvalues;
    Mat _mean;
    vector<Mat> _projections;
    Mat _labels;
    double getSimilarity(const Mat A, const Mat B);
    void train(InputArrayOfArrays src, InputArray _lbls);
    
public:
    Mat getMean();
    Mat getEigenvectors();
    Mat getEigenvalues();
    vector<Mat> getProjections();
    
    void learnCollectedFaces(const vector<Mat> preprocessedFaces, const vector<int> faceLabels, const string facerecAlgorithm = "FaceRecognizer.Fisherfaces");
    Mat reconstructFace(FaceRecognizer faceRecognizer, const Mat preprocessedFace);
    
    int predict(InputArray src);

    void load(const FileStorage& fs);

    void save(FileStorage& fs);
};