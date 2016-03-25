#include <stdio.h>
#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"

class FaceRecognizer : public cv::Algorithm
{
private:
    int _num_components;
    double _threshold;
    cv::Mat _eigenvectors;
    cv::Mat _eigenvalues;
    cv::Mat _mean;
    std::vector<cv::Mat> _projections;
    cv::Mat _labels;
    double getSimilarity(const cv::Mat A, const cv::Mat B);
    cv::Mat reconstructFace(const cv::Mat preprocessedFace);
    void train(cv::InputArrayOfArrays src, cv::InputArray _lbls);
public:
    cv::Mat getMean();
    cv::Mat getEigenvectors();
    cv::Mat getEigenvalues();
    std::vector <cv::Mat> getProjections();
    
    void learnCollectedFaces(const std::vector<cv::Mat> preprocessedFaces, const std::vector<int> faceLabels, const std::string facerecAlgorithm = "FaceRecognizer.Fisherfaces");
    cv::Mat reconstructFace(FaceRecognizer faceRecognizer, const cv::Mat preprocessedFace);
    
    int predict(cv::InputArray _src);

    void load(const cv::FileStorage& fs);

    void save(cv::FileStorage& fs);
};