#include "facerecognizer.h"
#include <set>

template<typename _Tp>
inline void readFileNodeList(const FileNode& fn, vector<_Tp>& result) {
    if (fn.type() == FileNode::SEQ) {
        for (FileNodeIterator it = fn.begin(); it != fn.end();) {
            _Tp item;
            it >> item;
            result.push_back(item);
        }
    }
}

template<typename _Tp>
inline void writeFileNodeList(FileStorage& fs, const string& name,
                              const vector<_Tp>& items) {

    typedef typename vector<_Tp>::const_iterator constVecIterator;
    fs << name << "[";
    for (constVecIterator it = items.begin(); it != items.end(); ++it) {
        fs << *it;
    }
    fs << "]";
}

template<typename _Tp>
inline vector<_Tp> remove_dups(const vector<_Tp>& src) {
    typedef typename set<_Tp>::const_iterator constSetIterator;
    typedef typename vector<_Tp>::const_iterator constVecIterator;
    set<_Tp> set_elems;
    for (constVecIterator it = src.begin(); it != src.end(); ++it)
        set_elems.insert(*it);
    vector<_Tp> elems;
    for (constSetIterator it = set_elems.begin(); it != set_elems.end(); ++it)
        elems.push_back(*it);
    return elems;
}

static Mat asRowMatrix(InputArrayOfArrays src, int rtype, double alpha=1, double beta=0) {
    if(src.kind() != _InputArray::STD_VECTOR_MAT && src.kind() != _InputArray::STD_VECTOR_VECTOR) {
        string error_message = "Data is expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< vector<...> >).";
        CV_Error(CV_StsBadArg, error_message);
    }
    size_t nsamples = src.total();
    if(nsamples == 0)
        return Mat();
    size_t dimensionOfSamples = src.getMat(0).total();
    Mat data((int)nsamples, (int)dimensionOfSamples, rtype);
    for(unsigned int i = 0; i < nsamples; i++) {
        if(src.getMat(i).total() != dimensionOfSamples) {
            string error_message = format("Incorrent number of elements in matrix #%d! Expected %d was %d.", i, dimensionOfSamples, src.getMat(i).total());
            CV_Error(CV_StsBadArg, error_message);
        }
        Mat xi = data.row(i);
        if(src.getMat(i).isContinuous()) {
            src.getMat(i).reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src.getMat(i).clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

Mat FaceRecognizer::getMean() {
    return _mean;
}
Mat FaceRecognizer::getEigenvectors() {
    return _eigenvectors;
}

Mat FaceRecognizer::getEigenvalues() {
    return _eigenvalues;
}

vector<Mat> FaceRecognizer::getProjections() {
    return _projections;
}

Mat FaceRecognizer::reconstructFace(FaceRecognizer faceRecognizer, const Mat preprocessedFace)
{
    try {
        Mat eigenvectors = faceRecognizer.getEigenvectors();
        Mat averageFaceRow = faceRecognizer.getMean();
        int faceHeight = preprocessedFace.rows;
        Mat projection = cv::LDA::subspaceProject(eigenvectors, averageFaceRow, preprocessedFace.reshape(1,1));
        Mat reconstructionRow = cv::LDA::subspaceReconstruct (eigenvectors, averageFaceRow, projection);
        Mat reconstructionMat = reconstructionRow.reshape(1, faceHeight);
        Mat reconstructedFace = Mat(reconstructionMat.size(), CV_8U);
        reconstructionMat.convertTo(reconstructedFace, CV_8U, 1, 0);
        return reconstructedFace;
    } catch (cv::Exception e) {
        return Mat();
    }
}

double FaceRecognizer::getSimilarity(const Mat A, const Mat B)
{
    if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols) {
        double errorL2 = norm(A, B, CV_L2);
        double similarity = errorL2 / (double)(A.rows * A.cols);
        return similarity;
    }
    else {
        return 100000000.0;
    }
}

void FaceRecognizer::learnCollectedFaces(const vector<Mat> preprocessedFaces, const vector<int> faceLabels, const string facerecAlgorithm)
{
    
    train(preprocessedFaces, faceLabels);
        
}

void FaceRecognizer::train(InputArrayOfArrays src, InputArray _lbls) {
    
    if(src.total() == 0) {
        string error_message = format("Empty training data was provided.");
        CV_Error(CV_StsBadArg, error_message);
    } else if(_lbls.getMat().type() != CV_32SC1) {
        string error_message = format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _lbls.type());
        CV_Error(CV_StsBadArg, error_message);
    }
    if(src.total() > 1) {
        for(int i = 1; i < static_cast<int>(src.total()); i++) {
            if(src.getMat(i-1).total() != src.getMat(i).total()) {
                string error_message = format("In the Fisherfaces method, all training images must be of equal size. Expected %d pixels, but was %d pixels.", src.getMat(i-1).total(), src.getMat(i).total());
                CV_Error(CV_StsUnsupportedFormat, error_message);
            }
        }
    }
    Mat labels = _lbls.getMat();
    Mat data = asRowMatrix(src, CV_64FC1);
    int numberOfSamples = data.rows;
    if(labels.total() != (size_t) numberOfSamples) {
        string error_message = format("The number of samples must equal the number of labels. len(src)=%d, len(labels)=%d.", numberOfSamples, labels.total());
        CV_Error(CV_StsBadArg, error_message);
    } else if(labels.rows != 1 && labels.cols != 1) {
        string error_message = format("Expected the labels in a matrix with one row or column. Given dimensions are rows=%s, cols=%d.", labels.rows, labels.cols);
        CV_Error(CV_StsBadArg, error_message);
    }
    _labels.release();
    _projections.clear();
    vector<int> ll;
    for(unsigned int i = 0; i < labels.total(); i++) {
        ll.push_back(labels.at<int>(i));
    }
    int C = (int) remove_dups(ll).size();
    if((_num_components <= 0) || (_num_components > (C-1)))
        _num_components = (C-1);
    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, (numberOfSamples-C));
    LDA lda(pca.project(data),labels, _num_components);
    _mean = pca.mean.reshape(1,1);
    _labels = labels.clone();
    lda.eigenvalues().convertTo(_eigenvalues, CV_64FC1);
    gemm(pca.eigenvectors, lda.eigenvectors(), 1.0, Mat(), 0.0, _eigenvectors, GEMM_1_T);
    for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
        Mat p = cv::LDA::subspaceProject(_eigenvectors, _mean, data.row(sampleIdx));
        _projections.push_back(p);
    }
    cout << "Finished training." << endl;
}

int FaceRecognizer::predict(InputArray _src) {
    Mat src = _src.getMat();
    if(_projections.empty()) {
        string error_message = "This Fisherfaces model isn't computed yet. Did you call Fisherfaces::train?";
        CV_Error(CV_StsBadArg, error_message);
    }
    else if(src.total() != (size_t) _eigenvectors.rows) {
        string error_message = format("Incorrect input image size. Expected an image with %d elements, but got %d.", _eigenvectors.rows, src.total());
        CV_Error(CV_StsBadArg, error_message);
    }
    Mat q = cv::LDA::subspaceProject(_eigenvectors, _mean, src.reshape(1,1));
    double minDist = DBL_MAX;
    int minClass = -1;
    for(size_t sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
        double dist = norm(_projections[sampleIdx], q, NORM_L2);
        if((dist < minDist) && (dist < _threshold)) {
            minDist = dist;
            minClass = _labels.at<int>((int)sampleIdx);
        }
    }
    return minClass;
}

void FaceRecognizer::load(const FileStorage& fs) {
    fs["num_components"] >> _num_components;
    fs["mean"] >> _mean;
    fs["eigenvalues"] >> _eigenvalues;
    fs["eigenvectors"] >> _eigenvectors;
    readFileNodeList(fs["projections"], _projections);
    fs["labels"] >> _labels;
}

void FaceRecognizer::save(FileStorage& fs) {
    fs << "num_components" << _num_components;
    fs << "mean" << _mean;
    fs << "eigenvalues" << _eigenvalues;
    fs << "eigenvectors" << _eigenvectors;
    writeFileNodeList(fs, "projections", _projections);
    fs << "labels" << _labels;
}