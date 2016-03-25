#include "facerecognizer.h"
#include <set>

const int THRESHOLD = 3000;
const double UNKNOWN_PERSON_THRESHOLD = 0.8f;

template<typename _Tp>
inline void readFileNodeList(const cv::FileNode& fn, std::vector<_Tp>& result) {
    if (fn.type() == cv::FileNode::SEQ) {
        for (cv::FileNodeIterator it = fn.begin(); it != fn.end();) {
            _Tp item;
            it >> item;
            result.push_back(item);
        }
    }
}

template<typename _Tp>
inline void writeFileNodeList(cv::FileStorage& fs, const std::string& name,
                              const std::vector<_Tp>& items) {

    typedef typename std::vector<_Tp>::const_iterator constVecIterator;
    fs << name << "[";
    for (constVecIterator it = items.begin(); it != items.end(); ++it) {
        fs << *it;
    }
    fs << "]";
}

template<typename _Tp>
inline std::vector<_Tp> remove_dups(const std::vector<_Tp>& src) {
    typedef typename std::set<_Tp>::const_iterator constSetIterator;
    typedef typename std::vector<_Tp>::const_iterator constVecIterator;
    std::set<_Tp> set_elems;
    for (constVecIterator it = src.begin(); it != src.end(); ++it)
        set_elems.insert(*it);
    std::vector<_Tp> elems;
    for (constSetIterator it = set_elems.begin(); it != set_elems.end(); ++it)
        elems.push_back(*it);
    return elems;
}

static cv::Mat asRowMatrix(cv::InputArrayOfArrays src, int rtype, double alpha=1, double beta=0) {
    if(src.kind() != cv::_InputArray::STD_VECTOR_MAT && src.kind() != cv::_InputArray::STD_VECTOR_VECTOR) {
        std::string error_message = "Data is expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< vector<...> >).";
        CV_Error(CV_StsBadArg, error_message);
    }
    size_t nsamples = src.total();
    if(nsamples == 0)
        return cv::Mat();
    size_t dimensionOfSamples = src.getMat(0).total();
    cv::Mat data((int)nsamples, (int)dimensionOfSamples, rtype);
    for(unsigned int i = 0; i < nsamples; i++) {
        if(src.getMat(i).total() != dimensionOfSamples) {
            std::string error_message = cv::format("Incorrent number of elements in matrix #%d! Expected %d was %d.", i, dimensionOfSamples, src.getMat(i).total());
            CV_Error(CV_StsBadArg, error_message);
        }
        cv::Mat xi = data.row(i);
        if(src.getMat(i).isContinuous()) {
            src.getMat(i).reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src.getMat(i).clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

cv::Mat FaceRecognizer::getMean() {
    return _mean;
}
cv::Mat FaceRecognizer::getEigenvectors() {
    return _eigenvectors;
}

cv::Mat FaceRecognizer::getEigenvalues() {
    return _eigenvalues;
}

std::vector<cv::Mat> FaceRecognizer::getProjections() {
    return _projections;
}

cv::Mat FaceRecognizer::reconstructFace(FaceRecognizer faceRecognizer, const cv::Mat preprocessedFace)
{
    try {
        cv::Mat eigenvectors = faceRecognizer.getEigenvectors();
        cv::Mat averageFaceRow = faceRecognizer.getMean();
        int faceHeight = preprocessedFace.rows;
        cv::Mat projection = cv::LDA::subspaceProject(eigenvectors, averageFaceRow, preprocessedFace.reshape(1,1));
        cv::Mat reconstructionRow = cv::LDA::subspaceReconstruct (eigenvectors, averageFaceRow, projection);
        cv::Mat reconstructionMat = reconstructionRow.reshape(1, faceHeight);
        cv::Mat reconstructedFace = cv::Mat(reconstructionMat.size(), CV_8U);
        reconstructionMat.convertTo(reconstructedFace, CV_8U, 1, 0);
        return reconstructedFace;
    } catch (cv::Exception e) {
        return cv::Mat();
    }
}

double FaceRecognizer::getSimilarity(const cv::Mat A, const cv::Mat B)
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

void FaceRecognizer::learnCollectedFaces(const std::vector<cv::Mat> preprocessedFaces, const std::vector<int> faceLabels, const std::string facerecAlgorithm)
{
    
    train(preprocessedFaces, faceLabels);
        
}

void FaceRecognizer::train(cv::InputArrayOfArrays src, cv::InputArray _lbls) {
    
    if(src.total() == 0) {
        std::string error_message = cv::format("Empty training data was provided.");
        CV_Error(CV_StsBadArg, error_message);
    } else if(_lbls.getMat().type() != CV_32SC1) {
        std::string error_message = cv::format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _lbls.type());
        CV_Error(CV_StsBadArg, error_message);
    }
    if(src.total() > 1) {
        for(int i = 1; i < static_cast<int>(src.total()); i++) {
            if(src.getMat(i-1).total() != src.getMat(i).total()) {
                std::string error_message = cv::format("In the Fisherfaces method, all training images must be of equal size. Expected %d pixels, but was %d pixels.", src.getMat(i-1).total(), src.getMat(i).total());
                CV_Error(CV_StsUnsupportedFormat, error_message);
            }
        }
    }
    cv::Mat labels = _lbls.getMat();
    cv::Mat data = asRowMatrix(src, CV_64FC1);
    int numberOfSamples = data.rows;
    if(labels.total() != (size_t) numberOfSamples) {
        std::string error_message = cv::format("The number of samples must equal the number of labels. len(src)=%d, len(labels)=%d.", numberOfSamples, labels.total());
        CV_Error(CV_StsBadArg, error_message);
    } else if(labels.rows != 1 && labels.cols != 1) {
        std::string error_message = cv::format("Expected the labels in a matrix with one row or column. Given dimensions are rows=%s, cols=%d.", labels.rows, labels.cols);
        CV_Error(CV_StsBadArg, error_message);
    }
    _labels.release();
    _projections.clear();
    std::vector<int> ll;
    for(unsigned int i = 0; i < labels.total(); i++) {
        ll.push_back(labels.at<int>(i));
    }
    int C = (int) remove_dups(ll).size();
    if((_num_components <= 0) || (_num_components > (C-1)))
        _num_components = (C-1);
    cv::PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW, (numberOfSamples-C));
    cv::LDA lda(pca.project(data),labels, _num_components);
    _mean = pca.mean.reshape(1,1);
    _labels = labels.clone();
    lda.eigenvalues().convertTo(_eigenvalues, CV_64FC1);
    cv::gemm(pca.eigenvectors, lda.eigenvectors(), 1.0, cv::Mat(), 0.0, _eigenvectors, cv::GEMM_1_T);
    for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
        cv::Mat p = cv::LDA::subspaceProject(_eigenvectors, _mean, data.row(sampleIdx));
        _projections.push_back(p);
    }
    //cout << "Finished training." << endl;
}


// Generate an approximately reconstructed face by back-projecting the eigenvectors & eigenvalues of the given (preprocessed) face.
cv::Mat FaceRecognizer::reconstructFace(const cv::Mat preprocessedFace)
{
    // Since we can only reconstruct the face for some types of FaceRecognizer models (ie: Eigenfaces or Fisherfaces),
    // we should surround the OpenCV calls by a try/catch block so we don't crash for other models.
    try {
        
        int faceHeight = preprocessedFace.rows;
        
        // Project the input image onto the PCA subspace.
        cv::Mat projection = cv::LDA::subspaceProject(_eigenvectors, _mean, preprocessedFace.reshape(1,1));
        //printMatInfo(projection, "projection");
        
        // Generate the reconstructed face back from the PCA subspace.
        cv::Mat reconstructionRow = cv::LDA::subspaceReconstruct (_eigenvectors, _mean, projection);
        //printMatInfo(reconstructionRow, "reconstructionRow");
        
        // Convert the float row matrix to a regular 8-bit image. Note that we
        // shouldn't use "getImageFrom1DFloatMat()" because we don't want to normalize
        // the data since it is already at the perfect scale.
        
        // Make it a rectangular shaped image instead of a single row.
        cv::Mat reconstructionMat = reconstructionRow.reshape(1, faceHeight);
        // Convert the floating-point pixels to regular 8-bit uchar pixels.
        cv::Mat reconstructedFace = cv::Mat(reconstructionMat.size(), CV_8U);
        reconstructionMat.convertTo(reconstructedFace, CV_8U, 1, 0);
        //printMatInfo(reconstructedFace, "reconstructedFace");
        
        return reconstructedFace;
        
    } catch (cv::Exception e) {
        //cout << "WARNING: Missing FaceRecognizer properties." << endl;
        return cv::Mat();
    }
}


int FaceRecognizer::predict(cv::InputArray _src) {
    cv::Mat preprocessedFace = _src.getMat();
    int minClass = -1;
    if(_projections.empty()) {
        std::string error_message = "This Fisherfaces model isn't computed yet. Did you call Fisherfaces::train?";
        CV_Error(CV_StsBadArg, error_message);
    }
    else if(preprocessedFace.total() == 0){ //return. not a valid preprocessed facial image
        return minClass;
    }
    else if(preprocessedFace.total() != (size_t) _eigenvectors.rows) {
        std::string error_message = cv::format("Incorrect input image size. Expected an image with %d elements, but got %d.", _eigenvectors.rows, preprocessedFace.total());
        CV_Error(CV_StsBadArg, error_message);
    }
    

    cv::Mat q = cv::LDA::subspaceProject(_eigenvectors, _mean, preprocessedFace.reshape(1,1));
        double minDist = DBL_MAX;
        for(size_t sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
            double dist = norm(_projections[sampleIdx], q, cv::NORM_L2);
            if((dist < minDist) && (dist < THRESHOLD)) {
                minDist = dist;
                minClass = _labels.at<int>((int)sampleIdx);
            }
            else {
                cv::Mat reconstructedFace = reconstructFace(preprocessedFace);
                double similarity = getSimilarity(preprocessedFace, reconstructedFace);
                if (similarity > UNKNOWN_PERSON_THRESHOLD) {
                    minClass = _labels.at<int>((int)sampleIdx);
                }
            }
        }

    

    return minClass;
}

void FaceRecognizer::load(const cv::FileStorage& fs) {
    fs["num_components"] >> _num_components;
    fs["mean"] >> _mean;
    fs["eigenvalues"] >> _eigenvalues;
    fs["eigenvectors"] >> _eigenvectors;
    readFileNodeList(fs["projections"], _projections);
    fs["labels"] >> _labels;
}

void FaceRecognizer::save(cv::FileStorage& fs) {
    fs << "num_components" << _num_components;
    fs << "mean" << _mean;
    fs << "eigenvalues" << _eigenvalues;
    fs << "eigenvectors" << _eigenvectors;
    writeFileNodeList(fs, "projections", _projections);
    fs << "labels" << _labels;
}