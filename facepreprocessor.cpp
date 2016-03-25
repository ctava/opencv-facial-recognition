#include "detectobjects.h"
#include "facepreprocessor.h"
#import <CoreImage/CoreImage.h>

const double DESIRED_LEFT_EYE_X = 0.16;
const double DESIRED_LEFT_EYE_Y = 0.14;
const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.50;
const double FACE_ELLIPSE_H = 0.80;
const int FACE_WIDTH = 70;
const bool PREPROCESS_SIDES_OF_FACE_SEPERATELY = true;

const char *FACE_CASCADE_FILENAME = "./face.xml";
const char *EYE_CASCADE1_FILENAME = "./lefteye.xml";
const char *EYE_CASCADE2_FILENAME = "./righteye.xml";

void FacePreprocessor::initCascadeClassifiers()
{
    try {
        faceCascade.load(FACE_CASCADE_FILENAME);
    } catch (cv::Exception &e) {}
    
    try {
        eyeCascade1.load(EYE_CASCADE1_FILENAME);
    } catch (cv::Exception &e) {}
    
    try {
        eyeCascade2.load(EYE_CASCADE2_FILENAME);
    } catch (cv::Exception &e) {}
}

void FacePreprocessor::detectBothEyes(const cv::Mat &face, cv::Point &leftEye, cv::Point &rightEye)
{
    const float EYE_SX = 0.10f;
    const float EYE_SY = 0.19f;
    const float EYE_SW = 0.40f;
    const float EYE_SH = 0.36f;

    int leftX = cvRound(face.cols * EYE_SX);
    int topY = cvRound(face.rows * EYE_SY);
    int widthX = cvRound(face.cols * EYE_SW);
    int heightY = cvRound(face.rows * EYE_SH);
    int rightX = cvRound(face.cols * (1.0-EYE_SX-EYE_SW) );

    cv::Mat topLeftOfFace = face(cv::Rect(leftX, topY, widthX, heightY));
    cv::Mat topRightOfFace = face(cv::Rect(rightX, topY, widthX, heightY));
    cv::Rect leftEyeRect, rightEyeRect;
    findLargestObject(topLeftOfFace, eyeCascade1, leftEyeRect, topLeftOfFace.cols);
    findLargestObject(topRightOfFace, eyeCascade1, rightEyeRect, topRightOfFace.cols);
    if (leftEyeRect.width <= 0 && !eyeCascade2.empty()) {
        findLargestObject(topLeftOfFace, eyeCascade2, leftEyeRect, topLeftOfFace.cols);
    }

    if (rightEyeRect.width <= 0 && !eyeCascade2.empty()) {
        findLargestObject(topRightOfFace, eyeCascade2, rightEyeRect, topRightOfFace.cols);
    }

    if (leftEyeRect.width > 0) {
        leftEyeRect.x += leftX;
        leftEyeRect.y += topY;
        leftEye = cv::Point(leftEyeRect.x + leftEyeRect.width/2, leftEyeRect.y + leftEyeRect.height/2);
    }
    else {
        leftEye = cv::Point(-1, -1);
    }

    if (rightEyeRect.width > 0) { 
        rightEyeRect.x += rightX;
        rightEyeRect.y += topY;
        rightEye = cv::Point(rightEyeRect.x + rightEyeRect.width/2, rightEyeRect.y + rightEyeRect.height/2);
    }
    else {
        rightEye = cv::Point(-1, -1);
    }
}

void FacePreprocessor::equalizeLeftAndRightHalves(cv::Mat &faceImg)
{
    int w = faceImg.cols;
    int h = faceImg.rows;
    cv::Mat wholeFace;
    equalizeHist(faceImg, wholeFace);
    int midX = w/2;
    cv::Mat leftSide = faceImg(cv::Rect(0,0, midX,h));
    cv::Mat rightSide = faceImg(cv::Rect(midX,0, w-midX,h));
    equalizeHist(leftSide, leftSide);
    equalizeHist(rightSide, rightSide);
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            int v;
            if (x < w/4) {
                v = leftSide.at<uchar>(y,x);
            }
            else if (x < w*2/4) {
                int lv = leftSide.at<uchar>(y,x);
                int wv = wholeFace.at<uchar>(y,x);
                float f = (x - w*1/4) / (float)(w*0.25f);
                v = cvRound((1.0f - f) * lv + (f) * wv);
            }
            else if (x < w*3/4) {
                int rv = rightSide.at<uchar>(y,x-midX);
                int wv = wholeFace.at<uchar>(y,x);
                float f = (x - w*2/4) / (float)(w*0.25f);
                v = cvRound((1.0f - f) * wv + (f) * rv);
            }
            else {
                v = rightSide.at<uchar>(y,x-midX);
            }
            faceImg.at<uchar>(y,x) = v;
        }
    }
}

cv::Mat FacePreprocessor::getPreprocessedFace(cv::Mat srcImg)
{
    int desiredFaceHeight = FACE_WIDTH;

//    NSDictionary *detectorOptions = @{CIDetectorTracking: @YES};
//    CIDetector *detector = [CIDetector detectorOfType:CIDetectorTypeFace context:nil options:detectorOptions];

    
    cv::Rect faceRect;
    findLargestObject(srcImg, faceCascade, faceRect);
    
    

    if (faceRect.width > 0) {

        cv::Mat faceImg = srcImg(faceRect);
        cv::Mat gray;
        if (faceImg.channels() == 3) {
            cvtColor(faceImg, gray, CV_BGR2GRAY);
        }
        else if (faceImg.channels() == 4) {
            cvtColor(faceImg, gray, CV_BGRA2GRAY);
        }
        else {
            gray = faceImg;
        }

        cv::Point leftEye, rightEye;
        detectBothEyes(gray, leftEye, rightEye);

        //
        if (leftEye.x >= 0 && rightEye.x >= 0) {

            cv::Point2f eyesCenter = cv::Point2f( (leftEye.x + rightEye.x) * 0.5f, (leftEye.y + rightEye.y) * 0.5f );
            double dy = (rightEye.y - leftEye.y);
            double dx = (rightEye.x - leftEye.x);
            double len = sqrt(dx*dx + dy*dy);
            double angle = atan2(dy, dx) * 180.0/CV_PI;
            const double DESIRED_RIGHT_EYE_X = (1.0f - DESIRED_LEFT_EYE_X);
            double desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * FACE_WIDTH;
            double scale = desiredLen / len;
            cv::Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
            rot_mat.at<double>(0, 2) += FACE_WIDTH * 0.5f - eyesCenter.x;
            rot_mat.at<double>(1, 2) += desiredFaceHeight * DESIRED_LEFT_EYE_Y - eyesCenter.y;
            cv::Mat warped = cv::Mat(desiredFaceHeight, FACE_WIDTH, CV_8U, cv::Scalar(128));
            warpAffine(gray, warped, rot_mat, warped.size());
            if (!PREPROCESS_SIDES_OF_FACE_SEPERATELY) {
                equalizeHist(warped, warped);
            }
            else {
                equalizeLeftAndRightHalves(warped);
            }

            cv::Mat filtered = cv::Mat(warped.size(), CV_8U);
            bilateralFilter(warped, filtered, 0, 20.0, 2.0);

            cv::Mat mask = cv::Mat(warped.size(), CV_8U, cv::Scalar(0));
            cv::Point faceCenter = cv::Point( FACE_WIDTH/2, cvRound(desiredFaceHeight * FACE_ELLIPSE_CY) );
            cv::Size size = cv::Size( cvRound(FACE_WIDTH * FACE_ELLIPSE_W), cvRound(desiredFaceHeight * FACE_ELLIPSE_H) );
            ellipse(mask, faceCenter, size, 0, 0, 360, cv::Scalar(255), CV_FILLED);

            cv::Mat dstImg = cv::Mat(warped.size(), CV_8U, cv::Scalar(128));
            filtered.copyTo(dstImg, mask);

            return dstImg;
        }
    }
    return cv::Mat();
}