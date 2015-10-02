/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#include <cxcore.h>
#include <cv.h>

#include <vector>
#include <iostream>
#include <functional>

#include "HoG.h"

// inspired by https://github.com/rahul411/GLCM-parameters-using-Opencv-Library/blob/master/glcm_parameters.cpp
inline
cv::Mat GLCM(cv::Mat img)
{
    int rows = img.rows;
    int cols = img.cols;
    cv::Mat glcm = cv::Mat::zeros(256,256,CV_32FC1);
  
    if (img.channels() == 3)
        cvtColor(img, img, cv::COLOR_BGR2GRAY);

    for (int j=0; j<rows; j++)
    {
        for (int i=0; i<cols-1; i++)
            glcm.at<float>(img.at<uchar>(j,i),img.at<uchar>(j,i+1)) = 1.f + glcm.at<float>(img.at<uchar>(j,i),img.at<uchar>(j,i+1));
    }

    // normalize glcm matrix
    glcm = glcm + glcm.t();            
    glcm = glcm / sum(glcm)[0];
 
    return glcm;
}

// inspired by https://github.com/rahul411/GLCM-parameters-using-Opencv-Library/blob/master/glcm_parameters.cpp
inline
float const GLCM_contrast(cv::Mat const &img)
{
    cv::Mat glcm = GLCM(img);
  
    float contrast = 0.0f;
    for (int j=0; j<glcm.rows; j++)
    {
        for (int i=0; i<glcm.cols-1; i++)
            contrast += (j-i) * (j-i) * glcm.at<float>(j,i);
    }
 
    return contrast / float(img.cols * img.rows);
}

inline
bool select_positive_training_patch(cv::Mat const &image, cv::Rect const &roi, float scale)
{
    auto selector = [](cv::Mat const &patch) {
        return (GLCM_contrast(patch) > 0.20f);
    };

    if (scale == 1.f)
        return selector(image(roi));

    cv::Mat patch;
    cv::resize(image, patch, cv::Size(), scale, scale);
    return selector(
        patch(
            cv::Rect(
                cvRound(roi.x * scale),
                cvRound(roi.y * scale),
                cvRound(roi.width * scale),
                cvRound(roi.height * scale))));
}

namespace gall {

// structure for image patch
struct PatchFeature {
	explicit PatchFeature() : src_index{-1} {}
	explicit PatchFeature(int frame) : src_index{frame} {}

    bool const empty() const { return src_index == -1; }

    int            const src_index;  // index of the src image in the full training
	CvRect               roi;
	std::vector<CvPoint> center;
	std::vector<CvMat *> vPatch;

    // not copyable
    PatchFeature(PatchFeature const &)            = delete;

    // we need a move ctor to be storable in a vector<>
    PatchFeature(PatchFeature &&other)            = default;

    // not assignable because of the const index
    PatchFeature &operator=(PatchFeature &&)      = delete;
    PatchFeature &operator=(PatchFeature const &) = delete;
};

static HoG hog; 

class CRPatch {
public:
	CRPatch(CvRNG rng, int w, int h, int num_l) : rng(rng), width(w), height(h) { vLPatches.resize(num_l);}

	// Extract patches from image
	void extractPatches(IplImage *img, unsigned int n, int label, CvRect const * const box = 0, std::vector<CvPoint>* vCenter = 0);

	// Extract patches from feature channels
	void extractPatches(std::vector<IplImage*> const &vImg, unsigned int n, int label, CvRect const * const box = 0, std::vector<CvPoint>* vCenter = 0);

	// Extract patches from feature channels, ignoring areas of low texture
	void extract_patches_of_texture(cv::Mat                 const &image,
                                    std::vector<IplImage *> const &vImg,
                                    unsigned int                   n,
                                    std::function<bool (cv::Rect const &)> patch_selector,
                                    CvRect const          * const  box     = nullptr,
                                    std::vector<CvPoint>          *vCenter = nullptr);

	// Extract features from image
	static void extractFeatureChannels(IplImage *img, std::vector<IplImage*>& vImg);

	// min/max filter
	static void maxfilt(uchar* data, uchar* maxvalues, unsigned int step, unsigned int size, unsigned int width);
	static void maxfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width);
	static void minfilt(uchar* data, uchar* minvalues, unsigned int step, unsigned int size, unsigned int width);
	static void minfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width);
	static void maxminfilt(uchar* data, uchar* maxvalues, uchar* minvalues, unsigned int step, unsigned int size, unsigned int width);
	static void maxfilt(IplImage *src, unsigned int width);
	static void maxfilt(IplImage *src, IplImage *dst, unsigned int width);
	static void minfilt(IplImage *src, unsigned int width);
	static void minfilt(IplImage *src, IplImage *dst, unsigned int width);

	std::vector<std::vector<PatchFeature> > vLPatches;

private:
    CRPatch &operator=(CRPatch const &) = delete;

private:
	int const width;
	int const height;
	CvRNG rng;
};

}   // namespace gall
