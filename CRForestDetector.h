/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#include "CRForest.h"


class CRForestDetector {
public:
	// Constructor
	CRForestDetector(CRForest const &RF, int w, int h) : crForest_(RF), width(w), height(h)  {}

	// detect multi scale
	void detectPyramid(IplImage *img, std::vector<std::vector<IplImage*> >& imgDetect, std::vector<float> const &ratios);

	// Get/Set functions
	size_t GetNumCenter() const {return crForest_.GetNumCenter();}

    void accumulate_votes(CvSize                 const &size,
                          std::vector<IplImage*>       &imgDetect,
                          std::vector<IplImage*> const &features,
                          std::vector<float>     const &ratios);

private:
    CRForestDetector &operator=(CRForestDetector const &) = delete;
	void detectColor(IplImage *img, std::vector<IplImage*>& imgDetect, std::vector<float> const &ratios);

	CRForest const &crForest_;
	int width;
	int height;
};
