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
	void detectPyramid(IplImage *img, std::vector<std::vector<IplImage*> >& imgDetect, std::vector<float>& ratios);

	// Get/Set functions
	size_t GetNumCenter() const {return crForest_.GetNumCenter();}

private:
	void detectColor(IplImage *img, std::vector<IplImage*>& imgDetect, std::vector<float>& ratios);

	CRForest const &crForest_;
	int width;
	int height;
};
