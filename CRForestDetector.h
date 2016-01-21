/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#include <functional>
#include "CRForest.h"

namespace gall {

class CRForestDetector {
public:
	// Constructor
	CRForestDetector(CRForest const &RF, int w, int h) : crForest_(RF), width(w), height(h)
    {
    }

	size_t const GetNumCenter() const
    {
        return crForest_.GetNumCenter();
    }

    void accumulate_votes(cv::Rect                         const &roi,
                          std::vector<IplImage*>           const &features,
                          std::vector<float>               const &ratios,
                          std::function<bool (cv::Rect const &)> patch_selector,
                          std::vector<IplImage*>                 &imgDetect) const;

private:
    CRForestDetector &operator=(CRForestDetector &&)      = delete;
    CRForestDetector &operator=(CRForestDetector const &) = delete;

	CRForest const &crForest_;
	int      const  width;
	int      const  height;
};

}   // namespace gall
