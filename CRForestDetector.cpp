/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "CRForestDetector.h"
#include <vector>
#ifdef CR_PROGRESS
#include "timer.h"
#endif

namespace gall {

using namespace std;


std::vector<std::vector<cv::Mat>>
CRForestDetector::accumulate_votes(
    cv::Rect               const &roi,
    std::vector<IplImage*> const &features,
    std::vector<float>     const &ratios,
    std::vector<IplImage*>       &imgDetect,
    bool                   const inverted_forest_training) const
{
#ifdef CR_PROGRESS
    cdmh::timer t("CRForestDetector::accumulate_votes");
#endif

    assert(features.size() == 32);
    assert(features[0]->width >= roi.width);
    assert(features[0]->height >= roi.height);
    assert(features[0]->width == features[0]->width);
    assert(features[0]->height == features[0]->height);

    // reset output image
    for(int c=0; c<(int)imgDetect.size(); ++c)
        cvSetZero( imgDetect[c] );

    // get pointers to feature channels
    int stepImg = 0;
    uchar** ptFCh     = new uchar*[features.size()];
    uchar** ptFCh_row = new uchar*[features.size()];
	for(unsigned int c=0; c<features.size(); ++c) {
        cvGetRawData( features[c], (uchar**)&(ptFCh[c]), &stepImg);
	}
    stepImg /= sizeof(ptFCh[0][0]);

    // get pointer to output image
    int stepDet = 0;
    float** ptDet = new float*[imgDetect.size()];
    for(unsigned int c=0; c<imgDetect.size(); ++c)
        cvGetRawData( imgDetect[c], (uchar**)&(ptDet[c]), &stepDet);
    stepDet /= sizeof(ptDet[0][0]);

    int const xoffset = width/2;
    int const yoffset = height/2;

    // data structure to record the leaf contributors to each pixel
    std::vector<std::vector<cv::Mat>> contributor_map(imgDetect.size());

    // cx,cy center of patch
	vector<const LeafNode*> result(crForest_.GetSize());
    for (int cy=yoffset+roi.y; cy<=roi.y+roi.height+yoffset-height; ++cy)
    {
        // Get start of row
        for (unsigned int c=0; c<features.size(); ++c)
            ptFCh_row[c] = &ptFCh[c][0];

        for(int cx=xoffset+roi.x; cx<=roi.x+roi.width+xoffset-width; ++cx)
        {
            // regression for a single patch
            crForest_.regression(result, ptFCh_row, stepImg);
            
            for (size_t c=0; c<imgDetect.size(); ++c)
            {
                auto &contrib = contributor_map[c];

                // vote for all trees (leafs) 
                for (auto &leaf : result)
                {
                    // To speed up the voting, one can vote only for patches 
                    // with a probability for foreground > 0.5
                    // !!! CH This condition was commented out in the original code,
                    //        with no indication why. It reduces processing from
                    //        4m to 20s on a debug build, so worth having, and
                    //        produces a good result. I haven't compared the
                    //        accuracy fully yet, though
                    if (leaf->pfg > 0.5)
                    {
                        // voting weight for leaf 
                        float const w = leaf->pfg / float(leaf->vCenter.size() * result.size());

                        // vote for all points stored in the leaf
                        for (size_t ndx=0; ndx<leaf->vCenter.size(); ++ndx)
                        {
                            /// if we have trained the forest using the search dataset and the
                            /// application is to find a smaller query image in that dataset,
                            /// then we use a different voting mechanism
                            int x, y;
                            if (inverted_forest_training)
                            {
                                x = int(cx - (leaf->roi[ndx].width/2.) * ratios[c] + 0.5);
                                y = int(cy - (leaf->roi[ndx].height/2.) + 0.5);
                            }
                            else
                            {
                                x = int(cx - leaf->vCenter[ndx][0].x * ratios[c] + 0.5);
                                y = cy - leaf->vCenter[ndx][0].y;
                            }

                            if (x>=0  &&  y>=0  &&  x<imgDetect[c]->width  &&  y<imgDetect[c]->height)
                            {
                                *(ptDet[c] + x + y*stepDet) += w;

                                size_t const frame = leaf->src_indices[ndx];
                                for (size_t i=contrib.size(); i<=frame; ++i)
                                    contrib.push_back(cv::Mat::zeros(features[0]->height,features[0]->width,CV_32FC1));
//!!!!!!!!!!!!!!!!!!!!!!!!!
// we don't know the original image size, so we'll resize as we go. this is very inefficient, so needs fixing
                                {
                                    auto const mx = 1+leaf->roi[ndx].x+leaf->roi[ndx].width;
                                    auto const my = 1+leaf->roi[ndx].y+leaf->roi[ndx].height;
                                    if (contrib[frame].cols < mx  ||  contrib[frame].rows < my)
                                    {
                                        cv::Mat newimage = cv::Mat::zeros(std::max(my,contrib[frame].rows), std::max(mx,contrib[frame].cols), CV_32FC1);
                                        cv::Rect roi(cv::Point(0,0),contrib[frame].size());
                                        contrib[frame].copyTo(newimage(roi));
                                        swap(contrib[frame], newimage);
                                    }
                                }
//!!!!!!!!!!!!!!!!!!!!!!!!!
                                for (int cy1=leaf->roi[ndx].y; cy1<=leaf->roi[ndx].y+leaf->roi[ndx].height; ++cy1)
                                    for (int cx1=leaf->roi[ndx].x; cx1<=leaf->roi[ndx].x+leaf->roi[ndx].width; ++cx1)
                                        contrib[frame].at<float>(cy1,cx1) += w;
                            }
                        }
                    }
                } // end if
            }

            // increase pointer - x
            for(unsigned int c=0; c<features.size(); ++c)
                ++ptFCh_row[c];

        } // end for x

        // increase pointer - y
        for(unsigned int c=0; c<features.size(); ++c)
            ptFCh[c] += stepImg;

    } // end for y 	

    // smooth result image
	for(auto &i: imgDetect)
		cvSmooth(i, i, CV_GAUSSIAN, 3);

    delete[] ptFCh;
    delete[] ptFCh_row;
    delete[] ptDet;

    return contributor_map;
}

}   // namespace gall
