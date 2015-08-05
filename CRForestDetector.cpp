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


void CRForestDetector::accumulate_votes(
    CvSize                 const &size,
    std::vector<IplImage*> const &features,
    std::vector<float>     const &ratios,
    std::vector<IplImage*>       &imgDetect) const
{
#ifdef CR_PROGRESS
    cdmh::timer t("CRForestDetector::accumulate_votes");
#endif

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
	
    // cx,cy center of patch
	vector<const LeafNode*> result(crForest_.GetSize());
	for (int cy=yoffset; cy<size.height+yoffset-height; ++cy)
    {
		// Get start of row
		for (unsigned int c=0; c<features.size(); ++c)
			ptFCh_row[c] = &ptFCh[c][0];

		for(int cx=xoffset; cx<size.width+xoffset-width; ++cx)
        {
			// regression for a single patch
			crForest_.regression(result, ptFCh_row, stepImg);
			
			for (size_t c=0; c<imgDetect.size(); ++c)
            {
			    // vote for all trees (leafs) 
			    for (auto const &leaf : result)
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
					    for (auto const &centre : leaf->vCenter)
                        {
						    int const x = int(cx - centre[0].x * ratios[c] + 0.5);
						    int const y = cy - centre[0].y;
						    if (y>=0  &&  x>=0  &&  y<imgDetect[c]->height  &&  x<imgDetect[c]->width)
						        *(ptDet[c] + x + y*stepDet) += w;
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
	for(int c=0; c<(int)imgDetect.size(); ++c)
		cvSmooth( imgDetect[c], imgDetect[c], CV_GAUSSIAN, 3);
	
	delete[] ptFCh;
	delete[] ptFCh_row;
	delete[] ptDet;

}

}   // namespace gall
