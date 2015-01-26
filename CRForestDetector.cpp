/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "CRForestDetector.h"
#include <vector>
#include "timer.h"


using namespace std;

void CRForestDetector::detectColor(
    IplImage                 *img,
    vector<IplImage*>        &imgDetect,
    std::vector<float> const &ratios)
{
#ifdef CR_PROGRESS
    cdmh::timer t("CRForestDetector::detectColor");
#endif

	// extract features
	vector<IplImage*> features;
	CRPatch::extractFeatureChannels(img, features);
    accumulate_votes({img->width, img->height}, imgDetect, features, ratios);
}


void CRForestDetector::accumulate_votes(
    CvSize                 const &size,
    vector<IplImage*>            &imgDetect,
    std::vector<IplImage*> const &features,
    std::vector<float>     const &ratios)
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

	int xoffset = width/2;
	int yoffset = height/2;
	
	int x, y, cx, cy; // x,y top left; cx,cy center of patch
	cy = yoffset; 

	for(y=0; y<size.height-height; ++y, ++cy) {
		// Get start of row
		for(unsigned int c=0; c<features.size(); ++c)
			ptFCh_row[c] = &ptFCh[c][0];
		cx = xoffset; 
		
		for(x=0; x<size.width-width; ++x, ++cx) {					

			// regression for a single patch
			vector<const LeafNode*> result;
			crForest_.regression(result, ptFCh_row, stepImg);
			
			// vote for all trees (leafs) 
			for(vector<const LeafNode*>::const_iterator itL = result.begin(); itL!=result.end(); ++itL) {

				// To speed up the voting, one can vote only for patches 
			    // with a probability for foreground > 0.5
                // !!! CH This was commented out in the original code, with no
                //        indication why. It reduces processing from 4m to 20s
                //        on a debug build, so worth having, and produces a
                //        good result. I haven't compared the accuracy fully
                //        yet, though
				if((*itL)->pfg>0.5) {

					// voting weight for leaf 
					float w = (*itL)->pfg / float( (*itL)->vCenter.size() * result.size() );

					// vote for all points stored in the leaf
					for(vector<vector<CvPoint> >::const_iterator it = (*itL)->vCenter.begin(); it!=(*itL)->vCenter.end(); ++it) {

						for(int c=0; c<(int)imgDetect.size(); ++c) {
						  int const x = int(cx - (*it)[0].x * ratios[c] + 0.5);
						  int const y = cy-(*it)[0].y;
						  if(y>=0 && y<imgDetect[c]->height && x>=0 && x<imgDetect[c]->width) {
						    *(ptDet[c]+x+y*stepDet) += w;
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
	for(int c=0; c<(int)imgDetect.size(); ++c)
		cvSmooth( imgDetect[c], imgDetect[c], CV_GAUSSIAN, 3);
	
	delete[] ptFCh;
	delete[] ptFCh_row;
	delete[] ptDet;

}

void CRForestDetector::detectPyramid(IplImage *img, vector<vector<IplImage*> >& vImgDetect, std::vector<float> const &ratios) {

	if(img->nChannels==1) {

		std::cerr << "Gray color images are not supported." << std::endl;

	} else { // color

		for(int i=0; i<int(vImgDetect.size()); ++i) {
			IplImage* cLevel = cvCreateImage( cvSize(vImgDetect[i][0]->width,vImgDetect[i][0]->height) , IPL_DEPTH_8U , 3);				
			cvResize( img, cLevel, CV_INTER_LINEAR );	

			// detection
			detectColor(cLevel,vImgDetect[i],ratios);

			cvReleaseImage(&cLevel);
		}

	}

}








