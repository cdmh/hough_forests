/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
 
// You may use, copy, reproduce, and distribute this Software for any 
// non-commercial purpose, subject to the restrictions of the 
// Microsoft Research Shared Source license agreement ("MSR-SSLA"). 
// Some purposes which can be non-commercial are teaching, academic 
// research, public demonstrations and personal experimentation. You 
// may also distribute this Software with books or other teaching 
// materials, or publish the Software on websites, that are intended 
// to teach the use of the Software for academic or other 
// non-commercial purposes.
// You may not use or distribute this Software or any derivative works 
// in any form for commercial purposes. Examples of commercial 
// purposes would be running business operations, licensing, leasing, 
// or selling the Software, distributing the Software for use with 
// commercial products, using the Software in the creation or use of 
// commercial products or any other activity which purpose is to 
// procure a commercial gain to you or others.
// If the Software includes source code or data, you may create 
// derivative works of such portions of the Software and distribute 
// the modified Software for non-commercial purposes, as provided 
// herein.

// THE SOFTWARE COMES "AS IS", WITH NO WARRANTIES. THIS MEANS NO 
// EXPRESS, IMPLIED OR STATUTORY WARRANTY, INCLUDING WITHOUT 
// LIMITATION, WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A 
// PARTICULAR PURPOSE, ANY WARRANTY AGAINST INTERFERENCE WITH YOUR 
// ENJOYMENT OF THE SOFTWARE OR ANY WARRANTY OF TITLE OR 
// NON-INFRINGEMENT. THERE IS NO WARRANTY THAT THIS SOFTWARE WILL 
// FULFILL ANY OF YOUR PARTICULAR PURPOSES OR NEEDS. ALSO, YOU MUST 
// PASS THIS DISCLAIMER ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR 
// DERIVATIVE WORKS.

// NEITHER MICROSOFT NOR ANY CONTRIBUTOR TO THE SOFTWARE WILL BE 
// LIABLE FOR ANY DAMAGES RELATED TO THE SOFTWARE OR THIS MSR-SSLA, 
// INCLUDING DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL OR INCIDENTAL 
// DAMAGES, TO THE MAXIMUM EXTENT THE LAW PERMITS, NO MATTER WHAT 
// LEGAL THEORY IT IS BASED ON. ALSO, YOU MUST PASS THIS LIMITATION OF 
// LIABILITY ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE 
// WORKS.

// When using this software, please acknowledge the effort that 
// went into development by referencing the paper:
//
// Gall J. and Lempitsky V., Class-Specific Hough Forests for 
// Object Detection, IEEE Conference on Computer Vision and Pattern 
// Recognition (CVPR'09), 2009.

// Note that this is not the original software that was used for 
// the paper mentioned above. It is a re-implementation for Linux. 

*/


#define PATH_SEP "/"

#include <stdexcept>

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include <highgui.h>

#include "CRForestDetector.h"

using namespace std;

// Path to trees
string treepath;
// Number of trees
int ntrees;
// Patch width
int p_width;
// Patch height
int p_height;
// Path to images
string impath;
// File with names of images
string imfiles;
// Extract features
bool xtrFeature;
// Scales
vector<float> scales;
// Ratio
vector<float> ratios;
// Output path
string outpath;
// scale factor for output image (default: 128)
int out_scale;
// Path to positive examples
string trainpospath;
// File with postive examples
string trainposfiles;
// Subset of positive images -1: all images
int subsamples_pos;
// Sample patches from pos. examples
unsigned int samples_pos; 
// Path to positive examples
string trainnegpath;
// File with postive examples
string trainnegfiles;
// Subset of neg images -1: all images
int subsamples_neg;
// Samples from pos. examples
unsigned int samples_neg;

// offset for saving tree number
int off_tree;


// load config file for dataset
void loadConfig(const char* filename, int mode) {
	char buffer[400];
	ifstream in(filename);

	if(in.is_open()) {

		// Path to trees
		in.getline(buffer,400);
		in.getline(buffer,400); 
		treepath = buffer;
		// Number of trees
		in.getline(buffer,400); 
		in >> ntrees;
		in.getline(buffer,400); 
		// Patch width
		in.getline(buffer,400); 
		in >> p_width;
		in.getline(buffer,400); 
		// Patch height
		in.getline(buffer,400); 
		in >> p_height;
		in.getline(buffer,400); 
		// Path to images
		in.getline(buffer,400); 
		in.getline(buffer,400); 
		impath = buffer;
		// File with names of images
		in.getline(buffer,400);
		in.getline(buffer,400);
		imfiles = buffer;
		// Extract features
		in.getline(buffer,400);
		in >> xtrFeature;
		in.getline(buffer,400); 
		// Scales
		in.getline(buffer,400);
		int size;
		in >> size;
		scales.resize(size);
		for(int i=0;i<size;++i)
			in >> scales[i];
		in.getline(buffer,400); 
		// Ratio
		in.getline(buffer,400);
		in >> size;
		ratios.resize(size);
		for(int i=0;i<size;++i)
			in >> ratios[i];
		in.getline(buffer,400); 
		// Output path
		in.getline(buffer,400);
		in.getline(buffer,400);
		outpath = buffer;
		// Scale factor for output image (default: 128)
		in.getline(buffer,400);
		in >> out_scale;
		in.getline(buffer,400);
		// Path to positive examples
		in.getline(buffer,400);
		in.getline(buffer,400);
		trainpospath = buffer;
		// File with postive examples
		in.getline(buffer,400);
		in.getline(buffer,400);
		trainposfiles = buffer;
		// Subset of positive images -1: all images
		in.getline(buffer,400);
		in >> subsamples_pos;
		in.getline(buffer,400);
		// Samples from pos. examples
		in.getline(buffer,400);
		in >> samples_pos;
		in.getline(buffer,400);
		// Path to positive examples
		in.getline(buffer,400);
		in.getline(buffer,400);
		trainnegpath = buffer;
		// File with postive examples
		in.getline(buffer,400);
		in.getline(buffer,400);
		trainnegfiles = buffer;
		// Subset of negative images -1: all images
		in.getline(buffer,400);
		in >> subsamples_neg;
		in.getline(buffer,400);
		// Samples from pos. examples
		in.getline(buffer,400);
		in >> samples_neg;
		//in.getline(buffer,400);

	} else {
		cerr << "File not found " << filename << endl;
		exit(-1);
	}
	in.close();

	switch ( mode ) { 
		case 0:
		cout << endl << "------------------------------------" << endl << endl;
		cout << "Training:         " << endl;
		cout << "Patches:          " << p_width << " " << p_height << endl;
		cout << "Train pos:        " << trainpospath << endl;
		cout << "                  " << trainposfiles << endl;
		cout << "                  " << subsamples_pos << " " << samples_pos << endl;
		cout << "Train neg:        " << trainnegpath << endl;
		cout << "                  " << trainnegfiles << endl;
		cout << "                  " << subsamples_neg << " " << samples_neg << endl;
		cout << "Trees:            " << ntrees << " " << off_tree << " " << treepath << endl;
		cout << endl << "------------------------------------" << endl << endl;
		break;

		case 1:
		cout << endl << "------------------------------------" << endl << endl;
		cout << "Show:             " << endl;
		cout << "Trees:            " << ntrees << " " << treepath << endl;
		cout << endl << "------------------------------------" << endl << endl;
		break;

		default:
		cout << endl << "------------------------------------" << endl << endl;
		cout << "Detection:        " << endl;
		cout << "Trees:            " << ntrees << " " << treepath << endl;
		cout << "Patches:          " << p_width << " " << p_height << endl;
		cout << "Images:           " << impath << endl;
		cout << "                  " << imfiles << endl;
		cout << "Scales:           "; for(unsigned int i=0;i<scales.size();++i) cout << scales[i] << " "; cout << endl;
		cout << "Ratios:           "; for(unsigned int i=0;i<ratios.size();++i) cout << ratios[i] << " "; cout << endl;
		cout << "Extract Features: " << xtrFeature << endl;
		cout << "Output:           " << out_scale << " " << outpath << endl;
		cout << endl << "------------------------------------" << endl << endl;
		break;
	}

}

// load test image filenames
void loadImFile(std::vector<string>& vFilenames) {
	
	char buffer[400];

	ifstream in(imfiles.c_str());
	if(in.is_open()) {

		unsigned int size;
		in >> size; //size = 10;
		in.getline(buffer,400); 
		vFilenames.resize(size);

		for(unsigned int i=0; i<size; ++i) {
			in.getline(buffer,400);      
			vFilenames[i] = buffer;	
		}

	} else {
		cerr << "File not found " << imfiles.c_str() << endl;
		exit(-1);
	}

	in.close();
}

// load positive training image filenames
void loadTrainPosFile(std::vector<string>& vFilenames, std::vector<CvRect>& vBBox, std::vector<std::vector<CvPoint> >& vCenter) {

	unsigned int size, numop; 
	ifstream in(trainposfiles.c_str());

	if(in.is_open()) {
		in >> size;
		in >> numop;
		cout << "Load Train Pos Examples: " << size << " - " << numop << endl;

		vFilenames.resize(size);
		vCenter.resize(size);
		vBBox.resize(size);

		for(unsigned int i=0; i<size; ++i) {
			// Read filename
			in >> vFilenames[i];

			// Read bounding box
			in >> vBBox[i].x; in >> vBBox[i].y; 
			in >> vBBox[i].width;
			vBBox[i].width -= vBBox[i].x; 
			in >> vBBox[i].height;
			vBBox[i].height -= vBBox[i].y;

			if(vBBox[i].width<p_width || vBBox[i].height<p_height) {
			  cout << "Width or height are too small" << endl; 
			  cout << vFilenames[i] << endl;
			  exit(-1); 
			}

			// Read center points
			vCenter[i].resize(numop);
			for(unsigned int c=0; c<numop; ++c) {			
				in >> vCenter[i][c].x;
				in >> vCenter[i][c].y;
			}				
		}

		in.close();
	} else {
		cerr << "File not found " << trainposfiles.c_str() << endl;
		exit(-1);
	}
}

// load negative training image filenames
void loadTrainNegFile(std::vector<string>& vFilenames, std::vector<CvRect>& vBBox) {

	unsigned int size, numop; 
	ifstream in(trainnegfiles.c_str());

	if(in.is_open()) {
		in >> size;
		in >> numop;
		cout << "Load Train Neg Examples: " << size << " - " << numop << endl;

		vFilenames.resize(size);
		if(numop>0)
			vBBox.resize(size);
		else
			vBBox.clear();

		for(unsigned int i=0; i<size; ++i) {
			// Read filename
			in >> vFilenames[i];

			// Read bounding box (if available)
			if(numop>0) {
				in >> vBBox[i].x; in >> vBBox[i].y; 
				in >> vBBox[i].width;
				vBBox[i].width -= vBBox[i].x; 
				in >> vBBox[i].height;
				vBBox[i].height -= vBBox[i].y;

				if(vBBox[i].width<p_width || vBBox[i].height<p_height) {
				  cout << "Width or height are too small" << endl; 
				  cout << vFilenames[i] << endl;
				  exit(-1); 
				}

			}

				
		}

		in.close();
	} else {
		cerr << "File not found " << trainposfiles.c_str() << endl;
		exit(-1);
	}
}

// Show leaves
void show() {
	// Init forest with number of trees
	CRForest crForest( ntrees ); 

	// Load forest
	crForest.loadForest(treepath.c_str());	

	// Show leaves
	crForest.show(100,100);
}


// Run detector
void detect(CRForestDetector& crDetect) {

	// Load image names
	vector<string> vFilenames;
	loadImFile(vFilenames);
				
	char buffer[200];

	// Storage for output
	vector<vector<IplImage*> > vImgDetect(scales.size());	

	// Run detector for each image
	for(unsigned int i=0; i<vFilenames.size(); ++i) {

		// Load image
		IplImage *img = 0;
		img = cvLoadImage((impath + "/" + vFilenames[i]).c_str(),CV_LOAD_IMAGE_COLOR);
		if(!img) {
			cout << "Could not load image file: " << (impath + "/" + vFilenames[i]).c_str() << endl;
			exit(-1);
		}	

		// Prepare scales
		for(unsigned int k=0;k<vImgDetect.size(); ++k) {
			vImgDetect[k].resize(ratios.size());
			for(unsigned int c=0;c<vImgDetect[k].size(); ++c) {
				vImgDetect[k][c] = cvCreateImage( cvSize(int(img->width*scales[k]+0.5),int(img->height*scales[k]+0.5)), IPL_DEPTH_32F, 1 );
			}
		}

		// Detection for all scales
		crDetect.detectPyramid(img, vImgDetect, ratios);

		// Store result
		for(unsigned int k=0;k<vImgDetect.size(); ++k) {
			IplImage* tmp = cvCreateImage( cvSize(vImgDetect[k][0]->width,vImgDetect[k][0]->height) , IPL_DEPTH_8U , 1);
			for(unsigned int c=0;c<vImgDetect[k].size(); ++c) {
				cvConvertScale( vImgDetect[k][c], tmp, out_scale); //80 128
				sprintf_s(buffer,"%s/detect-%d_sc%d_c%d.png",outpath.c_str(),i,k,c);
				cvSaveImage( buffer, tmp );
				cvReleaseImage(&vImgDetect[k][c]);
			}
			cvReleaseImage(&tmp);
		}

		// Release image
		cvReleaseImage(&img);

	}

}

// Extract patches from training data
void extract_Patches(CRPatch& Train, CvRNG* pRNG) {
		
	vector<string> vFilenames;
	vector<CvRect> vBBox;
	vector<vector<CvPoint> > vCenter;

	// load positive file list
	loadTrainPosFile(vFilenames,  vBBox, vCenter);

	// load postive images and extract patches
	for(int i=0; i<(int)vFilenames.size(); ++i) {

	  if(i%50==0) cout << i << " " << flush;

	  if(subsamples_pos <= 0 || (int)vFilenames.size()<=subsamples_pos || (cvRandReal(pRNG)*double(vFilenames.size()) < double(subsamples_pos)) ) {

			// Load image
			IplImage *img = 0;
			img = cvLoadImage((trainpospath + "/" + vFilenames[i]).c_str(),CV_LOAD_IMAGE_COLOR);
			if(!img) {
				cout << "Could not load image file: " << (trainpospath + "/" + vFilenames[i]).c_str() << endl;
				exit(-1);
			}	

			// Extract positive training patches
			Train.extractPatches(img, samples_pos, 1, &vBBox[i], &vCenter[i]); 

			// Release image
			cvReleaseImage(&img);

	  }
			
	}
	cout << endl;

	// load negative file list
	loadTrainNegFile(vFilenames,  vBBox);

	// load negative images and extract patches
	for(int i=0; i<(int)vFilenames.size(); ++i) {

		if(i%50==0) cout << i << " " << flush;

		if(subsamples_neg <= 0 || (int)vFilenames.size()<=subsamples_neg || ( cvRandReal(pRNG)*double(vFilenames.size()) < double(subsamples_neg) ) ) {

			// Load image
			IplImage *img = 0;
			img = cvLoadImage((trainnegpath + "/" + vFilenames[i]).c_str(),CV_LOAD_IMAGE_COLOR);

			if(!img) {
				cout << "Could not load image file: " << (trainnegpath + "/" + vFilenames[i]).c_str() << endl;
				exit(-1);
			}	

			// Extract negative training patches
			if(vBBox.size()==vFilenames.size())
				Train.extractPatches(img, samples_neg, 0, &vBBox[i]); 
			else
				Train.extractPatches(img, samples_neg, 0); 

			// Release image
			cvReleaseImage(&img);

		}
			
	}
	cout << endl;
}

// Init and start detector
void run_detect() {
	// Init forest with number of trees
	CRForest crForest( ntrees ); 

	// Load forest
	crForest.loadForest(treepath.c_str());	

	// Init detector
	CRForestDetector crDetect(&crForest, p_width, p_height);

	// create directory for output
	string execstr = "mkdir ";
	execstr += outpath;
	system( execstr.c_str() );

	// run detector
	detect(crDetect);
}

// Init and start training
void run_train() {
	// Init forest with number of trees
	CRForest crForest( ntrees ); 

	// Init random generator
	time_t t = time(NULL);
	int seed = (int)t;

	CvRNG cvRNG(seed);
						
	// Create directory
	string tpath(treepath);
	tpath.erase(tpath.find_last_of(PATH_SEP));
	string execstr = "mkdir ";
	execstr += tpath;
	system( execstr.c_str() );

	// Init training data
	CRPatch Train(&cvRNG, p_width, p_height, 2); 
			
	// Extract training patches
	extract_Patches(Train, &cvRNG); 

	// Train forest
	crForest.trainForest(20, 15, &cvRNG, Train, 2000);

	// Save forest
	crForest.saveForest(treepath.c_str(), off_tree);

}

int main(int argc, char* argv[])
{
	int mode = 1;

	// Check argument
	if(argc<2) {
		cout << "Usage: CRForest-Detector.exe mode [config.txt] [tree_offset]" << endl;
		cout << "mode: 0 - train; 1 - show; 2 - detect" << endl;
		cout << "tree_offset: output number for trees" << endl;
		cout << "Load default: mode - 2" << endl; 
	} else
		mode = atoi(argv[1]);
	
	off_tree = 0;	
	if(argc>3)
		off_tree = atoi(argv[3]);

	// load configuration for dataset
	if(argc>2)
		loadConfig(argv[2], mode);
	else
		loadConfig("config.txt", mode);

	switch ( mode ) { 
		case 0: 	
			// train forest
			run_train();
			break;	

		case 1: 
					
			// train forest
			show();
			break;	

		default:

			// detection
			run_detect();
			break;
	}


	return 0;
}


