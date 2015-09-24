/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#ifndef _MSC_VER
#define sprintf_s sprintf 
#endif

#include "CRPatch.h"
#include <iostream>
#include <fstream>

namespace gall {

// Auxilary structure
struct IntIndex {
	int val;
	unsigned int index;
	bool operator<(const IntIndex& a) const { return val<a.val; }
};

// Structure for the leafs
struct LeafNode {
	// Constructors
	LeafNode() : pfg() {}

	// IO functions
	void show(int delay, int width, int height); 
	void print() const {
		std::cout << "Leaf " << vCenter.size() << " "  << pfg << std::endl;
	}

	// Probability of foreground
	float pfg;

	// Vectors from object center to training patches
	std::vector<std::vector<CvPoint> > vCenter;	

    // patch roi
	std::vector<CvRect> roi;

    // index of the src image
    std::vector<int> src_indices;
};

inline
bool const operator==(LeafNode const &first, LeafNode const &second)
{
    if (first.pfg != second.pfg)
        return false;

    if (first.vCenter.size() != second.vCenter.size())
        return false;

    for (size_t i=0; i<first.vCenter.size(); ++i)
    {
        if (first.vCenter[i].size() != second.vCenter[i].size())
            return false;

        for (size_t j=0; j<first.vCenter[i].size(); ++j)
        {
            if (first.vCenter[i][j].x != second.vCenter[i][j].x)
                return false;
            else if (first.vCenter[i][j].y != second.vCenter[i][j].y)
                return false;
        }
    }

    return true;
}


inline
bool const operator!=(LeafNode const &first, LeafNode const &second)
{
    return !(first == second);
}

class CRTree {
public:
	// Constructors
	explicit CRTree(const char* filename);
    explicit CRTree(std::ifstream &in);
	CRTree(int min_s, int max_d, size_t cp, CvRNG rng) : min_samples(min_s), max_depth(max_d), num_leaf(0), num_cp(cp), rng(rng) {
		num_nodes = (int)pow(2.0,int(max_depth+1))-1;
		// num_nodes x 7 matrix as vector
		treetable = new int[num_nodes * 7];
		for(unsigned int i=0; i<num_nodes * 7; ++i) treetable[i] = 0;
		// allocate memory for leafs
		leaf = new LeafNode[(int)pow(2.0,int(max_depth))];
	}
	~CRTree() {delete[] leaf; delete[] treetable;}

	// Set/Get functions
	unsigned int    GetDepth()       const { return max_depth; }
	size_t          GetNumCenter()   const { return num_cp;    }
	unsigned int    GetNumLeaves()   const { return num_leaf;  }
	LeafNode const &GetLeaf(int ndx) const { return leaf[ndx]; }

	// Regression
	const LeafNode* regression(uchar** ptFCh, int stepImg) const;

	// Training
	unsigned int growTree(const CRPatch& TrData, int samples);

	// IO functions
	bool saveTree(const char* filename) const;
    bool const save(std::ofstream &out) const;
	void showLeaves(int width, int height) const {
		for(unsigned int l=0; l<num_leaf; ++l)
			leaf[l].show(5000, width, height);
	}
    void stats() const;

private: 
    bool const load(std::ifstream &in);

	// Private functions for training
	unsigned int grow(const std::vector<std::vector<const PatchFeature*> >& TrainSet, int node, unsigned int depth, int samples, float pnratio);
	void makeLeaf(const std::vector<std::vector<const PatchFeature*> >& TrainSet, float pnratio, int node);
	bool optimizeTest(std::vector<std::vector<const PatchFeature*> >& SetA, std::vector<std::vector<const PatchFeature*> >& SetB, const std::vector<std::vector<const PatchFeature*> >& TrainSet, int* test, unsigned int iter, unsigned int mode);
	void generateTest(int* test, size_t max_w, size_t max_h, size_t max_c);
	void evaluateTest(std::vector<std::vector<IntIndex> >& valSet, const int* test, const std::vector<std::vector<const PatchFeature*> >& TrainSet);
	void split(std::vector<std::vector<const PatchFeature*> >& SetA, std::vector<std::vector<const PatchFeature*> >& SetB, const std::vector<std::vector<const PatchFeature*> >& TrainSet, const std::vector<std::vector<IntIndex> >& valSet, int t);
	double measureSet(const std::vector<std::vector<const PatchFeature*> >& SetA, const std::vector<std::vector<const PatchFeature*> >& SetB, unsigned int mode) {
	  if (mode==0) return InfGain(SetA, SetB); else return -distMean(SetA[1],SetB[1]);
	}
	double distMean(const std::vector<const PatchFeature*>& SetA, const std::vector<const PatchFeature*>& SetB);
	double InfGain(const std::vector<std::vector<const PatchFeature*> >& SetA, const std::vector<std::vector<const PatchFeature*> >& SetB);


	// Data structure

	// tree table
	// 2^(max_depth+1)-1 x 7 matrix as vector
	// column: leafindex x1 y1 x2 y2 channel thres
	// if node is not a leaf, leaf=-1
	int* treetable;

	// stop growing when number of patches is less than min_samples
	unsigned int min_samples;

	// depth of the tree: 0-max_depth
	unsigned int max_depth;

	// number of nodes: 2^(max_depth+1)-1
	unsigned int num_nodes;

	// number of leafs
	unsigned int num_leaf;

	// number of center points per patch
	size_t num_cp;

	//leafs as vector
	LeafNode* leaf;

	CvRNG rng;

    friend bool const operator==(CRTree const &first, CRTree const &second);
};

inline
bool const operator==(CRTree const &first, CRTree const &second)
{
	if (first.min_samples != second.min_samples)
        return false;

	if (first.max_depth != second.max_depth)
        return false;

	if (first.num_nodes != second.num_nodes)
        return false;

	if (first.num_leaf != second.num_leaf)
        return false;

	if (first.num_cp != second.num_cp)
        return false;

	if (memcmp(first.treetable, second.treetable, sizeof(int) * first.num_nodes * 7) != 0)
        return false;

    for (unsigned i=0; i<first.num_leaf; ++i)
    {
        if (first.leaf[i] != second.leaf[i])
            return false;
    }

    return true;
}

inline const LeafNode* CRTree::regression(uchar** ptFCh, int stepImg) const {
	// pointer to current node                                                     
	const int* pnode = &treetable[0];
	int node = 0;

	// Go through tree until one arrives at a leaf, i.e. pnode[0]>=0)
	while(pnode[0]==-1) {
		// binary test 0 - left, 1 - right
		// Note that x, y are changed since the patches are given as matrix and not as image 
		// p1 - p2 < t -> left is equal to (p1 - p2 >= t) == false
		
		// pointer to channel
		uchar* ptC = ptFCh[pnode[5]];
		// get pixel values 
		int p1 = *(ptC+pnode[1]+pnode[2]*stepImg);
		int p2 = *(ptC+pnode[3]+pnode[4]*stepImg);
		// test
		bool test = ( p1 - p2 ) >= pnode[6];

		// next node: 2*node_id + 1 + test
		// increment node/pointer by node_id + 1 + test
		int incr = node+1+test;
		node += incr;
		pnode += incr*7;
	}

	// return leaf
	return &leaf[pnode[0]];
}

inline void CRTree::generateTest(int* test, size_t max_w, size_t max_h, size_t max_c) {
	test[0] = cvRandInt( &rng ) % max_w;
	test[1] = cvRandInt( &rng ) % max_h;
	test[2] = cvRandInt( &rng ) % max_w;
	test[3] = cvRandInt( &rng ) % max_h;
	test[4] = cvRandInt( &rng ) % max_c;
}

}   // namespace gall
