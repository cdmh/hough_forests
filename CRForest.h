/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#include "CRTree.h"

#include <vector>
#include <omp.h>

namespace gall {

class CRForest {
public:
	// Constructors
	CRForest(size_t trees = 0) {
		vTrees.resize(trees);
	}
	~CRForest() {
		for(std::vector<CRTree*>::iterator it = vTrees.begin(); it != vTrees.end(); ++it) delete *it;
		vTrees.clear();
	}

    CRForest(CRForest &) = delete;  // it's not safe to copy a forest

	// Set/Get functions
	void SetTrees(int n) {vTrees.resize(n);}
	size_t GetSize() const {return vTrees.size();}
	unsigned int GetDepth() const {return vTrees[0]->GetDepth();}
	size_t       GetNumCenter() const {return vTrees[0]->GetNumCenter();}
	
	// Regression 
	void regression(std::vector<const LeafNode*>& result, uchar** ptFCh, int stepImg) const;

	// Training
	void trainForest(int min_s, int max_d, CvRNG* pRNG, const CRPatch& TrData, int samples);

	// IO functions
	void saveForest(const char* filename, unsigned int offset = 0, int type = 0);
	void loadForest(const char* filename, int type = 0);
	void show(int w, int h) const {vTrees[0]->showLeaves(w,h);}

	// Trees
	std::vector<CRTree*> vTrees;
};

inline void CRForest::regression(std::vector<const LeafNode*>& result, uchar** ptFCh, int stepImg) const {
	result.resize( vTrees.size() );

    #pragma omp parallel for 
	for(int i=0; i<(int)vTrees.size(); ++i) {
		result[i] = vTrees[i]->regression(ptFCh, stepImg);
	}
}

//Training
inline void CRForest::trainForest(int min_s, int max_d, CvRNG* pRNG, const CRPatch& TrData, int samples) {
    for(int i=0; i < (int)vTrees.size(); ++i)
        vTrees[i] = new CRTree( min_s, max_d, TrData.vLPatches[1][0].center.size(), pRNG);

    #pragma omp parallel for 
    for(int i=0; i < (int)vTrees.size(); ++i)
    {
        std::cout << "(Thread " << omp_get_thread_num() << ") Tree "
                  << i << " trained to depth: "
                  << vTrees[i]->growTree(TrData, samples)
                  << " (max depth=" << max_d << ")\n";
    }
}

// IO Functions
inline void CRForest::saveForest(const char* filename, unsigned int offset, int type) {
    if (type == 0)
    {
	    char buffer[200];
	    for(unsigned int i=0; i<vTrees.size(); ++i) {
		    sprintf_s(buffer,"%s%03u.txt",filename,i+offset);
		    vTrees[i]->saveTree(buffer);
	    }
    }
    else if (type == 1  ||  type == 2)
    {
        // composite forest storage // CDMH
        std::ofstream out(filename, std::ios_base::out | ((type == 2)? std::ios::binary : 0));
        size_t const size = vTrees.size();
        out.write((char const *)&size, sizeof(size));
	    for(unsigned int i=0; i<vTrees.size(); ++i) {
		    vTrees[i]->save(out, (type == 2));
	    }
    }
}

inline void CRForest::loadForest(const char* filename, int type)
{
    if (type == 0)
    {
	    char buffer[200];
	    for(unsigned int i=0; i<vTrees.size(); ++i) {
		    sprintf_s(buffer,"%s%03u.txt",filename,i);
		    vTrees[i] = new CRTree(buffer);
	    }
    }
    else if (type == 1  ||  type == 2)
    {
        // composite forest storage // CDMH
        std::ifstream in(filename, std::ios_base::in | ((type == 2)? std::ios::binary : 0));
        size_t size;
        in.read((char *)&size, sizeof(size));
        vTrees.resize(size);
	    for(unsigned int i=0; i<vTrees.size(); ++i)
		    vTrees[i] = new CRTree(in, (type == 2));
    }
}

}   // namespace gall
