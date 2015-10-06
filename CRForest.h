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
	size_t       GetSize() const {return vTrees.size();}
	unsigned int GetDepth() const {return vTrees[0]->GetDepth();}
	size_t       GetNumCenter() const {return vTrees[0]->GetNumCenter();}
	void         SetTrees(int n) {vTrees.resize(n);}
	
	// Regression 
	void regression(std::vector<const LeafNode*>& result, uchar** ptFCh, int stepImg) const;

	// Training
	bool const trainForest(int min_s, int max_d, CvRNG rng, const CRPatch& TrData, int samples);

	// IO functions
	void saveForest(const char* filename, unsigned int offset = 0, int type = 0);
	void saveForest(std::ofstream &out);
	void loadForest(const char* filename, int type = 0);
	void loadForest(std::ifstream &in);
	void show(int w, int h) const {vTrees[0]->showLeaves(w,h);}
    void stats() const;

	// Trees
	std::vector<CRTree*> vTrees;
};

inline void CRForest::regression(std::vector<const LeafNode*>& result, uchar** ptFCh, int stepImg) const
{
	assert(result.size() == vTrees.size());
	for(int i=0; i<(int)vTrees.size(); ++i) {
		result[i] = vTrees[i]->regression(ptFCh, stepImg);
	}
}

//Training
inline bool const CRForest::trainForest(int min_s, int max_d, CvRNG rng, const CRPatch& TrData, int samples)
{
    for (auto const &patches : TrData.vLPatches)
    {
        if (patches.size() == 0)
        {
            std::cerr << "Unable to train forest -- missing training data";
            return false;
        }
    }

    // not thread safe because of the RNG
    for(int i=0; i < (int)vTrees.size(); ++i)
        vTrees[i] = new CRTree(min_s, max_d, TrData.vLPatches[1][0].center.size(), cvRandInt(&rng));

#   pragma omp parallel for 
    for (int i=0; i < (int)vTrees.size(); ++i)
    {
        auto const depth = vTrees[i]->growTree(TrData, samples);
        std::cout << "(Thread " << omp_get_thread_num() << ") Tree "
                  << i     << " trained to depth: "
                  << depth << " (max depth=" << max_d << ")\n";
    }
    return true;
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
        std::ofstream out(filename, std::ios_base::out | std::ios::binary);
        saveForest(out);
    }
}

inline void CRForest::saveForest(std::ofstream &out)
{
    // composite forest storage // CDMH
    size_t const size = vTrees.size();
    out.write((char const *)&size, sizeof(size));
	for(unsigned int i=0; i<vTrees.size(); ++i) {
		vTrees[i]->save(out);
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
        std::ifstream in(filename, std::ios_base::in | std::ios::binary);
        loadForest(in);
    }
}

inline void CRForest::loadForest(std::ifstream &in)
{
    // composite forest storage // CDMH
    size_t size;
    in.read((char *)&size, sizeof(size));
    vTrees.resize(size);
	for(unsigned int i=0; i<vTrees.size(); ++i)
		vTrees[i] = new CRTree(in);
}

inline void CRForest::stats() const
{
	for(unsigned int i=0; i<vTrees.size(); ++i)
    {
        std::cout << "Tree " << i << '\n';
		vTrees[i]->stats();
    }
}

}   // namespace gall
