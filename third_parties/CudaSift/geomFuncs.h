#ifndef CUDA_FIST_GEOM_FUNS_H_
#define CUDA_FIST_GEOM_FUNS_H_

#include "cudaImage.h"
#include "cudaSift.h"

int ImproveHomography(SiftData &data, float *homography, int numLoops,
                      float minScore, float maxAmbiguity, float thresh);

#endif  // CUDA_FIST_GEOM_FUNS_H_
