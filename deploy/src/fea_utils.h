
#ifndef DTLN_DEPLOY_FEA_UTILS_H
#define DTLN_DEPLOY_FEA_UTILS_H

#include "fftw3.h"

namespace ns
{
    void S16ToFloat(const short * src, size_t size, float* dest);
    void FloatToS16(const float * src, size_t size, short * dest);

    void  LogPowerSpecFea(const fftwf_complex * in, size_t size, float * dest);
    void MagAndPhaseFea(const fftwf_complex * in, size_t size, float * mag, float * phase);
}
#endif //DTLN_DEPLOY_FEA_UTILS_H
