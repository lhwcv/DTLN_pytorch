
#include "fea_utils.h"
#include <numeric>
#include <cmath>

namespace ns
{
    static inline float FloatS16ToFloat(float v)
    {
        v = fmin(v, 32768.f);
        v = fmax(v, -32768.f);
        constexpr float kScaling = 1.f / 32768.f;
        return v * kScaling;
    }
    static inline int16_t Float32ToS16(float v)
    {
        v = fmin(v, 1.f);
        v = fmax(v, -1.f);
        v = v * 32768.f;
        return static_cast<int16_t>(v + std::copysign(0.5f, v));
    }
    static inline float LogPowerSpec(float real, float  imag)
    {
        float  eps = 1e-12;
        float p = real * real + imag * imag;
        p = p < eps?eps: p;
        p = log10f(p);
        return p;
    }


    void S16ToFloat(const short * src, size_t size, float* dest)
    {
        for (size_t i = 0; i < size; ++i)
            dest[i] = FloatS16ToFloat(static_cast<float>(src[i]));
    }

    void FloatToS16(const float * src, size_t size, short * dest)
    {
        for (size_t i = 0; i < size; ++i)
            dest[i] = Float32ToS16(src[i]);
    }
    void  LogPowerSpecFea(const fftwf_complex * in, size_t size, float * dest)
    {
        for(size_t i = 0; i < size; i++)
            dest[i] = LogPowerSpec(in[i][0], in[i][1]);
    }


    static inline void MagAndPhase(float real, float  imag, float &mag, float  &phase)
    {
        mag = sqrtf(real * real + imag * imag);
        phase = atan2f(imag, real);
//        if(imag == 0.0f)
//        {
//            phase = 0.0f;
//        }
//        else if(real == 0.0f)
//        {
//            phase = PI  / 2;
//        }
//        else
//        {
//            phase = atan2f(imag, real);
//        }
    }

    void MagAndPhaseFea(const fftwf_complex * in, size_t size, float * mag, float * phase)
    {
        for(size_t i = 0; i < size; i++)
            MagAndPhase(in[i][0], in[i][1], mag[i], phase[i]);
    }

}