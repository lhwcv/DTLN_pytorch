#ifndef _NSNET_NCNN_H
#define _NSNET_NCNN_H

#include <vector>
#include <string>
#include <iostream>
#include "fftw3.h"
#include "ncnn/net.h"

namespace ns
{
    class  NSNet_ncnn
    {
    public:
        NSNet_ncnn();
        ~NSNet_ncnn();

        void set_post_gain(float postDB = 3.0f);
        int load_model(std::vector<std::string> &modelPath);
        int process_1_channel_trunk_by_trunk(const char * indata, char *outdata, size_t buffsize);
        int trunk_size() const {return  _trunk_size; }
        int trunk_size_in_bytes() const {return  _trunk_size * 2; }

    private:
        int _trunk_size;
        int _block_size;
        int _block_shift;
        int _fea_size;

        float * _inblock_Real = nullptr;
        float * _inblock_windowed_Real = nullptr;

        float * _logPow = nullptr;

        float * _window = nullptr;

        float * _outblock_Real = nullptr;
        float * _outblock_Buf = nullptr;

        float * _indata = nullptr;
        float * _outdata = nullptr;

        fftwf_complex *_inSpec_Complex = nullptr;
        fftwf_plan _fft_plan;
        fftwf_plan _ifft_plan;

        ncnn::Net _net;
        ncnn::Mat _h1, _h2;
        float _mingain;
        float _postgainDB;


    };
}

#endif //_NSNET_NCNN_H
