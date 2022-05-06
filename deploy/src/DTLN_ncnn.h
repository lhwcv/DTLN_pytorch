
#ifndef DTLN_DEPLOY_DTLN_NCNN_H
#define DTLN_DEPLOY_DTLN_NCNN_H

#include <vector>
#include <string>
#include <iostream>
#include "fftw3.h"
#include "ncnn/net.h"

namespace ns
{
    class  DTLN_ncnn
    {
    public:
        DTLN_ncnn();

        int load_model(std::vector<std::string> &modelPath);
        int process_1_channel_trunk_by_trunk(const char * indata, char *outdata, size_t buffsize);
        int trunk_size() const {return  _trunk_size; }
        int trunk_size_in_bytes() const {return  _trunk_size * 2; }

        ~DTLN_ncnn();

    private:
        int _trunk_size;
        int _block_size;
        int _block_shift;
        int _fea_size;

        float * _inblock_Real = nullptr;
        float * _mag = nullptr;
        float * _phase = nullptr;

        float * _outblock_Real = nullptr;
        float * _outblock_Buf = nullptr;

        float * _indata = nullptr;
        float * _outdata = nullptr;

        fftwf_complex *_inSpec_Complex = nullptr;
        fftwf_plan _fft_plan;
        fftwf_plan _ifft_plan;

        ncnn::Net _net1, _net2;
        ncnn::Mat _h1,_c1, _h2, _c2;
        std::vector<ncnn::Mat> _net1_States;
        std::vector<ncnn::Mat> _net2_States;


    };
}

#endif //DTLN_DEPLOY_DTLN_NCNN_H
