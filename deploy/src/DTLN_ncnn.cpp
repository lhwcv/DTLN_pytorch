#include "DTLN_ncnn.h"
#include "fea_utils.h"
#include "fftw3.h"
#include <cmath>

namespace ns
{
    #define  PI 3.141592653589793f

    DTLN_ncnn::DTLN_ncnn()
    {
        _block_size = 512;
        _trunk_size = 128;
        _block_shift = 128;
        _fea_size = _block_size / 2 + 1;

        //_inblock = new float [_block_size];
        //memset(_inblock, 0, sizeof(float) * _block_size);

        _indata = new float [_trunk_size];
        _outdata = new float [_trunk_size];

        int nfft = _block_size;

        _inblock_Real = fftwf_alloc_real(nfft);
        _outblock_Real = fftwf_alloc_real(nfft);
        _outblock_Buf = new float [_block_size];
        memset(_inblock_Real, 0, sizeof(float) * _block_size);
        memset(_outblock_Buf, 0, sizeof(float) * _block_size);

        _inSpec_Complex = fftwf_alloc_complex(nfft/2+1);
        _fft_plan = fftwf_plan_dft_r2c_1d(nfft, _inblock_Real, _inSpec_Complex, FFTW_ESTIMATE);
        _ifft_plan = fftwf_plan_dft_c2r_1d(nfft, _inSpec_Complex, _outblock_Real, FFTW_ESTIMATE);

        _mag = new float [_fea_size];
        _phase = new float [_fea_size];

    }
    DTLN_ncnn::~DTLN_ncnn()
    {
        if(_inblock_Real != nullptr)
        {
            fftwf_free(_inblock_Real);
            _inblock_Real = nullptr;
        }
        if(_outblock_Real != nullptr)
        {
            fftwf_free(_outblock_Real);
            _outblock_Real = nullptr;
        }
        if(_outblock_Buf != nullptr)
        {
            delete [] _outblock_Buf;
            _outblock_Buf = nullptr;
        }

        if(_mag != nullptr)
        {
            delete [] _mag;
            _mag = nullptr;
        }
        if(_phase != nullptr)
        {
            delete [] _phase;
            _phase = nullptr;
        }

        if(_inSpec_Complex != nullptr)
        {
            fftwf_free(_inSpec_Complex);
            _inSpec_Complex = nullptr;

            fftwf_destroy_plan(_fft_plan);
            fftwf_destroy_plan(_ifft_plan);
        }

        if(_indata != nullptr)
        {
            delete [] _indata;
            _indata = nullptr;
        }
        if(_outdata != nullptr)
        {
            delete [] _outdata;
            _outdata = nullptr;
        }
    }

    int DTLN_ncnn::load_model(std::vector<std::string> &modelPath)
    {
        if(modelPath.size() != 4)
        {
            std::cout<<"model laod err!"<<std::endl;
            return -1;
        }

        _net1.load_param(modelPath[0].c_str());
        _net1.load_model(modelPath[1].c_str());
        _net2.load_param(modelPath[2].c_str());
        _net2.load_model(modelPath[3].c_str());

        int nStates = 4;
        _net1_States.resize(nStates);
        _net2_States.resize(nStates);
        for(int i=0; i< nStates; i++)
        {
            _net1_States[i].create(128, 1, 1);
            _net1_States[i].fill(0.0f);

            _net2_States[i].create(128, 1, 1);
            _net2_States[i].fill(0.0f);
        }

        _h1.create(128, 1, 1);
        _h1.fill(0.0f);
        _h2.create(128, 1, 1);
        _h2.fill(0.0f);
        _c1.create(128, 1, 1);
        _c1.fill(0.0f);
        _c2.create(128, 1, 1);
        _c2.fill(0.0f);


        return 0;
    }

    int DTLN_ncnn::process_1_channel_trunk_by_trunk(const char *indata, char *outdata, size_t buffsize)
    {
        if(buffsize != trunk_size_in_bytes() )
        {
            std::cout<<"[NS] only support buffsize == trunk_size_in_bytes()!"<<std::endl;
            return -1;
        }

        //memcpy(outdata, indata, buffsize);

        S16ToFloat((short *)indata,  _trunk_size, _indata);


        memmove(_inblock_Real, _inblock_Real + _block_shift, (_block_size - _block_shift) * sizeof(float));
        memcpy(_inblock_Real + (_block_size - _block_shift), _indata, _block_shift * sizeof(float));


        fftwf_execute(_fft_plan);

        MagAndPhaseFea(_inSpec_Complex, _fea_size, _mag, _phase);

        ncnn::Mat ncnn_inFlt(_fea_size, 1, 1, _mag);
        ncnn::Extractor ex1 = _net1.create_extractor();
        ex1.set_num_threads(1);
        ex1.input("mag", ncnn_inFlt);
        ex1.input("h1_in", _net1_States[0]);
        ex1.input("c1_in", _net1_States[1]);
        ex1.input("h2_in", _net1_States[2]);
        ex1.input("c2_in", _net1_States[3]);

        ncnn::Mat mag;
        ex1.extract("76", mag);

        ex1.extract("38", _net1_States[0]);
        ex1.extract("39", _net1_States[1]);
        ex1.extract("63", _net1_States[2]);
        ex1.extract("64", _net1_States[3]);

        float * mag_p = (float *)mag.data;

        for(size_t i = 0; i < _fea_size; i++)
        {
            //Euler's equation
            _inSpec_Complex[i][0] = mag_p[i] * cosf(_phase[i]);
            _inSpec_Complex[i][1] = mag_p[i] * sinf(_phase[i]);
        }
        fftwf_execute(_ifft_plan);// ifft to _outblock_Real

        for(size_t i = 0; i < _block_size; i++)
        {
            _outblock_Real[i] = _outblock_Real[i] / _block_size;
        }


        ncnn::Mat ncnn_y1(1, _block_size, 1,_outblock_Real);
        ncnn::Extractor ex2 = _net2.create_extractor();
        ex2.set_num_threads(1);
        ex2.input("y1", ncnn_y1);
        ex2.input("h1_in", _net2_States[0]);
        ex2.input("c1_in", _net2_States[1]);
        ex2.input("h2_in", _net2_States[2]);
        ex2.input("c2_in", _net2_States[3]);


        ncnn::Mat y;
        ex2.extract("y", y);

        ex2.extract("54", _net2_States[0]);
        ex2.extract("55", _net2_States[1]);
        ex2.extract("79", _net2_States[2]);
        ex2.extract("80", _net2_States[3]);

        //std::cout<<"out shape: w: "<<y.w <<",h: "<<y.h <<",c: "<<y.c<<std::endl;

        float * yp = (float *)y.data;

        //shift to left
        memmove(_outblock_Buf, _outblock_Buf + _block_shift, (_block_size - _block_shift) * sizeof(float));
        memset(_outblock_Buf + (_block_size - _block_shift), 0, _block_shift * sizeof(float));

        for (int i = 0; i < _block_size; i++)
        {
            _outblock_Buf[i] = _outblock_Buf[i] + yp[i];
        }
        for(int i=0;  i < _trunk_size; i++)
        {
            _outdata[i] = _outblock_Buf[i];
        }

#if 0
        float sum =0.0;
        static int ii=0;
        if(ii < 10)
        {
            for(int i=0; i< _block_size; i++)
            {
                sum += abs(_outblock_Buf[i]);
            }
            std::cout<<ii<<" ==> out: "<<sum<<std::endl;
        }
        ii +=1;
#endif
        FloatToS16(_outdata, _trunk_size, (short *)outdata);


        return 0;
    }


}