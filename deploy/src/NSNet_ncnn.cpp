#include "NSNet_ncnn.h"
#include "fea_utils.h"
#include "fftw3.h"
#include <cmath>
#include <numeric>

namespace ns
{

    NSNet_ncnn::NSNet_ncnn()
    {
        _block_size = 320;
        _trunk_size = 160;
        _block_shift = _trunk_size;
        _fea_size = _block_size / 2 + 1;

        //_inblock = new float [_block_size];
        //memset(_inblock, 0, sizeof(float) * _block_size);

        _indata = new float [_trunk_size];
        _outdata = new float [_trunk_size];

        _logPow = new float [_fea_size];

        int nfft = _block_size;

        _inblock_Real = fftwf_alloc_real(nfft);
        _inblock_windowed_Real = fftwf_alloc_real(nfft);

        for(int i=0; i < nfft; i++)
            _inblock_Real[i] = 0;

        _outblock_Real = fftwf_alloc_real(nfft);
        _outblock_Buf = new float [_block_size];

        memset(_inblock_Real, 0, sizeof(float) * _block_size);
        memset(_outblock_Buf, 0, sizeof(float) * _block_size);

        _inSpec_Complex = fftwf_alloc_complex(nfft/2+1);
        _fft_plan = fftwf_plan_dft_r2c_1d(nfft, _inblock_windowed_Real, _inSpec_Complex, FFTW_ESTIMATE);
        _ifft_plan = fftwf_plan_dft_c2r_1d(nfft, _inSpec_Complex, _outblock_Real, FFTW_ESTIMATE);

        // Hanning Sqrt
        static float window_np[320]={0.0, 0.00984809363948123, 0.01969523213894739, 0.029540460451028643, 0.03938282371361059, 0.0492213673424555, 0.059055137123780875, 0.06888317930680748, 0.0787045406962607, 0.08851826874481623, 0.09832341164548851, 0.10811901842394182, 0.11790413903072322, 0.12767782443340545, 0.13743912670863034, 0.1471870991340458, 0.15692079628012645, 0.16663927410186744, 0.17634159003034497, 0.18602680306413374, 0.19569397386057194, 0.2053421648268658, 0.214970440211024, 0.22457786619261355, 0.23416351097332916, 0.24372644486736467, 0.2532657403915822, 0.2627804723554652, 0.2722697179508507, 0.2817325568414298, 0.2911680712520081, 0.3005753460575194, 0.30995346887178016, 0.31930153013597995, 0.32861862320689683, 0.3379038444448305, 0.34715629330124315, 0.3563750724061022, 0.36555928765491347, 0.3747080482954376, 0.38382046701408207, 0.39289566002195897, 0.40193274714060206, 0.41093085188733197, 0.4198891015602646, 0.4288066273229519, 0.43768256428864744, 0.4465160516041904, 0.4553062325334967, 0.4640522545406519, 0.47275326937259604, 0.48140843314139387, 0.49001690640608014, 0.49857785425407636, 0.5070904463821657, 0.5155538571770217, 0.5239672657952832, 0.5323298562431645, 0.5406408174555977, 0.5488993433748942, 0.5571046330289237, 0.5652558906087972, 0.5733523255460505, 0.58139315258932, 0.589377591880501, 0.5973048690303849, 0.6051742151937651, 0.6129848671440047, 0.6207360673470598, 0.6284270640349514, 0.6360571112786766, 0.6436254690605547, 0.6511314033459994, 0.6585741861547103, 0.6659530956312787, 0.6732674161151978, 0.6805164382102729, 0.6876994588534234, 0.6948157813828715, 0.7018647156057095, 0.7088455778648395, 0.7157576911052795, 0.7226003849398293, 0.7293729957140892, 0.7360748665708269, 0.7427053475136828, 0.749263795470213, 0.7557495743542584, 0.7621620551276366, 0.7685006158611513, 0.7747646417949114, 0.7809535253979546, 0.7870666664271707, 0.7931034719855166, 0.7990633565795209, 0.8049457421760687, 0.8107500582584638, 0.8164757418817611, 0.8221222377273655, 0.8276889981568906, 0.8331754832652729, 0.8385811609331353, 0.8439055068783965, 0.849148004707119, 0.8543081459635935, 0.8593854301796519, 0.864379364923207, 0.8692894658460119, 0.8741152567306354, 0.8788562695366494, 0.8835120444460229, 0.8880821299077181, 0.8925660826814854, 0.8969634678808521, 0.901273859015301, 0.9054968380316338, 0.9096319953545184, 0.9136789299262107, 0.9176372492454534, 0.9215065694055438, 0.9252865151315663, 0.9289767198167914, 0.9325768255582302, 0.936086483191347, 0.9395053523239244, 0.9428331013690763, 0.9460694075774078, 0.9492139570683178, 0.9522664448604417, 0.9552265749012303, 0.9580940600956634, 0.9608686223340943, 0.9635499925192229, 0.9661379105921949, 0.9686321255578235, 0.971032395508934, 0.9733384876498244, 0.9755501783188447, 0.9776672530100885, 0.9796895063941976, 0.9816167423382764, 0.9834487739249144, 0.9851854234703145, 0.9868265225415261, 0.9883719119727817, 0.9898214418809328, 0.9911749716799872, 0.9924323700947445, 0.9935935151735275, 0.9946582943000098, 0.9956266042041394, 0.9964983509721527, 0.9972734500556846, 0.9979518262799678, 0.9985334138511238, 0.9990181563625447, 0.9994060068003627, 0.9996969275480115, 0.9998908903898729, 0.9999878765140148, 0.9999878765140148, 0.9998908903898729, 0.9996969275480115, 0.9994060068003627, 0.9990181563625447, 0.9985334138511238, 0.9979518262799678, 0.9972734500556846, 0.9964983509721527, 0.9956266042041394, 0.9946582943000098, 0.9935935151735275, 0.9924323700947445, 0.9911749716799872, 0.9898214418809328, 0.9883719119727817, 0.9868265225415261, 0.9851854234703145, 0.9834487739249144, 0.9816167423382764, 0.9796895063941976, 0.9776672530100885, 0.9755501783188447, 0.9733384876498244, 0.971032395508934, 0.9686321255578235, 0.9661379105921949, 0.9635499925192229, 0.9608686223340943, 0.9580940600956634, 0.9552265749012303, 0.9522664448604417, 0.9492139570683178, 0.9460694075774078, 0.9428331013690763, 0.9395053523239244, 0.936086483191347, 0.9325768255582302, 0.9289767198167914, 0.9252865151315663, 0.9215065694055438, 0.9176372492454534, 0.9136789299262107, 0.9096319953545184, 0.9054968380316338, 0.901273859015301, 0.8969634678808521, 0.8925660826814854, 0.8880821299077181, 0.8835120444460229, 0.8788562695366494, 0.8741152567306354, 0.8692894658460119, 0.864379364923207, 0.8593854301796519, 0.8543081459635935, 0.849148004707119, 0.8439055068783965, 0.8385811609331353, 0.8331754832652729, 0.8276889981568906, 0.8221222377273655, 0.8164757418817611, 0.8107500582584638, 0.8049457421760687, 0.7990633565795209, 0.7931034719855166, 0.7870666664271707, 0.7809535253979546, 0.7747646417949114, 0.7685006158611513, 0.7621620551276366, 0.7557495743542584, 0.749263795470213, 0.7427053475136828, 0.7360748665708269, 0.7293729957140892, 0.7226003849398293, 0.7157576911052795, 0.7088455778648395, 0.7018647156057095, 0.6948157813828715, 0.6876994588534234, 0.6805164382102729, 0.6732674161151978, 0.6659530956312787, 0.6585741861547103, 0.6511314033459994, 0.6436254690605547, 0.6360571112786766, 0.6284270640349514, 0.6207360673470598, 0.6129848671440047, 0.6051742151937651, 0.5973048690303849, 0.589377591880501, 0.58139315258932, 0.5733523255460505, 0.5652558906087972, 0.5571046330289237, 0.5488993433748942, 0.5406408174555977, 0.5323298562431645, 0.5239672657952832, 0.5155538571770217, 0.5070904463821657, 0.49857785425407636, 0.49001690640608014, 0.48140843314139387, 0.47275326937259604, 0.4640522545406519, 0.4553062325334967, 0.4465160516041904, 0.43768256428864744, 0.4288066273229519, 0.4198891015602646, 0.41093085188733197, 0.40193274714060206, 0.39289566002195897, 0.38382046701408207, 0.3747080482954376, 0.36555928765491347, 0.3563750724061022, 0.34715629330124315, 0.3379038444448305, 0.32861862320689683, 0.31930153013597995, 0.30995346887178016, 0.3005753460575194, 0.2911680712520081, 0.2817325568414298, 0.2722697179508507, 0.2627804723554652, 0.2532657403915822, 0.24372644486736467, 0.23416351097332916, 0.22457786619261355, 0.214970440211024, 0.2053421648268658, 0.19569397386057194, 0.18602680306413374, 0.17634159003034497, 0.16663927410186744, 0.15692079628012645, 0.1471870991340458, 0.13743912670863034, 0.12767782443340545, 0.11790413903072322, 0.10811901842394182, 0.09832341164548851, 0.08851826874481623, 0.0787045406962607, 0.06888317930680748, 0.059055137123780875, 0.0492213673424555, 0.03938282371361059, 0.029540460451028643, 0.01969523213894739, 0.00984809363948123, 0.0,};

        _window = new float [_block_size];
        for (int i = 0; i < _block_size; i++)
        {
            _window[i] = window_np[i];
            //_window[i] = sqrtf( 0.5f * (1 - cosf(2 * static_cast<float>(M_PI) * i / ((float)_block_size - 1))) );
        }
        float mingain = -80.0f;
        _mingain = powf(10.0f, mingain / 20.0f);

        set_post_gain(3.0f);

    }

    void NSNet_ncnn::set_post_gain(float postgainDB)
    {
        _postgainDB = powf(10.0f, postgainDB / 20.0f);
    }

    NSNet_ncnn::~NSNet_ncnn()
    {
        if(_inblock_Real != nullptr)
        {
            fftwf_free(_inblock_Real);
            _inblock_Real = nullptr;
        }
        if(_inblock_windowed_Real != nullptr)
        {
            fftwf_free(_inblock_windowed_Real);
            _inblock_windowed_Real = nullptr;
        }
        if(_logPow != nullptr)
        {
            delete _logPow;
            _logPow = nullptr;
        }

        if(_window != nullptr)
        {
            delete [] _window;
            _window = nullptr;
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



    int NSNet_ncnn::load_model(std::vector<std::string> &modelPath)
    {
        if(modelPath.size() != 2)
        {
            std::cout<<"model laod err!"<<std::endl;
            return -1;
        }
        //TODO use encrpyted
        _net.load_param(modelPath[0].c_str());
        _net.load_model(modelPath[1].c_str());

        _h1.create(400, 1, 1);
        _h2.create(400, 1, 1);
        _h1.fill(0.0f);
        _h2.fill(0.0f);

        return 0;
    }

    int NSNet_ncnn::process_1_channel_trunk_by_trunk(const char *indata, char *outdata, size_t buffsize)
    {
        if(buffsize != trunk_size_in_bytes() )
        {
            std::cout<<"[NS] only support buffsize == trunk_size_in_bytes()!"<<std::endl;
            return -1;
        }
        //memcpy(outdata, indata, buffsize);

        S16ToFloat((short *)indata,  _trunk_size, _indata);

        //shift to left
        memmove(_inblock_Real, _inblock_Real + _block_shift, (_block_size - _block_shift) * sizeof(float));
        memcpy(_inblock_Real + (_block_size - _block_shift), _indata, _block_shift * sizeof(float));

        // add window
        for (int i = 0; i < _block_size; i++)
        {
            _inblock_windowed_Real[i] = _inblock_Real[i] * _window[i];
        }

        fftwf_execute(_fft_plan);

        LogPowerSpecFea(_inSpec_Complex, _fea_size, _logPow);

        ncnn::Mat ncnn_inFlt(_fea_size, 1, 1, _logPow);
        ncnn::Extractor ex = _net.create_extractor();
        ex.set_num_threads(1);
        ex.input("x", ncnn_inFlt);
        ex.input("h1_in", _h1);
        ex.input("h2_in", _h2);

        ncnn::Mat out;//, h1_out, h2_out;
        ex.extract("y", out);
        ex.extract("h1_out", _h1);
        ex.extract("h2_out", _h2);

        //_h1 = h1_out.clone();
        //_h2 = h2_out.clone();
        //std::cout<<"out shape: w: "<<out.w <<",h: "<<out.h <<",c: "<<out.c<<std::endl;

        const float *gainp = (float *)out.data;
        for(int i=0; i< _fea_size; i++)
        {
            float gain = gainp[i];
            gain = gain < _mingain? _mingain:  gain;

            gain = _postgainDB * gain;
            _inSpec_Complex[i][0] *= gain;
            _inSpec_Complex[i][1] *= gain;
        }

        fftwf_execute(_ifft_plan);// ifft to _outblock_Real
        float scale = 1.0f / _block_size; // FFTW ifft should div N
        // add window
        for (int i = 0; i < _block_size; i++)
        {
            _outblock_Real[i] = _outblock_Real[i] * _window[i] * scale;
        }
        //shift to left
        memmove(_outblock_Buf, _outblock_Buf + _block_shift, (_block_size - _block_shift) * sizeof(float));
        memset(_outblock_Buf + (_block_size - _block_shift), 0, _block_shift * sizeof(float));

        for (int i = 0; i < _block_size; i++)
        {
            _outblock_Buf[i] = _outblock_Buf[i] + _outblock_Real[i];
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