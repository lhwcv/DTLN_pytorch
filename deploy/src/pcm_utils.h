#include <stdio.h>
#include <stdlib.h>

class  PCMReader
{
public:
    explicit PCMReader(const char * filepath)
    {
        _file = fopen(filepath, "rb");
        if(nullptr == _file)
        {
            printf("==> open file %s error.\n", filepath);
        }

    }
    long  sample_size_in_16bit()
    {
        if(_file == nullptr)
            return 0;
        fseek(_file, 0, SEEK_END);
        long len = ftell(_file);
        len /= 2;
        rewind(_file);
        return len;
    }

    size_t read_bytes(void * to, size_t size)
    {
        if(nullptr == _file)
        {
            printf("==> file empty!");
            exit(0);
        }
        size_t  readsize = fread(to, sizeof(char), size, _file);
        if(readsize != size)
        {
            printf("read size err!\n");
            exit(0);
        }
        return readsize;
    }

    ~PCMReader()
    {
        if(nullptr != _file)
        {
            fclose(_file);
            _file = nullptr;
        }
    }

private:
    FILE  * _file = nullptr;

};

class  PCMWriter
{
public:
    explicit PCMWriter(const char * filepath)
    {
        _file = fopen(filepath, "wb");
        if(nullptr == _file)
        {
            printf("==> open file %s error.\n", filepath);
        }

    }

    void write_bytes(void * src, size_t size)
    {
        if(nullptr == _file)
        {
            printf("==> file empty!");
            exit(0);
        }
        fwrite(src, sizeof(char), size, _file);
    }

    ~PCMWriter()
    {
        if(nullptr != _file)
        {
            fclose(_file);
            _file = nullptr;
        }
    }

private:
    FILE  * _file = nullptr;

};

