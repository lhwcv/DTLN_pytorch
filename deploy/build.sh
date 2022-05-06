#!/usr/bin/env bash

if [ ! -d "./third_party/fftw3/" ];then
  echo "build fftw3 from source."
  cd ./third_party/source/
  unzip fftw-3.3.10.zip
  mkdir build_fftw3/ && cd build_fftw3/
  cmake -DCMAKE_BUILD_TYPE=Release  -DBUILD_SHARED_LIBS=OFF -DENABLE_FLOAT=ON\
        -DCMAKE_INSTALL_PREFIX=../../fftw3/ ../fftw-3.3.10/
  make -j8
  make install
  cd ../
  rm -r build_fftw3/
  rm -r fftw-3.3.10
  cd ../../

else
  echo "fftw3 exist."
fi


if [ ! -d "./third_party/ncnn/" ];then
  echo "build ncnn from source."
  cd ./third_party/source/
  unzip ncnn-20220420.zip
  mkdir build_ncnn/ && cd build_ncnn
  cmake  -DNCNN_PIXEL=ON -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_BENCHMARK=OFF\
       -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF\
       -DNCNN_SSE2=OFF -DNCNN_AVX2=OFF  -DCMAKE_BUILD_TYPE=Release\
       -DCMAKE_INSTALL_PREFIX=../../ncnn/ -DBUILD_SHARED_LIBS=OFF ../ncnn-20220420/
  make -j8
  make install
  cd ../
  rm -r build_ncnn/
  rm -r ncnn-20220420
  rm -r __MACOSX
  cd ../../

else
  echo "ncnn exist."
fi

if [ -d "./build/" ];then
  rm -r build/
fi

mkdir build && cd build/
cmake -DCMAKE_BUILD_TYPE=Release ../
make
