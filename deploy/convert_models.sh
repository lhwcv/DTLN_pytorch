#!/usr/bin/env bash

if [ ! -d "./third_party/ncnn/bin/" ];then
  echo "build ncnn from source."
  cd ./third_party/source/
  unzip ncnn-20220420.zip
  mkdir build_ncnn/ && cd build_ncnn
  cmake  -DNCNN_PIXEL=ON -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_BENCHMARK=OFF\
       -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_EXAMPLES=OFF\
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


export PYTHONPATH=../
python ../convert_to_onnx.py  --model_path ../pretrained/model.pth \
     --model_1 ./model_p1.onnx  --model_2 ./model_p2.onnx

python -m onnxsim model_p1.onnx  model_p1.onnx

./third_party/ncnn/bin/onnx2ncnn  model_p1.onnx  ./models/dtln/model_p1.param ./models/dtln/model_p1.bin

./third_party/ncnn/bin/onnx2ncnn  model_p2.onnx  ./models/dtln/model_p2.param ./models/dtln/model_p2.bin

