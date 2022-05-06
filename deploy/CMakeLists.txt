project(DTLN_deploy)
cmake_minimum_required(VERSION 3.1)

include_directories(3rd/webrtc/)


set(CMAKE_CXX_STANDARD 11)

# FFTW3
set(FFTW3_LIBRARIES  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fftw3/lib/libfftw3f.a)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}//third_party/fftw3/include/)

# ncnn
set(NCNN_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/lib/libncnn.a)
include_directories(./third_party/ncnn/include/)

if(NOT  APPLE)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fopenmp")
endif()

FILE(GLOB SRCS1
        src/main_NSNet.cpp
        src/fea_utils.cpp
        src/NSNet_ncnn.cpp)

FILE(GLOB SRCS2
        src/main_DTLN.cpp
        src/fea_utils.cpp
        src/DTLN_ncnn.cpp)

add_executable(nsnet  ${SRCS1})
target_link_libraries(nsnet ${FFTW3_LIBRARIES} ${NCNN_LIBS})

add_executable(dtln  ${SRCS2})
target_link_libraries(dtln ${FFTW3_LIBRARIES} ${NCNN_LIBS})