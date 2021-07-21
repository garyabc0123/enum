cmake_minimum_required(VERSION 3.19)
#Linux
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
#windows
#set(CMAKE_CUDA_COMPILER C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin/nvcc.exe)
project(enum CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
#Linux
include_directories(/usr/inclue/c++/9)

set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 75 80)
set(CMAKE_VERBOSE_MAKEFILE ON)

#Linux
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -lineinfo --compiler-bindir /usr/bin/g++-9 --extended-lambda ")
set(CMAKE_CUDA_FLAGS_DEBUG " --compiler-bindir /usr/bin/g++-9 --extended-lambda ")
#Windows
#set(CMAKE_CUDA_FLAGS_DEBUG "--compiler-bindir  --extended-lambda")
add_definitions(-DUNICODE -D_UNICODE)
find_package(Boost COMPONENTS thread REQUIRED)


add_executable(enum main.cu stdIncluded.h initial_sys.cu initial_sys.cuh utility.cuh utility.cu myThreadPool.cu myThreadPool.cuh ngram.cu ngram.cuh)

target_link_libraries(enum pthread)
target_link_libraries(enum Boost::thread)

set_target_properties(
        enum
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)