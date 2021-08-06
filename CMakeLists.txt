cmake_minimum_required(VERSION 3.19)
set(CMAKE_CUDA_COMPILER ${LD_LIBRARY_PATH})
project(enum CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)

#Linux
include_directories(/usr/inclue/c++/9)

set(CMAKE_CUDA_ARCHITECTURES 52)
set(CMAKE_VERBOSE_MAKEFILE ON)


#set(CMAKE_BUILD_TYPE Debug)
#LINUX
#set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -lineinfo --compiler-bindir /usr/bin/g++-9 --extended-lambda ")
set(CMAKE_CUDA_FLAGS_DEBUG " --compiler-bindir /usr/bin/g++-9 --extended-lambda ")
#WINDOWS
#set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -lineinfo  --extended-lambda ")
add_definitions(-DUNICODE -D_UNICODE)
#find_package(Boost COMPONENTS thread REQUIRED)


add_executable(enum main.cu stdIncluded.h initial_sys.cu initial_sys.cuh utility.cuh utility.cu ngram.cu ngram.cuh)

#target_link_libraries(enum pthread)
#target_link_libraries(enum Boost::thread)

set_target_properties(
        enum
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)