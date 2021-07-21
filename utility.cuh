//
// Created by ascdc on 2021-05-27.
//

#ifndef ENUMV7_UTILITY_CUH
#define ENUMV7_UTILITY_CUH

#include "stdIncluded.h"

#define reCodeNotDef 0x399999

__host__ __device__ bool ifchinese(int);

__host__ __device__ bool ifchinese2(int);

__host__ __device__  unsigned int utf32ChineseReCoding(unsigned int tmp);

__host__ __device__  unsigned int Reutf32ChineseReCoding(unsigned int tmp);

void catchError();

template<typename myType>
struct array {
    myType *ptr;
    size_t size;
};

template<typename myT>
array<myT> alloc_unified(size_t size) {
    array<myT> temp;
    temp.size = size;
    cudaMallocManaged(reinterpret_cast<void **>(&(temp.ptr)), sizeof(myT) * size);
    catchError();
    return temp;
}

template<typename myT>
array<myT> alloc_device(size_t size) {
    array<myT> temp;
    temp.size = size;
    cudaMalloc(reinterpret_cast<void **>(&(temp.ptr)), sizeof(myT) * size);
    catchError();
    return temp;
}

template<class myFunc, class returnType>
returnType benchmark(std::function <myFunc> run, std::wstring name) {
    auto start = std::chrono::system_clock::now();
    returnType ret = reinterpret_cast<returnType>(run());
    auto end = std::chrono::system_clock::now();
    auto period1 = std::chrono::duration_cast < std::chrono::duration < double >> (end - start);
    wcout << L"function " << name << L" execute time : "
          << std::chrono::duration_cast<std::chrono::microseconds>(period1).count() << L"us\n";
    return ret;
}

template<class myFunc>
void benchmark(std::function <myFunc> run, std::wstring name) {
    auto start = std::chrono::system_clock::now();
    run();
    auto end = std::chrono::system_clock::now();
    auto period1 = std::chrono::duration_cast < std::chrono::duration < double >> (end - start);
    wcout << L"function " << name << L" execute time : "
          << std::chrono::duration_cast<std::chrono::microseconds>(period1).count() << L"us\n";

}

#endif //ENUMV7_UTILITY_CUH
