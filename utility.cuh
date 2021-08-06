//
// Created by ascdc on 2021-05-27.
//

#ifndef ENUMV7_UTILITY_CUH
#define ENUMV7_UTILITY_CUH

#include "stdIncluded.h"

#define reCodeNotDef 0x399999

__host__ __device__ bool ifchinese(int);


__host__ __device__  unsigned int utf32ChineseReCoding(unsigned int tmp);

__host__ __device__  unsigned int Reutf32ChineseReCoding(unsigned int tmp);

void catchError();



template<typename myType>
struct array {
    myType *ptr;
    size_t size;
};

//跟系統要size個myT型別的空間(unified)
template<typename myT>
auto alloc_unified(size_t size) -> array<myT>{
    array<myT> temp;
    temp.size = size;
    cudaMallocManaged(reinterpret_cast<void **>(&(temp.ptr)), sizeof(myT) * size);
    catchError();
    return temp;
}
//跟系統要size個myT型別的空間(device)
template<typename myT>
auto alloc_device(size_t size) -> array<myT>{
    array<myT> temp;
    temp.size = size;
    cudaMalloc(reinterpret_cast<void **>(&(temp.ptr)), sizeof(myT) * size);
    catchError();
    return temp;
}
//backup
//template<class myFunc, class returnType>
//returnType benchmark(std::function <myFunc> run, std::wstring name) {
//    auto start = std::chrono::system_clock::now();
//    returnType ret = run();
//    auto end = std::chrono::system_clock::now();
//    auto period1 = std::chrono::duration_cast < std::chrono::duration < double >> (end - start);
//    wcout << L"function " << name << L" execute time : "
//          << std::chrono::duration_cast<std::chrono::microseconds>(period1).count() << L"us\n";
//    return ret;
//}

template<class myFunc>
auto benchmarkR(std::function<myFunc> run, std::wstring name) -> decltype(run()) {
    auto start = std::chrono::system_clock::now();
    auto ret = run();
    auto end = std::chrono::system_clock::now();
    auto period1 = std::chrono::duration_cast<std::chrono::duration<double >>(end - start);
    wcout << L"function " << name << L" execute time : "
          << std::chrono::duration_cast<std::chrono::microseconds>(period1).count() << L"us\n";
    return ret;
}

template<class myFunc>
void benchmark(std::function<myFunc> run, std::wstring name) {
    auto start = std::chrono::system_clock::now();
    run();
    auto end = std::chrono::system_clock::now();
    auto period1 = std::chrono::duration_cast<std::chrono::duration<double >>(end - start);
    wcout << L"function " << name << L" execute time : "
          << std::chrono::duration_cast<std::chrono::microseconds>(period1).count() << L"us\n";

}

/*
 * 使用:
不用回傳質:
benchmark<void(void)>(std::bind(你的函式, 參數一, 參數二, ..., 最多參數9), L"輸出用的標籤");

需要回傳質時:
benchmarkR<回傳型別(void)>(std::bind(你的函式, 參數一, 參數二, ..., 最多參數9), L"輸出用的標籤");

要測試的是一個class或namespace的member function, 以ngram::function這個為例:
benchmark<void(void)>(std::bind(&ngram::function, this), L"onegram");

要測試的function有多載過
benchmark<void(void)>(std::bind(static_cast<void(ngram::*)(bool)>(&ngram::calTwoGram), this, true), L"twogram");
 */

#endif //ENUMV7_UTILITY_CUH
