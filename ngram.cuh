//
// Created by HIMONO on 2021/6/11.
//

#ifndef ENUMV8_NGRAM_CUH
#define ENUMV8_NGRAM_CUH

#define COMPARE_UTF_MAX 0x188DE
//reCodeMax

#include "stdIncluded.h"
#include "utility.cuh"

#define CHINESE_MAX 0x2EBEF

namespace ngramNameSpace {

    //struct of array 排列方式
    /* example : 3-gram
     * onCPU: std::map<std::string.resize(3), int>
     * 1. 1-1 1-2 1-3 1-val
     * 2. 2-1 2-2 2-3 2-val
     * 3. 3-1 3-2 3-3 3-val
     * 4. 4-1 4-2 4-3 4-val
     * 5. 5-1 5-2 5-3 5-val
     * =>                       |                   |
     * key: 1-1 2-1 3-1 4-1 5-1 1-2 2-2 3-2 4-2 5-2 1-3 2-3 3-3 4-3 5-3
     * val: 1-val 2-val 3-val 4-val 5-val
     */
    //https://www.researchgate.net/profile/Daniel-Wilke/publication/261860648/figure/fig27/AS:667891238178831@1536249088802/AOS-vs-SOA-data-access-patterns.png
    struct resultSOA {
        thrust::universal_vector<unsigned int> result_key;
        thrust::universal_vector<unsigned int> result_value;
    };


    /*
     * Use:
     * auto myNgram = ngram(&string);
     *
     * calculate N-Gram:
     * myNgram.calculate(N, top);
     *
     * Notice:
     * If you want to calculate N, N > 1, you need to calculate N-1 first.
     *
     * output:
     *
     */
    class ngram {
    public:
        ngram(std::wstring *input);

        ~ngram();

        void calculate(int n_gram, size_t top);

        void output(std::basic_ostream<wchar_t> &outStream);

    private:

        int deviceCount;
        thrust::universal_vector<int> string;

        //字典簿
        thrust::universal_vector<unsigned int> dictBook;
        //保存按頻率排序的資料
        std::map<int, resultSOA> result_sort;
        //保存按字元排序的資料
        std::map<int, resultSOA> result_nonSort;

        void calculateOneGram(size_t top);

        void calculateTwoGram(size_t top);


    };

    namespace initStruct {

        __global__ void createDictBook(array<int> string, array<unsigned int> dict);
    }

    namespace calculate {

        __global__ void calculateTwo(array<int> string, array<int> twoGramDict, array<unsigned int> twoGramDictCache,
                                     array<unsigned int> myMap);

        __global__ void
        DenseMatrix2CSR(array<int> twoGramDict, array<unsigned int> myMap, array<unsigned int> scan, array<bool> pred,
                        array<unsigned int> key, array<unsigned int> val, unsigned int newEnd);

        __global__ void canCombineFunc(array<unsigned int> in, array<bool> out, int inGram);

        __global__ void createCombinePair(array<bool> pred, array<unsigned int> scan, array<unsigned int> in,
                                          array<unsigned int> out, int ngram);

        __global__ void
        calculateN(array<int> string, array<unsigned int> key, array<unsigned int> keyCache, array<unsigned int> val,
                   int ngram);
    }


    //thrust 1元運算子
    struct is_large_zero {
        __host__ __device__
        bool operator()(const unsigned int x) {
            return x > 0;
        }
    };

    //thrust 1元運算子
    struct isAboveTh {


        int kThreshold;

        isAboveTh(int threshold) : kThreshold{threshold} {}

        __host__ __device__ bool operator()(const unsigned int x) {
            int v = x;
            return (v > kThreshold);
        }

    };


}

#endif //ENUMV8_NGRAM_CUH
