//
// Created by HIMONO on 2021/6/11.
//

#ifndef ENUMV8_NGRAM_CUH
#define ENUMV8_NGRAM_CUH

#define COMPARE_UTF_MAX 0x188DE
//reCodeMax

#include "myThreadPool.cuh"
#include "stdIncluded.h"
#include "utility.cuh"

namespace ngramNameSpace {

    struct array_in_texture{
        cudaTextureObject_t texObj;
        cudaArray_t cuArray;
        int * devPtr;
        size_t size;

    };

    struct n_gram_struct {
        int *word;
        size_t *count;
        bool *enable;
        size_t word_count;
        unsigned short gram;
    };

    n_gram_struct alloc_n_gram_stuct(unsigned short gram, size_t word_count);

    void free_n_gram_struct(n_gram_struct);


    class ngram {
    public:
        ngram(std::wstring *input, size_t top);

        ~ngram();

        void calculate(int n_gram, size_t top);

        void output();

    private:
        //不會順便重新編碼





        array<int> string;
        array_in_texture string_tex;
        std::map<unsigned short, n_gram_struct> n_gram_map;
        size_t top_pa;

        n_gram_struct merge_n_gram_struct_by_top(n_gram_struct input, size_t top);

        n_gram_struct merge_n_gram_struct_by_threshold(n_gram_struct input, size_t threshold);


    };

    namespace __init__ {
        namespace cal_one_gram {
            __global__ void cal_one_gram_kernel(array<int> string, array<size_t> onegram);
            __global__ void cal_one_gram_kernel_texture(array_in_texture string, array<size_t> onegram);
        }
        namespace convert_to_standard_struct {
            __global__ void write_word(n_gram_struct input);
        }
    }

    namespace merge_n_gram_struct {
        __global__ void pred(n_gram_struct input, array<bool> pred, size_t threshold);

        __global__ void pack(n_gram_struct input, array<bool> pred, array<size_t> scan, n_gram_struct output);
    }

    namespace calculate {
        __global__ void write_word(n_gram_struct my_gram, n_gram_struct prev_gram, n_gram_struct one_gram);

        __global__ void add_on_table(n_gram_struct my_gram, array<int> string);
        __global__ void add_on_table_texture(n_gram_struct my_gram, array_in_texture string);

    }


}

#endif //ENUMV8_NGRAM_CUH
