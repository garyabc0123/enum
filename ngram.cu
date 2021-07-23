//
// Created by HIMONO on 2021/6/11.
//

#include "ngram.cuh"

ngramNameSpace::n_gram_struct ngramNameSpace::alloc_n_gram_stuct(unsigned short gram, size_t word_count) {
    n_gram_struct temp;
    temp.gram = gram;
    temp.word_count = word_count;
    cudaMallocManaged(reinterpret_cast<void **>(&(temp.word)), sizeof(int) * word_count * gram);
    cudaMallocManaged(reinterpret_cast<void **>(&(temp.count)), sizeof(size_t) * word_count);
    cudaMallocManaged(reinterpret_cast<void **>(&(temp.enable)), sizeof(bool) * word_count);
    return temp;

}

void ngramNameSpace::free_n_gram_struct(n_gram_struct input) {
    cudaFree(input.count);
    cudaFree(input.word);
    cudaFree(input.enable);
}

ngramNameSpace::ngram::ngram(std::wstring *input, size_t top_pa) {
    auto func_load_to_gpu = [](std::wstring *input) -> array<int> {

        size_t end = input->size() - 1;
        array<int> ret;

        //GCC-11以下對於unicode編碼有長度bug...，要自己重新計算
        for (; input->operator[](end) == input->operator[](input->size() - 1); end--) {
            if (end == 0)
                break;
        }
        cudaMallocManaged(reinterpret_cast<void **>(&(ret.ptr)), sizeof(int) * end);
        catchError();
        for (size_t it = 0; it < end; it++) {
            ret.ptr[it] = input->operator[](it);
        }
        ret.size = end;
        return ret;
    };
    auto func_load_to_texture = [=](std::wstring *input){
        size_t end = input->size() - 1;
        //GCC-11以下對於unicode編碼有長度bug...，要自己重新計算
        for (; input->operator[](end) == input->operator[](input->size() - 1); end--) {
            if (end == 0)
                break;
        }

        string_tex.size = end;

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();

        cudaMalloc(reinterpret_cast<void **>(&(string_tex.devPtr)), sizeof(int) * end);
        catchError();
        cudaMemcpy(string_tex.devPtr, input->c_str(), sizeof(int) * end, cudaMemcpyHostToDevice);
        catchError();
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));

        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = string_tex.devPtr;
        resDesc.res.linear.desc = channelDesc;
        resDesc.res.linear.sizeInBytes = sizeof(int) * end;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaCreateTextureObject(&(string_tex.texObj), &resDesc, &texDesc, NULL);
        catchError();

    };

    auto func_cal_one_gram = [=]() -> array<size_t> {
        auto ret = alloc_unified<size_t>(0x2EBF0);
        if(string_tex.texObj){
            __init__::cal_one_gram::cal_one_gram_kernel_texture<<<string_tex.size / 1024 + 1, 1024>>>(string_tex, ret);
        }else{
            __init__::cal_one_gram::cal_one_gram_kernel<<<string.size / 1024 + 1, 1024>>>(string, ret);
        }
        cudaDeviceSynchronize();
        catchError();
        return ret;
    };

    auto func_convert_to_standard_struct = [=](array<size_t> onegram_temp) -> n_gram_struct {
        n_gram_struct onegram = alloc_n_gram_stuct(1, onegram_temp.size);
        n_gram_struct ret;
        cudaFree(onegram.count);
        onegram.count = onegram_temp.ptr;
        __init__::convert_to_standard_struct::write_word<<<onegram.word_count / 1024 + 1, 1024>>>(onegram);
        cudaDeviceSynchronize();
        catchError();
        ret = merge_n_gram_struct_by_top(onegram, top_pa);
        free_n_gram_struct(onegram);
        return ret;
    };
    auto debug = [=]() {
        for (auto i = 0; i < n_gram_map[1].word_count; i++) {
            if (n_gram_map[1].count[i] != 0) {
                wcout << static_cast<wchar_t>(n_gram_map[1].word[i]) << " " << n_gram_map[1].count[i] << endl;
            }
        }
    };

    this->top_pa = top_pa;
    //this->string = func_load_to_gpu(input);

    func_load_to_texture(input);
    auto onegram_temp = func_cal_one_gram();
    n_gram_map[1] = func_convert_to_standard_struct(onegram_temp);


    debug();
}

ngramNameSpace::ngram::~ngram() {
    if(string.ptr != nullptr && string.size != 0)
        cudaFree(string.ptr);
    if(string_tex.cuArray)
        cudaFreeArray(string_tex.cuArray);
    if(string_tex.devPtr)
        cudaFree(string_tex.devPtr);
    if(string_tex.texObj)
        cudaDestroyTextureObject(string_tex.texObj);
    for (auto x : n_gram_map) {
        free_n_gram_struct(x.second);
    }
}


void ngramNameSpace::ngram::calculate(int n_gram, size_t top) {
    auto func_write_words = [](n_gram_struct my_gram, n_gram_struct prev_gram, n_gram_struct one_gram) {
        calculate::write_word<<<my_gram.word_count / 1024 + 1, 1024>>>(my_gram, prev_gram, one_gram);
        cudaDeviceSynchronize();
        catchError();
        auto debug = [](n_gram_struct my_gram) {
            for (auto i = 0; i < my_gram.word_count; i++) {
                for (auto j = 0; j < my_gram.gram; j++)
                    wcout << static_cast<wchar_t>(my_gram.word[i * my_gram.gram + j]);
                wcout << endl;
            }

        };
        //debug(my_gram);
    };
    auto func_add_on_table = [=](n_gram_struct my_gram, array<int> string, array_in_texture string_tex) {
        auto debug = [=](n_gram_struct my_gram) {
            for (auto i = 0; i < my_gram.word_count; i++) {
                for (auto k = 0; k < my_gram.gram; k++)
                    wcout << static_cast<wchar_t>(my_gram.word[i * my_gram.gram + k]);
                wcout << " " << my_gram.count[i];
                wcout << endl;
            }
        };

        if(string_tex.texObj){
            calculate::add_on_table_texture<<<string_tex.size / 512 + 1, 512>>>(my_gram, string_tex);
        }else{
            calculate::add_on_table<<<string.size / 1024 + 1, 1024>>>(my_gram, string);
        }

        cudaDeviceSynchronize();

//        for(auto i = 0 ; i < string.size ; i++){
//            kernel(my_gram, string, i);
//        }
        catchError();

        //debug(my_gram);
    };


    if (n_gram < 2 || n_gram > 200 || n_gram_map.find(n_gram - 1) == n_gram_map.end())
        return;

    auto one_gram = n_gram_map[1];
    auto prev_gram = n_gram_map[n_gram - 1];
    auto my_gram = alloc_n_gram_stuct(n_gram, one_gram.word_count * prev_gram.word_count);

    benchmark<void(void)>(std::bind(func_write_words, my_gram, prev_gram, one_gram), L"func_write_words");
    //func_write_words(my_gram, prev_gram, one_gram);


    benchmark<void(void)>(std::bind(func_add_on_table, my_gram, string, string_tex), L"func_add_on_table");
    //func_add_on_table(my_gram, string, string_tex);


    n_gram_map[n_gram] = benchmark<n_gram_struct(void), n_gram_struct>(std::bind(&ngram::merge_n_gram_struct_by_top, this, my_gram, top), L"merge_n_gram");
    //n_gram_map[n_gram] = merge_n_gram_struct_by_top(my_gram, top);

    free_n_gram_struct(my_gram);


}

void ngramNameSpace::ngram::output() {
    for (auto it : n_gram_map) {
        wcout << "start print " << it.second.gram << endl;
        wcout << "-------------------------------------------------------------------\n";
        wcout << "size : " << it.second.word_count << endl;
        for (auto i = 0; i < it.second.word_count; i++) {
            for (auto j = 0; j < it.second.gram; j++) {
                wcout << static_cast<wchar_t>(it.second.word[i * it.second.gram + j]);
            }
            wcout << " count: ";
            wcout << it.second.count[i] << endl;
        }
    }
}


ngramNameSpace::n_gram_struct ngramNameSpace::ngram::merge_n_gram_struct_by_top(n_gram_struct input, size_t top) {
    array<size_t> sort_ans = alloc_unified<size_t>(input.word_count);
    memcpy(sort_ans.ptr, input.count, input.word_count);
    thrust::sort(thrust::device, sort_ans.ptr, &(sort_ans.ptr[sort_ans.size - 1]), thrust::greater<size_t>());
    size_t threshold = 1;
    if (top > input.word_count || top <= 0)
        threshold = 1;
    else
        threshold = sort_ans.ptr[top];
    if (threshold == 0)
        threshold = 1;
    cudaFree(sort_ans.ptr);
    return merge_n_gram_struct_by_threshold(input, threshold);

}

ngramNameSpace::n_gram_struct ngramNameSpace::ngram::merge_n_gram_struct_by_threshold(n_gram_struct input,
                                                                                      size_t threshold) {
    array<bool> pred = alloc_unified<bool>(input.word_count);
    array<size_t> scan = alloc_unified<size_t>(input.word_count);

    merge_n_gram_struct::pred<<<input.word_count / 1024 + 1, 1024>>>(input, pred, threshold);
    cudaDeviceSynchronize();
    catchError();
    thrust::exclusive_scan(thrust::device, pred.ptr, &(pred.ptr[pred.size]), scan.ptr, 0);
    auto new_size = scan.ptr[scan.size - 1];
    auto output = alloc_n_gram_stuct(input.gram, new_size);
    merge_n_gram_struct::pack<<<input.word_count / 1024 + 1, 1024>>>(input, pred, scan, output);
    cudaDeviceSynchronize();
    catchError();
    cudaFree(pred.ptr);
    cudaFree(scan.ptr);
    return output;

}

__global__ void ngramNameSpace::__init__::cal_one_gram::cal_one_gram_kernel(array<int> string, array<size_t> onegram) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= string.size) return;
    auto prep = string.ptr[idx];
    if (ifchinese(prep))
        atomicAdd(reinterpret_cast<int *>(&(onegram.ptr[prep])), (int) 1);
    return;
}

__global__ void ngramNameSpace::__init__::cal_one_gram::cal_one_gram_kernel_texture(array_in_texture string,
                                                                                    array<size_t> onegram) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= string.size) return;
    auto prep = tex1Dfetch<int>(string.texObj, idx);

    if (ifchinese(prep))
        atomicAdd(reinterpret_cast<int *>(&(onegram.ptr[prep])), (int) 1);
    return;
}

__global__ void ngramNameSpace::__init__::convert_to_standard_struct::write_word(n_gram_struct input) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= input.word_count) return;
    input.word[idx] = idx;
}

__global__ void ngramNameSpace::merge_n_gram_struct::pred(n_gram_struct input, array<bool> pred,
                                                          size_t threshold) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= input.word_count)
        return;
    if (input.count[idx] >= threshold)
        pred.ptr[idx] = true;
    else if (input.count[idx] < threshold)
        pred.ptr[idx] = false;

}

__global__ void ngramNameSpace::merge_n_gram_struct::pack(n_gram_struct input, array<bool> pred, array<size_t> scan,
                                                          n_gram_struct output) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= input.word_count)
        return;
    if (!pred.ptr[idx])
        return;
    auto new_addr = scan.ptr[idx];
    output.enable[new_addr] = input.enable[idx];
    output.count[new_addr] = input.count[idx];
    for (auto i = 0; i < input.gram; i++) {
        output.word[input.gram * new_addr + i] = input.word[input.gram * idx + i];
    }


}

__global__ void ngramNameSpace::calculate::write_word(n_gram_struct my_gram, n_gram_struct prev_gram,
                                                      n_gram_struct one_gram) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= my_gram.word_count)return;
    size_t prev_id = idx / one_gram.word_count;
    size_t one_id = idx % one_gram.word_count;

    for (auto i = 0; i < prev_gram.gram; i++) {
        my_gram.word[idx * my_gram.gram + i] = prev_gram.word[prev_id * prev_gram.gram + i];
    }
    my_gram.word[idx * my_gram.gram + prev_gram.gram] = one_gram.word[one_id];

}

__global__ void ngramNameSpace::calculate::add_on_table(n_gram_struct my_gram, array<int> string) {
    auto equal = [=] __device__(size_t a, size_t b, n_gram_struct my_gram, array<int> string) {
        for (auto i = 0; i < my_gram.gram; i++) {
            if (my_gram.word[a * my_gram.gram + i] != string.ptr[b + i])
                return false;
        }
        return true;
    };
    auto a_large_than_b = [=] __device__(size_t a, size_t b, n_gram_struct my_gram, array<int> string) {
        for (auto i = 0; i < my_gram.gram; i++) {
            if (my_gram.word[a * my_gram.gram + i] < string.ptr[b + i]) {
                return false;
            }
            if (my_gram.word[a * my_gram.gram + i] > string.ptr[b + i]) {
                return true;
            }
        }
        return true;
    };
    auto a_small_than_b = [=] __device__(size_t a, size_t b, n_gram_struct my_gram, array<int> string) {
        for (auto i = 0; i < my_gram.gram; i++) {
            if (my_gram.word[a * my_gram.gram + i] > string.ptr[b + i]) {
                return false;
            }
            if (my_gram.word[a * my_gram.gram + i] < string.ptr[b + i]) {
                return true;
            }
        }
        return true;
    };


    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= string.size)
        return;

    size_t begin, end, middle;
    begin = 0;
    end = my_gram.word_count - 1;

    while (end > begin) {
        if (end - begin <= 1) {
            if (equal(end, idx, my_gram, string)) {
                atomicAdd(reinterpret_cast<unsigned int *>(&(my_gram.count[end])), 1);
            } else if (equal(begin, idx, my_gram, string)) {
                atomicAdd(reinterpret_cast<unsigned int *>(&(my_gram.count[begin])), 1);
            }
            return;
        }
        middle = (begin + end) / 2;
        if (equal(middle, idx, my_gram, string)) {
            atomicAdd(reinterpret_cast<unsigned int *>(&(my_gram.count[middle])), 1);
            return;
        } else if (a_large_than_b(middle, idx, my_gram, string)) {
            end = middle;
        }
//        else if(a_small_than_b(middle, idx, my_gram, string)){
//            begin = middle;
//        }else{
//
//            //printf("%d :%d\n",idx, middle);
//            assert(false) ;
//        }
            //speed!
        else {
            begin = middle;
        }
    }

}

__global__ void ngramNameSpace::calculate::add_on_table_texture(n_gram_struct my_gram, array_in_texture string) {
    auto equal = [=] __device__(size_t a, size_t b, n_gram_struct my_gram, array_in_texture string) {
        for (auto i = 0; i < my_gram.gram; i++) {
            if (my_gram.word[a * my_gram.gram + i] != tex1Dfetch<int>(string.texObj, b + i))
                return false;
        }
        return true;
    };
    auto a_large_than_b = [=] __device__(size_t a, size_t b, n_gram_struct my_gram, array_in_texture string) {
        for (auto i = 0; i < my_gram.gram; i++) {
            if (my_gram.word[a * my_gram.gram + i] < tex1Dfetch<int>(string.texObj, b + i)) {
                return false;
            }
            if (my_gram.word[a * my_gram.gram + i] > tex1Dfetch<int>(string.texObj, b + i)) {
                return true;
            }
        }
        return true;
    };
    auto a_small_than_b = [=] __device__(size_t a, size_t b, n_gram_struct my_gram, array_in_texture string) {
        for (auto i = 0; i < my_gram.gram; i++) {
            if (my_gram.word[a * my_gram.gram + i] > tex1Dfetch<int>(string.texObj, b + i)) {
                return false;
            }
            if (my_gram.word[a * my_gram.gram + i] < tex1Dfetch<int>(string.texObj, b + i)) {
                return true;
            }
        }
        return true;
    };


    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= string.size)
        return;

    size_t begin, end, middle;
    begin = 0;
    end = my_gram.word_count - 1;

    while (end > begin) {
        if (end - begin <= 1) {
            if (equal(end, idx, my_gram, string)) {
                atomicAdd(reinterpret_cast<unsigned int *>(&(my_gram.count[end])), 1);
            } else if (equal(begin, idx, my_gram, string)) {
                atomicAdd(reinterpret_cast<unsigned int *>(&(my_gram.count[begin])), 1);
            }
            return;
        }
        middle = (begin + end) / 2;
        if (equal(middle, idx, my_gram, string)) {
            atomicAdd(reinterpret_cast<unsigned int *>(&(my_gram.count[middle])), 1);
            return;
        } else if (a_large_than_b(middle, idx, my_gram, string)) {
            end = middle;
        }
//        else if(a_small_than_b(middle, idx, my_gram, string)){
//            begin = middle;
//        }else{
//
//            //printf("%d :%d\n",idx, middle);
//            assert(false) ;
//        }
            //speed!
        else {
            begin = middle;
        }
    }
}