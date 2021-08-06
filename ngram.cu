//
// Created by HIMONO on 2021/6/11.
//

#include "ngram.cuh"



//初始化class
//並將string讀入universal_vector
ngramNameSpace::ngram::ngram(std::wstring *input) {
    auto func_load_to_gpu = [](std::wstring *input) -> thrust::universal_vector<int> {

        size_t end = input->size() - 1;
        //GCC-11以下對於unicode編碼有長度bug...，要自己重新計算
        for (; input->operator[](end) == input->operator[](input->size() - 1); end--) {
            if (end == 0)
                break;
        }
        return thrust::universal_vector<int>(input->begin(), input->begin() + end);


    };
    //創建字典簿
    /*
     *  dict[0] = U+0 次數
     *  dict[1] = U+1 次數
     *  dict[2] = U+2 次數
     *  dict[3] = U+3 次數
     *  dict[4] = U+4 次數
     *  dict[5] = U+5 次數
     *  dict[6] = U+6 次數
     */
    auto func_create_dict = [](thrust::universal_vector<int> input) -> thrust::universal_vector<unsigned int> {
        thrust::universal_vector<unsigned int> myDict(CHINESE_MAX, 0);

        //package
        array<int> str;
        str.size = input.size();
        str.ptr = thrust::raw_pointer_cast(input.data());
        array<unsigned int> dict;
        dict.ptr = thrust::raw_pointer_cast(myDict.data());
        dict.size = myDict.size();


        initStruct::createDictBook<<<str.size / 1024 + 1, 1024>>>(str, dict);
        cudaDeviceSynchronize();
        catchError();
        return myDict;
    };


    auto debug = [=]() {
        auto it_dict = dictBook.begin();
        auto it_result_nonSort_key = result_nonSort[1].result_key.begin();
        auto it_result_nonSort_val = result_nonSort[1].result_value.begin();
        for (; it_dict != dictBook.end(); it_dict++) {
            wchar_t chinese = static_cast<wchar_t >(thrust::distance(dictBook.begin(), it_dict));
            wcout << chinese << ": " << *it_dict << " | ";
            if (*it_result_nonSort_key == chinese) {
                wcout << ": " << *it_result_nonSort_val << endl;
                it_result_nonSort_key++;
                it_result_nonSort_val++;
            } else {
                wcout << endl;
            }
        }

    };


    cudaGetDeviceCount(&deviceCount);
    string = func_load_to_gpu(input);
    dictBook = func_create_dict(string);


    //debug();


}

//Free memory
ngramNameSpace::ngram::~ngram() {

}


void ngramNameSpace::ngram::calculate(int n_gram, size_t top) {
    if (n_gram == 1) {

        //1-gram 特別解
        calculateOneGram(top);
        return;
    } else if (n_gram == 2) {

        //2-gram 特別解
        calculateTwoGram(top);
        return;
    }

    //一般解


    auto func_create_my_dict_and_cache = [](size_t top, size_t threshold, int ngram, resultSOA prevGramSort,
            thrust::universal_vector<unsigned int> dictBook) ->  std::tuple<thrust::universal_vector<unsigned int>, thrust::universal_vector<unsigned int>>  {

        //存放： 將兩兩組合在一起的可能性
        /*
         * 二十 十三 -> True
         * 二十 三十 -> false
         *
         * |     A  | B  | C  | D  |
         * |   A|
         * |   B|
         * |   C|
         * |   D|
         *
         * =>
         * |AA|AB|AC|AD|BA|BB| ... (top*2)^2
         */
        thrust::universal_vector<bool> combinePairPred(top * top * 4);

        //package
        array<unsigned int> prevGramSortKeyArr = {thrust::raw_pointer_cast(prevGramSort.result_key.data()),
                                                  prevGramSort.result_key.size()};
        array<bool> combinePairPredArr = {thrust::raw_pointer_cast(combinePairPred.data()), combinePairPred.size()};

        calculate::canCombineFunc<<<combinePairPredArr.size / 1024 + 1, 1024>>>(prevGramSortKeyArr, combinePairPredArr,
            ngram - 1);
        cudaDeviceSynchronize();
        catchError();

        //debug
        [=]() {
            for (auto x = 0; x < prevGramSort.result_value.size(); x++) {
                for (auto y = 0; y < prevGramSort.result_value.size(); y++) {
                    for (auto i = 0; i < ngram - 1; i++) {
                        std::wcout << static_cast<wchar_t >(prevGramSort.result_key[x + i *
                                                                                        prevGramSort.result_value.size()]);
                    }
                    std::wcout << " and ";
                    for (auto i = 0; i < ngram - 1; i++) {
                        std::wcout << static_cast<wchar_t >(prevGramSort.result_key[y + i *
                                                                                        prevGramSort.result_value.size()]);
                    }
                    std::wcout << ": ";
                    std::wcout << combinePairPred[x * prevGramSort.result_value.size() + y];
                    std::wcout << endl;

                }
            }
        };//();

        //compact - scan
        thrust::universal_vector<unsigned int> combinePairScan(combinePairPred.size());
        thrust::exclusive_scan(thrust::device, combinePairPred.begin(), combinePairPred.end(), combinePairScan.begin(),
                               0);

        //組合表格 and b-tree cache
        thrust::universal_vector<unsigned int> combinePairCache(CHINESE_MAX, -1);
        thrust::universal_vector<unsigned int> combinePair(combinePairScan[combinePairScan.size() - 1] * ngram);

        //package
        array<unsigned int> combinePairScanArr = {thrust::raw_pointer_cast(combinePairScan.data()), combinePairScan.size()};
        array<unsigned int> combinePairArr = {thrust::raw_pointer_cast(combinePair.data()), combinePair.size()};

        calculate::createCombinePair<<<combinePairScanArr.size / 1024 + 1, 1024>>>(combinePairPredArr, combinePairScanArr, prevGramSortKeyArr, combinePairArr, ngram);
        cudaDeviceSynchronize();
        catchError();
        /*
        for(auto idx = 0 ; idx < combinePairScanArr.size ; idx++){
            [](array<bool> pred, array<unsigned int> scan, array<unsigned int> in,
                    array<unsigned int> out, int ngram, size_t idx) {
                size_t predShape = static_cast<size_t >(sqrt(static_cast<float>(pred.size)));
                size_t x = idx / predShape;
                size_t y = idx % predShape;
                if (!pred.ptr[idx])
                    return;
                size_t newAddr = scan.ptr[idx];
                size_t newSize = out.size / ngram;
                size_t oldSize = in.size / (ngram - 1);
                for (auto i = 0; i < ngram - 1; i++) {
                    out.ptr[newAddr + i * newSize] = in.ptr[x + i * oldSize];
                }
                out.ptr[newAddr + (ngram - 1) * newSize] = in.ptr[y + (ngram - 2) * oldSize];
            }(combinePairPredArr, combinePairScanArr, prevGramSortKeyArr, combinePairArr, ngram, idx);
        }*/

        //debug
        /*
        for(auto i = 0 ; i < combinePair.size() / ngram ; i++){
            for(auto j = 0 ; j < ngram ; j++){
                wcout << static_cast<wchar_t >(combinePair[i + j * combinePair.size() / ngram]);
            }
            wcout << endl;


        }
         */


        //TODO func 把按頻率排序轉成按字元排序
        //CPU version
        auto sortByUnicodeID = [](thrust::universal_vector<unsigned int> combinePair, int ngram){
            size_t pairSize = combinePair.size() / ngram;
            std::vector<std::vector<unsigned int>> TempVec;
            for(auto it = 0 ; it < pairSize ; it++){
                std::vector<unsigned int> TTempVec;
                for(auto itt = 0 ; itt < ngram ; itt ++){
                    TTempVec.push_back(combinePair[it + itt * pairSize]);
                }
                TempVec.push_back(TTempVec);
            }
            std::sort(TempVec.begin(), TempVec.end());

            thrust::universal_vector<unsigned int> ret(combinePair.size());
            for(auto it = 0 ; it < TempVec.size() ; it++){
                for(auto itt = 0 ; itt < ngram ; itt++){
                    ret[it + itt * pairSize] = TempVec[it][itt];
                }
            }


            return ret;
        };

        combinePair = sortByUnicodeID(combinePair, ngram);


        //create cache
        for(auto it = 0; it < combinePair.size() / ngram ; it++){
            if(combinePairCache[combinePair[it]] == -1){
                combinePairCache[combinePair[it]] = it;
            }
        }

        return {combinePair, combinePairCache};


    };\

    auto func_sort = [](resultSOA in, int ngram) -> resultSOA{
        auto outVal(in.result_value);
        auto outKey(in.result_key);
        for(auto i = 0 ; i < ngram ; i++){
            thrust::stable_sort_by_key(thrust::device, outVal.begin(), outVal.end(), outKey.begin() + i * in.result_value.size(), thrust::greater<unsigned int>());
            outVal = in.result_value;
        }
        thrust::stable_sort(thrust::device, outVal.begin(), outVal.end(), thrust::greater<unsigned int>());
        return {outKey, outVal};
    };
    unsigned int threshold = 1;

    //檢查是否超越邊界
    if (top > result_sort[n_gram - 1].result_value.size()) {
        top = result_sort[n_gram - 1].result_value.size() - 1;
    }
    threshold = result_sort[n_gram - 1].result_value[top];

    thrust::universal_vector<unsigned int> key, keyCache, val;
    auto tempTup = func_create_my_dict_and_cache(top, threshold, n_gram, result_sort[n_gram - 1], dictBook);
    key = std::get<0>(tempTup);
    keyCache = std::get<1>(tempTup);
    val.resize(key.size() / n_gram);

    array<int> stringArr = {thrust::raw_pointer_cast(string.data()), string.size()};
    array<unsigned int> keyArr = {thrust::raw_pointer_cast(key.data()), key.size()};
    array<unsigned int> keyCacheArr = {thrust::raw_pointer_cast(keyCache.data()), key.size()};
    array<unsigned int> valArr = {thrust::raw_pointer_cast(val.data()), val.size()};


    calculate::calculateN<<<stringArr.size / 1024 + 1, 1024>>>(stringArr, keyArr, keyCacheArr, valArr, n_gram);
    cudaDeviceSynchronize();
    catchError();



    //debug
    //simulate device
    /*for(auto idx = 0 ; idx < stringArr.size ; idx++){

        [](array<int> string, array<unsigned int>key, array<unsigned int>keyCache, array<unsigned int> val, int ngram, size_t idx){

            auto atomicAdd = [](auto addr, int n){
                *addr+=1;
            };
            //TODO
            if(idx + ngram>= string.size)
                return;



            unsigned int begin, end, mid;
            unsigned int keyShape = key.size / ngram;

            if(!ifchinese(string.ptr[idx]))
                return;
            begin = keyCache.ptr[string.ptr[idx]];
            if(begin == -1){
                return;
            }
            if(string.ptr[idx] + 1 != keyCache.size){
                end = keyCache.ptr[string.ptr[idx] + 1];
                if(end == -1){
                    end = val.size - 1;
                }
            }else{
                end = val.size - 1;
            }

            //itS: string iterator, itK: key Iterator
            auto equal = [=](unsigned int itS, unsigned int itK) -> bool{
                for(auto i = 0 ; i < ngram ; i++){
                    if(string.ptr[itS + i] != key.ptr[itK + i * keyShape]){
                        return false;
                    }
                }
                return true;
            };
            auto SLargeThanK = [=](unsigned int itS, unsigned int itK) -> bool{
                for(auto i = 0 ; i < ngram ; i++){
                    if(string.ptr[itS + i] > key.ptr[itK + i * keyShape]){
                        return true;
                    }else if(string.ptr[itS + i] < key.ptr[itK + i * keyShape]){
                        return false;
                    }
                }
                return false;
            };
            auto SSmallThanK = [=](unsigned int itS, unsigned int itK) -> bool{
                for(auto i = 0 ; i < ngram ; i++){
                    if(string.ptr[itS + i] < key.ptr[itK + i * keyShape]){
                        return true;
                    }else if(string.ptr[itS + i] > key.ptr[itK + i * keyShape]){
                        return false;
                    }
                }
                return false;
            };


            for(;begin < end;){
                if(end - begin == 1){
                    if(equal(idx, end)){
                        atomicAdd(&(val.ptr[end]), 1);
                    }else if(equal(idx, begin)){
                        atomicAdd(&(val.ptr[begin]), 1);
                    }
                    return;
                }
                mid = (begin + end) / 2;
                if(equal(idx, mid)){
                    atomicAdd(&(val.ptr[mid]), 1);
                    return;
                }
                if(SLargeThanK(idx, mid)){
                    begin = mid;
                }else{
                    end = mid;
                }




            }


        }(stringArr, keyArr, keyCacheArr, valArr, n_gram, idx);
    }*/



    [=](){
        for(auto it = 0 ; it < val.size() ; it ++){
            for(auto itt = 0 ; itt < n_gram ; itt++){
                wcout << static_cast<wchar_t >(key[it + itt * val.size()]);
            }
            wcout << ": ";
            wcout << val[it];
            wcout << endl;
        }
    };

    result_nonSort[n_gram] = {key, val};

    result_sort[n_gram] = func_sort(result_nonSort[n_gram], n_gram);

}

void ngramNameSpace::ngram::calculateOneGram(size_t top) {

    //將字典簿轉成resultSOA
    auto func_convert_dict_to_result = [](thrust::universal_vector<unsigned int> input) -> resultSOA {
        thrust::universal_vector<unsigned int> ans_key, temp_key;
        thrust::universal_vector<unsigned int> ans_val;
        temp_key.resize(input.size());
        ans_val.resize(input.size());
        ans_key.resize(input.size());

        thrust::sequence(temp_key.begin(), temp_key.end(), 0, 1);

        //delete element which element in dictBook is 0
        auto ans_val_size =
                thrust::copy_if(thrust::device, input.begin(), input.end(), ans_val.begin(), is_large_zero()) -
                ans_val.begin();

        auto ans_key_size =
                thrust::copy_if(thrust::device, temp_key.begin(), temp_key.end(), input.begin(), ans_key.begin(),
                                is_large_zero()) - ans_key.begin();
        ans_val.resize(ans_val_size);
        ans_key.resize(ans_key_size);
        return {ans_key, ans_val};
    };

    //sort by freq
    auto func_sort = [](resultSOA input) -> resultSOA {
        thrust::universal_vector<unsigned int> ans_key(input.result_key);
        thrust::universal_vector<unsigned int> ans_value(input.result_value);

        thrust::sort_by_key(thrust::device, ans_value.begin(), ans_value.end(), ans_key.begin(),
                            thrust::greater<unsigned int>());
        thrust::sort(thrust::device, ans_value.begin(), ans_value.end(), thrust::greater<unsigned int>());
        return {ans_key, ans_value};
    };

    auto debug = [=]() {
        for (auto it = result_sort[1].result_key.begin(), it2 = result_sort[1].result_value.begin();
             it != result_sort[1].result_key.end() && it2 != result_sort[1].result_value.end(); it++, it2++) {
            wcout << static_cast<wchar_t >(*it) << ": " << *it2 << endl;
        }
    };
    result_nonSort[1] = func_convert_dict_to_result(dictBook);
    result_sort[1] = func_sort(result_nonSort[1]);
    //debug();


}

void ngramNameSpace::ngram::calculateTwoGram(size_t top) {

    //https://imgur.com/a/20FTpUU
    auto func_create_my_dict_and_cache = [](size_t threshold,
                                            thrust::universal_vector<unsigned int> dictBook) -> std::tuple<thrust::universal_vector<int>, thrust::universal_vector<unsigned int>> {

        thrust::universal_vector<int> twoGramDict(dictBook.size());
        // -1 : not exist
        thrust::universal_vector<unsigned int> twoGramDictCache(CHINESE_MAX, -1);
        //TODO write readme
        unsigned int time = 0;
        for (auto it = dictBook.begin(); it != dictBook.end(); it++) {
            if (*it > 0) {
                twoGramDict[time] = thrust::distance(dictBook.begin(), it);
                twoGramDictCache[thrust::distance(dictBook.begin(), it)] = time;
                time++;
            }
        }
        twoGramDict.resize(time);
        return {twoGramDict, twoGramDictCache};

    };
    auto debug = [=](thrust::universal_vector<int> twoGramDict, thrust::universal_vector<unsigned int> twoGramDictCache,
                     thrust::universal_vector<unsigned int> myMap) {
        for (auto it1 = twoGramDict.begin(); it1 != twoGramDict.end(); it1++) {
            for (auto it2 = twoGramDict.begin(); it2 != twoGramDict.end(); it2++) {
                wcout << static_cast<wchar_t >(*it1);
                wcout << static_cast<wchar_t >(*it2);
                wcout << ": ";
                auto addr1 = thrust::distance(twoGramDict.begin(), it1);
                auto addr2 = thrust::distance(twoGramDict.begin(), it2);
                wcout << myMap[addr1 * twoGramDict.size() + addr2];
                wcout << endl;
            }
        }
    };

    //將稀疏矩陣轉換成resultSOA (CSR format)
    //https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
    auto func_convert_to_result = [](thrust::universal_vector<int> twoGramDict,
                                 thrust::universal_vector<unsigned int> twoGramDictCache,
                                 thrust::universal_vector<unsigned int> myMap) -> resultSOA {

        thrust::universal_vector<unsigned int> scan(myMap.size());
        thrust::universal_vector<bool> pred(myMap.size());
        thrust::transform(thrust::device, myMap.begin(), myMap.end(), pred.begin(), isAboveTh{2});
        thrust::exclusive_scan(thrust::device, pred.begin(), pred.end(), scan.begin(), 0);
        auto newSize = scan[scan.size() - 1];

        thrust::universal_vector<unsigned int> key(newSize * 2);
        thrust::universal_vector<unsigned int> val(newSize);


        //package
        array<int> twoGramDictArr = {thrust::raw_pointer_cast(twoGramDict.data()), twoGramDict.size()};
        array<unsigned int> myMapArr = {thrust::raw_pointer_cast(myMap.data()), myMap.size()};
        array<unsigned int> scanArr = {thrust::raw_pointer_cast(scan.data()), scan.size()};
        array<bool> predArr = {thrust::raw_pointer_cast(pred.data()), pred.size()};
        array<unsigned int> keyArr = {thrust::raw_pointer_cast(key.data()), key.size()};
        array<unsigned int> valArr = {thrust::raw_pointer_cast(val.data()), val.size()};

        calculate::DenseMatrix2CSR<<<myMap.size() / 1024 +
                                     1, 1024>>>(twoGramDictArr, myMapArr, scanArr, predArr, keyArr, valArr, newSize);
        cudaDeviceSynchronize();
        catchError();

        return {key, val};


    };
    auto debugSec = [](resultSOA in) {
        auto it = 0;
        auto itFirst = in.result_key.begin();
        auto itSec = in.result_key.begin() + thrust::distance(in.result_value.begin(), in.result_value.end());
        auto itVal = in.result_value.begin();
        for (; itVal != in.result_value.end(); itFirst++, itSec++, itVal++, it++) {
            wcout << it << " ";
            wcout << static_cast<wchar_t >(*itFirst) << static_cast<wchar_t >(*itSec);
            wcout << ": ";
            wcout << *itVal << endl;
        }
    };
    //TODO
    auto func_sort = [](resultSOA in) -> resultSOA {

        thrust::universal_vector<unsigned int> key(in.result_key);
        thrust::universal_vector<unsigned int> val(in.result_value);
        auto mid = key.begin() + val.size();

        thrust::stable_sort_by_key(thrust::device, val.begin(), val.end(), key.begin(),
                                   thrust::greater<unsigned int>());
        val = in.result_value;
        thrust::stable_sort_by_key(thrust::device, val.begin(), val.end(), mid, thrust::greater<unsigned int>());
        thrust::stable_sort(thrust::device, val.begin(), val.end(), thrust::greater<unsigned int>());
        key.reserve(in.result_key.size());
        return {key, val};
    };


    unsigned int threshold = 1;
    //檢查是否超越邊界
    if (top > result_sort[1].result_value.size()) {
        top = result_sort[1].result_value.size() - 1;
    }
    threshold = result_sort[1].result_value[top];

    //key
    //TODO write readMe
    thrust::universal_vector<int> twoGramDict;
    //key B-tree index
    thrust::universal_vector<unsigned int> twoGramDictCache;
    //value
    thrust::universal_vector<unsigned int> myMap;

    //auto tempTup = benchmarkR<decltype(func_create_my_dict_and_cache(threshold, dictBook))(void)>(std::bind(func_create_my_dict_and_cache, threshold, dictBook),L"func_create_my_dict_and_cache");
    auto tempTup = func_create_my_dict_and_cache(threshold, dictBook);
    twoGramDict = std::get<0>(tempTup);
    twoGramDictCache = std::get<1>(tempTup);
    myMap.resize(twoGramDict.size() * twoGramDict.size());

    //package
    array<int> twoGramDictArr = {thrust::raw_pointer_cast(twoGramDict.data()), twoGramDict.size()};
    array<unsigned int> twoGramDictCacheArr = {thrust::raw_pointer_cast(twoGramDictCache.data()),
                                               twoGramDictCache.size()};
    array<unsigned int> myMapArr = {thrust::raw_pointer_cast(myMap.data()), myMap.size()};
    array<int> stringArr = {thrust::raw_pointer_cast(string.data()), string.size()};


    calculate::calculateTwo<<<stringArr.size / 1024 +
                              1, 1024>>>(stringArr, twoGramDictArr, twoGramDictCacheArr, myMapArr);
    cudaDeviceSynchronize();
    catchError();

    //test program run on CPU
    /*for(auto idx = 0 ; idx < stringArr.size ; idx++){
        [](array<int> string, array<int> twoGramDict, array<unsigned int> twoGramDictCache,array<unsigned int> myMap, size_t idx){
            if(idx + 2 > string.size)
                return;
            int nGram = 2;
            {

                int prep = string.ptr[idx + 0];
                unsigned int addrToDict = twoGramDictCache.ptr[prep];
                if(addrToDict == -1)
                    return;
                unsigned int addr = addrToDict * twoGramDict.size;

                prep = string.ptr[idx + 1];
                addrToDict = twoGramDictCache.ptr[prep];
                if(addrToDict == -1)
                    return;
                addr += addrToDict;

                myMap.ptr[addr]++;

            }
        }(stringArr, twoGramDictArr, twoGramDictCacheArr, myMapArr, idx);
    }*/

    //debug(twoGramDict, twoGramDictCache, myMap);
    //result_nonSort[2] = benchmarkR<decltype( convert_to_resault(twoGramDict, twoGramDictCache, myMap))(void)>(std::bind(convert_to_resault, twoGramDict, twoGramDictCache, myMap), L"convert_to_resault");
    result_nonSort[2] = func_convert_to_result(twoGramDict, twoGramDictCache, myMap);
    //debugSec(result_nonSort[2]);
    result_sort[2] = func_sort(result_nonSort[2]);


}

//out put data to std::stream
void ngramNameSpace::ngram::output(std::basic_ostream<wchar_t> &outStream) {
    for (auto ngr : result_sort) {
        auto size = ngr.second.result_value.size();
        outStream << ngr.first << "gram start!\n";
        for (auto i = 0; i < size; i++) {
            for (auto n = 0; n < ngr.first; n++) {
                outStream << static_cast<wchar_t >(ngr.second.result_key[i + n * size]);
            }
            outStream << ", ";
            outStream << ngr.second.result_value[i];
            outStream << endl;
        }
    }
}


//only call by ngram().func_create_dict()
__global__ void ngramNameSpace::initStruct::createDictBook(array<int> string, array<unsigned int> dict) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    int pred = string.ptr[idx];
    if (ifchinese(pred)) {
        atomicAdd(&(dict.ptr[pred]), 1);
    }
}


//only call by calculateTwoGram
__global__ void
ngramNameSpace::calculate::calculateTwo(array<int> string, array<int> twoGramDict, array<unsigned int> twoGramDictCache,
                                        array<unsigned int> myMap) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx + 2 > string.size)
        return;
    int nGram = 2;
    {

        int prep = string.ptr[idx + 0];
        unsigned int addrToDict = twoGramDictCache.ptr[prep];
        if (addrToDict == -1)
            return;
        unsigned int addr = addrToDict * twoGramDict.size;

        prep = string.ptr[idx + 1];
        addrToDict = twoGramDictCache.ptr[prep];
        if (addrToDict == -1)
            return;
        addr += addrToDict;

        atomicAdd(&(myMap.ptr[addr]), 1);

    }

}


//only call by calculateTwoGram.func_convert_to_result
__global__ void
ngramNameSpace::calculate::DenseMatrix2CSR(array<int> twoGramDict, array<unsigned int> myMap, array<unsigned int> scan,
                                           array<bool> pred,
                                           array<unsigned int> key, array<unsigned int> val, unsigned int newEnd) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx > myMap.size)
        return;
    if (!(pred.ptr[idx]))
        return;

    auto addr = scan.ptr[idx];
    unsigned int x = idx / twoGramDict.size;
    unsigned int y = idx % twoGramDict.size;
    key.ptr[addr] = twoGramDict.ptr[x];
    key.ptr[addr + newEnd] = twoGramDict.ptr[y];
    val.ptr[addr] = myMap.ptr[idx];

}

//only call by calculate.func_create_my_dict_and_cache
__global__ void ngramNameSpace::calculate::canCombineFunc(array<unsigned int> in, array<bool> out, int inGram) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= out.size)
        return;
    //in.size = in Ture size * in Gram
    size_t inSize = in.size / inGram;
    size_t outShape = static_cast<size_t >(sqrt(static_cast<float>(out.size)));
    size_t x = idx / outShape;
    size_t y = idx % outShape;
    for (auto i = 0; i < inGram - 1; i++) {
        if (in.ptr[x + (i + 1) * inSize] != in.ptr[y + i * inSize]) {
            out.ptr[idx] = false;
            return;
        }
    }
    out.ptr[idx] = true;

}

//only call by calculate.func_create_my_dict_and_cache
__global__ void
ngramNameSpace::calculate::createCombinePair(array<bool> pred, array<unsigned int> scan, array<unsigned int> in,
                                             array<unsigned int> out, int ngram) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t predShape = static_cast<size_t >(sqrt(static_cast<float>(pred.size)));
    size_t x = idx / predShape;
    size_t y = idx % predShape;
    if (!pred.ptr[idx])
        return;
    size_t newAddr = scan.ptr[idx];
    size_t newSize = out.size / ngram;
    size_t oldSize = in.size / (ngram - 1);
    for (auto i = 0; i < ngram - 1; i++) {
        out.ptr[newAddr + i * newSize] = in.ptr[x + i * oldSize];
    }
    out.ptr[newAddr + (ngram - 1) * newSize] = in.ptr[y + (ngram - 2) * oldSize];
}

__global__ void ngramNameSpace::calculate::calculateN(array<int> string, array<unsigned int>key, array<unsigned int>keyCache, array<unsigned int> val, int ngram){
    //TODO
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx +  ngram>= string.size)
        return;



    unsigned int begin, end, mid;
    unsigned int keyShape = key.size / ngram;

    if(!ifchinese(string.ptr[idx]))
        return;
    begin = keyCache.ptr[string.ptr[idx]];
    if(begin == -1){
        return;
    }
    if(string.ptr[idx] + 1 != keyCache.size){
        end = keyCache.ptr[string.ptr[idx] + 1];
        if(end == -1){
            end = val.size - 1;
        }
    }else{
        end = val.size - 1;
    }

    //itS: string iterator, itK: key Iterator
    auto equal = [=](unsigned int itS, unsigned int itK) -> bool{
        for(auto i = 0 ; i < ngram ; i++){
            if(string.ptr[itS + i] != key.ptr[itK + i * keyShape]){
                return false;
            }
        }
        return true;
    };
    auto SLargeThanK = [=](unsigned int itS, unsigned int itK) -> bool{
        for(auto i = 0 ; i < ngram ; i++){
            if(string.ptr[itS + i] > key.ptr[itK + i * keyShape]){
                return true;
            }else if(string.ptr[itS + i] < key.ptr[itK + i * keyShape]){
                return false;
            }
        }
        return false;
    };
    auto SSmallThanK = [=](unsigned int itS, unsigned int itK) -> bool{
        for(auto i = 0 ; i < ngram ; i++){
            if(string.ptr[itS + i] < key.ptr[itK + i * keyShape]){
                return true;
            }else if(string.ptr[itS + i] > key.ptr[itK + i * keyShape]){
                return false;
            }
        }
        return false;
    };


    for(;begin < end;){
        if(end - begin == 1){
            if(equal(idx, end)){
                atomicAdd(&(val.ptr[end]), 1);
            }else if(equal(idx, begin)){
                atomicAdd(&(val.ptr[begin]), 1);
            }
            return;
        }
        mid = (begin + end) / 2;
        if(equal(idx, mid)){
            atomicAdd(&(val.ptr[mid]), 1);
            return;
        }
        if(SLargeThanK(idx, mid)){
            begin = mid;
        }else{
            end = mid;
        }




    }


}