//
// Created by ascdc on 2021-05-27.
//

#include "utility.cuh"

__host__ __device__ bool ifchinese2(int in) {
    if (in < L'㐀')
        return false; //U+3400
    if (in >= L'！' && in <= L'～')
        return false;
    if (in >= L'￨' && in <= L'￮')
        return false;
    if (in == L'，')
        return false;
    if (in == L'。')
        return false;
    return true;
}

__host__ __device__ bool ifchinese(int tmpUCS4) {
    return (tmpUCS4 >= 0x4E00 && tmpUCS4 <= 0x62FF) ||
           (tmpUCS4 >= 0x6300 && tmpUCS4 <= 0x77FF) ||
           (tmpUCS4 >= 0x7800 && tmpUCS4 <= 0x8CFF) ||
           (tmpUCS4 >= 0x8D00 && tmpUCS4 <= 0x9FFF) ||
           /* CJK Unified Ideographs Extension A */
           (tmpUCS4 >= 0x3400 && tmpUCS4 <= 0x4DBF) ||
           /* CJK Unified Ideographs Extension B */
           (tmpUCS4 >= 0x20000 && tmpUCS4 <= 0x215FF) ||
           (tmpUCS4 >= 0x21600 && tmpUCS4 <= 0x230FF) ||
           (tmpUCS4 >= 0x23100 && tmpUCS4 <= 0x245FF) ||
           (tmpUCS4 >= 0x24600 && tmpUCS4 <= 0x260FF) ||
           (tmpUCS4 >= 0x26100 && tmpUCS4 <= 0x275FF) ||
           (tmpUCS4 >= 0x27600 && tmpUCS4 <= 0x290FF) ||
           (tmpUCS4 >= 0x29100 && tmpUCS4 <= 0x2A6DF) ||
           /* CJK Unified Ideographs Extension C */
           (tmpUCS4 >= 0x2A700 && tmpUCS4 <= 0x2B73F) ||
           /* CJK Unified Ideographs Extension D */
           (tmpUCS4 >= 0x2B740 && tmpUCS4 <= 0x2B81F) ||
           /* CJK Unified Ideographs Extension E */
           (tmpUCS4 >= 0x2B820 && tmpUCS4 <= 0x2CEAF) ||
           /* CJK Unified Ideographs Extension F */
           (tmpUCS4 >= 0x2CEB0 && tmpUCS4 <= 0x2EBEF) ||
           /* Hiragana */
           (tmpUCS4 >= 0x3040 && tmpUCS4 <= 0x309F) ||
           /* Katakana */
           (tmpUCS4 >= 0x30A0 && tmpUCS4 <= 0x30FF) ||
           /* Katakana Phonetic Extensions */
           (tmpUCS4 >= 0x31F0 && tmpUCS4 <= 0x31FF) ||
           /* Hangul Jamo */
           (tmpUCS4 >= 0x1100 && tmpUCS4 <= 0x11FF) ||
           /* Hangul Jamo Extended-A */
           (tmpUCS4 >= 0xA960 && tmpUCS4 <= 0xA97F) ||
           /* Hangul Jamo Extended-B */
           (tmpUCS4 >= 0xD7B0 && tmpUCS4 <= 0xD7FF) ||
           /* Hangul Compatibility Jamo */
           (tmpUCS4 >= 0x3130 && tmpUCS4 <= 0x318F) ||
           /* Hangul Syllables */
           (tmpUCS4 >= 0xAC00 && tmpUCS4 <= 0xD7AF);
}

__host__ __device__  unsigned int utf32ChineseReCoding(unsigned int tmp) {
    unsigned int a = reCodeNotDef;
    if (tmp < 0x1100) {
        return a;
    }
    //0x1100~0x11FF
    if (tmp <= 0x11FF) {
        return tmp - 0x1100;
    }
    if (tmp < 0x3040) {
        return a;
    }
    //0x3040~0x9FFF
    if (tmp <= 0x9FFF) {
        return tmp - 0x3040 + 0x100;
    }
    if (tmp < 0xA960) {
        return a;
    }
    //0xA960~0xA97F
    if (tmp <= 0xA97F) {
        return tmp - 0xA960 + 0x100 + 0x6FC0;
    }
    if (tmp < 0xAC00) {
        return a;
    }
    //0xAC00~0xD7FF
    if (tmp <= 0xD7FF) {
        return tmp - 0xAC00 + 0x100 + 0x6FC0 + 0x20;
    }

    if (tmp < 0x20000) {
        return a;
    }
    //0x20000~0x2EBFF
    if (tmp <= 0x2EBFF) {
        return tmp - 0x20000 + 0x100 + 0x6FC0 + 0x20 + 0x2C00;
    }
    //MAX 188DE



    return a;
}

__host__ __device__  unsigned int Reutf32ChineseReCoding(unsigned int tmp) {
    if (tmp == reCodeNotDef)
        return tmp;
    //0x1100~0x11FF
    if (tmp < 0x100) {
        return tmp + 0x1100;
    }
    //0x3040~0x9FFF
    if (tmp < 0x100 + 0x6FC0) {
        return tmp + 0x3040 - 0x100;
    }
    //0xA960~0xA97F
    if (tmp < 0x100 + 0x6FC0 + 0x20) {
        return tmp + 0xA960 - 0x100 - 0x6FC0;
    }
    //0xAC00~0xD7FF
    if (tmp < 0x100 + 0x6FC0 + 0x20 + 0x2C00) {
        return tmp + 0xAC00 - 0x100 - 0x6FC0 - 0x20;
    }
    return tmp + 0x20000 - 0x100 - 0x6FC0 - 0x20 - 0x2C00;

}

void catchError() {
    cudaError_t err;
    err = cudaGetLastError();
    if (err != cudaSuccess) {

        std::wcout << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}
