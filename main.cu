#include "stdIncluded.h"
#include "initial_sys.cuh"
#include "utility.cuh"
#include "myThreadPool.cuh"
#include "ngram.cuh"
//./enumV8 /home/ascdc/solr.dat | tee $(date +'%Y%m%d:%H%M').txt


int main(int argc, char **argv) {
    if (argc != 2) {
        wcout << L"error command\n";
        return 1;
    }
    srand(time(NULL));
    std::locale utf8{"zh_TW.utf8"}; //use locale -a check if your system support
    initLocale(utf8);
    std::wstring orgStr = readFile(utf8, argv[1]);


    ngramNameSpace::ngram *myNgram = new ngramNameSpace::ngram(&orgStr, 2000);
    //myNgram->calculate(2, 500000);
    //myNgram->calculate(3, 400000);
    //myNgram->calculate(4, 300000);

    benchmark<void(void)>(std::bind(&ngramNameSpace::ngram::calculate, myNgram, 2, 10000), L"2 - gram");
    benchmark<void(void)>(std::bind(&ngramNameSpace::ngram::calculate, myNgram, 3, 5000), L"3 - gram");
    benchmark<void(void)>(std::bind(&ngramNameSpace::ngram::calculate, myNgram, 4, 2500), L"4 - gram");


    myNgram->output();


}
