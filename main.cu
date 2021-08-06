#include "stdIncluded.h"
#include "initial_sys.cuh"
#include "utility.cuh"
#include "ngram.cuh"
//./enum /home/ascdc/solr200m.dat | tee $(date +'%Y%m%d:%H%M').txt

//
int main(int argc, char **argv) {
    if (argc != 2) {
        wcout << L"error command\n";
        return 1;
    }
    //set rand seed
    srand(time(NULL));

    std::locale utf8{"zh_TW.utf8"};
    //use locale -a check if your system support
    //install zh_TW.utf8
    initLocale(utf8);

    std::wstring orgStr = readFile(utf8, argv[1]);


    ngramNameSpace::ngram *myNgram = new ngramNameSpace::ngram(&orgStr);
    benchmark<void(void)>(std::bind(&ngramNameSpace::ngram::calculate, myNgram, 1, 0), L"one gram");
    benchmark<void(void)>(std::bind(&ngramNameSpace::ngram::calculate, myNgram, 2, 2000), L"two gram");
    benchmark<void(void)>(std::bind(&ngramNameSpace::ngram::calculate, myNgram, 3, 2000), L"3 gram");
    benchmark<void(void)>(std::bind(&ngramNameSpace::ngram::calculate, myNgram, 4, 2000), L"4 gram");
    benchmark<void(void)>(std::bind(&ngramNameSpace::ngram::calculate, myNgram, 5, 2000), L"5 gram");
    benchmark<void(void)>(std::bind(&ngramNameSpace::ngram::calculate, myNgram, 6, 2000), L"6 gram");
    benchmark<void(void)>(std::bind(&ngramNameSpace::ngram::calculate, myNgram, 7, 2000), L"7 gram");
    benchmark<void(void)>(std::bind(&ngramNameSpace::ngram::calculate, myNgram, 8, 2000), L"8 gram");

    //equal
    //myNgram->calculate(2, 500000);
    //myNgram->calculate(3, 400000);
    //myNgram->calculate(4, 300000);




    myNgram->output(std::wcout);


}
