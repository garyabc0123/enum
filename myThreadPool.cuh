//
// Created by HIMONO on 2021/6/3.
//

#ifndef ENUMV7_MYTHREADPOOL_CUH
#define ENUMV7_MYTHREADPOOL_CUH

#include "stdIncluded.h"


namespace myThreadPoolSpace {

    //工作駔列
    //會先從最前面開始做，後加入的會被放在最後面
    //FIFO
    struct workList {
        std::function<void(void)> work;
    };


    class myThreadPool {
    public:
        myThreadPool(int thread);

        ~myThreadPool();

        void go(std::function<void(void)> func);

        void killAll();

    private:
        int threadNum;


        std::queue <workList> myWorkList;
        bool killSignal;

        std::mutex mux;
        std::thread **threadPool;


        void worker();
    };
}


#endif //ENUMV7_MYTHREADPOOL_CUH
