//
// Created by HIMONO on 2021/6/3.
//

#include "myThreadPool.cuh"


myThreadPoolSpace::myThreadPool::myThreadPool(int thread) {
    threadNum = thread;
    while (!myWorkList.empty())
        myWorkList.pop();

    killSignal = false;

    threadPool = new std::thread *[threadNum];
    for (auto i = 0; i < threadNum; i++) {
        threadPool[i] = new std::thread(&myThreadPool::worker, this);
    }


}

void myThreadPoolSpace::myThreadPool::worker() {
    while (!killSignal) {
        mux.lock();
        if (myWorkList.empty()) {
            mux.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(5));
            continue;
        }
        workList my = myWorkList.front();
        myWorkList.pop();

        mux.unlock();


        (my.work)();


    }
}

void myThreadPoolSpace::myThreadPool::go(std::function<void(void)> func) {


    mux.lock();

    myWorkList.push(workList{func});

    mux.unlock();
}

void myThreadPoolSpace::myThreadPool::killAll() {
    killSignal = true;
    for (auto i = 0; i < threadNum; i++) {
        threadPool[i]->join();
    }
}

myThreadPoolSpace::myThreadPool::~myThreadPool() {
    while (!myWorkList.empty());
    killAll();
    delete[] threadPool;
}