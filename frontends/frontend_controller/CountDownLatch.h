#ifndef RAY_TRACER_COUNTDOWNLATCH_H
#define RAY_TRACER_COUNTDOWNLATCH_H

#include <mutex>
#include <condition_variable>

class CountDownLatch {
private:
    std::condition_variable condition;
    std::mutex lock;
    unsigned count;

public:
    explicit CountDownLatch(unsigned count);

    void await();

    void countDown();
};


#endif //RAY_TRACER_COUNTDOWNLATCH_H
