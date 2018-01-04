#include "CountDownLatch.h"

CountDownLatch::CountDownLatch(unsigned count) {
    this->count = count;
}

void CountDownLatch::await() {
    std::unique_lock<std::mutex> localLock(lock);
    if (count == 0) {
        return;
    }
    condition.wait(localLock);
}

void CountDownLatch::countDown() {
    std::unique_lock<std::mutex> localLock(lock);
    if (count == 0) {
        return;
    }

    --count;
    if (count == 0) {
        condition.notify_all();
    }
}
