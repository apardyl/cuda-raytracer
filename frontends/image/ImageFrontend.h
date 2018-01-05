#ifndef RAY_TRACER_IMAGEFRONTEND_H
#define RAY_TRACER_IMAGEFRONTEND_H

#include <mutex>
#include <condition_variable>
#include "frontends/Frontend.h"

class ImageFrontend : public Frontend {
private:
    std::condition_variable condition;
    std::mutex lock;

    std::unique_ptr<Image> image = nullptr;
    bool shouldTerminate = false;

    void savePNG(Image const &image);
public:
    ImageFrontend();

    ~ImageFrontend() override;

    void run() override;

    void setImage(Image image) override;

    bool isDaemon() override;

    void terminate() override;
};

#endif //RAY_TRACER_IMAGEFRONTEND_H
