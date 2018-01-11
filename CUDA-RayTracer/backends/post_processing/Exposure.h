#pragma once
#include "PostBackend.h"

class Exposure : PostBackend {
    float exposure;
    void autoExposure(const Image * image);

public:
    explicit Exposure(float exposure = 0);
    ~Exposure() override = default;
    std::unique_ptr<Image> process(std::unique_ptr<Image>) override;
};
