#pragma once
#include "PostBackend.h"

struct SuperSampler : PostBackend {
    std::unique_ptr<Image> process(std::unique_ptr<Image> image) override;
};
