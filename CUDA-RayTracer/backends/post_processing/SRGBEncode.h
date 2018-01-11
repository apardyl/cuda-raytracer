#pragma once
#include "PostBackend.h"

class SRGBEncode : PostBackend {
    static float encodeSrgb(float f);
public:
    ~SRGBEncode() override = default;
    std::unique_ptr<Image> process(std::unique_ptr<Image> image) override;

};
