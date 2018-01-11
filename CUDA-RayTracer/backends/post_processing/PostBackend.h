#pragma once
#include <memory>
#include "backends/Image.h"

class PostBackend {
public:
    virtual ~PostBackend() = default;

    virtual std::unique_ptr<Image> process(std::unique_ptr<Image> image) = 0;
};
