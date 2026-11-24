#pragma once
#include "matrix.h"
#include <vector>
#include <string>

class ModelBase {
public:
    virtual ~ModelBase() = default;
    virtual Matrix<float> forward(const Matrix<float>& input) const = 0;
};