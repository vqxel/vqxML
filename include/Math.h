// Math.h

#pragma once

#include <vector>

namespace vml {
  std::vector<float> softmax(const std::vector<float> &outputs);

  float leakyRelu(const float value, const float slope);

  float relu(const float value);

  float crossEntropyLoss(const int points, const float *exp, const float *values);

  float dotProduct(const std::vector<float>& v1, const std::vector<float>& v2);
}
