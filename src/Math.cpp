// Math.cpp

#include "Math.h"
#include <cmath>
#include <numeric>

namespace vml {
  std::vector<float> softmax(const std::vector<float> &outputs) {
    std::vector<float> softOutputs;
    
    float denom = 0;
    for (float denomOutput : outputs) {
      denom += exp(denomOutput);
    }

    for (float output : outputs) {
      float num = exp(output);
      float soft = num / denom;
      softOutputs.push_back(soft);
    }

    return softOutputs; 
  }

  float leakyRelu(const float value, const float slope) {
    return value > 0 ? value : slope * value;
  }

  float relu(const float value) {
    return leakyRelu(value, 0.0f);
  }

  float crossEntropyLoss(const int points, const float *exp, const float *values) {
    float loss = 0;
    for (int i = 0; i < points; i++) {
      loss += exp[i] * std::log(values[i]);
    }
    loss *= -1;
    return loss;
  }

  float dotProduct(const std::vector<float>& v1, const std::vector<float>& v2) {
    if (v1.size() != v2.size()) {
        return 0.0;
    }

    return std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);
  }
}
