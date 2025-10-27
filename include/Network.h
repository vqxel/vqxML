// Network.h

#pragma once

#include <vector>

namespace vml {
  class Perceptron {
  public:
    std::vector<float> weights;
    float bias;

    std::vector<float> weightsGrad;
    float biasGrad;

    std::vector<float> input;
    float rawOutput;
    float reluOutput;

  public:
    Perceptron(const std::vector<float> &weights, float bias);

    Perceptron(int inputCount);

    float forward(const std::vector<float> &inputs, const float leakyReluSlope);

    std::vector<float> serialize();
  };

  class Layer {
  public:
    std::vector<Perceptron> perceptrons;
    std::vector<float> output;

  public:
    Layer(const std::vector<Perceptron> &perceptrons);

    Layer(int width, int prevWidth);

    int width() const;
  };

  class Network {
  public:
    std::vector<Layer> layers;

    std::vector<float> expectedData;
    std::vector<float> softOutput;

    float leakyReluSlope;

    float loss;

    int epochs;

  public:
    Network(const std::vector<Layer> &layers, float leakyReluSlope);

    Network(const std::vector<int> &layerSizes, int inputs, float leakyReluSlope);

    std::vector<float> forwardProp(const std::vector<float> &input, const std::vector<float> &expectedData);
  };
}
