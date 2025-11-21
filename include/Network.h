// Network.h

#pragma once

#include <vector>

namespace vml {
  class Perceptron {
  public:
    std::vector<float> weights;
    float bias;

    // Intermediate calculations for backprop (dr_o / dr_i)
    std::vector<float> passthroughGrad;

    // For final backprop (dr_o / dw) (dr_o / db = dr_o / do * do / db = dr_o / do)
    std::vector<float> weightsGrad;
    float biasGrad;

    float leakyReluSlope;

    std::vector<float> input;
    float rawOutput;
    float reluOutput;

  public:
    Perceptron(int inputCount, float leakyReluSlope);

    float forward(const std::vector<float> &inputs);

    void populatePassthroughGrad();

    std::vector<float> serialize();
  };

  class Layer {
  public:
    std::vector<Perceptron> perceptrons;
    std::vector<float> output;

    std::vector<float> cascadingGrad;

  public:
    Layer(const std::vector<Perceptron> &perceptrons);

    Layer(int width, int prevWidth);

    void populateCascadingGrad(vml::Layer &nextLayer);

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
