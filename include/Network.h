// Network.h

#pragma once

#include <vector>

namespace vml {
  class Perceptron {
  private:
    std::vector<float> weights;
    float bias;

    // Backprop
    // Intermediate calculations for backprop (dr_o / dr_i)
    std::vector<float> passthroughGrad;
    // For final backprop (dr_o / dw) (dr_o / db = dr_o / do * do / db = dr_o / do)
    std::vector<float> weightsGrad;
    float biasGrad;

    // Stateful
    std::vector<float> &input;
    float output;

  public:
    Perceptron(int inputCount, float leakyReluSlope);

    float forward(const std::vector<float> &inputs);

    void populatePassthroughGrad(std::vector<float> &input);

    int getInputWidth();

    //std::vector<float> serialize();
  };

  class Layer {
  protected:
    std::vector<float> output;

    float cascadingGradSum;

    virtual void calculateForwardProp(std::vector<float> &input) = 0;

  public:
    //Layer(int width, int prevWidth);

    float getCascadingGradSum();

    void forwardProp(std::vector<float> &input);
    
    std::vector<float> getOutput();

    virtual void populateCascadingGradSum(std::vector<float> &nextLayerCascadingGradSum) = 0;

    virtual int width() const = 0;

    virtual int getInputWidth() const = 0;
  };

  class DenseLayer : public Layer {
  private:
    std::vector<Perceptron> perceptrons;

    void calculateForwardProp(std::vector<float> &input) override;
  public:

    void populateCascadingGradSum(std::vector<float> &nextLayerCascadingGradSum) override;

    int width() const override;

    int getInputWidth() const override;
  }

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

    void populateCascadingGrads();

    void backwardProp();
  };
}
