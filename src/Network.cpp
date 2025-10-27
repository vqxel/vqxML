// Network.cpp

#include "Network.h"
#include "Math.h"
#include <iostream>
#include <ostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <string>

using namespace vml;

Perceptron::Perceptron(const std::vector<float> &weights, float bias): weights(weights), bias(bias), weightsGrad(weights.size(), 0.0f), biasGrad(0.0f), input(weights.size(), 0.0f), rawOutput(0.0f), reluOutput(0.0f) {
}

Perceptron::Perceptron(int inputCount): weights(inputCount, 0.0f), bias(0.0f), weightsGrad(inputCount, 0.0f), biasGrad(0.0f), input(inputCount, 0.0f), rawOutput(0.0f), reluOutput(0.0f) {
}

Layer::Layer(const std::vector<Perceptron> &perceptrons): perceptrons(perceptrons) {
}

Layer::Layer(int width, int prevWidth): perceptrons(width, prevWidth) {
}

int Layer::width() const {
  return perceptrons.size();
}

Network::Network(const std::vector<Layer> &layers, float leakyReluSlope): layers(layers), expectedData(layers.back().width()), leakyReluSlope(leakyReluSlope), loss(loss), epochs(0) {
}

Network::Network(const std::vector<int> &layerSizes, int inputs, float leakyReluSlope): leakyReluSlope(leakyReluSlope), loss(0.0f), epochs(0) { 
  // TODO: Verify layers is large enough
  for (int i = 0; i < layerSizes.size(); i++) {
    if (i == 0) {
      layers.emplace_back(layerSizes[i], inputs);
    } else {
      layers.emplace_back(layerSizes[i], layerSizes[i - 1]);
    }
  }
}

float Perceptron::forward(const std::vector<float> &input, const float leakyReluSlope) {
  this->input = input;
  
  this->rawOutput = dotProduct(weights, input) + bias;

  this->reluOutput = leakyRelu(this->rawOutput, leakyReluSlope);

  return this->reluOutput;
}

std::vector<float> Network::forwardProp(const std::vector<float> &input, const std::vector<float> &expectedData) {
  for (int i = 0; i < layers.size(); i++) {
    Layer layer = layers[i];
    // Go through each perceptron, calculate forward value, and add to the results vector for the layer
    for (Perceptron &perceptron : layer.perceptrons) {
      const std::vector<float> &layerInput = i > 0 ? layers[i-1].output : input;
      float perceptronResult = perceptron.forward(layerInput, this->leakyReluSlope);
      layer.output.push_back(perceptronResult);
    }
  }

  softOutput = softmax(layers.back().output);
  return softOutput;
}

// Disclaimer: AI generated utility function
std::ostream& operator<<(std::ostream& os, const Network& net) {
    // --- 1. Print Network Header ---
    os << "========================================\n";
    os << "         NETWORK ARCHITECTURE         \n";
    os << "========================================\n";
    os << "Leaky ReLU Slope: " << net.leakyReluSlope << "\n";

    if (net.layers.empty()) {
        os << "Network has no layers." << std::endl;
        os << "========================================\n";
        return os;
    }

    // --- 2. Find and print the overall structure ---
    int netInputSize = 0;
    if (!net.layers[0].perceptrons.empty()) {
        netInputSize = net.layers[0].perceptrons[0].weights.size();
    }
    
    os << "Structure: " << netInputSize;
    for (const Layer& layer : net.layers) {
        os << " -> " << layer.perceptrons.size();
    }
    os << "\n\n";
    
    // Set formatting for all floating point numbers
    os << std::fixed << std::setprecision(4);

    // --- 3. Loop Through Each Layer (as a vertical block) ---
    int layerInputSize = netInputSize;
    for (size_t i = 0; i < net.layers.size(); ++i) {
        const Layer& layer = net.layers[i];
        int layerOutputSize = layer.perceptrons.size();
        
        // Print Layer Header
        os << "----------------------------------------\n";
        os << "  LAYER " << i << " (Input: " << layerInputSize << ", Output: " << layerOutputSize << ")\n";
        os << "----------------------------------------\n";

        // --- 4. Loop Through Each Perceptron in the Layer ---
        for (size_t p_idx = 0; p_idx < layer.perceptrons.size(); ++p_idx) {
            const Perceptron& p = layer.perceptrons[p_idx];
            
            // Print Perceptron Index and Bias
            os << "  P" << p_idx << ": Bias: " << p.bias << "\n";
            
            // Print All Weights
            os << "       Weights (" << p.weights.size() << "): [";
            for (size_t w_idx = 0; w_idx < p.weights.size(); ++w_idx) {
                os << p.weights[w_idx];
                if (w_idx < p.weights.size() - 1) {
                    os << ", ";
                }
            }
            os << "]\n"; // End of weights array
            
            if (p_idx < layer.perceptrons.size() - 1) {
                 os << "  ...\n"; // Small separator between perceptrons
            }
        }
        
        // The output size of this layer is the input size of the next
        layerInputSize = layerOutputSize;
    }
    
    os << "========================================\n";
    
    // Reset ostream formatting to default
    os.unsetf(std::ios_base::floatfield); 
    
    return os; // Always return the stream
}

#include <fstream>

int main(int argv, char **argc) {
  float f1[2] = {-187.33667, -423.43}, f2 = 0.0;
  std::ofstream out("test2.bin",std::ios_base::binary);
  if(out.good())
  {
    std::cout << "Writing floating point number: " << std::fixed << f1 << std::endl;
    out.write(reinterpret_cast<char *>(f1),2 * sizeof(float));
    out.close();
  }
  std::ifstream in("test2.bin",std::ios_base::binary);
  if(in.good())
  {
    in.read((char *)&f2,2 * sizeof(float));
    std::cout << "Reading floating point number: " << std::fixed << f2 << std::endl;
  }
  return 0;
}
