#include <iostream>
#include <cmath>
#include <string>
#include <optional>
#include <fstream>
#include <vector>
#include <array>

typedef struct {
  std::string input_data_filepath;
  std::string output_data_filepath;
  float input_x;
} Params;

typedef struct {
  float x;
  float y;
  float x_act;
  float y_act;
} DataPoint;

typedef struct {
  float x;

  float pam;
  float pab;
  float pbm;
  float pbb;

  float leakyReluSlope;

  std::array<float, 2> data;
  std::array<float, 2> reluData;
  std::array<float, 2> softData;
  std::array<float, 2> expData;

  float loss;

  float dLdPam;
  float dLdPab;
  float dLdPbm;
  float dLdPbb;
  float alpha; 
} Network;

std::ostream& operator<<(std::ostream& os, const Network& net) {
    os << "Network {\n";
    
    os << "  Inputs:\n";
    os << "    x: " << net.x << "\n";

    os << "  Parameters:\n";
    os << "    pam: " << net.pam << "\n";
    os << "    pab: " << net.pab << "\n";
    os << "    pbm: " << net.pbm << "\n";
    os << "    pbb: " << net.pbb << "\n";
    os << "    leakyReluSlope: " << net.leakyReluSlope << "\n";

    os << "  Data Arrays:\n";
    os << "    data:     [" << net.data[0] << ", " << net.data[1] << "]\n";
    os << "    reluData: [" << net.reluData[0] << ", " << net.reluData[1] << "]\n";
    os << "    softData: [" << net.softData[0] << ", " << net.softData[1] << "]\n";
    os << "    expData:     [" << net.expData[0] << ", " << net.expData[1] << "]\n";

    os << "  Backprop:\n";
    os << "    loss:   " << net.loss << "\n";
    os << "    dLdPam: " << net.dLdPam << "\n";
    os << "    dLdPab: " << net.dLdPab << "\n";
    os << "    dLdPbm: " << net.dLdPbm << "\n";
    os << "    dLdPbb: " << net.dLdPbb << "\n";

    os << "  Other:\n";
    os << "    alpha: " << net.alpha << "\n";
    
    os << "}";
    
    return os;
}

std::optional<Params> processIO(int argc, char* argv[]) {
  Params params = {};
  
  // Validate arg length
  if (argc != 4) {
    std::cout << "Not enough args. Please input input file path and output file path and input x." << std::endl;
      return std::nullopt;
  }

  params.input_data_filepath = argv[1];
  params.output_data_filepath = argv[2];
  params.input_x = std::stof(argv[3]);

  return params;
}

float softmax(const int points, const float* data, const int index) {
  float num = exp(data[index]);
  float denom = 0;
  for (int i = 0; i < points; i++) {
    denom += exp(data[i]);
  }

  return num / denom;
}

float leakyRelu(const float value, const float slope) {
  return value > 0 ? value : slope * value;
}

float relu(const float value) {
  return leakyRelu(value, 0);
}

float crossEntropyLoss(const int points, const float *exp, const float *values) {
  float loss = 0;
  for (int i = 0; i < points; i++) {
    loss += exp[i] * std::log(values[i]);
  }
  loss *= -1;
  return loss;
}

std::vector<DataPoint> getTrainingPoints(std::ifstream *input_file) {
  // Read input data from file
  std::vector<DataPoint> training_points;
  training_points.push_back(DataPoint{});

  int pointIndex = 0;

  while (*input_file >> training_points[pointIndex].x >> training_points[pointIndex].y >> training_points[pointIndex].x_act >> training_points[pointIndex].y_act) {
    training_points.push_back(DataPoint{});
    pointIndex++;
  }

  training_points.pop_back();

  return training_points;
}

void forwardProp(Network *network, const float x, const float *expData) {
  network->x = x;

  network->data = {network->pam * network->x + network->pab, network->pbm * network->x + network->pbb};

  network->reluData = {leakyRelu(network->data[0], network->leakyReluSlope), leakyRelu(network->data[1], network->leakyReluSlope)};

  network->softData = {softmax(2, network->reluData.data(), 0), softmax(2, network->reluData.data(), 1)};

  network->expData = {expData[0], expData[1]};

  network->loss = crossEntropyLoss(2, network->expData.data(), network->softData.data());
}

void calculateGradients(Network *network) {
  network->dLdPam = network->x * (network->softData[0] - network->expData[0]);
  network->dLdPab = network->softData[0] - network->expData[0];
  network->dLdPbm = network->x * (network->softData[1] - network->expData[1]);
  network->dLdPbb = network->softData[1] - network->expData[1];
}

void backprop(Network *network) {
  network->pam -= network->dLdPam*network->alpha;
  network->pab -= network->dLdPab*network->alpha;
  network->pbm -= network->dLdPbm*network->alpha;
  network->pbb -= network->dLdPbb*network->alpha;
}

int main(int argc, char* argv[]) {
  
  std::optional<Params> optParams = processIO(argc, argv);
  Params params;
  if (optParams) {
    params = *optParams;
  } else {
    return 1;
  }

  std::ifstream input_file(params.input_data_filepath);
  if (!input_file.is_open()) {
    std::cout << "Input file doesn't exist." << std::endl;
    return 1;
  }

  std::ofstream output_file(params.output_data_filepath);

  std::vector<DataPoint> training_points = getTrainingPoints(&input_file);

  Network network = {
    .pam = 1,
    .pab = 1,
    .pbm = 2,
    .pbb = 1,
    .leakyReluSlope = 0.01,
    .alpha = 0.1
  };

  for (int i = 0; i < 200; i ++) {
    for (DataPoint training_point : training_points) {
      float x = training_point.x;

      float exp[2] = {training_point.x_act, training_point.y_act};

      forwardProp(&network, x, exp);

      calculateGradients(&network);

      backprop(&network);

      std::cout << network << std::endl;
      output_file << network << std::endl;
    }
  }

  return 0;
}
