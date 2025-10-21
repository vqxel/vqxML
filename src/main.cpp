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
} Params;

typedef struct {
  float x;
  float y;
  float x_act;
  float y_act;
} DataPoint;

std::optional<Params> processIO(int argc, char* argv[]) {
  Params params = {};
  
  // Validate arg length
  if (argc != 4) {
    std::cout << "Not enough args. Please input input file path and output file path and input x." << std::endl;
      return std::nullopt; }

  params.input_data_filepath = argv[1];
  params.output_data_filepath = argv[2];
  params.input_x = std::stof(argv[3]);

  return params;
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
    .loss = 1,
    .alpha = 0.1
  };

  while (network.loss >= 0.001) {
    for (DataPoint training_point : training_points) {
      float x = training_point.x;

      float exp[2] = {training_point.x_act, training_point.y_act};

      forwardProp(&network, x, exp);

      calculateGradients(&network);

      backprop(&network);

      network.epochs++;

      std::cout << network << std::endl;
      output_file << network << std::endl;
    }
  }

  std::cout << "Training complete with " << network.epochs << " epochs" << std::endl;

  return 0;
}
