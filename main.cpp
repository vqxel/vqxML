#include <iostream>
#include <cmath>
#include <string>
#include <optional>
#include <fstream>
#include <vector>

typedef struct {
  std::string input_data_filepath;
  std::string output_data_filepath;
  float input_x;
} Params;

typedef struct {
  float x;
  float y;
  float a_act;
  float b_act;
} DataPoint;

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

float relu(const float value) {
  return value > 0 ? value : 0;
}

std::vector<DataPoint> getTrainingPoints(std::ifstream *input_file) {
  // Read input data from file
  std::vector<DataPoint> training_points;
  training_points.push_back(DataPoint{});

  int pointIndex = 0;

  while (*input_file >> training_points[pointIndex].x >> training_points[pointIndex].y >> training_points[pointIndex].a_act >> training_points[pointIndex].b_act) {
    training_points.push_back(DataPoint{});
    pointIndex++;
  }

  training_points.pop_back();

  return training_points;
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


  float pam = 1;
  float pab = 1;
  
  float pbm = 0.5;
  float pbb = 0.5;

  float x = params.input_x;

  float data[2] = {pam * x + pab, pbm * x + pbb};

  float a_out = softmax(2, data, 0);
  float b_out = softmax(2, data, 1);

  std::cout << "Output: " << a_out << "   " << b_out << std::endl;

  return 0;
}
