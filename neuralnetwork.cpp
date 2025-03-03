#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <omp.h>

using namespace std;

// Header for the neural network class
// Small enough that I'm just keeping it in the main file
class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hiddenSize1, int hiddenSize2, int outputSize);
    vector<double> forwardPass(const vector<double>& inputs);
    void backpropagate(const vector<double>& inputs, const vector<double>& targets, double learningRate);
    double error(double x);
    double errorDerivative(double x);

private:
    int inputSize;
    int hiddenSize1;
    int hiddenSize2;
    int outputSize;
    vector<vector<double>> weightsInputHidden1;
    vector<vector<double>> weightsHidden1Hidden2;
    vector<vector<double>> weightsHidden2Output;
    vector<double> hiddenLayer1;
    vector<double> hiddenLayer2;
    vector<double> outputLayer;
};

// Constructor
// Initializes the neural network with the given sizes
// Generates random weights for the connections between the layers
NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize1, int hiddenSize2, int outputSize)
    : inputSize(inputSize), hiddenSize1(hiddenSize1), hiddenSize2(hiddenSize2), outputSize(outputSize) {
    srand(time(0));

    weightsInputHidden1.resize(inputSize, vector<double>(hiddenSize1));
    weightsHidden1Hidden2.resize(hiddenSize1, vector<double>(hiddenSize2));
    weightsHidden2Output.resize(hiddenSize2, vector<double>(outputSize));

    for (int i = 0; i < inputSize; ++i)
        for (int j = 0; j < hiddenSize1; ++j)
            weightsInputHidden1[i][j] = ((double) rand() / (RAND_MAX));

    for (int i = 0; i < hiddenSize1; ++i)
        for (int j = 0; j < hiddenSize2; ++j)
            weightsHidden1Hidden2[i][j] = ((double) rand() / (RAND_MAX));

    for (int i = 0; i < hiddenSize2; ++i)
        for (int j = 0; j < outputSize; ++j)
            weightsHidden2Output[i][j] = ((double) rand() / (RAND_MAX));
}

// Forward pass through the network
// Given an input vector, computes the output of the network
vector<double> NeuralNetwork::forwardPass(const vector<double>& inputs) {
    hiddenLayer1.resize(hiddenSize1);
    hiddenLayer2.resize(hiddenSize2);
    outputLayer.resize(outputSize);

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (int j = 0; j < hiddenSize1; ++j) {
                hiddenLayer1[j] = 0.0;
                for (int i = 0; i < inputSize; ++i) {
                    hiddenLayer1[j] += inputs[i] * weightsInputHidden1[i][j];
                }
                hiddenLayer1[j] = error(hiddenLayer1[j]);
            }
        }

        #pragma omp section
        {
            for (int j = 0; j < hiddenSize2; ++j) {
                hiddenLayer2[j] = 0.0;
                for (int i = 0; i < hiddenSize1; ++i) {
                    hiddenLayer2[j] += hiddenLayer1[i] * weightsHidden1Hidden2[i][j];
                }
                hiddenLayer2[j] = error(hiddenLayer2[j]);
            }
        }

        #pragma omp section
        {
            for (int k = 0; k < outputSize; ++k) {
                outputLayer[k] = 0.0;
                for (int j = 0; j < hiddenSize2; ++j) {
                    outputLayer[k] += hiddenLayer2[j] * weightsHidden2Output[j][k];
                }
                outputLayer[k] = error(outputLayer[k]);
            }
        }
    }

    return outputLayer;
}

// Backpropagation algorithm
// Given an input vector, the target output vector, and a learning rate,
// updates the weights of the network to minimize error
void NeuralNetwork::backpropagate(const vector<double>& inputs, const vector<double>& targets, double learningRate) {
    vector<double> outputErrors(outputSize);
    vector<double> hiddenErrors2(hiddenSize2);
    vector<double> hiddenErrors1(hiddenSize1);

    for (int k = 0; k < outputSize; ++k) {
        outputErrors[k] = targets[k] - outputLayer[k];
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (int j = 0; j < hiddenSize2; ++j) {
                hiddenErrors2[j] = 0.0;
                for (int k = 0; k < outputSize; ++k) {
                    hiddenErrors2[j] += outputErrors[k] * weightsHidden2Output[j][k];
                }
                hiddenErrors2[j] *= errorDerivative(hiddenLayer2[j]);
            }
        }

        #pragma omp section
        {
            for (int j = 0; j < hiddenSize1; ++j) {
                hiddenErrors1[j] = 0.0;
                for (int k = 0; k < hiddenSize2; ++k) {
                    hiddenErrors1[j] += hiddenErrors2[k] * weightsHidden1Hidden2[j][k];
                }
                hiddenErrors1[j] *= errorDerivative(hiddenLayer1[j]);
            }
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (int j = 0; j < hiddenSize2; ++j) {
                for (int k = 0; k < outputSize; ++k) {
                    #pragma omp atomic
                    weightsHidden2Output[j][k] += learningRate * outputErrors[k] * hiddenLayer2[j];
                }
            }
        }

        #pragma omp section
        {
            for (int i = 0; i < hiddenSize1; ++i) {
                for (int j = 0; j < hiddenSize2; ++j) {
                    #pragma omp atomic
                    weightsHidden1Hidden2[i][j] += learningRate * hiddenErrors2[j] * hiddenLayer1[i];
                }
            }
        }

        #pragma omp section
        {
            for (int i = 0; i < inputSize; ++i) {
                for (int j = 0; j < hiddenSize1; ++j) {
                    #pragma omp atomic
                    weightsInputHidden1[i][j] += learningRate * hiddenErrors1[j] * inputs[i];
                }
            }
        }
    }
}

// error function
double NeuralNetwork::error(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the error function
double NeuralNetwork::errorDerivative(double x) {
    return x * (1.0 - x);
}

vector<vector<double>> loadDataset(const string& filename) {
    vector<vector<double>> dataset;
    ifstream file(filename);
    string line;
    string::size_type sz;

    // Skip the first line (header)
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<double> row;

        while (getline(ss, value, ';') && !value.empty()) {
            if (!value.empty()){
                // Remove quotes if present
                value.erase(remove(value.begin(), value.end(), '\"'), value.end());
                row.push_back(stod(value, &sz));
            }
        }

        dataset.push_back(row);
    }

    return dataset;
}

void normalizeDataset(vector<vector<double>>& dataset) {
    for (size_t i = 0; i < dataset[0].size() - 1; ++i) {
        double minVal = dataset[0][i];
        double maxVal = dataset[0][i];

        for (size_t j = 1; j < dataset.size(); ++j) {
            if (dataset[j][i] < minVal) minVal = dataset[j][i];
            if (dataset[j][i] > maxVal) maxVal = dataset[j][i];
        }

        for (size_t j = 0; j < dataset.size(); ++j) {
            dataset[j][i] = (dataset[j][i] - minVal) / (maxVal - minVal);
        }
    }
}

void splitDataset(const vector<vector<double>>& dataset, vector<vector<double>>& trainingSet, vector<vector<double>>& testingSet, double trainRatio) {
    size_t trainSize = static_cast<size_t>(dataset.size() * trainRatio);
    trainingSet.assign(dataset.begin(), dataset.begin() + trainSize);
    testingSet.assign(dataset.begin() + trainSize, dataset.end());
}

int main() {
    NeuralNetwork nn(11, 8, 8, 1);

    vector<vector<double>> dataset = loadDataset("./winequality-white.csv");
    normalizeDataset(dataset);

    vector<vector<double>> trainingSet, testingSet;
    splitDataset(dataset, trainingSet, testingSet, 0.1); // 10% training, 90% testing

    auto trainingstart = chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < 10000; ++epoch) {
        #pragma omp parallel for
        for (size_t i = 0; i < trainingSet.size(); ++i) {
            const auto& row = trainingSet[i];
            vector<double> inputs(row.begin(), row.end() - 1);
            vector<double> targets = { row.back() / 10.0 }; // Normalize target to [0, 1]
            #pragma omp critical
            {
                nn.forwardPass(inputs);
                nn.backpropagate(inputs, targets, 0.1);
            }
        }
    }

    auto trainingend = chrono::high_resolution_clock::now();
    chrono::duration<double> trainingduration = trainingend - trainingstart;

    auto testingstart = chrono::high_resolution_clock::now();

    double totalError = 0.0;
    #pragma omp parallel for reduction(+:totalError)
    for (size_t i = 0; i < testingSet.size(); ++i) {
        const auto& row = testingSet[i];
        vector<double> inputs(row.begin(), row.end() - 1);
        vector<double> targets = { row.back() / 10.0 }; // Normalize target to [0, 1]
        vector<double> outputs = nn.forwardPass(inputs);
        totalError += pow(targets[0] - outputs[0], 2);
    }

    auto testingend = chrono::high_resolution_clock::now();
    chrono::duration<double> testingduration = testingend - testingstart;

    cout << "Training set size: " << trainingSet.size() << endl;
    cout << "Testing set size: " << testingSet.size() << endl;
    cout << "Total error on testing set: " << totalError << endl;

    cout << "Mean Squared Error on testing set: " << totalError / testingSet.size() << endl;
    cout << "Time taken to train the model: " << trainingduration.count() << " seconds" << endl;
    cout << "Time taken to evaluate the testing set: " << testingduration.count() << " seconds" << endl;

    return 0;
}