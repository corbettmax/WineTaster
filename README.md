Simple Feed Forward Neural Network

- Trains on a portion of the winequality-white.csv dataset
- Tests the model on the remaining data, outputting the mean squared error
- Parallelized using OpenMP

Compile with:

**g++ -fopenmp -O neuralnetwork neuralnetwork.cpp**


Run with:

**./neuralnetwork**
