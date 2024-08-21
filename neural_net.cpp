#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;

struct connection {
  double weight;
  double deltaweights;
};
class Neuron;
typedef vector<Neuron> layer;

// ===================================================================
class Neuron {

public:
  Neuron(unsigned numOutputs, unsigned tindex) {
    for (unsigned c = 0; c < numOutputs; c++) {
      weights.push_back(connection());
      weights.back().weight = randomWeight();
    }
    index = tindex;
  };

  void setoutputVal(double val) { output = val; };
  double getOutput(void) { return output; }

  void feedForward(layer &prevLayer) {
    double sum = 0.0; // output = summation(i to num of neuron (i x w))

    for (unsigned n = 0; n < prevLayer.size(); n++) {
      sum += prevLayer[n].getOutput() * prevLayer[n].weights[index].weight;
    }

    output = Neuron::transferFunction(sum);
  };

  void outputGrad(double target);

  void calcHiddenGradients(layer &nextLayer) {
    double dow = sumDOW(nextLayer);
    gradient = dow * Neuron::transferFunctionDerivative(output);
  };
  /*void updateWeight(layer &prevLayer) {*/
  /*  for (unsigned n = 0; n < prevLayer.size(); ++n) {*/
  /*    Neuron &neuron = prevLayer[n];*/
  /*    double oldDeltaweight = neuron.weights[index].deltaweights;*/
  /**/
  /*    double newDeltaWeight =*/
  /*        eta * neuron.getOutput() * gradient + alpha * oldDeltaweight;*/
  /**/
  /*    neuron.weights[index].deltaweights = newDeltaWeight;*/
  /*    neuron.weights[index].weight += newDeltaWeight;*/
  /*  }*/
  /*}*/

  void updateWeight(layer &prevLayer) {
    for (unsigned n = 0; n < prevLayer.size(); n++) {
      Neuron &neuron = prevLayer[n];
      if (index < neuron.weights.size()) { // Ensure index is valid
        double oldDeltaweight = neuron.weights[index].deltaweights;

        double newDeltaWeight =
            eta * neuron.getOutput() * gradient + alpha * oldDeltaweight;

        neuron.weights[index].deltaweights = newDeltaWeight;
        neuron.weights[index].weight += newDeltaWeight;
      }
    }
  }

private:
  static double transferFunction(double temp);
  static double transferFunctionDerivative(double temp);
  static double randomWeight() { return rand() / double(RAND_MAX); };
  double output;
  vector<connection> weights;
  unsigned index;
  double sumDOW(layer &nextLayer);
  double gradient;
  static double eta;
  static double alpha;
};

double Neuron::eta = 0.01;
double Neuron::alpha = 0.5;

double Neuron::sumDOW(layer &nextLayer) {
  double sum = 0.0;

  for (unsigned n = 0; n < nextLayer.size() - 1; n++) {
    sum += weights[n].weight * nextLayer[n].gradient;
  }
  return sum;
};

void Neuron::outputGrad(double target) {
  double delta = target - output;
  gradient = delta * Neuron::transferFunctionDerivative(output);
};

double Neuron::transferFunction(double temp) {
  // tanh
  return tanh(temp);
};

double Neuron::transferFunctionDerivative(double temp) {
  // derivative of tanh

  return 1 - pow(temp, 2);
};

// ===================================================================
class Net {
public:
  // constructor for class net
  Net(vector<unsigned> mapping) {

    unsigned numLayers = mapping.size();

    for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
      layers.push_back(layer()); // creates new layers
      unsigned numOutputs =
          layerNum == mapping.size() - 1 ? 0 : mapping[layerNum + 1];

      for (unsigned neuronNum = 0; neuronNum <= mapping[layerNum];
           neuronNum++) {
        layers.back().push_back(
            Neuron(numOutputs, neuronNum)); // appends neuron to new layer
        cout << "Generated a Neuron" << endl;
      }

      layers.back().back().setoutputVal(1.0);
    }
  };

  void feedforward(vector<double> &input) {
    assert(input.size() == layers[0].size() - 1); // subtarct neuron

    for (unsigned i = 0; i < input.size(); i++) {
      layers[0][i].setoutputVal(input[i]);
    }
    // looping each layer

    for (unsigned layerNum = 1; layerNum < layers.size(); ++layerNum) {
      layer &prevlayer = layers[layerNum - 1];
      for (unsigned n = 0; n < layers[layerNum].size() - 1; n++) {
        layers[layerNum][n].feedForward(prevlayer);
      }
    }
  };

  void backprop(vector<double> &target) {
    layer &outputLayer = layers.back();
    error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
      double delta = target[n] - outputLayer[n].getOutput();
      error += delta * delta;
    }
    error /= outputLayer.size() - 1;
    error = sqrt(error); // FINAL RMS

    // Output gradient
    for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
      outputLayer[n].outputGrad(target[n]);
    }

    // Hidden layer gradient
    for (unsigned layerNum = layers.size() - 2; layerNum > 0; layerNum--) {
      layer &hiddenLayer = layers[layerNum];
      layer &nextLayer = layers[layerNum + 1];

      for (unsigned n = 0; n < hiddenLayer.size(); n++) {
        hiddenLayer[n].calcHiddenGradients(nextLayer);
      }
    }

    // Update connected weights
    for (unsigned layerNum = layers.size() - 1; layerNum > 0; --layerNum) {
      layer &prevLayer = layers[layerNum - 1];
      layer &currLayer = layers[layerNum];

      for (unsigned n = 0; n < currLayer.size(); n++) {
        currLayer[n].updateWeight(prevLayer);
      }
    }
  };

  // This only results back values
  void returnResult(vector<double> &results) {
    results.clear();
    for (unsigned n = 0; n < layers.back().size() - 1; n++) {
      results.push_back(layers.back()[n].getOutput());
    }
  };

private:
  vector<layer> layers; // layers[layerNum][neuronNum]
  double error;
};
// ===================================================================

int main() {

  // setting up my net
  vector<unsigned> mapping;
  mapping.push_back(3);
  mapping.push_back(2);
  mapping.push_back(1);
  Net network(mapping);

  vector<double> input, target, results;

  // Back propogation
  network.backprop(target);

  // get back our results
  network.returnResult(results);
}
