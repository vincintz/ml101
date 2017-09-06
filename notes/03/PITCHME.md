# Machine Learning 101
## Multilayer Perceptron (MLP)

---
## Recall
* Multilayer Perceptron (MLP)
* Forward Computation
- Training: Backpropagation

---
### Multilayer Perceptron (MLP)

![Picture of MLP](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/mlp.png)

+++
### Multilayer Perceptron (MLP)

#### Neuron aka node, perceptron
![Neuron](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/recall-neuron.png)

+++
#### Layer
![Neuron](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/recall-layer.png)

+++
#### Multi-layer Perceptron
![Neuron](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/recall-mlp.png)

---
### Forward Computation
![Picture of MLP](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/xor-mlp.png)

+++
### Forward Computation
```
class MLP:
    public double[] feedForward(double... input) {
        double[] layerOutput = input;
        for (int l = 0; l < layers.length; l++) {
            layerOutput = layers[l].feedForward(layerOutput);
        }
        return layerOutput;
    }
```
```
class Layer:
    double[] feedForward(double[] input) {
        crossMultiply(output, weights, input);
        vectorAdd(output, output, bias);
        activate(output, activationFn);
        return output;
    }
```

---
### Training Algorithm

* Run forward computation
* Compute cost at the output layer
* Update the weights and bias such that a certain error function is minimized

---
## Gradient Descent

![Gradient)](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/gradientdescent.png)

* Compute cost at the output layer
* Change in weigh is based on the cost gradient
  * derivative of cost wrt weight
* Cost function is not simple
* We need a simple way to compute the gradient

+++

---
### Gradient Intuition

* We isolate a sample weight/bias (first bias at first layer)

![Gradient)](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/gradient1.png)

+++
* Gradient is the slope of the line that 'touches' the graph at a certain point

![Gradient](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/gradient2.png)

+++
* When the slope is positive, direction is away from the minimum

![Gradient](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/gradient2.png)

+++
* When the slope is negative, direction is away from the minimum

![Gradient](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/gradient3.png)

+++
* When the slope zero, gradient is horizontal

![Gradient](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/gradient4.png)

---
### Compute Change in Weight

![Computation](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/compute-deltaw.png)

* https://en.wikipedia.org/wiki/Backpropagation

---
## Back Propagation

![Neuron](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/recall-mlp.png)

+++
## Back Propagation

* At the output layer 
  * Compute error term at the output layer - $(y_j - t_j)$
  * Compute delta weights and delta bias
* For each hidden layer (going backards)
  * Propagate the error from the next layer
  * Compute delta weights and delta bias

+++
```
public void train(final TrainingData trainingData) {
  for (int epoch = 0; epoch < epochs; epoch++) {
    double totalCost = 0.0;
    for (int n = 0; n < trainingData.length(); n++) {
      double[] output =
          feedForward(trainingData.input(n));
      totalCost +=
          computeCost(trainingData.output(n), output);
      computeDeltaWeightsAndBias(
          trainingData.output(n),
          output,
          trainingData.input(n));
    }
    updateTotalWeightsAndBias();
  }
}
```
+++
## Back Propagation
```
private void computeDeltaWeightsAndBias(
    double[] expected,
    double[] output,
    double[] input) {
  layers[layers.length-1]
      .computeErrorAtOutputLayer(expected, output);
  for (int l = layers.length-1; l > 0; l--) {
    final Layer currentLayer = layers[l];
    final Layer previousLayer = layers[l-1];
    previousLayer.propagateErrors(currentLayer);
    currentLayer.computeDeltaWeightsAndBias(
        previousLayer.output, learningRate);
  }
  layers[0].computeDeltaWeightsAndBias(input, learningRate);
}
```

---
## Back Propagation

### When to update weights
* Batch
* Stochastic
* Hybrid
* Realtime / Online (very new!)

---
### Next Step

* Fixed bugs
  * sometimes used f(net) instead of net
  * indexing mistakes?

* Use Numerical Method library (for matrix operations)

* MNIST dataset
  * Try using stochastic back-propagation
  * Try using less data

---
### References
* https://en.wikipedia.org/wiki/Backpropagation
