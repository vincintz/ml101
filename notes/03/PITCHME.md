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

+++
### Multilayer Perceptron (MLP)

#### Conventions
* we use the following indexes
  * _l_ : for layer
  * _j_ : node in current layer / output node
  * _i_ : node in previous layer / input node
* each layer has:
  * biases\[j\]
  * weights\[j\]\[i\]

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

* Cost function
$$ Cost = \frac 12 (t_j - y_j)^2 $$
* For the derivation, we treat bias as a normal weight with input = 1
* Change in weight
$$ \Delta w_{ji} = -\alpha * \frac {\delta Cost}{\delta w_{ji}} $$

---
### Gradient Intuition

* We isolate a sample weight/bias (first bias at first layer)

![Gradient)](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/gradient1.png)

![cost = sumSquare(expected - output)](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/cost.png)

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

![Gradient](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/gradient1.png)

---
### Compute Change in Weight

* Output Layer

$$\frac {\delta E}{\delta w_{ji}} = (y_j-t_j) * fn'(net_j) * input_i$$

* Hidden Layers

$$ \frac {\delta E}{\delta w_{ji}} = (\sum_{l \epsilon L} err_j*w_{ji}) * fn'(net_j) * input_i $$

* https://en.wikipedia.org/wiki/Backpropagation

---
## Back Propagation

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
### References
* https://en.wikipedia.org/wiki/Backpropagation
