# Machine Learning 101
## Multilayer Perceptron (MLP)

---
## Definition of Terms
* A **multilayer perceptron (MLP)** is a class of _feedforward_, _artificial neural network_
* **Feedforward**: information always moves one direction
* **Artificial neural networks (ANNs)** are computing systems inspired by the _biological neural networks_

+++
### Biological Neural Network
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Chemical_synapse_schema_cropped.jpg/250px-Chemical_synapse_schema_cropped.jpg" width="45%">

+++
* Neurons connects to each other to form neural networks.
* The dendrites of a neuron are cellular extensions with many branches. This is where the majority of **input** to the neuron occurs via the dendritic spine.
* The axon terminal contains synapses, specialized structures where neurotransmitter **chemicals are released** to communicate with target neurons.

+++
### Neuron: Simplified

<img src="https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/neuron.png" width="70%">

+++
* Neurons connects to each other to form neural networks.
* Neurons receives _weighted_ inputs from other neurons. Each input paths has a specific weight.
* When a certain _threshold_ is met, it sends an _activated_ value to other neurons.

---
### Sample Activation Functions
#### Step Function
<img src="https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/stepfn.png" width="80%">
+++
#### Step Function
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Dirac_distribution_CDF.svg/325px-Dirac_distribution_CDF.svg.png" width="80%">

+++
#### Step Function
* Good for demo purposes
* But we need the derivative during training
+++
#### Logistic Function
<img src="https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/logisticfn.png" width="80%">
+++
#### Logistic Function
<img src="https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg" width="80%">

+++
#### Logistic Function - derivative
<img src="https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/dlogisticfn.png" width="80%">

+++
#### Step Function
```
public class StepFn implements ActivationFn {
  public double compute(double z) {
    if (z < 0)
      return 0.0;
    else
      return 1.0;
  }
  public double derivative(double z) {
    throw new ArithmeticException(
          "StepFn does not support derivative");
  }
}
```
+++
#### Logistic Function
```
public class LogisticFn implements ActivationFn {
  public double compute(double z) {
    return 1.0 / (1.0 + Math.exp(-z));
  }
  public double derivative(double z) {
    double fz = compute(z);
    return fz * (1.0 - fz);
  }
}
```
---
### What can our neuron do?

+++
#### Boolean OR
<img src="https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/or.png" width="90%">

+++
#### Boolean AND
<img src="https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/and.png" width="90%">

+++
#### Challenge XOR
<img src="https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/xor.png" width="50%">

+++
* Can't be represented by a neuron
* But:<br/>
<img src="https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/xorfn.png" width="80%">
+++
<img src="https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/xor-mlp.png" width="80%">

+++
<img src="https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/feedforward1.png" width="80%">

+++
### Forward Computation
```
// Feed forward computation
private double[] feedForward(final double[][] outputValues,
                             double... input) {
  System.arraycopy(input, 0, outputValues[0], 0, input.length);
  for (int l = 0; l < weights.length; l++) {
    crossMultiply(outputValues[l+1], weights[l], outputValues[l]);
    vectorAdd(outputValues[l+1], outputValues[l+1], bias[l]);
    activate(outputValues[l+1], activationFn);
  }
  return outputValues[weights.length];
}
```

+++
### Matrix-Vector Multiply
```
public static void crossMultiply(double[]   result,
                                 double[][] matrix,
                                 double[]   vector) {
  for (int j = 0; j < matrix.length; j++) {
    result[j] = 0.0;
    for (int i = 0; i < matrix[j].length; i++) {
      result[j] += vector[i] * matrix[j][i];
    }
  }
}
```

+++
### Matrix-Vector Multiply
```
public static void vectorAdd(double[] result,
                             double[] v1,
                             double[] v2) {
  int length = Math.min(v1.length, v2.length);
  for (int i = 0; i < length; i++) {
    result[i] = v1[i] + v2[i];
  }
}
```

+++
### To generalize:
![Picture of MLP](http://www.cse.unsw.edu.au/~cs9417ml/MLP2/BPNeuralNetwork.jpg)

---
* But our XOR network was programmed
  * We chose the weights to solve the problem
* We need an algorithm to _learn_ the weights
* First, we need to define the MLP network architecture
  * Number of layers (2 in our example, not counting the input layer)
  * Number of nodes per layer
  * Activation function

+++
* We also need the following:
  * A _learning rate_
  * Stopping criteria
  * Performance measure
    * Error/cost function

---
### Sample API
#### Instantiate
```
final MLP mlp =
        new MLP.Builder()
               .activation(new LogisticFn())
               .layers(2, 2, 1)
               .randomWeights()
               .learningRate(0.05)
               .epochs(1000000)
               .build();
```

+++
#### Train
```
mlp.train(
        new double[][] {{0.0, 0.0},
                        {0.0, 1.0},
                        {1.0, 0.0},
                        {1.0, 1.0}},
        new double[][] {{0.0},
                        {1.0},
                        {1.0},
                        {0.0}});
```

+++
#### Use
```
        assertEquals(0.0, mlp.compute(0.0, 0.0)[0], DELTA);
        assertEquals(1.0, mlp.compute(0.0, 1.0)[0], DELTA);
        assertEquals(1.0, mlp.compute(1.0, 0.0)[0], DELTA);
        assertEquals(0.0, mlp.compute(1.0, 1.0)[0], DELTA);
```

---
### Training Algorithms
* Randomly update the weights
* Change each weight by a small amount
  * keep the update that decrease the error
* Change each weight by a small amount
  * predict which update will decrease the error
  * derivative == gradient == direction of growth
  * use negative gradient :-)

---
### Gradient Descent Algorithm

* Run forward computation
* Compute difference between expected output and forward computation output
* Update the weights to minimize the difference
* Repeat the above steps

+++
### Backpropagation Algorithm

* Run forward computation
* At the output layer
  * Compute difference between expected output and forward computation output
  * Update the weights to minimize the difference
* For each hidden layer (nearest to the OL, going backwards)
  * Compute a difference that would improve the output from this layer to the next
  * Update the weights to minimize the difference
* Repeat the above steps

+++
### Back-propagation

Backpropagation is a method used in artificial neural networks to calculate the error contribution of each neuron after
a batch of data (in image recognition, multiple images) is processed

+++
### Gradient Descent
Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function.
To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the
gradient (or of the approximate gradient) of the function at the current point.

---
```
public void train(final double[][] input,
                  final double[][] expected) {
  for (int ep = 0; ep < epochs; ep++) {
    doBatchBackProp(deltaWeights, deltaBias,
                    input, expected);
    updateWeightsAndBias(deltaWeights, deltaBias);
  }
}
```

+++
```
private double doBatchBackProp(double[][][] deltaWeights, double[][]   deltaBias,
                               double[][] input, double[][]   expected) {
  for (int n = 0; n < input.length; n++) {
    feedForward(outputValues, input[n]);
    computeNodeErrors(errorValues, outputValues, expected[n]);
    computeDeltaWeightsAndBias(deltaWeights, deltaBias, outputValues, errorValues);
  }
}
```

+++
```
private void updateWeightsAndBias(double[][][] deltaWeights,
                                  double[][] deltaBias) {
  for (int k = 0; k < weights.length; k++) {
    for (int j = 0; j < weights[k].length; j++) {
      for (int i = 0; i < weights[k][j].length; i++) {
        weights[k][j][i] += deltaWeights[k][j][i];
        deltaWeights[k][j][i] = 0.0;
      }
      bias[k][j] += deltaBias[k][j];
      deltaBias[k][j] = 0.0;
    }
  }
}
```

+++
```
private double computeNodeErrors(double[][] errorValues, double[][] outputValues, double[] expected) {
  int layers = errorValues.length;
  double sumSquareError = 0.0;
  // compute cost at the output layer
  double[] output = outputValues[layers - 1];
  for (int j = 0; j < output.length; j++) {
    double delta = expected[j] - output[j];
    errorValues[layers-1][j] = delta * activationFn.derivative(output[j]);
    sumSquareError += delta * delta;
  }
  // compute error at hidden layers
  for (int currentLayer = layers-1; currentLayer > 1; currentLayer--) {
    int previousLayer = currentLayer - 1;
    double[] hidden = outputValues[previousLayer];
    for (int i = 0; i < errorValues[previousLayer].length; i++) {
      double delta = 0.0;
      for (int j = 0; j < errorValues[currentLayer].length; j++) {
        delta += weights[previousLayer][j][i] * errorValues[currentLayer][j];
      }
      errorValues[previousLayer][i] = delta * activationFn.derivative(hidden[i]);
    }
  }
  return sumSquareError;
}
```

+++
```
private void computeDeltaWeightsAndBias(double[][][] deltaWeights,
                                        double[][]   deltaBias,
                                        double[][]   outputValues,
                                        double[][]   errorValues) {
    int layers = errorValues.length;
    for (int currentLayer = layers-1; currentLayer > 0; currentLayer--) {
        int previousLayer = currentLayer - 1;
        for (int j = 0; j < errorValues[currentLayer].length; j++) {
            deltaBias[previousLayer][j] +=
                    learningRate * errorValues[currentLayer][j];
            for (int i = 0; i < errorValues[previousLayer].length; i++) {
                deltaWeights[previousLayer][j][i] +=
                        learningRate * errorValues[currentLayer][j] * outputValues[previousLayer][i];
            }
        }
    }
}
```

---
### Next Step
* Gradient descent intuition
  * Error function over time (epoch)
  * Weight/Bias update computation
* Possible improvements
  * regularization
  * momentum
  * Batch vs SGD vs mixed
  * use a numeric math library

+++
### Next Step
* Real world application
  * preparing the data
* Troubleshooting
  * tuning the parameters
  * bias vs variance

---
### References
* https://en.wikipedia.org/wiki/Multilayer_perceptron
* http://neuralnetworksanddeeplearning.com/chap2.html
* http://www.cse.unsw.edu.au/~cs9417ml/MLP2/