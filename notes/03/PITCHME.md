# Machine Learning 101
## Multilayer Perceptron (MLP)

---
## Recall
* Multilayer Perceptron (MLP)
* Forward Computation
* Backpropagation

+++
### Multilayer Perceptron (MLP)

![Picture of MLP](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/mlp.png)

* input / hidden / output
* weights / bias


+++
### Multilayer Perceptron (MLP)

* input/output 

+++
### Forward Computation

---
## Input/Output Samples

### XOR
|           | Input | Hidden | Output |
| --------- |:-----:|:------:|:------:|
| **Nodes** | 2     | 2      | 1      |

* Input Layer: 2
* 1 Hidde

---
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
![Picture of MLP](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/mlp.png)

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

+++
![Picture of MLP](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/mlp.png)

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
### train
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
### doBatchBackProp
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
### updateWeightsAndBias
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
### compute cost at the output layer 
```
  double[] output = outputValues[layers - 1];
  for (int j = 0; j < output.length; j++) {
    double delta = expected[j] - output[j];
    errorValues[layers-1][j] = delta * activationFn.derivative(output[j]);
    sumSquareError += delta * delta;
  }
```
+++
### compute error at hidden layers 
```
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
```

+++
### computeDeltaWeightsAndBias
```
int layers = errorValues.length;
for (int currentLayer = layers-1; currentLayer > 0; currentLayer--) {
  int previousLayer = currentLayer - 1;
  for (int j = 0; j < errorValues[currentLayer].length; j++) {
    deltaBias[previousLayer][j] +=
            learningRate * errorValues[currentLayer][j];
    for (int i = 0; i < errorValues[previousLayer].length; i++) {
        deltaWeights[previousLayer][j][i] +=
                learningRate * errorValues[currentLayer][j]
                             * outputValues[previousLayer][i];
    }
  }
}
```

---
### Results
#### Cost vs Epoch
![Cost vs Epoch](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/xor-train.png)

+++
#### Can get stuck in a local minima
![Cost vs Epoch](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/xor-train-stuck.png)

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
  * Data preparation
  * Train/Test/Validation data
* Troubleshooting
  * Tuning the parameters
  * Bias vs Variance

---
### References
* https://en.wikipedia.org/wiki/Multilayer_perceptron
* http://neuralnetworksanddeeplearning.com/chap2.html
* http://www.cse.unsw.edu.au/~cs9417ml/MLP2/