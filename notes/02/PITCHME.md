# Machine Learning 101
## Multilayer perceptron (MLP)

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
public double[] compute(double... inp) {
  System.arraycopy(inp, 0, buffer[0], 1, inp.length);
  for (int l = 0; l < weights.length; l++) {
    buffer[l][0] = 1.0;
    multMatrixVector(buffer[l+1], weights[l], buffer[l]);
    activate(buffer[l+1]);
  }
  return buffer[weights.length];
}
```

+++
### Matrix-Vector Multiply
```
private void multMatrixVector(
        double[] result,
        double[][] matrix,
        double[] vector) {
  for (int j = 0; j < matrix.length; j++) {
    result[j+1] = 0.0;
    for (int i = 0; i < matrix[j].length; i++) {
      result[j+1] += vector[i] * matrix[j][i];
    }
  }
}
```
---
### To generalize:
![Picture of MLP](wala)

+++
* But our XOR network was programmed
  * We chose the weights to solve the problem
* We need an algorithm to _learn_ the weights
  * Backpropagation
* First, we need to define the MLP network architecture
  * Number of layers (2 in our example, not counting the input layer)
  * Number of nodes per layer
  * Activation function

+++
* For learning, we also need a learning
  * A _learning rate_
  * Stopping criteria (_number of iterations_)
  * Performance measure (remember _<P, M, T>_)
    * Error/cost function 
