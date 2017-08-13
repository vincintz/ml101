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
$$z = \sum_iw_i*input_i \\
step(z) = 1: if (z + bias \geq 0) \\
= 0: otherwise$$
![Step Function](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Dirac_distribution_CDF.svg/325px-Dirac_distribution_CDF.svg.png)
+++
#### Step Function
* Good for demo purposes
* We need the derivative during training

+++
#### Logistic Function
![Step Function](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg)

---
### Boolean functions

#### OR
| x1 | x2 | output |
| -- | -- | ------ |
| 0  | 0  | 0      |
| 0  | 1  | 1      |
| 1  | 0  | 1      |
| 1  | 1  | 1      |

+++
#### AND
| x1 | x2 | output |
| -- | -- | ------ |
| 0  | 0  | 0      |
| 0  | 1  | 0      |
| 1  | 0  | 0      |
| 1  | 1  | 1      |

---
#### Challenge - XOR

+++
* XOR output is **not** linearly seperable

---
### We need to implement
| Implement   | AKA              | Sample           |
| ----------- | ---------------- | ---------------- |
| Task        | Predict/Classify | Feed forward     |
| Performance | Cost function    | Cost function    |
| Experience  | Training         | Back propagation |

---
## Recall
### What is Machine Learning?

* Tom Mitchell (1998): Study of algorithms that:[2][3]
    * improve on a given task _T_
    * a certain performance measure _P_
    * given experience _E_

* A well defined learning task is given by _< T, P, E >_

+++
### For NN, we need to implement
| What        | AKA              | Sample           |
| ----------- | ---------------- | ---------------- |
| Task        | Predict/Classify | Feed forward     |
| Performance | Cost function    | Cost function    |
| Experience  | Training         | Back propagation |

---
### Two phases

* Training
  * Data -> Training Algo => Model
* Usage
  * Data -> Model => Prediction

---







---
### Q: What's the difference between statistical modelling tools and Machine Learning?
* Statistical modelling (like regression analysis) overlaps with ML)
* That is: Some ML are subranches of statistics / some statistical tools are ML algo
* https://en.wikipedia.org/wiki/Regression_analysis

---
## Prerequisite

1. Linear Algebra
2. Calculus

+++
### Linear Algebra

* Matrix-Vector cross multiplication
![Image](./assets/md/assets/metrix-vector.png.jpg)

+++
* Matrix-Vector cross multiplication
```
private void multiplyMatrixVector(double[] result,
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

+++
### Calculus

* Derivatives
* We need the derivative of a specific function during training

+++
* Our interface
```
public interface ActivationFn {
    double compute(double z);
    double derivative(double z);
}
```
* Sample implementation
```
public class LogisticFn implements ActivationFn {
    @Override
    public double compute(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    @Override
    public double derivative(double z) {
        double fz = compute(z);
        return fz * (1.0 - fz);
    }
}
```

---
## Recall
### What is Machine Learning?

* Tom Mitchell (1998): Study of algorithms that:[2][3]
    * improve on a given task _T_
    * a certain performance measure _P_
    * given experience _E_

* A well defined learning task is given by _< T, P, E >_

+++
### We need to implement
| What        | AKA              | Sample           |
| ----------- | ---------------- | ---------------- |
| Task        | predict/classify | Feed forward     |
| Performance | Cost function    | Cost function    |
| Experience  | Training         | Back propagation |

+++
### Two phases

* Training
* Usage


+++
### Training
1. Configure a learning algo
2. Run training => trained model

### Usage
* We use the model to make predictions

---
### Neural Networks

---
### Concepts

* Neuron
* Can represent an AND/OR function
* How to represent XOR

+++

Combine Neurons

* 2 layer XOR representation

---

### Multi-layered Perceptron

* Def'n: Type of ANN
* Picture of MLP

+++

### Program representation

* OOP? Matrix?

---

### Training

mlp.train(x, y);

+++

### Usage

y' = mlp.predict(x')


---
Resources

<a target="_blank" href="https://intelligentjava.wordpress.com/2015/04/28/machine-learning-decision-tree/">https://intelligentjava.wordpress.com/2015/04/28/machine-learning-decision-tree/</a>
<a target="_blank" href="https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x">https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x</a>
<a target="_blank" href="https://en.wikipedia.org/wiki/Matrix_multiplication">https://en.wikipedia.org/wiki/Matrix_multiplication</a>