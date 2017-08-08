# Machine Learning 101
## Linear Regression

---
# Machine Learning 101
## Artificial Neural Networks

---
## Recall
### Q: What's the difference between statistical modelling tools and Machine Learning?
* Statistical modelling (like regression analysis) overlaps with ML)
* That is: Some ML are subranches of statistics / some statistical tools are ML algo
* https://en.wikipedia.org/wiki/Regression_analysis

---
## Prerequisite

1. Linear Algebra
2. Calculus
3. Biology

+++
### Linear Algebra

* Matrix-Vector cross multiplication
```
private void multiplyMatrixVector(double[] result, double[][] matrix, double[] vector) {
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

* Derivative of complicated functions
* Can be googled!

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
### Two phase

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

https://intelligentjava.wordpress.com/2015/04/28/machine-learning-decision-tree/
https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x