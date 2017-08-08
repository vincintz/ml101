# Machine Learning 101
## Linear Regression

---
# Machine Learning 101
## Artificial Neural Networks

---
## Recall

* Q: What's the difference between statistical modelling tools and Machine Learning?
  * Statistical modelling (like regression analysis) overlaps with ML)
  * That is: Some ML are subranches of statistics / some statistical tools are ML algo
  * https://en.wikipedia.org/wiki/Regression_analysis

---
## Recall

![Image](https://www.ibm.com/developerworks/community/blogs/jfp/resource/BLOGS_UPLOADED_IMAGES/MLworkflownotsimple.png)

Data > _Data Prep_ > ML Algorithm > _Model_ > _Deploy_ > _Predict_ > $$$  [4]

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

### Sample run

Neural net training

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