# Machine Learning 101
## Multilayer Perceptron (MLP)

---
## Recall
* Multilayer Perceptron (MLP)
* Forward Computation
- Training: Backpropagation

+++
### Multilayer Perceptron (MLP)

![Picture of MLP](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/mlp.png)

+++
### Multilayer Perceptron (MLP)

#### Terms
* neuron/node
* layer
* weight and bias


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

+++
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
+++
### Training

* Compute cost at the output layer
  * ![cost = sumSquare(expected - output)](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/cost.png)
* Update the weights and bias such that a certain error function is minimized

---
## Gradient Descent

* We isolate a sample weight/bias (first bias at first layer)

![Gradient)](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/gradient1.png)

![cost = sumSquare(expected - output)](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/cost.png)

+++
* Gradient is the slope of the line that 'touches' the graph at a certain point

![Gradient](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/gradient2.png)

+++
* When the slope is positive

![Gradient](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/gradient2.png)

+++
* Gradient is the slope of the line that 'touches' the graph at a certain point

![Gradient](https://raw.githubusercontent.com/vincintz/ml101/master/notes/assets/gradient3.png)

---
## Back Propagation

---
### References
* https://en.wikipedia.org/wiki/Multilayer_perceptron
* http://neuralnetworksanddeeplearning.com/chap2.html
* http://www.cse.unsw.edu.au/~cs9417ml/MLP2/