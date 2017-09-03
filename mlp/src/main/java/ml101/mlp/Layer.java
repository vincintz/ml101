package ml101.mlp;

import ml101.mlp.activation.ActivationFn;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.Arrays;

import static java.lang.Math.random;
import static ml101.mlp.NumUtilities.activate;
import static ml101.mlp.NumUtilities.crossMultiply;
import static ml101.mlp.NumUtilities.vectorAdd;

public class Layer implements Serializable {
    transient final private static Logger logger = LoggerFactory.getLogger(Layer.class);
    final private ActivationFn activationFn;
    final private double[][]   weights;
    final private double[]     bias;

    transient double[]   output;
    transient double[]   errors;
    transient double[][] deltaWeights;
    transient double[] deltaBias;

    Layer(ActivationFn activationFn, int numInputs, int numOutputs) {
        this.activationFn = activationFn;
        this.weights = new double[numOutputs][];
        this.bias    = new double[numOutputs];
        for (int j = 0; j < numOutputs; j++) {
            this.weights[j] = new double[numInputs];
            this.bias[j]    = random();
            for (int i = 0; i < numInputs; i++) {
                this.weights[j][i] = random();
            }
        }
    }

    int numOutputs() {
        return weights.length;
    }

    int numInputs() {
        return weights[0].length;
    }

    double[] feedForward(double[] input) {
        crossMultiply(output, weights, input);
        vectorAdd(output, output, bias);
        activate(output, activationFn);
        return output;
    }

    /*
     * Demo method for manually setting weights and biases. Useful for testing manually selecting weights and biases
     * for an XOR mlp.
     */
    int setWeightsAndBiases(int start, double[] rawWeights) {
        int numOutputs = weights.length;
        int numInputs = weights[0].length;
        for (int j = 0; j < numOutputs; j++) {
            bias[j] = rawWeights[start++];
            weights[j] = Arrays.copyOfRange(rawWeights, start, start + numInputs);
            start += numInputs;
        }
        return start;
    }

    /*
     * Initializes computation buffers. These arrays are used to avoid multiple calls to 'new'.
     */
    void initializeComputationBuffers() {
        int numOutputs = weights.length;
        int numInputs = weights[0].length;
        this.output       = new double[numOutputs];
        this.errors       = new double[numOutputs];
        this.deltaWeights = new double[numOutputs][];
        this.deltaBias = new double[numOutputs];
        for (int j = 0; j < numOutputs; j++) {
            this.output[j] = 0.0;
            this.errors[j] = 0.0;
            this.deltaWeights[j] = new double[numInputs];
            this.deltaBias[j] = 0.0;
            for (int i = 0; i < numInputs; i++) {
                this.deltaWeights[j][i] = 0.0;
            }
        }
    }

    void displayWeightsAndBias() {
        for (int j = 0; j < weights.length; j++) {
            final StringBuilder builder = new StringBuilder();
            builder.append("  ")
                    .append(bias[j]);
            for (int i = 0; i < weights[j].length; i++) {
                builder.append("  ")
                        .append(weights[j][i]);
            }
            logger.info(builder.toString());
        }
    }

    void computeErrorAtOutputLayer(double[] expected, double[] output) {
        for (int j = 0; j < output.length; j++) {
            double delta = expected[j] - output[j];
            errors[j] = delta * activationFn.derivative(output[j]);
        }
    }

    void propagateErrors(final Layer currentLayer) {
        final Layer previousLayer = this;
        for (int i = 0; i < currentLayer.numInputs(); i++) {
            double delta = 0.0;
            for (int j = 0; j < currentLayer.numOutputs(); j++) {
                delta += currentLayer.weights[j][i] * currentLayer.errors[j];
            }
            previousLayer.errors[i] = delta * activationFn.derivative(previousLayer.output[i]);
        }
    }

    void computeDeltaWeightsAndBias(final double[] input, final double learningRate) {
        final Layer currentLayer = this;
        for (int j = 0; j < currentLayer.numOutputs(); j++) {
            currentLayer.deltaBias[j] += learningRate * currentLayer.errors[j];
            for (int i = 0; i < currentLayer.numInputs(); i++) {
                currentLayer.deltaWeights[j][i] +=
                        learningRate * currentLayer.errors[j] * input[i];
            }
        }
    }

    void updateTotalWeightsAndBias() {
        for (int j = 0; j < weights.length; j++) {
            for (int i = 0; i < weights[j].length; i++) {
                weights[j][i] += deltaWeights[j][i];
                deltaWeights[j][i] = 0.0;
            }
            bias[j] += deltaBias[j];
            deltaBias[j] = 0.0;
        }
    }
}
