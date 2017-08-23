package ml101.mlp;

import ml101.mlp.activation.ActivationFn;

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static ml101.mlp.NumUtilities.*;

/**
 * Multi-layer Perceptron
 */
public class MLP {
    private final static Logger logger = LoggerFactory.getLogger(MLP.class);
    private double[][][] weights;
    private double[][] bias;
    private ActivationFn activationFn;
    private double learningRate;
    private int epochs;

    /**
     * Feed forward computation
     */
    public double[] compute(double... input) {
        final double[][] outputValues = createComputationBuffer(weights);
        return feedForward(outputValues, input);
    }

    // Feed forward computation
    private double[] feedForward(final double[][] outputValues, double... input) {
        System.arraycopy(input, 0, outputValues[0], 0, input.length);
        for (int l = 0; l < weights.length; l++) {
            crossMultiply(outputValues[l+1], weights[l], outputValues[l]);
            vectorAdd(outputValues[l+1], outputValues[l+1], bias[l]);
            activate(outputValues[l+1], activationFn);
        }
        return outputValues[weights.length];
    }

    /**
     * Trains the network using batch back-propagation
     */
    public void train(final double[][] input, final double[][] expected) {
        double[][]   outputValues  = createComputationBuffer(weights);
        double[][]   errorValues   = createComputationBuffer(weights);
        double[][][] deltaWeights  = zerosFrom(weights);
        double[][]   deltaBias     = zerosFrom(bias);
        for (int ep = 0; ep < epochs; ep++) {
            double totalSumSquareError = doBatchBackProp(outputValues, errorValues,
                                                         deltaWeights, deltaBias,
                                                         input, expected);
            updateWeightsAndBias(deltaWeights, deltaBias);
            if (ep % 1000 == 0) {
                logger.info("\t{}\t{}", ep, totalSumSquareError);
            }
        }
    }

    // Performs one batch of back propagation
    private double doBatchBackProp(double[][]   outputValues,
                                 double[][]   errorValues,
                                 double[][][] deltaWeights,
                                 double[][]   deltaBias,
                                 double[][]   input,
                                 double[][]   expected) {
        double totalSumSquareError = 0.0;
        for (int n = 0; n < input.length; n++) {
            feedForward(outputValues, input[n]);
            totalSumSquareError += computeNodeErrors(errorValues, outputValues, expected[n]);
            computeDeltaWeightsAndBias(deltaWeights, deltaBias, outputValues, errorValues);
        }
        return totalSumSquareError;
    }

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

    private void updateWeightsAndBias(double[][][] deltaWeights, double[][] deltaBias) {
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

    // Initialize network weights from input array
    private void initializeWeights(double[] rawWeights, int... nodesPerLayer) {
        final int numLayers = nodesPerLayer.length - 1;
        weights = new double[numLayers][][];
        bias = new double[numLayers][];
        int start = 0;
        for (int l = 0; l < nodesPerLayer.length - 1; l++) {
            int rows = nodesPerLayer[l + 1];
            int cols = nodesPerLayer[l];
            weights[l] = new double[rows][];
            bias[l] = new double[cols];
            for (int j = 0; j < rows; j++) {
                bias[l][j] = rawWeights[start++];
                weights[l][j] = Arrays.copyOfRange(rawWeights, start, start + cols);
                start += cols;
            }
        }
    }

    //  Initialize network weights with random values
    private void initializeWeights(int... nodesPerLayer) {
        final int numLayers = nodesPerLayer.length - 1;
        weights = new double[numLayers][][];
        bias = new double[numLayers][];
        for (int l = 0; l < nodesPerLayer.length - 1; l++) {
            int rows = nodesPerLayer[l + 1];
            int cols = nodesPerLayer[l];
            weights[l] = new double[rows][];
            bias[l] = new double[rows];
            for (int j = 0; j < rows; j++) {
                bias[l][j] = Math.random();
                weights[l][j] = new double[cols];
                for (int i = 0; i < cols; i++) {
                    weights[l][j][i] = Math.random();
                }
            }
        }
    }

    // Creates output values of each neuron to avoid multiple calls to new
    private double[][] createComputationBuffer(double[][][] weights) {
        final int layers = weights.length;
        double[][] outputValues = new double[layers+1][];
        outputValues[0] = new double[weights[0][0].length];
        for (int l = 0; l < layers; l++) {
            outputValues[l+1] = new double[weights[l].length];
        }
        return outputValues;
    }

    public void displayWeightsAndBias(final String text) {
        logger.info(text);
        for (int l = 0; l < weights.length; l++) {
            logger.info("Layer " + (l+1));
            for (int j = 0; j < weights[l].length; j++) {
                final StringBuilder builder = new StringBuilder();
                builder.append("  ")
                        .append(bias[l][j]);
                for (int i = 0; i < weights[l][j].length; i++) {
                    builder.append("  ")
                            .append(weights[l][j][i]);
                }
                logger.info(builder.toString());
            }
        }
    }

    /**
     * MLP configuration object.
     * Collects configuration info, then builds a MLP.
     */
    public static class Builder
    {
        private ActivationFn activationFn;
        private int[] nodesPerLayer;
        private double[] rawWeights;
        private double learningRate;
        private int epochs;

        public MLP build() {
            final MLP mlp;
            mlp = new MLP();
            mlp.activationFn = activationFn;
            mlp.learningRate = learningRate;
            mlp.epochs = epochs;
            if (rawWeights != null) {
                mlp.initializeWeights(rawWeights, nodesPerLayer);
            }
            else {
                mlp.initializeWeights(nodesPerLayer);
            }
            return mlp;
        }

        public Builder activation(final ActivationFn fn) {
            this.activationFn = fn;
            return this;
        }

        public Builder layers(final int... nodesPerLayer) {
            this.nodesPerLayer = nodesPerLayer;
            return this;
        }

        public Builder weights(double... weights) {
            this.rawWeights = weights;
            return this;
        }

        public Builder randomWeights() {
            this.rawWeights = null;
            return this;
        }

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder epochs(int epochs) {
            this.epochs = epochs;
            return this;
        }
    }

}
