package ml101.mlp;

import ml101.mlp.activation.ActivationFn;

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static ml101.mlp.math.NumUtilities.crossMultiply;
import static ml101.mlp.math.NumUtilities.vectorAdd;
import static ml101.mlp.math.NumUtilities.zerosFrom;

public class MLP {
    private final static Logger logger = LoggerFactory.getLogger(MLP.class);
    private double[][][] weights;
    private double[][] bias;
    private ActivationFn activationFn;
    private double learningRate;
    private int epochs;

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
        displayWeightsAndBias();
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
            bias[l] = new double[cols];
            for (int j = 0; j < rows; j++) {
                bias[l][j] = Math.random();
                weights[l][j] = new double[cols];
                for (int i = 0; i < cols; i++) {
                    weights[l][j][i] = Math.random();
                }
            }
        }
        displayWeightsAndBias();
    }

    /**
     * Feed forward computation
     */
    public double[] compute(double... input) {
        final double[][] outputValues = createComputationBuffer(weights);
        return compute(outputValues, input);
    }

    // Feed forward computation
    private double[] compute(final double[][] outputValues, double... input) {
        System.arraycopy(input, 0, outputValues[0], 0, input.length);
        for (int l = 0; l < weights.length; l++) {
            crossMultiply(outputValues[l+1], weights[l], outputValues[l]);
            vectorAdd(outputValues[l+1], outputValues[l+1], bias[l]);
            activate(outputValues[l+1]);
        }
        return outputValues[weights.length];
    }

    /**
     * Trains the network using batch back-propagation
     */
    public void train(final double[][] x, final double[][] y) {
        double[][][] deltaWeights = zerosFrom(weights);
        double[][] deltaBias = zerosFrom(bias);
        for (int ep = 0; ep < epochs; ep++) {
            doBatchBackprop(deltaWeights, deltaBias, x, y);
            updateWeightsAndBias(deltaWeights, deltaBias);
        }
        displayWeightsAndBias();
    }

    /**
     * Performs one batch of back propagation
     */
    private void doBatchBackprop(double[][][] deltaWeights,
                                 double[][] deltaBias,
                                 final double[][] x,
                                 final double[][] y) {
        for (int n = 0; n < x.length; n++) {
            double[] h = compute(x[n]);
            double cost = cost(h, y[n]);
            logger.info("cost: {}", cost);
            //throw new UnsupportedOperationException();
        }
    }

    private double cost(double[] h, double[] y) {
        double sumSq = 0.0;
        for (int i = 0; i < y.length; i++) {
            sumSq += Math.pow(h[i] - y[i], 2);
        }
        return (1.0 / 2.0*y.length) * sumSq;
    }

    private void updateWeightsAndBias(double[][][] deltaWeights, double[][] deltaBias) {
        for (int k = 0; k < weights.length; k++) {
            for (int j = 0; j < weights[k].length; j++) {
                for (int i = 0; i < weights[k][j].length; i++) {
                    weights[k][j][i] += deltaWeights[k][j][i];
                }
                bias[k][j] += deltaBias[k][j];
            }
        }
    }

    /**
     * Fires activation function for each node in a layer
     * @param layer Output values for a layer
     */
    private void activate(double[] layer) {
        for (int i = 0; i < layer.length; i++) {
            layer[i] = activationFn.compute(layer[i]);
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

    private void displayWeightsAndBias() {
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
