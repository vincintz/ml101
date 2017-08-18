package ml101.mlp;

import ml101.mlp.activation.ActivationFn;

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MLP {
    private final static Logger logger = LoggerFactory.getLogger(MLP.class);
    private double[][][] weights;
    private double[][] bias;
    private ActivationFn activationFn;
    private double learningRate;
    private int epochs;
    // buffer to store the intermediate computation results per layer
    private double[][] outputValues;

    private MLP(double[] rawWeights, int... nodesPerLayer) {
        final int numLayers = nodesPerLayer.length - 1;
        weights = new double[numLayers][][];
        bias = new double[numLayers][];
        outputValues = new double[nodesPerLayer.length][];
        outputValues[0] = new double[nodesPerLayer[0]];
        int start = 0;
        for (int l = 0; l < nodesPerLayer.length - 1; l++) {
            int rows = nodesPerLayer[l + 1];
            int cols = nodesPerLayer[l];
            outputValues[l + 1] = new double[rows];
            weights[l] = new double[rows][];
            bias[l] = new double[cols];
            for (int j = 0; j < rows; j++) {
                bias[l][j] = rawWeights[start++];
                weights[l][j] = Arrays.copyOfRange(rawWeights, start, start + cols);
                start += cols;
            }
        }
        displayWeights();
    }

    private MLP(int... nodesPerLayer) {
        final int numLayers = nodesPerLayer.length - 1;
        weights = new double[numLayers][][];
        bias = new double[numLayers][];
        outputValues = new double[nodesPerLayer.length][];
        outputValues[0] = new double[nodesPerLayer[0]];
        for (int l = 0; l < nodesPerLayer.length - 1; l++) {
            int rows = nodesPerLayer[l + 1];
            int cols = nodesPerLayer[l];
            outputValues[l + 1] = new double[rows];
            weights[l] = new double[rows][];
            bias[l] = new double[cols];
            for (int j = 0; j < rows; j++) {
                bias[l][j] = Math.random();
                weights[l][j] = new double[cols];
                for (int i = 0; i < cols; i++) {
                    weights[l][j][i] = 0.0;
                }
            }
        }
        displayWeights();
    }

    /**
     * Feed forward computation
     */
    public double[] compute(double... input) {
        System.arraycopy(input, 0, outputValues[0], 0, input.length);
        for (int l = 0; l < weights.length; l++) {
            crossMultiply(outputValues[l+1], weights[l], outputValues[l]);
            vectorAdd(outputValues[l+1], outputValues[l+1], bias[l]);
            activate(outputValues[l+1]);
        }
        return outputValues[weights.length];
    }

    /**
     * Cross Multiply a Matrix with a Vector.
     */
    private void crossMultiply(double[] result, double[][] matrix, double[] vector) {
        for (int j = 0; j < matrix.length; j++) {
            result[j] = 0.0;
            for (int i = 0; i < matrix[j].length; i++) {
                result[j] += vector[i] * matrix[j][i];
            }
        }
    }

    /**
     * Adds two vectors.
     */
    private void vectorAdd(double[] result, double[] v1, double[] v2) {
        int length = Math.min(v1.length, v2.length);
        for (int i = 0; i < length; i++) {
            result[i] = v1[i] + v2[i];
        }
    }

    /**
     * Trains the network using batch back-propagation
     */
    public void train(final double[][] x, final double[][] y) {
        double[][][] deltaWeights = zerosFrom(weights);
        double[][] deltaBias = zerosFrom(bias);
        for (int ep = 0; ep < epochs; ep++) {
            doBatchBackprop(deltaWeights, deltaBias, x, y);
            updateWeights(deltaWeights);
            updateBias(deltaBias);
        }
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

    private void updateWeights(double[][][] deltaWeights) {
        for (int k = 0; k < weights.length; k++) {
            for (int j = 0; j < weights[k].length; j++) {
                for (int i = 0; i < weights[k][j].length; i++) {
                    weights[k][j][i] += deltaWeights[k][j][i];
                }
            }
        }
    }

    private void updateBias(double[][] deltaBias) {
        for (int k = 0; k < bias.length; k++) {
            for (int j = 0; j < bias[k].length; j++) {
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

    private double[][][] zerosFrom(double[][][] shape) {
        double[][][] zeros = new double[shape.length][][];
        for (int l = 0; l < shape.length; l++) {
            zeros[l] = new double[shape[l].length][];
            for (int j = 0; j < shape[l].length; j++) {
                zeros[l][j] = new double[shape[l][j].length];
                for (int i = 0; i < shape[l][j].length; i++) {
                    zeros[l][j][i] = 0.0d;
                }
            }
        }
        return zeros;
    }

    private double[][] zerosFrom(double[][] shape) {
        double[][] zeros = new double[shape.length][];
        for (int l = 0; l < shape.length; l++) {
            zeros[l] = new double[shape[l].length];
            for (int j = 0; j < shape[l].length; j++) {
                zeros[l][j] = 0.0d;
            }
        }
        return zeros;
    }

    private void displayWeights() {
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
            if (rawWeights != null) {
                mlp = new MLP(rawWeights, nodesPerLayer);
            }
            else {
                mlp = new MLP(nodesPerLayer);
            }
            mlp.activationFn = activationFn;
            mlp.learningRate = learningRate;
            mlp.epochs = epochs;
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
