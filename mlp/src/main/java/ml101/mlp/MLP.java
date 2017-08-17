package ml101.mlp;

import ml101.mlp.activation.ActivationFn;

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MLP {
    private final static Logger logger = LoggerFactory.getLogger(MLP.class);
    private double[][][] weights;
    private ActivationFn activationFn;
    private double learningRate;
    private int epochs;
    // buffer to store the intermediate computation results per layer
    private double[][] outputValues;

    private MLP(double[] rawWeights, int... nodesPerLayer) {
        final int numLayers = nodesPerLayer.length - 1;
        weights = new double[numLayers][][];
        outputValues = new double[nodesPerLayer.length][];
        outputValues[0] = new double[nodesPerLayer[0] + 1];
        int start = 0;
        for (int l = 0; l < nodesPerLayer.length - 1; l++) {
            int rows = nodesPerLayer[l + 1];
            int cols = nodesPerLayer[l] + 1;
            outputValues[l + 1] = new double[rows + 1];
            weights[l] = new double[rows][];
            for (int j = 0; j < rows; j++) {
                weights[l][j] = Arrays.copyOfRange(rawWeights, start, start + cols);
                start += cols;
            }
        }
        displayWeights();
    }

    private MLP(int... nodesPerLayer) {
        final int numLayers = nodesPerLayer.length - 1;
        double[][][] weights = new double[numLayers][][];
        double[][] outputValues = new double[nodesPerLayer.length][];
        outputValues[0] = new double[nodesPerLayer[0] + 1];
        int start = 0;
        for (int l = 0; l < nodesPerLayer.length - 1; l++) {
            int rows = nodesPerLayer[l + 1];
            int cols = nodesPerLayer[l] + 1;
            outputValues[l + 1] = new double[rows + 1];
            weights[l] = new double[rows][];
            for (int j = 0; j < rows; j++) {
                weights[l][j] = new double[cols];
                for (int i = 0; i < cols; i++) {
                    weights[l][j][i] = Math.random();
                    start += cols;
                }
            }
        }
        this.weights      = weights;
        this.outputValues = outputValues;
    }

    // Setter
    private void activationFn(final ActivationFn activationFn) {
        this.activationFn = activationFn;
    }

    // Setter
    private void epochs(final int epochs) {
        this.epochs = epochs;
    }

    // Setter
    private void learningRate(final double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Feed forward computation
     * @param input Input to the MLP
     * @return Returns the MLP output
     */
    public double[] compute(double... input) {
        clear(outputValues);
        System.arraycopy(input, 0, outputValues[0], 1, input.length);
        for (int l = 0; l < weights.length; l++) {
            outputValues[l][0] = 1.0;
            multiplyMatrixVector(outputValues[l+1], weights[l], outputValues[l]);
            activate(outputValues[l+1]);
        }
        return outputValues[weights.length];
    }

    /**
     * Cross Multiply a Matrix with a Vector. Pass the array to store the result in.
     * @param result where to store the result
     * @param matrix the Matrix
     * @param vector the Vector
     */
    private void multiplyMatrixVector(double[] result, double[][] matrix, double[] vector) {
        for (int j = 0; j < matrix.length; j++) {
            result[j+1] = 0.0;
            for (int i = 0; i < matrix[j].length; i++) {
                result[j+1] += vector[i] * matrix[j][i];
            }
        }
    }

    /**
     * Trains the network using batch back-propagation
     */
    public void train(final double[][] x, final double[][] y) {
        double[][][] deltaWeights = clone(weights);
        for (int ep = 0; ep < epochs; ep++) {
            deltaWeights = backprop(deltaWeights, x, y);
            updateWeights(deltaWeights);
        }
    }

    private double[][][] backprop(final double[][][] deltaWeights, final double[][] x, final double[][] y) {
        for (int n = 0; n < x.length; n++) {
            double[] h = compute(x[n]);
            double cost = cost(h, y[n]);
            logger.info("cost: {}", cost);
            //throw new UnsupportedOperationException();
        }
        return deltaWeights;
    }

    private double cost(double[] h, double[] y) {
        double sumSq = 0.0;
        for (int i = 0; i < y.length; i++) {
            sumSq += Math.pow(h[i+1] - y[i], 2);
        }
        return (1.0 / 2.0*y.length) * sumSq;
    }

    private void updateWeights(double[][][] deltaWeights) {
        for (int k = 0; k < weights.length; k++) {
            for (int j = 0; j < weights[k].length; j++) {
                for (int i = 0; i < weights[k][j].length; i++) {
                    weights[k][j][i] += deltaWeights[k][j][i];
                    deltaWeights[k][j][i] = 0.0d;
                }
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

    private double[][][] clone(double[][][] weights) {
        double[][][] zeros = new double[weights.length][][];
        for (int l = 0; l < weights.length; l++) {
            zeros[l] = new double[weights[0].length][];
            for (int j = 0; j < weights[0].length; j++) {
                zeros[l][j] = new double[weights[0][0].length];
            }
        }
        return zeros;
    }

    private void clear(double[][] buffer) {
        for (int j = 0; j < buffer.length; j++) {
            for (int i = 0; i < buffer[j].length; i++) {
                buffer[j][i] = 0.0d;
            }
        }
    }

    private void displayWeights() {
        for (int l = 0; l < weights.length; l++) {
            logger.info("Layer " + (l+1));
            for (int j = 0; j < weights[l].length; j++) {
                final StringBuilder builder = new StringBuilder();
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
            mlp.activationFn(activationFn);
            mlp.learningRate(learningRate);
            mlp.epochs(epochs);
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
