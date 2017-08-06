package ml101.mlp;

import ml101.mlp.activation.ActivationFn;

import java.util.Arrays;

/**
 * Multi Layered Perceptron
 */
public class MLP {
    private static final int MAX_NODES = 1000;
    private final ActivationFn activationFn;
    private final double[][][] weights;

    public MLP(final ActivationFn activationFn, final double[][][] weights) {
        this.activationFn  = activationFn;
        this.weights       = weights;
        displayWeights();
    }

    public double[] compute(double... input) {
        double[] vector = input.clone();
        for (int l = 0; l < weights.length; l++) {
            vector = multiplyMatrixVector(weights[l], vector);
            activate(vector);
        }
        return vector;
    }

    private void displayWeights() {
        for (int l = 0; l < weights.length; l++) {
            System.out.println("-----------");
            System.out.println("Layer " + (l+1));
            System.out.println("-----------");
            for (int j = 0; j < weights[l].length; j++) {
                for (int i = 0; i < weights[l][j].length; i++) {
                    System.out.print(weights[l][j][i] + "  ");
                }
                System.out.println();
            }
        }
    }

    private double[] multiplyMatrixVector(double matrix[][], double[] vector) {
        double[] output = new double[matrix.length];
        for (int j = 0; j < matrix.length; j++) {
            output[j] = -1.0 * matrix[j][0];
            for (int i = 0; i < matrix[j].length-1; i++) {
                output[j] += vector[i] * matrix[j][i+1];
            }
        }
        return output;
    }

    private void activate(double[] vector) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = activationFn.compute(vector[i]);
        }
    }

    /**
     * MLP configuration object.
     * Collects configuration info, then builds a MLP.
     */
    public static class Config {
        private ActivationFn activationFn;
        private int[] nodesPerLayer;
        private double[] rawWeights;

        public MLP build() {
            final int numLayers = nodesPerLayer.length - 1;
            double[][][] weights = new double[numLayers][][];
            int start = 0;
            for (int l = 0; l < nodesPerLayer.length - 1; l++) {
                int rows = nodesPerLayer[l+1];
                int cols = nodesPerLayer[l] + 1;
                weights[l] = new double[rows][];
                for (int j = 0; j < rows; j++) {
                    weights[l][j] = Arrays.copyOfRange(rawWeights, start, start + cols);
                    start += cols;
                }
//                start += rows;
            }
            return new MLP(activationFn, weights);
        }

        public Config activation(final ActivationFn fn) {
            this.activationFn = fn;
            return this;
        }

        public MLP.Config layers(final int... nodesPerLayer) {
            this.nodesPerLayer = nodesPerLayer;
            return this;
        }

        public Config weights(double... weights) {
            this.rawWeights = weights;
            return this;
        }

    }
}
