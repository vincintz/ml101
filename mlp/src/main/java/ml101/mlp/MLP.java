package ml101.mlp;

import ml101.mlp.activation.ActivationFn;

import java.util.Arrays;

/**
 * MultiLayer-Perceptron
 */
public class MLP {
    private final ActivationFn activationFn;
    private final double[][][] weights;

    public MLP(final ActivationFn activationFn, final double[][][] weights) {
        this.activationFn  = activationFn;
        this.weights       = weights;
        displayWeights();
    }

    public double[] compute(double... input) {
        double[] vector = new double[input.length+1];
        System.arraycopy(input, 0, vector, 1, input.length);
        for (double[][] weight : weights) {
            vector[0] = -1;
            double[] output = new double[weight.length + 1];
            multiplyMatrixVector(output, weight, vector);
            vector = output;
            activate(vector);
        }
        return Arrays.copyOfRange(vector, 1, vector.length);
    }

    /**
     * Cross Multiply a Matrix with a Vector. Pass the array to store the result in.
     * @param result where to store the result
     * @param matrix the Matrix
     * @param vector the Vector
     */
    private void multiplyMatrixVector(
            double[]   result,
            double[][] matrix,
            double[]   vector) {
        for (int j = 0; j < matrix.length; j++) {
            for (int i = 0; i < matrix[j].length; i++) {
                result[j+1] += vector[i] * matrix[j][i];
            }
        }
    }

    private void activate(double[] vector) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = activationFn.compute(vector[i]);
        }
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
