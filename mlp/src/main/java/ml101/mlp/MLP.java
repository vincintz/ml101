package ml101.mlp;

import ml101.mlp.activation.ActivationFn;

import java.util.Arrays;

public class MLP {
    final private ActivationFn activationFn;
    final private double[][][] weights;
    final private double[][] computeBuffer;

    MLP(final ActivationFn activationFn, final double[][][] weights, final double[][] computeBuffer) {
        this.activationFn  = activationFn;
        this.weights       = weights;
        this.computeBuffer = computeBuffer;
        displayWeights();
    }

    public double[] compute(double... input) {
        System.arraycopy(input, 0, computeBuffer[0], 1, input.length);
        for (int l = 0; l < weights.length; l++) {
            computeBuffer[l][0] = -1;
            multiplyMatrixVector(computeBuffer[l+1], weights[l], computeBuffer[l]);
            activate(computeBuffer[l+1]);
        }
        return computeBuffer[weights.length];
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
            result[j+1] = 0.0;
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
            double[][] computeBuffer = new double[nodesPerLayer.length][];
            computeBuffer[0] = new double[nodesPerLayer[0] + 1];
            int start = 0;
            for (int l = 0; l < nodesPerLayer.length - 1; l++) {
                int rows = nodesPerLayer[l+1];
                int cols = nodesPerLayer[l] + 1;
                computeBuffer[l+1] = new double[rows+1];
                weights[l] = new double[rows][];
                for (int j = 0; j < rows; j++) {
                    weights[l][j] = Arrays.copyOfRange(rawWeights, start, start + cols);
                    start += cols;
                }
            }
            return new MLP(activationFn, weights, computeBuffer);
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
