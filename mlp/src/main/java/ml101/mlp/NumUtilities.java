package ml101.mlp;

import ml101.mlp.activation.ActivationFn;

public class NumUtilities {
    /**
     * Cross Multiply a Matrix with a Vector.
     */
    public static void crossMultiply(double[] result, double[][] matrix, double[] vector) {
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
    public static void vectorAdd(double[] result, double[] v1, double[] v2) {
        int length = Math.min(v1.length, v2.length);
        for (int i = 0; i < length; i++) {
            result[i] = v1[i] + v2[i];
        }
    }

    public static double[][][] zerosFrom(double[][][] shape) {
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

    public static  double[][] zerosFrom(double[][] shape) {
        double[][] zeros = new double[shape.length][];
        for (int l = 0; l < shape.length; l++) {
            zeros[l] = new double[shape[l].length];
            for (int j = 0; j < shape[l].length; j++) {
                zeros[l][j] = 0.0d;
            }
        }
        return zeros;
    }

    /**
     * Fires activation function for each node in a vector
     * @param vector Output values for a vector
     */
    public static void activate(double[] vector, ActivationFn fn) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = fn.compute(vector[i]);
        }
    }

    public static double square(double v) {
        return v * v;
    }
}
