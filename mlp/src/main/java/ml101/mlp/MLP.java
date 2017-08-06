package ml101.mlp;

import ml101.mlp.activation.ActivationFn;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * Multi Layered Perceptron
 */
public class MLP {
    private final ActivationFn activationFn;
    private final INDArray[] weights;

    private MLP(final ActivationFn activationFn, final INDArray... weights) {
        this.activationFn = activationFn;
        this.weights = weights;
    }

    public double[] compute(double... x) {
        INDArray vec = Nd4j.create(x);
        for (final INDArray w : this.weights) {
            vec = Nd4j.prepend(vec, 1, -1.0, 1);
            vec = w.mmul(vec.transpose()).transpose();
            System.out.println("$ " + vec);
        }
        return new double[] {vec.getDouble(0)};
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
            INDArray[] weights = new INDArray[numLayers];
            int start = 0;
            for (int l = 0; l < nodesPerLayer.length - 1; l++) {
                int cols = nodesPerLayer[l] + 1;
                int rows = nodesPerLayer[l+1];
                double[] subWeights = Arrays.copyOfRange(rawWeights, start, start + cols*rows);
                weights[l] = Nd4j.create(subWeights, new int[] {rows, cols});
                start += cols*rows;
            }
            final MLP mlp = new MLP(activationFn, weights);
            return mlp;
        }

        public Config activation(final ActivationFn fn) {
            this.activationFn = fn;
            return this;
        }

        public MLP.Config layers(final int... ll) {
            this.nodesPerLayer = ll;
            return this;
        }

        public Config weights(double... rawWeights) {
            this.rawWeights = rawWeights;
            return this;
        }

    }
}
